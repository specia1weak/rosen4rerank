"""Argon 5 v4: reuse v3 data and native training, change only the requested budget mechanism."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import polars as pl
import torch
from rosen.core.train import EarlyStopper
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import TwoStageRunner
from rosen.evaluate.evaluator import Evaluator
from rosen.evaluate.manager import EvaluatorManager, DomainFilter
from rosen.utils.optimize import split_model_params_by_decay
from workspace.kr_time_budget.pipeline.run_exit import state_hash, on_device
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.time_budget_v3 import ExitModelV3
from workspace.kr_time_budget.pipeline.time_budget_v4 import ExitModelV4, weibull_parameters
from workspace.kr_time_budget.pipeline.run_exit_v3 import ExitTrainer, load_batches, predict, scores


def verify_common_v3_initialization(model, manager, mode, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        reference=ExitModelV3(manager,mode)
    for name,t in reference.state_dict().items():
        if not name.startswith('budget_net.2.'):
            assert torch.equal(t,model.state_dict()[name]),name
    return True


class ExitTrainerV4(ExitTrainer):
    transition_reference=None

    def run(self, *, stop_condition=None):
        if self.transition_reference is not None:
            self.model.eval();self.transition_reference.eval()
            max_error=0.;rows=0;warm_loss=0.;product_loss=0.;gamma_sum=0.
            for batch in self.valid_loader:
                b=on_device(batch,self.cfg.device)
                with torch.no_grad():
                    old=self.transition_reference.predict(b)
                    new=self.model.predict(b)
                    gamma=self.model.budget_components(b)[0]
                    assert torch.equal(old,new), 'nonzero stage transition'
                    max_error=max(max_error,(old-new).abs().max().item())
                    n=len(new);rows+=n;gamma_sum+=gamma.sum().item()
                    y=b['y_continue'].float()
                    warm_loss+=torch.nn.functional.binary_cross_entropy(old.clamp(1e-6,1-1e-6),y,reduction='sum').item()
                    product_loss+=torch.nn.functional.binary_cross_entropy(new.clamp(1e-6,1-1e-6),y,reduction='sum').item()
            self.transition_audit={'transition_full_valid_rows':rows,'transition_prediction_max_abs_error':max_error,
                'transition_logloss_difference':(product_loss-warm_loss)/rows,'transition_gamma_mean':gamma_sum/rows,
                'transition_content_stay':self.model.prior_stay.item(),'warmup_prediction_ceiling':self.model.prior_stay.item()}
            print('ZERO_TRANSITION',json.dumps(self.transition_audit),flush=True)
            (self.output/'transition.json').write_text(json.dumps(self.transition_audit,indent=2))
        return super().run(stop_condition=stop_condition)


@torch.no_grad()
def parameter_summary(model,batches,device):
    values=[]
    for batch in batches:
        b=on_device(batch,device)
        scale,shape=weibull_parameters(model.budget_raw(b))
        values.append(pl.DataFrame({'scale':scale.cpu().numpy(),'shape':shape.cpu().numpy()}))
    frame=pl.concat(values)
    return {c:{'min':frame[c].min(),'mean':frame[c].mean(),'max':frame[c].max()} for c in ['scale','shape']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('workspace/kr_time_budget/data/v3'))
    ap.add_argument('--output',type=Path,default=Path('workspace/kr_time_budget/runs/exit_v4'))
    ap.add_argument('--models',nargs='+',choices=['content','budget','fusion','ours'],default=['content','budget','fusion','ours'])
    ap.add_argument('--batch-size',type=int,default=4096);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--device',default='cpu');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(args.threads);args.output.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(protocol='Argon5 single IFR Weibull + fixed content prior; single epoch baselines; ours 1+1 epochs',weight_decay=1e-6,embedding_decay=0.,lr=.001,
                  optimizer='AdamW',physical_effective_batch=args.batch_size,gradient_accumulation=1,scheduler=None,ema=False,
                  selection='valid LogLoss at .25/.5/1 epoch; stage2 includes untrained product start',
                  comparison='architecture screening; same initial tensors and data, different routing and objectives; ours has two epochs')
    config.update(shape='1.05+softplus capped5',scale='softplus*30+0.5',prior_stay_logit=2.6,stage2_head='zero weight, bias2.6 at transition; unchanged baseline heads')
    config['dataset_manifest_sha256']=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()
    cp=args.output/'config.json'
    if cp.exists() and json.loads(cp.read_text())!=config:raise ValueError('Use a fresh output directory for a different configuration')
    cp.write_text(json.dumps(config,indent=2));manager=request_manager(args.data/'schema',load=True)
    train=load_batches(args.data/'train.parquet',manager,args.batch_size,args.smoke)
    valid=load_batches(args.data/'valid.parquet',manager,args.batch_size,args.smoke)
    test=load_batches(args.data/'test.parquet',manager,args.batch_size,args.smoke)
    print('BATCHES',len(train),len(valid),len(test),flush=True)
    order=hashlib.sha256(torch.cat([b['request_id'] for b in train]).numpy().tobytes()).hexdigest()
    expected=None;summary={}
    for mode in args.models:
        out=args.output/mode;out.mkdir(parents=True,exist_ok=True)
        if (out/'metrics.json').exists():
            old=json.loads((out/'manifest.json').read_text())
            if expected is not None:assert old['initial_sha256']==expected
            expected=old['initial_sha256'];summary[mode]=json.loads((out/'metrics.json').read_text());continue
        if list(out.glob('*/progress.jsonl')):raise RuntimeError(f'Incomplete run {out}; use a fresh output or archive it')
        torch.manual_seed(args.seed);np.random.seed(args.seed)
        model=ExitModelV4(manager,mode);initial=state_hash(model)
        if expected is not None:assert expected==initial
        expected=initial
        common_check=verify_common_v3_initialization(model,manager,mode,args.seed)
        (out/'manifest.json').write_text(json.dumps({'initial_sha256':initial,'common_v3_state_exact':common_check,'train_order_sha256':order,
            'train_rows':sum(len(b['y_continue']) for b in train),'batch_size':args.batch_size,
            'full_train':not args.smoke,'full_test':not args.smoke},indent=2))
        def factory(current,stage=None):
            name=stage.value if stage else 'single'
            phase=out/name
            evaluator=EvaluatorManager(log_path=phase/'evaluations.log',title=f'v4 {mode} {name}')
            evaluator.register('overall',Evaluator('auc','gauc','logloss'))
            for domain in range(5):evaluator.register(f'domain{domain}',Evaluator('auc','gauc','logloss'),DomainFilter('tab',domain))
            cfg=SimpleNamespace(device=args.device,max_epochs=1,ckpt_dir=str(phase),experiment_name=f'exit_v4_{mode}_{name}',dataset_name='kuairand_requests_v3',train_log_interval=50)
            trainer=ExitTrainerV4(current,torch.optim.AdamW(split_model_params_by_decay(current,weight_decay=1e-6),lr=.001,weight_decay=0),
                manager,TrainerDataLoaders(train,valid),TrainerComponents(evaluator_manager=evaluator,early_stopper=EarlyStopper(preferred_metrics=('logloss',))),cfg)
            if mode=='ours' and current.stage==2:trainer.transition_reference=model
            return trainer
        if mode=='ours':
            result=TwoStageRunner(factory).run(model);model=result.stage_two_model
            # Snapshot hashes must cover exactly the stage-one budget tables and net.
            for prefix,module in [('budget_embedding.',model.budget_embedding),('budget_net.',model.budget_net)]:
                for name,t in module.state_dict().items():
                    assert torch.equal(t.cpu(),result.stage_snapshot.model_state[prefix+name]),f'budget drift {prefix+name}'
            ref=on_device(valid[0],args.device)
            with torch.no_grad():
                stage1=result.stage_one_trainer.model.budget_components(ref)[0]
                stage2=model.budget_components(ref)[0]
                gamma_drift=(stage1-stage2).abs().max().item()
                assert gamma_drift==0
                stay=model.components(ref)[2].mean().item()
            switch={'budget_exactly_frozen':True,'gamma_max_change':gamma_drift,'selected_stage2_content_stay_mean':stay,
                    'stage1_steps_executed':result.stage_one_trainer.global_step,'stage2_steps_executed':result.stage_two_trainer.global_step,**result.stage_two_trainer.transition_audit}
            (out/'stage_switch.json').write_text(json.dumps(switch,indent=2))
            steps=result.stage_one_trainer.global_step+result.stage_two_trainer.global_step
        else:
            trainer=factory(model);trainer.run();steps=trainer.global_step
        pred=predict(model,test,args.device)
        if mode=='budget':pred=pred.with_columns(pl.lit(model.prior_stay.item()).alias('content_stay'))
        pred.write_parquet(out/'predictions.parquet')
        metric=scores(pred);metric.update(full_train=not args.smoke,full_test=not args.smoke,optimizer_steps_executed=steps)
        if mode in ['budget','ours']:
            p=pred.sort(['session_id','req_idx_raw']).with_columns(pl.col('log_survival').diff().over('session_id').alias('ds'),pl.col('elapsed_min').diff().over('session_id').alias('dt'))
            p=p.with_columns(pl.col('gamma').diff().over('session_id').alias('dg'))
            metric['survival_diagnostics']={'gamma_increase_with_nondecreasing_time':p.filter((pl.col('dt')>=0)&(pl.col('dg')>1e-7)).height,'clock_decreases':p.filter(pl.col('dt')<0).height,'S_increase_with_nondecreasing_time':p.filter((pl.col('dt')>=0)&(pl.col('ds')>1e-6)).height}
            assert metric['survival_diagnostics']['S_increase_with_nondecreasing_time']==0
            assert metric['survival_diagnostics']['gamma_increase_with_nondecreasing_time']==0
            metric['budget_parameters']=parameter_summary(model,test,args.device)
        if mode in ['content','fusion'] and not args.smoke:
            reference=pl.read_parquet(Path('workspace/kr_time_budget/runs/exit_v3')/mode/'predictions.parquet')
            assert pred['request_id'].equals(reference['request_id'])
            error=(pred['prediction']-reference['prediction']).abs().max()
            metric['unchanged_baseline_max_prediction_error_vs_v3']=error
            assert error==0.,error
        if mode=='ours':metric['product_max_abs_error']=(pred['prediction']-pred['gamma']*pred['content_stay']).abs().max()
        (out/'metrics.json').write_text(json.dumps(metric,indent=2));summary[mode]=metric
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print('RESULT',mode,json.dumps(metric),flush=True)
        del model
    print('COMPLETE',json.dumps({k:{c:v[c] for c in ['overall.auc','user_gauc','logloss']} for k,v in summary.items()}),flush=True)

if __name__=='__main__':main()
