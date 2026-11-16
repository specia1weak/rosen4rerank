"""Argon 4 date-split comparison using native rosen Trainer/TwoStageRunner."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from rosen.core.train import EarlyStopper
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer, TwoStageRunner
from rosen.evaluate.evaluator import Evaluator
from rosen.evaluate.manager import EvaluatorManager, DomainFilter
from rosen.data.dataset import ParquetStreamDataset
from rosen.utils.optimize import split_model_params_by_decay
from workspace.kr_time_budget.pipeline.run_exit import state_hash, grouped_auc, on_device
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager, EXTRA_COLUMNS
from workspace.kr_time_budget.pipeline.time_budget_v3 import ExitModelV3


def load_batches(path,manager,size,smoke=False):
    loader=DataLoader(ParquetStreamDataset(path,manager,batch_size=size,shuffle=False,drop_last=False,extra_col_names=EXTRA_COLUMNS),batch_size=None,num_workers=0)
    batches=[]
    for b in loader:
        batches.append(b)
        if smoke:break
    return batches


def budget_hash(model):
    digest=hashlib.sha256()
    for module in model.budget_modules:
        digest.update(state_hash(module).encode())
    return digest.hexdigest()


class ExitTrainer(BaseTrainer):
    """Keep core lifecycle; add within-epoch validation and restore the selected state."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.output=Path(self.cfg.ckpt_dir);self.output.mkdir(parents=True,exist_ok=True)
        self.best_loss=float('inf');self.selected=None;self.eval_step=0
        self.marks={max(1,math.ceil(len(self.train_loader)*f)) for f in [.25,.5]}

    def calculate_train_loss(self,b,ctx):
        loss=super().calculate_train_loss(b,ctx)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
        return loss

    def train_step(self,b,ctx):
        loss=super().train_step(b,ctx)
        if ctx.batch_idx+1 in self.marks:
            self.eval_step=self.global_step+1
            self.evaluate_epoch();self.model.train()
        return loss

    def evaluate_epoch(self):
        metrics=super().evaluate_epoch()
        step=max(self.eval_step,self.global_step)
        overall=metrics['overall']
        row={'step':step,'epoch':self.epoch,'stage':self.model.stage,'valid':metrics}
        print('VALID',json.dumps(row),flush=True)
        with (self.output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if overall['logloss']<self.best_loss:
            self.best_loss=overall['logloss'];self.selected=row
            torch.save({'state_dict':self.model.state_dict(),'selection':row},self.output/'best.pt')
        return metrics

    def run(self,*,stop_condition=None):
        # Product start is a valid stage-two checkpoint too.
        if self.model.mode=='ours' and self.model.stage==2:self.evaluate_epoch()
        super().run(stop_condition=stop_condition)
        best=torch.load(self.output/'best.pt',map_location=self.cfg.device,weights_only=False)
        self.model.load_state_dict(best['state_dict'])
        (self.output/'selection.json').write_text(json.dumps({'selected':best['selection'],'optimizer_steps_executed':self.global_step},indent=2))
        return best['selection']['valid']


@torch.no_grad()
def predict(model,batches,device):
    model.eval();parts=[]
    for batch in batches:
        b=on_device(batch,device);p=model.predict(b)
        if not torch.isfinite(p).all():raise FloatingPointError('nonfinite test predictions')
        data={c:b[c].cpu().numpy() for c in ['request_id','session_id','user_id_raw','tab_raw','req_idx_raw','y_continue','elapsed_min']}
        data['prediction']=p.cpu().numpy()
        if model.mode in ['budget','ours']:
            _,gamma,stay,s,log_s=model.components(b)
            data.update(gamma=gamma.cpu().numpy(),content_stay=stay.cpu().numpy(),survival=s.cpu().numpy(),log_survival=log_s.cpu().numpy())
        parts.append(pl.DataFrame(data))
    return pl.concat(parts)


def scores(df):
    from sklearn.metrics import roc_auc_score,log_loss
    y,p=df['y_continue'].to_numpy(),df['prediction'].to_numpy()
    r={'overall.auc':float(roc_auc_score(y,p)),'user_gauc':grouped_auc(y,p,df['user_id_raw'].to_numpy()),
       'logloss':float(log_loss(y,p,labels=[0,1])),'rows':df.height,
       'intra_step_gauc':grouped_auc(y,p,df['req_idx_raw'].to_numpy())}
    r['domains']={str(tab):scores_domain(part) for tab,part in df.partition_by('tab_raw',as_dict=True).items()}
    return r


def scores_domain(df):
    from sklearn.metrics import roc_auc_score,log_loss
    y,p=df['y_continue'].to_numpy(),df['prediction'].to_numpy()
    return {'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'gauc':grouped_auc(y,p,df['user_id_raw'].to_numpy()),'logloss':float(log_loss(y,p,labels=[0,1])),'rows':df.height}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('workspace/kr_time_budget/data/v3'))
    ap.add_argument('--output',type=Path,default=Path('workspace/kr_time_budget/runs/exit_v3'))
    ap.add_argument('--models',nargs='+',choices=['content','budget','fusion','ours'],default=['content','budget','fusion','ours'])
    ap.add_argument('--batch-size',type=int,default=4096);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--device',default='cpu');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(args.threads);args.output.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(protocol='Argon4 date split; single epoch baselines; ours 1+1 epochs',weight_decay=1e-6,embedding_decay=0.,lr=.001,
                  optimizer='AdamW',physical_effective_batch=args.batch_size,gradient_accumulation=1,scheduler=None,ema=False,
                  selection='valid LogLoss at .25/.5/1 epoch; stage2 includes untrained product start',
                  comparison='architecture screening; same initial tensors and data, different routing and objectives; ours has two epochs')
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
        model=ExitModelV3(manager,mode);initial=state_hash(model)
        if expected is not None:assert expected==initial
        expected=initial
        (out/'manifest.json').write_text(json.dumps({'initial_sha256':initial,'train_order_sha256':order,
            'train_rows':sum(len(b['y_continue']) for b in train),'batch_size':args.batch_size,
            'full_train':not args.smoke,'full_test':not args.smoke},indent=2))
        def factory(current,stage=None):
            name=stage.value if stage else 'single'
            phase=out/name
            evaluator=EvaluatorManager(log_path=phase/'evaluations.log',title=f'v3 {mode} {name}')
            evaluator.register('overall',Evaluator('auc','gauc','logloss'))
            for domain in range(5):evaluator.register(f'domain{domain}',Evaluator('auc','gauc','logloss'),DomainFilter('tab',domain))
            cfg=SimpleNamespace(device=args.device,max_epochs=1,ckpt_dir=str(phase),experiment_name=f'exit_v3_{mode}_{name}',dataset_name='kuairand_requests_v3',train_log_interval=50)
            return ExitTrainer(current,torch.optim.AdamW(split_model_params_by_decay(current,weight_decay=1e-6),lr=.001,weight_decay=0),
                manager,TrainerDataLoaders(train,valid),TrainerComponents(evaluator_manager=evaluator,early_stopper=EarlyStopper(preferred_metrics=('logloss',))),cfg)
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
                    'stage1_steps_executed':result.stage_one_trainer.global_step,'stage2_steps_executed':result.stage_two_trainer.global_step}
            (out/'stage_switch.json').write_text(json.dumps(switch,indent=2))
            steps=result.stage_one_trainer.global_step+result.stage_two_trainer.global_step
        else:
            trainer=factory(model);trainer.run();steps=trainer.global_step
        pred=predict(model,test,args.device);pred.write_parquet(out/'predictions.parquet')
        metric=scores(pred);metric.update(full_train=not args.smoke,full_test=not args.smoke,optimizer_steps_executed=steps)
        if mode in ['budget','ours']:
            p=pred.sort(['session_id','req_idx_raw']).with_columns(pl.col('log_survival').diff().over('session_id').alias('ds'),pl.col('elapsed_min').diff().over('session_id').alias('dt'))
            metric['survival_diagnostics']={'clock_decreases':p.filter(pl.col('dt')<0).height,'S_increase_with_nondecreasing_time':p.filter((pl.col('dt')>=0)&(pl.col('ds')>1e-6)).height}
            assert metric['survival_diagnostics']['S_increase_with_nondecreasing_time']==0
        if mode=='ours':metric['product_max_abs_error']=(pred['prediction']-pred['gamma']*pred['content_stay']).abs().max()
        (out/'metrics.json').write_text(json.dumps(metric,indent=2));summary[mode]=metric
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
        print('RESULT',mode,json.dumps(metric),flush=True)
        del model
    print('COMPLETE',json.dumps({k:{c:v[c] for c in ['overall.auc','user_gauc','logloss']} for k,v in summary.items()}),flush=True)

if __name__=='__main__':main()
