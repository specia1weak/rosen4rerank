"""Argon feedback v2: inner-user validation and tag-only full 5-fold comparison."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from rosen.data.dataset import ParquetStreamDataset
from rosen.utils.optimize import split_model_params_by_decay
from rosen.utils.expr_tensor import ExprTensor
from workspace.kr_time_budget.pipeline.features_v2 import prepare_fold_v2, EXTRAS
from workspace.kr_time_budget.pipeline.run_exit import grouped_auc, state_hash, on_device
from workspace.kr_time_budget.pipeline.time_budget_v2 import ExitModelV2


def scores(df):
    from sklearn.metrics import roc_auc_score, log_loss
    y,p = df['y_continue'].to_numpy(), df['prediction'].to_numpy()
    result = {'overall.auc': float(roc_auc_score(y,p)), 'logloss': float(log_loss(y,p,labels=[0,1])), 'rows':len(y)}
    for field,name in [('user_id','user_gauc'),('intra_step_group','intra_step_gauc'),('active_step_group','active_step_gauc')]:
        result[name] = grouped_auc(y,p,df[field].to_numpy())
    return result


def load_batches(path,manager,batch_size,smoke=False):
    ds = ParquetStreamDataset(path,manager,batch_size=batch_size,shuffle=False,drop_last=False,extra_col_names=EXTRAS)
    result=[]
    for b in DataLoader(ds,batch_size=None,num_workers=0):
        result.append(b)
        if smoke:
            break
    return result


def evaluate(model,batches,device,components=False):
    model.eval()
    pieces=[]
    with torch.no_grad():
        for batch in batches:
            b=on_device(batch,device)
            pred=model.predict(b)
            if not torch.isfinite(pred).all():
                raise FloatingPointError('nonfinite eval probability')
            data={c:b[c].cpu().numpy() for c in ['request_id','user_id','session_id','y_continue','intra_step_group','active_step_group','req_idx_raw','elapsed_minutes']}
            data['prediction']=pred.cpu().numpy()
            if components:
                _,gamma,stay,survival,log_s=model.components(b)
                data.update(gamma=gamma.cpu().numpy(),content_stay=stay.cpu().numpy(),survival=survival.cpu().numpy(),log_survival=log_s.cpu().numpy())
            pieces.append(pl.DataFrame(data))
    pred=pl.concat(pieces)
    return pred,scores(pred)


def better(metrics,best):
    return best is None or (metrics['logloss'],-metrics['user_gauc']['auc']) < (best['logloss'],-best['user_gauc']['auc'])


def budget_hash(model):
    h=hashlib.sha256()
    for module in [model.budget_encoder,model.budget_net]:
        for name,t in sorted(module.state_dict().items()):
            h.update(name.encode());h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=Path('workspace/kr_time_budget/data/v1'))
    ap.add_argument('--data',type=Path,default=Path('workspace/kr_time_budget/data/v2'))
    ap.add_argument('--output',type=Path,default=Path('workspace/kr_time_budget/runs/exit_v2'))
    ap.add_argument('--folds',type=int,nargs='+',default=list(range(5)))
    ap.add_argument('--models',nargs='+',default=['content','budget','fusion','ours'])
    ap.add_argument('--batch-size',type=int,default=4096)
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if 'ours' in args.models and args.epochs<2:
        raise ValueError('Ours requires 1 warmup epoch and >=1 product epoch')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True,exist_ok=True)
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    cfg.update(version=2,item_ids=False,own_user_history='all splits; strict past complete sessions at session start; m=5 train global smoothing',elapsed='end_of_screen wall clock per Leader',delta_minutes=1.,validation_fraction_users=.1,eval_fractions=[.25,.5,1.],selection='valid LogLoss then UserGAUC; warmup and product selected separately; final test once',warmup_epochs=1,stage2_budget='independent encoder and net completely frozen; gamma.detach',content_final_bias=2.6,weight_decay=1e-6,embedding_weight_decay=0.,optimizer='AdamW',learning_rate=.001,ema=False,scheduler=None,comparison='multi-fix v2 screening, not a single-variable v1-v2 intervention')
    config=args.output/'config.json'
    if config.exists() and json.loads(config.read_text())!=cfg:
        raise ValueError('Use a new output for another experiment config')
    config.write_text(json.dumps(cfg,indent=2))
    for fold in args.folds:
        manager,data=prepare_fold_v2(args.source,args.data,fold,args.seed)
        start=time.monotonic()
        train=load_batches(data/'train.parquet',manager,args.batch_size,args.smoke)
        valid=load_batches(data/'valid.parquet',manager,args.batch_size,args.smoke)
        print('V2 cached',fold,'train batches',len(train),'valid',len(valid),'sec',time.monotonic()-start,flush=True)
        first_ids=torch.cat([b['request_id'] for b in train]).numpy()
        order_hash=hashlib.sha256(first_ids.tobytes()).hexdigest()
        expected_hash=None
        for mode in args.models:
            out=args.output/f'fold_{fold}'/mode
            out.mkdir(parents=True,exist_ok=True)
            if (out/'metrics.json').exists():
                prior=json.loads((out/'manifest.json').read_text())
                if expected_hash is not None:
                    assert expected_hash==prior['initial_sha256']
                expected_hash=prior['initial_sha256']
                print('SKIP complete',fold,mode,flush=True)
                continue
            if (out/'progress.jsonl').exists():
                raise RuntimeError(f'Incomplete model {out}; move directory aside before restarting it')
            torch.manual_seed(args.seed);np.random.seed(args.seed)
            model=ExitModelV2(manager,mode).to(args.device)
            init_hash=state_hash(model)
            if expected_hash is not None:
                assert expected_hash==init_hash
            expected_hash=init_hash
            manifest={'fold':fold,'mode':mode,'initial_sha256':init_hash,'train_order_sha256':order_hash,'batch_size':args.batch_size,'physical_effective_batch':args.batch_size,'gradient_accumulation':1,'seed':args.seed,'schema_dir':str(data/'schema'),'no_item_id_schema':not any('items' in s.field_name for s in manager.settings)}
            assert manifest['no_item_id_schema']
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
            optimizer=torch.optim.AdamW(split_model_params_by_decay(model,weight_decay=1e-6),lr=.001,weight_decay=0)
            best=None;warm_best=None;steps=0;warm_steps=0;stage2_reference=None
            marks={max(1,math.ceil(len(train)*f)) for f in [.25,.5,1.]}
            def record(value):
                print(json.dumps({'fold':fold,'model':mode,**value}),flush=True)
                with (out/'progress.jsonl').open('a') as f:
                    f.write(json.dumps(value)+'\n')
            def checkpoint(path,metrics,epoch,step,optimizer_state=False):
                payload={'state_dict':model.state_dict(),'stage':model.stage,'epoch':epoch,'step':step,'metrics':metrics,'config':cfg}
                if optimizer_state:
                    payload['optimizer']=optimizer.state_dict()
                torch.save(payload,path)
            for epoch in range(args.epochs):
                if mode=='ours' and epoch==1:
                    warm=torch.load(out/'best_warmup.pt',map_location=args.device,weights_only=False)
                    model.load_state_dict(warm['state_dict']);optimizer.load_state_dict(warm['optimizer'])
                    warm_steps=warm['step']
                    model.set_stage(1)
                    warm_pred,warm_score=evaluate(model,valid,args.device)
                    ref_batch=on_device(valid[0],args.device)
                    with torch.no_grad():
                        _,gamma,stay,_,_=model.components(ref_batch)
                        ratio=stay.mean().item()
                    model.set_stage(2)
                    stage2_reference=budget_hash(model)
                    baseline_gamma=model.budget_components(ref_batch)[0].detach().clone()
                    product_pred,product_score=evaluate(model,valid,args.device)
                    shock={'stage1_selected_step':warm_steps,'warmup_valid_logloss':warm_score['logloss'],'product_start_valid_logloss':product_score['logloss'],'product_to_gamma_mean_ratio_first_valid_batch':ratio,'budget_hash_before_stage2':stage2_reference}
                    (out/'stage_switch.json').write_text(json.dumps(shock,indent=2))
                    record({'event':'stage_switch',**shock})
                    optimizer=torch.optim.AdamW(split_model_params_by_decay(model,weight_decay=1e-6),lr=.001,weight_decay=0)
                    best=product_score
                    checkpoint(out/'best.pt',best,1,warm_steps)
                model.set_stage(1 if mode=='ours' and epoch==0 else 2)
                model.train();seen=0;loss_sum=0.;tic=time.monotonic()
                for i,batch in enumerate(train,1):
                    b=on_device(batch,args.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss=model.calculate_loss(b)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'V2 {fold} {mode} {epoch} {i}')
                    loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                    if not torch.isfinite(grad):
                        raise FloatingPointError('nonfinite gradient norm')
                    optimizer.step();steps+=1
                    n=len(b['y_continue']);seen+=n;loss_sum+=loss.item()*n
                    if i in marks:
                        _,metric=evaluate(model,valid,args.device)
                        phase='warmup' if mode=='ours' and epoch==0 else 'product_or_baseline'
                        record({'event':'validation','epoch':epoch+1,'epoch_fraction':i/len(train),'stage':model.stage,'step':steps,'train_rows_so_far':seen,'train_loss_so_far':loss_sum/seen,'seconds_so_far':time.monotonic()-tic,'phase':phase,'valid':metric})
                        if phase=='warmup':
                            if better(metric,warm_best):
                                warm_best=metric;checkpoint(out/'best_warmup.pt',metric,epoch+1,steps,True)
                        elif better(metric,best):
                            best=metric;checkpoint(out/'best.pt',metric,epoch+1,steps)
                        model.train()
                record({'event':'epoch_complete','epoch':epoch+1,'trained_rows':seen,'train_loss':loss_sum/seen,'optimizer_steps_executed':steps,'seconds':time.monotonic()-tic})
            if mode=='ours':
                assert budget_hash(model)==stage2_reference,'budget path drifted during content training'
                with torch.no_grad():
                    delta=(model.budget_components(ref_batch)[0]-baseline_gamma).abs().max().item()
                assert delta==0.,delta
                switch=json.loads((out/'stage_switch.json').read_text());switch.update(budget_hash_after_stage2=budget_hash(model),gamma_max_change_after_content_training=delta)
                (out/'stage_switch.json').write_text(json.dumps(switch,indent=2))
            selected=torch.load(out/'best.pt',map_location=args.device,weights_only=False)
            model.load_state_dict(selected['state_dict']);model.set_stage(selected['stage']);model.eval()
            test=load_batches(data/'test.parquet',manager,args.batch_size,args.smoke)
            predictions,metric=evaluate(model,test,args.device,components=mode in ['ours','budget'])
            predictions=predictions.with_columns(pl.lit(fold).alias('fold'))
            predictions.write_parquet(out/'predictions.parquet')
            metric.update(full_train=not args.smoke,full_test=not args.smoke,epochs_executed=args.epochs,optimizer_steps_executed=steps,selected_epoch=selected['epoch'],selected_step=selected['step'],selected_valid=selected['metrics'],test_checkpoint='best.pt')
            if mode in ['ours','budget']:
                p=predictions.sort(['session_id','req_idx_raw'])
                p=p.with_columns(pl.col('log_survival').diff().over('session_id').alias('log_s_diff'),pl.col('elapsed_minutes').diff().over('session_id').alias('time_diff'))
                metric['survival_diagnostics']={'adjacent_log_s_increases':p.filter(pl.col('log_s_diff')>1e-6).height,'adjacent_clock_decreases':p.filter(pl.col('time_diff')< -1e-6).height,'log_s_increase_when_clock_nondecreasing':p.filter((pl.col('log_s_diff')>1e-6)&(pl.col('time_diff')>=0)).height}
                assert metric['survival_diagnostics']['log_s_increase_when_clock_nondecreasing']==0
            if mode=='ours':
                metric['product_max_abs_error']=(predictions['prediction']-predictions['gamma']*predictions['content_stay']).abs().max()
                expr=ExprTensor(torch.tensor([.9]),'gamma')*ExprTensor(torch.tensor([.93]),'stay')
                metric['expression_checked']=expr.expr
            (out/'metrics.json').write_text(json.dumps(metric,indent=2))
            record({'event':'RESULT','test':metric})
            del model,optimizer,test,predictions
        del train,valid
    summary={}
    for mode in args.models:
        paths=sorted(args.output.glob(f'fold_*/{mode}/metrics.json'))
        if paths:
            df=pl.read_parquet([p.with_name('predictions.parquet') for p in paths])
            m=scores(df);m['completed_folds']=len(paths);m['fold_metrics']=[json.loads(p.read_text()) for p in paths]
            m['prediction_request_id_unique']=df['request_id'].n_unique()==df.height
            summary[mode]=m
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print('V2 SUMMARY',json.dumps({k:{c:v[c] for c in ['overall.auc','user_gauc','logloss','intra_step_gauc','completed_folds']} for k,v in summary.items()}),flush=True)

if __name__=='__main__':
    main()
