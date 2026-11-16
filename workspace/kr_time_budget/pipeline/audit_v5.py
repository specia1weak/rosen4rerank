"""Read-only mechanism checks: direct clock masking, calibration, temporal trends."""
import json
from pathlib import Path
import polars as pl
import torch
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.run_exit_v3 import load_batches
from workspace.kr_time_budget.pipeline.run_exit import on_device
from workspace.kr_time_budget.pipeline.time_budget_v4 import ExitModelV4
from workspace.kr_time_budget.pipeline.time_budget_v5 import ExitModelV5

ROOT=Path('workspace/kr_time_budget')


def main():
    torch.set_num_threads(4)
    manager=request_manager(ROOT/'data/v3/schema',load=True)
    test=load_batches(ROOT/'data/v3/test.parquet',manager,4096)
    results={}
    for version,cls in [('v4',ExitModelV4),('v5',ExitModelV5)]:
        for mode in ['budget','ours']:
            out=ROOT/f'runs/exit_{version}'/mode
            phase='single' if mode=='budget' else 'stage_two'
            model=cls(manager,mode)
            checkpoint=torch.load(out/phase/'best.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['state_dict']);model.set_stage(2);model.eval()
            p=pl.read_parquet(out/'predictions.parquet')
            trend=p.sort(['session_id','req_idx_raw']).with_columns(
                pl.col('elapsed_min').diff().over('session_id').alias('dt'),pl.col('gamma').diff().over('session_id').alias('dg'),
                pl.col('prediction').diff().over('session_id').alias('dp')).filter(pl.col('dt')>=0)
            info={'gamma_mean':p['gamma'].mean(),'gamma_std':p['gamma'].std(),'gamma_min':p['gamma'].min(),'gamma_max':p['gamma'].max(),
                  'prediction_mean':p['prediction'].mean(),'label_mean':p['y_continue'].mean(),
                  'stay_mean':p['content_stay'].mean(),'stay_over_099':(p['content_stay']>.99).mean(),'stay_below_085':(p['content_stay']<.85).mean(),
                  'adjacent_nondecreasing_time_pairs':trend.height,'gamma_increases':trend.filter(pl.col('dg')>1e-7).height,
                  'prediction_increases':trend.filter(pl.col('dp')>1e-7).height}
            if mode=='ours':
                max_x=0.;max_stay=0.;max_prediction_error=0.;rows=0
                with torch.no_grad():
                    for batch in test:
                        b=on_device(batch,'cpu');changed=dict(b,elapsed_min=b['elapsed_min']+10.,req_idx=b['req_idx']+5.)
                        original_x=model.content_x(b);modified_x=model.content_x(changed)
                        max_x=max(max_x,(original_x-modified_x).abs().max().item())
                        stay=torch.sigmoid(model.content_net(original_x))
                        changed_stay=torch.sigmoid(model.content_net(modified_x))
                        max_stay=max(max_stay,(stay-changed_stay).abs().max().item())
                        pred=model.predict(b)
                        reference=pl.DataFrame({'request_id':b['request_id'].numpy()}).join(p.select('request_id','prediction'),on='request_id',validate='1:1',maintain_order='left')['prediction'].to_numpy()
                        max_prediction_error=max(max_prediction_error,float(abs(pred.numpy()-reference).max()));rows+=len(pred)
                info['clock_intervention']={'rows':rows,'elapsed_shift_minutes':10,'req_idx_shift':5,
                    'content_x_max_abs_change':max_x,'content_stay_max_abs_change':max_stay,'loaded_checkpoint_prediction_max_error':max_prediction_error}
                assert max_prediction_error==0
                if version=='v5':assert max_x==max_stay==0
            assert info['gamma_increases']==0
            results[version+'_'+mode]=info
    (ROOT/'runs/exit_v5/mechanism_audit.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
