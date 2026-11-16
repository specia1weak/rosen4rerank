"""Read-only fixed-checkpoint mechanism diagnostics for v3 versus v4."""
import json
from pathlib import Path
import polars as pl
import torch
from torch.nn import functional as F
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.run_exit_v3 import load_batches
from workspace.kr_time_budget.pipeline.run_exit import on_device
from workspace.kr_time_budget.pipeline.time_budget_v3 import ExitModelV3
from workspace.kr_time_budget.pipeline.time_budget_v4 import ExitModelV4


def main():
    root=Path('workspace/kr_time_budget');runs=root/'runs';torch.set_num_threads(4)
    mgr=request_manager(root/'data/v3/schema',load=True)
    batches=load_batches(root/'data/v3/test.parquet',mgr,4096)
    audit={}
    for version in ['v3','v4']:
        for mode in ['budget','ours']:
            path=runs/f'exit_{version}'/mode;phase='stage_two' if mode=='ours' else 'single'
            model=(ExitModelV3 if version=='v3' else ExitModelV4)(mgr,mode)
            model.load_state_dict(torch.load(path/phase/'best.pt',map_location='cpu',weights_only=False)['state_dict']);model.set_stage(2);model.eval()
            df=pl.read_parquet(path/'predictions.parquet')
            p=df.sort(['session_id','req_idx_raw']).with_columns(pl.col('elapsed_min').diff().over('session_id').alias('dt'),pl.col('gamma').diff().over('session_id').alias('dg'),pl.col('prediction').diff().over('session_id').alias('dp'))
            valid=p.filter(pl.col('dt')>=0)
            d={'gamma_mean':df['gamma'].mean(),'prediction_mean':df['prediction'].mean(),'adjacent_nondecreasing_time_pairs':valid.height,
               'adjacent_gamma_increases':valid.filter(pl.col('dg')>1e-7).height,'adjacent_prediction_increases':valid.filter(pl.col('dp')>1e-7).height}
            if mode=='ours':d.update(stay_mean=df['content_stay'].mean(),stay_over_099=(df['content_stay']>.99).mean(),stay_below_085=(df['content_stay']<.85).mean())
            with torch.no_grad():
                if version=='v3':
                    shapes=[]
                    for batch in batches:
                        raw=model.budget_raw(on_device(batch,'cpu')).reshape(-1,2,3)
                        shapes.append((F.softplus(raw[:,:,2])+.5).clamp(max=5).flatten())
                    shape=torch.cat(shapes)
                    d.update(component_shape_below_one_fraction=(shape<1).float().mean().item(),component_shape_min=shape.min().item(),component_shape_max=shape.max().item())
                else:
                    b=on_device(batches[0],'cpu');n=256;grid=torch.tensor([0.,1.,5.,10.,20.,60.,120.]);k=len(grid)
                    fixed={c:v[:n].repeat_interleave(k,dim=0) for c,v in b.items()};fixed['elapsed_min']=grid.repeat(n)
                    gamma=model.budget_components(fixed)[0].reshape(n,k);pred=model.predict(fixed).reshape(n,k)
                    d['fixed_context_grid']={'contexts':n,'elapsed_grid':grid.tolist(),'gamma_increases':int((gamma.diff(dim=1)>1e-7).sum()),'prediction_increases':int((pred.diff(dim=1)>1e-7).sum())}
            audit[version+'_'+mode]=d
    (runs/'exit_v4/mechanism_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
