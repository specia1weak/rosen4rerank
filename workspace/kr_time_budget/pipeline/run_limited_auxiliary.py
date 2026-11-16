"""Plan21低强度复验：辅助项合计<=主Loss10%，每项系数<=.1。

UID置0、单/混合+pace、左/右/双辅助×3seed，完整训练5轮，每轮完整test。
继承原训练/评估链路；旧入口、模型及已有run不修改，避免历史哈希失效。
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from workspace.kr_time_budget.pipeline.limited_auxiliary_exit import LimitedAuxiliaryExitModel
from workspace.kr_time_budget.pipeline.run_auxiliary_exit import AuxiliaryTrainer, session_audit, MODELS
from workspace.kr_time_budget.pipeline.exit_training import (
    EXPERIMENTS, WORKSPACE, joint_parameter_groups, load_batches, state_hash,
)
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.run_epochwise_exit import write_json
from workspace.kr_time_budget.pipeline.run_uid_zero import zero_uid_inputs, uid_tables, tensor_hash


VARIANTS = {'left':(.1,0.),'right':(0.,.1),'both':(.1,.1)}


class LimitedAuxiliaryTrainer(AuxiliaryTrainer):
    """逐batch留存有效系数和占比；每步断言合计<=.1*main。"""
    def train_epoch(self):
        self.batch_loss_records = []
        result = super().train_epoch()
        frame = pl.DataFrame(self.batch_loss_records)
        frame.write_parquet(self.output/f'loss_steps_epoch_{self.epoch:02d}.parquet')
        audit = dict(batches=frame.height, max_auxiliary_main_ratio=frame['auxiliary_main_ratio'].max(),
            min_auxiliary_main_ratio=frame['auxiliary_main_ratio'].min(),
            max_effective_left_weight=frame['effective_left_weight'].max(),
            max_effective_right_weight=frame['effective_right_weight'].max(),
            every_step_cap_verified=True, cap_is_loss_value_not_gradient_norm=True)
        write_json(self.output/f'loss_cap_epoch_{self.epoch:02d}.json',audit)
        print('LOSS_CAP_AUDIT',self.epoch,json.dumps(audit),flush=True)
        return result

    def train_step(self,batch,ctx):
        result=super().train_step(batch,ctx)
        row=dict(self.model.last_loss_terms)
        assert all(np.isfinite(v) for v in row.values())
        assert 0 <= row['auxiliary_main_ratio'] <= .1+1e-12
        assert 0 <= row['effective_left_weight'] <= .1
        assert 0 <= row['effective_right_weight'] <= .1
        assert abs(row['total']-row['joint_bce']-row['auxiliary_weighted']) < 1e-12
        self.batch_loss_records.append(dict(step=self.global_step+1,rows=len(batch['y_continue']),**row))
        return result


def run(args,output):
    torch.set_num_threads(4)
    baseline=WORKSPACE/'runs/epochwise5_uid_zero'
    previous=WORKSPACE/'runs/plan21_auxiliary'
    config=json.loads((previous/'config.json').read_text())
    assert json.loads((previous/'complete.json').read_text())['trajectories']==18
    for p,h in config['source_hashes'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    sources={**config['source_hashes'],**{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),Path(__file__).with_name('limited_auxiliary_exit.py')]}}
    data=WORKSPACE/'data/v3'
    assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()==config['dataset_manifest_sha256']
    manager=request_manager(data/'schema',load=True)
    original=load_batches(data,manager)
    batches=zero_uid_inputs(original)
    coverage={}
    old_coverage=json.loads((baseline/'coverage.json').read_text())
    for split,loader in batches.items():
        frame=pl.read_parquet(data/f'{split}.parquet')
        coverage[split]=session_audit(frame)
        ids=torch.cat([b['request_id'] for b in loader]).numpy()
        assert np.array_equal(ids,frame['request_id'].to_numpy())
        coverage[split]['request_order_sha256']=hashlib.sha256(ids.astype(frame['request_id'].to_numpy().dtype).tobytes()).hexdigest()
        assert coverage[split]['request_order_sha256']==old_coverage[split]['request_order_sha256']
        for before,after in zip(original[split],loader):
            assert set(before)==set(after)
            assert torch.count_nonzero(after['user_id'])==0
            assert all(torch.equal(before[f],after[f]) for f in before if f!='user_id')
    del original
    write_json(output/'coverage.json',coverage)
    config.update(seeds=args.seeds,experiments={m:EXPERIMENTS[m] for m in args.models},
        variants={v:dict(left_weight=VARIANTS[v][0],right_weight=VARIANTS[v][1]) for v in args.variants},
        source_hashes=sources,auxiliary_loss='individual coefficients<=.1, combined contribution<=.1*joint BCE',
        max_auxiliary_main_ratio=.1,cap_scaling='detach main and weighted auxiliary in scale; main gradient coefficient 1',
        previous_strong_run=str(previous),baseline=str(baseline),
        previous_strong_config_sha256=hashlib.sha256((previous/'config.json').read_bytes()).hexdigest(),
        training_strategy='one backward, joint BCE + capped auxiliaries; neither tower frozen or detached',
        left_normalization=coverage['train']['requests_per_session'])
    write_json(output/'config.json',config)
    print('DATA',json.dumps(coverage),flush=True)
    summary={}
    for seed in args.seeds:
        for name in args.models:
            for variant in args.variants:
                key=f'{seed}/{name}/{variant}'
                dest=output/key;dest.mkdir(parents=True)
                torch.manual_seed(seed);np.random.seed(seed)
                a,b=VARIANTS[variant]
                model=LimitedAuxiliaryExitModel(manager,**EXPERIMENTS[name],left_weight=a,right_weight=b,
                    requests_per_session=coverage['train']['requests_per_session'],max_aux_ratio=.1)
                reference=json.loads((baseline/f'{seed}/{name}/initialization.json').read_text())
                initial=state_hash(model)
                assert initial==reference['state_sha256']
                groups=joint_parameter_groups(model)
                group_records=[dict(scope=g['scope'],lr=g['lr'],weight_decay=g['weight_decay'],tensors=len(g['params'])) for g in groups]
                assert group_records==reference['optimizer_groups']
                uid_initial={n:tensor_hash(t.weight) for n,t in uid_tables(model).items()}
                write_json(dest/'initialization.json',dict(state_sha256=initial,baseline_initialization_exact=True,
                    optimizer_groups=group_records,uid_table_sha256=uid_initial))
                cfg=SimpleNamespace(device='cpu',max_epochs=5,ckpt_dir=str(dest),
                    experiment_name=f'limited_{name}_{variant}_{seed}',dataset_name='kuairand_requests_v3',train_log_interval=200)
                trainer=LimitedAuxiliaryTrainer(model,torch.optim.AdamW(groups,weight_decay=0),manager,
                    TrainerDataLoaders(batches['train'],batches['valid'],batches['test']),TrainerComponents(),cfg,output=dest)
                print('START',key,flush=True)
                trainer.run()
                assert trainer.global_step==2105 and len(trainer.records)==5
                assert {n:tensor_hash(t.weight) for n,t in uid_tables(model).items()}==uid_initial
                summary[key]=trainer.records
                write_json(dest/'complete.json',dict(epochs=5,optimizer_steps=2105,uid_embeddings_unchanged=True,
                    best_valid_epoch=trainer.best_valid_epoch,restored_best=False))
                write_json(output/'summary.json',summary)
                print('TRAJECTORY_COMPLETE',key,flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
    write_json(output/'complete.json',dict(trajectories=len(summary),epoch_test_evaluations=5*len(summary)))
    print('COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028])
    parser.add_argument('--models',nargs='+',choices=MODELS,default=MODELS)
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=list(VARIANTS))
    parser.add_argument('--output',type=Path,default=WORKSPACE/'runs/plan21_auxiliary_cap10')
    args=parser.parse_args()
    if any(len(x)!=len(set(x)) for x in [args.seeds,args.models,args.variants]):
        parser.error('Duplicate seeds/models/variants')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'logs').mkdir()
    print('日志：',args.output/'logs/train.log',flush=True)
    with (args.output/'logs/train.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
        run(args,args.output)
    print('完成：',args.output/'summary.json',flush=True)


if __name__=='__main__':
    main()
