# 参考入口：从仓库根使用 uv run python -m experiments.kuairand.pipeline.run_pace_completion 启动。
# 保留历史实验的严格对齐检查；需要对应数据/schema及历史run，不是下载即跑的独立教程。
"""补齐纯时间戳上一刷pace矩阵：无辅助六配置+有pace仅左/仅右，各3seed×5轮。

已完成的有pace双辅助及预算梯度隔离共12条轨迹只读复用，不重跑。
新跑30条；每轮完整valid/test、固定epoch5，不恢复best。
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import polars as pl
import torch
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from experiments.kuairand.pipeline.exit_training import WORKSPACE, EXPERIMENTS, state_hash, joint_parameter_groups
from experiments.kuairand.pipeline.previous_pace import load_previous_batches, PACE_DEFINITION
from experiments.kuairand.pipeline.pace_completion import build_model, VARIANTS
from experiments.kuairand.pipeline.kr_request_schema import request_manager
from experiments.kuairand.pipeline.run_epochwise_exit import EpochwiseExitTrainer, write_json
from experiments.kuairand.pipeline.uid import zero_uid_inputs, uid_tables, tensor_hash
from experiments.kuairand.pipeline.build_timestamp_data import sha

MATRIX = [(name, 'none') for name in EXPERIMENTS] + [
    (name, variant) for name in ['single_step_pace','mixture_step_pace'] for variant in ['left','right']]


class CompletionTrainer(EpochwiseExitTrainer):
    """保留原BaseTrainer step；统一留存无辅助与弱辅助Loss记录。"""
    def train_epoch(self):
        self.loss_rows = []
        result = super().train_epoch()
        pl.DataFrame(self.loss_rows).write_parquet(self.output/f'loss_steps_epoch_{self.epoch:02d}.parquet')
        return result

    def train_step(self, batch, ctx):
        result = super().train_step(batch, ctx)
        terms = dict(getattr(self.model, 'last_loss_terms', {}))
        if terms:
            assert all(np.isfinite(v) for v in terms.values())
            assert 0 <= terms['auxiliary_main_ratio'] <= .1+1e-12
        self.loss_rows.append(dict(step=self.global_step+1,rows=len(batch['y_continue']),**terms))
        if self.global_step == 0:
            watched = {'fusion':self.model.fusion_net[-1]} if self.model.mode == 'fusion' else {
                'budget':self.model.budget_net[-1], 'content':self.model.content_net[-1]}
            grads = {name:float(layer.weight.grad.norm()) for name,layer in watched.items()}
            assert all(np.isfinite(v) and v > 0 for v in grads.values())
            write_json(self.output/'first_gradient.json',grads)
        return result


def run(output):
    torch.set_num_threads(4)
    reference = WORKSPACE/'runs/plan22_previous_pace'
    old = json.loads((reference/'config.json').read_text())
    assert json.loads((reference/'complete.json').read_text())['trajectories'] == 12
    assert all(sha(p) == h for p,h in old['source_hashes'].items())
    sources = {**old['source_hashes'], **{str(p.resolve()):sha(p) for p in [
        Path(__file__),Path(__file__).with_name('pace_completion.py'),reference/'config.json',
        reference/'coverage.json',WORKSPACE/'runs/plan22_budget_stop_gradient/config.json']}}
    data = Path(old['data'])
    manager = request_manager(data/'schema',load=True)
    batches = zero_uid_inputs(load_previous_batches(data,manager))
    coverage = json.loads((reference/'coverage.json').read_text())
    for split,loader in batches.items():
        ids = torch.cat([b['request_id'] for b in loader]).numpy()
        expected = pl.read_parquet(data/f'{split}.parquet',columns=['request_id'])['request_id'].to_numpy()
        assert np.array_equal(ids,expected) and len(ids) == coverage[split]['rows']
        assert all(torch.count_nonzero(b['user_id']) == 0 for b in loader)
    write_json(output/'coverage.json',coverage)
    config = {**old,'models':EXPERIMENTS,'matrix':MATRIX,'pace_scopes':['previous'],
        'previous_pace':PACE_DEFINITION,'variants':VARIANTS,'source_hashes':sources,
        'reuse':{'both':str(reference),'both_detached':str(WORKSPACE/'runs/plan22_budget_stop_gradient')},
        'training_strategy':'joint BCE to both towers; optional left/right auxiliaries <=10% joint',
        'warm_start':False,'new_trajectories':30}
    write_json(output/'config.json',config)
    summary = {}
    for seed in [2026,2027,2028]:
        for name,variant in MATRIX:
            dest = output/f'{seed}/{name}/{variant}'
            # A complete trajectory is safe to reuse after interruption; no partial continuation.
            if (dest/'complete.json').exists():
                assert json.loads((dest/'complete.json').read_text())['steps'] == 2105
                summary[f'{seed}/{name}/{variant}'] = json.loads((dest/'epochs.json').read_text())
                continue
            dest.mkdir(parents=True,exist_ok=False)
            torch.manual_seed(seed);np.random.seed(seed)
            model = build_model(manager,EXPERIMENTS[name],variant,coverage['train']['requests_per_session'])
            init_ref = json.loads((WORKSPACE/f'runs/epochwise5_uid_zero/{seed}/{name}/initialization.json').read_text())
            initial = state_hash(model)
            assert initial == init_ref['state_sha256']
            groups = joint_parameter_groups(model)
            gmeta = [dict(scope=g['scope'],lr=g['lr'],weight_decay=g['weight_decay'],tensors=len(g['params'])) for g in groups]
            assert gmeta == init_ref['optimizer_groups']
            uid_initial = {n:tensor_hash(t.weight) for n,t in uid_tables(model).items()}
            write_json(dest/'initialization.json',dict(state_sha256=initial,optimizer_groups=gmeta,
                uid_initial=uid_initial,reference=str(WORKSPACE/f'runs/epochwise5_uid_zero/{seed}/{name}'),pair_exact=True))
            cfg = SimpleNamespace(device='cpu',max_epochs=5,ckpt_dir=str(dest),
                experiment_name=f'pace_complete_{seed}_{name}_{variant}',dataset_name='kr_v5_previous_pace',train_log_interval=200)
            trainer = CompletionTrainer(model,torch.optim.AdamW(groups,weight_decay=0),manager,
                TrainerDataLoaders(batches['train'],batches['valid'],batches['test']),TrainerComponents(),cfg,output=dest)
            print('START',seed,name,variant,flush=True)
            trainer.run()
            assert trainer.global_step == 2105 and len(trainer.records) == 5
            assert uid_initial == {n:tensor_hash(t.weight) for n,t in uid_tables(model).items()}
            write_json(dest/'complete.json',dict(epochs=5,steps=2105,uid_unchanged=True,restored_best=False))
            summary[f'{seed}/{name}/{variant}'] = trainer.records
            write_json(output/'summary.json',summary)
            print('TRAJECTORY_COMPLETE',seed,name,variant,flush=True)
    assert all(sha(p) == h for p,h in sources.items())
    write_json(output/'complete.json',dict(trajectories=len(summary),test_evaluations=5*len(summary),
        valid_evaluations=5*len(summary),source_hashes_unchanged=True,common_initialization_optimizer_exact=True))
    print('COMPLETE',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=WORKSPACE/'runs/plan22_pace_completion')
    parser.add_argument('--resume',action='store_true',help='Only skip completed trajectories, reject partial folders')
    args = parser.parse_args()
    if not args.resume:args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'logs').mkdir(exist_ok=True)
    print('日志：',args.output/'logs/train.log',flush=True)
    with (args.output/'logs/train.log').open('a' if args.resume else 'w') as f,redirect_stdout(f),redirect_stderr(f):
        run(args.output)


if __name__ == '__main__':main()
