"""Plan21：UID置0，单/混合+pace × 左/右/双辅助Loss，3seed连续5轮。

主对照直接复用epochwise5_uid_zero同配置；权重预先固定1，不以test调权。
每轮末完整valid/test，固定第5轮停止；不回滚best，不改数据或旧模型源码。
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
from workspace.kr_time_budget.pipeline.auxiliary_exit_model import AuxiliaryExitModel
from workspace.kr_time_budget.pipeline.exit_training import (
    EXPERIMENTS, WORKSPACE, joint_parameter_groups, load_batches, state_hash,
)
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager, BUDGET_PARAMETER_FIELDS
from workspace.kr_time_budget.pipeline.run_epochwise_exit import EpochwiseExitTrainer, write_json
from workspace.kr_time_budget.pipeline.run_uid_zero import zero_uid_inputs, uid_tables, tensor_hash


VARIANTS = {'left': (1., 0.), 'right': (0., 1.), 'both': (1., 1.)}
MODELS = ['single_step_pace', 'mixture_step_pace']


def session_audit(frame):
    """每场唯一终屏；预算输入静态。L仅作训练监督，不添加到特征列。"""
    assert frame['y_continue'].is_in([0, 1]).all()
    g = frame.group_by('session_id').agg(pl.len().alias('n'),
        pl.col('req_idx_raw').min().alias('first'), pl.col('req_idx_raw').max().alias('last'),
        pl.col('req_idx_raw').n_unique().alias('unique_steps'),
        (pl.col('y_continue') == 0).sum().alias('exits'),
        pl.col('req_idx_raw').filter(pl.col('y_continue') == 0).max().alias('exit_step'),
        *[pl.col(f).n_unique().alias(f) for f in BUDGET_PARAMETER_FIELDS])
    assert g.select(((pl.col('first') == 1) & (pl.col('last') == pl.col('n')) &
        (pl.col('unique_steps') == pl.col('n')) & (pl.col('exits') == 1) &
        (pl.col('exit_step') == pl.col('n'))).all()).item()
    assert g.select(pl.all().exclude('session_id','n','first','last','unique_steps','exits','exit_step').max()).row(0) == (1,)*len(BUDGET_PARAMETER_FIELDS)
    return dict(rows=frame.height, sessions=g.height, requests_per_session=frame.height/g.height,
                one_exit_per_complete_session=True, all_budget_inputs_static=True)


class AuxiliaryTrainer(EpochwiseExitTrainer):
    def train_epoch(self):
        self.term_sums, self.term_rows = {}, 0
        result = super().train_epoch()
        values = {k: v/self.term_rows for k, v in self.term_sums.items()}
        write_json(self.output/f'train_loss_epoch_{self.epoch:02d}.json', dict(
            rows=self.term_rows, online_batch_weighted_terms=values,
            note='Parameters change during epoch; left term normalized per session, others per request'))
        print('LOSS_TERMS', self.epoch, json.dumps(values), flush=True)
        return result

    def train_step(self, batch, ctx):
        value = super().train_step(batch, ctx)
        n = len(batch['y_continue'])
        self.term_rows += n
        for name, number in self.model.last_loss_terms.items():
            self.term_sums[name] = self.term_sums.get(name, 0.)+number*n
        if self.global_step == 0:
            audit = {}
            for name, layer in [('budget', self.model.budget_net[-1]), ('content', self.model.content_net[-1])]:
                norm = float(layer.weight.grad.norm())
                assert np.isfinite(norm) and norm > 0
                audit[name] = dict(gradient_norm_after_clipping=norm)
            write_json(self.output/'first_gradient.json', audit)
        return value


def run(args, output):
    torch.set_num_threads(4)
    baseline = WORKSPACE/'runs/epochwise5_uid_zero'
    config = json.loads((baseline/'config.json').read_text())
    assert json.loads((baseline/'complete.json').read_text())['trajectories'] == 18
    for p, h in config['source_hashes'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h, p
    data = WORKSPACE/'data/v3'
    manager = request_manager(data/'schema', load=True)
    batches = zero_uid_inputs(load_batches(data, manager))
    coverage = {}
    for split, loader in batches.items():
        frame = pl.read_parquet(data/f'{split}.parquet')
        coverage[split] = session_audit(frame)
        ids = torch.cat([b['request_id'] for b in loader]).numpy()
        assert np.array_equal(ids, frame['request_id'].to_numpy())
        coverage[split]['request_order_sha256'] = hashlib.sha256(
            ids.astype(frame['request_id'].to_numpy().dtype).tobytes()).hexdigest()
        old = json.loads((baseline/'coverage.json').read_text())[split]
        assert coverage[split]['request_order_sha256'] == old['request_order_sha256']
        assert all(torch.count_nonzero(b['user_id']) == 0 for b in loader)
    write_json(output/'coverage.json', coverage)
    sources = {**config['source_hashes'], **{str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), Path(__file__).with_name('auxiliary_exit_model.py')]}}
    config.update(seeds=args.seeds, experiments={n: EXPERIMENTS[n] for n in args.models},
        variants={v: dict(left_weight=VARIANTS[v][0], right_weight=VARIANTS[v][1]) for v in args.variants},
        baseline=str(baseline), source_hashes=sources,
        baseline_config_sha256=hashlib.sha256((baseline/'config.json').read_bytes()).hexdigest(),
        auxiliary_loss='per-session discrete NLL + per-request content BCE; weights fixed 1 when enabled',
        left_normalization=coverage['train']['requests_per_session'],
        training_strategy='one backward of joint BCE plus selected auxiliaries; no freeze or detach',
        test_role='observation only; no weight search or test-based epoch selection')
    write_json(output/'config.json', config)
    print('DATA', json.dumps(coverage), flush=True)
    summary = {}
    for seed in args.seeds:
        for name in args.models:
            for variant in args.variants:
                key = f'{seed}/{name}/{variant}'
                dest = output/key
                dest.mkdir(parents=True)
                torch.manual_seed(seed)
                np.random.seed(seed)
                a, b = VARIANTS[variant]
                model = AuxiliaryExitModel(manager, **EXPERIMENTS[name], left_weight=a, right_weight=b,
                    requests_per_session=coverage['train']['requests_per_session'])
                reference = json.loads((baseline/f'{seed}/{name}/initialization.json').read_text())
                initial = state_hash(model)
                assert initial == reference['state_sha256']
                groups = joint_parameter_groups(model)
                group_records = [dict(scope=g['scope'], lr=g['lr'], weight_decay=g['weight_decay'],
                                      tensors=len(g['params'])) for g in groups]
                assert group_records == reference['optimizer_groups']
                uid_initial = {n: tensor_hash(t.weight) for n, t in uid_tables(model).items()}
                write_json(dest/'initialization.json', dict(state_sha256=initial,
                    baseline_initialization_exact=True, optimizer_groups=group_records, uid_table_sha256=uid_initial))
                cfg = SimpleNamespace(device='cpu', max_epochs=5, ckpt_dir=str(dest),
                    experiment_name=f'aux_{name}_{variant}_{seed}', dataset_name='kuairand_requests_v3',
                    train_log_interval=200)
                trainer = AuxiliaryTrainer(model, torch.optim.AdamW(groups, weight_decay=0), manager,
                    TrainerDataLoaders(batches['train'], batches['valid'], batches['test']),
                    TrainerComponents(), cfg, output=dest)
                print('START', key, flush=True)
                trainer.run()
                assert trainer.global_step == 2105 and len(trainer.records) == 5
                assert {n: tensor_hash(t.weight) for n, t in uid_tables(model).items()} == uid_initial
                summary[key] = trainer.records
                write_json(dest/'complete.json', dict(epochs=5, optimizer_steps=trainer.global_step,
                    uid_embeddings_unchanged=True, best_valid_epoch=trainer.best_valid_epoch, restored_best=False))
                write_json(output/'summary.json', summary)
                print('TRAJECTORY_COMPLETE', key, flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in sources.items())
    write_json(output/'complete.json', dict(trajectories=len(summary), epoch_test_evaluations=5*len(summary)))
    print('COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--output', type=Path, default=WORKSPACE/'runs/plan21_auxiliary')
    args = parser.parse_args()
    if any(len(v) != len(set(v)) for v in [args.seeds, args.models, args.variants]):
        parser.error('Duplicate seeds/models/variants')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'logs').mkdir()
    print('日志：', args.output/'logs/train.log', flush=True)
    with (args.output/'logs/train.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        run(args, args.output)
    print('完成：', args.output/'summary.json', flush=True)


if __name__ == '__main__':
    main()
