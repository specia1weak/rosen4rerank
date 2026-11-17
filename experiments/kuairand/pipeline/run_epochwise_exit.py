# 参考入口：从仓库根使用 uv run python -m experiments.kuairand.pipeline.run_epochwise_exit 启动。
# 保留历史实验的严格对齐检查；需要对应数据/schema及历史run，不是下载即跑的独立教程。
"""KR v3: 连续训练5epoch，每轮直接评估该轮末模型的完整valid/test。

固定在第5轮结束停止；test只记录，不用于梯度、选点、调参或提前终止。
保留六配置×三seed、原物理顺序/模型/AdamW；不在轮间恢复best。
输出与plan13的预算内best评估分开，不覆盖历史run。
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch
from rosen.core.trainers import BaseTrainer
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from experiments.kuairand.pipeline.exit_model import KRExitModel
from experiments.kuairand.pipeline.exit_training import (
    EXPERIMENTS, WORKSPACE, joint_parameter_groups, load_batches, predict_batches, state_hash,
)
from experiments.kuairand.pipeline.kr_request_schema import request_manager
# 复用历史审计的纯指标函数，保持Float32 LogLoss和request-weighted GAUC口径。
from experiments.kuairand.pipeline.evaluation import scores


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def optimizer_hash(optimizer):
    digest = hashlib.sha256()
    for parameter, state in optimizer.state.items():
        digest.update(str(tuple(parameter.shape)).encode())
        for key, value in sorted(state.items()):
            digest.update(key.encode())
            digest.update(value.detach().cpu().numpy().tobytes() if torch.is_tensor(value)
                          else repr(value).encode())
    return digest.hexdigest()


class EpochwiseExitTrainer(BaseTrainer):
    """只改变评估调度，训练step仍调用原生BaseTrainer的联合BCE与clip(10)。"""
    def __init__(self, *args, output, **kwargs):
        super().__init__(*args, **kwargs)
        if self.early_stopper is not None:
            raise ValueError('This protocol stops at max_epochs, not a metric-dependent patience')
        self.output = Path(output)
        self.records = []
        self.best_valid_epoch = None
        self.best_valid_loss = float('inf')
        self.started = time.monotonic()

    def calculate_train_loss(self, batch, ctx):
        loss = super().calculate_train_loss(batch, ctx)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        return loss

    def evaluate_epoch(self):
        """评估当前轮末模型；绝不调用load_state_dict或optimizer.step。"""
        start = time.monotonic()
        destination = self.output/f'epoch_{self.epoch:02d}'
        destination.mkdir(exist_ok=False)
        model_hash = state_hash(self.model)
        opt_hash = optimizer_hash(self.optimizer)
        rng = torch.get_rng_state().clone()
        step_values = sorted({int(s['step'].item()) for s in self.optimizer.state.values() if 'step' in s})
        assert self.global_step == self.epoch*len(self.train_loader)
        assert step_values == [self.global_step]
        # 同一时刻模型和Adam moments一并保存，可审计轮间优化器连续性。
        payload = self._checkpoint_payload()
        payload.update(state_sha256=model_hash, torch_rng_state=rng,
                       numpy_rng_state=np.random.get_state())
        torch.save(payload, destination/'checkpoint.pt')
        metrics = {}
        for split, loader in [('valid', self.valid_loader), ('test', self.test_loader)]:
            prediction = predict_batches(self.model, loader)
            assert prediction['request_id'].n_unique() == prediction.height
            metric = scores(prediction)
            metric.update(epoch=self.epoch, optimizer_steps=self.global_step,
                          state_sha256=model_hash, evaluation='current epoch-end weights',
                          full_train=True, full_eval=True)
            prediction.write_parquet(destination/f'{split}_predictions.parquet')
            write_json(destination/f'{split}_metrics.json', metric)
            metrics[split] = metric
        assert state_hash(self.model) == model_hash
        assert optimizer_hash(self.optimizer) == opt_hash
        assert torch.equal(torch.get_rng_state(), rng)
        valid_loss = metrics['valid']['logloss']
        if valid_loss < self.best_valid_loss:
            self.best_valid_loss, self.best_valid_epoch = valid_loss, self.epoch
        row = dict(epoch=self.epoch, step=self.global_step, valid=metrics['valid'], test=metrics['test'],
                   state_sha256=model_hash, optimizer_state_sha256=opt_hash,
                   optimizer_state_steps=step_values, evaluation_preserved_model_optimizer_rng=True,
                   evaluation_seconds=time.monotonic()-start,
                   elapsed_seconds=time.monotonic()-self.started)
        self.records.append(row)
        write_json(self.output/'epochs.json', self.records)
        write_json(self.output/'status.json', dict(completed_epochs=self.epoch,
            optimizer_steps=self.global_step, best_valid_epoch=self.best_valid_epoch,
            selection='epoch-end minimum valid LogLoss; diagnostic only, never restored'))
        print('EPOCH_RESULT', json.dumps(dict(epoch=self.epoch, step=self.global_step,
            valid_auc=metrics['valid']['overall.auc'], valid_logloss=valid_loss,
            test_auc=metrics['test']['overall.auc'], test_logloss=metrics['test']['logloss'],
            test_user_auc=metrics['test']['user_gauc']['auc'],
            test_intra_step_gauc=metrics['test']['intra_step_gauc']['auc'])), flush=True)
        return {'overall': {'auc': metrics['valid']['overall.auc'], 'logloss': valid_loss}}


def historical_alignment(output, name, seed, initialization, records):
    """与旧轨迹轮末hash比对，不能拿旧预算内best代替当前轮末。"""
    old = WORKSPACE/f'runs/plan13_ablation/{seed}/{name}'
    result = dict(reference=str(old), epochs={})
    for row in records:
        ref = old/f'joint/budget_{row["epoch"]}ep.json'
        if row['epoch'] in {1, 3} and ref.exists():
            expected = json.loads(ref.read_text())['terminal_state_sha256']
            assert row['state_sha256'] == expected, (seed, name, row['epoch'])
            result['epochs'][str(row['epoch'])] = dict(terminal_state_exact=True, sha256=expected)
    manifest = old/'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        # 不同历史manifest可能用不同键，仅以可验证的终点逐字节等价作强检查。
        result['historical_manifest_present'] = True
    result['initial_state_sha256'] = initialization
    write_json(output/'historical_alignment.json', result)


def run(args, output):
    torch.set_num_threads(4)
    data = WORKSPACE/'data/v3'
    sources = [Path(__file__), *[Path(__file__).with_name(n) for n in
        ['exit_model.py', 'exit_training.py', 'kr_request_schema.py', 'evaluation.py']]]
    config = dict(data=str(data), seeds=args.seeds,
        experiments={name: EXPERIMENTS[name] for name in args.experiments}, epochs=5,
        stop='fixed end of epoch 5; no metric-based early termination',
        evaluation='full valid and full test at each epoch end; current weights; no best restore',
        test_role='observation only; never train, checkpoint selection or stop criterion',
        training_strategy='joint product BCE for Ours; BCE for Fusion', auxiliary_loss=None,
        warmup=None, freeze=False, detach_gamma=False, optimizer='AdamW', batch_size=4096,
        gradient_accumulation=1, budget_lr=.005, content_or_fusion_lr=.001, weight_decay=1e-6,
        zero_decay='Embedding,bias,one-dimensional parameters', scheduler=None, ema=False,
        device='cpu', threads=4, train_order='same physical train order each epoch',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        dataset_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
    write_json(output/'config.json', config)
    manager = request_manager(data/'schema', load=True)
    batches = load_batches(data, manager)
    manifest = json.loads((data/'manifest.json').read_text())
    coverage = {}
    for split, loader in batches.items():
        actual = torch.cat([b['request_id'] for b in loader]).numpy()
        expected = pl.read_parquet(data/f'{split}.parquet', columns=['request_id'])['request_id'].to_numpy()
        assert np.array_equal(actual, expected)
        digest = hashlib.sha256(actual.astype(expected.dtype).tobytes()).hexdigest()
        assert len(actual) == manifest['splits'][split]['rows']
        assert digest == manifest['splits'][split]['request_order_sha256']
        coverage[split] = dict(rows=len(actual), batches=len(loader), request_order_sha256=digest)
    write_json(output/'coverage.json', coverage)
    print('DATA', json.dumps(coverage), flush=True)
    summary = {}
    for seed in args.seeds:
        for name in args.experiments:
            key = f'{seed}/{name}'
            destination = output/key
            destination.mkdir(parents=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = KRExitModel(manager, **EXPERIMENTS[name])
            initial = state_hash(model)
            groups = joint_parameter_groups(model)
            write_json(destination/'initialization.json', dict(state_sha256=initial, optimizer_groups=[
                dict(scope=g['scope'], lr=g['lr'], weight_decay=g['weight_decay'],
                     tensors=len(g['params'])) for g in groups]))
            cfg = SimpleNamespace(device='cpu', max_epochs=5, ckpt_dir=str(destination),
                experiment_name=f'epochwise_{name}_{seed}', dataset_name='kuairand_requests_v3',
                train_log_interval=200)
            trainer = EpochwiseExitTrainer(model, torch.optim.AdamW(groups, weight_decay=0), manager,
                TrainerDataLoaders(batches['train'], batches['valid'], batches['test']),
                TrainerComponents(), cfg, output=destination)
            print('START', key, flush=True)
            trainer.run()
            assert trainer.global_step == 5*coverage['train']['batches']
            assert [r['epoch'] for r in trainer.records] == [1, 2, 3, 4, 5]
            historical_alignment(destination, name, seed, initial, trainer.records)
            summary[key] = trainer.records
            write_json(destination/'complete.json', dict(epochs=5, optimizer_steps=trainer.global_step,
                best_valid_epoch=trainer.best_valid_epoch, restored_best=False,
                wall_seconds=time.monotonic()-trainer.started))
            write_json(output/'summary.json', summary)
            print('TRAJECTORY_COMPLETE', key, flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
               for p, h in config['source_hashes'].items())
    write_json(output/'complete.json', dict(trajectories=len(summary), epoch_test_evaluations=5*len(summary)))
    print('COMPLETE', output/'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or len(args.experiments) != len(set(args.experiments)):
        parser.error('seeds and experiments must be unique')
    output = args.output or WORKSPACE/'runs'/f'epochwise5_{datetime.now():%Y%m%d_%H%M%S}'
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    print(f'运行目录：{output}\n训练日志：{output / "logs/train.log"}', flush=True)
    with (output/'logs/train.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        run(args, output)
    print(f'完成：{output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
