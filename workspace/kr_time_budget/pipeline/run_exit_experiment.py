"""当前KR单配置训练入口：一个模型、一个seed、一次连续训练。

uv run python -m workspace.kr_time_budget.pipeline.run_exit_experiment
uv run python -m workspace.kr_time_budget.pipeline.run_exit_experiment --experiment fusion --epochs 3

只使用data/v3，全量train/valid/test；默认mixture_step_pace、seed2026、1epoch。
模型/训练/优化器实现见exit_model.py与exit_training.py，不导入历史run脚本。
本入口用于后续单次运行；第13份矩阵仍由原入口精确复现。
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch
from rosen.core.train import EarlyStopper
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from rosen.evaluate.evaluator import Evaluator
from rosen.evaluate.manager import EvaluatorManager, DomainFilter
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.exit_model import KRExitModel
from workspace.kr_time_budget.pipeline.exit_training import (
    WORKSPACE, EXPERIMENTS, JointExitTrainer, load_batches, joint_parameter_groups,
    predict_batches, state_hash,
)


def make_evaluator(path):
    evaluator = EvaluatorManager(log_path=path, title='KR single run')
    evaluator.register('overall', Evaluator('auc', 'logloss'))
    for domain in range(5):
        evaluator.register(f'domain{domain}', Evaluator('auc', 'logloss'), DomainFilter('tab', domain))
    return evaluator


@torch.no_grad()
def evaluate_batches(model, manager, batches, evaluator, epoch):
    model.eval()
    for batch in batches:
        evaluator.collect(batch[manager.uid_field], batch[manager.label_field], batch,
                          batch_preds_1d=model.predict(batch))
    metrics = evaluator.summary(epoch)
    evaluator.clear()
    return metrics


def run(args, output):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data = WORKSPACE/'data/v3'
    source_files = [Path(__file__), *[WORKSPACE/'pipeline'/name for name in
                    ['exit_model.py', 'exit_training.py', 'kr_request_schema.py']]]
    config = dict(experiment=args.experiment, model_options=EXPERIMENTS[args.experiment], seed=args.seed,
        epochs=args.epochs, data=str(data), training_strategy='joint', auxiliary_loss=None,
        warmup=None, detach_gamma=False, freeze=False, batch_size=4096, gradient_accumulation=1,
        optimizer='AdamW', budget_lr=.005, content_or_fusion_lr=.001, weight_decay=1e-6,
        zero_decay='Embedding,bias,one-dimensional parameters', scheduler=None, ema=False,
        device='cpu', threads=4, shuffle='same physically shuffled train order each epoch',
        selection='minimum full-valid LogLoss within declared 1/3epoch budget',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        dataset_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
    (output/'config.json').write_text(json.dumps(config, indent=2))

    # 1. 已编码数据 -> Rosen Interaction；记录真实覆盖和物理顺序。
    manager = request_manager(data/'schema', load=True)
    batches = load_batches(data, manager)
    manifest = json.loads((data/'manifest.json').read_text())
    coverage = {}
    for split, loader in batches.items():
        ids = torch.cat([batch['request_id'] for batch in loader]).numpy()
        expected = pl.read_parquet(data/f'{split}.parquet', columns=['request_id'])['request_id'].to_numpy()
        assert np.array_equal(ids, expected)
        order_hash = hashlib.sha256(ids.astype(expected.dtype).tobytes()).hexdigest()
        assert len(ids) == manifest['splits'][split]['rows']
        assert order_hash == manifest['splits'][split]['request_order_sha256']
        coverage[split] = dict(rows=len(ids), batches=len(loader), request_order_sha256=order_hash)
    (output/'coverage.json').write_text(json.dumps(coverage, indent=2))
    print('DATA', coverage, flush=True)

    # 2. 模型与单一优化器；test不交给Trainer，不参与checkpoint选择。
    # DataLoader迭代会消耗Torch RNG；与原实验一样，在数据读取后重置模型seed。
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = KRExitModel(manager, **EXPERIMENTS[args.experiment])
    groups = joint_parameter_groups(model)
    initialization = dict(state_sha256=state_hash(model), optimizer_groups=[
        dict(scope=g['scope'], lr=g['lr'], weight_decay=g['weight_decay'],
             tensors=len(g['params']), elements=sum(p.numel() for p in g['params'])) for g in groups])
    (output/'initialization.json').write_text(json.dumps(initialization, indent=2))
    phase = output/'joint'
    cfg = SimpleNamespace(device='cpu', max_epochs=args.epochs, ckpt_dir=str(phase),
        experiment_name=f'{args.experiment}_{args.seed}', dataset_name='kuairand_requests_v3', train_log_interval=200)
    trainer = JointExitTrainer(model, torch.optim.AdamW(groups, weight_decay=0), manager,
        TrainerDataLoaders(batches['train'], batches['valid']),
        TrainerComponents(evaluator_manager=make_evaluator(phase/'evaluations.log'),
                          early_stopper=EarlyStopper(preferred_metrics=('logloss',))), cfg)

    # 3. 从头连续训练到声明预算；run结束才恢复best，不在epoch边界回滚。
    trainer.run()
    assert trainer.global_step == args.epochs*len(batches['train'])

    # 4. 全部训练结束后，分别重载各预算内的best，评估完整test。
    summary = {}
    for budget in ([1] if args.epochs == 1 else [1, 3]):
        checkpoint = torch.load(phase/f'best_within_{budget}ep.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        destination = output/f'{budget}ep'
        destination.mkdir()
        prediction = predict_batches(model, batches['test'])
        prediction.write_parquet(destination/'predictions.parquet')
        metrics = evaluate_batches(model, manager, batches['test'],
                                  make_evaluator(destination/'evaluations.log'), checkpoint['selection']['epoch'])
        metrics.update(rows=prediction.height, full_train=True, full_valid=True, full_test=True,
            training_budget_epochs=budget, budget_optimizer_steps=budget*len(batches['train']),
            trajectory_optimizer_steps_executed=trainer.global_step, selected=checkpoint['selection'],
            selected_state_sha256=state_hash(model))
        (destination/'metrics.json').write_text(json.dumps(metrics, indent=2))
        summary[f'{budget}ep'] = metrics
        print('RESULT', budget, metrics['overall'], flush=True)
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', output/'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=list(EXPERIMENTS), default='mixture_step_pace')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, choices=[1, 3], default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or WORKSPACE/'runs/current'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_{args.experiment}_{args.seed}'
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    print(f'运行目录：{output}\n持续日志：{output / "logs/train.log"}', flush=True)
    with (output/'logs/train.log').open('w') as log:
        with redirect_stdout(log), redirect_stderr(log):
            run(args, output)
    print(f'完成：{output / "summary.json"}')


if __name__ == '__main__':
    main()
