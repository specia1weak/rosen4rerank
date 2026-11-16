"""Reuse v5 warmup; train paired content-routing probes for one full epoch."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import polars as pl
import torch
from rosen.core.train import EarlyStopper
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from rosen.core.train.two_stage import StageSnapshot, StageReport
from rosen.evaluate.evaluator import Evaluator
from rosen.evaluate.manager import EvaluatorManager, DomainFilter
from rosen.utils.optimize import split_model_params_by_decay
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.run_exit import state_hash
from workspace.kr_time_budget.pipeline.run_exit_v3 import load_batches, predict, scores, budget_hash
from workspace.kr_time_budget.pipeline.run_exit_v4 import ExitTrainerV4
from workspace.kr_time_budget.pipeline.time_budget_v5 import ExitModelV5
from workspace.kr_time_budget.pipeline.time_budget_v6_probe import ExitModelV6Probe, CONTENT_MASKS

ROOT = Path('workspace/kr_time_budget')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'runs/exit_v6_probe')
    ap.add_argument('--variants', nargs='+', choices=list(CONTENT_MASKS), default=list(CONTENT_MASKS))
    ap.add_argument('--smoke', action='store_true', help='First batch only; use a separate output')
    args = ap.parse_args()
    torch.set_num_threads(4)
    data = ROOT/'data/v3'
    reference_run = ROOT/'runs/exit_v5'
    warmup_path = reference_run/'ours/stage_one/best.pt'
    old_cfg = json.loads((reference_run/'config.json').read_text())
    old_manifest = json.loads((reference_run/'ours/manifest.json').read_text())
    manifest_hash = hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()
    assert manifest_hash == old_cfg['dataset_manifest_sha256']
    source_files = [Path(__file__), Path(__file__).with_name('time_budget_v6_probe.py')]
    config = {
        'data': str(data), 'output': str(args.output), 'variants': args.variants, 'smoke': args.smoke,
        'warmup_checkpoint': str(warmup_path),
        'warmup_checkpoint_sha256': hashlib.sha256(warmup_path.read_bytes()).hexdigest(),
        'dataset_manifest_sha256': manifest_hash, 'seed': 2026, 'batch_size': 4096,
        'optimizer': 'AdamW', 'lr': .001, 'weight_decay': 1e-6,
        'decay_excludes': 'Embedding modules, biases and one-dimensional parameters',
        'gradient_accumulation': 1, 'physical_effective_batch': 4096,
        'scheduler': None, 'ema': False, 'device': 'cpu', 'threads': 4,
        'epochs_executed_here': 1, 'warmup_epochs_reused': 1,
        'selection': 'full valid LogLoss at step0 and 25/50/100 percent; test selected checkpoint once',
        'content_masks': {name: sorted(CONTENT_MASKS[name]) for name in args.variants},
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        'comparison': 'paired stage-two routing ablation; same v5 warmup and exact initial tensors; exploratory single seed',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    cfg_path = args.output/'config.json'
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text()) == config, 'Use a fresh output for a changed configuration'
    cfg_path.write_text(json.dumps(config, indent=2))
    manager = request_manager(data/'schema', load=True)
    train = load_batches(data/'train.parquet', manager, 4096, args.smoke)
    valid = load_batches(data/'valid.parquet', manager, 4096, args.smoke)
    test = load_batches(data/'test.parquet', manager, 4096, args.smoke)
    train_rows = sum(len(b['y_continue']) for b in train)
    order = hashlib.sha256(torch.cat([b['request_id'] for b in train]).numpy().tobytes()).hexdigest()
    if not args.smoke:
        assert order == old_manifest['train_order_sha256'] and train_rows == old_manifest['train_rows']
        assert (train_rows, sum(len(b['y_continue']) for b in valid), sum(len(b['y_continue']) for b in test)) == (1721745, 57872, 48898)
    print('DATA', train_rows, 'train;', len(train), 'steps per variant;', len(valid), 'valid batches;', len(test), 'test batches', flush=True)
    torch.manual_seed(2026)
    reference = ExitModelV5(manager)
    warmup = torch.load(warmup_path, map_location='cpu', weights_only=False)
    reference.load_state_dict(warmup['state_dict'])
    selected = warmup['selection']
    snapshot = StageSnapshot.capture(reference, StageReport(selected['epoch'], selected['step'], {}, 'student', selected['valid']))
    baseline_start = reference.initialize_stage_two(snapshot)
    expected_start_hash = state_hash(baseline_start)
    expected_budget_hash = budget_hash(baseline_start)
    summary = {}
    for variant in args.variants:
        started = time.monotonic()
        out = args.output/variant
        out.mkdir(exist_ok=True)
        if (out/'metrics.json').exists():
            summary[variant] = json.loads((out/'metrics.json').read_text())
            continue
        if (out/'stage_two/progress.jsonl').exists():
            raise RuntimeError(f'Incomplete run: {out}; use a fresh output')
        torch.manual_seed(2026)
        np.random.seed(2026)
        model = ExitModelV6Probe(manager, variant).initialize_stage_two(snapshot)
        assert state_hash(model) == expected_start_hash
        for name, value in model.state_dict().items():
            assert torch.equal(value, baseline_start.state_dict()[name]), name
        manifest = {'initial_state_exact_vs_v5_stage2': True, 'initial_sha256': state_hash(model),
                    'budget_sha256': expected_budget_hash, 'train_order_sha256': order,
                    'train_rows': train_rows, 'content_mask': sorted(model.content_mask),
                    'full_train': not args.smoke, 'full_test': not args.smoke,
                    'warmup_selection': selected, 'warmup_checkpoint': str(warmup_path)}
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        print('START', variant, 'mask=', sorted(model.content_mask), flush=True)
        phase = out/'stage_two'
        evaluator = EvaluatorManager(log_path=phase/'evaluations.log', title=f'v6 probe {variant}')
        evaluator.register('overall', Evaluator('auc', 'gauc', 'logloss'))
        for domain in range(5):
            evaluator.register(f'domain{domain}', Evaluator('auc', 'gauc', 'logloss'), DomainFilter('tab', domain))
        cfg = SimpleNamespace(device='cpu', max_epochs=1, ckpt_dir=str(phase),
                              experiment_name=f'exit_v6_{variant}', dataset_name='kuairand_requests_v3', train_log_interval=50)
        optimizer = torch.optim.AdamW(split_model_params_by_decay(model, weight_decay=1e-6), lr=.001, weight_decay=0)
        trainer = ExitTrainerV4(model, optimizer, manager, TrainerDataLoaders(train, valid),
                               TrainerComponents(evaluator_manager=evaluator, early_stopper=EarlyStopper(preferred_metrics=('logloss',))), cfg)
        trainer.transition_reference = reference
        trainer.run()
        assert trainer.global_step == len(train)
        assert budget_hash(model) == expected_budget_hash
        assert all(not p.requires_grad and p.grad is None for module in model.budget_modules for p in module.parameters())
        pred = predict(model, test, 'cpu')
        if not args.smoke:
            old = pl.read_parquet(reference_run/'ours/predictions.parquet')
            assert pred['request_id'].equals(old['request_id'])
            assert pred['gamma'].equals(old['gamma']), 'Frozen budget must match saved v5 gamma exactly'
            if variant == 'control':
                assert pred['prediction'].equals(old['prediction']), 'Control replay differs from v5'
        product_error = (pred['prediction'] - pred['gamma']*pred['content_stay']).abs().max()
        assert product_error == 0
        pred.write_parquet(out/'predictions.parquet')
        metric = scores(pred)
        metric.update(full_train=not args.smoke, full_test=not args.smoke,
                      optimizer_steps_executed_here=trainer.global_step, warmup_epochs_reused=1,
                      initial_state_exact=True, budget_exactly_frozen=True,
                      control_predictions_exact_vs_v5=(variant == 'control' and not args.smoke),
                      product_max_abs_error=product_error, selected=trainer.selected,
                      wall_seconds=time.monotonic()-started,
                      content_stay_mean=pred['content_stay'].mean(), content_stay_std=pred['content_stay'].std())
        gamma_only = pred.with_columns(pl.col('gamma').alias('prediction'))
        metric['gamma_only'] = scores(gamma_only)
        pairs = pred.sort(['session_id', 'req_idx_raw']).with_columns(
            pl.col('elapsed_min').diff().over('session_id').alias('dt'),
            pl.col('prediction').diff().over('session_id').alias('dp'))
        eligible = pairs.filter(pl.col('dt') >= 0)
        metric['nondecreasing_time_pairs'] = eligible.height
        metric['prediction_increases'] = eligible.filter(pl.col('dp') > 1e-7).height
        (out/'metrics.json').write_text(json.dumps(metric, indent=2))
        summary[variant] = metric
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
        print('RESULT', variant, json.dumps({k: metric[k] for k in ['overall.auc', 'user_gauc', 'logloss', 'wall_seconds']}), flush=True)
    print('COMPLETE', str(args.output/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
