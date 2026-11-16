"""Plan 13: Single/Mixture x pace and fixed-architecture 1-vs-3 epoch budgets.

One uninterrupted optimizer trajectory per architecture/seed. Save the best-valid
checkpoint within each declared budget without restoring weights between epochs.
Test is read only after the full trajectory ends. Model/preprocessing are unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import numpy as np
import polars as pl
import torch
from rosen.core.train import EarlyStopper
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from rosen.evaluate.evaluator import Evaluator
from rosen.evaluate.manager import EvaluatorManager, DomainFilter
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.run_exit import state_hash
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores
from workspace.kr_time_budget.pipeline.run_exit_v7_joint import JointTrainer, load_data, predict_joint
from workspace.kr_time_budget.pipeline.run_exit_v9_pooled import calibration
from workspace.kr_time_budget.pipeline.run_exit_v10_plan12 import diagnostics, initial_alignment
from workspace.kr_time_budget.pipeline.time_budget_v8 import joint_parameter_groups
from workspace.kr_time_budget.pipeline.time_budget_v10 import MixtureTimeBudgetModel

ROOT = Path('workspace/kr_time_budget')
EXPERIMENTS = {
    'single_step': dict(mode='ours', mixture=False, step_only=True, add_pace=False),
    'single_step_pace': dict(mode='ours', mixture=False, step_only=True, add_pace=True),
    'mixture_step': dict(mode='ours', mixture=True, step_only=True, add_pace=False),
    'mixture_step_pace': dict(mode='ours', mixture=True, step_only=True, add_pace=True),
    'fusion': dict(mode='fusion', mixture=False, step_only=False, add_pace=False),
    'fusion_pace': dict(mode='fusion', mixture=False, step_only=False, add_pace=True),
}


class BudgetTrainer(JointTrainer):
    """Save selection at epoch 1 and 3; never reset model/optimizer in between."""
    def evaluate_epoch(self):
        metrics = super().evaluate_epoch()
        batches = len(self.train_loader)
        step = max(self.eval_step, self.global_step)
        if step in {batches, 3*batches}:
            budget = step//batches
            # Copy the best checkpoint so later validation cannot overwrite this horizon.
            shutil.copy2(self.output/'best.pt', self.output/f'best_within_{budget}ep.pt')
            record = dict(budget_epochs=budget, budget_optimizer_steps=step,
                selected=self.selected, terminal_state_sha256=state_hash(self.model),
                optimizer_state_entries=len(self.optimizer.state),
                optimizer_state_steps=sorted({int(v['step'].item()) for v in self.optimizer.state.values() if 'step' in v}))
            (self.output/f'budget_{budget}ep.json').write_text(json.dumps(record, indent=2))
            print('BUDGET_SAVED', budget, json.dumps(record), flush=True)
        return metrics


def validation_steps(batches, epochs=3):
    return [0]+[offset+step for offset in range(0, batches*epochs, batches)
                for step in [math.ceil(batches*.25), math.ceil(batches*.5), batches]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/plan13_ablation')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    data = ROOT/'data/v3'
    data_manifest = json.loads((data/'manifest.json').read_text())
    assert data_manifest['version'] == 3 and data_manifest['protocol']['clock'] == 'Leader screen end; delta1min'
    sources = [Path(__file__), *[Path(__file__).with_name(n) for n in ['time_budget_v10.py', 'time_budget_v9.py',
        'time_budget_v8.py', 'kr_request_schema.py', 'run_exit_v7_joint.py', 'run_exit_v8_branches.py',
        'run_exit_v9_pooled.py', 'run_exit_v10_plan12.py', 'run_exit_v3.py', 'run_exit.py']]]
    config = dict(data=str(data), output=str(args.output), smoke=args.smoke, seeds=args.seeds,
        experiments={n: EXPERIMENTS[n] for n in args.experiments}, epochs=3, reported_epoch_budgets=[1, 3],
        training_strategy='joint', loss='BCE(gamma_step*content_stay,y_continue)',
        auxiliary_loss=None, frozen_budget=False, detach_gamma=False, warmup=None,
        primary_metrics=['overall.auc', 'logloss'], auxiliary_metric='intra_step_gauc',
        batch_size=4096, gradient_accumulation=1, optimizer='AdamW', budget_lr=.005, content_lr=.001,
        weight_decay=1e-6, zero_decay='Embedding,bias,one-dimensional parameters',
        scheduler=None, ema=False, device='cpu', threads=4,
        model='unchanged time_budget_v10.MixtureTimeBudgetModel; same shape/caps/init/pace as plan12',
        epoch_order='same physically shuffled train order repeated in each epoch',
        optimizer_continuity='one optimizer for 3 epochs; no restore/reinit at intermediate horizon',
        selection='initial and 25/50/100 percent of each epoch; best full-valid LogLoss within each budget',
        evaluation='all test evaluation occurs after the entire 3-epoch trajectory; no test-based selection',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        dataset_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
    config_path = args.output/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Changed source/config needs fresh output'
    config_path.write_text(json.dumps(config, indent=2))
    manager = request_manager(data/'schema', load=True)
    batches = load_data(data, manager, args.smoke)
    coverage = {}
    for split, loader in batches.items():
        ids = torch.cat([b['request_id'] for b in loader]).numpy()
        expected = pl.read_parquet(data/f'{split}.parquet', columns=['request_id'])['request_id'].to_numpy()[:len(ids)]
        assert np.array_equal(ids, expected)
        order = hashlib.sha256(ids.astype(expected.dtype).tobytes()).hexdigest()
        coverage[split] = dict(rows=len(ids), batches=len(loader), request_order_sha256=order)
        if not args.smoke:
            assert len(ids) == data_manifest['splits'][split]['rows']
            assert order == data_manifest['splits'][split]['request_order_sha256']
    (args.output/'coverage.json').write_text(json.dumps(coverage, indent=2))
    print('DATA', json.dumps(coverage), flush=True)
    summary = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        reference = {k: v.clone() for k, v in MixtureTimeBudgetModel(manager).state_dict().items()}
        for name in args.experiments:
            started = time.monotonic()
            key, options = f'{seed}/{name}', EXPERIMENTS[name]
            out = args.output/key
            out.mkdir(parents=True, exist_ok=True)
            if (out/'complete.json').exists():
                for budget in [1, 3]:
                    summary[f'{key}/{budget}ep'] = json.loads((out/f'{budget}ep/metrics.json').read_text())
                continue
            if (out/'joint/progress.jsonl').exists():
                raise RuntimeError(f'Incomplete run {out}; use fresh output')
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = MixtureTimeBudgetModel(manager, **options)
            changed = initial_alignment(model, reference)
            groups = joint_parameter_groups(model, .005)
            manifest = dict(initial_sha256=state_hash(model), common_initial_tensors_exact=True,
                initial_changed_tensors=changed, seed=seed, experiment=options,
                train_order_sha256=coverage['train']['request_order_sha256'],
                full_train=not args.smoke, full_valid=not args.smoke, full_test=not args.smoke,
                optimizer_groups=[dict(scope=g['scope'], lr=g['lr'], weight_decay=g['weight_decay'],
                    tensors=len(g['params']), elements=sum(p.numel() for p in g['params'])) for g in groups])
            (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
            phase = out/'joint'
            evaluator = EvaluatorManager(log_path=phase/'evaluations.log', title=f'plan13 {key}')
            evaluator.register('overall', Evaluator('auc', 'logloss'))
            for domain in range(5):
                evaluator.register(f'domain{domain}', Evaluator('auc', 'logloss'), DomainFilter('tab', domain))
            cfg = SimpleNamespace(device='cpu', max_epochs=3, ckpt_dir=str(phase), experiment_name=f'plan13_{name}_{seed}',
                dataset_name='kuairand_requests_v3', train_log_interval=200)
            trainer = BudgetTrainer(model, torch.optim.AdamW(groups, weight_decay=0), manager,
                TrainerDataLoaders(batches['train'], batches['valid']),
                TrainerComponents(evaluator_manager=evaluator, early_stopper=EarlyStopper(preferred_metrics=('logloss',))), cfg)
            print('START', key, json.dumps(options), flush=True)
            if model.mode == 'fusion':
                trainer.evaluate_epoch()
            trainer.run()
            steps = coverage['train']['batches']
            assert trainer.global_step == 3*steps
            assert all(p.requires_grad for p in model.parameters())
            progress = [json.loads(line) for line in (phase/'progress.jsonl').read_text().splitlines()]
            expected_steps = validation_steps(steps)
            # For a 2-batch smoke, 25% and 50% marks coincide (native trainer uses a set).
            expected_steps = list(dict.fromkeys(expected_steps))
            assert [r['step'] for r in progress] == expected_steps
            for budget in [1, 3]:
                checkpoint = torch.load(phase/f'best_within_{budget}ep.pt', map_location='cpu', weights_only=False)
                selection = min((r for r in progress if r['step'] <= budget*steps),
                                key=lambda r: r['valid']['overall']['logloss'])
                assert checkpoint['selection'] == selection
                audit = json.loads((phase/f'budget_{budget}ep.json').read_text())
                assert audit['optimizer_state_steps'] == [budget*steps]
                model.load_state_dict(checkpoint['state_dict'])
                pred = predict_joint(model, batches['test'])
                sub = out/f'{budget}ep'
                sub.mkdir(exist_ok=True)
                pred.write_parquet(sub/'predictions.parquet')
                cal = calibration(pred)
                (sub/'calibration.json').write_text(json.dumps(cal, indent=2))
                (sub/'diagnostics.json').write_text(json.dumps(diagnostics(model, batches['test'], pred), indent=2))
                reloaded = MixtureTimeBudgetModel(manager, **options)
                reloaded.load_state_dict(checkpoint['state_dict'])
                assert state_hash(model) == state_hash(reloaded)
                assert predict_joint(reloaded, batches['test']).equals(pred)
                metric = scores(pred)
                early = pred.filter(pl.col('req_idx_raw').is_between(2, 10))
                early_scores = scores(early)
                metric.update(steps2to10=dict(rows=early.height, auc=early_scores['overall.auc'],
                    logloss=early_scores['logloss'], bias=calibration(early)['mean_prediction_minus_observed']),
                    full_train=not args.smoke, full_valid=not args.smoke, full_test=not args.smoke,
                    training_budget_epochs=budget, budget_optimizer_steps=budget*steps,
                    trajectory_optimizer_steps_executed=trainer.global_step, selected=selection,
                    checkpoint_reload_predictions_exact=True, selected_state_sha256=state_hash(model),
                    calibration={k: v for k, v in cal.items() if k != 'by_step'})
                if model.mode == 'ours':
                    assert (pred['prediction']-pred['gamma']*pred['content_stay']).abs().max() == 0
                (sub/'metrics.json').write_text(json.dumps(metric, indent=2))
                summary[f'{key}/{budget}ep'] = metric
                print('RESULT', f'{key}/{budget}ep', json.dumps({k: metric[k] for k in
                    ['overall.auc', 'logloss', 'intra_step_gauc', 'selected']}), flush=True)
            (out/'complete.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-started,
                trajectory_steps=trainer.global_step, tested_budgets=[1, 3]), indent=2))
            (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', args.output/'summary.json', flush=True)


if __name__ == '__main__':
    main()
