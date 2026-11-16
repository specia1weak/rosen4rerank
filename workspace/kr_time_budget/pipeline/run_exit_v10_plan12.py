"""Argon 12: K=2 survival mixtures, then screen-only clock and pace ablations."""
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
from rosen.evaluate.evaluator import Evaluator
from rosen.evaluate.manager import EvaluatorManager, DomainFilter
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.run_exit import state_hash
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores
from workspace.kr_time_budget.pipeline.run_exit_v7_joint import JointTrainer, load_data, predict_joint
from workspace.kr_time_budget.pipeline.run_exit_v8_branches import describe
from workspace.kr_time_budget.pipeline.time_budget_v8 import joint_parameter_groups
from workspace.kr_time_budget.pipeline.time_budget_v10 import (
    MixtureTimeBudgetModel, mixture_parts, pace_feature,
)
from workspace.kr_time_budget.pipeline.run_exit_v9_pooled import calibration
from workspace.kr_time_budget.pipeline.time_budget_v9 import weibull_parameters

ROOT = Path('workspace/kr_time_budget')
EXPERIMENTS = {
    'single_dual': dict(mode='ours', mixture=False, step_only=False, add_pace=False),
    'mixture_dual': dict(mode='ours', mixture=True, step_only=False, add_pace=False),
    'mixture_step': dict(mode='ours', mixture=True, step_only=True, add_pace=False),
    'mixture_step_pace': dict(mode='ours', mixture=True, step_only=True, add_pace=True),
    'fusion': dict(mode='fusion', mixture=False, step_only=False, add_pace=False),
    'fusion_pace': dict(mode='fusion', mixture=False, step_only=False, add_pace=True),
}


def diagnostics(model, batches, pred):
    result = {'pace_log1p': describe(pace_feature({k: torch.cat([b[k] for b in batches])
        for k in ['elapsed_min_raw', 'req_idx_raw']}).numpy())}
    if model.mode == 'fusion':
        return result
    with torch.no_grad():
        raw = torch.cat([model.budget_raw(b) for b in batches])
    result.update(components={c: describe(pred[c].to_numpy()) for c in
        ['gamma_time', 'gamma_step', 'gamma', 'content_stay']}, clocks={})
    for i, clock in enumerate(['time', 'step']):
        if clock == 'time' and model.step_only:
            result['clocks'][clock] = {'active': False}
            continue
        if model.mixture:
            r = raw[:, i*6:i*6+6]
            t = torch.from_numpy(pred['elapsed_min_raw'].to_numpy().copy()) if clock == 'time' else torch.from_numpy(pred['req_idx_raw'].to_numpy().copy())-1
            parts = mixture_parts(r, t)
            components = []
            for k in range(2):
                components.append(dict(scale=describe(parts['scale'][:, k].numpy()),
                    shape=describe(parts['shape'][:, k].numpy()), prior=describe(parts['prior'][:, k].numpy()),
                    posterior=describe(parts['posterior'][:, k].numpy()),
                    scale_upper_clamped_fraction=float((r[:, 2*k] >= 4.5).float().mean())))
            result['clocks'][clock] = dict(active=True, components=components,
                posterior_component0_near_extreme_fraction=float(((parts['posterior'][:, 0] < .01) | (parts['posterior'][:, 0] > .99)).float().mean()))
        else:
            scale, shape = weibull_parameters(raw[:, i*2:i*2+2], True)
            result['clocks'][clock] = dict(active=True, scale=describe(scale.numpy()), shape=describe(shape.numpy()))
    pairs = pred.sort(['session_id', 'req_idx_raw']).with_columns(
        pl.col('elapsed_min_raw').diff().over('session_id').alias('dt'),
        pl.col('log_survival').diff().over('session_id').alias('ds'))
    eligible = pairs.filter(pl.col('dt') >= 0)
    result['clock_diagnostics'] = dict(nonrewinding_pairs=eligible.height,
        clock_rewinds=pairs.filter(pl.col('dt') < 0).height,
        logS_increases=eligible.filter(pl.col('ds') > 1e-6).height)
    assert result['clock_diagnostics']['logS_increases'] == 0
    return result


def initial_alignment(model, reference):
    excluded = []
    for name, value in model.state_dict().items():
        old = reference[name]
        if name in {'budget_net.2.weight', 'budget_net.2.bias'} and model.mixture:
            excluded.append(name)
        elif name in {'content_net.0.weight', 'fusion_net.0.weight'} and model.add_pace:
            assert torch.equal(value[:, :-1], old) and value[:, -1].abs().sum() == 0
            excluded.append(name)
        else:
            assert torch.equal(value, old), name
    return excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/exit_v10_plan12')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    data = ROOT/'data/v3'
    data_manifest = json.loads((data/'manifest.json').read_text())
    assert data_manifest['version'] == 3 and data_manifest['protocol']['clock'] == 'Leader screen end; delta1min'
    sources = [Path(__file__), *[Path(__file__).with_name(n) for n in ['time_budget_v10.py', 'time_budget_v9.py', 'time_budget_v8.py',
        'kr_request_schema.py', 'run_exit_v7_joint.py', 'run_exit_v8_branches.py', 'run_exit_v9_pooled.py', 'run_exit_v3.py', 'run_exit.py']]]
    config = dict(data=str(data), output=str(args.output), smoke=args.smoke, seeds=args.seeds,
        experiments={n: EXPERIMENTS[n] for n in args.experiments},
        primary_metrics=['overall.auc', 'logloss'], auxiliary_metric='intra_step_gauc',
        user_gauc_role='diagnostic only; never used for loss or checkpoint selection',
        training_strategy='joint', loss='BCE(budget_gamma*stay,y_continue); budget_gamma=gamma_step or gamma_time*gamma_step',
        auxiliary_loss=None, warmup=None, detach_gamma=False, frozen_budget=False,
        shape_floor=.3, shape_ceiling=5., scale_log_offset=1.6, initial_content_bias=3.,
        mixture_components=2, mixture_scale_caps='both clocks retain original caps',
        mixture_initialization='copy per-clock single head twice; scale raw bias offsets +-log(2); equal prior',
        derived_feature='log1p(max(elapsed_min_raw,0)/max(req_idx_raw,1)); current screen end',
        pace_initialization='zero appended first-layer column; common columns copied exactly',
        epochs=1, batch_size=4096, gradient_accumulation=1, optimizer='AdamW',
        budget_lr=.005, content_lr=.001, weight_decay=1e-6, zero_decay='Embedding,bias,one-dimensional parameters',
        scheduler=None, ema=False, device='cpu', threads=4,
        selection='full valid LogLoss at initial/25/50/100 percent, then full test',
        preprocessing='v3 unchanged; physical elapsed_min aliases elapsed_min_raw; req_idx_raw preserved',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        dataset_manifest_sha256=hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest())
    config_path = args.output/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Changed source/config needs fresh output'
    config_path.write_text(json.dumps(config, indent=2))
    manager = request_manager(data/'schema', load=True)
    assert manager.get_setting('elapsed_min').min_val == 0 and manager.get_setting('elapsed_min').max_val == 1
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
        reference_state = {k: v.clone() for k, v in MixtureTimeBudgetModel(manager).state_dict().items()}
        for name in args.experiments:
            started = time.monotonic()
            key, options = f'{seed}/{name}', EXPERIMENTS[name]
            out = args.output/key
            out.mkdir(parents=True, exist_ok=True)
            if (out/'metrics.json').exists():
                summary[key] = json.loads((out/'metrics.json').read_text())
                continue
            if (out/'joint/progress.jsonl').exists():
                raise RuntimeError(f'Incomplete run {out}; use fresh output')
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = MixtureTimeBudgetModel(manager, **options)
            excluded = initial_alignment(model, reference_state)
            groups = joint_parameter_groups(model, .005)
            manifest = dict(initial_sha256=state_hash(model), common_initial_tensors_exact=True, initial_changed_tensors=excluded,
                train_order_sha256=coverage['train']['request_order_sha256'], full_train=not args.smoke,
                full_valid=not args.smoke, full_test=not args.smoke, seed=seed, experiment=options,
                budget_fields=list(model.budget_view.field_slices), content_fields=list(model.content_view.field_slices),
                optimizer_groups=[dict(scope=g['scope'], lr=g['lr'], weight_decay=g['weight_decay'],
                    tensors=len(g['params']), elements=sum(p.numel() for p in g['params'])) for g in groups])
            (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
            initial_pred = predict_joint(model, batches['test'])
            initial_pred.write_parquet(out/'initial_predictions.parquet')
            (out/'initial_diagnostics.json').write_text(json.dumps(diagnostics(model, batches['test'], initial_pred), indent=2))
            phase = out/'joint'
            evaluator = EvaluatorManager(log_path=phase/'evaluations.log', title=f'plan12 mixture {key}')
            evaluator.register('overall', Evaluator('auc', 'logloss'))
            for domain in range(5):
                evaluator.register(f'domain{domain}', Evaluator('auc', 'logloss'), DomainFilter('tab', domain))
            cfg = SimpleNamespace(device='cpu', max_epochs=1, ckpt_dir=str(phase), experiment_name=f'exit_v10_{name}_{seed}',
                dataset_name='kuairand_requests_v3', train_log_interval=100)
            trainer = JointTrainer(model, torch.optim.AdamW(groups, weight_decay=0), manager,
                TrainerDataLoaders(batches['train'], batches['valid']),
                TrainerComponents(evaluator_manager=evaluator, early_stopper=EarlyStopper(preferred_metrics=('logloss',))), cfg)
            print('START', key, json.dumps(options), flush=True)
            if model.mode == 'fusion':
                trainer.evaluate_epoch()
            trainer.run()
            assert trainer.global_step == coverage['train']['batches']
            assert all(p.requires_grad for p in model.parameters())
            pred = predict_joint(model, batches['test'])
            pred.write_parquet(out/'predictions.parquet')
            (out/'final_diagnostics.json').write_text(json.dumps(diagnostics(model, batches['test'], pred), indent=2))
            cal = calibration(pred)
            (out/'calibration.json').write_text(json.dumps(cal, indent=2))
            reloaded = MixtureTimeBudgetModel(manager, **options)
            reloaded.load_state_dict(torch.load(phase/'best.pt', map_location='cpu', weights_only=False)['state_dict'])
            assert state_hash(reloaded) == state_hash(model)
            assert predict_joint(reloaded, batches['test']).equals(pred)
            metric = scores(pred)
            early = pred.filter(pl.col('req_idx_raw').is_between(2, 10))
            early_cal = calibration(early)
            metric['steps2to10'] = dict(rows=early.height, auc=scores(early)['overall.auc'],
                logloss=scores(early)['logloss'], bias=early_cal['mean_prediction_minus_observed'])
            metric.update(full_train=not args.smoke, full_valid=not args.smoke, full_test=not args.smoke,
                optimizer_steps_executed=trainer.global_step, selected=trainer.selected,
                checkpoint_reload_predictions_exact=True, selected_state_sha256=state_hash(model),
                calibration={k: v for k, v in cal.items() if k != 'by_step'}, wall_seconds=time.monotonic()-started)
            if model.mode == 'ours':
                assert (pred['prediction']-pred['gamma']*pred['content_stay']).abs().max() == 0
                metric['product_max_abs_error'] = 0.
            (out/'metrics.json').write_text(json.dumps(metric, indent=2))
            summary[key] = metric
            (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
            print('RESULT', key, json.dumps({k: metric[k] for k in ['overall.auc', 'logloss', 'intra_step_gauc', 'wall_seconds']}), flush=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', args.output/'summary.json', flush=True)


if __name__ == '__main__':
    main()
