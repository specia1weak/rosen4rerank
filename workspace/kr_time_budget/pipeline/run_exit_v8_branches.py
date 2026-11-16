"""Argon 9: paired, full-data ablations and two joint-training branches."""
import argparse
from dataclasses import asdict, dataclass
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
from workspace.kr_time_budget.pipeline.time_budget_v8 import (
    DualBranchModel, weibull_parameters, single_weibull_survival, joint_parameter_groups,
)

ROOT = Path('workspace/kr_time_budget')


@dataclass(frozen=True)
class Experiment:
    mode: str = 'ours'
    scale_mapping: str = 'log'
    budget_lr: float = .005
    shape_floor: float = 1.05
    content_bias: float = 3.0
    bounded_content: bool = False

    def build(self, manager):
        options = asdict(self)
        options.pop('budget_lr')
        return DualBranchModel(manager, **options)


# Declared before evaluating test; adjacent stages isolate one explicitly named change.
EXPERIMENTS = {
    'reference': Experiment(scale_mapping='legacy', budget_lr=.001, content_bias=3.5),
    'log_scale': Experiment(budget_lr=.001, content_bias=3.5),
    'faster_budget': Experiment(content_bias=3.5),
    'monotone': Experiment(),
    'flexible_shape': Experiment(shape_floor=.3),
    'bounded_content': Experiment(bounded_content=True),
    'fusion': Experiment(mode='fusion', scale_mapping='legacy', budget_lr=.001, content_bias=3.5),
}


def describe(values):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all()
    return {'mean': float(a.mean()), 'std': float(a.std()), 'min': float(a.min()),
            'p05': float(np.quantile(a, .05)), 'p50': float(np.quantile(a, .5)),
            'p95': float(np.quantile(a, .95)), 'max': float(a.max())}


def diagnostics(model, batches, prediction):
    """Measurements, not an additional training objective or optimizer update."""
    if model.mode == 'fusion':
        return {}
    with torch.no_grad():
        raw = torch.cat([model.budget_raw(b) for b in batches])
        logit = torch.cat([model.content_net(model.content_view(b)).squeeze(-1) for b in batches])
    result = {'components': {c: describe(prediction[c].to_numpy()) for c in
                             ['gamma_time', 'gamma_step', 'gamma', 'content_stay', 'prediction']},
              'content_logit': describe(logit.numpy()), 'clocks': {}}
    for name, r in [('time', raw[:, :2]), ('step', raw[:, 2:])]:
        scale, shape = weibull_parameters(r, model.scale_mapping, model.shape_floor)
        result['clocks'][name] = {'scale': describe(scale.numpy()), 'shape': describe(shape.numpy()),
            'shape_below_one_fraction': float((shape < 1).float().mean()),
            'shape_upper_clamp_fraction': float((shape >= 5).float().mean()),
            'log_scale_clamped_fraction': float(((r[:, 0] <= -2) | (r[:, 0] >= 4.5)).float().mean())
                if model.scale_mapping == 'log' else None}
    # Per-row derivatives on the same first batch: compare actual raw/logit paths,
    # without inferring AdamW speed from differently-sized parameter gradient norms.
    b = batches[0]
    r = raw[:len(b['y_continue'])].detach().requires_grad_(True)
    z = logit[:len(b['y_continue'])].detach().requires_grad_(True)
    gt = single_weibull_survival(r[:, :2], b['elapsed_min_raw'], torch.ones_like(z), model.scale_mapping, model.shape_floor)[0]
    gs = single_weibull_survival(r[:, 2:], b['req_idx_raw'].float()-1, torch.ones_like(z), model.scale_mapping, model.shape_floor)[0]
    gamma, stay = gt*gs, model.content_stay(z)
    dg = torch.autograd.grad(gamma.sum(), r, retain_graph=True)[0]
    ds = torch.autograd.grad(stay.sum(), z, retain_graph=True)[0]
    loss = torch.nn.functional.binary_cross_entropy((gamma*stay).clamp(1e-6, 1-1e-6), b['y_continue'].float(), reduction='sum')
    grad_r, grad_z = torch.autograd.grad(loss, (r, z))
    result['first_batch_derivatives'] = {
        'abs_mean_dgamma_draw': dg.abs().mean(0).tolist(),
        'abs_mean_dstay_dlogit': float(ds.abs().mean()),
        'abs_mean_drow_bce_draw': grad_r.abs().mean(0).tolist(),
        'abs_mean_drow_bce_dlogit': float(grad_z.abs().mean()),
        'rows': len(z), 'source': 'first batch of full test; diagnostic only',
    }
    pairs = prediction.sort(['session_id', 'req_idx_raw']).with_columns(
        pl.col('elapsed_min_raw').diff().over('session_id').alias('dt'),
        pl.col('gamma').diff().over('session_id').alias('dg'),
        pl.col('log_survival').diff().over('session_id').alias('ds'),
        pl.col('prediction').diff().over('session_id').alias('dp'))
    eligible = pairs.filter(pl.col('dt') >= 0)
    result['monotonicity'] = {'nonrewinding_pairs': eligible.height, 'clock_rewinds': pairs.filter(pl.col('dt') < 0).height,
        'gamma_increases': eligible.filter(pl.col('dg') > 1e-7).height,
        'logS_increases': eligible.filter(pl.col('ds') > 1e-6).height,
        'prediction_increases': eligible.filter(pl.col('dp') > 1e-7).height}
    assert result['monotonicity']['logS_increases'] == 0
    if model.shape_floor >= 1.05:
        assert result['monotonicity']['gamma_increases'] == 0
    # Fixed-checkpoint component readouts: no retraining, no causal attribution.
    result['component_readouts'] = {}
    for component in ['gamma', 'content_stay']:
        metric = scores(prediction.with_columns(pl.col(component).alias('prediction')))
        result['component_readouts'][component] = {'auc': metric['overall.auc'], 'user_gauc': metric['user_gauc']['auc']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/exit_v8_branches')
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    data = ROOT/'data/v3'
    dataset_manifest = json.loads((data/'manifest.json').read_text())
    assert dataset_manifest['version'] == 3 and dataset_manifest['protocol']['clock'] == 'Leader screen end; delta1min'
    sources = [Path(__file__), Path(__file__).with_name('time_budget_v8.py'), Path(__file__).with_name('kr_request_schema.py'),
               Path(__file__).with_name('run_exit_v7_joint.py'), Path(__file__).with_name('run_exit_v3.py')]
    config = {'data': str(data), 'output': str(args.output), 'smoke': args.smoke, 'seeds': args.seeds,
        'experiments': {name: asdict(EXPERIMENTS[name]) for name in args.experiments},
        'training_strategy': 'joint', 'loss': 'BCE(gamma_time*gamma_step*stay,y_continue)',
        'content_auxiliary_loss': None, 'warmup': None, 'detach_gamma': False, 'frozen_budget': False,
        'epochs': 1, 'batch_size': 4096, 'gradient_accumulation': 1, 'optimizer': 'AdamW',
        'content_lr': .001, 'weight_decay': 1e-6, 'zero_decay': 'Embedding,bias,one-dimensional parameters',
        'scheduler': None, 'ema': False, 'device': 'cpu', 'threads': 4,
        'selection': 'full valid LogLoss at initial/25/50/100 percent; full test on selected checkpoint',
        'preprocessing': 'v3 unchanged; elapsed_min_raw aliases physical elapsed_min; raw step uses req_idx_raw',
        'dataset_manifest_sha256': hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'interpretation': 'paired ablation chain; two branch packages vs Fusion are exploratory architecture comparisons'}
    cfg_path = args.output/'config.json'
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text()) == config, 'Changed source/configuration needs fresh output'
    cfg_path.write_text(json.dumps(config, indent=2))
    manager = request_manager(data/'schema', load=True)
    assert manager.get_setting('elapsed_min').min_val == 0 and manager.get_setting('elapsed_min').max_val == 1
    batches = load_data(data, manager, args.smoke)
    coverage = {}
    for split, loader in batches.items():
        ids = torch.cat([b['request_id'] for b in loader]).numpy()
        expected = pl.read_parquet(data/f'{split}.parquet', columns=['request_id'])['request_id'].to_numpy()[:len(ids)]
        assert np.array_equal(ids, expected)
        h = hashlib.sha256(ids.astype(expected.dtype).tobytes()).hexdigest()
        coverage[split] = dict(rows=len(ids), batches=len(loader), request_order_sha256=h)
        if not args.smoke:
            assert len(ids) == dataset_manifest['splits'][split]['rows']
            assert h == dataset_manifest['splits'][split]['request_order_sha256']
    (args.output/'coverage.json').write_text(json.dumps(coverage, indent=2))
    print('DATA', json.dumps(coverage), flush=True)
    summary = {}
    for seed in args.seeds:
        torch.manual_seed(seed)
        reference = EXPERIMENTS['reference'].build(manager)
        reference_state = {k: v.clone() for k, v in reference.state_dict().items()}
        for name in args.experiments:
            started = time.monotonic()
            experiment = EXPERIMENTS[name]
            key = f'{seed}/{name}'
            out = args.output/key
            out.mkdir(parents=True, exist_ok=True)
            if (out/'metrics.json').exists():
                summary[key] = json.loads((out/'metrics.json').read_text())
                continue
            if (out/'joint/progress.jsonl').exists():
                raise RuntimeError(f'Incomplete training at {out}; use fresh output')
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = experiment.build(manager)
            differing = [k for k, v in model.state_dict().items() if not torch.equal(v, reference_state[k])]
            assert set(differing) <= {'prior_stay', 'content_net.4.bias'}
            groups = joint_parameter_groups(model, experiment.budget_lr)
            group_meta = [dict(scope=g['scope'], lr=g['lr'], weight_decay=g['weight_decay'],
                               tensors=len(g['params']), elements=sum(p.numel() for p in g['params'])) for g in groups]
            manifest = dict(initial_sha256=state_hash(model), initial_differences_from_reference=differing,
                initial_all_other_tensors_exact=True, train_order_sha256=coverage['train']['request_order_sha256'],
                full_train=not args.smoke, full_valid=not args.smoke, full_test=not args.smoke,
                optimizer_groups=group_meta, experiment=asdict(experiment), seed=seed,
                budget_fields=list(model.budget_view.field_slices), content_fields=list(model.content_view.field_slices))
            (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
            initial_prediction = predict_joint(model, batches['test'])
            initial_prediction.write_parquet(out/'initial_predictions.parquet')
            initial_diagnostics = diagnostics(model, batches['test'], initial_prediction)
            (out/'initial_diagnostics.json').write_text(json.dumps(initial_diagnostics, indent=2))
            phase = out/'joint'
            evaluator = EvaluatorManager(log_path=phase/'evaluations.log', title=f'v8 {key}')
            evaluator.register('overall', Evaluator('auc', 'gauc', 'logloss'))
            for domain in range(5):
                evaluator.register(f'domain{domain}', Evaluator('auc', 'gauc', 'logloss'), DomainFilter('tab', domain))
            cfg = SimpleNamespace(device='cpu', max_epochs=1, ckpt_dir=str(phase), experiment_name=f'exit_v8_{name}_{seed}',
                dataset_name='kuairand_requests_v3', train_log_interval=100)
            trainer = JointTrainer(model, torch.optim.AdamW(groups, weight_decay=0), manager,
                TrainerDataLoaders(batches['train'], batches['valid']),
                TrainerComponents(evaluator_manager=evaluator, early_stopper=EarlyStopper(preferred_metrics=('logloss',))), cfg)
            print('START', key, json.dumps(asdict(experiment)), flush=True)
            if model.mode == 'fusion':
                trainer.evaluate_epoch()
            trainer.run()
            assert trainer.global_step == coverage['train']['batches']
            assert all(p.requires_grad for p in model.parameters())
            prediction = predict_joint(model, batches['test'])
            prediction.write_parquet(out/'predictions.parquet')
            final_diagnostics = diagnostics(model, batches['test'], prediction)
            (out/'final_diagnostics.json').write_text(json.dumps(final_diagnostics, indent=2))
            reloaded = experiment.build(manager)
            reloaded.load_state_dict(torch.load(phase/'best.pt', map_location='cpu', weights_only=False)['state_dict'])
            assert state_hash(model) == state_hash(reloaded)
            reload_pred = predict_joint(reloaded, batches['test'])
            assert prediction.equals(reload_pred)
            metric = scores(prediction)
            metric.update(full_train=not args.smoke, full_valid=not args.smoke, full_test=not args.smoke,
                selected=trainer.selected, optimizer_steps_executed=trainer.global_step,
                selected_state_sha256=state_hash(model), checkpoint_reload_predictions_exact=True,
                wall_seconds=time.monotonic()-started)
            if model.mode == 'ours':
                assert (prediction['prediction']-prediction['gamma']*prediction['content_stay']).abs().max() == 0
                metric['product_max_abs_error'] = 0.
                metric['gamma_std'] = final_diagnostics['components']['gamma']['std']
            (out/'metrics.json').write_text(json.dumps(metric, indent=2))
            summary[key] = metric
            (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
            print('RESULT', key, json.dumps({k: metric[k] for k in ['overall.auc', 'user_gauc', 'logloss', 'wall_seconds']}), flush=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('COMPLETE', args.output/'summary.json', flush=True)


if __name__ == '__main__':
    main()
