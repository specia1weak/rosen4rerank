"""Full KR v3 joint-product training and a matched Fusion baseline.

One loss only: BCE(gamma_time * gamma_step * stay, y_continue).
No warmup, detach, frozen tower, auxiliary content loss or dataset recoding.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from rosen.core.train import EarlyStopper
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from rosen.data.dataset import ParquetStreamDataset
from rosen.evaluate.evaluator import Evaluator
from rosen.evaluate.manager import EvaluatorManager, DomainFilter
from rosen.utils.optimize import split_model_params_by_decay
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager, EXTRA_COLUMNS
from workspace.kr_time_budget.pipeline.run_exit import state_hash
from workspace.kr_time_budget.pipeline.run_exit_v3 import ExitTrainer, scores, on_device
from workspace.kr_time_budget.pipeline.time_budget_v7 import TimeBudgetModel

ROOT = Path('workspace/kr_time_budget')


class JointTrainer(ExitTrainer):
    """Native training with an audit of the first real optimizer update."""
    def train_step(self, batch, ctx):
        first_step = self.global_step == 0
        if first_step:
            model = self.model
            if model.mode == 'ours':
                watched = {
                    'budget_head': model.budget_net[-1].weight,
                    'content_head': model.content_net[-1].weight,
                    'budget_user_embedding': model.budget_embedding.emb_modules['user_id'].embedding.weight,
                    'content_user_embedding': model.omni_embedding.emb_modules['user_id'].embedding.weight,
                }
            else:
                watched = {'fusion_head': model.fusion_net[-1].weight,
                           'user_embedding': model.omni_embedding.emb_modules['user_id'].embedding.weight}
            before = {name: p.detach().clone() for name, p in watched.items()}
        loss = super().train_step(batch, ctx)
        if first_step:
            audit = {}
            for name, parameter in watched.items():
                assert parameter.requires_grad and parameter.grad is not None
                grad = parameter.grad.norm().item()
                change = (parameter.detach()-before[name]).abs().max().item()
                assert np.isfinite(grad) and grad > 0 and change > 0, name
                audit[name] = {'gradient_norm_after_clipping': grad, 'parameter_max_abs_update': change}
            (self.output/'first_update.json').write_text(json.dumps(audit, indent=2))
            print('FIRST_UPDATE', self.model.mode, json.dumps(audit), flush=True)
        return loss


def load_data(data, manager, smoke=False):
    batches = {}
    for split in ['train', 'valid', 'test']:
        # Only v3 is accepted below: elapsed_min is physical minutes in this schema.
        frame = pl.scan_parquet(data/f'{split}.parquet').with_columns(pl.col('elapsed_min').alias('elapsed_min_raw'))
        if smoke:
            frame = frame.head(8192 if split == 'train' else 4096)
        dataset = ParquetStreamDataset(frame, manager, batch_size=4096, shuffle=False,
                                       drop_last=False, extra_col_names=EXTRA_COLUMNS+['elapsed_min_raw'])
        batches[split] = list(DataLoader(dataset, batch_size=None, num_workers=0))
    return batches


@torch.no_grad()
def predict_joint(model, batches):
    model.eval()
    parts = []
    for batch in batches:
        b = on_device(batch, 'cpu')
        data = {c: b[c].numpy() for c in ['request_id', 'session_id', 'user_id_raw', 'tab_raw',
                    'req_idx_raw', 'y_continue', 'elapsed_min', 'elapsed_min_raw']}
        prediction = model.predict(b)
        assert torch.isfinite(prediction).all()
        data['prediction'] = prediction.numpy()
        if model.mode == 'ours':
            product, gamma, stay, survival, log_s = model.components(b)
            assert torch.equal(product, prediction) and gamma.requires_grad is False  # no_grad eval
            clocks = model.clock_components(b)
            data.update(gamma=gamma.numpy(), content_stay=stay.numpy(), survival=survival.numpy(),
                        log_survival=log_s.numpy(), gamma_time=clocks['time'][0].numpy(), gamma_step=clocks['step'][0].numpy())
            assert torch.equal(gamma, clocks['time'][0]*clocks['step'][0])
        parts.append(pl.DataFrame(data))
    return pl.concat(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'runs/exit_v7_joint')
    parser.add_argument('--models', nargs='+', choices=['ours', 'fusion'], default=['ours', 'fusion'])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    data = ROOT/'data/v3'
    dataset_manifest = json.loads((data/'manifest.json').read_text())
    assert dataset_manifest['version'] == 3 and dataset_manifest['protocol']['clock'] == 'Leader screen end; delta1min'
    sources = [Path(__file__), Path(__file__).with_name('time_budget_v7.py'),
               Path(__file__).with_name('kr_request_schema.py'), Path(__file__).with_name('time_budget_v4.py'),
               Path(__file__).with_name('run_exit_v3.py')]
    config = {
        'data': str(data), 'output': str(args.output), 'models': args.models, 'smoke': args.smoke,
        'training_strategy': 'joint', 'clock': 'dual', 'loss': 'BCE(gamma_time*gamma_step*stay,y_continue)',
        'content_auxiliary_loss': None, 'warmup': None, 'frozen_budget': False, 'detach_gamma': False,
        'seed': 2026, 'epochs': 1, 'batch_size': 4096, 'physical_effective_batch': 4096,
        'gradient_accumulation': 1, 'optimizer': 'AdamW', 'lr': .001, 'weight_decay': 1e-6,
        'zero_decay': 'Embedding,bias,one-dimensional parameters', 'scheduler': None, 'ema': False,
        'device': 'cpu', 'threads': 4, 'initial_content_bias': 3.5,
        'selection': 'full valid LogLoss at initial/25/50/100 percent; selected checkpoint full test',
        'preprocessing': 'v3 unchanged; lazy elapsed_min_raw alias of physical elapsed_min; no new normalization',
        'dataset_manifest_sha256': hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest(),
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'comparison': 'same initial tensors/data/order/updates; different routing, functional form and active parameters; single-seed screening',
    }
    cfg_path = args.output/'config.json'
    if cfg_path.exists():
        assert json.loads(cfg_path.read_text()) == config, 'Use a fresh output for changed configuration/source'
    cfg_path.write_text(json.dumps(config, indent=2))
    manager = request_manager(data/'schema', load=True)
    assert manager.get_setting('elapsed_min').min_val == 0 and manager.get_setting('elapsed_min').max_val == 1
    batches = load_data(data, manager, args.smoke)
    coverage = {}
    for split, loader in batches.items():
        ids = torch.cat([b['request_id'] for b in loader]).numpy()
        # Dataset manifest hashes the parquet's UInt32 representation; tensor extras are Int64.
        raw_ids = pl.read_parquet(data/f'{split}.parquet', columns=['request_id'])['request_id'].to_numpy()
        if args.smoke:
            raw_ids = raw_ids[:len(ids)]
        assert np.array_equal(ids, raw_ids), f'{split}: loader changed request order'
        order_hash = hashlib.sha256(ids.astype(raw_ids.dtype).tobytes()).hexdigest()
        coverage[split] = {'rows': len(ids), 'batches': len(loader), 'request_order_sha256': order_hash}
        if not args.smoke:
            assert len(ids) == dataset_manifest['splits'][split]['rows']
            assert order_hash == dataset_manifest['splits'][split]['request_order_sha256']
    (args.output/'coverage.json').write_text(json.dumps(coverage, indent=2))
    print('DATA', json.dumps(coverage), flush=True)
    initial_state = None
    summary = {}
    for mode in args.models:
        started = time.monotonic()
        out = args.output/mode
        out.mkdir(exist_ok=True)
        if (out/'metrics.json').exists():
            summary[mode] = json.loads((out/'metrics.json').read_text())
            continue
        if (out/'joint/progress.jsonl').exists():
            raise RuntimeError(f'Incomplete run {out}; use a fresh output')
        torch.manual_seed(2026)
        np.random.seed(2026)
        model = TimeBudgetModel(manager, mode=mode, clock='dual', training_strategy='joint')
        if initial_state is None:
            initial_state = {name: value.clone() for name, value in model.state_dict().items()}
        else:
            assert all(torch.equal(initial_state[name], value) for name, value in model.state_dict().items())
        initial_hash = state_hash(model)
        manifest = {'initial_sha256': initial_hash, 'common_initial_tensors_exact': True,
                    'train_order_sha256': coverage['train']['request_order_sha256'],
                    'full_train': not args.smoke, 'full_valid': not args.smoke, 'full_test': not args.smoke,
                    'train_rows': coverage['train']['rows'], 'batch_size': 4096,
                    'budget_fields': list(model.budget_view.field_slices),
                    'content_fields': list(model.content_view.field_slices),
                    'fusion_fields': list(model.fusion_view.field_slices)}
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        phase = out/'joint'
        evaluator = EvaluatorManager(log_path=phase/'evaluations.log', title=f'v7 joint {mode}')
        evaluator.register('overall', Evaluator('auc', 'gauc', 'logloss'))
        for domain in range(5):
            evaluator.register(f'domain{domain}', Evaluator('auc', 'gauc', 'logloss'), DomainFilter('tab', domain))
        cfg = SimpleNamespace(device='cpu', max_epochs=1, ckpt_dir=str(phase), experiment_name=f'exit_v7_joint_{mode}',
                              dataset_name='kuairand_requests_v3', train_log_interval=50)
        optimizer = torch.optim.AdamW(split_model_params_by_decay(model, weight_decay=1e-6), lr=.001, weight_decay=0)
        trainer = JointTrainer(model, optimizer, manager, TrainerDataLoaders(batches['train'], batches['valid']),
                               TrainerComponents(evaluator_manager=evaluator, early_stopper=EarlyStopper(preferred_metrics=('logloss',))), cfg)
        print('START', mode, initial_hash, flush=True)
        if mode != 'ours':
            trainer.evaluate_epoch()  # Match the initial-checkpoint selection afforded Ours.
        trainer.run()
        assert trainer.global_step == coverage['train']['batches']
        if mode == 'ours':
            assert all(p.requires_grad for module in model.budget_modules for p in module.parameters())
        pred = predict_joint(model, batches['test'])
        pred.write_parquet(out/'predictions.parquet')
        metric = scores(pred)
        metric.update(full_train=not args.smoke, full_valid=not args.smoke, full_test=not args.smoke,
                      optimizer_steps_executed=trainer.global_step, selected=trainer.selected,
                      selected_state_sha256=state_hash(model), wall_seconds=time.monotonic()-started)
        # Independently reload the selected checkpoint and verify every test prediction.
        reloaded = TimeBudgetModel(manager, mode=mode, clock='dual', training_strategy='joint')
        checkpoint = torch.load(phase/'best.pt', map_location='cpu', weights_only=False)
        reloaded.load_state_dict(checkpoint['state_dict'])
        reloaded.set_stage(2)
        reload_pred = predict_joint(reloaded, batches['test'])
        assert pred['request_id'].equals(reload_pred['request_id'])
        assert pred['prediction'].equals(reload_pred['prediction'])
        assert state_hash(model) == state_hash(reloaded)
        metric['checkpoint_reload_predictions_exact'] = True
        if mode == 'ours':
            assert pred['gamma'].equals(reload_pred['gamma'])
            assert (pred['prediction']-pred['gamma']*pred['content_stay']).abs().max() == 0
            metric['product_max_abs_error'] = 0.
            metric['components'] = {name: {'mean': pred[name].mean(), 'std': pred[name].std(),
                                          'min': pred[name].min(), 'max': pred[name].max()}
                                    for name in ['gamma_time', 'gamma_step', 'gamma', 'content_stay']}
            pairs = pred.sort(['session_id', 'req_idx_raw']).with_columns(
                pl.col('elapsed_min_raw').diff().over('session_id').alias('dt'),
                pl.col('gamma').diff().over('session_id').alias('dg'),
                pl.col('log_survival').diff().over('session_id').alias('ds'),
                pl.col('prediction').diff().over('session_id').alias('dp'))
            eligible = pairs.filter(pl.col('dt') >= 0)
            metric['monotonicity'] = {'nonrewinding_pairs': eligible.height,
                'clock_rewinds': pairs.filter(pl.col('dt') < 0).height,
                'gamma_increases_nonrewinding': eligible.filter(pl.col('dg') > 1e-7).height,
                'logS_increases_nonrewinding': eligible.filter(pl.col('ds') > 1e-6).height,
                'prediction_increases_nonrewinding': eligible.filter(pl.col('dp') > 1e-7).height}
            assert metric['monotonicity']['gamma_increases_nonrewinding'] == 0
            assert metric['monotonicity']['logS_increases_nonrewinding'] == 0
        (out/'metrics.json').write_text(json.dumps(metric, indent=2))
        summary[mode] = metric
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
        print('RESULT', mode, json.dumps({k: metric[k] for k in ['overall.auc', 'user_gauc', 'logloss', 'wall_seconds']}), flush=True)
    print('COMPLETE', args.output/'summary.json', flush=True)


if __name__ == '__main__':
    main()
