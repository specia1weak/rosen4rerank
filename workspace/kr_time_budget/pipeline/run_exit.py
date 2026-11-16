"""Full user-grouped 5-fold exit screening with auditable OOF predictions."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
from rosen.data.dataset import ParquetStreamDataset
from rosen.utils.optimize import split_model_params_by_decay
from workspace.kr_time_budget.pipeline.time_budget import ExitModel
from workspace.kr_time_budget.pipeline.features import prepare_fold, EXTRAS


def grouped_auc(y, p, groups):
    # Mann-Whitney rank formula, with average ranks for ties. No O(groups)
    # Python/sklearn loop, which is expensive for >100k sessions.
    ranked = pl.DataFrame({'g': groups, 'y': y, 'p': p}).with_columns(pl.col('p').rank('average').over('g').alias('rank'))
    stats = ranked.group_by('g').agg(pl.len().alias('n'), pl.col('y').sum().alias('pos'), (pl.col('rank') * pl.col('y')).sum().alias('rank_sum'))
    valid = stats.filter((pl.col('pos') > 0) & (pl.col('pos') < pl.col('n'))).with_columns(((pl.col('rank_sum') - pl.col('pos') * (pl.col('pos') + 1) / 2) / (pl.col('pos') * (pl.col('n') - pl.col('pos')))).alias('auc'))
    weight = valid['n'].sum()
    return {'auc': (valid['auc'] * valid['n']).sum() / weight if weight else None, 'eligible_groups': valid.height, 'eligible_rows': weight, 'total_groups': stats.height}


def metrics(df):
    y, p = df['y_continue'].to_numpy(), df['prediction'].to_numpy()
    return {'overall.auc': roc_auc_score(y, p), 'logloss': log_loss(y, p, labels=[0, 1]), 'user_gauc': grouped_auc(y, p, df['user_id'].to_numpy()), 'session_gauc': grouped_auc(y, p, df['session_id'].to_numpy()), 'rows': len(y)}


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def batches(path, manager, size):
    ds = ParquetStreamDataset(path, manager, batch_size=size, shuffle=False, drop_last=False, extra_col_names=EXTRAS)
    return DataLoader(ds, batch_size=None, num_workers=0)


def on_device(batch, device):
    return {k: v.to(device) for k, v in batch.interaction.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('workspace/kr_time_budget/data/v1'))
    ap.add_argument('--output', type=Path, default=Path('workspace/kr_time_budget/runs/exit_v1'))
    ap.add_argument('--folds', type=int, nargs='+', default=list(range(5)))
    ap.add_argument('--models', nargs='+', default=['content', 'budget', 'fusion', 'ours'])
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=4096)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config.update(weight_decay=1e-6, decay_exclusions='embedding, bias, normalization, scalar/vector parameters', optimizer='AdamW', lr=.001, ours_budget_stage2_lr=.0001, scheduler=None, checkpoint='last fixed epoch; no test-driven selection', comparison='non-causal architecture screening; identical initial tensors/data/batches, different masks and training objectives', ema=False)
    config_path = args.output / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError('Output already contains a different config; use a new output directory')
    config_path.write_text(json.dumps(config, indent=2))
    for fold in args.folds:
        mgr, data = prepare_fold(args.root, fold, args.seed)
        start = time.monotonic()
        train_batches = []
        for batch in batches(data / 'train.parquet', mgr, args.batch_size):
            train_batches.append(batch)
            if args.smoke:
                break
        print('Cached rosen train batches', fold, len(train_batches), 'seconds', time.monotonic() - start, flush=True)
        hashes = {}
        for mode in args.models:
            out = args.output / f'fold_{fold}' / mode
            out.mkdir(parents=True, exist_ok=True)
            if (out / 'metrics.json').exists():
                print('RESUME skip completed', fold, mode, flush=True)
                continue
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            model = ExitModel(mgr, mode=mode)
            init_hash = state_hash(model)
            hashes[mode] = init_hash
            assert len(set(hashes.values())) == 1, hashes
            model.to(args.device)
            optimizer = torch.optim.AdamW(split_model_params_by_decay(model, weight_decay=1e-6), lr=.001, weight_decay=0)
            manifest = {'initial_state_sha256': init_hash, 'fold': fold, 'mode': mode, 'train_rows': json.loads((data / 'manifest.json').read_text())['train_rows'], 'physical_batch': args.batch_size, 'effective_batch': args.batch_size, 'accumulation': 1, 'epochs': args.epochs, 'seed': args.seed}
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            steps = 0
            for epoch in range(args.epochs):
                model.train()
                model.stage = 1 if mode == 'ours' and epoch < 3 else 2
                if mode == 'ours' and epoch == 3:
                    # Preserve optimizer moments while assigning the predeclared lower tower LR.
                    budget_ids = {id(p) for p in model.budget_net.parameters()}
                    for group in list(optimizer.param_groups):
                        budget = [p for p in group['params'] if id(p) in budget_ids]
                        group['params'] = [p for p in group['params'] if id(p) not in budget_ids]
                        if budget:
                            optimizer.add_param_group({'params': budget, 'weight_decay': group['weight_decay'], 'lr': .0001})
                start = time.monotonic()
                rows, loss_sum = 0, 0.
                for batch in train_batches:
                    b = on_device(batch, args.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model.calculate_loss(b)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'nonfinite loss fold={fold} mode={mode} step={steps}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    n = len(b['y_continue'])
                    rows += n
                    loss_sum += loss.item() * n
                    steps += 1
                    if args.smoke:
                        break
                record = {'fold': fold, 'model': mode, 'epoch': epoch + 1, 'stage': model.stage, 'train_rows': rows, 'optimizer_steps': steps, 'loss': loss_sum / rows, 'seconds': time.monotonic() - start}
                print(json.dumps(record), flush=True)
                with (out / 'epochs.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch + 1, 'stage': model.stage, 'steps': steps, 'config': config}, out / 'last.pt')
                model.to(args.device)
            model.eval()
            model.stage = 2
            pieces = []
            with torch.no_grad():
                for batch in batches(data / 'test.parquet', mgr, args.batch_size):
                    b = on_device(batch, args.device)
                    pred = model.predict(b)
                    vals = {k: b[k].cpu().numpy() for k in ['request_id', 'user_id', 'session_id', 'y_continue']}
                    vals['prediction'] = pred.cpu().numpy()
                    if mode == 'ours':
                        _, gamma, stay, survival, log_survival = model.components(b)
                        vals.update(gamma=gamma.cpu().numpy(), content_stay=stay.cpu().numpy(), survival=survival.cpu().numpy(), log_survival=log_survival.cpu().numpy())
                    pieces.append(pl.DataFrame(vals))
                    if args.smoke:
                        break
            predictions = pl.concat(pieces).with_columns(pl.lit(fold).alias('fold'))
            predictions.write_parquet(out / 'predictions.parquet')
            result = metrics(predictions)
            result.update(full_train=not args.smoke, full_test=not args.smoke, optimizer_steps=steps)
            (out / 'metrics.json').write_text(json.dumps(result, indent=2))
            print('RESULT', fold, mode, json.dumps(result), flush=True)
            del model, optimizer, pieces
        del train_batches
    summaries = {}
    for mode in args.models:
        paths = sorted(args.output.glob(f'fold_*/{mode}/predictions.parquet'))
        if paths:
            df = pl.read_parquet(paths)
            result = metrics(df)
            fold_results = [json.loads(p.with_name('metrics.json').read_text()) for p in paths]
            result['fold_results'] = fold_results
            result['completed_folds'] = len(paths)
            summaries[mode] = result
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print('SUMMARY', json.dumps(summaries), flush=True)

if __name__ == '__main__':
    main()
