"""UID消融：只在内存中将编码后user_id置0，复用原5轮逐epoch训练协议。

raw UID保留给User-AUC；两塔和Fusion输入均被屏蔽，不重建parquet/schema。
保持模型形状、初始化和优化器不变；对照为runs/epochwise5。
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch
from rosen.core.interaction import Interaction
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from workspace.kr_time_budget.pipeline.exit_model import KRExitModel
from workspace.kr_time_budget.pipeline.exit_training import (
    EXPERIMENTS, WORKSPACE, joint_parameter_groups, load_batches, state_hash,
)
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.run_epochwise_exit import EpochwiseExitTrainer, write_json


def zero_uid_inputs(batches):
    """仅替换user_id张量，不改原batch；所有其余字段（包括raw UID）原样保留。"""
    return {split: [Interaction(dict(batch.interaction, user_id=torch.zeros_like(batch['user_id'])))
                    for batch in loader] for split, loader in batches.items()}


def uid_tables(model):
    return {'content_or_fusion': model.omni_embedding.emb_modules['user_id'].embedding,
            'budget': model.budget_embedding.emb_modules['user_id'].embedding}


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def run(args, output):
    torch.set_num_threads(4)
    baseline = WORKSPACE/'runs/epochwise5'
    original_config = json.loads((baseline/'config.json').read_text())
    assert json.loads((baseline/'complete.json').read_text())['trajectories'] == 18
    # 不改旧入口/模型源码；核对旧对照来源仍有效。
    for source, digest in original_config['source_hashes'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest, source
    config = dict(original_config)
    sources = {**original_config['source_hashes'], str(Path(__file__).resolve()):
               hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config.update(seeds=args.seeds, experiments={n: EXPERIMENTS[n] for n in args.experiments},
        source_hashes=sources, input_ablation='encoded user_id -> constant padding index 0 for all splits',
        raw_uid_for_metrics='user_id_raw unchanged', baseline=str(baseline),
        baseline_config_sha256=hashlib.sha256((baseline/'config.json').read_bytes()).hexdigest(),
        dataset_rebuilt=False, schema_refit=False, uid_embedding='padding_idx=0, row0 zero; dimensions retained')
    write_json(output/'config.json', config)
    data = WORKSPACE/'data/v3'
    assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest() == config['dataset_manifest_sha256']
    manager = request_manager(data/'schema', load=True)
    assert manager.get_setting('user_id').padding_zero
    original = load_batches(data, manager)
    batches = zero_uid_inputs(original)
    coverage, input_audit = {}, {}
    expected_coverage = json.loads((baseline/'coverage.json').read_text())
    for split, loader in batches.items():
        for before, after in zip(original[split], loader):
            assert set(before.interaction) == set(after.interaction)
            assert torch.count_nonzero(after['user_id']) == 0
            assert all(torch.equal(before[field], after[field]) for field in before if field != 'user_id')
        ids = torch.cat([batch['request_id'] for batch in loader]).numpy()
        expected = pl.read_parquet(data/f'{split}.parquet', columns=['request_id'])['request_id'].to_numpy()
        assert np.array_equal(ids, expected)
        coverage[split] = dict(rows=len(ids), batches=len(loader),
            request_order_sha256=hashlib.sha256(ids.astype(expected.dtype).tobytes()).hexdigest())
        assert coverage[split] == expected_coverage[split]
        input_audit[split] = dict(rows=len(ids), user_id_nonzero_after=0,
            user_id_nonzero_before=sum(int(torch.count_nonzero(b['user_id'])) for b in original[split]),
            raw_uid_unique=int(torch.cat([b['user_id_raw'] for b in loader]).unique().numel()),
            every_other_tensor_exact=True)
    write_json(output/'coverage.json', coverage)
    write_json(output/'input_audit.json', input_audit)
    del original
    print('DATA', json.dumps(coverage), 'UID_MASK', json.dumps(input_audit), flush=True)
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
            reference = json.loads((baseline/key/'initialization.json').read_text())
            assert initial == reference['state_sha256'], key
            tables = uid_tables(model)
            assert all(t.padding_idx == 0 and torch.count_nonzero(t.weight[0]) == 0 for t in tables.values())
            uid_initial = {n: tensor_hash(t.weight) for n, t in tables.items()}
            groups = joint_parameter_groups(model)
            group_records = [dict(scope=g['scope'], lr=g['lr'], weight_decay=g['weight_decay'],
                                  tensors=len(g['params'])) for g in groups]
            assert group_records == reference['optimizer_groups']
            write_json(destination/'initialization.json', dict(state_sha256=initial,
                baseline_initialization_exact=True, optimizer_groups=group_records,
                uid_table_sha256=uid_initial))
            cfg = SimpleNamespace(device='cpu', max_epochs=5, ckpt_dir=str(destination),
                experiment_name=f'uid_zero_{name}_{seed}', dataset_name='kuairand_requests_v3',
                train_log_interval=200)
            trainer = EpochwiseExitTrainer(model, torch.optim.AdamW(groups, weight_decay=0), manager,
                TrainerDataLoaders(batches['train'], batches['valid'], batches['test']),
                TrainerComponents(), cfg, output=destination)
            print('START', key, flush=True)
            trainer.run()
            assert trainer.global_step == 5*coverage['train']['batches']
            assert [r['epoch'] for r in trainer.records] == [1, 2, 3, 4, 5]
            uid_final = {n: tensor_hash(t.weight) for n, t in tables.items()}
            assert uid_final == uid_initial, key
            write_json(destination/'uid_audit.json', dict(padding_idx=0, row0_zero=True,
                uid_table_sha256=uid_final, all_uid_embeddings_unchanged_after_training=True,
                baseline_initialization_exact=True))
            summary[key] = trainer.records
            write_json(destination/'complete.json', dict(epochs=5, optimizer_steps=trainer.global_step,
                best_valid_epoch=trainer.best_valid_epoch, restored_best=False,
                wall_seconds=time.monotonic()-trainer.started))
            write_json(output/'summary.json', summary)
            print('TRAJECTORY_COMPLETE', key, flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in sources.items())
    write_json(output/'complete.json', dict(trajectories=len(summary), epoch_test_evaluations=5*len(summary)))
    print('COMPLETE', output/'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--experiments', nargs='+', choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument('--output', type=Path, default=WORKSPACE/'runs/epochwise5_uid_zero')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.experiments)) != len(args.experiments):
        parser.error('seeds and experiments must be unique')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'logs').mkdir()
    print(f'运行目录：{args.output}\n日志：{args.output / "logs/train.log"}', flush=True)
    with (args.output/'logs/train.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        run(args, args.output)
    print(f'完成：{args.output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
