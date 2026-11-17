# 参考入口：从仓库根使用 uv run python -m experiments.kuairand.pipeline.run_current_feedback_strong 启动。
# 保留历史实验的严格对齐检查；需要对应数据/schema及历史run，不是下载即跑的独立教程。
"""当刷信息可见，强双辅助，单/混合三seed各训练1完整epoch。Fusion复用。"""
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from experiments.kuairand.pipeline.current_feedback_strong import StrongCurrentFeedback
from experiments.kuairand.pipeline.current_feedback import load_feedback_batches
from experiments.kuairand.pipeline.exit_training import WORKSPACE, EXPERIMENTS, state_hash, joint_parameter_groups
from experiments.kuairand.pipeline.kr_request_schema import request_manager
from experiments.kuairand.pipeline.uid import zero_uid_inputs, uid_tables, tensor_hash
from experiments.kuairand.pipeline.run_epochwise_exit import EpochwiseExitTrainer, write_json
from experiments.kuairand.pipeline.build_timestamp_data import sha


def run(output):
    torch.set_num_threads(4)
    ref = WORKSPACE/'runs/plan22_current_feedback'
    config = json.loads((ref/'config.json').read_text())
    sources = {**config['source_hashes'], **{str(p.resolve()): sha(p) for p in [
        Path(__file__), Path(__file__).with_name('current_feedback_strong.py')]}}
    assert all(sha(p) == h for p, h in sources.items())
    manager = request_manager(Path(config['data'])/'schema', load=True)
    batches = zero_uid_inputs(load_feedback_batches(config['data'], manager))
    coverage = json.loads((ref/'coverage.json').read_text())
    write_json(output/'coverage.json', coverage)
    config.update(source_hashes=sources, epochs=1, new_trajectories=6, reused_trajectories=3,
        reference=str(ref), auxiliary='left NLL + right BCE, each coefficient 1, no cap',
        individual_aux_coefficients=1, max_aux_ratio=None,
        training_strategy='joint BCE to both towers; uncapped strong auxiliaries',
        selection='fixed end of epoch1; no best restore',
        matrix=['single_step_pace','mixture_step_pace'])
    write_json(output/'config.json', config)
    summary = {}
    for seed in [2026, 2027, 2028]:
        for name in config['matrix']:
            dest = output/f'{seed}/{name}'; dest.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(seed); np.random.seed(seed)
            model = StrongCurrentFeedback(manager, **EXPERIMENTS[name],
                requests_per_session=coverage['train']['requests_per_session'])
            initial = state_hash(model)
            expected = json.loads((ref/f'{seed}/{name}/on/initialization.json').read_text())
            assert initial == expected['state_sha256']
            groups = joint_parameter_groups(model)
            gmeta = [dict(scope=g['scope'], lr=g['lr'], weight_decay=g['weight_decay'], tensors=len(g['params'])) for g in groups]
            assert gmeta == expected['optimizer_groups']
            uid_initial = {n: tensor_hash(t.weight) for n,t in uid_tables(model).items()}
            batch = batches['train'][0].interaction
            loss = model.calculate_loss(batch)
            budget = list(model.budget_net.parameters())
            grad = torch.autograd.grad(model.loss_terms(batch)['joint_bce'], budget, allow_unused=True)
            assert sum(float(g.norm()) for g in grad if g is not None) > 0
            assert initial == state_hash(model)
            write_json(dest/'initialization.json', dict(state_sha256=initial, optimizer_groups=gmeta,
                baseline_initial_exact=True, joint_budget_connected=True, initial_terms=model.last_loss_terms))
            cfg = SimpleNamespace(device='cpu', max_epochs=1, ckpt_dir=str(dest),
                experiment_name=f'strong_current_{seed}_{name}', dataset_name='kr_v6_current_feedback', train_log_interval=200)
            trainer = EpochwiseExitTrainer(model, torch.optim.AdamW(groups, weight_decay=0), manager,
                TrainerDataLoaders(batches['train'], batches['valid'], batches['test']), TrainerComponents(), cfg, output=dest)
            print('START', seed, name, flush=True)
            trainer.run()
            assert trainer.global_step == 421 and len(trainer.records) == 1
            assert uid_initial == {n: tensor_hash(t.weight) for n,t in uid_tables(model).items()}
            summary[f'{seed}/{name}'] = trainer.records
            write_json(output/'summary.json', summary)
            write_json(dest/'complete.json', dict(epochs=1, steps=421, uid_unchanged=True))
            print('TRAJECTORY_COMPLETE', seed, name, flush=True)
    assert all(sha(p) == h for p,h in sources.items())
    write_json(output/'complete.json', dict(trajectories=6, test_evaluations=6, valid_evaluations=6, source_hashes_unchanged=True))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    output = WORKSPACE/'runs/plan22_current_feedback_strong_epoch1'
    output.mkdir(parents=True, exist_ok=False); (output/'logs').mkdir()
    print('日志：', output/'logs/train.log', flush=True)
    with (output/'logs/train.log').open('w') as f, redirect_stdout(f), redirect_stderr(f): run(output)
