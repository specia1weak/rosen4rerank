# 参考入口：从仓库根使用 uv run python -m experiments.kuairand.pipeline.run_current_feedback_nll_only 启动。
# 保留历史实验的严格对齐检查；需要对应数据/schema及历史run，不是下载即跑的独立教程。
"""当刷反馈预算仅NLL：单/混合×NLL系数0.1/1×三seed，逐轮完整评估到5轮。"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import polars as pl
import torch
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from experiments.kuairand.pipeline.current_feedback_nll_only import CurrentFeedbackNLLOnly
from experiments.kuairand.pipeline.current_feedback import load_feedback_batches
from experiments.kuairand.pipeline.exit_training import WORKSPACE, EXPERIMENTS, state_hash, joint_parameter_groups
from experiments.kuairand.pipeline.kr_request_schema import request_manager
from experiments.kuairand.pipeline.uid import zero_uid_inputs, uid_tables, tensor_hash
from experiments.kuairand.pipeline.run_epochwise_exit import EpochwiseExitTrainer, write_json
from experiments.kuairand.pipeline.build_timestamp_data import sha


class NLLOnlyTrainer(EpochwiseExitTrainer):
    def train_epoch(self):
        self.loss_rows = []
        result = super().train_epoch()
        pl.DataFrame(self.loss_rows).write_parquet(self.output/f'loss_steps_epoch_{self.epoch:02d}.parquet')
        return result

    def train_step(self, batch, ctx):
        result = super().train_step(batch, ctx)
        terms = dict(self.model.last_loss_terms)
        assert all(np.isfinite(v) for v in terms.values())
        assert 0 <= terms['right_main_ratio'] <= .1+1e-12
        assert terms['effective_left_weight'] == self.model.nll_weight
        self.loss_rows.append(dict(step=self.global_step+1, rows=len(batch['y_continue']), **terms))
        return result


def gradient_audit(model, batch):
    """用真实批次检验计算图；不赋.grad、不更新权重、不消耗随机数。"""
    budget = list(model.budget_embedding.parameters())+list(model.budget_net.parameters())
    content = list(model.omni_embedding.parameters())+list(model.content_net.parameters())+list(model.feedback_projection.parameters())
    terms = model.loss_terms(batch)
    joint_grads = torch.autograd.grad(terms['joint_bce'], budget, allow_unused=True, retain_graph=True)
    assert all(g is None for g in joint_grads)
    nll_content = torch.autograd.grad(terms['left_session_nll'], content, allow_unused=True, retain_graph=True)
    assert all(g is None for g in nll_content)
    nll_budget = torch.autograd.grad(terms['left_session_nll'], budget, allow_unused=True, retain_graph=True)
    norm = sum(float(g.norm()) for g in nll_budget if g is not None)
    assert norm > 0
    joint_content = torch.autograd.grad(terms['joint_bce'], content, allow_unused=True)
    assert sum(float(g.norm()) for g in joint_content if g is not None) > 0
    assert all(p.grad is None for p in model.parameters())
    return dict(joint_budget_disconnected=True, nll_content_disconnected=True,
        nll_budget_gradient_norm_sum=norm, joint_content_connected=True)


def run(output):
    torch.set_num_threads(4)
    reference = WORKSPACE/'runs/plan22_current_feedback'
    old = json.loads((reference/'config.json').read_text())
    sources = {**old['source_hashes'], **{str(p.resolve()): sha(p) for p in [
        Path(__file__), Path(__file__).with_name('current_feedback_nll_only.py')]}}
    assert all(sha(p) == h for p, h in sources.items())
    data = Path(old['data'])
    manager = request_manager(data/'schema', load=True)
    batches = zero_uid_inputs(load_feedback_batches(data, manager))
    coverage = json.loads((reference/'coverage.json').read_text())
    for split, loader in batches.items():
        actual = torch.cat([b['request_id'] for b in loader]).numpy()
        expected = pl.read_parquet(data/f'{split}.parquet', columns=['request_id'])['request_id'].to_numpy()
        assert np.array_equal(actual, expected) and len(actual) == coverage[split]['rows']
        assert all(torch.count_nonzero(b['user_id']) == 0 for b in loader)
    write_json(output/'coverage.json', coverage)
    config = {**old, 'source_hashes': sources, 'reference': str(reference),
        'matrix': [[name, weight] for name in ['single_step_pace', 'mixture_step_pace'] for weight in [.1, 1.]],
        'new_trajectories': 12, 'reused_trajectories': 0,
        'training_strategy': 'joint BCE gamma.detach()*stay; left only actual-weight NLL; right weak BCE',
        'left_nll_weights': [.1, 1.], 'left_nll_cap': None,
        'right_auxiliary': '0.1*BCE(stay,y), capped at 0.1*detached joint BCE',
        'auxiliary': 'left NLL uncapped by explicit request; only right BCE capped',
        'individual_aux_coefficients': None, 'max_aux_ratio': None, 'detach_gamma': True,
        'clip_semantics': 'unchanged global norm 10 across both towers; coefficient is before clipping'}
    write_json(output/'config.json', config)
    summary = {}
    for seed in [2026, 2027, 2028]:
        for name, weight in config['matrix']:
            key = f'{seed}/{name}/nll_{weight:g}'
            dest = output/key
            if (dest/'complete.json').exists():
                summary[key] = json.loads((dest/'epochs.json').read_text())
                continue
            dest.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(seed); np.random.seed(seed)
            model = CurrentFeedbackNLLOnly(manager, **EXPERIMENTS[name], nll_weight=weight,
                requests_per_session=coverage['train']['requests_per_session'])
            initial = state_hash(model)
            ref = json.loads((reference/f'{seed}/{name}/on/initialization.json').read_text())
            assert initial == ref['state_sha256']
            groups = joint_parameter_groups(model)
            meta = [dict(scope=g['scope'], lr=g['lr'], weight_decay=g['weight_decay'], tensors=len(g['params'])) for g in groups]
            assert meta == ref['optimizer_groups']
            rng = torch.get_rng_state().clone()
            audit = gradient_audit(model, batches['train'][0].interaction)
            assert initial == state_hash(model) and torch.equal(rng, torch.get_rng_state())
            uid_initial = {n: tensor_hash(t.weight) for n, t in uid_tables(model).items()}
            write_json(dest/'initialization.json', dict(state_sha256=initial, reference_state_exact=True,
                optimizer_groups=meta, **audit))
            cfg = SimpleNamespace(device='cpu', max_epochs=5, ckpt_dir=str(dest),
                experiment_name=f'nll_only_{seed}_{name}_{weight}', dataset_name='kr_v6_current_feedback', train_log_interval=200)
            trainer = NLLOnlyTrainer(model, torch.optim.AdamW(groups, weight_decay=0), manager,
                TrainerDataLoaders(batches['train'], batches['valid'], batches['test']), TrainerComponents(), cfg, output=dest)
            print('START', key, audit, flush=True)
            trainer.run()
            assert trainer.global_step == 2105 and len(trainer.records) == 5
            assert uid_initial == {n: tensor_hash(t.weight) for n, t in uid_tables(model).items()}
            write_json(dest/'complete.json', dict(epochs=5, steps=2105, uid_unchanged=True, restored_best=False))
            summary[key] = trainer.records
            write_json(output/'summary.json', summary)
            print('TRAJECTORY_COMPLETE', key, flush=True)
    assert all(sha(p) == h for p, h in sources.items())
    write_json(output/'complete.json', dict(trajectories=len(summary), test_evaluations=60,
        valid_evaluations=60, source_hashes_unchanged=True, initial_state_optimizer_exact=True))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=WORKSPACE/'runs/plan22_current_feedback_nll_only')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output/'logs').mkdir(exist_ok=args.resume)
    print('训练日志：', args.output/'logs/train.log', flush=True)
    with (args.output/'logs/train.log').open('a' if args.resume else 'w') as f, redirect_stdout(f), redirect_stderr(f):
        run(args.output)
