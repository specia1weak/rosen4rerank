"""真实KR数据上的Omni视图与联合训练调试，非正式效果实验。

uv run python workspace/kr_time_budget/debug_time_budget.py
uv run python workspace/kr_time_budget/debug_time_budget.py --stop-at forward
uv run python workspace/kr_time_budget/debug_time_budget.py --clock time

默认读取2个train batch和1个valid/test batch。v3的elapsed_min是原始分钟，
本入口显式别名为elapsed_min_raw后交给模型；不重建数据、不缩放、不写回v3。
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
import polars as pl
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosen.core.train import EarlyStopper
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import TwoStageRunner
from rosen.data.dataset import ParquetStreamDataset
from rosen.evaluate.evaluator import Evaluator
from rosen.evaluate.manager import EvaluatorManager
from rosen.utils.optimize import split_model_params_by_decay
from rosen.utils.expr_tensor import ExprTensor
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager, EXTRA_COLUMNS
from workspace.kr_time_budget.pipeline.run_exit_v4 import ExitTrainerV4
from workspace.kr_time_budget.pipeline.time_budget_v7 import TimeBudgetModel

WORKSPACE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clock', choices=['time', 'dual'], default='dual')
    parser.add_argument('--training-strategy', choices=['joint', 'two_stage'], default='joint')
    parser.add_argument('--output', type=Path, help='使用独立输出目录；默认runs/exit_v7_smoke/<strategy>_<clock>')
    parser.add_argument('--stop-at', choices=['views', 'forward', 'trained', 'stage-two'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    output = args.output or WORKSPACE/'runs/exit_v7_smoke'/f'{args.training_strategy}_{args.clock}'
    output.mkdir(parents=True, exist_ok=True)
    # 避免重复运行混合checkpoint和progress。
    if list(output.glob('*/progress.jsonl')):
        raise RuntimeError(f'调试输出已存在，请用--output选择新目录：{output}')
    data = WORKSPACE/'data/v3'
    manager = request_manager(data/'schema', load=True)
    loaders = {}
    for split, rows in [('train', 8192), ('valid', 4096), ('test', 4096)]:
        # v3中这个列仍是未缩放分钟；未来换数据版本须从真实raw列读取，不能盲目别名。
        frame = pl.scan_parquet(data/f'{split}.parquet').head(rows).with_columns(
            pl.col('elapsed_min').alias('elapsed_min_raw'))
        dataset = ParquetStreamDataset(frame, manager, batch_size=4096, shuffle=False,
                                       drop_last=False, extra_col_names=EXTRA_COLUMNS+['elapsed_min_raw'])
        loaders[split] = list(DataLoader(dataset, batch_size=None))
    batch = loaders['train'][0].interaction
    model = TimeBudgetModel(manager, clock=args.clock, training_strategy=args.training_strategy)
    views = {'budget': model.budget_view, 'content': model.content_view, 'fusion': model.fusion_view}
    view_fields = {name: list(view.field_slices) for name, view in views.items()}
    view_shapes = {name: tuple(view(batch).shape) for name, view in views.items()}
    view_slices = {name: view.field_slices for name, view in views.items()}
    field_embeddings = model.content_view(batch, split_by='name')
    source_embeddings = model.content_view(batch, split_by='source')
    print('VIEWS', {name: {'fields': len(view_fields[name]), 'shape': view_shapes[name]} for name in views}, flush=True)
    if args.stop_at == 'views':
        breakpoint()

    # 可单步进入clock_components，检查两个原始量与各自的gamma/S/logS。
    clock_parts = model.clock_components(batch)
    gamma, survival, log_survival = model.budget_components(batch)
    initial_prediction = model.predict(batch)
    if args.training_strategy == 'joint':
        stay = model.components(batch)[2]
        formula = ExprTensor(gamma, 'gamma') * ExprTensor(stay, 'stay')
    else:
        formula = ExprTensor(gamma, 'gamma') * torch.sigmoid(ExprTensor(torch.full_like(gamma, 3.5), 'prior_logit'))
    torch.testing.assert_close(formula.data, initial_prediction, rtol=0, atol=0)
    initial_loss = model.calculate_loss(batch)
    initial_loss.backward()
    initial_budget_grad = model.budget_net[-1].weight.grad.norm(dim=1).tolist()
    assert all(value > 0 for value in initial_budget_grad[:2])
    if args.clock == 'dual':
        assert all(value > 0 for value in initial_budget_grad[2:])
    initial_content_grad = model.content_net[-1].weight.grad
    initial_content_grad = None if initial_content_grad is None else initial_content_grad.norm().item()
    initial_embedding_grads = None
    if args.training_strategy == 'joint':
        assert initial_content_grad > 0
        initial_embedding_grads = {
            'budget_user': model.budget_embedding.emb_modules['user_id'].embedding.weight.grad.norm().item(),
            'content_user': model.omni_embedding.emb_modules['user_id'].embedding.weight.grad.norm().item(),
        }
        assert all(value > 0 for value in initial_embedding_grads.values())
    model.zero_grad(set_to_none=True)
    print('FORWARD', args.clock, args.training_strategy, 'loss=', initial_loss.item(),
          'budget_head_gradient=', initial_budget_grad, 'content_head_gradient=', initial_content_grad,
          'embedding_gradients=', initial_embedding_grads, flush=True)
    print('EXPRESSION', formula.expr, flush=True)
    if args.stop_at == 'forward':
        breakpoint()

    def factory(current, stage=None):
        phase_name = stage.value if stage is not None else 'joint'
        phase = output/phase_name
        evaluator = EvaluatorManager(log_path=phase/'evaluations.log', title=f'Omni model smoke {args.clock} {phase_name}')
        evaluator.register('overall', Evaluator('auc', 'gauc', 'logloss'))
        cfg = SimpleNamespace(device='cpu', max_epochs=1, ckpt_dir=str(phase),
                              experiment_name=f'omni_smoke_{args.clock}_{phase_name}',
                              dataset_name='kuairand_requests_v3', train_log_interval=1)
        trainer = ExitTrainerV4(
            current,
            torch.optim.AdamW(split_model_params_by_decay(current, weight_decay=1e-6), lr=.001, weight_decay=0),
            manager, TrainerDataLoaders(loaders['train'], loaders['valid']),
            TrainerComponents(evaluator_manager=evaluator, early_stopper=EarlyStopper(preferred_metrics=('logloss',))), cfg,
        )
        if args.training_strategy == 'two_stage' and current.stage == 2:
            trainer.transition_reference = model
        return trainer

    transition = None
    if args.training_strategy == 'two_stage':
        result = TwoStageRunner(factory).run(model)
        product = result.stage_two_model
        for prefix, module in [('budget_embedding.', product.budget_embedding), ('budget_net.', product.budget_net)]:
            for name, tensor in module.state_dict().items():
                assert torch.equal(tensor, result.stage_snapshot.model_state[prefix+name]), prefix+name
        assert all(p.grad is None for module in product.budget_modules for p in module.parameters())
        steps = result.stage_one_trainer.global_step + result.stage_two_trainer.global_step
        transition = result.stage_two_trainer.transition_audit
    else:
        trainer = factory(model)
        trainer.run()
        product = model
        steps = trainer.global_step
        assert all(p.requires_grad for module in product.budget_modules for p in module.parameters())
    test_batch = loaders['test'][0].interaction
    product.eval()
    with torch.no_grad():
        prediction, gamma, stay, survival, log_survival = product.components(test_batch)
        torch.testing.assert_close(prediction, product.predict(test_batch), rtol=0, atol=0)
        test_loss = product.calculate_loss(test_batch).item()
    prediction_table = pl.DataFrame({
        'request_id': test_batch['request_id'].numpy(),
        'elapsed_min_raw': test_batch['elapsed_min_raw'].numpy(),
        'req_idx_raw': test_batch['req_idx_raw'].numpy(),
        'gamma': gamma.numpy(), 'stay': stay.numpy(), 'prediction': prediction.numpy(),
    })
    print(prediction_table.head(8))
    if args.stop_at in {'trained', 'stage-two'}:
        breakpoint()
    summary = {
        'status': 'smoke passed; not a full-train/full-test effectiveness result',
        'data': str(data), 'clock': args.clock, 'training_strategy': args.training_strategy, 'seed': 2026,
        'train_rows_per_epoch': 8192, 'valid_rows': 4096, 'test_rows': 4096,
        'optimizer_steps': steps,
        'views': {name: {'fields': view_fields[name], 'shape': view_shapes[name]} for name in views},
        'budget_exactly_frozen': args.training_strategy == 'two_stage', 'transition': transition,
        'initial_budget_head_gradients': initial_budget_grad, 'test_probe_loss': test_loss,
        'initial_content_head_gradient': initial_content_grad, 'initial_shared_embedding_gradients': initial_embedding_grads,
        'optimizer': 'AdamW', 'lr': .001, 'weight_decay': 1e-6,
        'zero_decay': 'Embedding,bias,one-dimensional parameters', 'physical_batch': 4096,
        'formula_expression': formula.expr,
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('SMOKE_COMPLETE', output/'summary.json', flush=True)


if __name__ == '__main__':
    main()
