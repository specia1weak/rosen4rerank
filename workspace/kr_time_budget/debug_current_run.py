"""当前模型的一次可断点诊断；默认不重跑正式训练或调参。

uv run python workspace/kr_time_budget/debug_current_run.py
uv run python workspace/kr_time_budget/debug_current_run.py --stop-at forward

默认：mixture_step_pace / seed2026 / 3epoch预算。
先用真实8192条train、4096条valid做3epoch短程等价重放(6次更新)，
再重载已完成run的checkpoint，复算完整valid/test。短程结果不作为效果实验。
在views/forward/trained/checkpoint断点可检查同名局部变量。
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score, log_loss
from rosen.core.train import EarlyStopper
from rosen.core.train.context import TrainerComponents, TrainerDataLoaders
from rosen.evaluate.evaluator import Evaluator
from rosen.evaluate.manager import EvaluatorManager, DomainFilter
from workspace.kr_time_budget.pipeline.kr_request_schema import (
    request_manager, FEATURES, SHARED_FIELDS, BUDGET_PRIOR_FIELDS, BUDGET_PARAMETER_FIELDS,
    CONTENT_EXCLUSIVE_FIELDS, CONTENT_TOWER_FIELDS, SESSION_CLOCK_FIELDS,
)
from workspace.kr_time_budget.pipeline.exit_model import KRExitModel, pace_feature
from workspace.kr_time_budget.pipeline.exit_training import (
    WORKSPACE, EXPERIMENTS, JointExitTrainer, load_batches, joint_parameter_groups,
    predict_batches, state_hash,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def evaluator(path):
    result = EvaluatorManager(log_path=path, title='KR current-run diagnostic')
    result.register('overall', Evaluator('auc', 'logloss'))
    for domain in range(5):
        result.register(f'domain{domain}', Evaluator('auc', 'logloss'), DomainFilter('tab', domain))
    return result


def tensor_stats(value):
    x = value.detach().double()
    return dict(shape=list(x.shape), dtype=str(value.dtype), finite=bool(torch.isfinite(x).all()),
                rms=x.square().mean().sqrt().item(), min=x.min().item(), max=x.max().item())


def inspect_views(model, batch):
    views = {'budget': model.budget_view, 'content': model.content_view, 'fusion': model.fusion_view}
    fields = {name: view(batch, split_by='name') for name, view in views.items()}
    assert set(fields['budget']) == set(BUDGET_PARAMETER_FIELDS)
    assert set(fields['content']) == set(CONTENT_TOWER_FIELDS)
    assert set(fields['budget']) & set(fields['content']) == set(SHARED_FIELDS)
    independent = all(model.budget_embedding.emb_modules[name] is not model.omni_embedding.emb_modules[name]
                      for name in SHARED_FIELDS)
    assert independent
    routes = []
    for name in FEATURES:
        category = ('共享背景' if name in SHARED_FIELDS else '预算先验' if name in BUDGET_PRIOR_FIELDS
                    else '原始时钟' if name in SESSION_CLOCK_FIELDS else '内容反馈/列表')
        routes.append(dict(field=name, category=category, budget_parameter=name in fields['budget'],
            content=name in fields['content'], fusion=True,
            formula='req_idx_raw - 1' if name == 'req_idx' and model.mode == 'ours' else
                    '分钟预算关闭；pace开启时以派生量进入内容/Fusion' if name == 'elapsed_min' else None))
    return dict(routes=routes, independent_embeddings=independent,
        base_field_counts={name: len(values) for name, values in fields.items()},
        embedding_dimensions={name: view.embedding_dim for name, view in views.items()},
        add_pace=model.add_pace, pace_formula='log1p(max(elapsed_min_raw,0)/max(req_idx_raw,1))',
        field_tensors={name: {field: tensor_stats(value) for field, value in values.items()} for name, values in fields.items()})


def inspect_forward(model, batch):
    # 独立NumPy参考直接计算S(t+1)/S(t)，不调用生产mixture_parts或其增量公式。
    model.eval()
    prediction = model.predict(batch)
    manual_loss = torch.nn.functional.binary_cross_entropy(prediction.clamp(1e-6, 1-1e-6), batch['y_continue'].float())
    assert torch.equal(manual_loss, model.calculate_loss(batch))
    assert torch.equal(prediction, model.predict(dict(batch, y_continue=1-batch['y_continue'])))
    content_before = model.network_view(batch, model.content_view)
    changed = dict(batch, **{name: batch[name]+100 for name in BUDGET_PRIOR_FIELDS})
    assert torch.equal(content_before, model.network_view(changed, model.content_view))
    changed = dict(batch, **{name: torch.zeros_like(batch[name]) for name in CONTENT_EXCLUSIVE_FIELDS})
    assert torch.equal(model.budget_raw(batch), model.budget_raw(changed))
    data = {name: batch[name].numpy() for name in ['request_id', 'session_id', 'req_idx_raw', 'elapsed_min_raw', 'y_continue']}
    data.update(pace=pace_feature(batch).numpy(), prediction=prediction.detach().numpy())
    result = dict(loss=manual_loss.item(), label_not_model_input=True, budget_prior_excluded_from_content=True,
                  content_excluded_from_budget_parameters=True)
    if model.mode == 'ours':
        prediction, gamma, stay, survival, log_survival = model.components(batch)
        assert torch.equal(prediction, gamma*stay)
        raw = model.budget_raw(batch).detach().numpy().astype(np.float64)
        raw = raw[:, 6:] if model.mixture else raw[:, 2:]
        t = np.maximum(batch['req_idx_raw'].numpy().astype(np.float64)-1, 0)
        if model.mixture:
            scale_raw, shape_raw = raw[:, [0, 2]], raw[:, [1, 3]]
            logits = raw[:, 4:6]
            log_pi = logits-np.logaddexp.reduce(logits, axis=1)[:, None]
        else:
            scale_raw, shape_raw, log_pi = raw[:, :1], raw[:, 1:2], np.zeros((len(raw), 1))
        scale = np.exp(np.clip(scale_raw, -2, 4.5)+1.6)
        shape = np.minimum(.3+np.logaddexp(0, shape_raw), 5.)
        log_s0 = np.logaddexp.reduce(log_pi-(t[:, None]/scale)**shape, axis=1)
        log_s1 = np.logaddexp.reduce(log_pi-((t[:, None]+1)/scale)**shape, axis=1)
        expected = np.exp(log_s1-log_s0)
        np.testing.assert_allclose(gamma.detach().numpy(), expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(log_survival.detach().numpy(), log_s0, atol=2e-6, rtol=2e-6)
        assert torch.equal(model.clock_components(batch)['time'][0], torch.ones_like(gamma))
        data.update(gamma=gamma.detach().numpy(), content_stay=stay.detach().numpy(),
                    survival=survival.detach().numpy(), log_survival=log_survival.detach().numpy())
        result.update(product_max_error=(prediction-gamma*stay).abs().max().item(),
            independent_numpy_gamma_max_error=float(np.max(np.abs(gamma.detach().numpy()-expected))),
            independent_numpy_logS_max_error=float(np.max(np.abs(log_survival.detach().numpy()-log_s0))))
    return result, pl.DataFrame(data)


def make_trainer(model, manager, batches, output, trainer_type=JointExitTrainer):
    cfg = SimpleNamespace(device='cpu', max_epochs=3, ckpt_dir=str(output),
        experiment_name='current_run_diagnostic', dataset_name='kuairand_requests_v3', train_log_interval=1)
    return trainer_type(model, torch.optim.AdamW(joint_parameter_groups(model), weight_decay=0), manager,
        TrainerDataLoaders(batches['train'], batches['valid']),
        TrainerComponents(evaluator_manager=evaluator(output/'evaluations.log'),
                          early_stopper=EarlyStopper(preferred_metrics=('logloss',))), cfg)


def replay_training(manager, batches, options, seed, output):
    # 历史入口只在此回归比较中导入，当前模型和Trainer不依赖它。
    from workspace.kr_time_budget.pipeline.time_budget_v10 import MixtureTimeBudgetModel
    from workspace.kr_time_budget.pipeline.run_plan13_ablation import BudgetTrainer
    traces, trainers = {}, {}
    for label, cls, trainer_type in [('current', KRExitModel, JointExitTrainer),
                                     ('historical', MixtureTimeBudgetModel, BudgetTrainer)]:
        torch.manual_seed(seed)
        model = cls(manager, **options)
        trainer = make_trainer(model, manager, batches, output/label, trainer_type)
        trace = []
        original_step = trainer.train_step
        def observed_step(batch, ctx, original=original_step, current=trainer, records=trace):
            loss = original(batch, ctx)
            records.append(dict(step=current.global_step+1, loss=loss.item(), state_sha256=state_hash(current.model)))
            return loss
        trainer.train_step = observed_step
        if label == 'historical' and model.mode == 'fusion':
            trainer.evaluate_epoch()
        trainer.run()
        assert trainer.global_step == 6
        traces[label], trainers[label] = trace, trainer
    assert traces['current'] == traces['historical']
    current, historical = trainers['current'], trainers['historical']
    for budget in [1, 3]:
        a = json.loads((current.output/f'budget_{budget}ep.json').read_text())
        b = json.loads((historical.output/f'budget_{budget}ep.json').read_text())
        assert a == b
        assert a['optimizer_state_steps'] == [budget*2]
        assert a['selected']['step'] <= budget*2
    progress = [(trainer.output/'progress.jsonl').read_text() for trainer in [current, historical]]
    assert progress[0] == progress[1]
    assert [json.loads(line)['step'] for line in progress[0].splitlines()] == [0, 1, 2, 3, 4, 5, 6]
    assert state_hash(current.model) == state_hash(historical.model)
    # 连续AdamW的一阶/二阶矩及step也必须逐值相同，不能只比较最终权重。
    for a, b in zip(current.optimizer.state.values(), historical.optimizer.state.values()):
        assert a.keys() == b.keys()
        assert all(torch.equal(a[key], b[key]) for key in a)
    assert len(current.optimizer.state) == len(historical.optimizer.state)
    first = json.loads((current.output/'first_update.json').read_text())
    assert first == json.loads((historical.output/'first_update.json').read_text())
    return dict(steps=6, training_rows_per_epoch=8192, validation_rows=4096,
        step_loss_and_state_hash_exact=True, optimizer_moments_and_steps_exact=True,
        validation_and_selected_checkpoints_exact=True, first_update=first, trace=traces['current'])


def audit_data(data, batches):
    manifest = json.loads((data/'manifest.json').read_text())
    result = {}
    for split, loader in batches.items():
        expected = pl.read_parquet(data/f'{split}.parquet')
        ids = torch.cat([batch['request_id'] for batch in loader]).numpy()
        np.testing.assert_array_equal(ids, expected['request_id'].to_numpy()[:len(ids)])
        order_hash = hashlib.sha256(ids.astype(expected['request_id'].to_numpy().dtype).tobytes()).hexdigest()
        full = len(ids) == expected.height
        if full:
            assert order_hash == manifest['splits'][split]['request_order_sha256']
        assert all(torch.equal(batch['elapsed_min'], batch['elapsed_min_raw']) for batch in loader)
        assert all(torch.equal(batch['req_idx'].float(), batch['req_idx_raw'].float()) for batch in loader)
        assert all(torch.isfinite(value).all() for batch in loader for value in batch.interaction.values())
        detail = dict(rows=len(ids), full=full, batches=len(loader), request_order_sha256=order_hash,
                      raw_clock_contract_exact=True, finite=True)
        if full:
            counts = expected.group_by('session_id').agg(pl.col(BUDGET_PARAMETER_FIELDS).n_unique())
            assert counts.select(pl.all().exclude('session_id').max()).row(0) == (1,)*34
            assert expected.filter(pl.col('req_idx_raw') < 1).is_empty()
            expected_label = expected.with_columns((pl.col('req_idx_raw') < pl.col('req_idx_raw').max().over('session_id')).cast(pl.Int8).alias('expected_y'))
            assert (expected_label['expected_y'] == expected_label['y_continue']).all()
            detail.update(budget_parameters_fixed_within_session=True, terminal_labels_exact=True)
        result[split] = detail
    return result


def audit_saved_checkpoint(args, manager, batches, options, output):
    source = args.source_run/f'{args.seed}/{args.experiment}'
    metric = json.loads((source/f'{args.budget}ep/metrics.json').read_text())
    progress = [json.loads(line) for line in (source/'joint/progress.jsonl').read_text().splitlines()]
    best = min((row for row in progress if row['step'] <= args.budget*421), key=lambda row: row['valid']['overall']['logloss'])
    assert best == metric['selected']
    checkpoint = torch.load(source/f'joint/best_within_{args.budget}ep.pt', map_location='cpu', weights_only=False)
    assert checkpoint['selection'] == best
    model = KRExitModel(manager, **options)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    assert state_hash(model) == metric['selected_state_sha256']
    if args.stop_at == 'checkpoint':
        breakpoint()
    model.eval()
    valid_evaluator = evaluator(output/'full-valid-evaluations.log')
    with torch.no_grad():
        for batch in batches['valid']:
            valid_evaluator.collect(batch[manager.uid_field], batch[manager.label_field], batch,
                                    batch_preds_1d=model.predict(batch))
    valid_metrics = valid_evaluator.summary(best['epoch'])
    for domain, values in best['valid'].items():
        for key, value in values.items():
            assert abs(valid_metrics[domain][key]-value) < 1e-12, (domain, key)
    pred = predict_batches(model, batches['test'])
    saved = pl.read_parquet(source/f'{args.budget}ep/predictions.parquet')
    assert pred.equals(saved), '整理后checkpoint预测必须逐列逐值复现，不能只比AUC'
    y, p = pred['y_continue'].to_numpy(), pred['prediction'].to_numpy()
    auc, ll = float(roc_auc_score(y, p)), float(log_loss(y, p, labels=[0, 1]))
    assert auc == metric['overall.auc'] and ll == metric['logloss']
    details = dict(source=str(source), budget=args.budget, selected_step=best['step'],
        selected_state_sha256=state_hash(model), valid_rows=57872, test_rows=pred.height,
        full_valid_metrics_exact=True, all_test_prediction_columns_exact=True, auc=auc, logloss=ll)
    if model.mode == 'ours':
        forward, trace = inspect_forward(model, batches['test'][0].interaction)
        trace.head(16).write_parquet(output/'checkpoint_trace.parquet')
        pairs = pred.sort(['session_id', 'req_idx_raw']).with_columns(
            pl.col('log_survival').diff().over('session_id').alias('dlogS'),
            pl.col('gamma').diff().over('session_id').alias('dgamma'))
        s_increases = pairs.filter(pl.col('dlogS') > 1e-6).height
        assert s_increases == 0
        details.update(independent_formula=forward, S_increases=s_increases,
                       gamma_increases=pairs.filter(pl.col('dgamma') > 1e-6).height,
                       adjacent_pairs=pairs['dlogS'].count())
    return details


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def diagnose(args, output):
    torch.set_num_threads(4)
    options = EXPERIMENTS[args.experiment]
    original_config = json.loads((args.source_run/'config.json').read_text())
    assert original_config['experiments'][args.experiment] == options
    original_hashes = original_config['source_hashes']
    for filename, digest in original_hashes.items():
        assert hashlib.sha256((REPO_ROOT/filename).read_bytes()).hexdigest() == digest, filename
    data = WORKSPACE/'data/v3'
    assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest() == original_config['dataset_manifest_sha256']
    manager = request_manager(data/'schema', load=True)
    # train仅用于短程重放；valid/test用全量，随后单独截断valid用于短程验证。
    batches = load_batches(data, manager, row_limits={'train': 8192})
    small = dict(batches, valid=batches['valid'][:1], test=batches['test'][:1])
    data_audit = audit_data(data, batches)
    print('DATA_AUDIT', json.dumps(data_audit), flush=True)
    torch.manual_seed(args.seed)
    model = KRExitModel(manager, **options)
    batch = batches['train'][0].interaction
    views = inspect_views(model, batch)
    write_json(output/'routing.json', views)
    print('VIEWS', views['base_field_counts'], views['embedding_dimensions'], flush=True)
    if args.stop_at == 'views':
        breakpoint()
    forward, trace = inspect_forward(model, batch)
    trace.head(16).write_parquet(output/'initial_trace.parquet')
    print('FORWARD', json.dumps(forward), flush=True)
    if args.stop_at == 'forward':
        breakpoint()
    training = replay_training(manager, small, options, args.seed, output/'probe')
    if args.stop_at == 'trained':
        breakpoint()
    print('SHORT_REPLAY_EXACT', training['trace'], flush=True)
    saved = audit_saved_checkpoint(args, manager, batches, options, output)
    for filename, digest in original_hashes.items():
        assert hashlib.sha256((REPO_ROOT/filename).read_bytes()).hexdigest() == digest
    result = dict(status='PASS', experiment=args.experiment, seed=args.seed, data=data_audit,
        original_sources_unchanged=True, initial_forward=forward, short_replay=training, saved_checkpoint=saved,
        diagnostic_sources={str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), WORKSPACE/'pipeline/exit_model.py', WORKSPACE/'pipeline/exit_training.py']},
        scope='diagnostic only; 8192 train rows x 3 epochs, full saved-checkpoint valid/test; no new effectiveness experiment')
    write_json(output/'summary.json', result)
    print('DIAGNOSTIC_PASS', output/'summary.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=list(EXPERIMENTS), default='mixture_step_pace')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--budget', choices=[1, 3], type=int, default=3)
    parser.add_argument('--source-run', type=Path, default=WORKSPACE/'runs/plan13_ablation')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stop-at', choices=['views', 'forward', 'trained', 'checkpoint'])
    args = parser.parse_args()
    output = args.output or WORKSPACE/'runs/diagnostics'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'arguments.json', {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    (output/'logs').mkdir()
    with (output/'logs/run.log').open('w') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                diagnose(args, output)
            except Exception:
                write_json(output/'failure.json', dict(status='FAIL', traceback=traceback.format_exc()))
                traceback.print_exc()
                raise


if __name__ == '__main__':
    main()
