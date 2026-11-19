import polars as pl
import pytest
import torch

from experiments.kuairand.nxt_continue_v1 import TimeBudgetExitModel, WeibullHead, load_data, prepare_data
from experiments.kuairand.run_nxt_continue import (
    ExitExperimentConfig, build_optimizer, discrete_quantiles, evaluate_budget_hit_rate, run_experiment,
)
from experiments.kuairand.tests.test_nxt_continue_data import raw_source


@pytest.mark.parametrize("coupled", [False, True])
def test_structured_probability_identity_and_extreme_gradients(coupled):
    head = WeibullHead(4, num_components=2, structured=True, coupled_bathtub=coupled)
    raw = head.proj.bias.detach().double().repeat(5000, 1)
    length = torch.arange(1, 5001, dtype=torch.float64)
    before = head.survival(raw, length - 1)
    after = head.survival(raw, length)
    pmf = head.discrete_log_mass(raw, length).exp()
    torch.testing.assert_close(pmf, before.survival - after.survival, atol=1e-14, rtol=1e-10)
    nonzero = before.survival > 1e-200
    torch.testing.assert_close(before.gamma[nonzero], (after.survival / before.survival)[nonzero])
    assert pmf.sum().item() == pytest.approx(1.0)
    assert (before.shape[:, 0] < 1).all() and (before.shape[:, 1] > 1.2).all()
    assert (before.scale[:, 1] >= before.scale[:, 0] + 3).all()
    torch.testing.assert_close(head.survival(raw, length, delta=torch.zeros_like(length)).gamma, torch.ones_like(length))
    extreme = torch.tensor([[-100., -100., -100., -100., -100., 100.], [100., 100., 100., 100., 100., -100.]], requires_grad=True)
    loss = -head.discrete_log_mass(extreme, torch.tensor([1., 1000.])).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(extreme.grad).all()
    if coupled:
        assert (before.posterior[1:, 0] <= before.posterior[:-1, 0] + 1e-14).all()


def test_ordinary_path_preserves_original_math():
    head = WeibullHead(4, num_components=2, structured=False)
    torch.manual_seed(1)
    raw, t = torch.randn(30, 6), torch.arange(30).float()
    params = raw[:, :4].reshape(30, 2, 2)
    scale = torch.exp(params[..., 0].clamp(-2., 4.5) + 1.6).double()
    shape = (0.3 + torch.nn.functional.softplus(params[..., 1])).clamp(max=5.).double()
    log_pi = torch.nn.functional.log_softmax(raw[:, 4:].double(), dim=-1)
    h, inc = head._hazard_and_increment(scale, shape, t, torch.ones_like(t))
    log_s = torch.logsumexp(log_pi - h, dim=-1)
    log_post = log_pi - h - log_s[:, None]
    expected = torch.logsumexp(log_post - inc, dim=-1).exp().to(raw.dtype)
    assert torch.equal(head.survival(raw, t).gamma, expected)
    with pytest.raises(ValueError, match="num_components == 2"):
        WeibullHead(4, num_components=1, structured=True)


def test_discrete_quantiles_are_not_capped_at_200():
    head = WeibullHead(1)
    raw = torch.tensor([[4.5, -1.0]])
    quantiles = discrete_quantiles(head, raw)[0]
    scale, shape, _ = head._parse(raw)
    expected = torch.ceil(scale[0, 0] * (-torch.log(torch.tensor([.75, .5, .25], dtype=torch.float64))).pow(1 / shape[0, 0])).long()
    assert torch.equal(quantiles, expected)
    assert quantiles[1] > 200


def test_isolated_joint_gradient_and_uid_policy(raw_source, tmp_path):
    output = prepare_data(raw_source, tmp_path / "processed")
    manager, loaders = load_data(output, batch_size=20, zero_uid=True)
    batch = next(iter(loaders["train"]))
    assert (batch["user_id"] == 0).all() and batch["user_id_raw"].unique().numel() == 2
    for detach in (True, False):
        model = TimeBudgetExitModel(manager, num_components=2, structured=True, add_pace=True, left_weight=1., detach_budget_in_joint=detach)
        params = tuple(p for m in model.budget_modules for p in m.parameters())
        # 全为继续样本时无终屏 NLL；隔离模式必须完全切断预算参数梯度。
        continuing = batch[batch["y_continue"] == 1]
        model.calculate_loss(continuing).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params) == (not detach)
        model.zero_grad(set_to_none=True)
        model.calculate_loss(batch).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    cfg = ExitExperimentConfig()
    optimizer = build_optimizer(model, cfg)
    assigned = [id(p) for g in optimizer.param_groups for p in g["params"]]
    assert len(assigned) == len(set(assigned)) == len(list(model.parameters()))
    embedding_ids = {id(p) for m in model.modules() if isinstance(m, torch.nn.Embedding) for p in m.parameters()}
    assert all(g["weight_decay"] == 0 for g in optimizer.param_groups if any(id(p) in embedding_ids for p in g["params"]))


def test_trainer_and_opening_budget_audit(raw_source, tmp_path):
    data = prepare_data(raw_source, tmp_path / "processed")
    cfg = ExitExperimentConfig(data_dir=str(data), run_root=str(tmp_path / "runs"), experiment_name="test", batch_size=4, num_threads=1)
    result = run_experiment(cfg)
    assert result["trained_optimizer_steps"] == 2
    assert result["budget"]["n_sessions"] == 2
    assert set(result["test"]["overall"]) == {"auc", "logloss", "gauc"}
    assert result["data"]["valid"]["rows"] == 4
    manager, loaders = load_data(data, batch_size=4)
    model = TimeBudgetExitModel(manager, num_components=2, structured=True)
    # 改终屏背景不应改变开场预算预测。
    batches = list(loaders["test"])
    before = evaluate_budget_hit_rate(model, batches, "cpu", tmp_path / "before", train_baseline={"median": 2.})
    for batch in batches:
        terminal = batch["y_continue"] == 0
        batch["user_hist_req_cnt"][terminal] = 10000.
    after = evaluate_budget_hit_rate(model, batches, "cpu", tmp_path / "after", train_baseline={"median": 2.})
    assert before == after
    assert pl.read_parquet(tmp_path / "before/test_budget_centers.parquet").equals(pl.read_parquet(tmp_path / "after/test_budget_centers.parquet"))
