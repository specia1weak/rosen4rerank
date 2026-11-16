"""Plan21离散概率、会话权重、梯度路由与关闭辅助后的精确回归。"""
import pytest
import torch
from torch.nn import functional as F
from rosen.core.interaction import Interaction
from workspace.kr_time_budget.tests.test_v3 import model_batch
from workspace.kr_time_budget.pipeline.auxiliary_exit_model import AuxiliaryExitModel, discrete_log_mass
from workspace.kr_time_budget.pipeline.exit_model import KRExitModel, weibull_parameters
from workspace.kr_time_budget.pipeline.exit_training import joint_parameter_groups, state_hash


@pytest.mark.parametrize('mixture', [False, True])
def test_discrete_mass_matches_direct_distribution_and_normalizes(mixture):
    n = 10000
    raw = torch.tensor([[1., -.3, .4, .5, -.2, .6]] if mixture else [[1., -.3]], dtype=torch.float64)
    raw = raw.expand(n, -1).clone().requires_grad_()
    length = torch.arange(1, n+1, dtype=torch.float64)
    logp = discrete_log_mass(raw, length, mixture)
    scale, shape = weibull_parameters(raw[:, :4].reshape(-1, 2) if mixture else raw, True)
    l = length[:, None] if mixture else length
    if mixture:
        scale, shape = scale.reshape(-1, 2), shape.reshape(-1, 2)
    direct = torch.exp(-((l-1)/scale).pow(shape))-torch.exp(-(l/scale).pow(shape))
    if mixture:
        direct = (torch.softmax(raw[:, 4:6], -1)*direct).sum(-1)
    torch.testing.assert_close(logp.exp()[:100], direct[:100], rtol=1e-11, atol=1e-14)
    assert abs(logp.exp().sum().item()-1) < 1e-10
    (-logp[[0, 1, 9999]].mean()).backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        discrete_log_mass(raw[:1], torch.tensor([0.]), mixture)


@pytest.mark.parametrize('mixture', [False, True])
def test_aux_off_matches_all_parameters_gradients_and_adam_update(model_batch, mixture):
    _, raw, manager = model_batch
    batch = Interaction(dict(raw, elapsed_min_raw=raw['elapsed_min'],
        req_idx_raw=torch.arange(len(raw['y_continue']))+1))
    models = []
    for cls in [KRExitModel, AuxiliaryExitModel]:
        torch.manual_seed(2031)
        models.append(cls(manager, mixture=mixture, step_only=True, add_pace=True))
    assert state_hash(models[0]) == state_hash(models[1])
    losses = []
    for model in models:
        optimizer = torch.optim.AdamW(joint_parameter_groups(model), weight_decay=0)
        losses.append(model.calculate_loss(batch))
        losses[-1].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
    assert torch.equal(losses[0], losses[1])
    assert state_hash(models[0]) == state_hash(models[1])
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        assert (a.grad is None and b.grad is None) or torch.equal(a.grad, b.grad)


@pytest.mark.parametrize('mixture', [False, True])
def test_loss_terms_route_to_correct_tower_and_keep_forward_unchanged(model_batch, mixture):
    _, raw, manager = model_batch
    batch = Interaction(dict(raw, elapsed_min_raw=raw['elapsed_min'], req_idx_raw=torch.ones(len(raw['y_continue']))))
    model = AuxiliaryExitModel(manager, mixture=mixture, step_only=True, add_pace=True,
        left_weight=1., right_weight=1., requests_per_session=2.)
    before = model.predict(batch).detach().clone()
    terms = model.loss_terms(batch)
    assert torch.equal(before, model.predict(batch))
    for term, active, absent in [('left_session_nll', model.budget_net, model.content_net),
                                 ('right_bce', model.content_net, model.budget_net)]:
        model.zero_grad(set_to_none=True)
        terms[term].backward(retain_graph=True)
        assert active[-1].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in absent.parameters())
    model.zero_grad(set_to_none=True)
    terms['joint_bce'].backward()
    assert model.budget_net[-1].weight.grad.abs().sum() > 0
    assert model.content_net[-1].weight.grad.abs().sum() > 0


def test_session_normalization_counts_each_session_once_despite_shuffle(model_batch):
    _, raw, manager = model_batch
    # 12行、3场：L=2,4,6；终屏位置打散。固定参数使batch拆分不改变Q。
    model = AuxiliaryExitModel(manager, step_only=True, left_weight=1., requests_per_session=4.)
    for p in model.parameters():
        p.data.zero_()
    lengths = torch.tensor([2., 4., 6.])
    index = torch.tensor([0, 2, 5])
    y = torch.ones(12)
    y[index] = 0
    steps = torch.ones(12)
    steps[index] = lengths
    batch = Interaction(dict(raw, y_continue=y, req_idx_raw=steps, elapsed_min_raw=steps))
    direct = -discrete_log_mass(torch.zeros(3, 2), lengths).mean()
    parts = []
    for start, stop in [(0, 3), (3, 8), (8, 12)]:
        small = Interaction({k: batch[k][start:stop] for k in batch})
        parts.append(model.loss_terms(small)['left_session_nll']*(stop-start)/12)
    torch.testing.assert_close(sum(parts), direct)


def test_duplicate_label_product_creates_calibration_tradeoff():
    # 即使两个分支各自精确校准p，乘积也变p²；不是三项同时最优的保证。
    p = torch.tensor(.9)
    assert p*p < p
    logits = torch.tensor([2., 2.], requires_grad=True)
    g, c = logits.sigmoid()
    loss = F.binary_cross_entropy(g*c, p)+F.binary_cross_entropy(g, p)+F.binary_cross_entropy(c, p)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('mixture', [False, True])
def test_complete_session_binary_likelihood_equals_discrete_stop_likelihood(mixture):
    # 给每场固定同一组参数，continue直到L-1，L处退出：似然应望远镜消去。
    from workspace.kr_time_budget.pipeline.exit_model import single_weibull_survival, mixture_weibull_survival
    for length in [1, 2, 7, 50]:
        raw = torch.tensor([[.7, -.4, -.2, .6, .4, -.3]] if mixture else [[.7, -.4]], dtype=torch.float64)
        raw = raw.expand(length, -1)
        r = torch.arange(length, dtype=torch.float64)
        gamma = (mixture_weibull_survival(raw,r)[0] if mixture else
                 single_weibull_survival(raw,r,torch.ones_like(r),True)[0])
        log_likelihood = torch.log(gamma[:-1]).sum()+torch.log1p(-gamma[-1])
        direct = discrete_log_mass(raw[:1],torch.tensor([float(length)]),mixture)[0]
        torch.testing.assert_close(log_likelihood,direct,rtol=1e-12,atol=1e-12)


def test_personalized_one_hot_kl_is_not_kl_of_averaged_group_prediction():
    # 同组两场分别停在1、2，Q_s各自匹配；交换个体Q后，组均值不变但NLL改变。
    correct = torch.tensor([[.9,.1],[.1,.9]])
    swapped = correct.flip(0)
    individual_a = -torch.log(correct.diag()).mean()
    individual_b = -torch.log(swapped.diag()).mean()
    assert individual_a < individual_b
    assert torch.equal(correct.mean(0), swapped.mean(0))
