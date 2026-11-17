"""联合BCE只更新内容塔；预算塔保留左停止NLL训练。

数据/特征沿用v5：34静态预算字段、48内容字段+上一刷pace，UID置0。
两个Omni参数独立。仅在乘积处使用stopgrad(gamma)，不冻结预算参数，
不detach左NLL。前向仍为gamma*stay，左右辅助合计仍≤主Loss数值10%。
这改变梯度路由，不改变相同权重下的推理概率或Loss数值。
预算仍接受总退出标签；隔离联合梯度不等于获得纯预算真值。
"""
import torch
from experiments.kuairand.pipeline.previous_pace import PreviousPaceExitModel
from experiments.kuairand.pipeline.limited_auxiliary_exit import cap_auxiliary


class BudgetStopGradientExitModel(PreviousPaceExitModel):
    """阻断联合乘积到预算塔的边；预算辅助NLL的计算图完整保留。"""
    def components(self, batch):
        _, gamma, stay, survival, log_s = super().components(batch)
        return gamma.detach()*stay, gamma, stay, survival, log_s


def gradient_route_audit(model, batch):
    """用真实batch逐Loss查所有活跃参数；autograd.grad不写入.grad或执行更新。"""
    budget = [p for module in model.budget_modules for p in module.parameters()]
    content = list(model.omni_embedding.parameters()) + list(model.content_net.parameters())
    assert not {id(p) for p in budget} & {id(p) for p in content}
    terms = model.loss_terms(batch)
    params = budget + content
    def grads(loss):
        return torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    def norm(items):
        return sum(float(g.detach().double().square().sum()) for g in items if g is not None)**.5
    joint, left, right = [grads(terms[k]) for k in ['joint_bce', 'left_session_nll', 'right_bce']]
    n = len(budget)
    assert all(g is None for g in joint[:n])
    assert all(g is None for g in right[:n])
    assert all(g is None for g in left[n:])
    assert norm(left[:n]) > 0 and norm(joint[n:]) > 0
    aux = model.left_weight*terms['left_session_nll'] + model.right_weight*terms['right_bce']
    weighted, scale = cap_auxiliary(terms['joint_bce'], aux, model.max_aux_ratio)
    total = grads(terms['joint_bce'] + weighted)
    for actual, expected in zip(total[:n], left[:n]):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected*(scale*model.left_weight).to(expected.dtype),
                                       atol=1e-8, rtol=1e-6)
    return dict(joint_to_budget='all None', right_to_budget='all None', left_to_content='all None',
                joint_to_content_norm=norm(joint[n:]), left_to_budget_norm=norm(left[:n]),
                total_to_budget_norm=norm(total[:n]), total_budget_equals_scaled_left=True,
                budget_parameters_trainable=all(p.requires_grad for p in budget),
                budget_embedding_shared_with_content=False, cap_scale=float(scale))
