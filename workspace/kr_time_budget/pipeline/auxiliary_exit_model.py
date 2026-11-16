"""Plan21：保持双塔特征/前向不变，只添加可消融的监督项。

特征处理沿用KR v3：30共享背景，两塔独立Omni；预算另读4先验，
内容另读18内容/反馈（pace配置再加屏末速度）；编码UID在入口置0。
预算只用屏数时钟。停止屏数L是监督标签，仅在训练终屏计算NLL，
不作为任何网络输入；既有parquet/schema不重建。

总损失 = 请求均值BCE(gamma*stay,y) + a*会话均值NLL(L) + b*请求均值BCE(stay,y)。
NLL=-log[S(L-1)-S(L)]，等价于单场one-hot KL；不宣称与个性化Q的分组KL等价。
终屏随机分散在物理shuffle的batch中，按N_requests/N_sessions校正权重，
使全epoch按batch行数加权的辅助项精确等于每场一次的NLL均值。
右塔使用同一继续标签，是额外预测监督，不能解释为新增满意度真值。
"""
import torch
from torch.nn import functional as F
from rosen.core.interaction import Interaction
from workspace.kr_time_budget.pipeline.exit_model import KRExitModel, weibull_parameters


def discrete_log_mass(raw, length, mixture=False):
    """稳定计算log P(L=l)，避免直接相减的下溢；包括L=1的S(0)=1边界。"""
    if torch.any(length < 1) or torch.any(length != length.round()):
        raise ValueError('Complete positive integer screen counts required')
    heads = raw[:, :4].reshape(-1, 2) if mixture else raw[:, :2]
    scale, shape = (v.double() for v in weibull_parameters(heads, True))
    t = length.double()-1
    if mixture:
        t = t[:, None].expand(-1, 2).reshape(-1)
    safe = t.clamp(min=1e-12)
    hazard = torch.where(t > 0, (safe/scale).pow(shape), torch.zeros_like(t))
    increment = torch.where(t > 0,
        (safe/scale).pow(shape)*torch.expm1(shape*torch.log1p(1/safe)), (1/scale).pow(shape))
    component = -hazard+torch.log(-torch.expm1(-increment))
    if mixture:
        return torch.logsumexp(torch.log_softmax(raw[:, 4:6].double(), -1)+component.reshape(-1, 2), -1)
    return component


class AuxiliaryExitModel(KRExitModel):
    """不新增参数/随机初始化，不改predict，单次backward联合优化所有开启的Loss。"""
    def __init__(self, manager, *, left_weight=0., right_weight=0., requests_per_session=1., **kwargs):
        super().__init__(manager, **kwargs)
        if self.mode != 'ours' or not self.step_only:
            raise ValueError('Plan21 uses step-only Ours')
        if min(left_weight, right_weight) < 0 or requests_per_session <= 0:
            raise ValueError('Nonnegative weights and positive normalization required')
        self.left_weight, self.right_weight = left_weight, right_weight
        self.requests_per_session = requests_per_session
        self.last_loss_terms = {}

    def loss_terms(self, batch):
        prediction, _, stay, _, _ = self.components(batch)
        y = batch['y_continue'].float()
        joint = F.binary_cross_entropy(prediction.clamp(1e-6, 1-1e-6), y)
        right = F.binary_cross_entropy(stay.clamp(1e-6, 1-1e-6), y)
        left = joint.new_zeros((), dtype=torch.float64)
        terminal = y == 0
        if self.left_weight and terminal.any():
            end = Interaction({k: batch[k][terminal] for k in batch})
            raw = self.budget_raw(end)
            step_raw = raw[:, 6:12] if self.mixture else raw[:, 2:4]
            left = -discrete_log_mass(step_raw, end['req_idx_raw'], self.mixture).sum()
            left = left*self.requests_per_session/len(y)
        return dict(joint_bce=joint, left_session_nll=left, right_bce=right)

    def calculate_loss(self, batch):
        # 两项关闭时直接走基类，便于逐梯度/逐更新证明与旧联合训练一致。
        if not self.left_weight and not self.right_weight:
            return super().calculate_loss(batch)
        terms = self.loss_terms(batch)
        total = terms['joint_bce']
        if self.left_weight:
            total = total+self.left_weight*terms['left_session_nll']
        if self.right_weight:
            total = total+self.right_weight*terms['right_bce']
        self.last_loss_terms = {k: float(v.detach()) for k, v in terms.items()}
        self.last_loss_terms['total'] = float(total.detach())
        if self.lsp.active:
            for name, weight in [('joint_bce', 1.), ('left_session_nll', self.left_weight),
                                 ('right_bce', self.right_weight)]:
                self.lsp.add(name, terms[name], weight=weight)
        return total
