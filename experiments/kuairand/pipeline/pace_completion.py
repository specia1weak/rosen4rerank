"""第22份pace修正补全：只补日志接口与Fusion的上一刷pace路由。

沿用v5纯时间戳、既有54字段和独立Omni；预算34、内容48字段。
Ours有pace时只追加截止上一刷的标量，首刷0；无pace不读此旁路。
Fusion仍读54字段（包括当前分钟），有pace时追加同样上一刷标量。
不调整标签、分钟先验、模型参数、初始化、Weibull公式或优化器。
无辅助严格调用原始乘积BCE；辅助定义及合计10%上限保持既有实现。
"""
import torch
from experiments.kuairand.pipeline.exit_model import KRExitModel
from experiments.kuairand.pipeline.previous_pace import PreviousPaceExitModel, previous_pace


class CompletedPaceExitModel(PreviousPaceExitModel):
    """仅补无辅助训练的逐步记录，不额外前向或改变计算图。"""
    def calculate_loss(self, batch):
        loss = super().calculate_loss(batch)
        if not self.left_weight and not self.right_weight:
            value = float(loss.detach())
            self.last_loss_terms = dict(joint_bce=value, left_session_nll=0., right_bce=0.,
                auxiliary_before_cap=0., auxiliary_weighted=0., cap_scale=1.,
                effective_left_weight=0., effective_right_weight=0., auxiliary_main_ratio=0., total=value)
        return loss


class PreviousPaceFusion(KRExitModel):
    """Fusion直接复用原MLP；只把追加pace换成上一刷。"""
    def network_view(self, batch, view):
        x = view(batch)
        return torch.cat([x, previous_pace(batch).unsqueeze(-1)], -1) if self.add_pace else x


VARIANTS = {'none': (0., 0.), 'left': (.1, 0.), 'right': (0., .1), 'both': (.1, .1)}


def build_model(manager, config, variant, requests_per_session):
    if config['mode'] == 'fusion':
        assert variant == 'none'
        return PreviousPaceFusion(manager, **config)
    a, b = VARIANTS[variant]
    return CompletedPaceExitModel(manager, **config, pace_scope='previous',
        left_weight=a, right_weight=b, requests_per_session=requests_per_session, max_aux_ratio=.1)
