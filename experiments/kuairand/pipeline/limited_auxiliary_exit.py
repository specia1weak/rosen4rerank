"""Plan21低强度辅助：主Loss保持原乘积BCE，辅助合计不超过它的10%。

特征沿用54字段及独立Omni编码：两塔共享30背景；预算另读4先验，
内容另读18历史/列表及pace。入口将编码UID置0，raw UID保留评估。
数据/schema、屏数Weibull公式、前向预测均不变。左辅助沿用每场一次的
离散NLL，右辅助沿用每请求继续BCE；不新增满意度标签。

A=.1*left_NLL+.1*right_BCE（只累加开启的项）；
c=min(1, .1*stopgrad(main)/stopgrad(A))；L=main+c*A。
两项合计限制，且每项有效系数<=.1。c不反传，故主Loss梯度系数仍为1，
辅助仍有梯度；否则直接构造A/A会抵消辅助梯度。这是数值贡献上限，
不声称梯度范数也被限制到10%。不依赖valid/test动态选择系数。
"""
import math
import torch
from experiments.kuairand.pipeline.auxiliary_exit_model import AuxiliaryExitModel
from experiments.kuairand.pipeline.exit_model import KRExitModel


def cap_auxiliary(main, auxiliary, max_ratio=.1):
    """返回带图辅助项及无图缩放系数；主Loss不参与缩放系数的反传。"""
    if not math.isfinite(max_ratio) or not 0 <= max_ratio <= .1:
        raise ValueError('Auxiliary/main maximum ratio must be between 0 and 0.1')
    if not torch.isfinite(main) or not torch.isfinite(auxiliary) or main < 0 or auxiliary < 0:
        raise FloatingPointError('Losses must be finite and nonnegative')
    detached_aux = auxiliary.detach().double()
    scale = torch.minimum(torch.ones_like(detached_aux),
        max_ratio*main.detach().double()/detached_aux.clamp(min=torch.finfo(torch.float64).tiny))
    return scale*auxiliary, scale


class LimitedAuxiliaryExitModel(AuxiliaryExitModel):
    """复用原辅助项定义，仅降低系数并限制合计贡献；不新增参数或随机操作。"""
    def __init__(self, manager, *, max_aux_ratio=.1, **kwargs):
        super().__init__(manager, **kwargs)
        if not math.isfinite(max_aux_ratio) or not 0 <= max_aux_ratio <= .1:
            raise ValueError('Maximum auxiliary/main ratio is 0.1')
        if max(self.left_weight, self.right_weight) > .1:
            raise ValueError('Individual auxiliary coefficients must not exceed 0.1')
        self.max_aux_ratio = max_aux_ratio

    def calculate_loss(self, batch):
        if not self.max_aux_ratio or (not self.left_weight and not self.right_weight):
            return KRExitModel.calculate_loss(self, batch)
        terms = self.loss_terms(batch)
        auxiliary = self.left_weight*terms['left_session_nll']+self.right_weight*terms['right_bce']
        weighted, scale = cap_auxiliary(terms['joint_bce'], auxiliary, self.max_aux_ratio)
        total = terms['joint_bce']+weighted
        a, b = self.left_weight*scale, self.right_weight*scale
        self.last_loss_terms = {k: float(v.detach()) for k, v in terms.items()}
        self.last_loss_terms.update(auxiliary_before_cap=float(auxiliary.detach()),
            auxiliary_weighted=float(weighted.detach()), cap_scale=float(scale),
            effective_left_weight=float(a), effective_right_weight=float(b),
            auxiliary_main_ratio=float(weighted.detach()/terms['joint_bce'].detach()),
            total=float(total.detach()))
        if self.lsp.active:
            self.lsp.add('joint_bce', terms['joint_bce'])
            self.lsp.add('left_session_nll', terms['left_session_nll'], weight=float(a))
            self.lsp.add('right_bce', terms['right_bce'], weight=float(b))
        return total
