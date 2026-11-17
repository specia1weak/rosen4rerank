"""当刷反馈：联合BCE对预算塔stop-gradient，预算仅由系数0.1或1的NLL训练。

保留当刷pace、四反馈、UID=0及原参数初始化。右辅助仍为0.1*BCE，
单独限制到joint的10%；左NLL按本次明确要求不截值、不套辅助限幅。
预测保持gamma*stay，detach仅改变训练图，不改变前向数值。
"""
import math
import torch
from experiments.kuairand.pipeline.current_feedback import CurrentFeedbackExitModel
from experiments.kuairand.pipeline.limited_auxiliary_exit import cap_auxiliary


class CurrentFeedbackNLLOnly(CurrentFeedbackExitModel):
    def __init__(self, *args, nll_weight, **kwargs):
        if not math.isfinite(nll_weight) or nll_weight not in (.1, 1.):
            raise ValueError('This experiment compares actual NLL coefficients 0.1 and 1')
        super().__init__(*args, left_weight=.1, right_weight=.1, max_aux_ratio=.1, **kwargs)
        self.nll_weight = nll_weight
        self.training_strategy = 'joint BCE detached budget; budget trained only by NLL'

    def components(self, batch):
        _, gamma, stay, survival, log_s = super().components(batch)
        return gamma.detach()*stay, gamma, stay, survival, log_s

    def calculate_loss(self, batch):
        terms = self.loss_terms(batch)
        right, scale = cap_auxiliary(terms['joint_bce'], .1*terms['right_bce'], .1)
        left = self.nll_weight*terms['left_session_nll']
        total = terms['joint_bce']+left+right
        self.last_loss_terms = {k: float(v.detach()) for k, v in terms.items()}
        self.last_loss_terms.update(left_weighted=float(left.detach()),
            right_weighted=float(right.detach()), right_cap_scale=float(scale),
            effective_left_weight=self.nll_weight, effective_right_weight=.1*float(scale),
            right_main_ratio=float(right.detach()/terms['joint_bce'].detach()),
            total=float(total.detach()))
        return total
