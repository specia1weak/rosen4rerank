"""当前反馈强双辅助：J+NLL+右BCE，均无数值限幅，联合梯度回传双塔。"""
from experiments.kuairand.pipeline.current_feedback import CurrentFeedbackExitModel


class StrongCurrentFeedback(CurrentFeedbackExitModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, left_weight=.1, right_weight=.1, **kwargs)
        self.left_weight = self.right_weight = 1.
        self.training_strategy = 'joint BCE to both towers plus uncapped NLL and right BCE, weights 1'

    def calculate_loss(self, batch):
        terms = self.loss_terms(batch)
        total = terms['joint_bce']+terms['left_session_nll']+terms['right_bce']
        self.last_loss_terms = {k: float(v.detach()) for k, v in terms.items()}
        self.last_loss_terms['total'] = float(total.detach())
        return total
