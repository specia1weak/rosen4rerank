"""ML1M仅左/仅右辅助；联合BCE回传两塔，不使用stop-left。

×0.1沿既有协议限制辅助数值≤主项10%；×1使用实际系数1、无限幅。
不增加模型参数，不修改当前反馈数据、网络或历史训练源码。
"""
from experiments.movielens.pipeline.current_model import CurrentML1MModel
from experiments.kuairand.pipeline.limited_auxiliary_exit import cap_auxiliary


class SingleAuxiliaryModel(CurrentML1MModel):
    def __init__(self,*args,side,weight,**kwargs):
        if side not in ['left','right'] or weight not in [.1,1.]:raise ValueError((side,weight))
        super().__init__(*args,variant='none',**kwargs)
        self.side,self.aux_weight = side,weight
        self.variant = f'{side}_{weight:g}'

    def calculate_loss(self,batch):
        terms = self.loss_terms(batch)
        j = terms['joint_bce']
        auxiliary = terms['left_session_nll'] if self.side=='left' else terms['right_bce']
        weighted = self.aux_weight*auxiliary
        scale = j.new_ones(())
        if self.aux_weight==.1:weighted,scale = cap_auxiliary(j,weighted,.1)
        total = j+weighted
        self.last_loss_terms = {k:float(v.detach()) for k,v in terms.items()}
        self.last_loss_terms.update(effective_left_weight=self.aux_weight*float(scale) if self.side=='left' else 0.,
            effective_right_weight=self.aux_weight*float(scale) if self.side=='right' else 0.,
            auxiliary_weighted=float(weighted.detach()),total=float(total.detach()))
        return total
