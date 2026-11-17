"""ML1M评分活动继续预测：独立Omni + 单/K2屏数Weibull + 内容MLP。

预算11字段=7公共背景+4静态过去Session先验；内容15字段=7背景+8电影/历史。
UID在loader置0。内容有截止上一步pace和前序rating，无当前rating。
只用评分次数时钟r=step_idx-1，不把评分分钟解释成观影时间预算。
联合BCE使用stopgrad(gamma)*stay；预算仅左离散停止NLL，右BCE更新内容。
辅助合计≤主Loss数值10%，缩放系数停止梯度。Fusion是同全集特征黑盒对照。
数学函数与KR复用；模型不伪装为KR 54字段接口，不改已有KR源码或checkpoint。
"""
import torch
from torch import nn
from torch.nn import functional as F
from rosen.core.interaction import Interaction
from rosen.models.msr.base import MSRModel
from rosen.emb.emblayer import OmniEmbLayer, EmbView
from experiments.movielens.pipeline.schema import BUDGET_FIELDS, CONTENT_FIELDS
from experiments.kuairand.pipeline.exit_model import single_weibull_survival, mixture_weibull_survival
from experiments.kuairand.pipeline.auxiliary_exit_model import discrete_log_mass
from experiments.kuairand.pipeline.limited_auxiliary_exit import cap_auxiliary


class ML1MContinuationModel(MSRModel):
    def __init__(self, manager, mode='ours', mixture=False, requests_per_session=1.):
        super().__init__(manager, num_domains=1)
        self.mode, self.mixture = mode, mixture
        self.requests_per_session = requests_per_session
        self.left_weight = self.right_weight = self.max_aux_ratio = .1
        self.last_loss_terms = {}
        self.budget_embedding = OmniEmbLayer(emb_settings=self.omni_embedding.get_settings(include_fields=BUDGET_FIELDS))
        for name, table in self.budget_embedding.emb_modules.items():
            table.load_state_dict(self.omni_embedding.emb_modules[name].state_dict())
        self.budget_view = EmbView(self.budget_embedding, include_fields=BUDGET_FIELDS)
        self.content_view = EmbView(self.omni_embedding, include_fields=CONTENT_FIELDS)
        self.budget_net = nn.Sequential(nn.Linear(self.budget_view.embedding_dim,64),nn.SiLU(),nn.Linear(64,2))
        self.content_net = nn.Sequential(nn.Linear(self.content_view.embedding_dim,64),nn.SiLU(),
            nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        self.fusion_net = nn.Sequential(nn.Linear(self.omni_embedding.whole.embedding_dim,64),nn.SiLU(),
            nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.constant_(self.content_net[-1].bias,3.)
        if mixture:
            old = self.budget_net[-1]
            with torch.random.fork_rng():
                new = nn.Linear(64,6)
            with torch.no_grad():
                for i, shift in enumerate([-.6931471805599453,.6931471805599453]):
                    new.weight[i*2:i*2+2].copy_(old.weight)
                    new.bias[i*2:i*2+2].copy_(old.bias)
                    new.bias[i*2].add_(shift)
                new.weight[4:6].zero_(); new.bias[4:6].zero_()
            self.budget_net[-1] = new

    @property
    def budget_modules(self):
        return self.budget_embedding, self.budget_net

    def budget_raw(self,batch):
        return self.budget_net(self.budget_view(batch))

    def budget_components(self,batch):
        raw = self.budget_raw(batch)
        r = batch['req_idx_raw'].float()-1
        if self.mixture:
            return mixture_weibull_survival(raw,r)
        return single_weibull_survival(raw,r,torch.ones_like(r),True)

    def clock_components(self,batch):
        step = self.budget_components(batch)
        one = torch.ones_like(step[0])
        return dict(step=step,time=(one,one,torch.zeros_like(step[2])))

    def components(self,batch):
        gamma,s,ls = self.budget_components(batch)
        stay = self.content_net(self.content_view(batch)).squeeze(-1).sigmoid()
        return gamma.detach()*stay,gamma,stay,s,ls

    def predict(self,batch):
        if self.mode == 'fusion':
            return self.fusion_net(self.omni_embedding.whole(batch)).squeeze(-1).sigmoid()
        return self.components(batch)[0]

    def loss_terms(self,batch):
        p,_,stay,_,_ = self.components(batch)
        y = batch['y_continue'].float()
        joint = F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),y)
        right = F.binary_cross_entropy(stay.clamp(1e-6,1-1e-6),y)
        left = joint.new_zeros((),dtype=torch.float64)
        terminal = y==0
        if terminal.any():
            end = Interaction({k:batch[k][terminal] for k in batch})
            left = -discrete_log_mass(self.budget_raw(end),end['req_idx_raw'],self.mixture).sum()
            left = left*self.requests_per_session/len(y)
        return dict(joint_bce=joint,left_session_nll=left,right_bce=right)

    def calculate_loss(self,batch):
        if self.mode == 'fusion':
            return F.binary_cross_entropy(self.predict(batch).clamp(1e-6,1-1e-6),batch['y_continue'].float())
        terms = self.loss_terms(batch)
        aux = .1*terms['left_session_nll']+.1*terms['right_bce']
        weighted, scale = cap_auxiliary(terms['joint_bce'],aux,.1)
        total = terms['joint_bce']+weighted
        self.last_loss_terms = {k:float(v.detach()) for k,v in terms.items()}
        self.last_loss_terms.update(cap_scale=float(scale),auxiliary_weighted=float(weighted.detach()),
            auxiliary_main_ratio=float(weighted.detach()/terms['joint_bce'].detach()),total=float(total.detach()))
        return total
