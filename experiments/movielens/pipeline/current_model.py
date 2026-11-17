"""ML1M三主类及Loss子类：当前评分后预测、默认累计pace、UID恒0。

独立Omni预算11字段、内容15字段；当前rating/5额外投影到内容/Fusion首层。
原有ML1M模型只复用构造和屏数Weibull，覆写特征view、前向及Loss。
weak: J+c*(.1N+.1R)，strong: J+N+R；detach_0.1/1:
J(stopgrad(gamma)*stay)+w*N+c_r*.1R。无辅助仅J。推理均为gamma*stay。
"""
import torch
from torch import nn
from torch.nn import functional as F
from rosen.emb.emblayer import EmbView,OmniEmbLayer
from rosen.models.msr.base import MSRModel
from experiments.movielens.pipeline.model import ML1MContinuationModel
from experiments.movielens.pipeline.current_schema import BUDGET_FIELDS,CONTENT_FIELDS,CURRENT_RATING
from experiments.kuairand.pipeline.limited_auxiliary_exit import cap_auxiliary

VARIANTS = ['none','weak','strong','detach_0.1','detach_1']


class CurrentML1MModel(ML1MContinuationModel):
    def __init__(self,manager,mode='ours',mixture=False,requests_per_session=1.,variant='weak'):
        if variant not in VARIANTS: raise ValueError(variant)
        MSRModel.__init__(self,manager,num_domains=1)
        self.mode,self.mixture = mode,mixture
        self.requests_per_session = requests_per_session
        self.last_loss_terms = {}
        self.budget_embedding = OmniEmbLayer(emb_settings=self.omni_embedding.get_settings(include_fields=BUDGET_FIELDS))
        for name,table in self.budget_embedding.emb_modules.items():
            table.load_state_dict(self.omni_embedding.emb_modules[name].state_dict())
        self.budget_view = EmbView(self.budget_embedding,include_fields=BUDGET_FIELDS)
        self.content_view = EmbView(self.omni_embedding,include_fields=CONTENT_FIELDS)
        self.budget_net = nn.Sequential(nn.Linear(self.budget_view.embedding_dim,64),nn.SiLU(),nn.Linear(64,2))
        self.content_net = nn.Sequential(nn.Linear(self.content_view.embedding_dim,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        self.fusion_net = nn.Sequential(nn.Linear(self.omni_embedding.whole.embedding_dim,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.constant_(self.content_net[-1].bias,3.)
        if mixture:
            old = self.budget_net[-1]
            with torch.random.fork_rng():new = nn.Linear(64,6)
            with torch.no_grad():
                for i,shift in enumerate([-.6931471805599453,.6931471805599453]):
                    new.weight[i*2:i*2+2].copy_(old.weight)
                    new.bias[i*2:i*2+2].copy_(old.bias)
                    new.bias[i*2].add_(shift)
                new.weight[4:6].zero_();new.bias[4:6].zero_()
            self.budget_net[-1] = new
        self.variant = variant
        with torch.random.fork_rng():self.rating_projection = nn.Linear(1,64,bias=False)
        nn.init.zeros_(self.rating_projection.weight)

    def probability(self,batch,network,view):
        hidden = network[0](view(batch))+self.rating_projection(batch[CURRENT_RATING].float().unsqueeze(-1))
        return network[1:](hidden).squeeze(-1).sigmoid()

    def components(self,batch):
        gamma,s,ls = self.budget_components(batch)
        stay = self.probability(batch,self.content_net,self.content_view)
        joint_gamma = gamma.detach() if self.variant.startswith('detach_') else gamma
        return joint_gamma*stay,gamma,stay,s,ls

    def predict(self,batch):
        if self.mode == 'fusion':return self.probability(batch,self.fusion_net,self.omni_embedding.whole)
        return self.components(batch)[0]

    def calculate_loss(self,batch):
        if self.mode == 'fusion' or self.variant == 'none':
            loss = F.binary_cross_entropy(self.predict(batch).clamp(1e-6,1-1e-6),batch['y_continue'].float())
            self.last_loss_terms = dict(joint_bce=float(loss.detach()),total=float(loss.detach()))
            return loss
        terms = self.loss_terms(batch)
        j,n,r = terms['joint_bce'],terms['left_session_nll'],terms['right_bce']
        if self.variant == 'weak':
            aux,scale = cap_auxiliary(j,.1*n+.1*r,.1)
            a,b = .1*scale,.1*scale
        elif self.variant == 'strong':
            aux,scale = n+r,j.new_ones(())
            a,b = scale,scale
        else:
            weight = .1 if self.variant == 'detach_0.1' else 1.
            right,scale = cap_auxiliary(j,.1*r,.1)
            aux = weight*n+right
            a,b = j.new_tensor(weight,dtype=torch.float64),.1*scale
        total = j+aux
        self.last_loss_terms = {k:float(v.detach()) for k,v in terms.items()}
        self.last_loss_terms.update(effective_left_weight=float(a),effective_right_weight=float(b),
            auxiliary_weighted=float(aux.detach()),total=float(total.detach()))
        return total
