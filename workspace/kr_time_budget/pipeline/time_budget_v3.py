"""Request models with native schema encoding and a model-owned stage transition."""
import torch
from torch import nn
from torch.nn import functional as F
from rosen.models.msr.base import MSRModel
from rosen.emb.emblayer import OmniEmbLayer, EmbView
from workspace.kr_time_budget.pipeline.kr_request_schema import BUDGET_PARAMETER_FIELDS, CONTENT_EXCLUSIVE_FIELDS
from workspace.kr_time_budget.pipeline.time_budget import mixture_survival


class ExitModelV3(MSRModel):
    def __init__(self, manager, mode='ours'):
        super().__init__(manager, num_domains=5)
        self.mode=mode
        self.stage=1 if mode=='ours' else 2
        # One owner per table. Budget has only the settings it really consumes.
        settings=[s for s in manager.settings if s.field_name in BUDGET_PARAMETER_FIELDS]
        self.budget_embedding=OmniEmbLayer(emb_settings=settings)
        for name,table in self.budget_embedding.emb_modules.items():
            table.load_state_dict(self.omni_embedding.emb_modules[name].state_dict())
        self.budget_view=EmbView(self.budget_embedding,include_fields=BUDGET_PARAMETER_FIELDS)
        full_dim=self.omni_embedding.get_output_dim()
        budget_dim=self.budget_embedding.get_output_dim()
        self.budget_net=nn.Sequential(nn.Linear(budget_dim,64),nn.SiLU(),nn.Linear(64,6))
        self.content_net=nn.Sequential(nn.Linear(full_dim,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        self.fusion_net=nn.Sequential(nn.Linear(full_dim,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.constant_(self.content_net[-1].bias,2.6)

    @property
    def budget_modules(self):
        return (self.budget_embedding,self.budget_net)

    def set_stage(self,stage):
        self.stage=stage
        if self.mode=='ours' and stage==2:
            for module in self.budget_modules:
                for p in module.parameters():
                    p.requires_grad_(False);p.grad=None

    def budget_raw(self,b):
        return self.budget_net(self.budget_view(b))

    def budget_components(self,b):
        return mixture_survival(self.budget_raw(b),b['elapsed_min'],torch.ones_like(b['elapsed_min']))

    def content_x(self,b):
        fields=self.omni_embedding(b,split_by='name')
        return torch.cat([v if self.mode!='content' or c in CONTENT_EXCLUSIVE_FIELDS else torch.zeros_like(v) for c,v in fields.items()],dim=-1)

    def components(self,b):
        gamma,s,log_s=self.budget_components(b)
        if self.mode=='ours' and self.stage==2:gamma=gamma.detach()
        stay=torch.sigmoid(self.content_net(self.content_x(b)).squeeze(-1))
        return gamma*stay,gamma,stay,s,log_s

    def predict(self,b):
        if self.mode=='fusion':return torch.sigmoid(self.fusion_net(self.content_x(b)).squeeze(-1))
        if self.mode=='content':return torch.sigmoid(self.content_net(self.content_x(b)).squeeze(-1))
        if self.mode=='budget' or self.stage==1:return self.budget_components(b)[0]
        return self.components(b)[0]

    def calculate_loss(self,b):
        return F.binary_cross_entropy(self.predict(b).clamp(1e-6,1-1e-6),b['y_continue'].float())

    def should_finish_stage_one(self,report):
        return report.epoch>=1

    def initialize_stage_two(self,snapshot):
        with torch.random.fork_rng():
            model=ExitModelV3(self.manager,self.mode)
        model.load_state_dict(snapshot.model_state)
        model.set_stage(2)
        return model
