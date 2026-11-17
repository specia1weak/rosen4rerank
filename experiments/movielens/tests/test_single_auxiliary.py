import pytest
import torch
from torch.utils.data import DataLoader
from rosen.data.dataset import ParquetStreamDataset
from experiments.movielens.tests.test_training import example_features
from experiments.movielens.pipeline.current_schema import manager_at,EXTRA_COLUMNS
from experiments.movielens.pipeline.build_current_information import current_information
from experiments.movielens.pipeline.current_model import CurrentML1MModel
from experiments.movielens.pipeline.single_auxiliary import SingleAuxiliaryModel


@pytest.mark.parametrize('mixture',[False,True])
def test_single_auxiliary_selected_term_and_gradients(tmp_path,mixture):
    raw = current_information(example_features());mgr = manager_at(tmp_path);mgr.fit(raw.lazy())
    frame = mgr.transform(raw.lazy()).collect()
    batch = next(iter(DataLoader(ParquetStreamDataset(frame.lazy(),mgr,batch_size=32,shuffle=False,
        drop_last=False,extra_col_names=EXTRA_COLUMNS),batch_size=None)))
    for side in ['left','right']:
        for weight in [.1,1.]:
            torch.manual_seed(2026)
            model = SingleAuxiliaryModel(mgr,mixture=mixture,side=side,weight=weight,requests_per_session=3.)
            torch.manual_seed(2026)
            base = CurrentML1MModel(mgr,mixture=mixture,variant='none',requests_per_session=3.)
            assert all(torch.equal(v,base.state_dict()[k]) for k,v in model.state_dict().items())
            terms = model.loss_terms(batch)
            budget = list(model.budget_net.parameters());content = list(model.content_net.parameters())
            jgrad = torch.autograd.grad(terms['joint_bce'],budget,retain_graph=True)
            assert sum(float(g.norm()) for g in jgrad)>0
            selected = terms['left_session_nll'] if side=='left' else terms['right_bce']
            unrelated = content if side=='left' else budget
            assert all(g is None for g in torch.autograd.grad(selected,unrelated,allow_unused=True,retain_graph=True))
            expected = terms['joint_bce']+(torch.minimum(.1*selected.double(),.1*terms['joint_bce'].double()) if weight==.1 else selected)
            actual = model.calculate_loss(batch)
            assert torch.allclose(actual.double(),expected.double(),atol=1e-8,rtol=1e-6)
            assert model.last_loss_terms['effective_right_weight' if side=='left' else 'effective_left_weight']==0.
            actual.backward()
            assert model.budget_net[-1].weight.grad.norm()>0 and model.content_net[-1].weight.grad.norm()>0
