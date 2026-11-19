"""Behavior, equations and Trainer integration for the independent rerankers."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rosen.core.enum_type import FeatureSource
from rosen.core.interaction import Interaction
from rosen.core.train import TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer
from rosen.emb import SchemaManager, SlateLabel
from rosen.emb.schema import SparseEmbSetting, SequenceSetting, SeqGroupConfig
from rosen.evaluate import RankingEvaluator
from rosen.evaluate.evaluator import PointwiseEvaluator
from rosen.evaluate.manager import EvaluatorManager
from rosen.models.rerank import RERANK_MODEL_REGISTRY, select_by_order, build_rerank_model, masked_bce_with_logits
from rosen.models.rerank.dcn import CrossNetwork
from rosen.models.rerank.extr import EXTRAttentionBlock
from rosen.models.rerank.kuaishou import TargetContextPool
from rosen.models.rerank.mge import MultiGERerankModel


NAMES = ("dnn", "dcn", "prm", "extr", "kuaishou", "pier", "prs", "nar", "seq2slate",
         "pier_submodular", "mge_metric", "mge_cosine_metric")


def case(tmp_path, name, **overrides):
    torch.manual_seed(71)
    atoms = [SparseEmbSetting(n, FeatureSource.ITEM, embedding_dim=4) for n in ("item", "category")]
    for atom in atoms:
        atom._build_vocab_indices([str(i) for i in range(12)])
    current, history = SeqGroupConfig("slate", "slate_len", 3), SeqGroupConfig("history", "history_len", 6)
    settings = [SequenceSetting(f"cur_{s.field_name}", s, current) for s in atoms]
    settings += [SequenceSetting(f"hist_{s.field_name}", s, history) for s in atoms]
    manager = SchemaManager(settings, tmp_path / name / "schema", label_fields=SlateLabel("label", 3, "mask"))
    kwargs = dict(slate_cate_group="slate", hist_cate_group="history", slate_dense_field="prior",
                  page_mask_field="pages", poi_num=3, page_num=2, mlp_a=(8,), mlp_b=(8,))
    if name in ("pier", "prs", "pier_submodular", "nar", "seq2slate", "mge_metric", "mge_cosine_metric", "mge3", "mge_cosine"):
        kwargs.update(top_k=2, eval_all_perms=False, candidate_chunk_size=2)
    if name in ("nar", "seq2slate", "mge_metric", "mge_cosine_metric", "mge3", "mge_cosine"):
        kwargs["hidden_dim"] = 8
    kwargs.update(overrides)
    model = build_rerank_model(manager, name, **kwargs)
    batch = Interaction({
        "cur_item": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "cur_category": torch.tensor([[3, 1, 2], [6, 4, 5]]),
        "hist_item": torch.tensor([[1, 3, 2, 3, 2, 1], [6, 4, 5, 6, 5, 4]]),
        "hist_category": torch.tensor([[3, 1, 2, 1, 2, 3], [4, 5, 6, 5, 4, 6]]),
        "slate_len": torch.tensor([3, 3]), "history_len": torch.tensor([6, 6]),
        "prior": torch.tensor([[.1, .5, .9], [.7, .3, .2]]),
        "pages": torch.tensor([[1., 0.], [1., 1.]]),
        "label": torch.tensor([[1., 0., 0.], [0., 1., 1.]]),
        "mask": torch.tensor([[1., 1., 0.], [1., 1., 1.]]),
    })
    return model, batch


def test_registry_exposes_distinct_implementations():
    assert all(name in RERANK_MODEL_REGISTRY for name in NAMES)
    assert len({RERANK_MODEL_REGISTRY[name] for name in NAMES}) == len(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_trainable_predictions_loss_and_checkpoint_restore(tmp_path, name):
    model, batch = case(tmp_path, name)
    loader = DataLoader([batch], batch_size=None)
    evaluators = EvaluatorManager(model.manager)
    evaluators.register("overall", PointwiseEvaluator("auc", "logloss"))
    evaluators.register("order", RankingEvaluator("ndcg@3"))
    cfg = SimpleNamespace(device="cpu", max_epochs=1, train_log_interval=100,
                          ckpt_dir=str(tmp_path / name / "checkpoints"), experiment_name=name, dataset_name="synthetic")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    trainer = BaseTrainer(model, optimizer, model.manager, TrainerDataLoaders(loader, loader),
                          TrainerComponents(evaluator_manager=evaluators), cfg)
    before = {key: value.clone() for key, value in model.named_parameters()}
    trainer.run()
    assert any(not torch.equal(before[key], value) for key, value in model.named_parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        prediction = model.predict(batch)
    assert prediction.scores.shape == prediction.order.shape == (2, 3)
    assert model.outputs_candidate_pool == (prediction.candidates is not None)
    assert not {'topk', 'topk_order', 'chosen_scores'} & prediction.extra.keys()
    if model.outputs_candidate_pool:
        assert prediction.chosen_scores.shape == prediction.order.shape
        assert prediction.resolve_chosen_scores() is prediction.chosen_scores
    else:
        torch.testing.assert_close(prediction.resolve_chosen_scores(), prediction.scores.gather(-1, prediction.order))
    assert torch.isfinite(prediction.scores).all()
    assert ((prediction.scores >= 0) & (prediction.scores <= 1)).all()
    torch.testing.assert_close(prediction.order.sort(-1).values, torch.arange(3).expand(2, 3))
    terms = model.lsp.summary()["settlements"]["main"]
    assert terms["total"] == pytest.approx(sum(v["weighted"] for v in terms["terms"].values()), abs=1e-5)
    checkpoint = trainer.save_checkpoint("roundtrip")
    with torch.no_grad():
        next(model.parameters()).add_(10)
    trainer.load_checkpoint(checkpoint)
    unlabeled = Interaction({key: batch[key] for key in batch.columns if key != "label"})
    with torch.no_grad():
        restored = model.predict(unlabeled)
    torch.testing.assert_close(restored.scores, prediction.scores, rtol=0, atol=0)
    torch.testing.assert_close(restored.order, prediction.order, rtol=0, atol=0)
    teacher = copy.deepcopy(model)
    assert teacher.slate_view.omni is teacher.omni_embedding
    assert all(p.ndim <= 2 for p in model.parameters())


@pytest.mark.parametrize("name", ["dnn", "dcn", "prm"])
def test_item_scorers_are_permutation_equivariant(tmp_path, name):
    model, batch = case(tmp_path, name)
    model.eval()
    order = torch.tensor([2, 0, 1])
    changed = Interaction({key: batch[key].clone() for key in batch.columns})
    for key in ("cur_item", "cur_category", "prior", "label", "mask"):
        changed[key] = changed[key][:, order]
    torch.testing.assert_close(model.logits(changed), model.logits(batch)[:, order])


def test_dnn_is_item_local_and_prm_uses_other_items(tmp_path):
    for name in ("dnn", "dcn", "prm"):
        model, batch = case(tmp_path, name)
        batch["prior"].requires_grad_()
        model.logits(batch)[1, 0].backward()
        cross_slot = batch["prior"].grad[1, 1].abs()
        assert cross_slot > 0 if name == "prm" else cross_slot == 0


def test_cross_network_matches_hand_computed_vector_recurrence():
    network = CrossNetwork(2, 2)
    with torch.no_grad():
        network.weights[0].weight.copy_(torch.tensor([[1., 2.]]))
        network.weights[1].weight.copy_(torch.tensor([[-1., .5]]))
        network.biases[0].copy_(torch.tensor([.1, -.2]))
        network.biases[1].copy_(torch.tensor([.3, .4]))
    x = torch.tensor([[[1., 3.]]])
    first = torch.tensor([[[8.1, 23.8]]])
    second = torch.tensor([[[12.2, 35.6]]])
    torch.testing.assert_close(network(x), torch.cat((x, first, second), -1))


def test_extr_residual_structure_masks_and_causality():
    torch.manual_seed(4)
    block = EXTRAttentionBlock(8, heads=2, attention_dropout=0, ffn_dropout=0)
    query, context = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    valid = torch.ones(2, 3, dtype=torch.bool)
    baseline = block(query, context, valid, valid, causal=True)
    changed = context.clone()
    changed[:, -1] += torch.randn(2, 8) * 100
    actual = block(query, changed, valid, valid, causal=True)
    torch.testing.assert_close(actual[:, :2], baseline[:, :2], rtol=0, atol=0)
    assert not torch.allclose(actual[:, 2], baseline[:, 2])
    valid[:, -1] = False
    masked = block(query, context, valid, valid)
    torch.testing.assert_close(block(query, changed, valid, valid), masked, rtol=0, atol=0)
    assert masked[:, -1].count_nonzero() == 0
    for layer in block.modules():
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
    expected = F.layer_norm(query, (8,), eps=1e-8) * valid[..., None]
    torch.testing.assert_close(block(query, context, valid, valid), expected)


def test_attention_initialization_uses_linear_fan_in_and_keeps_delta_visible():
    torch.manual_seed(37)
    pool = TargetContextPool(3, 64)
    expected_std = (2 / (64 + 64)) ** .5
    for layer in (pool.q, pool.k, pool.v):
        assert .85 < layer.weight.std().item() / expected_std < 1.15
    x = torch.randn(32, 3, 64)
    weights = (pool.q(x) @ pool.k(x).transpose(-1, -2) / 8).softmax(-1)
    delta = weights @ pool.v(x)
    ratio = delta.square().mean().sqrt() / x.square().mean().sqrt()
    assert .1 < ratio < 3


def test_kuaishou_pool_residual_and_empty_history_formula():
    pool = TargetContextPool(2, 2, scale=1)
    with torch.no_grad():
        for layer in (pool.q, pool.k, pool.v):
            layer.weight.copy_(torch.eye(2))
        pool.pool.weight.copy_(torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]]))
    target = torch.tensor([[[1., 0.]]])
    context = torch.tensor([[[1., 2.], [3., 4.]]])
    weights = torch.tensor([1., 3.]).softmax(0)
    delta = weights[0] * context[:, 0] + weights[1] * context[:, 1]
    torch.testing.assert_close(pool(target, context, torch.ones(1, 2, dtype=torch.bool)),
                               (context.sum(1) + 2 * delta)[:, None])
    assert pool(target, context, torch.zeros(1, 2, dtype=torch.bool)).count_nonzero() == 0


@pytest.mark.parametrize("name", ["kuaishou", "pier", "prs"])
def test_invalid_history_cannot_change_current_scores(tmp_path, name):
    model, batch = case(tmp_path, name)
    changed = Interaction({key: batch[key].clone() for key in batch.columns})
    for key in ("hist_item", "hist_category"):
        changed[key][0, 3:] = 9
    torch.testing.assert_close(model.logits(changed)[0], model.logits(batch)[0], rtol=0, atol=0)
    batch["pages"].zero_()
    assert torch.isfinite(model.logits(batch)).all()


def test_pier_candidates_match_sample_and_permutation_loops(tmp_path):
    model, batch = case(tmp_path, "pier", eval_all_perms=True)
    cate, history, dense, pages = model._inputs(batch)
    orders = model.retriever.full_permutations
    actual = model.candidate_logits(cate, history, dense, pages, orders)
    expected = []
    for b in range(2):
        sample = []
        for order in orders:
            value, _ = model.evaluator(cate[b:b+1, order], model.oam(history[b:b+1])[0], dense[b:b+1, order], pages[b:b+1],
                                       model.oam, model.intra_poi, model.target_attention)
            sample.append(value[0])
        expected.append(torch.stack(sample))
    torch.testing.assert_close(actual, torch.stack(expected))
    pred = model.predict(batch)
    best = pred.candidates.scores.mean(-1).argmax(-1)
    torch.testing.assert_close(pred.order, pred.candidates.orders[torch.arange(2), best])
    assert pred.extra["all"].shape == (2, 6, 3)
    assert not any("generator" in name for name, _ in model.named_parameters())


def test_prs_supervises_raw_beam_logits_with_transported_labels(tmp_path, monkeypatch):
    model, batch = case(tmp_path, "prs", use_contrastive_loss=False)
    batch["mask"].fill_(1)
    monkeypatch.setattr(model, "candidate_logits", lambda cate, hist, prior, pages, order:
                        4 * select_by_order(prior.squeeze(-1), order))
    actual = model.loss_components(batch)["supervised"]
    # Gathering logits/labels/masks together leaves this additive BCE invariant.
    expected = masked_bce_with_logits(4 * batch["prior"], batch["label"], batch["mask"], "nonzero_mean")
    torch.testing.assert_close(actual, expected)
    wrong = F.binary_cross_entropy_with_logits((4 * batch["prior"]).sigmoid(), batch["label"])
    assert not torch.allclose(actual, wrong)


@pytest.mark.parametrize("name,index", [("nar", 1), ("seq2slate", 2)])
def test_reference_single_generator_selection_and_batch_mean_loss(tmp_path, name, index):
    model, batch = case(tmp_path, name)
    inputs = model._inputs(batch)
    full_orders, full_aux = MultiGERerankModel._generate(model, *inputs, teacher_forcing=False)
    selected, aux = model._generate(*inputs, teacher_forcing=False)
    torch.testing.assert_close(selected, full_orders[:, index:index+1])
    torch.testing.assert_close(aux["candidate_dense"], full_aux["candidate_dense"][:, index:index+1])
    terms = model.loss_components(batch)
    _, train = MultiGERerankModel._generate(model, *inputs, labels=batch["label"], label_mask=batch["mask"], teacher_forcing=True)
    torch.testing.assert_close(terms["gen.generative"], train["generative"].loss)
    assert model.predict(batch).candidates.scores.shape == (2, 1, 3)


def test_model_configuration_errors_fail_early(tmp_path):
    with pytest.raises(ValueError, match="partition"):
        case(tmp_path, "extr", natural_indices=[0, 1], ad_indices=[1, 2])
    with pytest.raises(ValueError, match="divisible"):
        case(tmp_path, "extr", num_heads=3)
    with pytest.raises(ValueError, match="requires hist"):
        case(tmp_path, "pier", hist_cate_group=None)
    with pytest.raises(ValueError, match="submodularity"):
        case(tmp_path, "nar", submodularity_mode="lc")
    with pytest.warns(UserWarning, match="unsupported"):
        case(tmp_path, "dnn", typo_width=10)


def test_pier_submodular_appends_farthest_non_retrieved_candidate(tmp_path):
    model, batch = case(tmp_path, "pier_submodular")
    cate, history, dense, pages = model._inputs(batch)
    distance = model.retriever(cate, history, pages)
    orders = model._top_orders(cate, distance)
    ids = model.retriever.top_k(distance, model.top_k)
    perms = model.retriever.full_permutations
    top = perms[ids]
    mean = select_by_order(cate, top).mean(1)
    for b in range(2):
        available = [i for i in range(len(perms)) if i not in ids[b].tolist()]
        farthest = max(available, key=lambda i: (cate[b, perms[i]] - mean[b]).square().sum().item())
        torch.testing.assert_close(orders[b, -1], perms[farthest])
    torch.testing.assert_close(orders[:, :2], top)
    pred = model.predict(batch)
    assert pred.candidates.scores.shape == (2, 3, 3)
    assert pred.extra["candidate_lc_index"].shape == (2,)
    model.top_k = len(perms)
    assert model._top_orders(cate, distance).shape == (2, 6, 3)


@pytest.mark.parametrize("name,count,batch_sum", [("mge_metric", 3, True), ("mge_cosine_metric", 4, False)])
def test_metric_variant_supervises_best_greedy_candidate(tmp_path, name, count, batch_sum):
    model, batch = case(tmp_path, name)
    cate, history, dense, pages = model._inputs(batch)
    orders, aux = model._generate(cate, history, dense, pages, teacher_forcing=False)
    assert orders.shape == (2, count, 3)
    logits = model.candidate_logits(cate.detach(), history, dense.detach(), pages, orders,
                                     candidate_dense=aux["candidate_dense"])
    candidates = logits.sigmoid() * select_by_order(batch["mask"], orders)
    best = candidates.mean(-1).argmax(-1)
    expected = []
    for b in range(2):
        order = orders[b, best[b]]
        expected.append(F.binary_cross_entropy_with_logits(logits[b, best[b]], batch["label"][b, order],
                                                           weight=batch["mask"][b, order], reduction="sum"))
    mask = select_by_order(batch["mask"], orders)[torch.arange(2), best]
    expected = sum(expected) / mask.count_nonzero()
    terms = model.loss_components(batch)
    torch.testing.assert_close(terms["evaluator"], expected)
    _, train = model._generate(cate, history, dense, pages, labels=batch["label"], label_mask=batch["mask"], teacher_forcing=True)
    torch.testing.assert_close(terms["gen.generative"], train["generative"].loss)
    pred = model.predict(batch)
    assert torch.isfinite(pred.extra["candidate_mean_l2"]).all()


def test_named_three_and_four_generator_presets(tmp_path):
    for name, mode, count in (("mge3", "none", 3), ("mge_cosine", "cosine", 4)):
        model, batch = case(tmp_path, name)
        assert model.submodularity_mode == mode
        assert model.candidate_dense_mode == "generator"
        assert model.predict(batch).candidates.scores.shape == (2, count, 3)
