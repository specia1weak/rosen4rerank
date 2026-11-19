import itertools
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rosen.core.enum_type import FeatureSource
from rosen.core.interaction import Interaction
from rosen.core.train import EMAConfig, EarlyStopper, TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer, EMATrainer
from rosen.emb import SchemaManager, SlateLabel
from rosen.emb.schema import SeqGroupConfig, SequenceSetting, SparseEmbSetting
from rosen.evaluate import BatchResolver, RankingEvaluator
from rosen.evaluate.evaluator import PointwiseEvaluator
from rosen.evaluate.manager import EvaluatorManager
from rosen.models.rerank import (
    MultiGERerankModel, select_by_order, build_full_permutations,
    build_rerank_model, fold_history_pages, masked_bce_with_logits,
)
from rosen.models.rerank.components import (
    BeamSearchRetriever, IntraPOISelfAttention, OAMBlock, PageTargetAttention, PPMLSimHashRetriever,
)
from rosen.models.rerank.components.generators import cosine_coverage, lc_exact_score, lc_soft_score


def example(tmp_path, mode="lc", k=3, p=2, **kwargs):
    atoms = [SparseEmbSetting(name, FeatureSource.ITEM, embedding_dim=4) for name in ("item", "cate")]
    for atom in atoms:
        atom._build_vocab_indices([str(i) for i in range(10)])
    slate = SeqGroupConfig("slate", "slate_len", max_len=k)
    history = SeqGroupConfig("history", "hist_len", max_len=p * k)
    settings = [SequenceSetting(f"cur_{s.field_name}", s, slate) for s in atoms]
    settings += [SequenceSetting(f"pre_{s.field_name}", s, history) for s in atoms]
    schema = SchemaManager(settings, tmp_path / "schema", label_fields=SlateLabel("label", k, "mask"))
    model = MultiGERerankModel(
        schema, slate_dense_field="prior", slate_cate_group="slate", hist_cate_group="history",
        page_mask_field="pages", poi_num=k, page_num=p, top_k=2, mlp_a=(8,), mlp_b=(8,),
        hidden_dim=8, submodularity_mode=mode, **kwargs,
    )
    ids = torch.arange(2 * k).reshape(2, k) % 8 + 1
    hist = torch.arange(2 * p * k).reshape(2, p * k) % 8 + 1
    batch = Interaction({
        "cur_item": ids, "cur_cate": ids.flip(-1), "pre_item": hist, "pre_cate": hist.flip(-1),
        "slate_len": torch.full((2,), k), "hist_len": torch.full((2,), p * k),
        "prior": torch.linspace(.1, .9, 2 * k).reshape(2, k),
        "label": (torch.arange(2 * k).reshape(2, k) % 2).float(),
        "mask": torch.ones(2, k), "pages": torch.ones(2, p),
    })
    batch["mask"][0, -1] = 0
    batch["pages"][0, -1] = 0
    return model, batch


@pytest.mark.parametrize("tail", [(), (1,), (2, 4)])
@pytest.mark.parametrize("batched", [False, True])
def test_permutation_gather_values_and_gradients(tail, batched):
    slate = torch.arange(2 * 3 * math.prod(tail), dtype=torch.float64).reshape(2, 3, *tail).requires_grad_()
    permutations = torch.tensor([[2, 0, 1], [1, 2, 0]])
    if batched:
        permutations = torch.stack((permutations, permutations.flip(0)))
    actual = select_by_order(slate, permutations)
    expected = torch.stack([slate[b, permutations[b] if batched else permutations] for b in range(2)])
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(slate.grad, torch.full_like(slate, 2))


def test_fold_full_permutations_and_validation():
    x = torch.arange(48).reshape(2, 4, 6).transpose(1, 2)
    torch.testing.assert_close(fold_history_pages(x, 2, 3).flatten(1, 2), x)
    assert build_full_permutations(3).tolist() == list(map(list, itertools.permutations(range(3))))
    for k in (0, 8, True, 2.5):
        with pytest.raises(ValueError):
            build_full_permutations(k)
    with pytest.raises(ValueError, match="length"):
        fold_history_pages(x, 3, 3)
    with pytest.raises(ValueError, match="batch/slot"):
        select_by_order(torch.ones(2, 3), torch.zeros(3, 2, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="long"):
        select_by_order(torch.ones(2, 3), torch.zeros(2, 3))


def test_masked_bce_reductions_sentinels_and_zero_mask():
    logits = torch.tensor([[0., math.log(3)], [0., -math.log(3)]], requires_grad=True)
    labels = torch.tensor([[1., 1.], [float("nan"), 0.]])
    mask = torch.tensor([[1., 1.], [0., 1.]])
    expected = math.log(2) + 2 * math.log(4 / 3)
    assert masked_bce_with_logits(logits, labels, mask).item() == pytest.approx(expected / 4)
    assert masked_bce_with_logits(logits, labels, mask, "valid_mean").item() == pytest.approx(expected / 3)
    loss = masked_bce_with_logits(logits, labels, torch.zeros_like(mask), "valid_mean")
    loss.backward()
    assert loss == 0
    assert logits.grad.count_nonzero() == 0
    with pytest.raises(ValueError, match="identical"):
        masked_bce_with_logits(logits, labels[0])


def test_masked_bce_nonzero_mean_matches_weighted_tf_reduction():
    logits = torch.zeros(1, 4, requires_grad=True)
    labels = torch.tensor([[1., 0., float("nan"), 1.]])
    mask = torch.tensor([[.5, 2., 0., 1.]])
    loss = masked_bce_with_logits(logits, labels, mask, "nonzero_mean")
    assert loss.item() == pytest.approx(3.5 * math.log(2) / 3)
    assert masked_bce_with_logits(logits, labels, mask, "valid_mean").item() == pytest.approx(math.log(2))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-.25, 1., 0., -.5]]) / 3)
    empty = masked_bce_with_logits(logits, labels, torch.zeros_like(mask), "nonzero_mean")
    assert torch.isfinite(empty) and empty == 0


def test_oam_matches_explicit_attribute_and_poi_loops():
    torch.manual_seed(17)
    oam = OAMBlock(3, 2, 4)
    x = torch.randn(2, 3, 2, 4, requires_grad=True)
    pools, deltas = [], []
    for f in range(2):
        values = x[:, :, f]
        delta = (oam.intra.q(values) @ oam.intra.k(values).transpose(-1, -2) / 8).softmax(-1) @ oam.intra.v(values)
        deltas.append(delta)
        pools.append(oam.intra.pool((values + delta).flatten(1)))
    pooled = torch.stack(pools, 1)
    inter = (oam.inter.q(pooled) @ oam.inter.k(pooled).transpose(-1, -2) / 8).softmax(-1) @ oam.inter.v(pooled)
    expected = oam.inter.pool((pooled + inter).flatten(1))
    page, per_poi = oam(x)
    torch.testing.assert_close(per_poi, torch.stack(deltas, 2))
    torch.testing.assert_close(page, expected)
    intra_poi = IntraPOISelfAttention(2, 4)
    expected_poi = []
    for slot in range(3):
        values = per_poi[:, slot]
        delta = (intra_poi.q(values) @ intra_poi.k(values).transpose(-1, -2) / 8).softmax(-1) @ intra_poi.v(values)
        expected_poi.append(intra_poi.pool((values + delta).flatten(1)))
    torch.testing.assert_close(intra_poi(per_poi), torch.stack(expected_poi, 1))
    # A new leading history-page axis must act as a batch axis.
    pages, _ = oam(x[:, None].expand(-1, 2, -1, -1, -1))
    torch.testing.assert_close(pages, page[:, None].expand(-1, 2, -1))
    page.square().mean().backward()
    assert x.grad.abs().sum() > 0
    assert all(param.ndim == 2 for param in oam.parameters())
    # Check branch scale without claiming a universal architecture threshold.
    ratio = per_poi.square().mean().sqrt() / x.square().mean().sqrt()
    assert .05 < ratio < 5


def test_target_attention_preserves_batch_alignment_and_zero_history():
    attention = PageTargetAttention()
    history = torch.tensor([[[1., 0.], [2., 0.]], [[0., 5.], [0., 7.]]])
    mask = torch.tensor([[1., 0.], [0., 1.]])
    target = torch.randn(2, 3, 2)
    expected = torch.tensor([[[1., 0.]] * 3, [[0., 7.]] * 3])
    torch.testing.assert_close(attention(target, history, mask), expected)
    torch.testing.assert_close(attention(target.flatten(0, 1), history, mask), expected.flatten(0, 1))
    assert attention(target, history, torch.zeros_like(mask)).count_nonzero() == 0


def test_ppml_simhash_and_exposure_ranking_match_loop_reference():
    torch.manual_seed(3)
    rng_state = torch.random.get_rng_state().clone()
    retriever = PPMLSimHashRetriever(3, 2, 2, position_encoding=[[1., .5], [.2, 2.], [3., .1]],
                                    hash_vector=[[1., -1.], [-2., 1.]], distance_weight=[[.5, .25]])
    assert torch.equal(rng_state, torch.random.get_rng_state())
    assert not list(retriever.parameters())
    slate = torch.randn(2, 3, 2, 2)
    hist = torch.randn(2, 2, 3, 2, 2)
    actual = retriever(slate, hist)
    expected = torch.zeros_like(actual)
    for b in range(2):
        for m, order in enumerate(retriever.full_permutations):
            candidate = (slate[b, order] * retriever.position_encoding[:, None]).mean((0, 1))
            for p in range(2):
                page = (hist[b, p] * retriever.position_encoding[:, None]).mean((0, 1))
                distance = (torch.sign(candidate * retriever.hash_vector[p]) - torch.sign(page * retriever.hash_vector[p])).abs().sum()
                expected[b, m] += distance * retriever.distance_weight[0, p]
    torch.testing.assert_close(actual, expected)
    assert retriever.top_k(actual, 2).shape == (2, 2)
    assert retriever.last_k(actual, 2).shape == (2, 2)
    assert retriever(slate, hist, torch.zeros(2, 2)).count_nonzero() == 0
    beam = BeamSearchRetriever(3)
    prior = torch.tensor([[.2, .9, .5], [.6, .1, .3]])
    candidate_priors = select_by_order(prior, retriever.full_permutations)
    scores = (candidate_priors * torch.tensor([1., .9, .8])).sum(-1)
    torch.testing.assert_close(beam(candidate_priors, 2), scores.argsort(dim=-1, descending=True, stable=True)[:, :2])
    with pytest.raises(ValueError, match="finite constant"):
        PPMLSimHashRetriever(3, 2, 2, hash_vector=torch.ones(3, 2))


def test_cosine_and_lc_statistics_match_definitions_and_soft_reward_has_gradient():
    orders = torch.tensor([[[0, 2, 1], [2, 1, 0], [0, 1, 2], [1, 0, 2]],
                           [[2, 2, 2], [1, 1, 2], [0, 1, 2], [2, 0, 1]]])
    expected = []
    for batch in orders.tolist():
        total = 0
        for a, b in itertools.combinations(batch, 2):
            total += sum((a[i] < a[j]) != (b[i] < b[j]) for i in range(3) for j in range(3))
        expected.append(total)
    torch.testing.assert_close(lc_exact_score(orders), torch.tensor(expected, dtype=torch.float32))
    assert not lc_exact_score(orders).requires_grad
    cate = torch.randn(2, 3, 2, 4, requires_grad=True)
    others = select_by_order(cate, orders[:, :3])
    expected_cosine = (cate[:, None] * others).sum((-1, -2)) / (
        cate[:, None].norm(dim=(-1, -2)) * others.norm(dim=(-1, -2))).clamp_min(1e-4)
    torch.testing.assert_close(cosine_coverage(cate, others), expected_cosine.mean(1))
    assert not cosine_coverage(cate, others).requires_grad
    logits = torch.randn(2, 3, 3, requires_grad=True)
    lc_soft_score(logits.softmax(-1), orders[:, :3]).mean().backward()
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["none", "cosine", "lc", "lc_soft"])
def test_modes_predictions_loss_and_legal_inference(tmp_path, mode):
    torch.manual_seed(7)
    model, batch = example(tmp_path, mode)
    terms = model.loss_components(batch)
    assert set(terms) == ({"gen.pointwise", "gen.listwise", "gen.generative", "evaluator"} |
                          ({"gen.submodularity"} if mode != "none" else set()) |
                          ({"gen.coverage"} if mode == "lc_soft" else set()))
    assert all(torch.isfinite(term) for term in terms.values())
    # predict uses greedy decoding even if the caller forgot eval().
    model.train()
    with torch.no_grad():
        outputs = model.predict(batch)
        generated, _ = model._generate(*model._inputs(batch))
    m = 3 if mode == "none" else 4
    assert generated.shape == (2, m, 3)
    torch.testing.assert_close(generated.sort(-1).values, torch.arange(3).expand(2, m, 3))
    assert outputs.scores.shape == (2, 3) and outputs.target == model.LABEL
    assert outputs.candidates.scores.shape == (2, m, 3)
    assert outputs.extra["all"].shape == (2, 6, 3)
    assert outputs.order.shape == (2, 3)
    for name, values in outputs.extra.items():
        if not name.endswith("_order"):
            assert torch.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()
    best = outputs.candidates.scores.mean(-1).argmax(-1)
    torch.testing.assert_close(outputs.order, generated[torch.arange(2), best])
    torch.testing.assert_close(outputs.candidates.orders, generated)



@pytest.mark.parametrize("mode", ["none", "cosine", "lc", "lc_soft"])
def test_generators_cannot_update_embeddings_but_share_attention(tmp_path, mode):
    torch.manual_seed(11)
    model, batch = example(tmp_path, mode)
    batch["prior"].requires_grad_()
    terms = model.loss_components(batch)
    sum(v for k, v in terms.items() if k.startswith("gen.")).backward()
    assert all(p.grad is None or p.grad.count_nonzero() == 0 for p in model.omni_embedding.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.oam.parameters())
    assert all(p.grad is None for p in model.evaluator.parameters())
    assert batch["prior"].grad is None
    if mode == "lc_soft":
        model.zero_grad(set_to_none=True)
        _, aux = model._generate(*model._inputs(batch), labels=batch["label"], label_mask=batch["mask"], teacher_forcing=True)
        aux["coverage"].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.submodularity_generator.parameters())
    model.zero_grad(set_to_none=True)
    model.loss_components(batch)["evaluator"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.omni_embedding.parameters())
    assert all(p.grad is None for p in model.generative_generator.parameters())
    assert batch["prior"].grad is not None and batch["prior"].grad.abs().sum() > 0


def test_teacher_forcing_records_may_repeat_but_greedy_is_a_permutation(tmp_path):
    model, batch = example(tmp_path)

    class FixedHead(nn.Module):
        def forward(self, candidates, state):
            return torch.arange(candidates.shape[1], device=candidates.device).float().expand(candidates.shape[:2])

    model.pointer_head = FixedHead()
    _, train = model._generate(*model._inputs(batch), labels=batch["label"], label_mask=batch["mask"], teacher_forcing=True)
    _, evaluate = model._generate(*model._inputs(batch), teacher_forcing=False)
    assert train["generative"].indices.tolist() == [[[2, 2, 2]], [[2, 2, 2]]]
    assert evaluate["generative"].indices.tolist() == [[[2, 1, 0]], [[2, 1, 0]]]
    probs = train["generative"].probabilities
    expected_loss = sum(-(batch["label"][:, t] * batch["mask"][:, t] * probs[:, t, t].clamp_min(1e-9).log()).mean() for t in range(3))
    torch.testing.assert_close(train["generative"].loss, expected_loss)


def test_chunked_ocpm_matches_individual_history_aligned_evaluation(tmp_path):
    model, batch = example(tmp_path, candidate_chunk_size=2)
    cate, history, dense, page_mask = model._inputs(batch)
    mask = model._mask(batch, cate)
    orders = model.retriever.full_permutations
    actual = model._score_candidates(cate, history, dense, page_mask, mask, orders)
    expected = []
    for b in range(2):
        scores = []
        for order in orders:
            logits, _ = model.evaluator(cate[b:b + 1, order], model.oam(history[b:b + 1])[0], dense[b:b + 1, order],
                                        page_mask[b:b + 1], model.oam, model.intra_poi, model.target_attention)
            scores.append(logits.sigmoid()[0] * mask[b, order])
        expected.append(torch.stack(scores))
    torch.testing.assert_close(actual, torch.stack(expected))
    model.candidate_chunk_size = 99
    torch.testing.assert_close(actual, model._score_candidates(cate, history, dense, page_mask, mask, orders))


@pytest.mark.parametrize("mode", ["none", "cosine", "lc", "lc_soft"])
def test_generator_dense_scores_reach_ocpm_in_candidate_coordinates(tmp_path, mode):
    model, batch = example(tmp_path, mode, candidate_dense_mode="generator", candidate_chunk_size=2)

    class DenseEvaluator(nn.Module):
        def forward(self, cate, history, dense, page_mask, *attention):
            logits = dense.squeeze(-1)
            return logits, logits.argsort(-1, descending=True)[:, None]

    model.evaluator = DenseEvaluator()
    inputs = model._inputs(batch)
    generated, aux = model._generate(*inputs, teacher_forcing=False)
    expected_dense = [select_by_order(aux["pointwise_logits"].sigmoid(), generated[:, 0:1])[:, 0],
                      select_by_order(aux["listwise_logits"].sigmoid(), generated[:, 1:2])[:, 0],
                      aux["generative"].max_scores]
    if mode != "none":
        expected_dense.append(aux["submodularity"].max_scores)
    torch.testing.assert_close(aux["candidate_dense"].squeeze(-1), torch.stack(expected_dense, 1))
    assert not aux["candidate_dense"].requires_grad
    pred = model.predict(batch)
    expected = aux["candidate_dense"].squeeze(-1).sigmoid() * select_by_order(batch["mask"], generated)
    torch.testing.assert_close(pred.candidates.scores, expected)
    best = expected.mean(-1).argmax(-1)
    torch.testing.assert_close(pred.order, generated[torch.arange(2), best])
    for family in ("lastk", "randk", "beamk", "all"):
        priors = select_by_order(batch["prior"], pred.extra[f"{family}_order"])
        torch.testing.assert_close(pred.extra[family], priors.sigmoid() * select_by_order(
            batch["mask"], pred.extra[f"{family}_order"]))


def test_custom_dense_candidate_chunking_matches_single_candidate_ocpm(tmp_path):
    model, batch = example(tmp_path, candidate_chunk_size=2)
    cate, history, dense, pages = model._inputs(batch)
    mask = model._mask(batch, cate)
    orders = model.retriever.full_permutations
    values = torch.linspace(-2, 2, 2 * 6 * 3).reshape(2, 6, 3, 1)
    actual = model._score_candidates(cate, history, dense, pages, mask, orders, candidate_dense=values)
    expected = []
    for b in range(2):
        scores = []
        for m, order in enumerate(orders):
            logits, _ = model.evaluator(cate[b:b + 1, order], model.oam(history[b:b + 1])[0], values[b:b + 1, m],
                                        pages[b:b + 1], model.oam, model.intra_poi, model.target_attention)
            scores.append(logits.sigmoid()[0] * mask[b, order])
        expected.append(torch.stack(scores))
    torch.testing.assert_close(actual, torch.stack(expected))
    model.candidate_chunk_size = 99
    torch.testing.assert_close(actual, model._score_candidates(
        cate, history, dense, pages, mask, orders, candidate_dense=values))
    with pytest.raises(ValueError, match="Candidate dense"):
        model._score_candidates(cate, history, dense, pages, mask, orders, candidate_dense=values[:, :-1])


def test_lc_mge_factory_presets_and_explicit_overrides(tmp_path):
    baseline, batch = example(tmp_path)
    options = dict(slate_dense_field="prior", slate_cate_group="slate", hist_cate_group="history",
                   poi_num=3, page_num=2, eval_all_perms=False)
    model = build_rerank_model(baseline.manager, "LC_MGE", **options)
    assert model.submodularity_mode == "lc"
    assert model.candidate_dense_mode == "generator"
    assert model.loss_reduction == "nonzero_mean"
    assert torch.isfinite(model.calculate_loss(batch))
    with torch.no_grad():
        assert model.predict(batch).order.shape == (2, 3)
    old = build_rerank_model(baseline.manager, "mge", **options)
    assert old.candidate_dense_mode == "prior" and old.loss_reduction == "mean"
    override = build_rerank_model(baseline.manager, "lc_mge", candidate_dense_mode="prior", **options)
    assert override.candidate_dense_mode == "prior"
    with pytest.raises(ValueError, match="candidate_dense_mode"):
        build_rerank_model(baseline.manager, "lc_mge", candidate_dense_mode="unknown", **options)


def test_common_initialization_constants_factory_and_label_free_prediction(tmp_path):
    torch.manual_seed(19)
    baseline, batch = example(tmp_path, "none")
    torch.manual_seed(19)
    enhanced, _ = example(tmp_path, "lc", eval_all_perms=False, share_pointer_head=False)
    for name, tensor in baseline.state_dict().items():
        torch.testing.assert_close(tensor, enhanced.state_dict()[name], rtol=0, atol=0)
    assert "retriever.full_permutations" in dict(enhanced.named_buffers())
    assert not any("oam" in name for name, _ in enhanced.generative_generator.named_parameters())
    params = list(enhanced.parameters())
    assert len(params) == len({id(p) for p in params})
    unlabeled = Interaction({k: batch[k] for k in batch.columns if k != "label"})
    with torch.no_grad():
        assert "all" not in enhanced.predict(unlabeled).extra
    with pytest.warns(UserWarning, match="unsupported"):
        rebuilt = build_rerank_model(baseline.manager, "MGE", unsupported=True,
                                    slate_dense_field="prior", slate_cate_group="slate", hist_cate_group="history",
                                    poi_num=3, page_num=2)
    assert isinstance(rebuilt, MultiGERerankModel)
    with pytest.raises(ValueError, match="Unknown"):
        build_rerank_model(baseline.manager, "missing")


@pytest.mark.parametrize("trainer_class", [BaseTrainer, EMATrainer])
def test_training_loss_ledger_evaluation_early_stop_and_checkpoint(tmp_path, trainer_class):
    torch.manual_seed(23)
    model, batch = example(tmp_path, "lc_soft", k=5, p=2, eval_all_perms=False,
                           use_contrastive_loss=True, contrastive_weight=.1)
    eval_manager = EvaluatorManager(log_path=tmp_path / "metrics.log")
    eval_manager.register("point", PointwiseEvaluator("auc", "logloss"))
    eval_manager.register("slate", RankingEvaluator("ndcg@5", "pcoc"), resolver=BatchResolver(view="score_only"))
    eval_manager.register("order", RankingEvaluator("ndcg@5"))
    loader = DataLoader([batch], batch_size=None)
    options = {"ema_config": EMAConfig(decays=(.5,), eval_student=True)} if trainer_class is EMATrainer else {}
    trainer = trainer_class(
        model, torch.optim.Adam(model.parameters(), lr=0.), model.manager,
        TrainerDataLoaders(loader, loader), TrainerComponents(evaluator_manager=eval_manager, early_stopper=EarlyStopper(patience=1)),
        SimpleNamespace(device="cpu", max_epochs=5, train_log_interval=100, ckpt_dir=str(tmp_path / "ckpt"),
                        experiment_name="mge", dataset_name="synthetic"), **options,
    )
    result = trainer.run()
    assert trainer.epoch == 2
    assert set(result["point"]) == {"auc", "logloss"}
    assert set(result["slate"]) == {"ndcg@5", "pcoc"}
    ledger = model.lsp.summary()["settlements"]["main"]
    assert set(ledger["terms"]) == {"gen.pointwise", "gen.listwise", "gen.generative", "gen.submodularity", "gen.coverage", "evaluator", "contrastive"}
    assert ledger["total"] == pytest.approx(sum(term["weighted"] for term in ledger["terms"].values()), abs=1e-5)
    # Zero LR above makes early stopping deterministic. Exercise a real update
    # separately so the integration also checks optimizer and EMA transactions.
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    for group in trainer.optimizer.param_groups:
        group["lr"] = 1e-3
    trainer.train_epoch()
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    with torch.no_grad():
        expected = model.predict(batch)
    path = trainer.save_checkpoint("roundtrip", metrics=result)
    with torch.no_grad():
        next(model.parameters()).add_(1)
        model.retriever.position_encoding.zero_()
        if trainer_class is EMATrainer:
            next(trainer.ema.teachers[.5].parameters()).add_(2)
    trainer.load_checkpoint(path)
    with torch.no_grad():
        restored = model.predict(batch)
    torch.testing.assert_close(restored.scores, expected.scores, rtol=0, atol=0)
    torch.testing.assert_close(restored.order, expected.order, rtol=0, atol=0)
    for key in expected.extra:
        torch.testing.assert_close(restored.extra[key], expected.extra[key], rtol=0, atol=0)
    if trainer_class is EMATrainer:
        assert set(trainer.last_eval_metrics_by_checkpoint) == {"student", "ema0p5"}
        teacher = trainer.ema.teachers[.5]
        assert teacher.slate_view.omni is teacher.omni_embedding
