from workspace.kr_time_budget.pipeline.kr_request_schema import (
    FEATURES, FEATURE_CATEGORIES, SHARED_FIELDS, BUDGET_EXCLUSIVE_FIELDS,
    CONTENT_EXCLUSIVE_FIELDS, BUDGET_PARAMETER_FIELDS, BUDGET_PRIOR_FIELDS,
    CONTENT_TOWER_FIELDS, CONTENT_MASK_FIELDS,
)


def test_feature_partition_and_legacy_semantics():
    groups = [set(fields) for fields in FEATURE_CATEGORIES.values()]
    assert [len(g) for g in groups] == [30, 6, 18]
    assert all(not left & right for i, left in enumerate(groups) for right in groups[i+1:])
    assert set.union(*groups) == set(FEATURES)
    assert set(BUDGET_PARAMETER_FIELDS) == set(SHARED_FIELDS + BUDGET_PRIOR_FIELDS)
    assert len(BUDGET_PARAMETER_FIELDS) == 34 and len(CONTENT_TOWER_FIELDS) == 48
    assert set(CONTENT_TOWER_FIELDS) == set(SHARED_FIELDS + CONTENT_EXCLUSIVE_FIELDS)
    assert not set(CONTENT_TOWER_FIELDS) & set(BUDGET_EXCLUSIVE_FIELDS)
    assert CONTENT_MASK_FIELDS == frozenset(BUDGET_EXCLUSIVE_FIELDS)
    assert len(CONTENT_EXCLUSIVE_FIELDS) == 18  # 历史Content-only不能悄然变成48字段
