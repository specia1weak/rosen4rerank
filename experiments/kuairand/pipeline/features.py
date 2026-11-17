"""严格早于查询日期的历史统计，供请求级特征处理复用。"""
import polars as pl

def past_daily(query, observations, keys, values):
    """Cumulative sums/count through the previous day; current day excluded."""
    by = keys + ['date']
    daily = observations.group_by(by).agg(pl.len().cast(pl.Float64).alias('prior_n'), *[pl.col(v).sum().alias('sum_' + v) for v in values]).sort(by)
    sums = ['prior_n'] + ['sum_' + v for v in values]
    if keys:
        daily = daily.with_columns(*[pl.col(c).cum_sum().over(keys).alias(c) for c in sums])
    else:
        daily = daily.with_columns(*[pl.col(c).cum_sum().alias(c) for c in sums])
    return query.sort('date').join_asof(daily.sort('date'), on='date', by=keys or None, strategy='backward', allow_exact_matches=False, check_sortedness=False).with_columns(pl.col(sums).fill_null(0))
