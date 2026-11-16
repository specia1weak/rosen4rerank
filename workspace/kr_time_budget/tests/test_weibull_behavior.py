"""经验停止屏数与预算曲线的单位、索引、混合和分桶语义。"""
import numpy as np
import polars as pl
import pytest

from workspace.kr_time_budget.pipeline.check_weibull_behavior import (
    empirical_curve, log_survival, fit_weibull, effective_scale, deciles, risk_curve, curve_error,
)


def test_empirical_survival_and_request_continuation_have_one_step_offset():
    lengths=np.array([1,2,3])
    result=empirical_curve(lengths)
    np.testing.assert_allclose(result['survival'],[1,2/3,1/3,0])
    np.testing.assert_allclose(result['continuation'],[1,2/3,1/2,0])
    prediction=pl.DataFrame(dict(session_id=[1,2,2,3,3,3],req_idx_raw=[1,1,2,1,2,3],
        y_continue=[0,1,0,1,1,0],prediction=[.7,.6,.5,.7,.5,.2]))
    sessions=pl.DataFrame(dict(session_id=[1,2,3],L=lengths))
    risk=risk_curve(prediction,sessions)
    np.testing.assert_allclose(risk['empirical_survival_product'],result['survival'][1:])
    # 已退出session没有后续特征，风险集均值连乘不伪称每场完整反事实预测。
    np.testing.assert_allclose(risk['risk_standardized_survival'],[2/3,1/3,1/15])
    with pytest.raises(ValueError):empirical_curve([0,1])


def test_discrete_mle_recovers_quantile_sample_without_continuous_density_bias():
    n=20000;scale=12.;shape=.72
    u=(np.arange(n)+.5)/n
    lengths=np.ceil(scale*(-np.log1p(-u))**(1/shape)).astype(int)
    fit=fit_weibull(lengths)
    np.testing.assert_allclose(fit['scale'],[scale],rtol=.005)
    np.testing.assert_allclose(fit['shape'],[shape],rtol=.005)
    grid=np.arange(10000)
    s=np.exp(log_survival(grid,np.array([[scale]]),np.array([[shape]])))[0]
    assert abs(np.diff(-s).sum()-1)<1e-10


def test_population_averages_survival_not_parameters_and_mixture_equivalent_scale():
    scale=np.array([[3.],[30.]]);shape=np.array([[.5],[1.5]])
    grid=np.arange(1,11)
    mean_curve=np.exp(log_survival(grid,scale,shape)).mean(axis=0)
    from_mean=np.exp(log_survival(grid,scale.mean(axis=0)[None,:],shape.mean(axis=0)[None,:]))[0]
    assert not np.allclose(mean_curve,from_mean)
    scale=np.array([[3.,30.]]);shape=np.array([[.5,1.5]]);prior=np.array([[.4,.6]])
    equivalent=effective_scale(scale,shape,prior)[0]
    assert 3<equivalent<30
    assert abs(log_survival([equivalent],scale,shape,prior)[0,0]+1)<1e-10


def test_constant_content_stay_adds_linear_hazard_not_fixed_scale_multiplier():
    r=np.arange(1,30);scale=12.;shape=.72;q=.95
    budget=(r/scale)**shape
    total=budget-r*np.log(q)
    ratio=total/budget
    assert np.ptp(ratio)>.1
    # 仅z=1时，线性风险相加仍是同shape的指数/Weibull。
    total_exponential=r/scale-r*np.log(q)
    effective=1/(1/scale-np.log(q))
    np.testing.assert_allclose(total_exponential,r/effective)


def test_quantile_buckets_do_not_split_equal_values():
    values=np.repeat([1.,2.,3.,4.],[20,10,30,40])
    labels=deciles(values)
    for value in np.unique(values):assert len(np.unique(labels[values==value]))==1
    assert len(set(labels))<=10


def test_curve_metrics_accept_polars_series_and_constant_target():
    result=curve_error(pl.Series([1.,.5,0.]),np.array([1.,.5,0.]))
    assert result==dict(rmse=0.,max_abs=0.,r2=1.)
    assert curve_error(pl.Series([1.,1.]),[1.,1.])['r2'] is None
