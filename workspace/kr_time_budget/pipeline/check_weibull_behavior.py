"""Plan19只读核对：真实停止屏数分布、经验拟合与模型预算/继续曲线。

不重训、不修改旧产物。主结果epoch5，epoch1/3作敏感性；有UID/置0均覆盖。
预算参数未存在旧预测中，仅重载checkpoint，对每场首屏运行预算网提取参数；
继续概率直接复用既有预测。全场群体曲线平均S，不把平均参数代回公式。
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import spearmanr
import torch

from workspace.kr_time_budget.pipeline.exit_model import KRExitModel, weibull_parameters
from workspace.kr_time_budget.pipeline.exit_training import WORKSPACE, EXPERIMENTS, load_batches
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager, BUDGET_PARAMETER_FIELDS


MODELS = ['single_step', 'single_step_pace', 'mixture_step', 'mixture_step_pace']


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def empirical_curve(lengths):
    """r=0为首屏之前；S(r)=P(L>r)，第t屏继续率为S(t)/S(t-1)。"""
    lengths = np.asarray(lengths, dtype=int)
    if len(lengths) == 0 or np.any(lengths < 1):
        raise ValueError('Positive complete session screen counts required')
    r = np.arange(int(lengths.max())+1)
    survivor = (lengths[:, None] > r).sum(axis=0)
    risk = (lengths[:, None] >= r).sum(axis=0)
    conditional = np.divide(survivor, risk, out=np.ones(len(r), dtype=float), where=risk > 0)
    return pl.DataFrame(dict(r=r, survivor=survivor, at_risk=risk,
                             survival=survivor/len(lengths), continuation=conditional))


def log_survival(r, scale, shape, prior=None):
    """[sessions, progress]，single也统一为一分量；双精度logsumexp。"""
    scale, shape = np.atleast_2d(scale), np.atleast_2d(shape)
    if prior is None:
        prior = np.ones_like(scale)/scale.shape[1]
    hazard = (np.asarray(r)[None, :, None]/scale[:, None, :])**shape[:, None, :]
    return logsumexp(np.log(prior)[:, None, :]-hazard, axis=-1)


def fit_weibull(lengths, mixture=False):
    """离散停止概率P(L=l)=S(l-1)-S(l)的MLE，不把整数长度当连续密度。"""
    lengths, counts = np.unique(np.asarray(lengths, dtype=float), return_counts=True)
    weights = counts/counts.sum()
    def decode(x):
        if not mixture:
            scale,shape=np.exp(x)
            return np.array([scale]), np.array([shape]), np.array([1.])
        scale,shape=np.exp(x[:2]),np.exp(x[2:4])
        p = 1/(1+np.exp(-x[4]))
        return scale, shape, np.array([p, 1-p])
    def objective(x):
        scale, shape, prior = decode(x)
        a = log_survival(lengths-1, scale[None, :], shape[None, :], prior[None, :])[0]
        b = log_survival(lengths, scale[None, :], shape[None, :], prior[None, :])[0]
        increment = np.minimum(b-a, -1e-15)
        return float(-np.dot(weights, a+np.log(-np.expm1(increment))))
    if mixture:
        starts = [np.r_[np.log([3., 25.]), np.log([a,b]), gate]
                  for a,b,gate in [(.6,.8,0.),(1.,.5,0.),(.8,1.,1.),(.4,.4,-1.)]]
        bounds = [(-8,15)]*2+[(-5,3)]*2+[(-8,8)]
    else:
        center = float(np.dot(lengths, weights))
        starts = [np.log([max(center,.1), z]) for z in [.5,1.,1.5]]
        bounds = [(-8,15),(-5,3)]
    results = [minimize(objective, x, method='L-BFGS-B', bounds=bounds) for x in starts]
    successful = [res for res in results if res.success and np.isfinite(res.fun)]
    if not successful:
        raise RuntimeError('Empirical Weibull fit did not converge')
    result = min(successful, key=lambda res: res.fun)
    scale, shape, prior = decode(result.x)
    order = np.argsort(scale)
    return dict(scale=scale[order].tolist(), shape=shape[order].tolist(), prior=prior[order].tolist(),
        negative_log_likelihood=float(result.fun), converged=True, sessions=int(counts.sum()),
        optimizer='L-BFGS-B, deterministic multistart', parameters=5 if mixture else 2)


def curve_error(empirical, predicted):
    empirical=np.asarray(empirical,dtype=float)
    residual = np.asarray(predicted)-empirical
    variance = float(np.square(empirical-np.mean(empirical)).sum())
    return dict(rmse=float(np.sqrt(np.mean(residual**2))), max_abs=float(abs(residual).max()),
                r2=float(1-np.square(residual).sum()/variance) if variance else None)


def deciles(values):
    """相同值不强行拆桶；固定分位点，不以结果标签选桶。"""
    cuts = np.unique(np.quantile(values, np.arange(1,10)/10))
    return np.searchsorted(cuts, values, side='right')+1


def effective_scale(scale, shape, prior):
    """数值求每场S(t)=exp(-1)的t，混合分布不伪造单一lambda。"""
    low, high = np.zeros(len(scale)), np.full(len(scale), 1e6)
    for _ in range(70):
        middle = (low+high)/2
        ls = logsumexp(np.log(prior)-(middle[:,None]/scale)**shape, axis=1)
        low = np.where(ls > -1, middle, low)
        high = np.where(ls <= -1, middle, high)
    return (low+high)/2


def risk_curve(predictions, sessions):
    """仅在真实仍在场的risk set平均各自概率，然后连乘。非候选策略全轨迹预测。"""
    selected = predictions.join(sessions.select('session_id'), on='session_id', how='semi')
    expressions = [pl.len().alias('at_risk'), pl.col('y_continue').mean().alias('observed_continue'),
                   pl.col('prediction').mean().alias('mean_prediction')]
    if 'gamma' in selected.columns:
        expressions += [pl.col('gamma').mean().alias('mean_gamma'),
                        pl.col('content_stay').mean().alias('mean_stay')]
    rows = selected.group_by('req_idx_raw').agg(expressions).sort('req_idx_raw')
    assert rows['req_idx_raw'].to_list() == list(range(1, int(sessions['L'].max())+1))
    return rows.with_columns(pl.col('mean_prediction').cum_prod().alias('risk_standardized_survival'),
                             pl.col('observed_continue').cum_prod().alias('empirical_survival_product'))


@torch.no_grad()
def extract_parameters(checkpoint, options, batches, manager, zero_uid):
    model = KRExitModel(manager, **options).eval()
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    rows = []
    for batch in batches:
        take = batch['req_idx_raw'] == 1
        if not take.any():
            continue
        b = {field: value[take] for field, value in batch.interaction.items()}
        if zero_uid:
            b['user_id'] = torch.zeros_like(b['user_id'])
        raw = model.budget_raw(b)
        if options['mixture']:
            scale, shape = weibull_parameters(raw[:,6:10].reshape(-1,2), True)
            scale, shape = scale.reshape(-1,2), shape.reshape(-1,2)
            prior = torch.softmax(raw[:,10:12].double(), dim=-1)
        else:
            scale, shape = weibull_parameters(raw[:,2:4], True)
            scale, shape = scale[:,None], shape[:,None]
            prior = torch.ones_like(scale).double()
        for i, session in enumerate(b['session_id'].tolist()):
            order = torch.argsort(scale[i])
            row = dict(session_id=session)
            for k, index in enumerate(order.tolist()):
                row.update({f'scale{k}':float(scale[i,index]), f'shape{k}':float(shape[i,index]),
                            f'prior{k}':float(prior[i,index])})
            rows.append(row)
    return pl.DataFrame(rows).sort('session_id')


def parameter_arrays(frame, mixture):
    k = 2 if mixture else 1
    return [frame.select([f'{name}{i}' for i in range(k)]).to_numpy() for name in ['scale','shape','prior']]


def make_sessions(test):
    ordered = test.sort('session_id','req_idx_raw')
    sessions = ordered.group_by('session_id', maintain_order=True).agg(
        pl.col('req_idx_raw').max().alias('L'), pl.len().alias('n'),
        pl.col('y_continue').sum().alias('continues'),
        pl.col('user_id_raw').first(), pl.col('tab_raw').first(),
        pl.col('user_hist_req_cnt').first().alias('history_req_log1p'))
    assert sessions.select((pl.col('L')==pl.col('n')).all()).item()
    assert sessions.select((pl.col('continues')==pl.col('L')-1).all()).item()
    assert ordered.filter(pl.col('req_idx_raw')==1).height == sessions.height
    end = ordered.group_by('session_id').agg(pl.col('y_continue').last().alias('terminal'))
    assert end['terminal'].sum() == 0
    # 静态预算字段逐场核对，允许从首屏读取一次后外推预算曲线。
    unique = ordered.group_by('session_id').agg(*[pl.col(f).n_unique().alias(f) for f in BUDGET_PARAMETER_FIELDS])
    assert unique.select(pl.all().exclude('session_id').max()).row(0) == (1,)*len(BUDGET_PARAMETER_FIELDS)
    return sessions.with_columns(pl.Series('history_decile', deciles(sessions['history_req_log1p'].to_numpy())))


def analyze_group(sessions, params, prediction, grouping, group, metadata):
    lengths = sessions['L'].to_numpy()
    emp = empirical_curve(lengths)
    fit = fit_weibull(lengths)
    scale, shape, prior = parameter_arrays(params, metadata['model'].startswith('mixture'))
    r = emp['r'].to_numpy()
    s_fit = np.exp(log_survival(r,np.array(fit['scale'])[None,:],np.array(fit['shape'])[None,:]))[0]
    s_model = np.exp(log_survival(r,scale,shape,prior)).mean(axis=0)
    conditional = risk_curve(prediction,sessions)
    # 首屏前S=1；经验S(r)对应观察过第r屏后的继续概率连乘。
    s_product = np.r_[1., conditional['risk_standardized_survival'].to_numpy()]
    np.testing.assert_allclose(emp['survival'].to_numpy()[1:],
                               conditional['empirical_survival_product'].to_numpy(),atol=1e-14)
    curve = emp.with_columns(pl.Series('empirical_fit_survival',s_fit),
        pl.Series('mean_budget_survival',s_model), pl.Series('observed_risk_product_survival',s_product),
        pl.lit(grouping).alias('grouping'),pl.lit(str(group)).alias('group'))
    row = dict(**metadata, grouping=grouping, group=str(group), sessions=len(lengths),
        users=sessions['user_id_raw'].n_unique(), length_mean=float(lengths.mean()),
        empirical_scale=fit['scale'][0],empirical_shape=fit['shape'][0],
        model_effective_scale_mean=float(effective_scale(scale,shape,prior).mean()),
        model_shape_component0_mean=float(shape[:,0].mean()),
        empirical_fit_rmse50=curve_error(emp['survival'][:51],s_fit[:51])['rmse'],
        empirical_fit_r2_50=curve_error(emp['survival'][:51],s_fit[:51])['r2'],
        budget_rmse50=curve_error(emp['survival'][:51],s_model[:51])['rmse'],
        product_rmse50=curve_error(emp['survival'][:51],s_product[:51])['rmse'])
    curves = curve.with_columns(*[pl.lit(value).alias(key) for key,value in metadata.items()])
    risk = conditional.with_columns(pl.lit(grouping).alias('grouping'),pl.lit(str(group)).alias('group'),
                                     *[pl.lit(value).alias(key) for key,value in metadata.items()])
    return row,curves,risk


def run(output):
    torch.set_num_threads(4)
    data = WORKSPACE/'data/v3'
    source_paths = {data/'test.parquet',data/'manifest.json',Path(__file__).resolve()}
    test = pl.read_parquet(data/'test.parquet')
    sessions = make_sessions(test)
    assert test.height==48898 and sessions.height==3281
    sessions.write_parquet(output/'sessions.parquet')
    empirical = empirical_curve(sessions['L'].to_numpy())
    empirical.write_parquet(output/'empirical_global.parquet')
    fits = {'discrete_single_mle':fit_weibull(sessions['L'].to_numpy()),
            'discrete_mixture_mle':fit_weibull(sessions['L'].to_numpy(),mixture=True),'double_log_ols':{}}
    for end in [10,50,100,int(sessions['L'].max())-1]:
        selected=empirical.filter((pl.col('r')>=1)&(pl.col('r')<=end)&
                                  (pl.col('survival')>0)&(pl.col('survival')<1))
        x=np.log(selected['r'].to_numpy()); y=np.log(-np.log(selected['survival'].to_numpy()))
        slope,intercept=np.polyfit(x,y,1)
        fits['double_log_ols'][str(end)] = dict(scale=float(np.exp(-intercept/slope)),shape=float(slope),
                                                points=len(x),**curve_error(y,slope*x+intercept))
    for name in ['discrete_single_mle','discrete_mixture_mle']:
        fitted=fits[name]
        curve=np.exp(log_survival(empirical['r'].to_numpy(),np.array(fitted['scale'])[None,:],
            np.array(fitted['shape'])[None,:],np.array(fitted['prior'])[None,:]))[0]
        fitted.update(in_sample_errors50=curve_error(empirical['survival'][:51],curve[:51]),
                      in_sample_errors_full=curve_error(empirical['survival'],curve))
    save_json(output/'empirical_fits.json',fits)
    print('EMPIRICAL',json.dumps(fits),flush=True)
    manager=request_manager(data/'schema',load=True)
    # 旧预测没有lambda/shape，读取已有test输入；不加载train、不执行训练。
    batches=load_batches(data,manager,{'train':1,'valid':1})['test']
    records, curves, risks, correlations, parameters, extraction = [],[],[],[],[],[]
    input_hashes={}
    max_log_error=0.
    def track(path):
        if str(path) not in input_hashes:
            input_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in source_paths:track(path)
    for state,run_name in [('uid','epochwise5'),('uid_zero','epochwise5_uid_zero')]:
        root=WORKSPACE/'runs'/run_name
        track(root/'config.json')
        for seed in [2026,2027,2028]:
            for epoch in [1,3,5]:
                for name in MODELS:
                    metadata=dict(uid=state,seed=seed,epoch=epoch,model=name)
                    folder=root/f'{seed}/{name}/epoch_{epoch:02d}'
                    checkpoint=folder/'checkpoint.pt'; predpath=folder/'test_predictions.parquet'
                    track(checkpoint);track(predpath)
                    pred=pl.read_parquet(predpath)
                    for col in ['request_id','session_id','user_id_raw','req_idx_raw','y_continue']:
                        assert pred[col].to_list()==test[col].to_list()
                    param=extract_parameters(checkpoint,EXPERIMENTS[name],batches,manager,state=='uid_zero')
                    joined=sessions.join(param,on='session_id',validate='1:1').sort('session_id')
                    scale,shape,prior=parameter_arrays(joined,name.startswith('mixture'))
                    model_decile=deciles(effective_scale(scale,shape,prior))
                    joined=joined.with_columns(pl.Series('model_scale_decile',model_decile))
                    parameters.append(joined.with_columns(*[pl.lit(v).alias(k) for k,v in metadata.items()]))
                    # 用独立NumPy公式核对每条已有预算logS，而非只看首屏S=1。
                    lookup=pred.select('session_id','req_idx_raw','log_survival').join(param,on='session_id',validate='m:1')
                    a,b,p=parameter_arrays(lookup,name.startswith('mixture'))
                    t=lookup['req_idx_raw'].to_numpy()-1
                    recovered=logsumexp(np.log(p)-(t[:,None]/a)**b,axis=1)
                    error=float(abs(recovered-lookup['log_survival'].to_numpy()).max())
                    assert error < 1e-8,(metadata,error)
                    max_log_error=max(max_log_error,error)
                    extraction.append(dict(**metadata,max_log_s_error=error,checkpoint=str(checkpoint)))
                    group_specs=[('global',joined)]
                    if epoch==5:
                        for grouping in ['history_decile','model_scale_decile','tab_raw']:
                            for group in joined[grouping].unique().sort().to_list():
                                group_specs.append((grouping,joined.filter(pl.col(grouping)==group)))
                    for grouping,groupframe in group_specs:
                        group='all' if grouping=='global' else groupframe[grouping][0]
                        row,curve,risk=analyze_group(groupframe,groupframe,pred,grouping,group,metadata)
                        records.append(row);curves.append(curve);risks.append(risk)
                    if epoch==5:
                        for grouping in ['history_decile','model_scale_decile','tab_raw']:
                            grouped=[row for row in records if all(row[k]==v for k,v in metadata.items()) and row['grouping']==grouping]
                            for target in ['empirical_scale','length_mean','empirical_shape']:
                                x=[row['model_effective_scale_mean'] for row in grouped]; y=[row[target] for row in grouped]
                                rho=float(spearmanr(x,y).statistic) if len(set(x))>1 and len(set(y))>1 else None
                                correlations.append(dict(**metadata,grouping=grouping,target=target,spearman=rho,buckets=len(grouped)))
                    print('CHECKED',metadata,'logS_error',error,flush=True)
                # Fusion曲线仅复用预测，供继续概率risk-set比较，不拟造预算分布。
                folder=root/f'{seed}/fusion/epoch_{epoch:02d}';path=folder/'test_predictions.parquet'
                track(path)
                fusion=pl.read_parquet(path)
                risk=risk_curve(fusion,sessions).with_columns(pl.lit('global').alias('grouping'),pl.lit('all').alias('group'),
                    pl.lit(state).alias('uid'),pl.lit(seed).alias('seed'),pl.lit(epoch).alias('epoch'),pl.lit('fusion').alias('model'))
                risks.append(risk)
    pl.DataFrame(records).write_parquet(output/'group_fits.parquet')
    pl.concat(curves,how='diagonal_relaxed').write_parquet(output/'curves.parquet')
    pl.concat(risks,how='diagonal_relaxed').write_parquet(output/'risk_curves.parquet')
    pl.concat(parameters,how='diagonal_relaxed').write_parquet(output/'model_session_parameters.parquet')
    save_json(output/'group_correlations.json',correlations)
    for filename,digest in input_hashes.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,filename
    save_json(output/'manifest.json',dict(sessions=sessions.height,requests=test.height,
        source_hashes=input_hashes,source_files_unchanged=True,models_extracted=len(extraction),
        parameter_readout='budget-only forward on first request; no optimization; original prediction reused',
        extraction_checks=extraction,maximum_independent_log_survival_error=max_log_error,
        primary='epoch5 both UID variants; single/mixture with/without pace, 3 seeds',
        sensitivity_epochs=[1,3],groupings=['history_decile','model_scale_decile','tab_raw'],
        censoring='selected complete reconstructed sessions; no claim of independent random censoring',
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('COMPLETE',output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=WORKSPACE/'runs/plan19_weibull_check')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'logs').mkdir()
    print('只读核对日志：',args.output/'logs/check.log',flush=True)
    with (args.output/'logs/check.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):
        run(args.output)
    print('完成：',args.output/'manifest.json',flush=True)


if __name__=='__main__':main()
