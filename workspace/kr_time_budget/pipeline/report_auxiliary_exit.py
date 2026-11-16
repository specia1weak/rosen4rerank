"""Plan21只读复核：逐epoch指标、离散NLL、预算/联合曲线与配对报告。"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.special import logsumexp
import torch

from workspace.kr_time_budget.pipeline.exit_training import WORKSPACE, EXPERIMENTS, load_batches, predict_batches
from workspace.kr_time_budget.pipeline.kr_request_schema import request_manager
from workspace.kr_time_budget.pipeline.check_weibull_behavior import (
    make_sessions, empirical_curve, extract_parameters, parameter_arrays, log_survival,
    effective_scale, curve_error, risk_curve, save_json,
)
from workspace.kr_time_budget.pipeline.report_epochwise_exit import compact, metric_difference
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores


VARIANTS = ['baseline', 'left', 'right', 'both']
COLORS = dict(zip(VARIANTS, ['#555555', '#e68613', '#3878b8', '#c13b4c']))
LABELS = dict(baseline='Joint only', left='+ Left NLL', right='+ Right BCE', both='+ Both')
METRICS = ['auc','logloss','user_auc','intra_step_gauc','session_nll','budget_rmse50','product_rmse50']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_log_survival(r, scale, shape, prior):
    return logsumexp(np.log(prior)-(r[:, None]/scale)**shape, axis=-1)


def analyze(run):
    torch.set_num_threads(4)
    config = json.loads((run/'config.json').read_text())
    baseline = Path(config['baseline'])
    assert digest(baseline/'config.json') == config['baseline_config_sha256']
    baseline_config = json.loads((baseline/'config.json').read_text())
    for key in ['epochs','batch_size','gradient_accumulation','budget_lr','content_or_fusion_lr',
                'weight_decay','zero_decay','train_order','input_ablation','dataset_manifest_sha256']:
        assert config[key] == baseline_config[key], key
    for p, h in config['source_hashes'].items():
        assert digest(Path(p)) == h, p
    summary = json.loads((run/'summary.json').read_text())
    reference = json.loads((baseline/'summary.json').read_text())
    expected_keys = {f'{s}/{m}/{v}' for s in config['seeds'] for m in config['experiments'] for v in config['variants']}
    assert set(summary) == expected_keys
    assert json.loads((run/'complete.json').read_text())['trajectories'] == len(summary)
    data = WORKSPACE/'data/v3'
    assert digest(data/'manifest.json') == config['dataset_manifest_sha256']
    expected = {s: pl.read_parquet(data/f'{s}.parquet') for s in ['valid','test']}
    sessions = make_sessions(expected['test']).sort('session_id')
    empirical = empirical_curve(sessions['L'].to_numpy())
    manager = request_manager(data/'schema', load=True)
    test_batches = load_batches(data, manager, {'train':1,'valid':1})['test']
    coverage = json.loads((run/'coverage.json').read_text())
    hashes = {p: h for p, h in config['source_hashes'].items()}
    hashes[str(run/'config.json')] = digest(run/'config.json')
    hashes[str(baseline/'summary.json')] = digest(baseline/'summary.json')
    hashes[str(data/'test.parquet')] = digest(data/'test.parquet')
    # 旧实验的180份预测也确认未被本次训练覆盖。
    old_audit = json.loads((baseline/'audit.json').read_text())
    for p, h in old_audit['prediction_sha256'].items():
        assert digest(baseline/p) == h, p
    rows, curve_rows, parameter_rows, loss_rows = [], [], [], []
    reloads = []
    max_metric_error, max_log_s_error, checkpoints, evaluations = 0., 0., 0, 0
    for seed in config['seeds']:
        for model_name in config['experiments']:
            for variant in VARIANTS:
                if variant != 'baseline' and variant not in config['variants']:
                    continue
                key = f'{seed}/{model_name}' if variant == 'baseline' else f'{seed}/{model_name}/{variant}'
                source = baseline if variant == 'baseline' else run
                records = reference[key] if variant == 'baseline' else summary[key]
                init = json.loads((source/key/'initialization.json').read_text())
                old_init = json.loads((baseline/f'{seed}/{model_name}/initialization.json').read_text())
                assert init['state_sha256'] == old_init['state_sha256']
                assert init['optimizer_groups'] == old_init['optimizer_groups']
                assert [r['epoch'] for r in records] == [1,2,3,4,5]
                for record in records:
                    epoch = record['epoch']
                    metadata = dict(seed=seed, model=model_name, variant=variant, epoch=epoch)
                    directory = source/key/f'epoch_{epoch:02d}'
                    checkpoint_path = directory/'checkpoint.pt'
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    state = checkpoint['model_state_dict']
                    h = hashlib.sha256()
                    for name, tensor in sorted(state.items()):
                        h.update(name.encode()); h.update(tensor.numpy().tobytes())
                    assert h.hexdigest() == checkpoint['state_sha256'] == record['state_sha256']
                    assert checkpoint['global_step'] == record['step'] == 421*epoch
                    assert {int(s['step']) for s in checkpoint['optimizer_state_dict']['state'].values()} == {421*epoch}
                    assert record['evaluation_preserved_model_optimizer_rng']
                    for tower, field in [('content_or_fusion','omni_embedding'),('budget','budget_embedding')]:
                        tensor = state[f'{field}.emb_modules.user_id.embedding.weight']
                        assert torch.count_nonzero(tensor[0]) == 0
                        assert hashlib.sha256(tensor.numpy().tobytes()).hexdigest() == init['uid_table_sha256'][tower]
                    hashes[str(checkpoint_path)] = digest(checkpoint_path)
                    checkpoints += 1
                    metrics = {}
                    for split in ['valid','test']:
                        path = directory/f'{split}_predictions.parquet'
                        prediction = pl.read_parquet(path)
                        for col in ['request_id','session_id','user_id_raw','req_idx_raw','y_continue']:
                            assert prediction[col].equals(expected[split][col])
                        assert prediction.height == prediction['request_id'].n_unique() == coverage[split]['rows']
                        assert (prediction['gamma_time'] == 1).all()
                        assert (prediction['prediction']-prediction['gamma']*prediction['content_stay']).abs().max() == 0
                        computed = scores(prediction)
                        max_metric_error = max(max_metric_error, metric_difference(computed, record[split]))
                        metrics[split] = compact(computed)
                        hashes[str(path)] = digest(path)
                        evaluations += 1
                    if seed == config['seeds'][0] and epoch == 5 and variant == 'both':
                        from workspace.kr_time_budget.pipeline.exit_model import KRExitModel
                        from workspace.kr_time_budget.pipeline.run_uid_zero import zero_uid_inputs
                        reloaded = KRExitModel(manager, **EXPERIMENTS[model_name])
                        reloaded.load_state_dict(state, strict=True)
                        recomputed = predict_batches(reloaded, zero_uid_inputs({'test':test_batches})['test'])
                        assert prediction.equals(recomputed), (model_name, 'full test reload')
                        reloads.append(dict(model=model_name, seed=seed, epoch=epoch,variant=variant,
                                            full_test_all_columns_exact=True,rows=prediction.height))
                    # 此时prediction为完整test，参数仅由每场首屏预算输入读取。
                    params = extract_parameters(checkpoint_path, EXPERIMENTS[model_name], test_batches, manager, True)
                    params = sessions.select('session_id','L').join(params,on='session_id',how='left').sort('session_id')
                    assert params.height == 3281 and params.null_count().sum_horizontal().item() == 0
                    scale, shape, prior = parameter_arrays(params, model_name.startswith('mixture'))
                    length = params['L'].to_numpy().astype(float)
                    # 独立NumPy直接计算各分量离散质量；与训练的expm1风险增量写法不同。
                    h0, h1 = ((length[:,None]-1)/scale)**shape, (length[:,None]/scale)**shape
                    component_log_mass = -h0+np.log(-np.expm1(-(h1-h0)))
                    log_mass = logsumexp(np.log(prior)+component_log_mass,axis=-1)
                    nll = float(-log_mass.mean())
                    budget_curve = np.exp(log_survival(empirical['r'].to_numpy(),scale,shape,prior)).mean(axis=0)
                    risk = risk_curve(prediction,sessions)
                    product_curve = np.r_[1.,risk['risk_standardized_survival'].to_numpy()]
                    np.testing.assert_allclose(np.r_[1.,risk['empirical_survival_product'].to_numpy()],
                                               empirical['survival'].to_numpy(),atol=1e-14)
                    joined = prediction.select('session_id','req_idx_raw','log_survival').join(params,on='session_id',how='left')
                    s2,h2,p2 = parameter_arrays(joined,model_name.startswith('mixture'))
                    ls = row_log_survival(joined['req_idx_raw'].to_numpy()-1,s2,h2,p2)
                    err = float(abs(ls-joined['log_survival'].to_numpy()).max())
                    assert err < 1e-8
                    max_log_s_error = max(max_log_s_error,err)
                    y = prediction['y_continue'].to_numpy()
                    branch_bce = {}
                    for field, label in [('gamma','gamma_bce'),('content_stay','stay_bce')]:
                        p = np.clip(prediction[field].to_numpy().astype(float),1e-6,1-1e-6)
                        branch_bce[label] = float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean())
                    row = dict(**metadata, session_nll=nll,
                        budget_rmse50=curve_error(empirical['survival'][:51],budget_curve[:51])['rmse'],
                        product_rmse50=curve_error(empirical['survival'][:51],product_curve[:51])['rmse'],
                        budget_effective_scale_mean=float(effective_scale(scale,shape,prior).mean()),
                        mean_gamma=float(prediction['gamma'].mean()), mean_stay=float(prediction['content_stay'].mean()),
                        **branch_bce, **metrics['test'], valid_auc=metrics['valid']['auc'], valid_logloss=metrics['valid']['logloss'])
                    rows.append(row)
                    curve_rows.append(empirical.with_columns(pl.Series('budget',budget_curve),
                        pl.Series('product',product_curve),*[pl.lit(v).alias(k) for k,v in metadata.items()]))
                    parameter_rows.append(params.with_columns(*[pl.lit(v).alias(k) for k,v in metadata.items()]))
                    if variant != 'baseline':
                        loss_path = source/key/f'train_loss_epoch_{epoch:02d}.json'
                        losses = json.loads(loss_path.read_text())
                        assert losses['rows'] == coverage['train']['rows']
                        loss_rows.append(dict(**metadata,**losses['online_batch_weighted_terms']))
                    print('CHECK',key,epoch,json.dumps(row),flush=True)
    for p,h in hashes.items():
        assert digest(Path(p)) == h,p
    frame = pl.DataFrame(rows)
    frame.write_parquet(run/'analysis_metrics.parquet')
    pl.concat(curve_rows).write_parquet(run/'analysis_curves.parquet')
    pl.concat(parameter_rows,how='diagonal_relaxed').write_parquet(run/'analysis_parameters.parquet')
    pl.DataFrame(loss_rows).write_parquet(run/'analysis_training_losses.parquet')
    aggregates = frame.group_by('model','variant','epoch').agg(
        *[pl.col(m).mean().alias(m+'_mean') for m in METRICS],
        *[pl.col(m).std().alias(m+'_std') for m in METRICS]).sort('model','variant','epoch')
    aggregates.write_parquet(run/'analysis_aggregates.parquet')
    save_json(run/'analysis_audit.json', dict(checkpoints=checkpoints,full_evaluations_recomputed=evaluations,
        parameter_session_readouts=checkpoints*3281, old_prediction_files_unchanged=len(old_audit['prediction_sha256']),
        maximum_metric_difference=max_metric_error, maximum_log_survival_error=max_log_s_error,
        all_initializations_exact=True, all_uid_tables_unchanged=True, all_sources_unchanged=True,
        source_hashes=hashes, checkpoint_reload=reloads, analysis_source_sha256=digest(Path(__file__))))


def generate(run, report):
    report.mkdir(parents=True,exist_ok=True)
    frame = pl.read_parquet(run/'analysis_metrics.parquet')
    agg = pl.read_parquet(run/'analysis_aggregates.parquet')
    curves = pl.read_parquet(run/'analysis_curves.parquet')
    models = sorted(frame['model'].unique().to_list(),reverse=True)
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,metric,title in zip(axes.flat,['auc','user_auc','budget_rmse50','product_rmse50'],
                               ['Test Pooled AUC','Test User-GAUC','Budget S RMSE (r=0..50)','Final risk-product S RMSE (r=0..50)']):
        for model in models:
            for variant in VARIANTS:
                selected = agg.filter((pl.col('model')==model)&(pl.col('variant')==variant)).sort('epoch')
                ax.plot(selected['epoch'],selected[metric+'_mean'],color=COLORS[variant],
                        linestyle='-' if model.startswith('mixture') else '--',
                        label=f'{model.split("_")[0]} {LABELS[variant]}')
        ax.set(title=title,xlabel='Epoch',xticks=[1,2,3,4,5]); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.savefig(report/'逐epoch对照.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,model in zip(axes,models):
        selected = curves.filter((pl.col('model')==model)&(pl.col('epoch')==5)&(pl.col('r')<=50))
        emp=selected.filter(pl.col('variant')=='baseline').group_by('r').agg(pl.col('survival').mean()).sort('r')
        ax.plot(emp['r'],emp['survival'],color='black',linewidth=2,label='Observed S')
        for variant in VARIANTS:
            c=selected.filter(pl.col('variant')==variant).group_by('r').agg(pl.col('budget').mean(),pl.col('product').mean()).sort('r')
            ax.plot(c['r'],c['budget'],color=COLORS[variant],linestyle='--',label=LABELS[variant]+' budget')
            ax.plot(c['r'],c['product'],color=COLORS[variant],label=LABELS[variant]+' final')
        ax.set(title=model,xlabel='Completed screens r',ylabel='Survival',ylim=(0,1));ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,ncol=2)
    fig.savefig(report/'全局生存曲线.png',dpi=170);plt.close(fig)
    text = ['# 第21份：分支辅助Loss实验报告','',
        'KR v3、编码UID置0、单/混合威布尔+pace、3seed；连续训练5epoch，每轮末完整valid/test。新增18条轨迹、90次test和90次valid；6条无辅助对照直接复用，没有重复训练。','',
        '数据与Loss定义见[实验说明](实验说明.md)。辅助权重预先固定为1，不使用test选择权重或轮次。主表为固定第5轮末权重，均值±seed标准差；User-AUC是按请求数加权的用户内AUC。','',
        '## 核心发现','']
    for model in models:
        values = {v: agg.filter((pl.col('model')==model)&(pl.col('variant')==v)&(pl.col('epoch')==5)).row(0,named=True)
                  for v in VARIANTS}
        old, left, both = values['baseline'], values['left'], values['both']
        text.append(f'{model}：仅左辅助使左塔S RMSE从{old["budget_rmse50_mean"]:.4f}变为{left["budget_rmse50_mean"]:.4f}；'
                    f'其AUC从{old["auc_mean"]:.6f}变为{left["auc_mean"]:.6f}，User-AUC从{old["user_auc_mean"]:.6f}变为{left["user_auc_mean"]:.6f}。'
                    f'双辅助的AUC={both["auc_mean"]:.6f}、LogLoss={both["logloss_mean"]:.6f}、左塔RMSE={both["budget_rmse50_mean"]:.4f}。')
        text.append('')
    text += [
        '## 固定第5轮结果','',
        '| 结构 | Loss变体 | Pooled AUC | LogLoss | User-AUC | Intra-Step GAUC | 左塔会话NLL | 左塔S RMSE | 最终曲线RMSE |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for model in models:
        for variant in VARIANTS:
            row=agg.filter((pl.col('model')==model)&(pl.col('variant')==variant)&(pl.col('epoch')==5)).row(0,named=True)
            cells=[f'{row[m+"_mean"]:.6f} ± {row[m+"_std"]:.6f}' for m in METRICS]
            text.append('| '+' | '.join([model,LABELS[variant],*cells])+' |')
    text += ['','左塔NLL按3,281场完整test会话各计一次；两个曲线RMSE都在r=0..50评估。左塔曲线是全体首屏预算分布的平均S；最终曲线是已观察风险集内平均继续概率的连乘，两者不是同一预测对象。','',
        '![逐epoch对照](逐epoch对照.png)','','![全局生存曲线](全局生存曲线.png)','','## 相对旧联合训练的第5轮配对变化','',
        '| 结构 | 变体 | ΔAUC均值 | ΔUser-AUC均值 | Δ左塔NLL均值 | Δ左塔RMSE均值 | 三seed的ΔAUC |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for model in models:
        old=frame.filter((pl.col('model')==model)&(pl.col('variant')=='baseline')&(pl.col('epoch')==5)).sort('seed')
        for variant in VARIANTS[1:]:
            new=frame.filter((pl.col('model')==model)&(pl.col('variant')==variant)&(pl.col('epoch')==5)).sort('seed')
            differences={m:new[m].to_numpy()-old[m].to_numpy() for m in ['auc','user_auc','session_nll','budget_rmse50']}
            text.append('| '+' | '.join([model,LABELS[variant],*[f'{differences[m].mean():+.6f}' for m in differences],
                ', '.join(f'{v:+.6f}' for v in differences['auc'])])+' |')
    text += ['','## 分支概率诊断（第5轮，三seed均值）','',
        '| 结构 | 变体 | 平均gamma | 平均stay | gamma独立BCE | stay独立BCE |',
        '| --- | --- | --- | --- | --- | --- |']
    for model in models:
        for variant in VARIANTS:
            group=frame.filter((pl.col('model')==model)&(pl.col('variant')==variant)&(pl.col('epoch')==5))
            text.append('| '+' | '.join([model,LABELS[variant],
                *[f'{group[m].mean():.6f}' for m in ['mean_gamma','mean_stay','gamma_bce','stay_bce']]])+' |')
    text += ['','这四列均按test请求汇总，单独BCE用1e-6概率截断，仅作预测诊断，不混同按场NLL。训练日志中的左项关闭时记0，表中的离散NLL始终独立计算，不将“未开启”误认作完美拟合。']
    text += ['','## 每轮完整test均值','',
        '| 结构 | 变体 | epoch | AUC | LogLoss | User-AUC | 左塔NLL | 左塔S RMSE |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for model in models:
        for variant in VARIANTS:
            for row in agg.filter((pl.col('model')==model)&(pl.col('variant')==variant)).sort('epoch').iter_rows(named=True):
                text.append('| '+' | '.join([model,LABELS[variant],str(row['epoch']),
                    *[f'{row[m+"_mean"]:.6f}' for m in ['auc','logloss','user_auc','session_nll','budget_rmse50']]])+' |')
    text += ['','## 如何解释这次实验','',
        '左塔离散NLL拟合的是观测到的总停止长度，不是单独标注的潜在时间预算。右塔独立BCE也使用总继续标签，不构成满意度监督。若两支各自逼近总继续概率p，乘积将趋向p²，因此三个目标存在取舍；第21份承诺“曲线完全贴合且AUC不下降”没有数学保证。','',
        '按场NLL等于每场one-hot KL。个性化Q_s下，它一般不等于先平均预测分布再对组直方图算KL；后者会留下更多个体层面的自由度。本次没有运行分组KL，没有使用test经验直方图作为训练目标，也没有把NLL误写成已计算的全局KL数值。','',
        '这是一组固定辅助权重下的对照，不穷尽弱正则、分组分布监督或其他标签的可能性。不依据本次test曲线追加调权。预算拟合与联合判别应分别汇报，不能把其中一个指标提升当作另外一个也必然提升。','',
        '量级也不能忽略：训练会话NLL约3.5、请求BCE约0.22。由于本次左项按场、主项按请求归一化，权重1不是等梯度强度；完整会话的逐屏gamma BCE求和等于单场NLL，故按场NLL均值与按请求gamma BCE均值相差训练平均屏数15.7427倍。终屏计一次避免把整场NLL在每屏重复计算；与按请求gamma BCE相比，两种归一化仅差全局系数，不能把同名权重1混为一谈。','',
        '若后续采用左停止监督，应把该分支明确描述成观测总停止分布预测；若仍要解释为纯预算，需要额外的监督或可检验识别假设。相同继续标签的右BCE不能替代CTR/满意度标签，也不能据此解决第16份组件语义的负结果。','',
        'v3沿用完整会话依赖的分屏起点、屏末elapsed和pace；结论仅针对当前离线协议。没有新增在线或因果预算识别证据。','',
        '## 数据、实现与核查证据','',
        'train为2022-04-08至05-06的1,721,745请求/109,368会话，valid 05-07为57,872请求/3,822会话，test 05-08为48,898请求/3,281会话、855用户。User-AUC沿用823个有双类别用户、48,855请求；其他用户不参与该项，不影响Pooled AUC。','',
        '18条新轨迹共有参数初始化hash、优化器参数组、物理请求顺序均与同seed旧对照完全一致。所有54特征及处理均未更改，仅编码UID沿用置0。每轮421更新，5轮2105更新，轮间不重置Adam、不恢复best。预算lr=0.005、其他lr=0.001、batch4096、clip10、普通权重decay1e-6、embedding/bias/1D为0。','',
        '- [训练日志](../../runs/plan21_auxiliary/logs/train.log) · [配置](../../runs/plan21_auxiliary/config.json) · [覆盖审计](../../runs/plan21_auxiliary/coverage.json)',
        '- [全部指标](../../runs/plan21_auxiliary/analysis_metrics.parquet) · [完整曲线](../../runs/plan21_auxiliary/analysis_curves.parquet) · [训练Loss分项](../../runs/plan21_auxiliary/analysis_training_losses.parquet)',
        '- [checkpoint、指标与NumPy独立公式审计](../../runs/plan21_auxiliary/analysis_audit.json)',
        '- [新模型](../../pipeline/auxiliary_exit_model.py) · [训练入口](../../pipeline/run_auxiliary_exit.py) · [分析入口](../../pipeline/report_auxiliary_exit.py) · [专项测试](../../tests/test_auxiliary_exit.py)',
        '', '复现：仓库根目录先运行`uv run python -m workspace.kr_time_budget.pipeline.run_auxiliary_exit --output 新目录`，再用本脚本`--run 新目录`读取；旧结果目录不能覆盖。仅重新生成报告使用`--report-only`。','']
    reference = json.loads((WORKSPACE/'runs/epochwise5_uid_zero/summary.json').read_text())
    fusion = [compact(reference[f'{s}/fusion'][-1]['test']) for s in [2026,2027,2028]]
    text += ['同期已有UID置0 Fusion固定第5轮参考：'+', '.join(
        f'{m}={np.mean([r[m] for r in fusion]):.6f}' for m in ['auc','logloss','user_auc','intra_step_gauc'])+
        '。这组参考来自第18份已有训练，不计入本次新增轨迹，也不以其test指标选择本次超参。','']
    (report/'实验报告.md').write_text('\n'.join(text))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=WORKSPACE/'runs/plan21_auxiliary')
    parser.add_argument('--report',type=Path,default=WORKSPACE/'回复/21新 Loss')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if not args.report_only:
        analyze(args.run)
    generate(args.run,args.report)
    print('REPORT',args.report/'实验报告.md',flush=True)


if __name__=='__main__':
    main()
