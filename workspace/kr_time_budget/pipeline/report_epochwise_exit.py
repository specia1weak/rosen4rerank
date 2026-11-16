"""只读审计逐epoch5轮run，生成均值曲线、完整表及回复文档；不训练/推理。"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from workspace.kr_time_budget.pipeline.exit_training import WORKSPACE
from workspace.kr_time_budget.pipeline.run_exit_v3 import scores


METRICS = ['auc', 'logloss', 'user_auc', 'intra_step_gauc']


def compact(record):
    return dict(auc=record['overall.auc'], logloss=record['logloss'],
                user_auc=record['user_gauc']['auc'], intra_step_gauc=record['intra_step_gauc']['auc'])


def metric_difference(actual, recorded):
    """Polars并行group sum可能有末位舍入；覆盖/计数严格相等，数值容差1e-12。"""
    maximum = 0.
    for key, value in actual.items():
        reference = recorded[key]
        if isinstance(value, dict):
            maximum = max(maximum, metric_difference(value, reference))
        elif isinstance(value, float):
            error = abs(value-reference)
            assert np.isfinite(value) and error <= 1e-12, (key, value, reference)
            maximum = max(maximum, error)
        else:
            assert value == reference, (key, value, reference)
    return maximum


def generate(run, report):
    config = json.loads((run/'config.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    coverage = json.loads((run/'coverage.json').read_text())
    assert set(summary) == {f'{seed}/{name}' for seed in config['seeds'] for name in config['experiments']}
    assert json.loads((run/'complete.json').read_text())['epoch_test_evaluations'] == len(summary)*5
    assert hashlib.sha256((WORKSPACE/'data/v3/manifest.json').read_bytes()).hexdigest() == config['dataset_manifest_sha256']
    for source, digest in config['source_hashes'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest
    expected = {split: pl.read_parquet(WORKSPACE/f'data/v3/{split}.parquet',
        columns=['request_id', 'session_id', 'user_id_raw', 'req_idx_raw', 'y_continue'])
        for split in ['valid', 'test']}
    rows, alignments, source_hashes = [], {}, {}
    maximum_metric_difference = 0.
    for key, records in summary.items():
        seed, name = key.split('/')
        assert [r['epoch'] for r in records] == [1, 2, 3, 4, 5]
        alignment = json.loads((run/key/'historical_alignment.json').read_text())
        assert set(alignment['epochs']) == {'1', '3'}
        assert all(a['terminal_state_exact'] for a in alignment['epochs'].values())
        alignments[key] = alignment
        for record in records:
            epoch = record['epoch']
            assert record['step'] == epoch*coverage['train']['batches']
            assert record['optimizer_state_steps'] == [record['step']]
            assert record['evaluation_preserved_model_optimizer_rng']
            checkpoint = torch.load(run/key/f'epoch_{epoch:02d}/checkpoint.pt',
                                    map_location='cpu', weights_only=False)
            digest = hashlib.sha256()
            for field, tensor in sorted(checkpoint['model_state_dict'].items()):
                digest.update(field.encode())
                digest.update(tensor.numpy().tobytes())
            assert digest.hexdigest() == checkpoint['state_sha256'] == record['state_sha256']
            assert checkpoint['epoch'] == epoch and checkpoint['global_step'] == record['step']
            assert {int(s['step']) for s in checkpoint['optimizer_state_dict']['state'].values()} == {record['step']}
            for split in ['valid', 'test']:
                path = run/key/f'epoch_{epoch:02d}/{split}_predictions.parquet'
                prediction = pl.read_parquet(path)
                for col in expected[split].columns:
                    assert prediction[col].equals(expected[split][col]), (key, epoch, split, col)
                assert prediction.height == prediction['request_id'].n_unique() == coverage[split]['rows']
                recomputed = scores(prediction)
                maximum_metric_difference = max(maximum_metric_difference,
                                               metric_difference(recomputed, record[split]))
                if not name.startswith('fusion'):
                    assert (prediction['gamma_time'] == 1).all()
                    assert (prediction['prediction']-prediction['gamma']*prediction['content_stay']).abs().max() == 0
                rows.append(dict(seed=int(seed), experiment=name, epoch=epoch, split=split,
                                 rows=prediction.height, **compact(recomputed)))
                source_hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pl.DataFrame(rows)
    report.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(run/'epoch_metrics.parquet')
    aggregated = frame.group_by('experiment', 'epoch', 'split').agg(
        *[expr for name in METRICS for expr in
          [pl.col(name).mean().alias(name+'_mean'), pl.col(name).std(ddof=1).alias(name+'_std')]]
    ).sort('experiment', 'split', 'epoch')
    aggregated.write_parquet(run/'epoch_aggregates.parquet')
    table = {(r['experiment'], r['epoch'], r['split']): r for r in aggregated.to_dicts()}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, METRICS):
        for name in config['experiments']:
            y = np.array([table[name, e, 'test'][metric+'_mean'] for e in range(1, 6)])
            spread = np.array([table[name, e, 'test'][metric+'_std'] or 0. for e in range(1, 6)])
            line, = axis.plot(range(1, 6), y, marker='o', markersize=3, label=name)
            axis.fill_between(range(1, 6), y-spread, y+spread, color=line.get_color(), alpha=.1)
        axis.set(xlabel='Epoch', ylabel=metric, xticks=range(1, 6), title=f'Full test: {metric}')
        axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=8)
    fig.suptitle('Epoch-end weights, 3 seeds; shaded bands = sample standard deviation')
    fig.savefig(report/'逐epoch曲线.png', dpi=170)
    plt.close(fig)
    def fmt(name, epoch, split, metric):
        r = table[name, epoch, split]
        return f'{r[metric+"_mean"]:.6f} ± {(r[metric+"_std"] or 0.):.6f}'
    lines = ['# 连续训练5轮：逐epoch完整test结果', '',
        '**已完成六配置×三seed共18条连续5epoch轨迹；每轮末直接测完整test，共90次。** 每条2105个optimizer step，没有轮间恢复best、重置优化器或按test提前停止。', '',
        '本轮使用当前轮末权重，区别于第13/14份“训练预算内完整valid最优checkpoint”。不能将两种口径的曲线和数值混称为同一个结果。', '',
        '## 数据与训练口径', '',
        '- KR v3：train截至2022-05-06为1,721,745请求，valid=05-07为57,872，test=05-08为48,898；数据与train-fit schema不变。',
        '- 配置：单/混合威布尔各自有无pace，Fusion及Fusion+pace；seed2026/2027/2028。',
        '- 原模型、原输入路由、同一train物理顺序每轮重复；batch4096，每轮421步，累积梯度1，CPU4线程。',
        '- 一个AdamW；预算LR0.005，内容/Fusion LR0.001；普通权重decay1e-6，Embedding/bias/一维参数零decay；无scheduler/EMA。Ours从第一步起使用乘积BCE联合训练，无预热、冻结或辅助loss。',
        '- 每轮完整valid/test只作观测；固定第5轮结束停止。额外记录轮末valid最优epoch，不据此回滚；不按test选模型。',
        '- User-AUC为用户内AUC按请求数加权，剔除单标签用户；Intra-Step GAUC按确切屏号分组。±是三个seed样本标准差，不是置信区间。', '',
        '## 逐轮曲线', '', '![逐epoch完整test曲线](逐epoch曲线.png)', '',
        '## 第5轮完整test', '',
        '| 配置 | Pooled AUC ↑ | LogLoss ↓ | User-AUC ↑ | Intra-Step GAUC ↑ |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in config['experiments']:
        lines.append('| '+name+' | '+' | '.join(fmt(name, 5, 'test', m) for m in METRICS)+' |')
    best_auc = max(config['experiments'], key=lambda name: table[name, 5, 'test']['auc_mean'])
    best_ll = min(config['experiments'], key=lambda name: table[name, 5, 'test']['logloss_mean'])
    best_user = max(config['experiments'], key=lambda name: table[name, 5, 'test']['user_auc_mean'])
    lines += ['', '第5轮均值中，AUC最高配置为`'+best_auc+'`，LogLoss最低为`'+best_ll+
              '`，User-AUC最高为`'+best_user+'`。这是固定test上的描述性排名，不是统计显著性结论。']
    if {'mixture_step_pace', 'fusion'} <= set(config['experiments']):
        differences = {m: table['mixture_step_pace', 5, 'test'][m+'_mean']-
                      table['fusion', 5, 'test'][m+'_mean'] for m in METRICS}
        lines += ['', '第5轮混合＋pace减Fusion：'+', '.join(f'{m} {differences[m]:+.6f}' for m in METRICS)+
                  '。应同时看总体排序、概率误差和用户内排序，不能用某一项代替全部指标。']
    lines += ['', '## 第3轮到第5轮变化（均值）', '',
        '| 配置 | ΔAUC ↑ | ΔLogLoss ↓ | ΔUser-AUC ↑ | ΔIntra-Step GAUC ↑ |',
        '| --- | ---: | ---: | ---: | ---: |']
    changes = {}
    for name in config['experiments']:
        changes[name] = {m: table[name, 5, 'test'][m+'_mean']-table[name, 3, 'test'][m+'_mean'] for m in METRICS}
        lines.append('| '+name+' | '+' | '.join(f'{changes[name][m]:+.6f}' for m in METRICS)+' |')
    lines += ['', '## 完整逐epoch表', '',
        '| 配置 | epoch | test AUC | test LogLoss | test User-AUC | test Intra-Step GAUC | valid LogLoss |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in config['experiments']:
        for epoch in range(1, 6):
            lines.append('| '+name+f' | {epoch} | '+' | '.join(fmt(name, epoch, 'test', m) for m in METRICS)+
                         ' | '+fmt(name, epoch, 'valid', 'logloss')+' |')
    lines += ['', '## 逐seed结果', '',
        '| seed | 配置 | epoch | test AUC | test LogLoss | test User-AUC | test Intra-Step GAUC |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in frame.filter(pl.col('split') == 'test').sort('seed', 'experiment', 'epoch').to_dicts():
        lines.append(f'| {row["seed"]} | {row["experiment"]} | {row["epoch"]} | '+
                     ' | '.join(f'{row[m]:.6f}' for m in METRICS)+' |')
    lines += ['', '## 审计与边界', '',
        '- 18条轨迹第1、第3轮末模型hash均与第13份保存的对应terminal hash完全一致；不是与预算内best混比。新增逐轮test观测未改变历史前三轮训练轨迹。',
        '- 每次评估前后模型hash、optimizer moment hash、Torch RNG均相同；各轮checkpoint保存同一时点模型及Adam状态，step计数为421/842/1263/1684/2105。',
        f'- 独立读取全部90份checkpoint核对模型hash和Adam step；读取180份valid/test预测，逐列核对请求/会话/用户/屏号/标签及覆盖。全部重算指标通过1e-12绝对容差，最大差异{maximum_metric_difference:.3g}（分组浮点求和顺序）；Ours乘积逐值相等。',
        '- 本轮逐epoch观测的是反复查看过的固定test，不新增独立测试样本，不做显著性或因果/线上收益判断。曲线改善也不能证明已达到模型上限。',
        '- 屏末pace和离线分屏边界仍按原实验保留，详见[第18份审阅](17与18审阅.md)。', '',
        '## 证据与复现', '',
        '- [训练日志](../../runs/epochwise5/logs/train.log) · [配置](../../runs/epochwise5/config.json) · [覆盖](../../runs/epochwise5/coverage.json) · [完整逐轮原始指标](../../runs/epochwise5/summary.json)。',
        '- [审计](../../runs/epochwise5/audit.json) · [聚合数据](../../runs/epochwise5/epoch_aggregates.parquet) · [训练入口](../../pipeline/run_epochwise_exit.py) · [只读报告生成](../../pipeline/report_epochwise_exit.py)。',
        '- [实验说明](逐epoch5轮实验说明.md)。模型、预处理和已有run未修改。', '',
        '仓库根目录运行（训练输出须为全新目录）：', '', '```bash',
        'uv run python -m workspace.kr_time_budget.pipeline.run_epochwise_exit --output workspace/kr_time_budget/runs/epochwise5_new',
        'uv run python -m workspace.kr_time_budget.pipeline.report_epochwise_exit', '```', '']
    # 参数化run时仍保持证据链接指向实际run。
    import os
    run_link = os.path.relpath(run, report)
    note = '\n'.join(lines).replace('../../runs/epochwise5', run_link)
    (report/'逐epoch5轮实验报告.md').write_text(note)
    audit = dict(trajectories=len(summary), epochs_per_trajectory=5,
        checkpoints_audited=len(summary)*5,
        valid_test_predictions_audited=len(source_hashes), full_coverage_exact=True,
        metric_absolute_tolerance=1e-12, maximum_metric_absolute_difference=maximum_metric_difference,
        historical_terminal_alignment=alignments, changes_5_minus_3=changes,
        prediction_sha256=source_hashes, reported_checkpoint='current epoch-end', test_based_selection=False,
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run/'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print('REPORT', report/'逐epoch5轮实验报告.md')
    print('EPOCH5', json.dumps({name: {m: table[name, 5, 'test'][m+'_mean'] for m in METRICS}
                              for name in config['experiments']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=WORKSPACE/'runs/epochwise5')
    args = parser.parse_args()
    generate(args.run.resolve(), WORKSPACE/'回复/18拍板')


if __name__ == '__main__':
    main()
