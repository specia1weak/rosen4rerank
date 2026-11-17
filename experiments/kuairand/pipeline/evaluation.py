"""完整预测评估：AUC、用户GAUC和同刷AUC；有效组按请求数加权。"""
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss

def grouped_auc(y, p, groups):
    # Mann-Whitney rank formula, with average ranks for ties. No O(groups)
    # Python/sklearn loop, which is expensive for >100k sessions.
    ranked = pl.DataFrame({'g': groups, 'y': y, 'p': p}).with_columns(pl.col('p').rank('average').over('g').alias('rank'))
    stats = ranked.group_by('g').agg(pl.len().alias('n'), pl.col('y').sum().alias('pos'), (pl.col('rank') * pl.col('y')).sum().alias('rank_sum'))
    valid = stats.filter((pl.col('pos') > 0) & (pl.col('pos') < pl.col('n'))).with_columns(((pl.col('rank_sum') - pl.col('pos') * (pl.col('pos') + 1) / 2) / (pl.col('pos') * (pl.col('n') - pl.col('pos')))).alias('auc'))
    weight = valid['n'].sum()
    return {'auc': (valid['auc'] * valid['n']).sum() / weight if weight else None, 'eligible_groups': valid.height, 'eligible_rows': weight, 'total_groups': stats.height}


def scores(df):
    from sklearn.metrics import roc_auc_score,log_loss
    y,p=df['y_continue'].to_numpy(),df['prediction'].to_numpy()
    r={'overall.auc':float(roc_auc_score(y,p)),'user_gauc':grouped_auc(y,p,df['user_id_raw'].to_numpy()),
       'logloss':float(log_loss(y,p,labels=[0,1])),'rows':df.height,
       'intra_step_gauc':grouped_auc(y,p,df['req_idx_raw'].to_numpy())}
    r['domains']={str(tab):scores_domain(part) for tab,part in df.partition_by('tab_raw',as_dict=True).items()}
    return r


def scores_domain(df):
    from sklearn.metrics import roc_auc_score,log_loss
    y,p=df['y_continue'].to_numpy(),df['prediction'].to_numpy()
    return {'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            'gauc':grouped_auc(y,p,df['user_id_raw'].to_numpy()),'logloss':float(log_loss(y,p,labels=[0,1])),'rows':df.height}
