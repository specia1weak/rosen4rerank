"""ML1M当前评分后预测：21个Omni字段＋1个已缩放评分旁路。

共享7字段：UID恒0、画像4、UTC开场小时/星期；预算独占4个过去场先验。
内容8字段：电影3、前序评分/类型历史4、当前累计pace1。
Fusion另读step_log/elapsed_log。当前rating/5仅追加给内容/Fusion，
不进预算参数网；rating_audit等extras不自动成为模型特征。
类别词表只fit train；genre原生mean聚合；数值缩放显式发生在上游。
"""
from rosen.core.enum_type import FeatureSource as FS
from rosen.emb import SchemaManager
from rosen.emb.schema import SparseEmbSetting, MultiSparseSetting, MinMaxDenseSetting
from experiments.movielens.pipeline.schema import SHARED, PRIORS, CLOCK_NETWORK, EXTRAS

CONTENT = ['movie_id','genres','release_year_scaled','previous_genres','previous_rating_scaled',
    'session_previous_rating_mean','session_previous_positive_rate','pace']
BUDGET_FIELDS = SHARED+PRIORS
CONTENT_FIELDS = SHARED+CONTENT
FEATURES = SHARED+PRIORS+CONTENT+CLOCK_NETWORK
CURRENT_RATING = 'current_rating_scaled'
EXTRA_COLUMNS = EXTRAS+[CURRENT_RATING]


def manager_at(path, load=False):
    settings = [SparseEmbSetting('user_id', FS.USER_ID,16,use_oov=True)]
    settings += [SparseEmbSetting(c,FS.USER,8,use_oov=True) for c in SHARED[1:]]
    settings += [SparseEmbSetting('movie_id',FS.ITEM_ID,16,use_oov=True)]
    settings += [MultiSparseSetting(c,FS.ITEM,16,max_tag_len=6,is_string_format=True,
        separator='|',agg='mean',use_oov=True) for c in ['genres','previous_genres']]
    numeric = PRIORS+[c for c in CONTENT if c not in ['movie_id','genres','previous_genres']]+CLOCK_NETWORK
    settings += [MinMaxDenseSetting(c,FS.INTERACTION,min_val=0.,max_val=1.) for c in numeric]
    assert len(FEATURES) == len(settings) == 21
    manager = SchemaManager(settings,str(path),label_fields='y_continue',domain_fields='tab')
    if load: manager.load_schema()
    return manager
