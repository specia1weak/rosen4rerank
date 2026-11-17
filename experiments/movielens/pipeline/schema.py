"""ML1M继续评分实验：共享背景、静态预算先验、电影和前序评分历史。

共享7列：UID（训练入口置0）、性别、年龄桶、职业、邮编前3位、开场UTC小时/星期。
预算另读4个仅由之前完整Session计算的先验，步号只进入屏数Weibull公式。
内容另读当前电影静态属性和前序评分反馈；当前rating不是输入。
同秒顺序沿源文件约定；历史按该约定而非假定真实顺序。
数值业务缩放在上游显式完成，Setting固定[0,1]仅透传；多类型用mean聚合。
"""
from rosen.core.enum_type import FeatureSource as FS
from rosen.emb import SchemaManager
from rosen.emb.schema import SparseEmbSetting, MultiSparseSetting, MinMaxDenseSetting

SHARED = ['user_id', 'gender', 'age', 'occupation', 'zip3', 'start_hour', 'start_weekday']
PRIORS = ['past_session_steps_log', 'past_session_min_log', 'past_session_count_log', 'previous_session_gap_log']
CONTENT = ['movie_id', 'genres', 'release_year_scaled', 'previous_genres', 'previous_rating_scaled',
           'session_previous_rating_mean', 'session_previous_positive_rate', 'pace_previous']
CLOCK_NETWORK = ['step_log', 'elapsed_log']
BUDGET_FIELDS = SHARED + PRIORS
CONTENT_FIELDS = SHARED + CONTENT
FEATURES = SHARED + PRIORS + CONTENT + CLOCK_NETWORK
EXTRAS = ['request_id', 'session_id', 'user_id_raw', 'tab_raw', 'req_idx_raw', 'y_continue',
          'elapsed_min', 'elapsed_min_raw', 'timestamp', 'rating_audit', 'same_timestamp_as_previous']


def manager_at(path, load=False):
    settings = [SparseEmbSetting('user_id', FS.USER_ID, 16, use_oov=True)]
    settings += [SparseEmbSetting(c, FS.USER, 8, use_oov=True) for c in SHARED[1:]]
    settings += [SparseEmbSetting('movie_id', FS.ITEM_ID, 16, use_oov=True)]
    settings += [MultiSparseSetting(c, FS.ITEM, 16, max_tag_len=6,
        is_string_format=True, separator='|', agg='mean', use_oov=True)
        for c in ['genres', 'previous_genres']]
    numerical = PRIORS + [c for c in CONTENT if c not in ['movie_id', 'genres', 'previous_genres']] + CLOCK_NETWORK
    settings += [MinMaxDenseSetting(c, FS.INTERACTION, min_val=0., max_val=1.) for c in numerical]
    assert {s.field_name for s in settings} == set(FEATURES)
    manager = SchemaManager(settings, str(path), label_fields='y_continue', domain_fields='tab')
    if load:
        manager.load_schema()
    return manager
