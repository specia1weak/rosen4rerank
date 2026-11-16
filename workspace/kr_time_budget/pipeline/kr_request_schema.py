"""KR Request 特征集与Rosen编码配置。

特征分类：54字段 = 30共享背景 + 6预算与时钟 + 18内容。
    用户画像26 + 开场上下文4：两塔共享，支持用户/场景与内容的交叉。
    预算先验4：历史场均分钟数/屏数、当天开场前场次/耗时，仅供预算塔。
    会话时钟2：elapsed_min（分钟）、req_idx（屏），仅归预算公式，不进内容塔。
    内容历史9 + 当屏列表9：仅供内容塔；历史只使用当前Request之前的信息。

目标路由：预算参数网34字段，内容塔48字段，Fusion使用全部54字段。
时钟归属不等于公式已接入：现有v5只在公式使用elapsed_min，req_idx尚未接入。
BUDGET_SCOPE_FIELDS表示36字段的目标预算归属。历史模型仍显式保留自己的路由。

处理理念：
    1. 业务分类与编码类型分开；FeatureSource不决定字段进入哪座塔。
    2. Schema仅在train上fit，valid/test加载同一词表；历史先验遵守时间边界。
    3. 上游仅对LOG1P_FIELDS做log1p；当前数值Setting固定[0,1]，只透传Float32。
       网络缩放与公式原始量必须分离；不能把对数分钟或屏数当作原始分钟。
    4. 标签与审计列放EXTRA_COLUMNS，不因被Dataset读取而成为模型特征。
    5. 多标签保留重复、按原顺序截断并sum聚合；空历史保留空列表、padding为0。
    6. FEATURES及settings顺序属于现有数据/checkpoint契约，重命名不改变其顺序。
"""
from rosen.core.enum_type import FeatureSource as FS
from rosen.emb import SchemaManager
from rosen.emb.schema import SparseEmbSetting, MultiSparseSetting, MinMaxDenseSetting


# 共享背景：用户画像26 + 开场上下文4。
USER_CATEGORY_FIELDS = [
    'user_active_degree', 'is_live_streamer', 'is_video_author',
    'friend_user_num_range', 'register_days_range',
    'follow_user_num_range', 'fans_user_num_range',
] + [f'onehot_feat{i}' for i in range(18)]
USER_PROFILE_FIELDS = ['user_id'] + USER_CATEGORY_FIELDS
SESSION_CONTEXT_FIELDS = ['tab', 'day_of_week', 'is_weekend', 'start_hour']
SHARED_FIELDS = USER_PROFILE_FIELDS + SESSION_CONTEXT_FIELDS

# 预算先验4与时钟2：两种时钟量纲分离，不传给内容塔。
USER_BUDGET_PRIOR_FIELDS = ['user_hist_session_min', 'user_hist_req_cnt']
DAY_BUDGET_PRIOR_FIELDS = ['day_prev_sess_cnt', 'day_prev_elapsed_min']
BUDGET_PRIOR_FIELDS = USER_BUDGET_PRIOR_FIELDS + DAY_BUDGET_PRIOR_FIELDS
SESSION_CLOCK_FIELDS = ['elapsed_min', 'req_idx']
BUDGET_EXCLUSIVE_FIELDS = BUDGET_PRIOR_FIELDS + SESSION_CLOCK_FIELDS

# 前序内容历史9：反馈只取前序Request，hist_tags取前两屏。
HISTORY_NUMERIC_FIELDS = [
    'prev1_click_cnt', 'prev1_longview_cnt', 'prev1_pos_cnt',
    'sess_cum_click_rate', 'sess_cum_longview_rate', 'sess_cum_pos_rate',
    'sess_cum_hate_cnt',
]
CONTENT_HISTORY_FIELDS = ['is_first_req'] + HISTORY_NUMERIC_FIELDS + ['hist_tags']

# 当屏候选列表9：静态属性和过去日物品先验，不含当屏真实反馈。
SLATE_NUMERIC_FIELDS = [
    'slate_prior_ctr_mean', 'slate_prior_longview_mean', 'slate_dur_mean_min',
    'slate_unique_tag_cnt', 'slate_unique_author_cnt',
]
CONTENT_SLATE_FIELDS = [
    'slate_tags', 'slate_video_types', 'slate_music_types',
] + SLATE_NUMERIC_FIELDS + ['slate_has_rand']
CONTENT_EXCLUSIVE_FIELDS = CONTENT_HISTORY_FIELDS + CONTENT_SLATE_FIELDS

# 业务分类与目标路由；历史Content-only基线使用CONTENT_EXCLUSIVE_FIELDS。
FEATURE_CATEGORIES = {
    'shared': SHARED_FIELDS,
    'budget_exclusive': BUDGET_EXCLUSIVE_FIELDS,
    'content_exclusive': CONTENT_EXCLUSIVE_FIELDS,
}
BUDGET_PARAMETER_FIELDS = SHARED_FIELDS + BUDGET_PRIOR_FIELDS
BUDGET_SCOPE_FIELDS = BUDGET_PARAMETER_FIELDS + SESSION_CLOCK_FIELDS
CONTENT_TOWER_FIELDS = SHARED_FIELDS + CONTENT_EXCLUSIVE_FIELDS
CONTENT_MASK_FIELDS = frozenset(BUDGET_EXCLUSIVE_FIELDS)

# 兼容已有parquet/checkpoint的54字段顺序；Fusion直接使用FEATURES。
FEATURES = (
    USER_PROFILE_FIELDS + USER_BUDGET_PRIOR_FIELDS
    + SESSION_CONTEXT_FIELDS + DAY_BUDGET_PRIOR_FIELDS + SESSION_CLOCK_FIELDS
    + CONTENT_HISTORY_FIELDS + CONTENT_SLATE_FIELDS
)
LOG1P_FIELDS = BUDGET_PRIOR_FIELDS
EXTRA_COLUMNS = [
    'request_id', 'session_id', 'user_id_raw', 'tab_raw', 'req_idx_raw',
    'y_continue', 'elapsed_min', 'intra_step_group', 'active_step_group',
]

assert [len(fields) for fields in FEATURE_CATEGORIES.values()] == [30, 6, 18]
assert sum(len(fields) for fields in FEATURE_CATEGORIES.values()) == len(set(FEATURES)) == 54
assert set().union(*(set(fields) for fields in FEATURE_CATEGORIES.values())) == set(FEATURES)
assert len(BUDGET_PARAMETER_FIELDS) == 34 and len(CONTENT_TOWER_FIELDS) == 48
assert set(BUDGET_SCOPE_FIELDS) & set(CONTENT_TOWER_FIELDS) == set(SHARED_FIELDS)


def request_manager(path, load=False):
    """按固定顺序构建54字段Schema；load=True时复用已有train-fit状态。"""
    settings = [SparseEmbSetting('user_id', FS.USER_ID, 16, min_freq=10, use_oov=True)]
    settings += [
        SparseEmbSetting(name, FS.USER, 8, min_freq=10, use_oov=True)
        for name in USER_CATEGORY_FIELDS
    ]
    settings += [
        SparseEmbSetting(name, FS.INTERACTION, 8, padding_zero=name != 'tab', use_null=name != 'tab')
        for name in SESSION_CONTEXT_FIELDS
    ]
    settings += [
        SparseEmbSetting(name, FS.INTERACTION, 4)
        for name in ['is_first_req', 'slate_has_rand']
    ]
    settings += [
        MultiSparseSetting(
            name, FS.ITEM, dim, max_tag_len=cap,
            is_string_format=True, separator=',', agg='sum',
        )
        for name, dim, cap in [
            ('hist_tags', 16, 15), ('slate_tags', 16, 12),
            ('slate_video_types', 8, 6), ('slate_music_types', 8, 6),
        ]
    ]
    # 固定[0,1]仅透传Float32；LOG1P_FIELDS的变换由上游执行。
    numeric_fields = LOG1P_FIELDS + SESSION_CLOCK_FIELDS + HISTORY_NUMERIC_FIELDS + SLATE_NUMERIC_FIELDS
    settings += [
        MinMaxDenseSetting(name, FS.INTERACTION, min_val=0., max_val=1.)
        for name in numeric_fields
    ]
    assert len(settings) == len(FEATURES) == 54
    manager = SchemaManager(settings, str(path), label_fields='y_continue', domain_fields='tab')
    if load:
        manager.load_schema()
    return manager
