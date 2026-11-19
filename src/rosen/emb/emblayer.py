"""Unified embedding owner and lightweight feature/sequence views."""
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import torch
from torch import nn

from rosen.core.enum_type import FeatureSource
from rosen.emb.manager import SchemaManager
from rosen.emb.schema import EmbSetting


SPLIT_METHODS = Literal["source", "name", "stack", "none", None]


class RecEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0, init_method='normal', init_std=1e-4):
        super(RecEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        self.init_method = init_method
        self.init_std = init_std
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_method == 'normal':
            nn.init.normal_(self.embedding.weight, mean=0.0, std=self.init_std)
        elif self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.embedding.weight.data)
        else:
            pass
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    def reinitialize(self, init_method: Optional[str] = None, init_std: Optional[float] = None):
        if init_method is not None:
            self.init_method = init_method
        if init_std is not None:
            self.init_std = init_std
        self.reset_parameters()

    def forward(self, x):
        return self.embedding(x)


class EmbView:
    """
    轻量级特征视图：
    本身没有任何参数，只负责将请求转发给底层的 OmniEmbLayer，并自动绑定 target_sources。
    """
    def __init__(self,
                 omni_layer: 'OmniEmbLayer',
                 target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
                 include_fields: Iterable[str] = None,
                 exclude_fields: Iterable[str] = None):
        super().__init__()
        # 引用同一个巨无霸底座
        self.omni = omni_layer
        self.target_sources = target_sources
        self.include_fields = tuple(include_fields) if include_fields is not None else None
        self.exclude_fields = tuple(exclude_fields) if exclude_fields is not None else None

    def forward(self, interaction, split_by: SPLIT_METHODS = "none") -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:
        return self.omni(
            interaction,
            split_by=split_by,
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )

    def __call__(self, interaction, split_by: SPLIT_METHODS = "none") -> Union[
        Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:
        return self.omni(
            interaction,
            split_by=split_by,
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )

    @property
    def embedding_dim(self) -> int:
        return self.omni.get_output_dim(
            target_sources=self.target_sources,
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
        )

    @property
    def selected_settings(self) -> tuple[EmbSetting, ...]:
        return tuple(
            self.omni.get_settings(
                target_sources=self.target_sources,
                include_fields=self.include_fields,
                exclude_fields=self.exclude_fields,
            )
        )

    @property
    def num_fields(self) -> int:
        """Number of selected fields, including explicitly selected sequences."""
        return len(self.selected_settings)

    @property
    def field_dim(self) -> int:
        """Common field width for stacked output; reject empty or mixed views."""
        dims = {setting.field_name: int(setting.embedding_dim) for setting in self.selected_settings}
        if not dims or len(set(dims.values())) != 1:
            raise ValueError(f"field_dim requires a non-empty view with equal embedding dimensions, got {dims}.")
        return next(iter(dims.values()))

    @property
    def field_segments(self) -> tuple[tuple[str, int, int], ...]:
        cursor = 0
        segments = []
        for setting in self.selected_settings:
            start = cursor
            cursor += int(setting.embedding_dim)
            segments.append((setting.field_name, start, cursor))
        return tuple(segments)

    @property
    def field_slices(self) -> Dict[str, slice]:
        return {
            field_name: slice(start, stop)
            for field_name, start, stop in self.field_segments
        }


class SeqGroupView(EmbView):
    """
    专为序列组设计的强化版视图：
    不仅能获取序列特征，还能自动获取严格对齐的 Target 特征和序列长度。
    """

    def __init__(self, omni_layer, group_name: str, include_fields: Iterable[str]):
        super().__init__(omni_layer, include_fields=include_fields)
        self.group_name = group_name

        # 预加载对齐的 element 设置与 seq_len 字段
        self.target_settings = []
        self.seq_len_field = None
        self.time_field = None
        self._target_forward_valid = True

        for field in self.include_fields:
            setting = next((s for s in self.omni.settings if s.field_name == field), None)
            if setting is None:
                continue

            if hasattr(setting, 'element_setting'):
                self.target_settings.append(setting.element_setting)
            else:
                self._target_forward_valid = False

            if self.seq_len_field is None and hasattr(setting, 'seq_len_field_name'):
                self.seq_len_field = setting.seq_len_field_name
            if hasattr(setting, 'time_field_name'):
                setting_time_field = setting.time_field_name
                if setting_time_field is None:
                    continue
                if self.time_field is None:
                    self.time_field = setting_time_field
                elif self.time_field != setting_time_field:
                    raise ValueError(
                        f"Seq group '{self.group_name}' has inconsistent time_field_name: "
                        f"{self.time_field} vs {setting_time_field}"
                    )

    def forward_target(self, interaction, split_by: str = "none"):
        # 实际上所有 setting 都应暴露 element_setting，forward_target 才不会有歧义
        if split_by == "source":
            raise ValueError("Target sequence view strictly supports split_by='none' or 'name'.")

        computed_results = []
        for target_s in self.target_settings:
            emb = target_s.compute_tensor(interaction, self.omni.emb_modules)
            computed_results.append((target_s, emb))
        return self.omni.format_output(computed_results, split_by)

    def fetch_all(self, interaction, include_time: bool = False):
        """默认返回 (seq_emb, target_emb, seq_len)，需要时间时返回四元组。"""
        seq_emb = self.forward(interaction, split_by="none")
        if not self._target_forward_valid:
            raise ValueError("存在部分 setting 没有 element_setting，函数中断")
        tar_emb = self.forward_target(interaction, split_by="none")
        seq_len = interaction[self.seq_len_field] if self.seq_len_field else None
        hist_time = interaction[self.time_field] if self.time_field else None
        if include_time:
            return seq_emb, tar_emb, seq_len, hist_time
        return seq_emb, tar_emb, seq_len


class OmniEmbLayer(nn.Module):
    """
    巨无霸统一 Embedding 层：
    内部纳管所有 settings。forward 时可通过 target_sources 动态指定范围。
    彻底取代 SideEmb、UserSideEmb、ItemSideEmb 等一众子类。
    """
    def __init__(self, emb_settings: Iterable[EmbSetting] = None, manager: 'SchemaManager' = None, init_method='normal', init_std=1e-3):
        super(OmniEmbLayer, self).__init__()
        if manager is not None:
            emb_settings = manager.settings
        if emb_settings is None:
            raise ValueError("OmniEmbLayer requires manager or emb_settings.")

        self.manager = manager
        self.settings: List[EmbSetting] = list(emb_settings)
        self.emb_modules = nn.ModuleDict()
        self.source2settings: Dict[FeatureSource, List[EmbSetting]] = {}

        self.group2fields = {}
        self._init_embedding(init_method=init_method, init_std=init_std)

        self.seq_groups: Dict[str, SeqGroupView] = {}
        for g_name, fields in self.group2fields.items():
            self.seq_groups[g_name] = SeqGroupView(self, group_name=g_name, include_fields=fields)

        self.whole = EmbView(self, target_sources=(FeatureSource.USER_ID, FeatureSource.USER,
                                                   FeatureSource.ITEM_ID, FeatureSource.ITEM,
                                                   FeatureSource.INTERACTION))
        self.user_all = EmbView(self, target_sources=(FeatureSource.USER_ID, FeatureSource.USER))
        self.item_all = EmbView(self, target_sources=(FeatureSource.ITEM_ID, FeatureSource.ITEM))
        self.inter = EmbView(self, target_sources=(FeatureSource.INTERACTION,))

        self.user_id = EmbView(self, target_sources=(FeatureSource.USER_ID,))
        self.item_id = EmbView(self, target_sources=(FeatureSource.ITEM_ID,))


        domain_fields = tuple(manager.domain_fields) if manager is not None else tuple()
        domain_field = manager.domain_field if manager is not None else None
        self.domain = EmbView(self, include_fields=domain_fields)
        self.domain_id = EmbView(
            self,
            include_fields=((domain_field,) if domain_field is not None else tuple()),
        )
        self.whole_without_domain = EmbView(
            self,
            target_sources=(FeatureSource.USER_ID, FeatureSource.USER,
                            FeatureSource.ITEM_ID, FeatureSource.ITEM,
                            FeatureSource.INTERACTION),
            exclude_fields=domain_fields,
        )
        self.inter_without_domain = EmbView(
            self,
            target_sources=(FeatureSource.INTERACTION,),
            exclude_fields=domain_fields,
        )

    def _init_embedding(self, init_method='normal', init_std=1e-3):
        for setting in self.settings:
            # A. 维护 Source 映射
            self.source2settings.setdefault(setting.source, []).append(setting)

            # B. 确定哪些特征需要物理权重 (nn.Embedding)
            if setting.requires_embedding_module and setting.embedding_field_name not in self.emb_modules:
                self.emb_modules[setting.embedding_field_name] = RecEmbedding(
                    num_embeddings=setting.num_embeddings,
                    embedding_dim=setting.embedding_dim,
                    padding_idx={True: 0, False: None}[setting.padding_zero],
                    init_method=init_method, init_std=init_std
                )

            # C. 自动收集带有组名标签的序列特征
            if hasattr(setting, 'group_name') and setting.group_name:
                self.group2fields.setdefault(setting.group_name, []).append(setting.field_name)

    def reinitialize_large_vocab_embeddings(
            self,
            vocab_size_threshold: int,
            *,
            init_method: Optional[str] = None,
            init_std: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Reinitialize embedding modules whose linked setting vocab size exceeds the threshold.

        Notes:
        - This method is opt-in only. Nothing in the default training flow calls it.
        - When multiple settings share one embedding table, the table is reinitialized at most once.
        - The threshold compares `setting.vocab_size`, not `num_embeddings`, so padding/OOV slots
          do not affect the trigger.
        """
        vocab_size_threshold = int(vocab_size_threshold)
        embedding_meta: Dict[str, Dict[str, Any]] = {}

        for setting in self.settings:
            if not setting.requires_embedding_module:
                continue

            emb_field_name = setting.embedding_field_name
            meta = embedding_meta.setdefault(
                emb_field_name,
                {
                    "embedding_field_name": emb_field_name,
                    "field_names": [],
                    "vocab_size": 0,
                    "num_embeddings": int(setting.num_embeddings),
                },
            )
            if setting.field_name not in meta["field_names"]:
                meta["field_names"].append(setting.field_name)
            meta["vocab_size"] = max(int(meta["vocab_size"]), int(setting.vocab_size))
            meta["num_embeddings"] = max(int(meta["num_embeddings"]), int(setting.num_embeddings))

        reinitialized = []
        for emb_field_name, meta in embedding_meta.items():
            if meta["vocab_size"] <= vocab_size_threshold:
                continue

            if emb_field_name not in self.emb_modules:
                continue
            emb_module = self.emb_modules[emb_field_name]

            emb_module.reinitialize(init_method=init_method, init_std=init_std)
            reinitialized.append(meta)

        return sorted(reinitialized, key=lambda item: item["vocab_size"], reverse=True)


    def forward(
            self,
            interaction,
            split_by: SPLIT_METHODS = "none",
            target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
            include_fields: Iterable[str] = None,
            exclude_fields: Iterable[str] = None,
    ) -> Union[Dict[Union[str, FeatureSource], torch.Tensor], torch.Tensor, None]:

        split_by = split_by.lower() if isinstance(split_by, str) else None
        if split_by not in ("source", "name", "stack", "none", None):
            raise ValueError(f"Invalid split_by: {split_by}")

        # 1. 路由过滤：找出本次 forward 需要处理的所有 Setting
        valid_settings = self._filter_settings(target_sources, include_fields, exclude_fields, split_by)
        computed = [(s, s.compute_tensor(interaction, self.emb_modules)) for s in valid_settings]
        return self.format_output(computed, split_by)


    def _filter_settings(self, target_sources, include_fields, exclude_fields, split_by) -> List[EmbSetting]:
        """职责一：专门负责解析 target_sources 和 include/exclude_fields，返回合法的 Setting 列表"""
        target_sources = ({target_sources} if isinstance(target_sources, FeatureSource) else set(target_sources)) if target_sources is not None else None
        include_fields = set(include_fields) if include_fields is not None else None
        exclude_fields = set(exclude_fields) if exclude_fields is not None else None
        valid_settings = []
        for source, settings in self.source2settings.items():
            if target_sources is not None and source not in target_sources:
                continue
            for setting in settings:
                if include_fields is not None and setting.field_name not in include_fields:
                    continue
                if exclude_fields and setting.field_name in exclude_fields:
                    continue

                if split_by in ("none", "stack", None) and setting.is_sequence_setting:
                    if not (include_fields and setting.field_name in include_fields):
                        continue

                valid_settings.append(setting)
        return valid_settings

    def format_output(self, computed_results: List[tuple], split_by: str):
        """职责三：根据 split_by 策略打包最终输出的 Tensor 或 Dict"""
        if not computed_results:
            return None

        if split_by == "name":
            return {setting.field_name: emb for setting, emb in computed_results}

        if split_by == "stack":
            shapes = {setting.field_name: tuple(emb.shape) for setting, emb in computed_results}
            if len(set(shapes.values())) != 1:
                raise ValueError(
                    "split_by='stack' requires matching tensor shapes for all selected fields "
                    f"(including embedding dimensions), got {shapes}. "
                    "Use include_fields/exclude_fields to select compatible fields."
                )
            return torch.stack([emb for _, emb in computed_results], dim=-2)

        if split_by == "source":
            output_dict = {}
            for setting, emb in computed_results:
                output_dict.setdefault(setting.source, []).append(emb)
            return {source: torch.cat(embs, dim=-1) for source, embs in output_dict.items()}

        all_embs = [emb for _, emb in computed_results]
        return torch.cat(all_embs, dim=-1)

    def get_output_dim(self,
                       target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
                       include_fields: Iterable[str] = None,
                       exclude_fields: Iterable[str] = None) -> int:
        """
        获取指定 target_sources 下拼接后 Embedding 的总维度大小。
        这是为了弥补移除 SideEmb 后，下游网络无法直接获取 self.embedding_size 的问题。
        """
        valid_settings = self.get_settings(
            target_sources=target_sources,
            include_fields=include_fields,
            exclude_fields=exclude_fields,
        )
        return sum(setting.embedding_dim for setting in valid_settings)

    def get_settings(
            self,
            target_sources: Union[FeatureSource, Iterable[FeatureSource]] = None,
            include_fields: Iterable[str] = None,
            exclude_fields: Iterable[str] = None,
    ) -> tuple[EmbSetting, ...]:
        return tuple(
            self._filter_settings(
                target_sources=target_sources,
                include_fields=include_fields,
                exclude_fields=exclude_fields,
                split_by="none",
            )
        )


_LEGACY_NAMES = frozenset(('BoleEmbLayer', 'SideEmb', 'UserSideEmb', 'ItemSideEmb', 'InterSideEmb', 'CustomSideEmb', 'ProfileEncoder', 'UserProfileEncoder', 'ItemProfileEncoder', 'SeqEmbedder'))


def __getattr__(name):
    """Resolve old import/pickle paths without loading profile code for new models."""
    if name in _LEGACY_NAMES:
        from . import legacy
        value = getattr(legacy, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["RecEmbedding", "EmbView", "SeqGroupView", "OmniEmbLayer", "SPLIT_METHODS", *_LEGACY_NAMES]
