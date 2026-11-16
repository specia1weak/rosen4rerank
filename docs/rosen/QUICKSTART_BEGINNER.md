# Beginner Quickstart

这份 quickstart 面向第一次接触 `rosen` 的人。

目标只有一个：用一个单文件脚本跑通最小链路，并且能在这个文件里看清楚完整流向。

如果你已经知道整条链路，只是想看一版更接近真实工程入口的 quickstart，去看 [`QUICKSTART.md`](QUICKSTART.md)。

## 你将看到什么

这份最小脚本会显式展示：

1. 造一份很小的原始数据
2. 定义 `ConfigBase`
3. 定义 `SchemaManager` 和特征规则
4. 手工切 train / valid / test
5. `fit -> transform -> save_as_dataset`
6. `ParquetStreamDataset -> build_model -> BaseTrainer`

## 运行前提

在仓库根目录执行，依赖至少包含：

```bash
pip install torch numpy polars pyarrow scikit-learn psutil matplotlib
```

## 一个最小但完整的单文件例子

```python
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

from rosen.core.enum_type import FeatureSource
from rosen.core.train import EarlyStopper, TrainerComponents, TrainerDataLoaders
from rosen.core.trainers import BaseTrainer
from rosen.data.dataset import ParquetStreamDataset
from rosen.emb import SchemaManager
from rosen.emb.schema import MinMaxDenseSetting, SparseEmbSetting
from rosen.evaluate.evaluator import Evaluator
from rosen.evaluate.manager import EvaluatorManager
from rosen.experiment import WORKSPACE, change_root_workdir
from rosen.experiment.param import ConfigBase, ParamManager
from rosen.models.msr import build_model
from rosen.utils.optimize import split_params_by_decay


change_root_workdir()


@dataclass
class DemoConfig(ConfigBase):
    experiment_name: str = "demo_sharedbottom"
    dataset_name: str = "demo_minimal"
    device: str = "cpu"
    max_epochs: int = 1
    batch_size: int = 4
    model: str = "sharedbottom"
    id_emb_dim: int = 8
    side_emb_dim: int = 4
    ckpt_dir: str = ""


cfg = ParamManager(DemoConfig).build()

work_dir = WORKSPACE / cfg.dataset_name
work_dir.mkdir(parents=True, exist_ok=True)
raw_path = work_dir / "demo_raw.parquet"

raw_df = pl.DataFrame(
    {
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4", "u5", "u5"],
        "item_id": ["i1", "i2", "i1", "i3", "i2", "i4", "i3", "i5", "i4", "i1"],
        "city_id": ["c1", "c1", "c2", "c2", "c1", "c1", "c3", "c3", "c2", "c2"],
        "domain_id": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "price": [10.0, 12.0, 9.0, 15.0, 11.0, 13.0, 14.0, 16.0, 8.0, 7.0],
        "label": [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        "timestamp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
)
raw_df.write_parquet(raw_path)
raw_lf = pl.scan_parquet(raw_path)

settings = [
    SparseEmbSetting("user_id", FeatureSource.USER_ID, embedding_dim=cfg.id_emb_dim, min_freq=1),
    SparseEmbSetting("item_id", FeatureSource.ITEM_ID, embedding_dim=cfg.id_emb_dim, min_freq=1),
    SparseEmbSetting("city_id", FeatureSource.USER, embedding_dim=cfg.side_emb_dim, min_freq=1),
    MinMaxDenseSetting("price", FeatureSource.INTERACTION),
]

manager = SchemaManager(
    settings_list=settings,
    work_dir=work_dir,
    time_field="timestamp",
    label_fields="label",
    domain_fields="domain_id",
)

train_raw = raw_lf.filter(pl.col("timestamp") <= 6)
valid_raw = raw_lf.filter((pl.col("timestamp") >= 7) & (pl.col("timestamp") <= 8))
test_raw = raw_lf.filter(pl.col("timestamp") >= 9)

manager.fit(train_raw)
train_lf = manager.transform(train_raw)
valid_lf = manager.transform(valid_raw)
test_lf = manager.transform(test_raw)

train_path, valid_path, test_path = manager.save_as_dataset(train_lf, valid_lf, test_lf)

train_ds = ParquetStreamDataset(train_path, manager, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
valid_ds = ParquetStreamDataset(valid_path, manager, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

model = build_model(manager, num_domains=2, model_cls=cfg.model)

evaluator_manager = EvaluatorManager(log_path=work_dir / "metrics.log", title=cfg.experiment_name)
evaluator_manager.register("overall", Evaluator("auc"))

optimizer = torch.optim.Adam(
    split_params_by_decay(model.named_parameters(), weight_decay=1e-5),
    lr=1e-3,
)

trainer = BaseTrainer(
    model=model,
    optimizer=optimizer,
    manager=manager,
    loaders=TrainerDataLoaders(train=train_ds, valid=valid_ds),
    components=TrainerComponents(
        evaluator_manager=evaluator_manager,
        early_stopper=EarlyStopper(patience=3),
    ),
    cfg=cfg,
)

trainer.run()
```

## 这个脚本里最重要的观察点

- 原始数据从 `raw_df` 开始就是显式可见的，没有被隐藏到外部工具里。
- `SchemaManager` 只负责特征拟合和编码规则，不负责替你决定训练逻辑。
- `domain_id` 没有注册成 `SparseEmbSetting`，但仍然会被保留到 batch，因为模型路由需要它。
- `ParquetStreamDataset` 输出的已经是 batch 级 `Interaction`。
- `build_model(...)` 只消费 `manager`、`num_domains` 和模型名，不接管你的数据处理过程。

## 初学者下一步看什么

- 想知道数据到底怎样从 raw 变成 batch：看 [`DATA_FLOW.md`](DATA_FLOW.md)
- 想知道 trainer 默认会做什么：看 [`TRAINING_FLOW.md`](TRAINING_FLOW.md)
- 想知道参数怎么从 CLI 进模型：看 [`PARAMS_AND_EXPERIMENTS.md`](PARAMS_AND_EXPERIMENTS.md)
