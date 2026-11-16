import polars as pl

from rosen.core.enum_type import FeatureSource
from rosen.emb import SchemaManager
from rosen.emb.schema import QuantileEmbSetting, SeqGroupConfig, SequenceSetting


def test_quantile_setting_can_share_scalar_buckets_with_sequence(tmp_path) -> None:
    price = QuantileEmbSetting(
        "price", FeatureSource.ITEM, bucket_count=3, embedding_dim=8
    )
    price_seq = SequenceSetting(
        "price_seq",
        price,
        SeqGroupConfig("history", "seq_len", max_len=3),
    )
    manager = SchemaManager([price, price_seq], str(tmp_path / "schema"))
    raw = pl.DataFrame(
        {
            "price": [0.0, 1.0, 2.0, 3.0],
            "price_seq": [[], [0.0], [0.0, 1.0], [1.0, 2.0, 3.0]],
        }
    ).lazy()

    manager.fit(raw)
    transformed = manager.transform(raw).collect()

    assert price.is_fitted
    assert price_seq.is_fitted
    assert transformed["price"].dtype == pl.UInt32
    assert transformed["price_seq"].dtype == pl.List(pl.UInt32)
    assert transformed["price_seq"].to_list()[0] == []
    assert transformed["seq_len"].to_list() == [0, 1, 2, 3]
