from __future__ import annotations

import ast
import gzip
import math
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from rosen.datasets.base import DatasetBase


class _Amazon2014DS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "Amazon2014"
    EXTRACTED_DIR = BASE_DIR / "extracted"
    METADATA_DIR = BASE_DIR / "metadata"

    BEAUTY_GZ = BASE_DIR / "reviews_Beauty_5.json.gz"
    CLOTHING_GZ = BASE_DIR / "reviews_Clothing_Shoes_and_Jewelry_5.json.gz"
    HEALTH_GZ = BASE_DIR / "reviews_Health_and_Personal_Care_5.json.gz"

    BEAUTY = EXTRACTED_DIR / "reviews_Beauty_5.json"
    CLOTHING = EXTRACTED_DIR / "reviews_Clothing_Shoes_and_Jewelry_5.json"
    HEALTH = EXTRACTED_DIR / "reviews_Health_and_Personal_Care_5.json"

    BEAUTY_METADATA = METADATA_DIR / "meta_Beauty.json"
    CLOTHING_METADATA = METADATA_DIR / "meta_Clothing_Shoes_and_Jewelry.json"
    HEALTH_METADATA = METADATA_DIR / "meta_Health_and_Personal_Care.json"

    AMAZON2014_DATASETS = {
        "Beauty": BEAUTY,
        "Clothing": CLOTHING,
        "Clothing_Shoes_and_Jewelry": CLOTHING,
        "Health": HEALTH,
        "Health_and_Personal_Care": HEALTH,
    }

    AMAZON2014_COMPRESSED_DATASETS = {
        "Beauty": BEAUTY_GZ,
        "Clothing": CLOTHING_GZ,
        "Clothing_Shoes_and_Jewelry": CLOTHING_GZ,
        "Health": HEALTH_GZ,
        "Health_and_Personal_Care": HEALTH_GZ,
    }

    AMAZON2014_METADATA = {
        "Beauty": BEAUTY_METADATA,
        "Clothing": CLOTHING_METADATA,
        "Clothing_Shoes_and_Jewelry": CLOTHING_METADATA,
        "Health": HEALTH_METADATA,
        "Health_and_Personal_Care": HEALTH_METADATA,
    }

    _metadata_cache: dict[str, pl.DataFrame] = {}

    @classmethod
    def resolve_name(cls, name: str) -> str:
        normalized = str(name).strip()
        aliases = {
            "beauty": "Beauty",
            "clothing": "Clothing",
            "clothing_shoes_and_jewelry": "Clothing_Shoes_and_Jewelry",
            "health": "Health",
            "health_and_personal_care": "Health_and_Personal_Care",
        }
        return aliases.get(normalized.lower(), normalized)

    @classmethod
    def resolve_names(cls, names: Sequence[str] | str | None = None) -> tuple[str, ...]:
        if names is None:
            return ("Beauty", "Clothing", "Health")
        if isinstance(names, str):
            return (cls.resolve_name(names),)
        return tuple(cls.resolve_name(name) for name in names)

    @classmethod
    def resolve_path(cls, name: str) -> Path:
        resolved = cls.resolve_name(name)
        try:
            return cls.AMAZON2014_DATASETS[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown Amazon2014 dataset: {name!r}") from exc

    @classmethod
    def resolve_compressed_path(cls, name: str) -> Path:
        resolved = cls.resolve_name(name)
        try:
            return cls.AMAZON2014_COMPRESSED_DATASETS[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown Amazon2014 dataset: {name!r}") from exc

    @classmethod
    def resolve_metadata_path(cls, name: str) -> Path:
        resolved = cls.resolve_name(name)
        try:
            return cls.AMAZON2014_METADATA[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown Amazon2014 metadata dataset: {name!r}") from exc

    @staticmethod
    def _parse_price(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            price = float(value)
            return price if math.isfinite(price) else None
        text = str(value).strip()
        if not text:
            return None
        match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
        if match is None:
            return None
        try:
            price = float(match.group(0).replace(",", ""))
        except ValueError:
            return None
        return price if math.isfinite(price) else None

    @staticmethod
    def _normalize_categories(value: Any) -> list[list[str]]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [[stripped]] if stripped else []
        if not isinstance(value, (list, tuple)):
            return []

        categories: list[list[str]] = []
        for path in value:
            if isinstance(path, str):
                stripped = path.strip()
                if stripped:
                    categories.append([stripped])
                continue
            if not isinstance(path, (list, tuple)):
                continue
            cleaned = [str(part).strip() for part in path if part is not None and str(part).strip()]
            if cleaned:
                categories.append(cleaned)
        return categories

    @classmethod
    def _load_item_metadata(cls, name: str) -> pl.DataFrame:
        resolved = cls.resolve_name(name)
        cached = cls._metadata_cache.get(resolved)
        if cached is not None:
            return cached

        path = cls.resolve_metadata_path(resolved)
        if not path.exists():
            raise FileNotFoundError(path)

        asins: list[str] = []
        brands: list[str | None] = []
        prices: list[float | None] = []
        categories: list[list[list[str]]] = []
        seen: set[str] = set()
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = ast.literal_eval(text)
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(f"Failed to parse Amazon2014 metadata line {line_no} in {path}") from exc
                if not isinstance(record, dict):
                    continue
                asin = record.get("asin")
                if asin is None:
                    continue
                asin = str(asin)
                if not asin or asin in seen:
                    continue
                seen.add(asin)
                asins.append(asin)
                brand = record.get("brand")
                brand = str(brand).strip() if brand is not None else None
                brands.append(brand or None)
                prices.append(cls._parse_price(record.get("price")))
                categories.append(cls._normalize_categories(record.get("categories")))

        df = pl.DataFrame(
            {
                "parent_asin": asins,
                "brand": brands,
                "price": prices,
                "categories": categories,
            },
            schema={
                "parent_asin": pl.Utf8,
                "brand": pl.Utf8,
                "price": pl.Float32,
                "categories": pl.List(pl.List(pl.Utf8)),
            },
        )
        cls._metadata_cache[resolved] = df
        return df

    @classmethod
    def scan_item_metadata(cls, names: Sequence[str] | str | None = None) -> pl.LazyFrame:
        """Return item-side metadata keyed by parent_asin.

        Brand, price, and categories are exposed. Callers that do not need metadata
        should keep include_metadata=False to avoid parsing the metadata files.
        """
        resolved_names = cls.resolve_names(names)
        if not resolved_names:
            raise ValueError("names must be non-empty.")

        frames = [cls._load_item_metadata(name) for name in resolved_names]
        df = pl.concat(frames, how="vertical")
        if len(frames) > 1:
            df = df.unique(subset=["parent_asin"], keep="first", maintain_order=True)
        return df.lazy()

    @classmethod
    def ensure_extracted(cls, names: Sequence[str] | str | None = None, redo: bool = False) -> list[Path]:
        resolved_names = cls.resolve_names(names)
        cls.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
        output_paths = []
        for name in resolved_names:
            src = cls.resolve_compressed_path(name)
            dst = cls.resolve_path(name)
            if dst.exists() and not redo:
                output_paths.append(dst)
                continue
            if not src.exists():
                raise FileNotFoundError(src)
            with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
                shutil.copyfileobj(fin, fout)
            output_paths.append(dst)
        return output_paths

    @classmethod
    def scan_domain(cls, name: str, domain_id: int = 0, include_metadata: bool = False) -> pl.LazyFrame:
        resolved = cls.resolve_name(name)
        path = cls.resolve_path(resolved)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist. Run Amazon2014Dataset.ensure_extracted() first.")
        lf = (
            pl.scan_ndjson(path)
            .select(
                pl.col("reviewerID").cast(pl.Utf8).alias("user_id"),
                pl.col("asin").cast(pl.Utf8).alias("parent_asin"),
                pl.col("overall").cast(pl.Float32).alias("rating"),
                pl.col("unixReviewTime").cast(pl.Int64).alias("timestamp"),
            )
            .with_columns(
                pl.lit(int(domain_id)).cast(pl.Int32).alias("domain_id"),
                pl.lit(str(resolved)).alias("domain_name"),
            )
        )
        if include_metadata:
            lf = lf.join(cls.scan_item_metadata(resolved), on="parent_asin", how="left")
        return lf

    @classmethod
    def scan_reviews(cls, names: Sequence[str] | str | None = None, include_metadata: bool = False) -> pl.LazyFrame:
        resolved_names = cls.resolve_names(names)
        frames = [
            cls.scan_domain(name, domain_id=domain_id, include_metadata=include_metadata)
            for domain_id, name in enumerate(resolved_names)
        ]
        if not frames:
            raise ValueError("names must be non-empty.")
        return pl.concat(frames, how="vertical")


Amazon2014Dataset = _Amazon2014DS()
