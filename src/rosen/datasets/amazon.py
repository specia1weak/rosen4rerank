import polars as pl

from rosen.datasets.base import DatasetBase


class _AmazonDS(DatasetBase):
    BASE_DIR = DatasetBase.SYSTEM_DATA_DIR / "Amazon"

    ALL_BEAUTY = BASE_DIR / "All_Beauty.csv"
    BOOKS = BASE_DIR / "Books.csv"
    CDS_AND_VINYL = BASE_DIR / "CDs_and_Vinyl.csv"
    CELL_PHONES_AND_ACCESSORIES = BASE_DIR / "Cell_Phones_and_Accessories.csv"
    CLOTHING_SHOES_AND_JEWELRY = BASE_DIR / "Clothing_Shoes_and_Jewelry.csv"
    GROCERY_AND_GOURMET_FOOD = BASE_DIR / "Grocery_and_Gourmet_Food.csv"
    HEALTH_AND_HOUSEHOLD = BASE_DIR / "Health_and_Household.csv"
    MOVIES_AND_TV = BASE_DIR / "Movies_and_TV.csv"
    MUSICAL_INSTRUMENTS = BASE_DIR / "Musical_Instruments.csv"
    PET_SUPPLIES = BASE_DIR / "Pet_Supplies.csv"
    SPORTS_AND_OUTDOORS = BASE_DIR / "Sports_and_Outdoors.csv"
    TOOLS_AND_HOME_IMPROVEMENT = BASE_DIR / "Tools_and_Home_Improvement.csv"
    TOYS_AND_GAMES = BASE_DIR / "Toys_and_Games.csv"
    VIDEO_GAMES = BASE_DIR / "Video_Games.csv"

    BEAUTY = ALL_BEAUTY
    CLOTHING = CLOTHING_SHOES_AND_JEWELRY
    HEALTH = HEALTH_AND_HOUSEHOLD

    AMAZON_DATASETS = {
        "All_Beauty": ALL_BEAUTY,
        "Beauty": BEAUTY,
        "Books": BOOKS,
        "CDs_and_Vinyl": CDS_AND_VINYL,
        "Cell_Phones_and_Accessories": CELL_PHONES_AND_ACCESSORIES,
        "Clothing_Shoes_and_Jewelry": CLOTHING_SHOES_AND_JEWELRY,
        "Clothing": CLOTHING,
        "Grocery_and_Gourmet_Food": GROCERY_AND_GOURMET_FOOD,
        "Health_and_Household": HEALTH_AND_HOUSEHOLD,
        "Health": HEALTH,
        "Movies_and_TV": MOVIES_AND_TV,
        "Musical_Instruments": MUSICAL_INSTRUMENTS,
        "Pet_Supplies": PET_SUPPLIES,
        "Sports_and_Outdoors": SPORTS_AND_OUTDOORS,
        "Tools_and_Home_Improvement": TOOLS_AND_HOME_IMPROVEMENT,
        "Toys_and_Games": TOYS_AND_GAMES,
        "Video_Games": VIDEO_GAMES,
    }

    ALL_BEAUTY_LF = pl.scan_csv(ALL_BEAUTY)
    BOOKS_LF = pl.scan_csv(BOOKS)
    CDS_AND_VINYL_LF = pl.scan_csv(CDS_AND_VINYL)
    CELL_PHONES_AND_ACCESSORIES_LF = pl.scan_csv(CELL_PHONES_AND_ACCESSORIES)
    CLOTHING_SHOES_AND_JEWELRY_LF = pl.scan_csv(CLOTHING_SHOES_AND_JEWELRY)
    GROCERY_AND_GOURMET_FOOD_LF = pl.scan_csv(GROCERY_AND_GOURMET_FOOD)
    HEALTH_AND_HOUSEHOLD_LF = pl.scan_csv(HEALTH_AND_HOUSEHOLD)
    MOVIES_AND_TV_LF = pl.scan_csv(MOVIES_AND_TV)
    MUSICAL_INSTRUMENTS_LF = pl.scan_csv(MUSICAL_INSTRUMENTS)
    PET_SUPPLIES_LF = pl.scan_csv(PET_SUPPLIES)
    SPORTS_AND_OUTDOORS_LF = pl.scan_csv(SPORTS_AND_OUTDOORS)
    TOOLS_AND_HOME_IMPROVEMENT_LF = pl.scan_csv(TOOLS_AND_HOME_IMPROVEMENT)
    TOYS_AND_GAMES_LF = pl.scan_csv(TOYS_AND_GAMES)
    VIDEO_GAMES_LF = pl.scan_csv(VIDEO_GAMES)

    BEAUTY_LF = ALL_BEAUTY_LF
    CLOTHING_LF = CLOTHING_SHOES_AND_JEWELRY_LF
    HEALTH_LF = HEALTH_AND_HOUSEHOLD_LF

    AMAZON_DATASET_LFS = {
        "All_Beauty": ALL_BEAUTY_LF,
        "Beauty": BEAUTY_LF,
        "Books": BOOKS_LF,
        "CDs_and_Vinyl": CDS_AND_VINYL_LF,
        "Cell_Phones_and_Accessories": CELL_PHONES_AND_ACCESSORIES_LF,
        "Clothing_Shoes_and_Jewelry": CLOTHING_SHOES_AND_JEWELRY_LF,
        "Clothing": CLOTHING_LF,
        "Grocery_and_Gourmet_Food": GROCERY_AND_GOURMET_FOOD_LF,
        "Health_and_Household": HEALTH_AND_HOUSEHOLD_LF,
        "Health": HEALTH_LF,
        "Movies_and_TV": MOVIES_AND_TV_LF,
        "Musical_Instruments": MUSICAL_INSTRUMENTS_LF,
        "Pet_Supplies": PET_SUPPLIES_LF,
        "Sports_and_Outdoors": SPORTS_AND_OUTDOORS_LF,
        "Tools_and_Home_Improvement": TOOLS_AND_HOME_IMPROVEMENT_LF,
        "Toys_and_Games": TOYS_AND_GAMES_LF,
        "Video_Games": VIDEO_GAMES_LF,
    }

    @classmethod
    def resolve_name(cls, name: str) -> str:
        normalized = str(name).strip()
        aliases = {
            "beauty": "Beauty",
            "all_beauty": "All_Beauty",
            "clothing": "Clothing",
            "clothing_shoes_and_jewelry": "Clothing_Shoes_and_Jewelry",
            "health": "Health",
            "health_and_household": "Health_and_Household",
        }
        return aliases.get(normalized.lower(), normalized)

    @classmethod
    def resolve_path(cls, name: str):
        resolved = cls.resolve_name(name)
        try:
            return cls.AMAZON_DATASETS[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown Amazon dataset: {name!r}") from exc

    @classmethod
    def resolve_lf(cls, name: str):
        resolved = cls.resolve_name(name)
        try:
            return cls.AMAZON_DATASET_LFS[resolved]
        except KeyError as exc:
            raise KeyError(f"Unknown Amazon dataset: {name!r}") from exc


AmazonDataset = _AmazonDS()


if __name__ == "__main__":
    books_lf = AmazonDataset.BOOKS_LF

    print("Books path:")
    print(AmazonDataset.BOOKS)

    print("\nBooks schema:")
    print(books_lf.collect_schema())

    print("\nBooks row count:")
    print(books_lf.select(pl.len().alias("rows")).collect())

    print("\nBooks head:")
    print(books_lf.head(5).collect())
