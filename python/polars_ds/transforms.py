import polars as pl
from typing import Union


class PipeConstructor:
    def __init__(self, df: Union[pl.DataFrame, pl.LazyFrame], name: str = "project"):
        self._frame = df.lazy()
        self._name = name
