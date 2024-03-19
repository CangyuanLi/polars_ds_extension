import polars as pl
import polars.selectors as cs
from typing import Union, List, Tuple, Dict, Any
from .type_alias import ScaleStrategy, ImputeStrategy


def scale(
    df: Union[pl.DataFrame, pl.LazyFrame],
    cols: List[str],
    *,
    strategy: ScaleStrategy = "standard",
    const: float = 1.0,
    qcuts: Tuple[float, float, float] = (0.25, 0.5, 0.75),
) -> List[pl.Expr]:
    """
    Scale the given columns with the given strategy.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        The columns to scale
    strategy
        One of `standard`, `min_max`, `const`, `mean`, `median`, `abs_max` or `robust`. If 'const',
        the const argument should be provided. If `robust`, the qcuts argument should be provided.
    const
        The constant value to scale by if strategy = 'const'
    qcuts
        The quantiles used in robust scaling. Must be three increasing numbers between 0 and 1. The formula is
        (X - qcuts[1]) / (qcuts[2] - qcuts[0]) for each column X.
    """
    if strategy == "standard":
        mean_std = (
            df.lazy()
            .select(
                pl.col(cols).mean().name.prefix("mean:"), pl.col(cols).std().name.prefix("std:")
            )
            .collect()
            .row(0)
        )
        exprs = [(pl.col(c) - mean_std[i]) / (mean_std[i + len(cols)]) for i, c in enumerate(cols)]
    elif strategy == "min_max":
        min_max = (
            df.lazy()
            .select(pl.col(cols).min().name.prefix("min:"), pl.col(cols).max().name.prefix("max:"))
            .collect()
            .row(0)
        )  # All mins come first, then maxs
        exprs = [
            (pl.col(c) - min_max[i]) / (min_max[i + len(cols)] - min_max[i])
            for i, c in enumerate(cols)
        ]
    elif strategy == "robust":
        quantiles = (
            df.lazy()
            .select(
                pl.col(cols).quantile(qcuts[0]).name.suffix("_1"),
                pl.col(cols).quantile(qcuts[1]).name.suffix("_2"),
                pl.col(cols).quantile(qcuts[2]).name.suffix("_3"),
            )
            .collect()
            .row(0)
        )
        exprs = [
            (pl.col(c) - quantiles[len(cols) + i]) / (quantiles[2 * len(cols) + i] - quantiles[i])
            for i, c in enumerate(cols)
        ]
    elif strategy == "max_abs":
        max_abs = (
            df.lazy()
            .select(
                pl.max_horizontal(pl.col(c).min().abs(), pl.col(c).max().abs()).name.prefix(
                    "absmax:"
                )
                for c in cols
            )
            .collect()
            .row(0)
        )
        exprs = [pl.col(c) / max_abs[i] for i, c in enumerate(cols)]
    elif strategy == "mean":
        mean = (
            df.lazy()
            .select(
                pl.col(cols).mean().name.prefix("mean:"),
            )
            .collect()
            .row(0)
        )
        exprs = [(pl.col(c) / m) for c, m in zip(cols, mean)]
    elif strategy == "median":
        med = (
            df.lazy()
            .select(
                pl.col(cols).median().name.prefix("median:"),
            )
            .collect()
            .row(0)
        )
        exprs = [(pl.col(c) / m) for c, m in zip(cols, med)]
    elif strategy in ("const", "constant"):
        exprs = [pl.col(cols) / const]
    else:
        raise TypeError(f"Unknown scaling strategy: {strategy}")

    return exprs


def impute(
    df: Union[pl.DataFrame, pl.LazyFrame],
    cols: list[str],
    *,
    strategy: ImputeStrategy = "median",
    const: float = 0.0,
) -> List[pl.Expr]:
    """
    Impute the given columns with the given strategy.

    Parameters
    ----------
    df
        Either a lazy or eager Polars DataFrame
    cols
        The columns to impute. Please make sure the columns are either numeric or string columns. If a
        column is string, then only mode makes sense.
    strategy
        One of 'median', 'mean', 'const', 'mode'. If 'const', the const argument
        must be provided. Note that if strategy is mode and if two values occur the same number
        of times, a random one will be picked.
    const
        The constant value to impute by if strategy = 'const'
    """
    if strategy == "median":
        all_medians = df.lazy().select(pl.col(cols).median()).collect().row(0)
        exprs = [pl.col(c).fill_null(all_medians[i]) for i, c in enumerate(cols)]
    elif strategy == "mean":
        all_means = df.lazy().select(pl.col(cols).mean()).collect().row(0)
        exprs = [pl.col(c).fill_null(all_means[i]) for i, c in enumerate(cols)]
    elif strategy == "const":
        exprs = [pl.col(cols).fill_null(const)]
    elif strategy == "mode":
        all_modes = df.lazy().select(pl.col(cols).mode().first()).collect().row(0)
        exprs = [pl.col(c).fill_null(all_modes[i]) for i, c in enumerate(cols)]
    else:
        raise TypeError(f"Unknown imputation strategy: {strategy}")

    return exprs


def feature_mapping(
    df: Union[pl.DataFrame, pl.LazyFrame], *, mapping: Dict[str, Dict[Any, Any]]
) -> List[pl.Expr]:
    """
    Maps values of features according to mappings in mthe mapping dict. It will keep
    original values if the mapping dict does not specify mapping for some values.

    Paramters
    ---------
    df
        Either a lazy or eager Polars DataFrame
    mapping
        A dict, whose key means column name, whose value is the mapping dict for the column
    """
    return [
        pl.col(c).replace(old=pl.Series(val.keys()), new=pl.Series(val.values()))
        for c, val in mapping.items()
    ]


def null_mask(df: Union[pl.DataFrame, pl.LazyFrame], cols: List[str]) -> List[pl.Expr]:
    """
    Returns a null mask for the given columns.

    Paramters
    ---------
    df
        Either a lazy or eager Polars DataFrame
    cols
        The columns to mask
    """
    return [pl.col(c).is_null().name.suffix("_is_null") for c in cols]


def one_hot_encode(
    df: Union[pl.DataFrame, pl.LazyFrame],
    cols: List[str],
    *,
    drop_first: bool = True,
    separator: str = "::",
) -> List[pl.Expr]:
    """
    One hot encode the given columns. The columns must be string columns. Null value
    in the column will be dropped. If null indicator is needed, please use `null_mask`.
    Constant columns will be skipped.

    Paramters
    ---------
    df
        Either a lazy or eager Polars DataFrame
    cols
        The columns to one-hot encode
    drop_first
        Whether to drop the first value
    separator
        The separator in the new column name
    """
    uniques: pl.DataFrame = (
        df.lazy().select(pl.col(cols).unique().drop_nulls().sort().implode()).collect()
    )

    exprs: List[pl.Expr] = []
    start_index = int(drop_first)
    for col in uniques.get_columns():
        unique: pl.Series = col[0]
        if len(unique) > 1:
            exprs.extend(
                (pl.col(col.name) == unique[i])
                .fill_null(False)
                .cast(pl.UInt8)
                .alias(col.name + separator + unique[i])
                for i in range(start_index, len(unique))
            )

    return exprs


class PipeConstructor:
    def __init__(self, df: Union[pl.DataFrame, pl.LazyFrame], name: str = "project"):
        self._frame = df.lazy()
        self._name = name
        self.in_columns: List[str] = list(self._frame.columns)
        self.numerics: List[str] = df.select(cs.numeric()).columns
        self.ints: List[str] = df.select(cs.integer()).columns
        self.floats: List[str] = df.select(cs.float()).columns
        self.strs: List[str] = df.select(cs.string()).columns
        self.bools: List[str] = df.select(cs.boolean()).columns
        self.valid: List[str] = self.numerics + self.strs + self.bools
        self.invalid: List[str] = [c for c in self._frame.columns if c not in self.valid]

        #
        self.selected: List[str] = []
        self.steps: List[List[pl.Expr]] = []

    def lower_columns(self) -> List[pl.Expr]:
        return [pl.col(c).alias(c.lower()) for c in self._frame.columns]
