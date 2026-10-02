from typing import List, Optional

import pandas as pd
import re

def _matches_one_of(name: str, patterns: list[str]):
    for pattern in patterns:
        if re.match(pattern, name):
            return True

    return False

def prepare_targets(
    df: pd.DataFrame,
    top_variance_filter: Optional[int],
    column_filter: Optional[List[str]],
) -> pd.DataFrame:
    if column_filter is not None:
        matching_column_names = [name for name in df.columns if _matches_one_of(name, column_filter)]
        df_ = df.loc[:, matching_column_names]
        assert isinstance(df_, pd.DataFrame) # I do wish pyright didn't get so confused by pandas 
        df = df_.copy()

    if top_variance_filter is not None:
        df = df.filter(
            items=df.var(axis=0)
            .sort_values(ascending=False)[:top_variance_filter]
            .index,
            axis="columns",
        )

    df.index.name = "Row.name"

    return df
