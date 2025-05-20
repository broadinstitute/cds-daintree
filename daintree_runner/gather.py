from .config import FILES
from pathlib import Path
from glob import glob
import pandas as pd
import re

def gather(
        src_dir: str,
        dst_prefix : str, 
):
    df_ensemble = read_concatenated_csvs(f"{src_dir}/**/*_features.csv", 0)
    df_predictions = read_concatenated_csvs(f"{src_dir}/**/*_predictions.csv", 1)

    # Identify best performing model for each target variable
    df_ensemble = df_ensemble.copy()
    ranked_pearson = df_ensemble.groupby("target_variable")["pearson"].rank(
        ascending=False
    )
    df_ensemble["best"] = ranked_pearson == 1

    # Build final column list including feature information
    top_n = _get_max_feature_index(df_ensemble.columns)+1
    ensb_cols = ["target_variable", "model", "pearson", "best"]
    for i in range(top_n):
        feature_cols = [
            f"feature{i}",  # Feature name
            f"feature{i}_importance",  # Feature importance score
            f"feature{i}_correlation",  # Feature correlation with target
        ]
        ensb_cols.extend(feature_cols)

    # Sort and select final columns
    df_ensemble = df_ensemble.sort_values(["target_variable", "model"])[ensb_cols]

    ensemble_filename = dst_prefix + "ensemble.csv"
    predictions_filename = dst_prefix + "predictions.csv"

    print(f"Writing merged {ensemble_filename} and {predictions_filename}")
    df_ensemble.to_csv(ensemble_filename, index=False)
    df_predictions.to_csv(predictions_filename, index=False)

def _get_max_feature_index(column_names):
    values = []
    for column_name in column_names:
        m = re.match("feature(\\d+)$", column_name)
        if m:
            values.append(int(m.group(1)))
    return max(values)

def read_concatenated_csvs(
    wildcard: str,
    axis: int
):
    """
    Read all csvs that match wildcard and return them as a concatenated pd.DataFrame
    """

    filenames = glob(wildcard, recursive=True)
    print(f"Reading {len(filenames)} matching {wildcard}...")
    dfs = [pd.read_csv(filename) for filename in filenames]
    return pd.concat(dfs, ignore_index=axis==0, axis=axis)

