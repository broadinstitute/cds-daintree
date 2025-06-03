from .config import FILES
from pathlib import Path
import fsspec
import pandas as pd
import re
import os

def match_files_by_names(full_paths: list[str], filenames: list[str]):
    filenames_set = set(filenames)
    matches = [x for x in full_paths if os.path.basename(x) in filenames_set]
    missing = filenames_set.difference([os.path.basename(x) for x in matches])
    assert len(matches) == len(filenames_set) and len(missing) == 0, f"Could not find the expected filenames: matches={matches}, filenames_set={filenames_set}, missing={missing}, full_paths={full_paths}"
    return matches

def find_files(wildcard: str):
    fs, path = fsspec.url_to_fs(wildcard)
    # there's got to be a better way to do this...
    if wildcard.startswith("gs://"):
        prefix = "gs://"
    else:
        prefix = ""
    return [prefix+x for x in fs.glob(path)]


def gather(
    src_dir: str,
    dst_prefix: str,
    partitions_csv: str
):
    partitions = pd.read_csv(partitions_csv)

    csv_paths = find_files(f"{src_dir}/**/*.csv")
    ensemble_filenames = match_files_by_names(csv_paths, list(partitions["ensemble_filename"]))
    predictions_filenames = match_files_by_names(csv_paths, list(partitions["predictions_filename"]))

    df_ensemble = read_concatenated_csvs(ensemble_filenames ,  0)
    df_predictions = read_concatenated_csvs(predictions_filenames, 1)

    # Identify best performing model for each target variable
    df_ensemble = df_ensemble.copy()
    ranked_pearson = df_ensemble.groupby("target_variable")["pearson"].rank(
        ascending=False
    )
    df_ensemble["best"] = ranked_pearson == 1

    # Build final column list including feature information
    top_n = _get_max_feature_index(df_ensemble.columns) + 1
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


def read_concatenated_csvs(filenames : list[str], axis: int):
    """
    Read all csvs and return them as a concatenated pd.DataFrame
    """

    print(f"Reading {len(filenames)}...")

    dfs = [pd.read_csv(filename) for filename in filenames]
    return pd.concat(dfs, ignore_index=axis == 0, axis=axis)
