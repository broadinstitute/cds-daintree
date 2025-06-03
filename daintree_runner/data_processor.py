import pandas as pd
from .config import TEST_LIMIT, FILES
import re
from pathlib import Path
import os
import subprocess
import numpy as np
from .config import DAINTREE_CORE_BIN_PATH
from glob import glob
from dataclasses import dataclass


@dataclass
class Partition:
    start_index: int
    end_index: int
    model_name: str


def _clean_dataframe(df: pd.DataFrame, index_col):
    """Clean and sort the dataframe.
    Args:
        df: DataFrame to clean and sort
        index_col: Index column number
    Returns:
        pd.DataFrame: Cleaned and sorted dataframe
    """
    df.sort_index(inplace=True, axis=1)

    if index_col is None:
        df.sort_values(df.columns.tolist(), inplace=True)
    else:
        df.sort_index(inplace=True)

    return df


def _process_model_correlations(model, partitions_df, targets_df):
    """Calculate correlation between model predictions and actual target values.
    Args:
        model: Name of the model whose predictions to process
        partitions_df: DataFrame containing paths to prediction files for each partition
        targets_df: DataFrame containing actual target values
    Returns:
        pd.DataFrame: DataFrame containing Pearson correlations for each target variable
                        with columns: [target_variable, pearson, model]
    """
    # Get paths to prediction files for this specific model
    predictions_filenames = partitions_df[partitions_df["model"] == model][
        "predictions_path"
    ]

    # Combine predictions from all partitions into a single DataFrame
    # Each file contains predictions for a subset of targets
    predictions = pd.DataFrame().join(
        [pd.read_csv(f, index_col=0) for f in predictions_filenames], how="outer"
    )

    # Calculate Pearson correlation between predictions and actual values
    cors = predictions.corrwith(targets_df)

    cors = (
        pd.DataFrame(cors)
        .reset_index()
        .rename(
            columns={
                "index": "target_variable",  # Name of the target variable
                0: "pearson",  # Correlation coefficient
            }
        )
    )

    cors["model"] = model

    return cors


def process_biomarker_matrix(df: pd.DataFrame, index_col: int = 0, test: bool = False):
    """Process biomarker matrix data.
    Args:
        df: Biomarker matrix dataframe
        index_col: Index column number
        test: Test flag
    Returns:
        pd.DataFrame: Processed biomarker matrix
    """
    print("Start Processing Biomarker Matrix")
    df = _clean_dataframe(df, index_col)
    # if test:
    #     df = df.iloc[:, :TEST_LIMIT]
    print("End Processing Biomarker Matrix")

    return df


def _process_dep_matrix(df: pd.DataFrame, test=False, restrict_targets_to=None):
    """Process dependency matrix data.
    Args:
        df: Dependency matrix dataframe
        test: Test flag
        restrict_targets_to: Comma Separated list of targets to restrict the matrix to
    Returns:
        pd.DataFrame: Processed dependency matrix
    """
    print("Start Processing Dependency Matrix")
    df = df.dropna(how="all", axis=0)
    df = df.dropna(how="all", axis=1)
    df.index.name = "Row.name"
    df = df.reset_index()

    if test:
        print("\033[93mWarning: Truncating datasets for testing...\033[0m")  # Yellow
        # If no specific targets, apply column filtering
        if restrict_targets_to:
            restrict_targets_to.insert(0, "Row.name")
            pattern = (
                r"\b("
                + "|".join(re.escape(col) for col in restrict_targets_to)
                + r")\b"
            )
            mask = df.columns.str.contains(pattern, regex=True)
            df = df.loc[:, mask]
        else:
            # If no filter columns provided, take first TEST_LIMIT+1 columns
            # (+1 because first column is Row.name)
            df = df.iloc[:, : TEST_LIMIT + 1]
    else:
        print(
            "\033[93mWarning: Not truncating datasets. This may take a while...\033[0m"
        )  # Yellow

    print("End Processing Dependency Matrix")

    return df


def process_dependency_data(
    tc, save_pref, runner_config, *, test=False, restrict_targets_to=None
):
    """Process dependency matrix data from Taiga and prepare it for model training.

    Args:
        runner_config: Dictionary containing input configuration with dataset metadata
        test: If True, limits data size for testing purposes
        restrict_targets_to: List of column names to filter the target matrix by

    Returns:
        pd.DataFrame: Processed dependency matrix ready for model training
    """
    print("Processing dependency data...")

    # Find the Taiga ID for the target matrix by looking through input dictionary
    # for the first entry with table_type="target_matrix"
    dep_matrix_taiga_id = next(
        (
            v.get("taiga_id")
            for v in runner_config["data"].values()
            if v.get("table_type") == "target_matrix"
        ),
        None,
    )

    df_dep = tc.get(dep_matrix_taiga_id)

    df_dep = _process_dep_matrix(df_dep, test, restrict_targets_to=restrict_targets_to)

    print("\033[92m================================================")  # Green
    print(f"Processed Target Matrix")
    print("================================================\033[0m")
    print(df_dep.head())

    # Save the processed matrix
    df_dep.to_csv(save_pref / FILES["target_matrix"])
    df_dep.to_feather(save_pref / FILES["target_matrix"])

    return df_dep


def prepare_data(save_pref: Path, out_rel, ensemble_config):
    """Prepare data for model fitting."""
    target_matrix = save_pref / FILES["target_matrix"]
    target_matrix_filtered = save_pref / FILES["target_matrix_filtered"]

    # Note that these parameters are from daintree_package or cds-ensemble
    print('Running "prepare-y"...')
    try:
        subprocess.check_call(
            [
                DAINTREE_CORE_BIN_PATH,
                "prepare-y",
                "--input",
                str(target_matrix),
                "--output",
                str(target_matrix_filtered),
            ]
        )
    except subprocess.CalledProcessError as e:
        print(f"Error preparing target data: {e}")
        raise

    # Note that these parameters are from daintree_package or cds-ensemble
    print('Running "prepare-x"...')
    prep_x_cmd = [
        DAINTREE_CORE_BIN_PATH,
        "prepare-x",
        "--model-config",
        str(ensemble_config),
        "--targets",
        str(target_matrix_filtered),
        "--feature-info",
        str(save_pref / FILES["feature_path_info"]),
        "--output",
        str(save_pref / FILES["feature_matrix"]),
    ]

    if out_rel:
        prep_x_cmd.extend(["--output-related", "related"])

    try:
        subprocess.check_call(prep_x_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error preparing feature data: {e}")
        raise


def partition_inputs(dep_matrix, ensemble_config, models_per_task):
    """
    Divides the dependency matrix columns (genes) into chunks for parallel processing.
    For each model in the ensemble, it creates partitions based on the number of jobs specified
    in the model config.
    Args:
        dep_matrix: DataFrame containing the dependency matrix
        ensemble_config: Dictionary containing configuration for each model in the ensemble
                        Each model config must have a "Jobs" key specifying number of partitions
                        So if a model specifies 10 jobs and there are 100 genes:
                            - Creates 10 partitions of ~10 genes each
                            - First 9 partitions will have exactly 10 genes
                            - Last partition will have remaining genes (also 10 in this case)
    """
    # Get total number of genes (columns) to partition
    num_genes = dep_matrix.shape[1]
    partitions: list[Partition] = []

    for model_name, model_config in ensemble_config.items():
        # Note: the parameter is called "jobs" but it's treated here as "the number of genes to process per job"
        # I'm not going to change that now, but I think this parameter name is misleading.

        num_jobs = models_per_task
        start_indices = np.array(range(0, num_genes, num_jobs))
        end_indices = start_indices + num_jobs

        # Ensure the last partition includes any remaining genes
        end_indices[-1] = num_genes

        # Store partition boundaries and model names
        for start_index, end_index in zip(start_indices, end_indices):
            partitions.append(
                Partition(
                    start_index=start_index, end_index=end_index, model_name=model_name
                )
            )

    return partitions
