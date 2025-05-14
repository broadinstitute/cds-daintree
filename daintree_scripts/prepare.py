from pathlib import Path
import re
import pandas as pd
import csv
from .prepare import _process_column_name
from dataclasses import dataclass
from typing import Optional, List
from . import config_manager
from . import data_processor


def _process_column_name(col, feature_dataset_name):
    """Process column name to generate feature name, label, and given id for breadbox.
    Args:
        col: Column name
        feature_dataset_name: Feature dataset name
    Returns:
        feature_name: Feature name
        feature_label: Feature label
        given_id: Given ID
    """
    # Checking if the column name is in the format "feature_label (given_id)" e.g. A1BG (1)
    match = re.match(r"(.+?) \((\d+)\)", col)
    if match:
        feature_label, given_id = match.groups()
        feature_name = (
            f"{feature_label.replace('-', '_')}_({given_id})_{feature_dataset_name}"
        )
    else:
        feature_label = col
        given_id = col
        feature_name = re.sub(r"[\s-]+", "_", col) + f"_{feature_dataset_name}"
    return feature_name, feature_label, given_id


def process_dataset_for_feature_metadata(
    dataset_metadata_df,
    dataset_name,
    dataset_metadata,
    model_name,
    related_dset,
    test=False,
):
    """Process a single dataset and generate feature metadata.
    Args:
        dataset_name: Name of the dataset
        dataset_metadata: Metadata for the dataset
        model_name: Name of the model
        related_dset: Related dataset
        test: Test flag
    Returns:
        pd.DataFrame: Downloaded feature matrix from taiga
        pd.DataFrame: Single Dataset Feature metadata
    """
    feature_metadata_rows = []
    _df = dataset_metadata_df  # self.tc.get(dataset_metadata["taiga_id"])

    if (related_dset is None) or (
        (related_dset is not None) and dataset_name != related_dset
    ):
        _df = data_processor.process_biomarker_matrix(_df, 0, test)
    print("\033[92m================================================")  # Green
    print(f"Processed Feature Dataset: {dataset_name}")
    print("================================================\033[0m")
    print(_df.head())

    for col in _df.columns:
        feature_name, feature_label, given_id = _process_column_name(col, dataset_name)
        feature_metadata_rows.append(
            {
                "model": model_name,
                "feature_name": feature_name,
                "feature_label": feature_label,
                "given_id": given_id,
                "taiga_id": dataset_metadata["taiga_id"],
                "dim_type": dataset_metadata["dim_type"],
            }
        )

    return _df, pd.DataFrame(feature_metadata_rows)


def generate_feature_metadata(
    tc, ipt_dict, feature_path_info, related_dset, test=False
):
    """Process feature information for all datasets and generate feature metadata.
    Args:
        ipt_dict: Input dictionary
        feature_path_info: DF with feature dataset names and their corresponding file paths
        related_dset: Related dataset
        test: Test flag
    Returns:
        pd.DataFrame: Concatenated feature metadata for all datasets
    """
    print("Generating feature metadata...")
    feature_metadata_df = pd.DataFrame(
        columns=[
            "model",
            "feature_name",
            "feature_label",
            "given_id",
            "taiga_id",
            "dim_type",
        ]
    )

    model_name = ipt_dict["model_name"]
    for dataset_name, dataset_metadata in ipt_dict["data"].items():
        if dataset_metadata["table_type"] not in ["feature", "relation"]:
            continue

        dataset_metadata_df = tc.get(dataset_metadata["taiga_id"])
        _df, dataset_info = process_dataset_for_feature_metadata(
            dataset_metadata_df,
            dataset_name,
            dataset_metadata,
            model_name,
            related_dset,
            test,
        )
        # Concatenate the feature metadata for all datasets
        feature_metadata_df = pd.concat(
            [feature_metadata_df, dataset_info], ignore_index=True
        )
        # Saving the downloaded feature matrix to a csv file
        _df.to_csv(feature_path_info.set_index("dataset").loc[dataset_name].filename)

    return feature_metadata_df


def generate_feature_path_info(save_pref, data: dict[str, dict]):
    """Generate feature path information.
    Args:
        ipt_dicts: Input dictionaries
    Returns:
        pd.DataFrame: DF with feature dataset names and their corresponding file paths
    """
    dsets = []
    for dset_name, dset_value in data.items():
        if dset_value["table_type"] == "feature":
            dsets.append(dset_name)
    fnames = [str(save_pref / (dset + ".csv")) for dset in dsets]

    df = pd.DataFrame(
        {
            "dataset": dsets,
            "filename": fnames,
        }
    )

    return df


def prepare(
    tc,
    ensemble_config,
    test: bool,
    restrict_targets_to: Optional[List[str]],
    input_config,
    save_pref: Path,
    nfolds: int
):
    ipt_dict = config_manager.load_input_config(input_config)

    # Setup and validate ensemble configuration
    config_path, config_dict = config_manager.setup_ensemble_config(
        save_pref, ensemble_config, ipt_dict
    )

    print("Generating feature index and files...")
    feature_path_info = generate_feature_path_info(save_pref, ipt_dict["data"])

    out_rel, related_dset = config_manager.determine_relations(config_dict)

    model_name = ipt_dict["model_name"]
    screen_name = ipt_dict["screen_name"]

    # This could probably be hardcoded and put in a config file.
    # However, I am keeping it this way for now to make it more flexible.
    feature_metadata_filename = f"FeatureMetadata{model_name}{screen_name}.csv"
    feature_metadata_df = generate_feature_metadata(
        ipt_dict, feature_path_info, related_dset, test
    )

    # Process dependency data
    df_dep = data_processor.process_dependency_data(
        tc, ipt_dict, test, restrict_targets_to=restrict_targets_to
    )

    # Save feature matrix file path information
    feature_path_info.to_csv(save_pref / "feature_path_info.csv")
    feature_metadata_df.to_csv(save_pref / feature_metadata_filename)

    # Prepare data
    data_processor.prepare_data(save_pref, out_rel, config_path)

    print("Partitioning inputs...")
    partitions = data_processor.partition_inputs(df_dep, config_dict)

    output_file = str(save_pref / "partitions.csv")
    _write_parameter_csv(output_file, partitions, nfolds)

from .data_processor import Partition
from .config import DAINTREE_BIN_PATH

def _write_parameter_csv(output_file: str, partitions: List[Partition], nfolds: int):
    with open(output_file, "wt") as fd:
        w = csv.writer(fd)
        w.writerow(["command"])
        for partition in partitions:
            w.writerow(
                [
                    f"{DAINTREE_BIN_PATH} fit-model --x X.ftr --y target.ftr --model-config model-config.yaml --n-folds {nfolds} --target-range {partition.start_index} {partition.end_index} --model {partition.model_name}"
                ]
            )
