from pathlib import Path
import re
import pandas as pd
import csv
from dataclasses import dataclass
from typing import Optional, List
from . import config_manager
from . import data_processor
from .data_processor import Partition
from .config import DAINTREE_CORE_BIN_PATH


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
    tc, runner_config, feature_path_info, related_dset, *, test=False
):
    """Process feature information for all datasets and generate feature metadata.
    Args:
        runner_config: Input dictionary
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

    model_name = runner_config["model_name"]
    for dataset_name, dataset_metadata in runner_config["data"].items():
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
        runner_configs: Input dictionaries
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
    test: bool,
    restrict_targets_to: Optional[List[str]],
    runner_config_path,
    save_pref: Path,
    nfolds: int,
    models_per_task: int,
    test_first_n_tasks : Optional[int]
):
    runner_config = config_manager.load_runner_config(runner_config_path)

    # Setup and validate ensemble configuration
    core_config_path = config_manager.generate_core_config(save_pref, runner_config)

    core_config_dict = config_manager.load_and_validate_core_config(
        core_config_path, runner_config
    )

    print("Generating feature index and files...")
    feature_path_info = generate_feature_path_info(save_pref, runner_config["data"])

    out_rel, related_dset = config_manager.determine_relations(core_config_dict)

    feature_metadata_df = generate_feature_metadata(
        tc, runner_config, feature_path_info, related_dset, test=test
    )

    # Process dependency data
    df_dep = data_processor.process_dependency_data(
        tc, save_pref, runner_config, test=test, restrict_targets_to=restrict_targets_to
    )

    # Save feature matrix file path information
    feature_path_info.to_csv(save_pref / "feature_path_info.csv")
    feature_metadata_df.to_csv(save_pref / "feature_metadata.csv")

    # Prepare data
    data_processor.prepare_data(save_pref, out_rel, core_config_path)

    print("Partitioning inputs...")
    partitions = data_processor.partition_inputs(df_dep, core_config_dict, models_per_task)

    if test_first_n_tasks is not None:
        print(f"Limiting run to the first {test_first_n_tasks} tasks")
        partitions = partitions[:test_first_n_tasks]

    output_file = str(save_pref / "partitions.csv")
    _write_parameter_csv(output_file, partitions, core_config_path, nfolds)


def _write_parameter_csv(
    output_file: str, partitions: List[Partition], core_config_path: str, nfolds: int
):
    # The parameter file being generated is going to contain the command line to execute. So when
    # sparkles runs this it will be a csv file named something like "parameters.csv" and
    # sparkles will be executed as: sparkles run ... --params parameters.csv '{command}'

    with open(output_file, "wt") as fd:
        w = csv.writer(fd)
        w.writerow(["model_config", "start_index", "end_index", "model_name", "predictions_filename", "ensemble_filename"])
        for partition in partitions:
            w.writerow(
                [
                    # f"{DAINTREE_CORE_BIN_PATH} fit-model --x X.ftr --y target.ftr --model-config {core_config_path} --n-folds {nfolds} --target-range {partition.start_index} {partition.end_index} --model {partition.model_name}"
                    core_config_path, partition.start_index, partition.end_index, partition.model_name,
                    f"{partition.model_name}_{partition.start_index}_{partition.end_index}_predictions.csv", f"{partition.model_name}_{partition.start_index}_{partition.end_index}_features.csv"
                ]
            )
