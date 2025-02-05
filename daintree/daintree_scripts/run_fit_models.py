import json
import pandas as pd
from taigapy import create_taiga_client_v3
import datetime as dt
import yaml
import subprocess
import os
from pathlib import Path
import click

from utils import (
    generate_config,
    generate_feature_info,
    partiton_inputs,
    process_dep_matrix,
    process_biomarker_matrix,
    process_column_name,
    create_output_config
)
from scipy.stats import pearsonr
from taiga_utils import update_taiga
from daintree_package import main

@click.group()
def cli():
    pass


def _build_sparkles_command(save_pref, config_fname, related, dt_hash, sparkles_config):
    """Build the sparkles command with all required arguments."""
    
    # Ensure config_fname is just the filename, not the full path
    config_file = Path(config_fname).name
    
    cmd = [
        "/install/sparkles/bin/sparkles",
        "--config", sparkles_config,
        "sub",
        "-i", "us.gcr.io/broad-achilles/daintree-sparkles:v1",
        "-u", main.__file__,
        "-u", f"{save_pref}/target_matrix.ftr:target.ftr",
        "-u", f"{save_pref}/{config_file}:model-config.yaml",
        "-u", f"{save_pref}/X.ftr:X.ftr",
        "-u", f"{save_pref}/X_feature_metadata.ftr:X_feature_metadata.ftr",
        "-u", f"{save_pref}/X_valid_samples.ftr:X_valid_samples.ftr",
        "-u", f"{save_pref}/partitions.csv",
        "--params", f"{save_pref}/partitions.csv",
        "--skipifexists",
        "--nodes", "100",
        "-n", f"ensemble_{dt_hash}",
        "/install/depmap-py/bin/daintree", "fit-model",
        "--x", "X.ftr",
        "--y", "target.ftr",
        "--model-config", "model-config.yaml",
        "--n-folds", "5",
        "--target-range", "{start}", "{end}",
        "--model", "{model}"
    ]

    if related:
        cmd.extend([
            "-u", f"{save_pref}/related.ftr:related.ftr",
            "--related-table", "related.ftr"
        ])

    return cmd


def fit_with_sparkles(config_fname, related, sparkles_config, save_pref):
    """Run model fitting using sparkles."""
    dt_hash = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    
    cmd = _build_sparkles_command(save_pref, config_fname, related, dt_hash, sparkles_config)
    print(f"Running sparkles with command: {cmd}")
    
    subprocess.check_call(cmd)
    print("sparkles run complete")
    
    return dt_hash


def validate_run(dt_hash, sparkles_config, save_pref):
    """Validate the sparkles run."""

    print("Validating sparkles run...")
    sparkles_path = "/install/sparkles/bin/sparkles"
    
    # Watch and reset sparkles jobs
    subprocess.check_call([sparkles_path, "--config", sparkles_config, "watch", f"ensemble_{dt_hash}", "--loglive"])
    subprocess.check_call([sparkles_path, "--config", sparkles_config, "reset", f"ensemble_{dt_hash}"])
    subprocess.check_call([sparkles_path, "--config", sparkles_config, "watch", f"ensemble_{dt_hash}", "--loglive"])
    
    # Create output directory
    os.makedirs(f"{save_pref}/data", exist_ok=True)
    
    # Get default_url_prefix from sparkles config
    with open(sparkles_config, 'r') as f:
        for line in f:
            if 'default_url_prefix' in line:
                default_url_prefix = line.split('=')[1].strip()
                break
    
    # Authenticate and list files
    subprocess.check_call([
        "/google-cloud-sdk/bin/gcloud", 
        "auth", 
        "activate-service-account", 
        "--key-file", 
        "/root/.sparkles-cache/service-keys/broad-achilles.json"
    ])
    
    # List and save completed jobs
    completed_jobs = subprocess.check_output([
        "/google-cloud-sdk/bin/gcloud", 
        "storage", 
        "ls", 
        f"{default_url_prefix}/ensemble_{dt_hash}/*/*.csv"
    ]).decode()
    
    with open(f"{save_pref}/completed_jobs.txt", 'w') as f:
        f.write(completed_jobs)
    
    # Validate jobs
    subprocess.check_call([
        "/install/depmap-py/bin/python3.9",
        "../daintree_scripts/validate_jobs_complete.py",
        f"{save_pref}/completed_jobs.txt",
        f"{save_pref}/partitions.csv",
        "features.csv",
        "predictions.csv"
    ])
    
    # Copy results
    subprocess.check_call([
        "/google-cloud-sdk/bin/gcloud",
        "storage",
        "cp",
        f"{default_url_prefix}/ensemble_{dt_hash}/*/*.csv",
        f"{save_pref}/data"
    ])


def _load_and_prepare_data(save_pref, features, targets, partitions):
    """Load and prepare the basic data needed for ensemble tasks."""
    features_df = pd.read_feather(save_pref / features)
    targets_df = pd.read_feather(save_pref / targets)
    targets_df = targets_df.set_index("Row.name")
    partitions_df = pd.read_csv(save_pref / partitions)
    
    return features_df, targets_df, partitions_df

def _prepare_partition_paths(partitions_df, save_pref, data_dir, features_suffix, predictions_suffix):
    """Add file paths to the partitions dataframe."""
    partitions_df["path_prefix"] = (
        data_dir
        + "/"
        + partitions_df["model"]
        + "_"
        + partitions_df["start"].map(str)
        + "_"
        + partitions_df["end"].map(str)
        + "_"
    )
    partitions_df["feature_path"] = save_pref / (
        partitions_df["path_prefix"] + features_suffix
    )
    partitions_df["predictions_path"] = save_pref / (
        partitions_df["path_prefix"] + predictions_suffix
    )
    
    # Validate paths exist
    assert all(os.path.exists(f) for f in partitions_df["feature_path"])
    assert all(os.path.exists(f) for f in partitions_df["predictions_path"])
    
    return partitions_df

def _calculate_feature_correlations(x, y):
    """Calculate correlation between feature and target."""
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    mask = ~pd.isna(x) & ~pd.isna(y)
    
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    if len(x_filtered) > 1 and len(y_filtered) > 1:
        corr, _ = pearsonr(x_filtered, y_filtered)
        return corr
    return None

def _process_model_correlations(model, partitions_df, targets_df):
    """Calculate correlations for a specific model."""
    predictions_filenames = partitions_df[partitions_df["model"] == model]["predictions_path"]
    predictions = pd.DataFrame().join(
        [pd.read_csv(f, index_col=0) for f in predictions_filenames], 
        how="outer"
    )
    
    cors = predictions.corrwith(targets_df)
    cors = (
        pd.DataFrame(cors)
        .reset_index()
        .rename(columns={"index": "target_variable", 0: "pearson"})
    )
    cors["model"] = model
    
    return cors

def gather_ensemble_tasks(
    save_pref,
    features="X.ftr",
    targets="target_matrix.ftr",
    data_dir="data",
    partitions="partitions.csv",
    features_suffix="features.csv",
    predictions_suffix="predictions.csv",
    top_n=50,
):
    """Gather and process ensemble tasks with improved organization."""
    # Load basic data
    features_df, targets_df, partitions_df = _load_and_prepare_data(
        save_pref, features, targets, partitions
    )
    
    # Prepare partition paths
    partitions_df = _prepare_partition_paths(
        partitions_df, save_pref, data_dir, features_suffix, predictions_suffix
    )
    
    # Process features
    all_features = pd.concat(
        [pd.read_csv(f) for f in partitions_df["feature_path"]], 
        ignore_index=True
    )
    all_features.drop(["score0", "score1", "best"], axis=1, inplace=True)
    
    # Load and combine predictions
    predictions = pd.DataFrame().join(
        [pd.read_csv(f, index_col=0) for f in partitions_df["predictions_path"]], 
        how="outer"
    )
    
    # Calculate correlations for all models
    all_cors = []
    for model in all_features["model"].unique():
        cors = _process_model_correlations(model, partitions_df, targets_df)
        all_cors.append(cors)
    
    all_cors = pd.concat(all_cors, ignore_index=True)
    ensemble = all_features.merge(all_cors, on=["target_variable", "model"])
    
    # Calculate rankings and best models
    ensemble = ensemble.copy()
    ranked_pearson = ensemble.groupby("target_variable")["pearson"].rank(ascending=False)
    ensemble["best"] = (ranked_pearson == 1)
    
    # Calculate feature correlations
    ensb_cols = ["target_variable", "model", "pearson", "best"]
    
    for index, row in ensemble.iterrows():
        target_variable = row['target_variable']
        y = targets_df[target_variable]
        
        for i in range(top_n):
            feature_col = f'feature{i}'
            feature_name = row[feature_col]
            
            if feature_name in features_df.columns:
                corr = _calculate_feature_correlations(features_df[feature_name], y)
                ensemble.loc[index, f'{feature_col}_correlation'] = corr
    
    # Prepare final columns
    for i in range(top_n):
        feature_cols = [
            f"feature{i}", 
            f"feature{i}_importance",
            f"feature{i}_correlation"
        ]
        ensb_cols.extend(feature_cols)
    
    ensemble = ensemble.sort_values(["target_variable", "model"])[ensb_cols]
    
    return ensemble, predictions


def check_file_locs(ipt, config):
    ipt_features = list(ipt["data"].keys())
    
    for model_name, model_config in config.items():
        f_set = set(model_config.get("Features", []) + model_config.get("Required", []))
        
        if model_config.get("Relation") not in ["All", "MatchTarget"]:
            f_set.add(model_config.get("Related", ""))
        
        features = list(f_set)
        for f in features:
            assert f in ipt_features, f"Feature {f} in model config file does not have corresponding input in {model_name}"


def _upload_results_to_taiga(upload_to_taiga, save_pref, model_name, screen_name, ipt_dict):
    """Upload results to Taiga and create output config."""
    feature_metadata_filename = f"FeatureMetadata{model_name}{screen_name}.csv"
    ensemble_filename = f"Ensemble{model_name}{screen_name}.csv"
    predictions_filename = f"Predictions{model_name}{screen_name}.csv"

    # Upload feature metadata
    feature_metadata_taiga_info = update_taiga(
        upload_to_taiga,
        f"Updated Feature Metadata for Model: {model_name} and Screen: {screen_name}",
        f"FeatureMetadata{model_name}{screen_name}",
        save_pref / feature_metadata_filename,
        "csv_table",
    )
    print(f"Feature Metadata uploaded to Taiga: {feature_metadata_taiga_info}")

    # Upload ensemble results
    ensemble_taiga_info = update_taiga(
        upload_to_taiga,
        f"Updated Ensemble for Model: {model_name} and Screen: {screen_name}",
        f"Ensemble{model_name}{screen_name}",
        save_pref / ensemble_filename,
        "csv_table",
    )
    print(f"Ensemble uploaded to Taiga: {ensemble_taiga_info}")

    # Upload predictions
    predictions_taiga_info = update_taiga(
        upload_to_taiga,
        f"Updated Predictions for Model: {model_name} and Screen: {screen_name}",
        f"Predictions{model_name}{screen_name}",
        save_pref / predictions_filename,
        "csv_table",
    )
    print(f"Predictions uploaded to Taiga: {predictions_taiga_info}")

    # Create and write output config
    output_config = create_output_config(
        model_name=model_name,
        screen_name=screen_name,
        input_config=ipt_dict,
        feature_metadata_id=feature_metadata_taiga_info,
        ensemble_id=ensemble_taiga_info,
        prediction_matrix_id=predictions_taiga_info
    )

    # Save output config
    output_config_dir = save_pref / "output_config_files"
    output_config_dir.mkdir(parents=True, exist_ok=True)
    output_config_filename = f"OutputConfig{model_name}{screen_name}.json"
    output_config_file = output_config_dir / output_config_filename

    with open(output_config_file, 'w') as f:
        json.dump(output_config, f, indent=4)
    print(f"Created output config file: {output_config_file}")

    return feature_metadata_taiga_info, ensemble_taiga_info, predictions_taiga_info

def _collect_and_fit(
    input_files,
    ensemble_config,
    sparkles_config,
    save_dir=None,
    test=False,
    skipfit=False,
    upload_to_taiga=None,
    restrict_targets=False,
    restrict_to=None,
):
    tc = create_taiga_client_v3()
    # Use the provided save_dir directly without modification
    save_pref = Path(save_dir) if save_dir else Path.cwd()
    print(f"_collect_and_fit save directory: {save_pref}")
    save_pref.mkdir(parents=True, exist_ok=True)
    
    print("loading input files...")
    with open(input_files, "r") as f:
        ipt_dict = json.load(f)
    with open(ensemble_config, "r") as f:
        config_dict = yaml.load(f, yaml.SafeLoader)

    print("validating inputs...")
    # check that all datasets in the model config are represented in the input file dict
    check_file_locs(ipt_dict, config_dict)

    # make sure there is only one dependency dataset
    assert (
        len([v for v in ipt_dict["data"].values() if v.get("table_type") == "target_matrix"]) == 1
    ), "Exactly one dataset labeled 'target_matrix' is required"

    print("generating feature index and files...")
    # generate feature info table
    feature_info = generate_feature_info(ipt_dict, save_pref)

    print("processing relations...")
    # determine whether to provide output_related
    relations = [m[1]["Relation"] for m in config_dict.items()]
    out_rel = len(set(relations).difference(set(["All", "MatchTarget"]))) > 0
    # out_rel = "MatchRelated" in relations
    related_dset = None
    if out_rel:
        related_dset = list(set(relations).difference(set(["All", "MatchTarget"])))[0]
    print("generating feature info...")
    print("#######################")
    feature_info_df = pd.DataFrame(columns=["model", "feature_name", "feature_label", "given_id", "taiga_id", "dim_type"])
    model_name = ipt_dict["model_name"]
    screen_name = ipt_dict["screen_name"]

    ensemble_filename = f"Ensemble{model_name}{screen_name}.csv"
    feature_metadata_filename = f"FeatureMetadata{model_name}{screen_name}.csv"

    if test:
        print("and truncating datasets for testing...")
    for dataset_name, dataset_metadata in ipt_dict["data"].items():
        if dataset_metadata["table_type"] not in ["feature", "relation"]:
            continue
        _df = tc.get(dataset_metadata["taiga_id"])

        if (related_dset is None) or (
            (related_dset is not None) and dataset_name != related_dset
        ):
            _df = process_biomarker_matrix(_df, 0, test)
        print(f"processed dataset: {dataset_name}")
        print(_df.head())

        for col in _df.columns:
            feature_name, feature_label, given_id = process_column_name(col, dataset_name)
            new_row = pd.DataFrame({
                "model": [model_name],
                "feature_name": [feature_name],
                "feature_label": [feature_label],
                "given_id": [given_id],
                "taiga_id": [dataset_metadata["taiga_id"]],
                "dim_type": [dataset_metadata["dim_type"]]
            })
            feature_info_df = pd.concat([feature_info_df, new_row], ignore_index=True)
        _df.to_csv(feature_info.set_index("dataset").loc[dataset_name].filename)
    
    print("feature info generated")
    print("#######################")

    print("processing dependency data...")
    # load, process, and save dependency matrix
    dep_matrix_taiga_id = next((v.get("taiga_id") for v in ipt_dict["data"].values() if v.get("table_type") == "target_matrix"), None)
    print(f"dep_matrix_taiga_id: {dep_matrix_taiga_id}")
    df_dep = tc.get(dep_matrix_taiga_id)
    df_dep = process_dep_matrix(df_dep, test, restrict_targets, restrict_to)
    df_dep.to_feather(save_pref / "target_matrix.ftr")

    # generate feature metadata file
    feature_info.to_csv(save_pref / "feature_info.csv")
    feature_info_df.to_csv(save_pref / feature_metadata_filename)

    print('running "prepare_y"...')
    subprocess.check_call(
        [
            "/install/depmap-py/bin/daintree", 
            "prepare-y",
            "--input",
            str(save_pref / "target_matrix.ftr"),
            "--output",
            str(save_pref / "target_matrix_filtered.ftr"),
        ]
    )
    # prepare_y(input="target_matrix.ftr", output="target_matrix_filtered.ftr")
    # pytest.set_trace()
    print('running "prepare_x"...')
    prep_x_cmd = [
        "/install/depmap-py/bin/daintree",
        "prepare-x",
        "--model-config",
        str(ensemble_config),
        "--targets",
        str(save_pref / "target_matrix_filtered.ftr"),
        "--feature-info",
        str(save_pref / "feature_info.csv"),
        "--output",
        str(save_pref / "X.ftr"),
    ]
    if out_rel:
        prep_x_cmd.extend(["--output-related", "related"])
    subprocess.check_call(prep_x_cmd)

    print("partitioning inputs...")
    partiton_inputs(df_dep, config_dict, save_pref, out_name="partitions.csv")

    if not skipfit:
        print("submitting fit jobs...")

        dt_hash = fit_with_sparkles(
            ensemble_config, out_rel, sparkles_config, save_pref
        )

        # watch sparkles run, resubmit failed jobs, and validate when complete
        validate_run(dt_hash, sparkles_config, save_pref)
        
        # gather the ensemble results and collect the top features
        df_ensemble, df_predictions = gather_ensemble_tasks(
            save_pref, features=str(save_pref / "X.ftr"), targets=str(save_pref / "target_matrix.ftr"), top_n=50
        )
        df_ensemble.to_csv(save_pref / ensemble_filename, index=False)
        predictions_filename = f"Predictions{model_name}{screen_name}.csv"
        df_predictions.to_csv(save_pref / predictions_filename)

        if upload_to_taiga:
            _upload_results_to_taiga(
                upload_to_taiga,
                save_pref,
                ipt_dict["model_name"],
                ipt_dict["screen_name"],
                ipt_dict
            )
    else:
        print("skipping fitting and ending run")


@cli.command()
@click.option(
    "--input-files",
    required=True,
    help="JSON file containing the set of files for prediction",
)
@click.option(
    "--sparkles-config", 
    required=True, 
    help="path to the sparkles config file to use"
)
@click.option(
    "--save-dir",
    required=False,
    help="Path to where the data should be stored if not the same directory as the script",
)
@click.option(
    "--test",
    default=False,
    type=bool,
    help="Run a test version with limited data",
)
@click.option(
    "--skipfit",
    default=False,
    type=bool,
    help="Only prepare files without fitting (for testing)",
)
@click.option(
    "--upload-to-taiga",
    default=None,
    type=str,
    help="Upload results to Taiga",
)
@click.option(
    "--restrict-targets",
    default=False,
    type=bool,
    help="Restrict dependencies to a user provided subset",
)
@click.option(
    "--restrict-to",
    required=False,
    type=str,
    default=None,
    help="""if "restrict" is true this provides a list of dependencies to restrict analysis to, separated by semicolons (;). If "restrict" and this is left empty it will not run. """,
)
def collect_and_fit_generate_config(
    input_files,
    sparkles_config,
    save_dir=None,
    test=False,
    skipfit=False,
    upload_to_taiga=None,
    restrict_targets=False,
    restrict_to=None,
):
    """Run model fitting with auto-generated config."""
    save_pref = Path(save_dir) if save_dir else Path.cwd()
    print(f"Save directory path: {save_pref}")
    save_pref.mkdir(parents=True, exist_ok=True)

    # Generate config
    with open(input_files, "r") as f:
        ipt_dict = json.load(f)
    config = generate_config(ipt_dict, relation="All")
    model_config_name = f"model-config_temp_{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.yaml"
    config_path = save_pref / model_config_name
    print(f"Config file path: {config_path}")

    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=True)

    # Run the model fitting
    _collect_and_fit(
        input_files,
        str(config_path),
        sparkles_config,
        save_dir=str(save_pref),
        test=test,
        skipfit=skipfit,
        upload_to_taiga=upload_to_taiga,
        restrict_targets=restrict_targets,
        restrict_to=restrict_to,
    )


@cli.command()
@click.option(
    "--input-files",
    required=True,
    help="JSON file containing the set of files for prediction",
)
@click.option(
    "--ensemble-config",
    required=True,
    help='YAML file for model configuration. Names of datasets must match those provided in "input-files"',
)
@click.option(
    "--sparkles-config", required=True, help="path to the sparkles config file to use"
)
@click.option(
    "--test",
    is_flag=False,
    help="Run a test version, using only five gene dependencies and five features from each dataset",
)
@click.option(
    "--skipfit",
    is_flag=False,
    help="Do not execute the fitting, only prepare files. Used for testing",
)
@click.option(
    "--upload-to-taiga",
    is_flag=True,
    help="Upload the Y matrix and the feature metadata to Taiga",
)
def collect_and_fit(
    input_files,
    ensemble_config,
    sparkles_config,
    test=False,
    skipfit=False,
    upload_to_taiga=None,
):

    _collect_and_fit(
        input_files,
        ensemble_config,
        sparkles_config,
        save_dir=None,
        test=test,
        skipfit=skipfit,
        upload_to_taiga=upload_to_taiga,
    )


if __name__ == "__main__":
    cli()
