import json
import pandas as pd
from taigapy import create_taiga_client_v3
import datetime as dt
import yaml
import subprocess
import os
from pathlib import Path
from string import Template
import click

from utils import (
    generate_config,
    generate_feature_info,
    partiton_inputs,
    process_dep_matrix,
    process_biomarker_matrix,
    process_column_name,
)
from scipy.stats import pearsonr
from taiga_utils import update_taiga


@click.group()
def cli():
    pass



def fit_with_sparkles(config_fname, related, sparkles_config, save_pref):
    dt_hash = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    cmd = []
    cmd.append("/install/sparkles/bin/sparkles")
    cmd.extend(["--config", sparkles_config])
    cmd.append("sub")
    cmd.extend(
        ["-i", "us.gcr.io/broad-achilles/daintree-sparkles:v6"]
    )
    cmd.extend(["-u", "/daintree/daintree_package/daintree_package/main.py"])
    cmd.extend(["-u", str(save_pref / "target_matrix.ftr") + ":target.ftr"])
    cmd.extend(["-u", str(save_pref / config_fname) + ":model-config.yaml"])
    cmd.extend(["-u", str(save_pref / "X.ftr") + ":X.ftr"])
    if related:
        cmd.extend(["-u", str(save_pref / "related.ftr") + ":related.ftr"])
    cmd.extend(
        ["-u", str(save_pref / "X_feature_metadata.ftr") + ":X_feature_metadata.ftr"]
    )
    cmd.extend(["-u", str(save_pref / "X_valid_samples.ftr") + ":X_valid_samples.ftr"])
    cmd.extend(["-u", str(save_pref / "partitions.csv")])
    cmd.extend(["--params", str(save_pref / "partitions.csv")])
    cmd.append("--skipifexists")
    cmd.extend(["--nodes", "100"])
    cmd.extend(["-n", "ensemble_" + dt_hash])
    cmd.extend(["/install/depmap-py/bin/daintree", "fit-model"])
    cmd.extend(["--x", "X.ftr"])
    cmd.extend(["--y", "target.ftr"])
    cmd.extend(["--model-config", "model-config.yaml"])
    cmd.extend(["--n-folds", "5"])
    if related:
        cmd.extend(["--related-table", "related.ftr"])
    cmd.extend(["--feature-metadata", "X_feature_metadata.ftr"])
    cmd.extend(["--model-valid-samples", "X_valid_samples.ftr"])
    cmd.extend(["--target-range", "{start}", "{end}"])
    cmd.extend(["--model", "{model}"])
    print(f"Running sparkles with command: {cmd}")
    print(cmd)
    subprocess.check_call(cmd)
    print("sparkles run complete")
    return dt_hash


validate_str = Template(
    """set -ex
$sparkles_path --config $sparkles_config watch ensemble_$HASH --loglive
$sparkles_path --config $sparkles_config reset ensemble_$HASH 
$sparkles_path --config $sparkles_config watch ensemble_$HASH --loglive
mkdir -p $save_pref/data
default_url_prefix=$(awk -F "=" '/default_url_prefix/ {print $2}' "$sparkles_config")
/google-cloud-sdk/bin/gcloud auth activate-service-account --key-file /root/.sparkles-cache/service-keys/broad-achilles.json
/google-cloud-sdk/bin/gcloud storage ls ${default_url_prefix}/ensemble_$HASH/*/*.csv > $save_pref/completed_jobs.txt
/install/depmap-py/bin/python3.9 validate_jobs_complete.py $save_pref/completed_jobs.txt $save_pref/partitions.csv features.csv predictions.csv
/google-cloud-sdk/bin/gcloud storage cp ${default_url_prefix}/ensemble_$HASH/*/*.csv $save_pref/data
"""
)


def validate_run(dt_hash, sparkles_config, save_pref):
    validate_cmd = validate_str.safe_substitute(
        {
            "sparkles_config": sparkles_config,
            "sparkles_path": "/install/sparkles/bin/sparkles",
            "HASH": dt_hash,
            "save_pref": save_pref,
        }
    )
    with open("validate.sh", "wt") as f:
        f.write(validate_cmd)
        f.close()

    # subprocess.check_call(validate_cmd, shell=True)
    subprocess.check_call("bash validate.sh", shell=True)


def save_and_run_bash(cmd_template, sub_dict):
    bash_cmd = cmd_template.safe_substitute(sub_dict)
    with open("bash_cmd.sh", "wt") as f:
        f.write(bash_cmd)
        f.close()

    # subprocess.check_call(validate_cmd, shell=True)
    subprocess.check_call("bash bash_cmd.sh", shell=True)


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
    features = pd.read_feather(save_pref / features)
    targets = pd.read_feather(save_pref / targets)
    targets = targets.set_index("Row.name")

    partitions = pd.read_csv(save_pref / partitions)
    partitions["path_prefix"] = (
        data_dir
        + "/"
        + partitions["model"]
        + "_"
        + partitions["start"].map(str)
        + "_"
        + partitions["end"].map(str)
        + "_"
    )
    partitions["feature_path"] = save_pref / (
        partitions["path_prefix"] + features_suffix
    )
    partitions["predictions_path"] = save_pref / (
        partitions["path_prefix"] + predictions_suffix
    )

    assert all(os.path.exists(f) for f in partitions["feature_path"])
    assert all(os.path.exists(f) for f in partitions["predictions_path"])

    all_features = pd.concat(
        [pd.read_csv(f) for f in partitions["feature_path"]], ignore_index=True
    )

    all_features.drop(["score0", "score1", "best"], axis=1, inplace=True)

    # Get pearson correlation of predictions by model
    all_cors = []
    for model in all_features["model"].unique():
        # Merge all files for model
        predictions_filenames = partitions[partitions["model"] == model][
            "predictions_path"
        ]
        predictions = pd.DataFrame().join(
            [pd.read_csv(f, index_col=0) for f in predictions_filenames], how="outer"
        )

        cors = predictions.corrwith(targets)

        # Reshape
        cors = (
            pd.DataFrame(cors)
            .reset_index()
            .rename(columns={"index": "target_variable", 0: "pearson"})
        )
        cors["model"] = model

        all_cors.append(cors)

    all_cors = pd.concat(all_cors, ignore_index=True)
    ensemble = all_features.merge(all_cors, on=["target_variable", "model"])

    ### The following code is implemented this way due to a
    # PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.
    # De-fragment the DataFrame
    ensemble = ensemble.copy()

    # Get the highest correlation across models per "target_variable" (entity)
    ranked_pearson = ensemble.groupby("target_variable")["pearson"].rank(ascending=False)
    ensemble["best"] = (ranked_pearson == 1)

    ensb_cols = ["target_variable", "model", "pearson", "best"]

    # Iterate over each row in the ensemble
    for index, row in ensemble.iterrows():
        target_variable = row['target_variable']

        # Extract the corresponding target data from `targets` DataFrame
        y = targets[target_variable]

        # Iterate over feature columns in the ensemble
        for i in range(top_n):  # Assuming there are 50 features (feature0 to feature49)
            feature_col = f'feature{i}'
            feature_name = row[feature_col]  # Get the feature name listed in the ensemble row
            
            if feature_name in features.columns:
                x = features[feature_name]

                x = x.reset_index(drop=True)
                y = y.reset_index(drop=True)

                mask = ~pd.isna(x) & ~pd.isna(y)

                x_filtered = x[mask]
                y_filtered = y[mask]

                if len(x_filtered) > 1 and len(y_filtered) > 1:
                    corr, _ = pearsonr(x_filtered, y_filtered)
                    # print("Correlation between", feature_name, "and", target_variable, "is", corr)
                else:
                    print("Not enough valid values to compute correlation between", feature_name, "and", target_variable)
                ensemble.loc[index, f'{feature_col}_correlation'] = corr

    for i in range(top_n):
        feature_importance_cols = ["feature" + str(i), "feature" + str(i) + "_importance"]
        ensb_cols += feature_importance_cols
        feature_correlations_cols = ["feature" + str(i) + "_correlation"]
        ensb_cols += feature_correlations_cols
        
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
    save_pref = Path(os.getcwd())
    if save_dir is not None:
        save_pref = Path(save_dir)
        save_pref.mkdir(parents=True, exist_ok=True)
    print("loading input files...")
    with open(input_files, "r") as f:
        ipt_dict = json.load(f)
        f.close()
    with open(ensemble_config, "r") as f:
        config_dict = yaml.load(f, yaml.SafeLoader)
        f.close()
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

    ensemble_filename = f"ensemble_{model_name}_{screen_name}.csv"
    feature_metadata_filename = f"feature_metadata_{model_name}_{screen_name}.csv"

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

    # generate feature info file
    feature_info.to_csv(save_pref / "feature_info.csv")
    feature_info_df.to_csv(save_pref / feature_metadata_filename)
    if upload_to_taiga:
        update_taiga(
            feature_info_df,
            "Feature Metadata",
            upload_to_taiga,
            "FeatureMetadata",
        )

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
        # validate_run(dt_hash, sparkles_path, sparkles_config, save_pref)
        validate_dict = {
            "sparkles_config": sparkles_config,
            "sparkles_path": "/install/sparkles/bin/sparkles",
            "HASH": dt_hash,
            "save_pref": save_pref,
        }
        save_and_run_bash(validate_str, validate_dict)
        # gather the ensemble results and collect the top features
        df_ensemble, df_predictions = gather_ensemble_tasks(
            save_pref, features=str(save_pref / "X.ftr"), targets=str(save_pref / "target_matrix.ftr"), top_n=50
        )
        df_ensemble.to_csv(save_pref / ensemble_filename, index=False)
        if upload_to_taiga:
            update_taiga(
                df_ensemble,
                "Y Matrix",
                upload_to_taiga,
                "PredictabilityYMatrix",
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
    "--sparkles-config", required=True, help="path to the sparkles config file to use"
)
@click.option(
    "--save-dir",
    required=False,
    help="""path to where the data should be stored if not the same directory as the script. path is relative to current working directory""",
)
@click.option(
    "--test",
    default=True,
    type=bool,
    help="Run a test version, using only five gene dependencies and five features from each dataset",
)
@click.option(
    "--skipfit",
    default=True,
    type=bool,
    help="Do not execute the fitting, only prepare files. Used for testing",
)
@click.option(
    "--upload-to-taiga",
    default=None,
    type=str,
    help="Upload the Y matrix and the feature metadata to Taiga",
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
    with open(input_files, "r") as f:
        ipt_dict = json.load(f)
        f.close()
    save_pref = Path(os.getcwd())
    if save_dir is not None:
        save_pref /= save_dir
        save_pref.mkdir(parents=True, exist_ok=True)
    config = generate_config(ipt_dict, relation="All")
    dt_hash = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    model_config_name = "model-config" + "_temp_" + ".yaml"

    with open(save_pref / model_config_name, "w") as f:
        yaml.dump(config, f, sort_keys=True)

    _collect_and_fit(
        input_files,
        save_pref / model_config_name,
        sparkles_config,
        save_dir,
        test,
        skipfit,
        upload_to_taiga,
        restrict_targets,
        restrict_to,
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
    is_flag=True,
    help="Run a test version, using only five gene dependencies and five features from each dataset",
)
@click.option(
    "--skipfit",
    is_flag=True,
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
