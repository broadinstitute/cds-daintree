import json
import pandas as pd
from taigapy import create_taiga_client_v3
import numpy as np
import datetime as dt
import yaml
import subprocess
import os
from pathlib import Path
import copy
from string import Template
import datetime
from cds_ensemble.__main__ import prepare_x, prepare_y, fit_model
import click
import pytest

tc = create_taiga_client_v3()


@click.group()
def cli():
    pass


def int_or_str(s: str):
    if s.isdecimal():
        return int(s)
    return s


def clean_dataframe(df, index_col):
    df.sort_index(inplace=True, axis=1)

    if index_col is None:
        df.sort_values(df.columns.tolist(), inplace=True)
    else:
        df.sort_index(inplace=True)

    return df


def process_biomarker_matrix(df, index_col=0, test=False):
    # consolidates the following conseq rules:
    ### create_biomarker_matrix_csv, make_pred_biomarker_matrix, thats it?
    df = df.T
    df = clean_dataframe(df, index_col)
    if test:
        df = df.iloc[:, :5]
    return df


def process_dep_matrix(df, test=False, restrict_targets=False, restrict_to=None):
    # drops rows and columns with all nulls, creates y matrix with cds-ensemble
    df = df.dropna(how="all", axis=0)
    df = df.dropna(how="all", axis=1)
    df.index.name = "Row.name"
    df = df.reset_index()
    if test:
        if restrict_targets:
            print("target restriction:", restrict_to)
            restrict_deps = restrict_to.split(";")
            df = df[["Row.name"] + restrict_deps]
        else:
            df = df.iloc[:, :6]
    elif restrict_targets:
        restrict_deps = restrict_to.split(";")
        df = df[["Row.name"] + restrict_deps]

    return df


def partiton_inputs(dep_matrix, ensemble_config, save_pref, out_name="partitions.csv"):
    num_genes = dep_matrix.shape[1]
    start_indexes = []
    end_indexes = []
    models = []

    for model_name, model_config in ensemble_config.items():

        num_jobs = int(model_config["Jobs"])
        start_index = np.array(range(0, num_genes, num_jobs))
        end_index = start_index + num_jobs
        end_index[-1] = num_genes
        start_indexes.append(start_index)
        end_indexes.append(end_index)
        models.append([model_name] * len(start_index))

    param_df = pd.DataFrame(
        {
            "start": np.concatenate(start_indexes),
            "end": np.concatenate(end_indexes),
            "model": np.concatenate(models),
        }
    )
    param_df.to_csv(save_pref / out_name, index=False)


def generate_config(ipt_dicts, relation="All", name="Model"):
    # input json format:
    # {'name':name, 'taiga_filename':taiga_filename, 'categorical': False, 'required': False, 'match_with': False, 'exempt': False, 'dep_or_feature':'feature'}
    features = [
        ipt_dicts[d]["name"]
        for d in ipt_dicts
        if (ipt_dicts[d]["table_type"] == "feature")
    ]
    required = [
        ipt_dicts[d]["name"]
        for d in ipt_dicts
        if ((ipt_dicts[d]["table_type"] == "feature") and ipt_dicts[d]["required"])
    ]
    exempt = [
        ipt_dicts[d]["name"]
        for d in ipt_dicts
        if ((ipt_dicts[d]["table_type"] == "feature") and ipt_dicts[d]["exempt"])
    ]
    model_config = dict()
    model_config["Features"] = features
    model_config["Required"] = required
    model_config["Relation"] = relation
    if relation == "MatchRelated":
        model_config["Related"] = [
            ipt_dicts[d]["name"]
            for d in ipt_dicts
            if ((ipt_dicts[d]["table_type"] == "relation"))
        ][0]
    model_config["Jobs"] = 10
    if exempt:
        model_config["Exempt"] = exempt

    return {name: model_config}


def generate_feature_info(ipt_dicts, save_pref):
    dsets = [d[1]["name"] for d in ipt_dicts.items() if (d[1]["table_type"] != "dep")]
    fnames = [str(save_pref / (dset + ".csv")) for dset in dsets]

    df = pd.DataFrame({"dataset": dsets, "filename": fnames,})

    return df


def fit_with_sparkles(config_fname, related, sparkles_path, sparkles_config, save_pref):
    dt_hash = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")

    cmd = []
    cmd.append(sparkles_path)
    cmd.extend(["--config", sparkles_config])
    cmd.append("sub")
    cmd.extend(
        ["-i", "us.gcr.io/broad-achilles/depmap-pipeline-tda-integrated-hermitshell:v0"]
    )
    cmd.extend(["-u", str(save_pref / "dep.ftr") + ":target.ftr"])
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
    cmd.extend(["/install/depmap-py/bin/cds-ensemble", "fit-model"])
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
    print(cmd)
    subprocess.check_call(cmd)
    return dt_hash


validate_str = Template(
    """set -ex
$sparkles_path --config $sparkles_config watch ensemble_$HASH --loglive
$sparkles_path --config $sparkles_config reset ensemble_$HASH 
$sparkles_path --config $sparkles_config watch ensemble_$HASH --loglive
mkdir -p $save_pref/data
default_url_prefix=$(awk -F "=" '/default_url_prefix/ {print $2}' "$sparkles_config")
/google-cloud-sdk/bin/gcloud storage ls ${default_url_prefix}/ensemble_$HASH/*/*.csv > $save_pref/completed_jobs.txt
/install/depmap-py/bin/python3.9 validate_jobs_complete.py $save_pref/completed_jobs.txt $save_pref/partitions.csv features.csv predictions.csv
/google-cloud-sdk/bin/gcloud storage cp ${default_url_prefix}/ensemble_$HASH/*/*.csv $save_pref/data
"""
)

upload_str = Template(
    """set -ex
/google-cloud-sdk/bin/gcloud auth activate-service-account --key-file /root/.sparkles-cache/service-keys/broad-achilles.json
/google-cloud-sdk/bin/gcloud storage cp -R $save_pref/* gs://cds-ensemble-pipeline/$gcp_bucket
/google-cloud-sdk/bin/gcloud storage cp /depmap/pipeline/predictability/scripts/model-map.json gs://cds-ensemble-pipeline/$gcp_bucket
"""
)


def validate_run(dt_hash, sparkles_path, sparkles_config, save_pref):
    validate_cmd = validate_str.safe_substitute(
        {
            "sparkles_config": sparkles_config,
            "sparkles_path": sparkles_path,
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
    targets="dep.ftr",
    data_dir="data",
    partitions="partitions.csv",
    features_suffix="features.csv",
    predictions_suffix="predictions.csv",
    top_n=10,
):

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

    all_features = pd.DataFrame().append(
        [pd.read_csv(f) for f in partitions["feature_path"]], ignore_index=True,
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
            .rename(columns={"index": "gene", 0: "pearson"})
        )
        cors["model"] = model

        all_cors.append(cors)

    all_cors = pd.concat(all_cors, ignore_index=True)
    ensemble = all_features.merge(all_cors, on=["gene", "model"])

    # Get the highest correlation across models per "gene" (entity)
    ensemble["best"] = ensemble.groupby("gene")["pearson"].rank(ascending=False) == 1

    ensb_cols = ["gene", "model", "pearson", "best"]

    for i in range(top_n):
        cols = ["feature" + str(i), "feature" + str(i) + "_importance"]
        ensb_cols += cols

    ensemble = ensemble.sort_values(["gene", "model"])[ensb_cols]

    return ensemble, predictions


def check_file_locs(ipt, config):
    ipt_features = [d[1]["name"] for d in ipt.items()]
    for _, model_config in config.items():
        f_set = model_config["Features"]
        f_set += model_config["Required"]

        if model_config["Relation"] not in ["All", "MatchTarget"]:
            f_set.append(model_config["Related"])

        features = list(set(f_set))
        for f in features:
            assert (
                f in ipt_features
            ), f"Feature {f} in model config file does not have corresponding input"


def _collect_and_fit(
    input_files,
    ensemble_config,
    sparkles_path,
    sparkles_config,
    save_dir=None,
    out_bucket=None,
    test=False,
    skipfit=False,
    restrict_targets=False,
    restrict_to=None,
):
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
        len([d[1] for d in ipt_dict.items() if (d[1]["table_type"] == "dep")]) == 1
    ), "Exactly one dataset labeled 'dep' is required"

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

    # generate the feature files for feature_info.csv to point to
    print("saving processed data locally and generating feature_info.csv...")
    if test:
        print("and truncating datasets for testing...")
    for _, d in ipt_dict.items():
        if d["table_type"] not in ["feature", "relation"]:
            continue
        _df = tc.get(d["taiga_filename"])
        if (related_dset is None) or (
            (related_dset is not None) and d["name"] != related_dset
        ):
            _df = process_biomarker_matrix(_df, 0, test)
        _df.to_csv(feature_info.set_index("dataset").loc[d["name"]].filename)

    print("processing dependency data...")
    # load, process, and save dependency matrix
    df_dep = tc.get(
        [d[1] for d in ipt_dict.items() if (d[1]["table_type"] == "dep")][0][
            "taiga_filename"
        ]
    )
    df_dep = process_dep_matrix(df_dep, test, restrict_targets, restrict_to)
    df_dep.to_feather(save_pref / "dep.ftr")

    # generate feature info file
    feature_info.to_csv(save_pref / "feature_info.csv")

    print('running "prepare_y"...')
    subprocess.check_call(
        [
            "/install/depmap-py/bin/cds-ensemble",
            "prepare-y",
            "--input",
            str(save_pref / "dep.ftr"),
            "--output",
            str(save_pref / "dep-filtered.ftr"),
        ]
    )
    # prepare_y(input="dep.ftr", output="dep-filtered.ftr")
    # pytest.set_trace()
    print('running "prepare_x"...')
    prep_x_cmd = [
        "/install/depmap-py/bin/cds-ensemble",
        "prepare-x",
        "--model-config",
        str(ensemble_config),
        "--targets",
        str(save_pref / "dep-filtered.ftr"),
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
            ensemble_config, out_rel, sparkles_path, sparkles_config, save_pref
        )

        # watch sparkles run, resubmit failed jobs, and validate when complete
        # validate_run(dt_hash, sparkles_path, sparkles_config, save_pref)
        validate_dict = {
            "sparkles_config": sparkles_config,
            "sparkles_path": sparkles_path,
            "HASH": dt_hash,
            "save_pref": save_pref,
        }
        save_and_run_bash(validate_str, validate_dict)
        # gather the ensemble results and collect the top features
        df_ensemble, df_predictions = gather_ensemble_tasks(
            save_pref, targets=str(save_pref / "dep.ftr"), top_n=10
        )
        df_ensemble.to_csv(save_pref / "ensemble.csv", index=False)
        df_predictions.to_csv(save_pref / "predictions.csv")

    else:
        print("skipping fitting and ending run")

    print("uploading files to GCP")
    upload_dict = {
        "ipt_files": input_files,
        "gcp_bucket": out_bucket,
        "save_pref": str(save_pref),
    }
    save_and_run_bash(upload_str, upload_dict)


@cli.command()
@click.option(
    "--input-files",
    required=True,
    help="JSON file containing the set of files for prediction",
)
@click.option("--sparkles-path", required=True, help="path to the sparkles command")
@click.option(
    "--sparkles-config", required=True, help="path to the sparkles config file to use"
)
@click.option(
    "--save-dir",
    required=False,
    help="""path to where the data should be stored if not the same directory as the script. path is relative to current working directory""",
)
@click.option(
    "--out-bucket",
    required=False,
    type=str,
    help="""location in the "cds-ensemble-pipeline" bucket where the data should be uploaded to""",
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
    sparkles_path,
    sparkles_config,
    save_dir=None,
    out_bucket=None,
    test=False,
    skipfit=False,
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
        sparkles_path,
        sparkles_config,
        save_dir,
        out_bucket,
        test,
        skipfit,
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
@click.option("--sparkles-path", required=True, help="path to the sparkles command")
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
def collect_and_fit(
    input_files,
    ensemble_config,
    sparkles_path,
    sparkles_config,
    test=False,
    skipfit=False,
):

    _collect_and_fit(
        input_files,
        ensemble_config,
        sparkles_path,
        sparkles_config,
        save_dir=None,
        out_bucket=None,
        test=test,
        skipfit=skipfit,
    )


if __name__ == "__main__":
    cli()
