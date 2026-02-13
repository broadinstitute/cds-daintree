import click
import json
from pathlib import Path
from .prepare import prepare
from taigapy import create_taiga_client_v3
import os
import sys
from glob import glob


# CLI Setup
@click.group()
@click.option(
    "--python-path",
    multiple=True,
    help="Add directory to sys.path for importing custom preprocessing functions",
)
def cli(python_path):
    for path in python_path:
        if path not in sys.path:
            sys.path.insert(0, path)


@cli.command()
@click.option(
    "--input-config",
    required=True,
    help="Path to JSON config file containing the set of files for prediction",
)
@click.option(
    "--ensemble-config",
    required=False,
    help="YAML file for model configuration. If not provided, will be auto-generated.",
)
@click.option(
    "--out",
    required=False,
    help="Path to where the data should be stored if not the same directory as the script",
)
@click.option(
    "--test",
    is_flag=True,
    help="Run a test by running on a subset of the data",
)
@click.option(
    "--upload-to-taiga",
    default=None,
    type=str,
    help="Upload results to Taiga",
)
@click.option(
    "--restrict-targets-to",
    default=None,
    type=str,
    help="Comma separated list of names to filter target columns. If not provided, uses TEST_LIMIT from config.py",
)
def run(
    input_config,
    ensemble_config,
    out=None,
    test=False,
    restrict_targets_to=None,
):
    pass


from .gather import gather as _gather


@cli.command()
@click.option(
    "--dir",
    default=".",
    help="Directory to scan for *_features.csv and *_predictions.csv",
)
@click.option("--dest-prefix", help="Prefix to prepend onto output files", default="")
@click.argument("partitions_csv")
def gather(partitions_csv, dir, dest_prefix):
    _gather(dir, dest_prefix, partitions_csv)


@cli.command()
@click.option(
    "--input-config",
    required=True,
    help="Path to JSON config file containing the set of files for prediction",
)
@click.option(
    "--out",
    required=False,
    help="Path to where the data should be stored if not the same directory as the script",
)
@click.option(
    "--test-first-n-models",
    type=int,
    default=None,
    help="If set, will only run a max of the first N models (for testing)",
)
@click.option(
    "--test-first-n-tasks",
    type=int,
    help="If set, will only run a max of N tasks (for testing)",
)
@click.option(
    "--restrict-targets-to",
    default=None,
    type=str,
    help="Comma separated list of names to filter target columns. If not provided, uses TEST_LIMIT from config.py",
)
@click.option(
    "--nfolds",
    default=5,
    type=int,
    help="Number of folds to use in cross validation (defaults to 5)",
)
@click.option(
    "--models-per-task",
    default=10,
    type=int,
    help="The number of models to fit per each sparkles task",
)
def prepare_and_partition(
    input_config,
    out,
    test_first_n_models,
    restrict_targets_to,
    nfolds,
    models_per_task,
    test_first_n_tasks,
):
    # import pdb

    # try:
    """Run model fitting with either provided or auto-generated config."""
    save_pref = Path(out) if out else Path.cwd()
    print(f"Save directory path: {save_pref}")
    save_pref.mkdir(parents=True, exist_ok=True)
    tc = create_taiga_client_v3()
    save_pref.mkdir(parents=True, exist_ok=True)
    prepare(
        tc,
        test_first_n_models=test_first_n_models,
        restrict_targets_to=(
            restrict_targets_to.split(",") if restrict_targets_to else None
        ),
        runner_config_path=input_config,
        save_pref=save_pref,
        nfolds=nfolds,
        models_per_task=models_per_task,
        test_first_n_tasks=test_first_n_tasks,
    )


# except Exception as ex:
#     print(f"Unhandled exception: {ex}")
#     pdb.post_mortem()


from typing import Optional
from .config import DAINTREE_CORE_BIN_PATH


@cli.command()
@click.option(
    "--config", help="Path to the json daintree model config file", required=True
)
@click.option(
    "--out",
    help="Path to write workflow to. If not specified, writes to stdout",
)
@click.option("--nfolds", default=5, type=int)
@click.option(
    "--models-per-task",
    default=10,
    type=int,
    help="The number of models to fit per each sparkles task",
)
@click.option(
    "--test-first-n-tasks",
    type=int,
    help="If set, will only run a max of N tasks (for testing)",
)
@click.option(
    "--test",
    is_flag=True,
    help="Run a test run (subsetting the data to make a fast, but incomplete, run)",
)
@click.option(
    "--python-to-upload",
    multiple=True,
    help="Directory which contains python files which should be in python search path so they can contain functions used for preprocessing data. All *.py files in this directory will be transferred to worker nodes into a single directory.",
)
def create_sparkles_workflow(
    config: str,
    out: Optional[str],
    test: bool,
    nfolds: int,
    models_per_task: int,
    test_first_n_tasks: Optional[int],
    python_to_upload: tuple,
):
    # Build prepare command with optional python-path arguments

    # I worry about the random names that
    transfered_python_files_dir = "extra_python_files"
    python_files_to_transfer = []
    for python_path_ in python_to_upload:
        python_files_to_transfer.extend(glob(f"{python_path_}/*.py"))

    python_path_parameter = []
    if len(python_files_to_transfer) > 0:
        print(
            f"Found the following files {python_files_to_transfer} which will be transferred to {repr(transfered_python_files_dir)} on node",
            file=sys.stderr,
        )
        python_path_parameter.extend(["--python-path", transfered_python_files_dir])

    prepare_command = ["daintree-runner"] + python_path_parameter

    prepare_command.extend(
        [
            "prepare-and-partition",
            "--input-config",
            "model_config.json",
            "--out",
            "out",
            "--models-per-task",
            str(models_per_task),
        ]
    )
    if test_first_n_tasks:
        prepare_command.extend(["--test-first-n-tasks", str(test_first_n_tasks)])
    if test:
        prepare_command.append("--test")

    # Build fit-model command with optional python-path arguments
    fit_model_command = [DAINTREE_CORE_BIN_PATH] + python_path_parameter
    fit_model_command.extend(
        [
            "fit-model",
            "--x",
            "out/X.ftr",
            "--y",
            "out/target_matrix.ftr",
            "--model-config",
            "{parameter.model_config}",
            "--n-folds",
            str(nfolds),
            "--target-range",
            "{parameter.start_index}",
            "{parameter.end_index}",
            "--model",
            "{parameter.model_name}",
        ]
    )

    taiga_token = _find_taiga_token()

    # Build paths_to_localize including python-path directories
    paths_to_localize = [{"src": taiga_token, "dst": ".taiga-token"}]
    for python_files_to_transfer in python_files_to_transfer:
        paths_to_localize.append(
            {
                "src": python_files_to_transfer,
                "dst": f"{transfered_python_files_dir}/{os.path.basename(python_files_to_transfer)}",
            }
        )

    workflow = {
        "paths_to_localize": paths_to_localize,
        "steps": [
            {
                "command": prepare_command,
                "files_to_localize": ["model_config.json"],
            },
            {
                "command": fit_model_command,
                "parameters_csv": "{step.1.job_path}/1/out/partitions.csv",
                "paths_to_localize": [{"src": "{step.1.job_path}/1/out", "dst": "out"}],
            },
            {"command": ["daintree-runner", "gather", "--dir", "{step.2.job_path}"]},
        ],
        "write_on_completion": [
            {
                "expression": {
                    "ensemble_path": "{step.3.job_path}/1/ensemble.csv",
                    "predictions_path": "{step.3.job_path}/1/predictions.csv",
                },
                "filename": "outputs.json",
            },
        ],
    }
    workflow_json = json.dumps(workflow, indent=2)

    if out:
        with open(out, "wt") as fd:
            fd.write(workflow_json)
    else:
        print(workflow_json)


def _find_taiga_token():
    search_path = [".taiga-token", f"{os.environ['HOME']}/.taiga/token"]
    for path in search_path:
        if os.path.exists(path):
            return path
    raise Exception(f"Could not find taiga token. Checked for it in: {search_path}")


def main():
    cli()
