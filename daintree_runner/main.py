import click
import json
from pathlib import Path
from .prepare import prepare
from taigapy import create_taiga_client_v3
import os


# CLI Setup
@click.group()
def cli():
    pass


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
    help="If set, will only run a max of N tasks (for testing)"    
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
    help="The number of models to fit per each sparkles task"
)
def prepare_and_partition(input_config, out, test_first_n_models, restrict_targets_to, nfolds, models_per_task, test_first_n_tasks):
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
            test_first_n_tasks=test_first_n_tasks
        )

    # except Exception as ex:
    #     print(f"Unhandled exception: {ex}")
    #     pdb.post_mortem()


from typing import Optional
from .config import DAINTREE_CORE_BIN_PATH

@cli.command()
@click.option(
    "--config",
    help="Path to the json daintree model config file",
    required=True
)
@click.option(
    "--out",
    help="Path to write workflow to. If not specified, writes to stdout",
)
@click.option(
    "--nfolds",
    default=5,
    type=int
)
@click.option(
    "--models-per-task",
    default=10,
    type=int,
    help="The number of models to fit per each sparkles task"
)
@click.option(
    "--test-first-n-tasks",
    type=int,
    help="If set, will only run a max of N tasks (for testing)"    
)
@click.option("--test", is_flag=True, help="Run a test run (subsetting the data to make a fast, but incomplete, run)")
def create_sparkles_workflow(config: str, out: Optional[str], test: bool, nfolds: int, models_per_task: int, test_first_n_tasks:Optional[int]):
    prepare_command = [
                    "daintree-runner",
                    "prepare-and-partition",
                    "--input-config",
                    "model_config.json",
                    "--out",
                    "out",
                    "--models-per-task",
                    str(models_per_task)
                ] + (["--test-first-n-tasks", str(test_first_n_tasks)] if test_first_n_tasks else [])
    if test:
        prepare_command.append("--test")

    taiga_token = _find_taiga_token()

    workflow = {
        "paths_to_localize": [
        {"src": taiga_token, "dst":".taiga-token"}
        ],
        "steps": [
            {
                "command": prepare_command,
                "files_to_localize": [f"model_config.json"],
            },
            {
                # f"{DAINTREE_CORE_BIN_PATH} fit-model --x X.ftr --y target.ftr --model-config {core_config_path} --n-folds {nfolds} --target-range {partition.start_index} {partition.end_index} --model {partition.model_name}"
                "command": [DAINTREE_CORE_BIN_PATH, "fit-model", "--x", "out/X.ftr", "--y", "out/target_matrix.ftr", "--model-config", "{parameter.model_config}", "--n-folds", str(nfolds), "--target-range", "{parameter.start_index}", "{parameter.end_index}", "--model", "{parameter.model_name}"],
                "parameters_csv": "{step.1.job_path}/1/out/partitions.csv",
                  "paths_to_localize": [
                    {"src": "{step.1.job_path}/1/out", "dst":"out"}
                ]
            },
            {"command": ["daintree-runner", "gather", "--dir", "{step.2.job_path}"]},
        ],
        "write_on_completion": [
            {
                "expression": {"ensemble_path":
                                "{step.3.job_path}/1/ensemble.csv",
                               "predictions_path": 
                                "{step.3.job_path}/1/predictions.csv"},
                "filename": "outputs.json"
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
