import click
import json
from pathlib import Path
from .prepare import prepare

from . import config_manager

from taigapy import create_taiga_client_v3

from .utils import update_taiga


# CLI Setup
@click.group()
def cli():
    print("\033[96mHello, World! Here I start my journey in Daintree.\033[0m")  # Teal
    pass


# class SparklesRunner:
#     def __init__(self, save_pref, config_fname, related, sparkles_config):
#         self.save_pref = Path(save_pref)
#         self.config_fname = Path(config_fname)
#         self.related = related
#         self.sparkles_config = sparkles_config
#         self.dt_hash = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
#         self.sparkles_path = PATHS['sparkles_bin']

#     def _build_sparkles_command(self):
#         base_cmd = [
#             self.sparkles_path,
#             "--config", self.sparkles_config,
#             "sub",
#             "-i", CONTAINER['image'],
#             "-u", main.__file__,
#             "-u", f"{self.save_pref}/{FILES['target_matrix']}:target.ftr",
#             "-u", f"{self.save_pref}/{self.config_fname.name}:model-config.yaml",
#             "-u", f"{self.save_pref}/{FILES['feature_matrix']}:X.ftr",
#             "-u", f"{self.save_pref}/{FILES['feature_metadata']}:X_feature_metadata.ftr",
#             "-u", f"{self.save_pref}/{FILES['valid_samples']}:X_valid_samples.ftr",
#             "-u", f"{self.save_pref}/{FILES['partitions']}",
#             "--params", f"{self.save_pref}/{FILES['partitions']}",
#             "--skipifexists",
#             "--nodes", "100",  # This could also be moved to config maybe?
#             "-n", f"ensemble_{self.dt_hash}",
#             PATHS['daintree_bin'], "fit-model", # TODO: This is essentially replacing cds-ensemble fit-model
#             "--x", "X.ftr",
#             "--y", "target.ftr",
#             "--model-config", "model-config.yaml",
#             "--n-folds", str(MODEL['n_folds']),
#             "--target-range", "{start}", "{end}",
#             "--model", "{model}"
#         ]

#         if self.related:
#             base_cmd.extend([
#                 "-u", f"{self.save_pref}/related.ftr:related.ftr",
#                 "--related-table", "related.ftr"
#             ])
#         return base_cmd

# def _watch_jobs(self):
#     subprocess.check_call([self.sparkles_path, "--config", self.sparkles_config,
#                          "watch", f"ensemble_{self.dt_hash}", "--loglive"])

# def _reset_jobs(self):
#     subprocess.check_call([self.sparkles_path, "--config", self.sparkles_config,
#                          "reset", f"ensemble_{self.dt_hash}"])

# def _validate_jobs_complete(self):
#     """Validate that all expected job outputs exist"""
#     with open(f"{self.save_pref}/completed_jobs.txt") as f:
#         completed_jobs = {l.split("/")[-1].strip() for l in f.readlines()}

#     partitions = pd.read_csv(f"{self.save_pref}/partitions.csv")
#     partitions["path_prefix"] = (
#         partitions["model"]
#         + "_"
#         + partitions["start"].map(str)
#         + "_"
#         + partitions["end"].map(str)
#         + "_"
#     )
#     partitions["feature_path"] = partitions["path_prefix"] + "features.csv"
#     partitions["predictions_path"] = partitions["path_prefix"] + "predictions.csv"

#     assert len(set(partitions["feature_path"]) - completed_jobs) == 0, "Missing feature files"
#     assert len(set(partitions["predictions_path"]) - completed_jobs) == 0, "Missing prediction files"

# def _process_completed_jobs(self):
#     """Process completed jobs and copy results to local directory.
#     """
#     os.makedirs(f"{self.save_pref}/data", exist_ok=True)
#     default_url_prefix = self._get_default_url_prefix()

#     self._authenticate_gcloud()

#     completed_jobs = subprocess.check_output([
#         "/google-cloud-sdk/bin/gcloud",
#         "storage",
#         "ls",
#         f"{default_url_prefix}/ensemble_{self.dt_hash}/*/*.csv"
#     ]).decode()

#     self._save_completed_jobs(completed_jobs)
#     self._validate_jobs_complete()
#     self._copy_results_to_local()

# def _get_default_url_prefix(self):
#     """Get the default URL prefix from the sparkles config file.
#     """
#     with open(self.sparkles_config, 'r') as f:
#         for line in f:
#             if 'default_url_prefix' in line:
#                 return line.split('=')[1].strip()

# def _authenticate_gcloud(self):
#     subprocess.check_call([
#         "/google-cloud-sdk/bin/gcloud",
#         "auth",
#         "activate-service-account",
#         "--key-file",
#         PATHS['service_account']
#     ])

# def _save_completed_jobs(self, completed_jobs):
#     with open(f"{self.save_pref}/completed_jobs.txt", 'w') as f:
#         f.write(completed_jobs)

# def _copy_results_to_local(self):
#     """Copy results from Google Cloud Storage to local directory.
#     """
#     default_url_prefix = self._get_default_url_prefix()
#     subprocess.check_call([
#         "/google-cloud-sdk/bin/gcloud",
#         "storage",
#         "cp",
#         f"{default_url_prefix}/ensemble_{self.dt_hash}/*/*.csv",
#         f"{self.save_pref}/data"
#     ])

# def run(self):
#     cmd = self._build_sparkles_command()
#     print(f"Running sparkles with command: {cmd}")
#     subprocess.check_call(cmd)
#     print("Sparkles run complete")
#     return self.dt_hash

# def validate(self):
#     print("Validating sparkles run...")
#     self._watch_jobs()
#     self._reset_jobs()
#     self._watch_jobs()
#     self._process_completed_jobs()


class TaigaUploader:
    def __init__(self, save_pref, upload_to_taiga):
        self.save_pref = Path(save_pref)
        self.upload_to_taiga = upload_to_taiga

    def upload_results(self, runner_config, model_name, screen_name, output_config):
        """Upload results to Taiga and create output config file.

        Args:
            runner_config: Input configuration dictionary

        Returns:
            tuple: (feature_metadata_taiga_info, ensemble_taiga_info, predictions_taiga_info)
        """
        feature_metadata_filename = f"FeatureMetadata{model_name}{screen_name}.csv"
        ensemble_filename = f"Ensemble{model_name}{screen_name}.csv"
        predictions_filename = f"Predictions{model_name}{screen_name}.csv"

        feature_metadata_taiga_info = update_taiga(
            self.upload_to_taiga,
            f"Updated Feature Metadata for Model: {model_name} and Screen: {screen_name}",
            f"FeatureMetadata{model_name}{screen_name}",
            self.save_pref / feature_metadata_filename,
            "csv_table",
        )
        print(f"Feature Metadata uploaded to Taiga: {feature_metadata_taiga_info}")

        ensemble_taiga_info = update_taiga(
            self.upload_to_taiga,
            f"Updated Ensemble for Model: {model_name} and Screen: {screen_name}",
            f"Ensemble{model_name}{screen_name}",
            self.save_pref / ensemble_filename,
            "csv_table",
        )
        print(f"Ensemble uploaded to Taiga: {ensemble_taiga_info}")

        predictions_taiga_info = update_taiga(
            self.upload_to_taiga,
            f"Updated Predictions for Model: {model_name} and Screen: {screen_name}",
            f"Predictions{model_name}{screen_name}",
            self.save_pref / predictions_filename,
            "csv_table",
        )
        print(f"Predictions uploaded to Taiga: {predictions_taiga_info}")

        output_config = config_manager.create_output_config(
            input_config=runner_config,
            feature_metadata_id=feature_metadata_taiga_info,
            ensemble_id=ensemble_taiga_info,
            prediction_matrix_id=predictions_taiga_info,
        )

        output_config_dir = self.save_pref / "output_config_files"
        output_config_dir.mkdir(parents=True, exist_ok=True)
        # This could probably be hardcoded and put in a config file.
        # However, I am keeping it this way for now to make it more flexible.
        output_config_filename = f"OutputConfig{model_name}{screen_name}.json"
        output_config_file = output_config_dir / output_config_filename

        with open(output_config_file, "w") as f:
            json.dump(output_config, f, indent=4)
        print(f"Created output config file: {output_config_file}")

        return feature_metadata_taiga_info, ensemble_taiga_info, predictions_taiga_info


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
    default=False,
    type=bool,
    help="Run a test version with limited data",
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
    "--test",
    default=False,
    type=bool,
    help="Run a test version with limited data",
)
@click.option(
    "--restrict-targets-to",
    default=None,
    type=str,
    help="Comma separated list of names to filter target columns. If not provided, uses TEST_LIMIT from config.py",
)
@click.option(
    "--nfolds", default=5, type=int, help="Number of folds to use in cross validation (defaults to 5)"
)
def prepare_and_partition(
    input_config,
    out,
    test,
    restrict_targets_to,
    nfolds
):
    import pdb
    try:
        """Run model fitting with either provided or auto-generated config."""
        save_pref = Path(out) if out else Path.cwd()
        print(f"Save directory path: {save_pref}")
        save_pref.mkdir(parents=True, exist_ok=True)
        tc = create_taiga_client_v3()
        save_pref.mkdir(parents=True, exist_ok=True)
        prepare(
            tc,
            test=test,
            restrict_targets_to=(
                restrict_targets_to.split(",") if restrict_targets_to else None
            ),
            runner_config_path=input_config,
            save_pref=save_pref,
            nfolds=nfolds
        )

        print("\033[96mMy journey in Daintree has finished.\033[0m")  # Teal
    except Exception as ex:
        print(f"Unhandled exception: {ex}")
        pdb.post_mortem()

#{DAINTREE_BIN_PATH} fit-model --x X.ftr --y target.ftr --model-config model-config.yaml --n-folds {nfolds} --target-range {partition.start_index} {partition.end_index} --model {partition.model_name}"
# @cli.command()
# @click.option(
#     "--x",
#     required=True,
#     help="Path to JSON config file containing the set of files for prediction",
# )
# def fit_model(x, y, model_config, nfolds, start, end, model):
#     # model_config is named this so that the command line parameter is called --model-config, but this really corresponds to core_config_path (Which is to say a daintree_core configuration file)
#     raise NotImplementedError()

def main():
    cli()
