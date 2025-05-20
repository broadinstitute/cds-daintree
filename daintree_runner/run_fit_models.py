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
    pass


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

from .gather import gather as _gather

@cli.command()
@click.option("--dir", required=True, help="Directory to scan for *_features.csv and *_predictions.csv")
@click.option("--dest-prefix", help="Prefix to prepend onto output files", default="")
def gather(dir,dest_prefix):
    _gather(dir, dest_prefix)


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

    except Exception as ex:
        print(f"Unhandled exception: {ex}")
        pdb.post_mortem()

def main():
    cli()
