import os
import pathlib

from typing import List, Optional, Tuple

import click
import pandas as pd

from .prepare_targets import prepare_targets
from .prepare_features import prepare_features, format_related
from .run_ensemble import filter_run_ensemble_inputs, run_model
from .parsing_utilities import (
    read_dataframe,
    read_dataframe_row_headers,
    read_model_config,
    read_feature_info,
)
import random
from time import time
import resource

@click.group()
def main():
    pass


@main.command()
@click.option("--input", required=True, help="The file to filter and reformat")
@click.option("--output", required=True, help="Path where to write the output")
@click.option(
    "--top-variance-filter",
    type=int,
    help="If specified, will only keep the top N targets, ranked by variance",
)
@click.option("--gene-filter", help="If specified, will only keep the listed genes")
def prepare_y(
    input: str,
    output: str,
    top_variance_filter: Optional[int],
    gene_filter: Optional[str],
):
    return prepare_y_command(input, output, top_variance_filter, gene_filter)


def prepare_y_command(
    input: str,
    output: str,
    top_variance_filter: Optional[int],
    gene_filter: Optional[str],
):
    if top_variance_filter is not None and top_variance_filter < 1:
        raise click.ClickException("Top variance filter must be >= 1")

    try:
        df = read_dataframe(input)
    except FileNotFoundError:
        raise click.ClickException(f"File {input} not found")
    except pd.ParserError:
        raise click.ClickException(f"Could not read {input} as CSV")

    try:
        df = df.astype(float)
    except ValueError:
        raise click.ClickException(f"Values in {input} must all be numbers")

    try:
        gene_filter_list: Optional[List[str]] = None

        if gene_filter is not None:
            gene_filter_list = [gene.strip() for gene in gene_filter.split(",")]

        # Filter targets based on variance and/or gene
        filtered_df = prepare_targets(df, top_variance_filter, gene_filter_list)

        # Make output parent directories if they don't already exist
        pathlib.Path(os.path.dirname(output)).mkdir(parents=True, exist_ok=True)

        # Reset index because feather does not support indexes, then output as feather
        filtered_df.reset_index().to_feather(output)
    except ValueError as e:
        raise click.ClickException(str(e))


@main.command()
@click.option("--targets", required=True, help="Matrix of the targets we are modeling")
@click.option(
    "--model-config",
    required=True,
    type=str,
    help="The file with model configurations (need to define format of this below) TODO",
)
@click.option(
    "--feature-info",
    required=True,
    help="Table containing feature datasets required and filename columns",
)
@click.option(
    "--output",
    required=True,
    type=str,
    help="Full path to where to write the merged matrix",
)
@click.option("--confounders", help="Table with target dataset specific QC, e.g. NNMD")
@click.option(
    "--output-format",
    type=click.Choice([".ftr", ".csv"], case_sensitive=False),
    help="Which format to write the output in (.ftr or .csv)",
)
@click.option(
    "--output-related",
    help='if specified, write out a file which can be used with "cds-ensemble fit-model --feature-subset ..." to select only related features for each target.',
)
def prepare_x(
    model_config: str,
    targets: str,
    feature_info: str,
    output: str,
    confounders: Optional[str],
    output_format: Optional[str],
    output_related: Optional[str],
):
    return prepare_x_command(
        model_config,
        targets,
        feature_info,
        output,
        confounders,
        output_format,
        output_related,
    )


def prepare_x_command(
    model_config: str,
    targets: str,
    feature_info: str,
    output: str,
    confounders: Optional[str],
    output_format: Optional[str],
    output_related: Optional[str],
):
    "This prepare_x_command() function exists so we can call prepare_x directly, bypassing click"

    if output_format is None:
        output_format = ".ftr"

    for p in [model_config, targets, feature_info]:
        if not os.path.exists(p):
            raise click.ClickException(f"File {p} not found")

    try:
        model_configs = read_model_config(model_config)

        target_samples = read_dataframe_row_headers(targets)
        feature_infos = read_feature_info(feature_info, confounders)
    except ValueError as e:
        raise click.ClickException(str(e))

    try:
        if output_related:
            file_prefix, _ = os.path.splitext(output_related)
            related_table = format_related(model_configs, feature_infos)
            if output_format == ".csv":
                related_table.to_csv(f"{file_prefix}.csv", index=False)
            else:
                related_table.reset_index(drop=True).to_feather(f"{file_prefix}.ftr")
    except ValueError as e:
        raise click.ClickException(str(e))

    try:
        combined_features, feature_metadata, model_valid_samples = prepare_features(
            model_configs, target_samples, feature_infos, confounders
        )
    except ValueError as e:
        raise click.ClickException(str(e))

    file_prefix, _ = os.path.splitext(output)
    if output_format == ".csv":
        combined_features.to_csv(f"{file_prefix}.csv")
        feature_metadata.to_csv(f"{file_prefix}_feature_metadata.csv", index=False)
        model_valid_samples.to_csv(f"{file_prefix}_valid_samples.csv")
    else:
        assert output_format == ".ftr", f"expected output_format=.ftr but was {output_format}, output={output}"
        combined_features.reset_index().to_feather(f"{file_prefix}.ftr")
        feature_metadata.reset_index(drop=True).to_feather(
            f"{file_prefix}_feature_metadata.ftr"
        )
        model_valid_samples.reset_index().to_feather(f"{file_prefix}_valid_samples.ftr")


@main.command()
@click.option(
    "--x",
    type=str,
    required=True,
    help="A feather file containing all features. The default is to use all features. A subset of features can be selected by specifying --feature-subset",
)
@click.option(
    "--y",
    type=str,
    required=True,
    help="A feature file containing all targets. The default is to fit each target sequentially. A subset of targets can be selected by specifying --target-range or --targets",
)
@click.option(
    "--model-config",
    type=str,
    required=True,
    help="The file with model configurations (need to define format of this below) TODO",
)
@click.option("--model", type=str, required=True)
@click.option(
    "--task-mode", type=click.Choice(["regress", "classify"]), default="regress"
)
@click.option("--n-folds", type=int, default=3)
@click.option("--related-table", type=str)
@click.option(
    "--feature-metadata",
    type=str,
    help="The feature metadata outputted by prepare-x. If not specified, will try to use the file that matches the '--x' parameter",
)
@click.option(
    "--model-valid-samples",
    type=str,
    help="The valid samples per model outputted by prepare-x. If not specified, will try to use the file that matches the '--x' parameter",
)
@click.option(
    "--valid-samples-file",
    type=str,
    help="If selected, only use the following samples in the training",
)
@click.option(
    "--feature-subset-file",
    type=str,
    # TODO Format
    help="if specified, use the given file to determine which features to subset. If not specified, all features will be used",
)
@click.option(
    "--target-range",
    nargs=2,
    type=int,
    callback=lambda ctx, param, value: value if len(value) == 2 else None,
    help="if specified, fit models for targets whose indices are in this inclusive range",
)
@click.option(
    "--targets", type=str, help="if specified, fit models for targets with these labels"
)
@click.option("--output-dir", type=str)
@click.option(
    "--top-n",
    help="Number of features to write to resulting file (defaults to 50)",
    type=int,
    default=50,
)
@click.option("--seed", help="random seed (defaults to 0)", type=int, default=0)
def fit_model(
    x: str,
    y: str,
    model_config: str,
    model: str,
    task_mode: str,
    n_folds: int,
    related_table: Optional[str],
    feature_metadata: Optional[str],
    model_valid_samples: Optional[str],
    valid_samples_file: Optional[str],
    feature_subset_file: Optional[str],
    target_range: Optional[Tuple[int, int]],
    targets: Optional[str],
    output_dir: Optional[str],
    top_n: int,
    seed: int,
):
    return fit_model_command(
        x,
        y,
        model_config,
        model,
        task_mode,
        n_folds,
        related_table,
        feature_metadata,
        model_valid_samples,
        valid_samples_file,
        feature_subset_file,
        target_range,
        targets,
        output_dir,
        top_n,
        seed,
    )


def fit_model_command(
    x: str,
    y: str,
    model_config: str,
    model: str,
    task_mode: str,
    n_folds: int,
    related_table: Optional[str],
    feature_metadata: Optional[str],
    model_valid_samples: Optional[str],
    valid_samples_file: Optional[str],
    feature_subset_file: Optional[str],
    target_range: Optional[Tuple[int, int]],
    targets: Optional[str],
    output_dir: Optional[str],
    top_n: int,
    seed: int,
):
    start_time = time()
    random.seed(seed)
    selected_model_config = read_model_config(model_config)[model]
    if selected_model_config.relation == "MatchRelated" and related_table is None:
        raise click.ClickException(
            "The model selected uses the MatchRelated relation, but no related table was provided."
        )
    file_prefix, ext = os.path.splitext(x)
    X = read_dataframe(x)
    Y = read_dataframe(y)

    related_table_df = (
        read_dataframe(related_table) if related_table is not None else None
    )
    try:
        feature_metadata_path = (
            feature_metadata
            if feature_metadata is not None
            else f"{file_prefix}_feature_metadata{ext}"
        )
        feature_metadata_df = read_dataframe(feature_metadata_path, set_index=False)
    except FileNotFoundError as e:
        raise click.ClickException(
            f"Expected file '{e.filename}' outputted by prepare-x command"
        )

    try:
        model_valid_samples_path = (
            model_valid_samples
            if model_valid_samples is not None
            else f"{file_prefix}_valid_samples{ext}"
        )
        model_valid_samples_df = read_dataframe(model_valid_samples_path)

        if not model_valid_samples_df.index.equals(X.index):
            raise ValueError("Model valid samples's index does not match X's index.")

        if selected_model_config.name not in model_valid_samples_df.columns:
            raise ValueError(
                f"Model valid samples does not have entry for model {selected_model_config.name}."
            )
    except ValueError as e:
        raise click.ClickException(str(e))
    except FileNotFoundError as e:
        raise click.ClickException(
            f"Expected file '{e.filename}' outputted by prepare-x command"
        )

    valid_samples = None
    if valid_samples_file is not None:
        valid_samples = read_dataframe_row_headers(valid_samples_file)

    feature_subset = None
    if feature_subset_file is not None:
        feature_subset = read_dataframe_row_headers(feature_subset_file)

    target_list = None
    if targets is not None:
        target_list = targets.split(",")

    try:
        X, Y, start_col, end_col = filter_run_ensemble_inputs(
            X,
            Y,
            selected_model_config,
            feature_metadata_df,
            model_valid_samples_df,
            valid_samples,
            feature_subset,
            target_range,
            target_list,
        )
    except ValueError as e:
        raise click.ClickException(str(e))

    # start_col, end_col not passed in because X, Y already filtered
    fit_start = time()
    ensemble = run_model(
        X=X,
        Y=Y,
        model=selected_model_config,
        nfolds=n_folds,
        task=task_mode,
        relation_table=related_table_df,
        feature_metadata=feature_metadata_df,
    )

    write_start = time()
    feature_file_path = (
        f"{selected_model_config.name}_{start_col}_{end_col}_features.csv"
    )
    predictions_file_path = (
        f"{selected_model_config.name}_{start_col}_{end_col}_predictions.csv"
    )

    if output_dir is not None:
        feature_file_path = os.path.join(output_dir, feature_file_path)
        predictions_file_path = os.path.join(output_dir, predictions_file_path)

    print(f"Writing {feature_file_path} and {predictions_file_path}...")
    ensemble.save_results(feature_file_path, predictions_file_path, top_n, X, Y)

    task_end = time()
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # recording the elapsed timings 
    timings_df = pd.DataFrame([{"model": model, 
                               "start_col": start_col,
                               "end_col": end_col,
                               "startup_secs": fit_start-start_time, 
                               "fit_secs": write_start-fit_start, 
                               "write_secs": task_end-write_start, 
                               "target_count": Y.shape[1],
                               "max_rss":max_rss}])
    timings_df.to_csv(f"timings.csv",index=False)


if __name__ == "__main__":
    main()
