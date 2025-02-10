import os
import subprocess
import click
import re
import json
import yaml
import numpy as np
import pandas as pd
import datetime as dt
from pathlib import Path
import time

from taigapy import create_taiga_client_v3
from daintree_package import main

from config import PATHS, CONTAINER, FILES, MODEL, TEST_LIMIT, filter_columns_gene, filter_columns_oncref
from utils import calculate_feature_correlations, update_taiga

# CLI Setup
@click.group()
def cli():
    print("Hello, World!")
    pass


class DataProcessor:
    def __init__(self, save_pref):
        self.save_pref = Path(save_pref)
        self.tc = create_taiga_client_v3()

    def _clean_dataframe(self, df, index_col):
        """Clean and sort the dataframe.
        Args:
            df: DataFrame to clean and sort
            index_col: Index column number
        Returns:
            pd.DataFrame: Cleaned and sorted dataframe
        """
        df.sort_index(inplace=True, axis=1)

        if index_col is None:
            df.sort_values(df.columns.tolist(), inplace=True)
        else:
            df.sort_index(inplace=True)

        return df
    
    
    def _process_model_correlations(self, model, partitions_df, targets_df):
        """Calculate correlation between model predictions and actual target values.
        Args:
            model: Name of the model whose predictions to process
            partitions_df: DataFrame containing paths to prediction files for each partition
            targets_df: DataFrame containing actual target values
        Returns:
            pd.DataFrame: DataFrame containing Pearson correlations for each target variable
                         with columns: [target_variable, pearson, model]
        """
        # Get paths to prediction files for this specific model
        predictions_filenames = partitions_df[partitions_df["model"] == model]["predictions_path"]
        
        # Combine predictions from all partitions into a single DataFrame
        # Each file contains predictions for a subset of targets
        predictions = pd.DataFrame().join(
            [pd.read_csv(f, index_col=0) for f in predictions_filenames], 
            how="outer"
        )
        
        # Calculate Pearson correlation between predictions and actual values
        cors = predictions.corrwith(targets_df)
        
        cors = (
            pd.DataFrame(cors)
            .reset_index()
            .rename(columns={
                "index": "target_variable",  # Name of the target variable
                0: "pearson"                 # Correlation coefficient
            })
        )
        
        cors["model"] = model
        
        return cors


    def _process_biomarker_matrix(self, df, index_col=0, test=False):
        """Process biomarker matrix data.
        Args:
            df: Biomarker matrix dataframe
            index_col: Index column number
            test: Test flag
        Returns:
            pd.DataFrame: Processed biomarker matrix
        """
        print("Start Processing Biomarker Matrix")
        df = self._clean_dataframe(df, index_col)
        # if test:
        #     df = df.iloc[:, :TEST_LIMIT]
        print(df.head())
        print("End Processing Biomarker Matrix")

        return df
    
    # TODO: restrict_targets, restrict_to, filter_columns are trying to accomplish the same thing
    # Remove either filter_columns or restrcit_targets, restrict_to in future
    def _process_dep_matrix(self, df, test=False, restrict_targets=False, restrict_to=None, filter_columns=None):
        """Process dependency matrix data.
        Args:
            df: Dependency matrix dataframe
            test: Test flag
            restrict_targets: If True, filters matrix to only specified target columns
            restrict_to: If restrict_targets is True, restrict to dependencies mentioned
            filter_columns: List of column names to filter the target matrix by
        Returns:
            pd.DataFrame: Processed dependency matrix
        """
        print("Start Processing Dependency Matrix")
        df = df.dropna(how="all", axis=0)
        df = df.dropna(how="all", axis=1)
        df.index.name = "Row.name"
        df = df.reset_index()

        if test:
            if restrict_targets:
                print("target restriction:", restrict_to)
                restrict_deps = restrict_to.split(",")
                df = df[["Row.name"] + restrict_deps]  # Keep Row.name and specified targets
            else:
                # If no specific targets, apply column filtering
                if filter_columns:
                    filter_columns.insert(0, "Row.name")
                    pattern = r'\b(' + '|'.join(re.escape(col) for col in filter_columns) + r')\b'
                    mask = df.columns.str.contains(pattern, regex=True)
                    df = df.loc[:, mask]
                else:
                    # If no filter columns provided, take first TEST_LIMIT+1 columns
                    # (+1 because first column is Row.name)
                    df = df.iloc[:, :TEST_LIMIT+1]
        elif restrict_targets:
            # If restrict targets is true, restrict the dataframe to the provided dependencies
            restrict_deps = restrict_to.split(",")
            df = df[["Row.name"] + restrict_deps]

        print(df.head())
        print("End Processing Dependency Matrix")

        return df
    

    def _process_column_name(self, col, feature_dataset_name):
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
            feature_name = f"{feature_label.replace('-', '_')}_({given_id})_{feature_dataset_name}"
        else:
            feature_label = col
            given_id = col
            feature_name = re.sub(r'[\s-]+', '_', col) + f"_{feature_dataset_name}"
        return feature_name, feature_label, given_id
    

    def _prepare_partition_paths(self, partitions_df, data_dir, features_suffix, predictions_suffix):
        partitions_df["path_prefix"] = (
            data_dir + "/" + partitions_df["model"] + "_" + 
            partitions_df["start"].map(str) + "_" + 
            partitions_df["end"].map(str) + "_"
        )
        partitions_df["feature_path"] = self.save_pref / (
            partitions_df["path_prefix"] + features_suffix
        )
        partitions_df["predictions_path"] = self.save_pref / (
            partitions_df["path_prefix"] + predictions_suffix
        )
        
        assert all(os.path.exists(f) for f in partitions_df["feature_path"])
        assert all(os.path.exists(f) for f in partitions_df["predictions_path"])
        
        return partitions_df
    

    def generate_feature_path_info(self, ipt_dicts):
        """Generate feature path information.
        Args:
            ipt_dicts: Input dictionaries
        Returns:
            pd.DataFrame: DF with feature dataset names and their corresponding file paths
        """
        dsets = []
        for dset_name, dset_value in ipt_dicts["data"].items():
            if dset_value["table_type"] == "feature":
                dsets.append(dset_name)
        fnames = [str(self.save_pref / (dset + ".csv")) for dset in dsets]

        df = pd.DataFrame({"dataset": dsets, "filename": fnames,})

        return df
    

    def process_dataset_for_feature_metadata(self, dataset_name, dataset_metadata, model_name, related_dset, test=False):
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
        _df = self.tc.get(dataset_metadata["taiga_id"])

        if (related_dset is None) or (
            (related_dset is not None) and dataset_name != related_dset
        ):
            _df = self._process_biomarker_matrix(_df, 0, test)
        print(f"Processing dataset: {dataset_name}")
        print(_df.head())

        for col in _df.columns:
            feature_name, feature_label, given_id = self._process_column_name(col, dataset_name)
            feature_metadata_rows.append({
                "model": model_name,
                "feature_name": feature_name,
                "feature_label": feature_label,
                "given_id": given_id,
                "taiga_id": dataset_metadata["taiga_id"],
                "dim_type": dataset_metadata["dim_type"]
            })

        return _df, pd.DataFrame(feature_metadata_rows)


    def generate_feature_metadata(self, ipt_dict, feature_path_info, related_dset, test=False):
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
        feature_metadata_df = pd.DataFrame(columns=["model", "feature_name", "feature_label", "given_id", "taiga_id", "dim_type"])

        if test:
            print("and truncating datasets for testing...")

        model_name = ipt_dict["model_name"]
        for dataset_name, dataset_metadata in ipt_dict["data"].items():
            if dataset_metadata["table_type"] not in ["feature", "relation"]:
                continue
                
            _df, dataset_info = self.process_dataset_for_feature_metadata(dataset_name, dataset_metadata, model_name, related_dset, test)
            # Concatenate the feature metadata for all datasets
            feature_metadata_df = pd.concat([feature_metadata_df, dataset_info], ignore_index=True)
            # Saving the downloaded feature matrix to a csv file
            _df.to_csv(feature_path_info.set_index("dataset").loc[dataset_name].filename)
        
        return feature_metadata_df


    def process_dependency_data(self, ipt_dict, test=False, restrict_targets=False, 
                              restrict_to=None, filter_columns=None):
        """Process dependency matrix data from Taiga and prepare it for model training.
        
        Args:
            ipt_dict: Dictionary containing input configuration with dataset metadata
            test: If True, limits data size for testing purposes
            restrict_targets: If True, filters matrix to only specified target columns
            restrict_to: Semicolon-separated string of target names to keep (used if restrict_targets=True)
            filter_columns: List of column names to filter the target matrix by
        
        Returns:
            pd.DataFrame: Processed dependency matrix ready for model training
        """
        print("Processing dependency data...")
        
        # Find the Taiga ID for the target matrix by looking through input dictionary
        # for the first entry with table_type="target_matrix"
        dep_matrix_taiga_id = next(
            (v.get("taiga_id") for v in ipt_dict["data"].values() 
             if v.get("table_type") == "target_matrix"), 
            None
        )
        
        df_dep = self.tc.get(dep_matrix_taiga_id)
        
        df_dep = self._process_dep_matrix(
            df_dep, test, restrict_targets, 
            restrict_to, filter_columns
        )
        
        # Save the processed matrix
        df_dep.to_feather(self.save_pref / FILES['target_matrix'])
        
        return df_dep


    def prepare_data(self, out_rel, ensemble_config):
        """Prepare data for model fitting."""
        daintree_bin = Path(PATHS['daintree_bin'])
        target_matrix = self.save_pref / FILES['target_matrix']
        target_matrix_filtered = self.save_pref / FILES['target_matrix_filtered']
        
        print('Running "prepare-y"...')
        try:
            subprocess.check_call([
                str(daintree_bin),
                "prepare-y",
                "--input", str(target_matrix),
                "--output", str(target_matrix_filtered),
            ])
        except subprocess.CalledProcessError as e:
            print(f"Error preparing target data: {e}")
            raise

        # Prepare feature (X) data
        print('Running "prepare-x"...')
        prep_x_cmd = [
            str(daintree_bin),
            "prepare-x",
            "--model-config", str(ensemble_config),
            "--targets", str(target_matrix_filtered),
            "--feature-info", str(self.save_pref / FILES['feature_path_info']),
            "--output", str(self.save_pref / FILES['feature_matrix']),
        ]
        
        if out_rel:
            prep_x_cmd.extend(["--output-related", "related"])
        
        try:
            subprocess.check_call(prep_x_cmd)
        except subprocess.CalledProcessError as e:
            print(f"Error preparing feature data: {e}")
            raise

    # TODO: Would like to add seeding in future
    def partition_inputs(self, dep_matrix, ensemble_config):
        """
        Divides the dependency matrix columns (genes) into chunks for parallel processing.
        For each model in the ensemble, it creates partitions based on the number of jobs specified
        in the model config.
        Args:
            dep_matrix: DataFrame containing the dependency matrix
            ensemble_config: Dictionary containing configuration for each model in the ensemble
                           Each model config must have a "Jobs" key specifying number of partitions
                            So if a model specifies 10 jobs and there are 100 genes:
                                - Creates 10 partitions of ~10 genes each
                                - First 9 partitions will have exactly 10 genes
                                - Last partition will have remaining genes (also 10 in this case)
        """
        # Get total number of genes (columns) to partition
        num_genes = dep_matrix.shape[1]
        start_indices = []
        end_indices = []
        models = []

        for model_name, model_config in ensemble_config.items():
            num_jobs = int(model_config["Jobs"])
            start_index = np.array(range(0, num_genes, num_jobs))
            end_index = start_index + num_jobs
            
            # Ensure the last partition includes any remaining genes
            end_index[-1] = num_genes
            
            # Store partition boundaries and model names
            start_indices.append(start_index)
            end_indices.append(end_index)
            models.append([model_name] * len(start_index))

        param_df = pd.DataFrame(
            {
                "start": np.concatenate(start_indices),  # Start index of each partition
                "end": np.concatenate(end_indices),      # End index of each partition
                "model": np.concatenate(models),         # Model name for each partition
            }
        )
        
        # Save partition information to CSV file
        param_df.to_csv(self.save_pref / FILES['partitions'], index=False)


    def gather_ensemble_tasks(self, features="X.ftr", targets="target_matrix.ftr", top_n=50):
        """Gather and process ensemble model results, combining predictions and feature importances.
        Args:
            features: Path to feature matrix file (default: "X.ftr")
            targets: Path to target matrix file (default: "target_matrix.ftr")
            top_n: Number of top features to analyze for correlations (default: 50)
        Returns:
            tuple: (ensemble_df, predictions_df)
                - ensemble_df: DataFrame with model performance and feature importance details
                - predictions_df: DataFrame containing all model predictions
        """
        features_df = pd.read_feather(self.save_pref / features)
        targets_df = pd.read_feather(self.save_pref / targets)
        targets_df = targets_df.set_index("Row.name")
        partitions_df = pd.read_csv(self.save_pref / FILES['partitions'])
        partitions_df = self._prepare_partition_paths(
            partitions_df, "data", "features.csv", "predictions.csv"
        )
        
        # Combine feature importance information from all partitions
        all_features = pd.concat(
            [pd.read_csv(f) for f in partitions_df["feature_path"]], 
            ignore_index=True
        )
        all_features.drop(["score0", "score1", "best"], axis=1, inplace=True)
        
        # Combine predictions from all partitions
        predictions = pd.DataFrame().join(
            [pd.read_csv(f, index_col=0) for f in partitions_df["predictions_path"]], 
            how="outer"
        )
        
        # Calculate correlations between predictions and actual values for each model
        all_cors = []
        for model in all_features["model"].unique():
            cors = self._process_model_correlations(model, partitions_df, targets_df)
            all_cors.append(cors)
        
        # Combine correlation results and merge with feature importance data
        all_cors = pd.concat(all_cors, ignore_index=True)
        ensemble = all_features.merge(all_cors, on=["target_variable", "model"])
        
        # Identify best performing model for each target variable
        ensemble = ensemble.copy()
        ranked_pearson = ensemble.groupby("target_variable")["pearson"].rank(ascending=False)
        ensemble["best"] = (ranked_pearson == 1)
        
        ensb_cols = ["target_variable", "model", "pearson", "best"]
        
        for index, row in ensemble.iterrows():
            target_variable = row['target_variable']
            y = targets_df[target_variable]  # Actual values for this target
            
            # Process top N(50 at this moment) features
            for i in range(top_n):
                feature_col = f'feature{i}'
                feature_name = row[feature_col]
                
                # Calculate correlation if feature exists in feature matrix
                if feature_name in features_df.columns:
                    corr = calculate_feature_correlations(features_df[feature_name], y)
                    ensemble.loc[index, f'{feature_col}_correlation'] = corr
        
        # Build final column list including feature information
        for i in range(top_n):
            feature_cols = [
                f"feature{i}",                # Feature name
                f"feature{i}_importance",     # Feature importance score
                f"feature{i}_correlation"     # Feature correlation with target
            ]
            ensb_cols.extend(feature_cols)
        
        # Sort and select final columns
        ensemble = ensemble.sort_values(["target_variable", "model"])[ensb_cols]
        
        return ensemble, predictions


class SparklesRunner:
    def __init__(self, save_pref, config_fname, related, sparkles_config):
        self.save_pref = Path(save_pref)
        self.config_fname = Path(config_fname)
        self.related = related
        self.sparkles_config = sparkles_config
        self.dt_hash = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        self.sparkles_path = PATHS['sparkles_bin']

    def _build_sparkles_command(self):
        base_cmd = [
            self.sparkles_path,
            "--config", self.sparkles_config,
            "sub",
            "-i", CONTAINER['image'],
            "-u", main.__file__,
            "-u", f"{self.save_pref}/{FILES['target_matrix']}:target.ftr",
            "-u", f"{self.save_pref}/{self.config_fname.name}:model-config.yaml",
            "-u", f"{self.save_pref}/{FILES['feature_matrix']}:X.ftr",
            "-u", f"{self.save_pref}/{FILES['feature_metadata']}:X_feature_metadata.ftr",
            "-u", f"{self.save_pref}/{FILES['valid_samples']}:X_valid_samples.ftr",
            "-u", f"{self.save_pref}/{FILES['partitions']}",
            "--params", f"{self.save_pref}/{FILES['partitions']}",
            "--skipifexists",
            "--nodes", "100",  # This could also be moved to config maybe?
            "-n", f"ensemble_{self.dt_hash}",
            PATHS['daintree_bin'], "fit-model",
            "--x", "X.ftr",
            "--y", "target.ftr",
            "--model-config", "model-config.yaml",
            "--n-folds", str(MODEL['n_folds']),
            "--target-range", "{start}", "{end}",
            "--model", "{model}"
        ]
        
        if self.related:
            base_cmd.extend([
                "-u", f"{self.save_pref}/related.ftr:related.ftr",
                "--related-table", "related.ftr"
            ])
        return base_cmd

    def _watch_jobs(self):
        subprocess.check_call([self.sparkles_path, "--config", self.sparkles_config, 
                             "watch", f"ensemble_{self.dt_hash}", "--loglive"])

    def _reset_jobs(self):
        subprocess.check_call([self.sparkles_path, "--config", self.sparkles_config, 
                             "reset", f"ensemble_{self.dt_hash}"])

    def _validate_jobs_complete(self):
        """Validate that all expected job outputs exist"""
        with open(f"{self.save_pref}/completed_jobs.txt") as f:
            completed_jobs = {l.split("/")[-1].strip() for l in f.readlines()}

        partitions = pd.read_csv(f"{self.save_pref}/partitions.csv")
        partitions["path_prefix"] = (
            partitions["model"]
            + "_"
            + partitions["start"].map(str)
            + "_"
            + partitions["end"].map(str)
            + "_"
        )
        partitions["feature_path"] = partitions["path_prefix"] + "features.csv"
        partitions["predictions_path"] = partitions["path_prefix"] + "predictions.csv"

        assert len(set(partitions["feature_path"]) - completed_jobs) == 0, "Missing feature files"
        assert len(set(partitions["predictions_path"]) - completed_jobs) == 0, "Missing prediction files"

    def _process_completed_jobs(self):
        """Process completed jobs and copy results to local directory.
        """
        os.makedirs(f"{self.save_pref}/data", exist_ok=True)
        default_url_prefix = self._get_default_url_prefix()
        
        self._authenticate_gcloud()
        
        completed_jobs = subprocess.check_output([
            "/google-cloud-sdk/bin/gcloud", 
            "storage", 
            "ls", 
            f"{default_url_prefix}/ensemble_{self.dt_hash}/*/*.csv"
        ]).decode()
        
        self._save_completed_jobs(completed_jobs)
        self._validate_jobs_complete()
        self._copy_results_to_local()

    def _get_default_url_prefix(self):
        """Get the default URL prefix from the sparkles config file.
        """
        with open(self.sparkles_config, 'r') as f:
            for line in f:
                if 'default_url_prefix' in line:
                    return line.split('=')[1].strip()

    def _authenticate_gcloud(self):
        subprocess.check_call([
            "/google-cloud-sdk/bin/gcloud", 
            "auth", 
            "activate-service-account", 
            "--key-file", 
            PATHS['service_account']
        ])

    def _save_completed_jobs(self, completed_jobs):
        with open(f"{self.save_pref}/completed_jobs.txt", 'w') as f:
            f.write(completed_jobs)

    def _copy_results_to_local(self):
        """Copy results from Google Cloud Storage to local directory.
        """
        default_url_prefix = self._get_default_url_prefix()
        subprocess.check_call([
            "/google-cloud-sdk/bin/gcloud",
            "storage",
            "cp",
            f"{default_url_prefix}/ensemble_{self.dt_hash}/*/*.csv",
            f"{self.save_pref}/data"
        ])
    
    def run(self):
        cmd = self._build_sparkles_command()
        print(f"Running sparkles with command: {cmd}")
        subprocess.check_call(cmd)
        print("sparkles run complete")
        return self.dt_hash

    def validate(self):
        print("Validating sparkles run...")
        self._watch_jobs()
        self._reset_jobs()
        self._watch_jobs()
        self._process_completed_jobs()


class ConfigManager:
    def __init__(self, save_pref):
        """Initialize ConfigManager with save directory path."""
        self.save_pref = Path(save_pref)


    def save_config(self, config, filename):
        """Save model configuration to a file.
        Args:
            config: Configuration dictionary
            filename: Name of the file to save to
        Returns:
            str: Path to saved configuration file
        """
        config_path = self.save_pref / filename
        with open(config_path, "w") as f:
            yaml.dump(config, f, sort_keys=True)

        return str(config_path)
    

    def load_config(self, config_path):
        """Load model configuration from a file.
        Args:
            config_path: Path to the configuration file
        Returns:
            dict: Loaded configuration
        """
        with open(config_path, "r") as f:
            return yaml.load(f, yaml.SafeLoader)
        

    def generate_model_config(self, input_dict, relation="All"):
        """Generate model configuration.
        Args:
            input_dict: Input dictionary containing model configuration
            relation: Relation type
        Returns:
            dict: Generated model configuration
        """
        model_name = input_dict["model_name"]
        data = input_dict["data"]

        features = [
            key for key, value in data.items()
            if value.get("table_type") == "feature"
        ]
        required = [
            key for key, value in data.items()
            if value.get("table_type") == "feature" and value.get("required", False)
        ]
        exempt = [
            key for key, value in data.items()
            if value.get("table_type") == "feature" and value.get("exempt", False)
        ]
        relation = next(
            (value["relation"] for value in data.values() if value.get("table_type") == "target_matrix"),
            "All"
        )
        model_config = {
            "Features": features,
            "Required": required,
            "Relation": relation,
            "Jobs": MODEL['default_jobs']  # Use config value
        }
        
        if relation == "MatchRelated":
            related = next(
                (key for key, value in data.items() if value.get("table_type") == "relation"),
                None
            )
            if related:
                model_config["Related"] = related
        
        if exempt:
            model_config["Exempt"] = exempt

        return {model_name: model_config}


    def check_file_locs(self, ipt, config):
        """Check if all files in config exist in input.
        Args:
            ipt: Input configuration dictionary(Input json file)
            config: Model configuration dictionary(Generated model-config.yaml file
        """
        ipt_features = list(ipt["data"].keys())
        for model_name, model_config in config.items():
            f_set = set(model_config.get("Features", []) + model_config.get("Required", []))
            if model_config.get("Relation") not in ["All", "MatchTarget"]:
                f_set.add(model_config.get("Related", ""))
            features = list(f_set)
            for f in features:
                assert f in ipt_features, f"Feature {f} in model config file does not have corresponding input in {model_name}"


    def load_input_config(self, input_files):
        """Load and validate input configuration.
        Args:
            input_files: Path to input configuration file
            
        Returns:
            dict: Validated configuration dictionary
        """
        print("Loading input json file...")
        with open(input_files, "r") as f:
            config = json.load(f)
        
        assert "model_name" in config, "Config missing required field 'model_name'"
        assert "screen_name" in config, "Config missing required field 'screen_name'"
        assert "data" in config, "Config missing required field 'data'"
        
        for dataset_name, dataset_config in config["data"].items():
            assert "taiga_id" in dataset_config, f"Dataset {dataset_name} missing required field 'taiga_id'"
            assert "table_type" in dataset_config, f"Dataset {dataset_name} missing required field 'table_type'"
            
            assert dataset_config["table_type"] in ["target_matrix", "feature", "relation"], \
                f"Dataset {dataset_name} has invalid table_type: {dataset_config['table_type']}"
        
        # Ensure at least one target_matrix exists
        target_matrices = [name for name, config in config["data"].items() 
                          if config["table_type"] == "target_matrix"]
        assert len(target_matrices) > 0, "Config must include at least one target_matrix"
        
        return config


    def setup_ensemble_config(self, ensemble_config, ipt_dict):
        """Setup and validate ensemble configuration.
        Args:
            ensemble_config: Path to existing ensemble config or None
            ipt_dict: Input configuration dictionary
            
        Returns:
            tuple: (config_path, config_dict)
        """
        print("Setting up ensemble config...")
        if not ensemble_config:
            config = self.generate_model_config(ipt_dict, relation="All")
            model_config_name = f"model-config_temp_{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.yaml"
            config_path = self.save_config(config, model_config_name)
        else:
            config_path = ensemble_config

        config_dict = self.load_config(config_path)
        self.check_file_locs(ipt_dict, config_dict)

        return config_path, config_dict

    #TODO: This is kind of a useless function at this moment that was copied over from
    # the old code since among the input configs there has not been any MatchTarget
    # and such. So out_rel is always False and related_dset is None. Maybe in future
    # we will have a use for it. 
    def determine_relations(self, config_dict):
        """Determine relation settings from config.
        Args:
            config_dict: Configuration dictionary
        Returns:
            tuple: (out_rel, related_dset)
        """
        print("Processing relations...")
        relations = [m[1]["Relation"] for m in config_dict.items()]
        out_rel = len(set(relations).difference(set(["All", "MatchTarget"]))) > 0
        related_dset = None
        if out_rel:
            related_dset = list(set(relations).difference(set(["All", "MatchTarget"])))[0]

        return out_rel, related_dset


    def create_output_config(self, input_config, 
                           feature_metadata_id, ensemble_id, prediction_matrix_id):
        """Create output configuration JSON file.
        
        Args:
            model_name: Name of the model
            screen_name: Name of the screen
            input_config: Input configuration
            feature_metadata_id: Taiga ID for feature metadata csv file
            ensemble_id: Taiga ID for ensemble csv file
            prediction_matrix_id: Taiga ID for prediction matrix csv file
            
        Returns:
            dict: Daintree output configuration dictionary
        """
        model_name = input_config["model_name"]
        screen_name = input_config["screen_name"]
        data = input_config["data"]

        config = {
            model_name: {
                "input": {
                    "model_name": model_name,
                    "screen_name": screen_name,
                    "data": data
                },
                "output": {
                    "ensemble_taiga_id": ensemble_id,
                    "feature_metadata_taiga_id": feature_metadata_id,
                    "prediction_matrix_taiga_id": prediction_matrix_id
                }
            }
        }
        return config


class TaigaUploader:
    def __init__(self, save_pref, upload_to_taiga, config_manager):
        self.save_pref = Path(save_pref)
        self.upload_to_taiga = upload_to_taiga
        self.config_manager = config_manager

    def upload_results(self, ipt_dict):
        """Upload results to Taiga and create output config file.
        
        Args:
            ipt_dict: Input configuration dictionary
            
        Returns:
            tuple: (feature_metadata_taiga_info, ensemble_taiga_info, predictions_taiga_info)
        """
        model_name = ipt_dict["model_name"]
        screen_name = ipt_dict["screen_name"]

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

        output_config = self.config_manager.create_output_config(
            input_config=ipt_dict,
            feature_metadata_id=feature_metadata_taiga_info,
            ensemble_id=ensemble_taiga_info,
            prediction_matrix_id=predictions_taiga_info
        )

        output_config_dir = self.save_pref / "output_config_files"
        output_config_dir.mkdir(parents=True, exist_ok=True)
        # This could probably be hardcoded and put in a config file.
        # However, I am keeping it this way for now to make it more flexible.
        output_config_filename = f"OutputConfig{model_name}{screen_name}.json"
        output_config_file = output_config_dir / output_config_filename

        with open(output_config_file, 'w') as f:
            json.dump(output_config, f, indent=4)
        print(f"Created output config file: {output_config_file}")

        return feature_metadata_taiga_info, ensemble_taiga_info, predictions_taiga_info


class ModelFitter:
    def __init__(self, input_files, ensemble_config, sparkles_config, 
                 save_dir=None, test=False, skipfit=False, 
                 upload_to_taiga=None, 
                 restrict_targets=False, restrict_to=None,
                 filter_columns=None):
        self.save_pref = Path(save_dir) if save_dir else Path.cwd()
        self.input_files = input_files
        self.ensemble_config = ensemble_config
        self.sparkles_config = sparkles_config
        self.test = test
        self.skipfit = skipfit
        self.upload_to_taiga = upload_to_taiga
        self.restrict_targets = restrict_targets
        self.restrict_to = restrict_to
        self.filter_columns = filter_columns.split(",") if filter_columns else None
        self.save_pref.mkdir(parents=True, exist_ok=True)
        
        self.data_processor = DataProcessor(self.save_pref)
        self.config_manager = ConfigManager(self.save_pref)
        self.taiga_uploader = TaigaUploader(self.save_pref, upload_to_taiga, self.config_manager)
        
    def _run_model_fitting(self, ipt_dict):
        """Execute model fitting if not skipped."""
        print("submitting fit jobs...")
        sparkles_runner = SparklesRunner(
            self.save_pref, 
            self.ensemble_config, 
            self.out_rel, 
            self.sparkles_config
        )
        dt_hash = sparkles_runner.run()
        sparkles_runner.validate()
        
        model_name = ipt_dict["model_name"]
        screen_name = ipt_dict["screen_name"]
        # This could probably be hardcoded and put in a config file. However, I am 
        # keeping it this way for now to make it more flexible.
        ensemble_filename = f"Ensemble{model_name}{screen_name}.csv"
        predictions_filename = f"Predictions{model_name}{screen_name}.csv"
        
        df_ensemble, df_predictions = self.data_processor.gather_ensemble_tasks(
            features=str(self.save_pref / FILES['feature_matrix']), 
            targets=str(self.save_pref / FILES['target_matrix']), 
            top_n=MODEL['top_n_features']
        )
        df_ensemble.to_csv(self.save_pref / ensemble_filename, index=False)
        df_predictions.to_csv(self.save_pref / predictions_filename)

        if self.upload_to_taiga:
            self.taiga_uploader.upload_results(ipt_dict)


    def run(self):
        # Load input configuration
        ipt_dict = self.config_manager.load_input_config(self.input_files)

        # Setup and validate ensemble configuration
        self.ensemble_config, config_dict = self.config_manager.setup_ensemble_config(
            self.ensemble_config, ipt_dict
        )

        print("Generating feature index and files...")
        feature_path_info = self.data_processor.generate_feature_path_info(ipt_dict)

        self.out_rel, related_dset = self.config_manager.determine_relations(config_dict)

        model_name = ipt_dict["model_name"]
        screen_name = ipt_dict["screen_name"]

        # This could probably be hardcoded and put in a config file.
        # However, I am keeping it this way for now to make it more flexible.
        feature_metadata_filename = f"FeatureMetadata{model_name}{screen_name}.csv"
        feature_metadata_df = self.data_processor.generate_feature_metadata(
            ipt_dict, feature_path_info, related_dset, self.test
        )

        # Process dependency data
        df_dep = self.data_processor.process_dependency_data(
            ipt_dict, self.test, self.restrict_targets, 
            self.restrict_to, self.filter_columns
        )

        # Save feature matrix file path information
        feature_path_info.to_csv(self.save_pref / "feature_path_info.csv")
        feature_metadata_df.to_csv(self.save_pref / feature_metadata_filename)

        # Prepare data
        self.data_processor.prepare_data(self.out_rel, self.ensemble_config)

        print("Partitioning inputs...")
        self.data_processor.partition_inputs(df_dep, config_dict)

        if not self.skipfit:
            self._run_model_fitting(ipt_dict)
        else:
            print("skipping fitting and ending run")


@cli.command()
@click.option(
    "--input-files",
    required=True,
    help="JSON file containing the set of files for prediction",
)
@click.option(
    "--ensemble-config",
    required=False,
    help='YAML file for model configuration. If not provided, will be auto-generated.',
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
    help="If restrict_targets is true, provide semicolon-separated list of dependencies",
)
@click.option(
    "--filter-columns",
    default=None,
    type=str,
    help="Comma separated list of names to filter target columns. If not provided, uses TEST_LIMIT from config.py",
)
def collect_and_fit(
    input_files,
    ensemble_config,
    sparkles_config,
    save_dir=None,
    test=False,
    skipfit=False,
    upload_to_taiga=None,
    restrict_targets=False,
    restrict_to=None,
    filter_columns=None,
):
    """Run model fitting with either provided or auto-generated config."""
    save_pref = Path(save_dir) if save_dir else Path.cwd()
    print(f"Save directory path: {save_pref}")
    save_pref.mkdir(parents=True, exist_ok=True)

    # Run the model fitting
    model_fitter = ModelFitter(
        input_files,
        ensemble_config,
        sparkles_config,
        save_dir=str(save_pref),
        test=test,
        skipfit=skipfit,
        upload_to_taiga=upload_to_taiga,
        restrict_targets=restrict_targets,
        restrict_to=restrict_to,
        filter_columns=filter_columns,
    )
    model_fitter.run()


if __name__ == "__main__":
    print("Starting Daintree CLI Instance Iteration 3")
    cli()
    