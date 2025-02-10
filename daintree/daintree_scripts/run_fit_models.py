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

from taigapy import create_taiga_client_v3
from daintree_package import main

from config import TEST_LIMIT, filter_columns_gene, filter_columns_oncref
from utils import calculate_feature_correlations, update_taiga

# CLI Setup
@click.group()
def cli():
    print("Hello, Aquib!")
    pass


class DataProcessor:
    def __init__(self, save_pref):
        self.save_pref = Path(save_pref)

    def _load_and_prepare_data(self, features, targets, partitions):
        features_df = pd.read_feather(self.save_pref / features)
        targets_df = pd.read_feather(self.save_pref / targets)
        targets_df = targets_df.set_index("Row.name")
        partitions_df = pd.read_csv(self.save_pref / partitions)
        return features_df, targets_df, partitions_df

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

    def _process_model_correlations(self, model, partitions_df, targets_df):
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

    def gather_ensemble_tasks(self, features="X.ftr", targets="target_matrix.ftr", 
                            data_dir="data", partitions="partitions.csv", 
                            features_suffix="features.csv", predictions_suffix="predictions.csv", 
                            top_n=50):
        features_df, targets_df, partitions_df = self._load_and_prepare_data(
            features, targets, partitions
        )
        partitions_df = self._prepare_partition_paths(
            partitions_df, data_dir, features_suffix, predictions_suffix
        )
        
        all_features = pd.concat(
            [pd.read_csv(f) for f in partitions_df["feature_path"]], 
            ignore_index=True
        )
        all_features.drop(["score0", "score1", "best"], axis=1, inplace=True)
        
        predictions = pd.DataFrame().join(
            [pd.read_csv(f, index_col=0) for f in partitions_df["predictions_path"]], 
            how="outer"
        )
        
        all_cors = []
        for model in all_features["model"].unique():
            cors = self._process_model_correlations(model, partitions_df, targets_df)
            all_cors.append(cors)
        
        all_cors = pd.concat(all_cors, ignore_index=True)
        ensemble = all_features.merge(all_cors, on=["target_variable", "model"])
        
        ensemble = ensemble.copy()
        ranked_pearson = ensemble.groupby("target_variable")["pearson"].rank(ascending=False)
        ensemble["best"] = (ranked_pearson == 1)
        
        ensb_cols = ["target_variable", "model", "pearson", "best"]
        
        for index, row in ensemble.iterrows():
            target_variable = row['target_variable']
            y = targets_df[target_variable]
            
            for i in range(top_n):
                feature_col = f'feature{i}'
                feature_name = row[feature_col]
                
                if feature_name in features_df.columns:
                    corr = calculate_feature_correlations(features_df[feature_name], y)
                    ensemble.loc[index, f'{feature_col}_correlation'] = corr
        
        for i in range(top_n):
            feature_cols = [
                f"feature{i}", 
                f"feature{i}_importance",
                f"feature{i}_correlation"
            ]
            ensb_cols.extend(feature_cols)
        
        ensemble = ensemble.sort_values(["target_variable", "model"])[ensb_cols]
        
        return ensemble, predictions
    
    def _clean_dataframe(self, df, index_col):
        """Clean and sort the dataframe.
        
        Args:
            df: Input dataframe
            index_col: Column to use as index
        """
        df.sort_index(inplace=True, axis=1)

        if index_col is None:
            df.sort_values(df.columns.tolist(), inplace=True)
        else:
            df.sort_index(inplace=True)

        return df

    def _process_biomarker_matrix(self, df, index_col=0, test=False):
        """Process biomarker matrix data.
        
        Args:
            df: Input dataframe
            index_col: Column to use as index
            test: Whether to limit data for testing
        """
        df = self._clean_dataframe(df, index_col)

        print("Start Processing Biomarker Matrix")
        if test:
            df = df.iloc[:, :]
        print(df.head())
        print("End Processing Biomarker Matrix")
        return df

    def _process_dep_matrix(self, df, test=False, restrict_targets=False, restrict_to=None, filter_columns=None):
        """Process dependency matrix data.
        
        Args:
            df: Input dataframe
            test: Whether to limit data for testing
            restrict_targets: Whether to restrict to specific targets
            restrict_to: Target restrictions if restrict_targets is True
        """
        # Drop null rows and columns
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
                if filter_columns:
                    filter_columns.insert(0, "Row.name")

                    # Create a regex pattern with word boundaries
                    pattern = r'\b(' + '|'.join(re.escape(col) for col in filter_columns) + r')\b'
                    
                    # Create a boolean mask for columns that contain any of the filter_columns as whole words
                    mask = df.columns.str.contains(pattern, regex=True)

                    df = df.loc[:, mask]
                else:
                    df = df.iloc[:, :TEST_LIMIT+1] # +1 is due to the Row.name column
        elif restrict_targets:
            restrict_deps = restrict_to.split(";")
            df = df[["Row.name"] + restrict_deps]

        print("Start Processing Dependency Matrix")
        print(df)
        print("End Processing Dependency Matrix")
        return df

    def _process_column_name(self, col, feature_dataset_name):
        """Process column name to generate feature name, label, and ID.
        
        Args:
            col: Column name to process
            feature_dataset_name: Name of the feature dataset
            
        Returns:
            tuple: (feature_name, feature_label, given_id)
        """
        match = re.match(r"(.+?) \((\d+)\)", col)
        if match:
            feature_label, given_id = match.groups()
            feature_name = f"{feature_label.replace('-', '_')}_({given_id})_{feature_dataset_name}"
        else:
            feature_label = col
            given_id = col
            feature_name = re.sub(r'[\s-]+', '_', col) + f"_{feature_dataset_name}"
        return feature_name, feature_label, given_id

    def _partition_inputs(self, dep_matrix, ensemble_config, save_pref, out_name="partitions.csv"):
        """Partition inputs for parallel processing.
        
        Args:
            dep_matrix: Dependency matrix
            ensemble_config: Configuration for ensemble models
            save_pref: Save directory path
            out_name: Output filename
        """
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

    def _generate_config(self, input_dict, relation="All"):
        """Generate configuration for model.
        
        Args:
            input_dict: Input dictionary containing model configuration
            relation: Relation type
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
            "Jobs": 10
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

    def _generate_feature_info(self, ipt_dicts, save_pref):
        """Generate feature information.
        
        Args:
            ipt_dicts: Input dictionaries
            save_pref: Save directory path
        """
        dsets = []
        for dset_name, dset_value in ipt_dicts["data"].items():
            if dset_value["table_type"] == "feature":
                dsets.append(dset_name)
        fnames = [str(save_pref / (dset + ".csv")) for dset in dsets]

        df = pd.DataFrame({"dataset": dsets, "filename": fnames,})
        return df


class SparklesRunner:
    def __init__(self, save_pref, config_fname, related, sparkles_config):
        self.save_pref = Path(save_pref)
        self.config_fname = Path(config_fname)
        self.related = related
        self.sparkles_config = sparkles_config
        self.dt_hash = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        self.sparkles_path = "/install/sparkles/bin/sparkles"

    def _build_sparkles_command(self):
        base_cmd = [
            self.sparkles_path,
            "--config", self.sparkles_config,
            "sub",
            "-i", "us.gcr.io/broad-achilles/daintree-sparkles:v1",
            "-u", main.__file__,
            "-u", f"{self.save_pref}/target_matrix.ftr:target.ftr",
            "-u", f"{self.save_pref}/{self.config_fname.name}:model-config.yaml",
            "-u", f"{self.save_pref}/X.ftr:X.ftr",
            "-u", f"{self.save_pref}/X_feature_metadata.ftr:X_feature_metadata.ftr",
            "-u", f"{self.save_pref}/X_valid_samples.ftr:X_valid_samples.ftr",
            "-u", f"{self.save_pref}/partitions.csv",
            "--params", f"{self.save_pref}/partitions.csv",
            "--skipifexists",
            "--nodes", "100",
            "-n", f"ensemble_{self.dt_hash}",
            "/install/depmap-py/bin/daintree", "fit-model",
            "--x", "X.ftr",
            "--y", "target.ftr",
            "--model-config", "model-config.yaml",
            "--n-folds", "5",
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
            "/root/.sparkles-cache/service-keys/broad-achilles.json"
        ])

    def _save_completed_jobs(self, completed_jobs):
        with open(f"{self.save_pref}/completed_jobs.txt", 'w') as f:
            f.write(completed_jobs)

    def _copy_results_to_local(self):
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


class ModelFitter:
    def __init__(self, input_files, ensemble_config, sparkles_config, 
                 save_dir=None, test=False, skipfit=False, 
                 upload_to_taiga=None, 
                 restrict_targets=False, restrict_to=None,
                 filter_columns=None):
        self.tc = create_taiga_client_v3()
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

    def _check_file_locs(self, ipt, config):
        ipt_features = list(ipt["data"].keys())
        for model_name, model_config in config.items():
            f_set = set(model_config.get("Features", []) + model_config.get("Required", []))
            if model_config.get("Relation") not in ["All", "MatchTarget"]:
                f_set.add(model_config.get("Related", ""))
            features = list(f_set)
            for f in features:
                assert f in ipt_features, f"Feature {f} in model config file does not have corresponding input in {model_name}"

    def _create_output_config(self, model_name, screen_name, input_config, 
                            feature_metadata_id, ensemble_id, prediction_matrix_id):
        """Create output configuration JSON file.
        
        Args:
            model_name: Name of the model
            screen_name: Name of the screen
            input_config: Input configuration
            feature_metadata_id: Taiga ID for feature metadata
            ensemble_id: Taiga ID for ensemble
            prediction_matrix_id: Taiga ID for prediction matrix
            
        Returns:
            dict: Output configuration dictionary
        """
        config = {
            model_name: {
                "input": {
                    "model_name": model_name,
                    "screen_name": screen_name,
                    "data": input_config["data"]
                },
                "output": {
                    "ensemble_taiga_id": ensemble_id,
                    "feature_metadata_taiga_id": feature_metadata_id,
                    "prediction_matrix_taiga_id": prediction_matrix_id
                }
            }
        }
        return config

    def _upload_results_to_taiga(self, model_name, screen_name, ipt_dict):
        """Upload results to Taiga and create output config file.
        
        Args:
            model_name: Name of the model
            screen_name: Name of the screen
            ipt_dict: Input configuration dictionary
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

        output_config = self._create_output_config(
            model_name=model_name,
            screen_name=screen_name,
            input_config=ipt_dict,
            feature_metadata_id=feature_metadata_taiga_info,
            ensemble_id=ensemble_taiga_info,
            prediction_matrix_id=predictions_taiga_info
        )

        output_config_dir = self.save_pref / "output_config_files"
        output_config_dir.mkdir(parents=True, exist_ok=True)
        output_config_filename = f"OutputConfig{model_name}{screen_name}.json"
        output_config_file = output_config_dir / output_config_filename

        with open(output_config_file, 'w') as f:
            json.dump(output_config, f, indent=4)
        print(f"Created output config file: {output_config_file}")

        return feature_metadata_taiga_info, ensemble_taiga_info, predictions_taiga_info

    def _partition_inputs(self, dep_matrix, ensemble_config, save_pref, out_name="partitions.csv"):
        """Partition inputs for parallel processing.
        
        Args:
            dep_matrix: Dependency matrix
            ensemble_config: Configuration for ensemble models
            save_pref: Save directory path
            out_name: Output filename
        """
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

    def _generate_config(self, input_dict, relation="All"):
        """Generate configuration for model.
        
        Args:
            input_dict: Input dictionary containing model configuration
            relation: Relation type
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
            "Jobs": 10
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

    def _generate_feature_info(self, ipt_dicts, save_pref):
        """Generate feature information.
        
        Args:
            ipt_dicts: Input dictionaries
            save_pref: Save directory path
        """
        dsets = []
        for dset_name, dset_value in ipt_dicts["data"].items():
            if dset_value["table_type"] == "feature":
                dsets.append(dset_name)
        fnames = [str(save_pref / (dset + ".csv")) for dset in dsets]

        df = pd.DataFrame({"dataset": dsets, "filename": fnames,})
        return df

    def run(self):
        print("loading input files...")
        with open(self.input_files, "r") as f:
            ipt_dict = json.load(f)

        # Auto-generate config if not provided
        if not self.ensemble_config:
            config = self._generate_config(ipt_dict, relation="All")
            model_config_name = f"model-config_temp_{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.yaml"
            config_path = self.save_pref / model_config_name
            
            with open(config_path, "w") as f:
                yaml.dump(config, f, sort_keys=True)
            
            self.ensemble_config = str(config_path)

        with open(self.ensemble_config, "r") as f:
            config_dict = yaml.load(f, yaml.SafeLoader)

        print("validating inputs...")
        self._check_file_locs(ipt_dict, config_dict)
        assert (
            len([v for v in ipt_dict["data"].values() if v.get("table_type") == "target_matrix"]) == 1
        ), "Exactly one dataset labeled 'target_matrix' is required"

        print("generating feature index and files...")
        feature_info = self._generate_feature_info(ipt_dict, self.save_pref)

        print("processing relations...")
        # determine whether to provide output_related
        relations = [m[1]["Relation"] for m in config_dict.items()]
        out_rel = len(set(relations).difference(set(["All", "MatchTarget"]))) > 0
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

        if self.test:
            print("and truncating datasets for testing...")
        for dataset_name, dataset_metadata in ipt_dict["data"].items():
            if dataset_metadata["table_type"] not in ["feature", "relation"]:
                continue
            _df = self.tc.get(dataset_metadata["taiga_id"])

            if (related_dset is None) or (
                (related_dset is not None) and dataset_name != related_dset
            ):
                _df = self.data_processor._process_biomarker_matrix(_df, 0, self.test)
            print(f"processed dataset: {dataset_name}")
            print(_df.head())

            for col in _df.columns:
                feature_name, feature_label, given_id = self.data_processor._process_column_name(col, dataset_name)
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
        dep_matrix_taiga_id = next((v.get("taiga_id") for v in ipt_dict["data"].values() if v.get("table_type") == "target_matrix"), None)
        print(f"dep_matrix_taiga_id: {dep_matrix_taiga_id}")
        df_dep = self.tc.get(dep_matrix_taiga_id)
        df_dep = self.data_processor._process_dep_matrix(df_dep, self.test, self.restrict_targets, self.restrict_to, self.filter_columns)
        df_dep.to_feather(self.save_pref / "target_matrix.ftr")

        feature_info.to_csv(self.save_pref / "feature_info.csv")
        feature_info_df.to_csv(self.save_pref / feature_metadata_filename)

        print('running "prepare_y"...')
        subprocess.check_call(
            [
                "/install/depmap-py/bin/daintree", 
                "prepare-y",
                "--input",
                str(self.save_pref / "target_matrix.ftr"),
                "--output",
                str(self.save_pref / "target_matrix_filtered.ftr"),
            ]
        )

        print('running "prepare_x"...')
        prep_x_cmd = [
            "/install/depmap-py/bin/daintree",
            "prepare-x",
            "--model-config",
            str(self.ensemble_config),
            "--targets",
            str(self.save_pref / "target_matrix_filtered.ftr"),
            "--feature-info",
            str(self.save_pref / "feature_info.csv"),
            "--output",
            str(self.save_pref / "X.ftr"),
        ]
        if out_rel:
            prep_x_cmd.extend(["--output-related", "related"])
        subprocess.check_call(prep_x_cmd)

        print("partitioning inputs...")
        self._partition_inputs(df_dep, config_dict, self.save_pref, out_name="partitions.csv")

        if not self.skipfit:
            print("submitting fit jobs...")
            sparkles_runner = SparklesRunner(
                self.save_pref, 
                self.ensemble_config, 
                out_rel, 
                self.sparkles_config
            )
            dt_hash = sparkles_runner.run()
            sparkles_runner.validate()
            
            data_processor = DataProcessor(self.save_pref)
            df_ensemble, df_predictions = data_processor.gather_ensemble_tasks(
                features=str(self.save_pref / "X.ftr"), 
                targets=str(self.save_pref / "target_matrix.ftr"), 
                top_n=50
            )
            df_ensemble.to_csv(self.save_pref / ensemble_filename, index=False)
            predictions_filename = f"Predictions{model_name}{screen_name}.csv"
            df_predictions.to_csv(self.save_pref / predictions_filename)

            if self.upload_to_taiga:
                self._upload_results_to_taiga(
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
    print("Starting Daintree CLI Instance Iteration 6")
    cli()
    