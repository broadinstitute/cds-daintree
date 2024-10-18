import pandas as pd
import numpy as np
import re

from config import TEST_GENE_LIMIT, filter_columns


def clean_dataframe(df, index_col):
    df.sort_index(inplace=True, axis=1)

    if index_col is None:
        df.sort_values(df.columns.tolist(), inplace=True)
    else:
        df.sort_index(inplace=True)

    return df


def process_column_name(col, feature_dataset_name):
    match = re.match(r"(.+?) \((\d+)\)", col)
    if match:
        feature_label, given_id = match.groups()
        feature_name = f"{feature_label.replace('-', '_')}_({given_id})_{feature_dataset_name}"
        # feature_name = f"{feature_label}_({given_id})_{feature_dataset_name}"
        # print(f"Gene name match: {feature_name}")
    # elif feature_dataset_name == "Lineage":
    #     feature_label = col
    #     given_id = col
    #     feature_name = re.sub(r'[^\w/,]+', '_', col) + f"_{feature_dataset_name}"
    #     # print(f"Gene name no match feature name: {feature_name}")
    # elif feature_dataset_name == "CytobandCN":
    #     feature_label = col
    #     given_id = col
    #     feature_name = re.sub(r'[^\w.]+', '_', col) + f"_{feature_dataset_name}"
    #     # print(f"Gene name no match: {feature_name}")
    else:
        feature_label = col
        given_id = col
        feature_name = re.sub(r'[\s-]+', '_', col) + f"_{feature_dataset_name}"
        # feature_name = re.sub(r'[^\w]+', '_', col) + f"_{feature_dataset_name}"
        # print(f"Gene name no match: {feature_name}")
    return feature_name, feature_label, given_id


def process_biomarker_matrix(df, index_col=0, test=False):
    # df = df.T
    df = clean_dataframe(df, index_col)
    print("Start Processing Biomarker Matrix")
    if test:
        df = df.iloc[:, :]
    print(df.head())
    print("End Processing Biomarker Matrix")
    # df.to_csv("features.csv", index=False)
    return df


def process_dep_matrix(df, test=False, restrict_targets=False, restrict_to=None):
    # drops rows and columns with all nulls, creates y matrix with cds-ensemble
    # df = df.T
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
            # df = df.iloc[:, :TEST_GENE_LIMIT+1]
            # Create a regex pattern with word boundaries
            pattern = r'\b(' + '|'.join(re.escape(col) for col in filter_columns) + r')\b'

            # Create a boolean mask for columns that contain any of the filter_columns as whole words
            mask = df.columns.str.contains(pattern, regex=True)

            # Use the mask to select the desired columns
            df = df.loc[:, mask]
            
    elif restrict_targets:
        restrict_deps = restrict_to.split(";")
        df = df[["Row.name"] + restrict_deps]

    print("Start Processing Dependency Matrix")
    print(df)
    print("End Processing Dependency Matrix")
    # df.to_csv("deps.csv", index=False)
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


def generate_config(input_dict, relation="All"):
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


def generate_feature_info(ipt_dicts, save_pref):
    dsets = []
    for dset_name, dset_value in ipt_dicts["data"].items():
        if dset_value["table_type"] == "feature":
            dsets.append(dset_name)
    fnames = [str(save_pref / (dset + ".csv")) for dset in dsets]

    df = pd.DataFrame({"dataset": dsets, "filename": fnames,})

    return df
