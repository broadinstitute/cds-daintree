import json
from pathlib import Path
import yaml
import datetime as dt
from .config import DEFAULT_JOB_COUNT

def check_file_locs(ipt, config):
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
            assert (
                f in ipt_features
            ), f"Feature {f} in model config file does not have corresponding input in {model_name}"


def load_input_config(input_config):
    """Load and validate input configuration.
    Args:
        input_config: Path to input configuration file

    Returns:
        dict: Validated configuration dictionary
    """
    print("Loading input json file...")
    with open(input_config, "r") as f:
        config = json.load(f)

    assert "model_name" in config, "Config missing required field 'model_name'"
    assert "screen_name" in config, "Config missing required field 'screen_name'"
    assert "data" in config, "Config missing required field 'data'"

    for dataset_name, dataset_config in config["data"].items():
        assert (
            "taiga_id" in dataset_config
        ), f"Dataset {dataset_name} missing required field 'taiga_id'"
        assert (
            "table_type" in dataset_config
        ), f"Dataset {dataset_name} missing required field 'table_type'"

        assert dataset_config["table_type"] in [
            "target_matrix",
            "feature",
            "relation",
        ], f"Dataset {dataset_name} has invalid table_type: {dataset_config['table_type']}"

    # Ensure at least one target_matrix exists
    target_matrices = [
        name
        for name, config in config["data"].items()
        if config["table_type"] == "target_matrix"
    ]
    assert len(target_matrices) > 0, "Config must include at least one target_matrix"

    return config


def setup_ensemble_config(save_pref: Path, ensemble_config, ipt_dict):
    """Setup and validate ensemble configuration.
    Args:
        ensemble_config: Path to existing ensemble config or None
        ipt_dict: Input configuration dictionary

    Returns:
        tuple: (config_path, config_dict)
    """
    print("Setting up ensemble config...")
    if not ensemble_config:
        config = generate_model_config(ipt_dict, relation="All")
        model_config_name = f"model-config_temp_{dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}.yaml"
        config_path = save_config(save_pref, config, model_config_name)
    else:
        config_path = ensemble_config

    config_dict = load_config(config_path)
    check_file_locs(ipt_dict, config_dict)

    return config_path, config_dict


def save_config(save_pref: Path, config, filename):
    """Save model configuration to a file.
    Args:
        config: Configuration dictionary
        filename: Name of the file to save to
    Returns:
        str: Path to saved configuration file
    """
    config_path = save_pref / filename
    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=True)

    return str(config_path)


def load_config(config_path):
    """Load model configuration from a file.
    Args:
        config_path: Path to the configuration file
    Returns:
        dict: Loaded configuration
    """
    with open(config_path, "r") as f:
        return yaml.load(f, yaml.SafeLoader)


def generate_model_config(input_dict, relation="All"):
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
        key for key, value in data.items() if value.get("table_type") == "feature"
    ]
    required = [
        key
        for key, value in data.items()
        if value.get("table_type") == "feature" and value.get("required", False)
    ]
    exempt = [
        key
        for key, value in data.items()
        if value.get("table_type") == "feature" and value.get("exempt", False)
    ]
    relation = next(
        (
            value["relation"]
            for value in data.values()
            if value.get("table_type") == "target_matrix"
        ),
        "All",
    )
    model_config = {
        "Features": features,
        "Required": required,
        "Relation": relation,
        "Jobs": DEFAULT_JOB_COUNT,  # Use config value
    }

    if relation == "MatchRelated":
        related = next(
            (
                key
                for key, value in data.items()
                if value.get("table_type") == "relation"
            ),
            None,
        )
        if related:
            model_config["Related"] = related

    if exempt:
        model_config["Exempt"] = exempt

    return {model_name: model_config}


# TODO: This is kind of a useless function at this moment that was copied over from
# the old code since among the input configs there has not been any MatchTarget
# and such. So out_rel is always False and related_dset is None. Maybe in future
# we will have a use for it.
def determine_relations(config_dict):
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


def create_output_config(
    input_config, feature_metadata_id, ensemble_id, prediction_matrix_id
):
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
                "data": data,
            },
            "output": {
                "ensemble_taiga_id": ensemble_id,
                "feature_metadata_taiga_id": feature_metadata_id,
                "prediction_matrix_taiga_id": prediction_matrix_id,
            },
        }
    }
    return config
