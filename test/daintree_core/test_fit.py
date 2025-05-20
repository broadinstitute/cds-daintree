import pandas as pd
import numpy as np

from daintree_core.main import prepare_x_command, prepare_y_command, fit_model_command

def test_prepare_and_fit(tmp_path):
    n_samples = 1000
    n_features = 50

    feature_names = [f"F{i}" for i in range(n_features)]
    sample_names = [f"S{i}" for i in range(n_samples)]

    # create input variables which are all random
    features = pd.DataFrame({f: np.random.uniform(size=n_samples) for f in feature_names}, index=sample_names)

    # create three target variables with relations to a few of the input features
    # the first target is positively correlated with F10
    target_1 = features["F10"] * 3 - 1
    # next, negatively correlated with F10
    target_2 = features["F10"] * -2 
    # the third has two variables. One determines two groups: a postive correlated group and a negative correlated group. Overall target will likely show negative correlation with F10
    target_3 = np.where( features["F11"] < 0.5, features["F10"] , -features["F10"] - 100 )

    targets = pd.DataFrame({"T1": target_1, "T2": target_2, "T3": target_3}, index=sample_names)

    features_path = tmp_path / "features.csv"
    targets_path = tmp_path / "targets.csv"
    features_info_path = tmp_path / "features_info.csv"
    model_config_path = tmp_path / "model_config.yaml"
    y_path = tmp_path / "y"
    x_path = tmp_path / "x"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    targets.to_csv(targets_path)
    features.to_csv(features_path)

    pd.DataFrame({"dataset": ["F", "T"], "filename": [str(features_path), str(targets_path)], }).to_csv(features_info_path)

    model_config_path.write_text("""
model_a:
  Features:
    - F
  Required:
    - F
  Relation: All
  Jobs: 10
""")

    prepare_x_command(str(model_config_path), str(targets_path), str(features_info_path), str(x_path), confounders=None, output_format=".csv", output_related=None)

    prepare_y_command(
        str(targets_path),
        str(y_path),
        top_variance_filter=None,
        gene_filter=None)

    def fit_target(start_index, end_index):
        fit_model_command(str(x_path)+".csv",
            str(y_path),
            str(model_config_path),
            "model_a",
            "regress",
            n_folds=3,
            related_table=None,
            feature_metadata=None,
            model_valid_samples=None,
            valid_samples_file=None,
            feature_subset_file=None,
            target_range=(start_index, end_index),
            targets=None,
            output_dir=str(output_dir),
            top_n=3,
            seed=0
        )

    # process these in one batch
    fit_target(start_index=0, end_index=3)

    # read the results
    results_df = pd.read_csv(str(output_dir / "model_a_0_3_features.csv"))
    results = {rec['target_variable']: rec for rec in results_df.to_records()}
    assert len(results) == 3
    out_predictions_1 = pd.read_csv(str(output_dir / "model_a_0_3_predictions.csv"), index_col=0)

    # target_1 and 2 should be easy for us to predict, so make sure the RMSE is lowish
    assert rmse(out_predictions_1["T1"], targets["T1"]) < 0.2
    assert results["T1"]["pearson"] > 0.9

    assert rmse(out_predictions_1["T2"], targets["T2"]) < 0.2
    assert results["T2"]["pearson"] > 0.9
    
    # T3 has a larger stddev, so I'm not sure what RMSE to expect offhand, so only check pearson
    assert results["T3"]["pearson"] > 0.9

    # make sure we've identified F10 as the most important feature for T1, and postively correlated with the output
    t1_results = results["T1"]
    assert t1_results["feature0"] == "F10_F"
    assert 0.90 < t1_results["feature0_importance"] <= 1.0 
    assert 0.90 < t1_results["feature0_correlation"] <= 1.0

    # make sure we've identified F10 as the most important feature for T2, and postively correlated with the output
    t2_results = results["T2"]
    assert t2_results["feature0"] == "F10_F"
    assert 0.90 < t2_results["feature0_importance"] <= 1.0 
    assert -0.90 > t2_results["feature0_correlation"] <= -1.0

    # make sure we've identified F10 and F11 as the most important features for T3
    t3_results = results["T3"]
    assert {t3_results["feature0"], t3_results["feature1"] }  == {"F10_F", "F11_F"}

def rmse(x, y):
    return np.mean((x-y)**2)**(0.5)