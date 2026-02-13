import pandas as pd
import sys
import os
from daintree_runner.data_processor import apply_preprocess


def test_apply_preprocess_with_none():
    """Test that apply_preprocess returns dataframe unchanged when preprocess_spec is None."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = apply_preprocess(df, None)
    pd.testing.assert_frame_equal(result, df)


def test_apply_preprocess_with_function(tmpdir):
    """Test that apply_preprocess calls the specified function."""
    # Create a module with a preprocessing function
    module_dir = tmpdir.mkdir("my_module")
    module_file = module_dir.join("transforms.py")
    module_file.write(
        """
import pandas as pd

def double_values(df):
    return df * 2

def add_column(df):
    df = df.copy()
    df['new_col'] = 100
    return df
"""
    )

    # Add the module directory to sys.path
    sys.path.insert(0, str(module_dir))

    try:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        # Test double_values function
        result = apply_preprocess(df, "transforms:double_values")
        expected = pd.DataFrame({"a": [2, 4, 6], "b": [8, 10, 12]})
        pd.testing.assert_frame_equal(result, expected)

        # Test add_column function
        result = apply_preprocess(df, "transforms:add_column")
        assert "new_col" in result.columns
        assert list(result["new_col"]) == [100, 100, 100]

    finally:
        # Clean up sys.path
        sys.path.remove(str(module_dir))


def test_apply_preprocess_in_prepare_flow(tmpdir):
    """Test that preprocess is applied during the prepare flow."""
    from daintree_runner.prepare import prepare
    from unittest.mock import MagicMock
    from pathlib import Path

    # Create a module with a preprocessing function
    module_dir = tmpdir.mkdir("preprocess_module")
    module_file = module_dir.join("my_preprocess.py")
    module_file.write(
        """
import pandas as pd

def filter_targets(df):
    # Only keep columns starting with 'KEEP'
    keep_cols = [c for c in df.columns if c.startswith('KEEP')]
    return df[keep_cols]

def scale_features(df):
    return df * 10
"""
    )

    # Add the module directory to sys.path
    sys.path.insert(0, str(module_dir))

    try:
        out_dir = tmpdir.mkdir("out")
        save_pref = Path(str(out_dir))
        input_config = tmpdir.join("input.json")

        # Config with preprocess specified for target matrix
        input_config.write(
            """
{
  "model_name": "TestModel",
  "screen_name": "Test",
  "data": {
      "Targets": {
          "taiga_id": "test-targets",
          "table_type": "target_matrix",
          "relation": "All",
          "preprocess": "my_preprocess:filter_targets"
      },
      "Features": {
          "taiga_id": "test-features",
          "table_type": "feature",
          "dim_type": "gene",
          "required": false,
          "exempt": false,
          "preprocess": "my_preprocess:scale_features"
      }
    }
  }
"""
        )

        n_samples = 10
        samples = [f"ACH-{i}" for i in range(n_samples)]

        def mock_tc_get(taiga_id):
            if taiga_id == "test-targets":
                # Return targets with some KEEP and some DROP columns
                return pd.DataFrame(
                    {
                        "KEEP_T1": list(range(n_samples)),
                        "KEEP_T2": list(range(n_samples)),
                        "DROP_T3": list(range(n_samples)),
                    },
                    index=samples,
                )
            else:
                assert taiga_id == "test-features"
                return pd.DataFrame(
                    {
                        "F1": [1.0] * n_samples,
                        "F2": [2.0] * n_samples,
                    },
                    index=samples,
                )

        tc = MagicMock()
        tc.get = mock_tc_get

        prepare(
            tc,
            test_first_n_models=None,
            restrict_targets_to=None,
            runner_config_path=str(input_config),
            save_pref=save_pref,
            nfolds=5,
            models_per_task=1,
            test_first_n_tasks=None,
        )

        # Check that only KEEP columns are in the partitions
        partitions = pd.read_csv(str(out_dir.join("partitions.csv")))
        # Should only have 2 targets (KEEP_T1, KEEP_T2), not 3
        assert len(partitions) == 2

        # Check that features were scaled (values should be 10x)
        features_df = pd.read_csv(str(out_dir.join("Features.csv")), index_col=0)
        assert features_df["F1"].iloc[0] == 10.0
        assert features_df["F2"].iloc[0] == 20.0

    finally:
        sys.path.remove(str(module_dir))
