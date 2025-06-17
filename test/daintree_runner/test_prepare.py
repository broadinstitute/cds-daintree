from daintree_runner.prepare import prepare
from unittest.mock import MagicMock
from pathlib import Path
import pandas as pd

def test_partition_sizes(tmpdir):
    out_dir = tmpdir.join("out")
    out_dir.mkdir()
    save_pref = Path(str(out_dir))
    input_config = tmpdir.join("input.json")

    input_config.write("""
{
  "model_name": "CellContext",
  "screen_name": "CRISPR",
  "data": {
      "CRISPR": {
          "taiga_id":"internal-24q2-3719.82/CRISPRGeneEffect",
          "table_type":"target_matrix",
          "relation":"All"
      },
      "Lineage": {
          "taiga_id": "predictability-76d5.94/PredictabilityLineageTransformed",
          "table_type": "feature",
          "dim_type": "lineage",
          "required": true,
          "exempt": false
      }
    }
  }
""")

    n_samples = 50
    n_targets = 10
    n_features = 20
    samples = ["ACH-{i}" for i in range(n_samples)]

    def mock_tc_get(taiga_id):
        if taiga_id == "internal-24q2-3719.82/CRISPRGeneEffect":
            targets = {f"T{i}" : list(range(n_samples)) for i in range(n_targets) }
            return pd.DataFrame(targets, index=samples)
        else:
            assert taiga_id == "predictability-76d5.94/PredictabilityLineageTransformed"
            features = {f"F{i}" : list(range(n_samples)) for i in range(n_features) }
            return pd.DataFrame(features, index=samples)

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
            test_first_n_tasks=None
        )
    
    partitions = pd.read_csv(str(out_dir.join("partitions.csv")))
    assert len(partitions) == n_targets
