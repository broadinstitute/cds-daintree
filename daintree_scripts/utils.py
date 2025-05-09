import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

from taigapy.client_v3 import UploadedFile, LocalFormat
from taigapy import create_taiga_client_v3
from typing import Union


def calculate_feature_correlations(x, y):
    """Calculate correlation between feature and target."""
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    mask = ~pd.isna(x) & ~pd.isna(y)
    
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    if len(x_filtered) > 1 and len(y_filtered) > 1:
        corr, _ = pearsonr(x_filtered, y_filtered)
        return corr
    return None


def update_taiga(
    dataset_id: str,
    description_of_changes: str,
    matrix_name_in_taiga: str,
    file_local_path: Path,
    file_format: Union[str, LocalFormat],
) -> str:
    """Update a dataset in Taiga with transformed data."""
    assert dataset_id, "Dataset ID cannot be empty"
    assert description_of_changes, "Description of changes cannot be empty"
    assert matrix_name_in_taiga, "Matrix name in Taiga cannot be empty"
    assert file_local_path, "File path cannot be empty"
    assert file_format, "File format cannot be empty"

    if file_format == "csv_table":
        file_format = LocalFormat.CSV_TABLE
    elif file_format == "csv_matrix":
        file_format = LocalFormat.CSV_MATRIX
    else:
        assert isinstance(file_format, LocalFormat)
        
    try:
        tc = create_taiga_client_v3()
        # Update the dataset with the transformed data
        version = tc.update_dataset(
            dataset_id,
            description_of_changes,
            additions=[
                UploadedFile(
                    matrix_name_in_taiga,
                    local_path=str(file_local_path),
                    format=file_format,
                )
            ],
        )
        print(
            f"Updated dataset: {version.permaname} to version number: {version.version_number}"
        )
        return f"{version.permaname}.{version.version_number}/{matrix_name_in_taiga}"
    except Exception as e:
        print(f"Error updating Taiga: {e}")
        raise
    