from taigapy.client_v3 import UploadedFile, LocalFormat
from taigapy import create_taiga_client_v3
from pathlib import Path


def update_taiga(
    dataset_id: str,
    description_of_changes: str,
    matrix_name_in_taiga: str,
    file_local_path: Path,
    file_format: str,
) -> None:
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
    try:
        tc = create_taiga_client_v3()
        # Update the dataset with the transformed data
        version = tc.update_dataset(
            dataset_id,
            description_of_changes,
            additions=[
                UploadedFile(
                    matrix_name_in_taiga,
                    local_path=file_local_path,
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
    