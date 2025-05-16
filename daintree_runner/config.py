# Paths to the daintree commands
DAINTREE_RUNNER_BIN_PATH = "daintree-runner"
DAINTREE_CORE_BIN_PATH = "daintree-core"

# Container image to use for sparkles.
DAINTREE_CONTAINER = "us.gcr.io/broad-achilles/daintree-sparkles:v1"

# Files to save locally.
FILES = {
    "target_matrix": "target_matrix.ftr",
    "target_matrix_filtered": "target_matrix_filtered.ftr",
    "feature_matrix": "X.ftr",
    "feature_metadata": "X_feature_metadata.ftr",
    "valid_samples": "X_valid_samples.ftr",
    "partitions": "partitions.csv",
    "feature_path_info": "feature_path_info.csv",
    "features": "features.csv",
    "predictions": "predictions.csv",
}

DEFAULT_JOB_COUNT = 10
 
TEST_LIMIT = 5

filter_columns_gene = [
    "Row.name",
    "SOX10",
    "PAX8",
    "EBF1",
    "MYB",
    "KRAS",
    "MDM2",
    "NRAS",
]
filter_columns_oncref = [
    "Row.name",
    "PRC-005868644-712-80",
    "PRC-005868644-712-80",
    "PRC-005868644-712-80",
    "PRC-005868644-712-80",
    "PRC-005868644-712-80",
]
