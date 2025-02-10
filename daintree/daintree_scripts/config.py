# Paths to the daintree and sparkles binaries. Also to the service account key for sparkles.
PATHS = {
    'daintree_bin': '/install/depmap-py/bin/daintree',
    'sparkles_bin': '/install/sparkles/bin/sparkles',
    'service_account': '/root/.sparkles-cache/service-keys/broad-achilles.json'
}

# Container image to use for sparkles.
CONTAINER = {
    'image': 'us.gcr.io/broad-achilles/daintree-sparkles:v1',
}

# Files to save locally.
FILES = {
    'target_matrix': 'target_matrix.ftr',
    'target_matrix_filtered': 'target_matrix_filtered.ftr',
    'feature_matrix': 'X.ftr',
    'feature_metadata': 'X_feature_metadata.ftr',
    'valid_samples': 'X_valid_samples.ftr',
    'partitions': 'partitions.csv',
    'feature_info': 'feature_info.csv',
    'features': 'features.csv',
    'predictions': 'predictions.csv'
}

# Model parameters.
MODEL = {
    'n_folds': 5,
    'top_n_features': 50,
    'default_jobs': 10
}

TEST_LIMIT = 5
filter_columns_gene = ["Row.name", "SOX10", "PAX8", "EBF1", "MYB", "KRAS", "MDM2", "NRAS"]
filter_columns_oncref = ["Row.name", "PRC-005868644-712-80", "PRC-005868644-712-80", "PRC-005868644-712-80", "PRC-005868644-712-80", "PRC-005868644-712-80"]
