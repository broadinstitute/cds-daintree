import pandas as pd
from scipy.stats import pearsonr

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
