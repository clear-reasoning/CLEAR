import numpy as np
from sklearn.metrics import mean_squared_error

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        print(f"[cosine_similarity] One of the vectors is zero: vec1={vec1}, vec2={vec2}.")
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)

def l2_loss(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return -np.linalg.norm(vec1 - vec2)

def calculate_l2_norm(predictions, ground_truth):
    """Euclidean distance (L2 norm) between two vectors"""
    if len(predictions) != len(ground_truth):
        return float('nan')
    return np.sqrt(mean_squared_error(ground_truth, predictions))

def compare_dataframes(df1, df2, columns: list =None):
    """
    Compare two DataFrames on specified columns using L2 norm
    Args:
    - the two dataframes
    - the list of columns you want to compare. NOTE: if no list provided, it will compare on all the common columns.
    """
    if columns is None:
        # Use columns common to both DataFrames
        columns = list(set(df1.columns) & set(df2.columns))
        if columns == []:
            raise ValueError(f"No columns in common between the two dataframe {df1} and {df2}")
    
    else:
        for key in columns:
            if key not in df1.columns:
                raise ValueError(f"{key} not a column in {df1}")
            if key not in df2.columns:
                raise ValueError(f"{key} not a column in {df2}")

    # Ensure DataFrames have the same length for the selected columns
    len1, len2 = len(df1), len(df2)
    
    # If lengths differ, trim the longer dataframe to match the shorter one
    if len1 > len2:
        # Keep only the last len2 rows of df1
        df1_trimmed = df1.iloc[-len2:].reset_index(drop=True)
        df2_trimmed = df2.reset_index(drop=True)
    elif len2 > len1:
        # Keep only the last len1 rows of df2
        df2_trimmed = df2.iloc[-len1:].reset_index(drop=True)
        df1_trimmed = df1.reset_index(drop=True)
    else:
        # Same length, just reset indices for consistency
        df1_trimmed = df1.reset_index(drop=True)
        df2_trimmed = df2.reset_index(drop=True)
    
    # Calculate L2 norm for each column and return average
    norms = []
    for col in columns:
        if col in df1_trimmed.columns and col in df2_trimmed.columns:
            if np.issubdtype(df1_trimmed[col].dtype, np.number) and np.issubdtype(df2_trimmed[col].dtype, np.number):
                norm = calculate_l2_norm(df1_trimmed[col].values, df2_trimmed[col].values)
                norms.append(norm)
    
    if not norms:
        return float('nan')
    
    return np.mean(norms)
