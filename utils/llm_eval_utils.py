import re
import numpy as np
from sklearn.metrics import mean_squared_error

def extract_tag_content(response: str, tag_name: str):
    """Extract the content within a specific XML-style tag from a response."""
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, response, re.DOTALL)
    return (match.group(1).strip(), True) if match else ([], False)

def extract_values(response: str, tag_name: str):
    """Extract a list of float values from a specific tag."""
    raw, found = extract_tag_content(response, tag_name)
    if not found:
        return [], False
    try:
        values = [float(v.strip()) for v in raw.split(',') if v.strip()]
        return values, True
    except ValueError:
        return [], False

def calculate_l2_norm(predictions, ground_truth):
    """Compute the L2 norm (RMSE) between predictions and ground truth."""
    if len(predictions) != len(ground_truth):
        return float('nan')
    return np.sqrt(mean_squared_error(ground_truth, predictions))
