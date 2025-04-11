import re 
from typing import List, Tuple, Optional
import numpy as np
from sklearn.metrics import mean_squared_error

def extract_tag_content(response, tag_name):
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip(), True
    else:
        return [], False

def extract_values(response: str, tag_name: str) -> Tuple[List[float], bool]:
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        values_str = match.group(1).strip()
        try:
            values = [float(v.strip()) for v in values_str.split(',') if v.strip()]
            return values, True
        except ValueError:
            return [], False
    return [], False

def calculate_l2_norm(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    # Ensuring that the type of the arguments is correct
    if not isinstance(predictions, np.ndarray):
        assert False, "Predictions should be a numpy array."
    if not isinstance(ground_truth, np.ndarray):
        assert False, "Ground truth should be a numpy array."
        
    if len(predictions) != len(ground_truth):
        return float('nan')
    return np.sqrt(mean_squared_error(ground_truth, predictions))