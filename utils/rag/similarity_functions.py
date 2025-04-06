import numpy as np

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