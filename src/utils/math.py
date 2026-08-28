from typing import List
import numpy as np

def l2_normalize(vec: List[float]) -> List[float]:
    # Apply L2 normalization to each embedding vector
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms = np.where(norms > 0, norms, 1.0)
    return vec / norms

def softmax(vec: List[float]) -> List[float]:
    # if vec is not a numpy array, convert it to one
    if not isinstance(vec, np.ndarray):
        vec = np.array(vec)

    max_value = np.max(vec)
    vec = vec - max_value
    return np.exp(vec) / np.sum(np.exp(vec))