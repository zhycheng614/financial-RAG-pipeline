from collections import defaultdict
from typing import List, Any, Tuple


def rrf(ranked_lists: List[List[Any]], k: int = 60) -> List[Any]:
    """
    Reciprocal Rank Fusion (RRF) algorithm to merge multiple ranked lists.
    
    Args:
        ranked_lists: List of lists, where each inner list contains elements in ranked order
        k: RRF constant parameter (default: 60, commonly used value)
    
    Returns:
        List of elements sorted by RRF score in descending order
    """
    # Use rrf_with_scores and extract only the elements
    return [element for element, _ in rrf_with_scores(ranked_lists, k)]


def rrf_with_scores(ranked_lists: List[List[Any]], k: int = 60) -> List[Tuple[Any, float]]:
    """
    Reciprocal Rank Fusion (RRF) algorithm that returns elements with their RRF scores.
    
    Args:
        ranked_lists: List of lists, where each inner list contains elements in ranked order
        k: RRF constant parameter (default: 60, commonly used value)
    
    Returns:
        List of tuples (element, rrf_score) sorted by RRF score in descending order
    """
    if not ranked_lists:
        return []
    
    # Dictionary to store RRF scores for each element
    rrf_scores = defaultdict(float)
    
    # Calculate RRF score for each element
    for ranked_list in ranked_lists:
        for rank, element in enumerate(ranked_list, start=1):
            # RRF formula: 1 / (k + rank)
            rrf_scores[element] += 1.0 / (k + rank)
    
    # Sort elements by RRF score in descending order
    sorted_elements = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_elements