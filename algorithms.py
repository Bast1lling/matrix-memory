from typing import List

import numpy as np


def generate_random_key(dim: int) -> np.ndarray:
    """Generate a random vector for key initialization."""
    return np.random.randn(dim)


def generate_orthogonal_key_rejection_sampling(keys: List[np.ndarray], dim: int, max_similarity: float) -> np.ndarray:
    """
    Generate a nearly orthogonal key by randomly sampling from the (gaussian) vector space.

    Args:
        keys: All already existing keys
        dim: Dimensionality of key vector
        max_similarity: Maximum allowed dot product with existing keys
    Returns:
        A unit vector that is approximately orthogonal to existing keys
    """
    # max_attempts = int(100 * (0.1 / max_similarity)**2 * min(512 / dim, 1))
    max_attempts = max(len(keys), dim) // 8
    lowest_similarity_yet = float("inf")
    best_candidate = None
    # If we have no keys yet, just return a random unit vector
    if not keys:
        key = generate_random_key(dim)
        return key / np.linalg.norm(key)

    # Try random vectors until we find one that's nearly orthogonal to all existing keys
    for _ in range(max_attempts):
        # Generate random vector from Gaussian distribution
        candidate = generate_random_key(dim)
        candidate = candidate / np.linalg.norm(candidate)

        # Check orthogonality against all existing keys
        max_dot_product = max(abs(np.dot(candidate, key)) for key in keys)

        # If it's sufficiently orthogonal, accept it
        if max_dot_product < max_similarity:
            return candidate
        if max_dot_product < lowest_similarity_yet:
            best_candidate = candidate

    return best_candidate


def generate_orthogonal_key_gram_schmidt(keys: List[np.ndarray], dim: int) -> np.ndarray:
    """
    Generate an orthogonal key using the Gram-Schmidt process.
    Only works when len(keys) < dim as it relies on finding a vector in the orthogonal complement.

    Args:
        keys: All already existing keys
        dim: Dimensionality of key vector

    Returns:
        A unit vector that is orthogonal to existing keys, or None if len(keys) >= dim
    """
    # Check if we have room for another orthogonal vector
    if len(keys) >= dim:
        return None

    # If we have no keys yet, just return a random unit vector
    if not keys:
        key = generate_random_key(dim)
        return key / np.linalg.norm(key)

    # Start with a random vector
    candidate = generate_random_key(dim)

    # Project the candidate onto each of the existing keys
    for key in keys:
        # Calculate the projection of candidate onto key
        projection = np.dot(candidate, key) * key
        # Subtract the projection to make candidate orthogonal to this key
        candidate = candidate - projection

    # Check if candidate has been reduced to near-zero (unlikely if dim >> len(keys))
    norm = np.linalg.norm(candidate)
    if norm < 1e-10:
        # This is very unlikely unless keys are nearly linearly dependent
        # Try again with a different random vector
        return generate_orthogonal_key_gram_schmidt(keys, dim)

    # Normalize to unit length
    return candidate / norm
