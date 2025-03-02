from typing import List, Optional

import numpy as np


def generate_random_key(dim: int) -> np.ndarray:
    """Generate a random vector for key initialization."""
    return np.random.randn(dim)


def generate_orthogonal_key_rejection_sampling(keys: List[np.ndarray], dim: int, max_similarity: float = 0.1,
                                               max_attempts: int = 100) -> Optional[np.ndarray]:
    """
    Generate a nearly orthogonal key by randomly sampling from the (gaussian) vector space.

    Args:
        keys: All already existing keys
        dim: Dimensionality of key vector
        max_similarity: Maximum allowed dot product with existing keys
        max_attempts: Maximum allowed retries
    Returns:
        A unit vector that is approximately orthogonal to existing keys
    """
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

    return None


def generate_orthogonal_key_lsh(keys: List[np.ndarray], dim: int, max_similarity: float = 0.1,
                                num_projections: int = 20) -> Optional[np.ndarray]:
    """
    Generate a nearly orthogonal key using an approach inspired by Locality-Sensitive Hashing.

    Args:
        keys: All already existing keys
        dim: Dimensionality of key vector
        num_projections: Number of random projection planes to consider
        max_similarity: Maximum allowed dot product with existing keys

    Returns:
        A unit vector that is approximately orthogonal to existing keys
    """
    if not keys:
        key = generate_random_key(dim)
        return key / np.linalg.norm(key)

    # Generate multiple random projection vectors
    projection_vecs = [generate_random_key(dim) for _ in range(num_projections)]
    projection_vecs = [v / np.linalg.norm(v) for v in projection_vecs]

    # Compute which side of each projection plane each existing key lies on
    key_signatures = []
    for key in keys:
        signature = [1 if np.dot(key, proj) > 0 else -1 for proj in projection_vecs]
        key_signatures.append(signature)

    # Generate candidate keys by flipping bits in signatures
    best_candidate = None
    min_max_dot = float('inf')

    # Try different signature patterns to find one far from existing keys
    for i in range(2 ** num_projections):
        # Convert integer to binary signature with 1/-1 values
        candidate_sig = [(2 * ((i >> j) & 1) - 1) for j in range(num_projections)]

        # Compute Hamming distances to existing signatures (approximating angular distance)
        hamming_distances = [sum(a != b for a, b in zip(candidate_sig, sig)) for sig in key_signatures]

        # If this signature is far from all existing ones, construct a vector
        if min(hamming_distances) > num_projections // 3:  # Require distance of at least 1/3 the bits
            # Construct a vector with this signature
            candidate = sum(s * v for s, v in zip(candidate_sig, projection_vecs))
            candidate = candidate / np.linalg.norm(candidate)

            # Check actual dot products
            max_dot_product = max(abs(np.dot(candidate, key)) for key in keys)

            if max_dot_product < max_similarity:
                return candidate

            if max_dot_product < min_max_dot:
                min_max_dot = max_dot_product
                best_candidate = candidate

    # If we found a reasonable candidate, return it
    if best_candidate is not None:
        return best_candidate
    return None


def generate_orthogonal_key_spherical(keys: List[np.ndarray], dim: int, max_similarity: float = 0.1, iterations: int = 50,
                                      repulsion_strength: float = 0.5) -> Optional[np.ndarray]:
    """
    Generate a nearly orthogonal key using a spherical coding inspired approach.
    This simulates repulsion forces between points on a sphere to maximize separation.

    Args:
        keys: All already existing keys
        dim: Dimensionality of key vector
        max_similarity: Target maximum dot product with existing keys
        iterations: Number of optimization iterations
        repulsion_strength: Strength of the repulsion force between vectors

    Returns:
        A unit vector that is approximately orthogonal to existing keys
    """
    if not keys:
        key = generate_random_key(dim)
        return key / np.linalg.norm(key)

    # Start with a random unit vector
    candidate = generate_random_key(dim)
    candidate = candidate / np.linalg.norm(candidate)

    # Iteratively update the vector to maximize orthogonality
    for _ in range(iterations):
        # Calculate gradient (force) from existing keys
        gradient = np.zeros(dim)

        for key in keys:
            # Calculate dot product
            dot_prod = np.dot(candidate, key)

            # If dot product is too large, apply a repulsive force
            if abs(dot_prod) > 0.01:  # Small threshold to avoid numerical issues
                # Force proportional to dot product and in opposite direction
                repulsion = -repulsion_strength * dot_prod * key
                gradient += repulsion

        # Update candidate vector
        if np.linalg.norm(gradient) > 1e-10:
            candidate = candidate - gradient
            # Project back to unit sphere
            candidate = candidate / np.linalg.norm(candidate)

        # Check if we've achieved sufficient orthogonality
        max_dot_product = max(abs(np.dot(candidate, key)) for key in keys)
        if max_dot_product < max_similarity:
            return candidate
    return None


def generate_orthogonal_key_gram_schmidt(keys: List[np.ndarray], dim: int) -> Optional[np.ndarray]:
    """
    Generate an orthogonal key using a simple algorithm that finds a base-vector for len(keys) < dim.

    Args:
        keys: All already existing keys
        dim: Dimensionality of key vector

    Returns:
        A unit vector that is approximately orthogonal to existing keys
    """
    pass