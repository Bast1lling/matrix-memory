import time
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from algorithms import (generate_orthogonal_key_rejection_sampling,
                        generate_orthogonal_key_gram_schmidt)


class MatrixMemory:
    """
    A high-dimensional associative memory that stores values using orthogonal keys.

    This implementation uses the Gram-Schmidt process to ensure all keys are
    orthogonal to each other, enabling efficient retrieval with minimal interference.
    """

    def __init__(self, dim: int = 100, max_key_similarity: float = 0.01):
        """
        Initialize a MatrixMemory with a given dimensionality.

        Args:
            dim: Dimension of the memory space
        """
        self.max_key_similarity = max_key_similarity
        self.dim = dim
        self.memory_matrix = np.zeros((dim, dim))  # The memory matrix
        self.keys = []  # Store all keys for orthogonalization

    def _generate_random_key(self) -> np.ndarray:
        if len(self.keys) < self.dim:
            return generate_orthogonal_key_gram_schmidt(self.keys, self.dim)
        else:
            return generate_orthogonal_key_rejection_sampling(self.keys, self.dim, self.max_key_similarity)

    def _dimensionalize(self, vector: np.ndarray) -> np.ndarray:
        if len(vector) != self.dim:
            # Resize or pad the value to match memory dimensions
            if len(vector) > self.dim:
                vector = vector[:self.dim]
            else:
                pad_width = self.dim - len(vector)
                vector = np.pad(vector, (0, pad_width), 'constant')
        return vector

    def insert(self, value: np.ndarray) -> np.ndarray:
        """
        Add a value to memory with a newly generated orthogonal key.

        Args:
            value: The value to store

        Returns:
            The ID assigned to this value
        """
        value_array = value.copy()

        # Ensure the value has the right dimensionality
        value_array = self._dimensionalize(value_array)

        # Generate new random key and orthogonalize it
        key = self._generate_random_key()

        # Add the key to our list
        self.keys.append(key)

        # Update memory matrix with outer product
        self.memory_matrix += np.outer(value_array, key)

        return key

    def insert_at(self, key: np.ndarray, value: np.ndarray):
        value_array = value.copy()
        value_array = self._dimensionalize(value_array)
        # Normalize the key
        key_norm = np.linalg.norm(key)
        if key_norm < 1e-10:
            raise ValueError("Key vector is too close to zero")

        normalized_key = key / key_norm
        # Check if this key (or a very similar one) already exists
        existing_key_idx = None
        for i, existing_key in enumerate(self.keys):
            if np.abs(np.dot(normalized_key, existing_key)) > 0.99:  # High similarity threshold
                existing_key_idx = i
                break

        if existing_key_idx is not None:
            # Key exists - remove old contribution from memory matrix
            old_key = self.keys[existing_key_idx]
            old_value = self.memory_matrix @ old_key
            self.memory_matrix -= np.outer(old_value, old_key)

            # Replace the key with the possibly new one
            self.keys[existing_key_idx] = normalized_key
        else:
            # New key - orthogonalize and add to key list
            orthogonal_key = self._generate_random_key()
            self.keys.append(orthogonal_key)
            normalized_key = orthogonal_key

        # Add new contribution to memory matrix
        self.memory_matrix += np.outer(value_array, normalized_key)

    def retrieve(self, key: np.ndarray) -> np.ndarray:
        """
        Retrieve a value directly using a key.

        Args:
            key: The key to use for retrieval

        Returns:
            The retrieved value vector
        """
        key = self._dimensionalize(key)
        # Normalize the key
        key = key / np.linalg.norm(key)

        # Multiply memory matrix by key to get the value
        return self.memory_matrix @ key

    def check_orthogonality(self) -> bool:
        """
        Check if all keys are orthogonal to each other.

        Returns:
            True if all keys are orthogonal, False otherwise
        """
        for i, key1 in enumerate(self.keys):
            for j, key2 in enumerate(self.keys):
                if i != j:
                    dot_product = np.abs(np.dot(key1, key2))
                    if dot_product > 1e-6:  # Allow for small numerical errors
                        return False
        return True

    def __getitem__(self, key: np.ndarray) -> np.ndarray:
        return self.retrieve(key)

    def __setitem__(self, key: np.ndarray, value: np.ndarray):
        return self.insert_at(key, value)


def stress_test_matrix_memory(
        dim: int = 100,
        num_items: int = 100,
        noise_levels=None,
        verbose: bool = True
) -> Dict:
    """
    Stress test for the MatrixMemory class.

    Args:
        dim: Dimension of the memory space
        num_items: Number of items to insert
        noise_levels: List of noise levels to test during retrieval
        verbose: Whether to print detailed results

    Returns:
        Dictionary containing test results and statistics
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.01]
    results = {}

    # Initialize memory
    memory = MatrixMemory(dim=dim)

    # Create test data - values and corresponding keys
    values = []
    keys = []

    if verbose:
        print(f"Creating {num_items} test items with dimension {dim}...")

    # Generate test data
    for i in range(num_items):
        print(i)
        # Create random value vectors
        value = np.random.randn(dim)
        values.append(value)
        # Add to memory and store the key
        key = memory.insert(value)
        keys.append(key)

    if verbose:
        print(f"Checking orthogonality of all keys: {memory.check_orthogonality()}")

    # Test retrieval accuracy for different noise levels
    accuracy_results = {}
    retrieval_times = []

    for noise_level in noise_levels:
        if verbose:
            print(f"\nTesting retrieval with noise level: {noise_level}")

        # Track cosine similarities between original and retrieved values
        similarities = []

        # Time each retrieval
        start_time = time.time()

        for i in range(num_items):
            # Get original value and key
            original_value = values[i]
            key = keys[i]

            # Add noise to key if needed
            if noise_level > 0:
                noisy_key = key + np.random.normal(0, noise_level, key.shape)
                # Renormalize the key
                noisy_key = noisy_key / np.linalg.norm(noisy_key)
            else:
                noisy_key = key

            # Retrieve value
            retrieved_value = memory.retrieve(noisy_key)

            # Calculate cosine similarity between original and retrieved
            norm_orig = np.linalg.norm(original_value)
            norm_retr = np.linalg.norm(retrieved_value)

            if norm_orig > 0 and norm_retr > 0:
                similarity = np.dot(original_value, retrieved_value) / (norm_orig * norm_retr)
                similarities.append(similarity)
            else:
                similarities.append(0)

        retrieval_time = (time.time() - start_time) / num_items
        retrieval_times.append(retrieval_time)

        # Calculate statistics
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        min_similarity = np.min(similarities)

        accuracy_results[noise_level] = {
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'min_similarity': min_similarity,
            'num_perfect': sum(s > 0.999 for s in similarities),
            'num_good': sum(s > 0.95 for s in similarities),
            'num_acceptable': sum(s > 0.8 for s in similarities),
            'all_similarities': similarities
        }

        if verbose:
            print(f"  Mean similarity: {mean_similarity:.6f}")
            print(f"  Min similarity: {min_similarity:.6f}")
            print(f"  Perfect retrievals (>0.999): {accuracy_results[noise_level]['num_perfect']}/{num_items}")
            print(f"  Good retrievals (>0.95): {accuracy_results[noise_level]['num_good']}/{num_items}")
            print(f"  Acceptable retrievals (>0.8): {accuracy_results[noise_level]['num_acceptable']}/{num_items}")
            print(f"  Average retrieval time: {retrieval_time * 1000:.2f} ms")

    # Test memory capacity by adding items until retrieval quality degrades
    if verbose:
        print("\nTesting memory capacity...")

    capacity_values = []
    capacity_keys = []
    capacity_similarities = []
    max_capacity_items = 5 * dim  # Upper limit to avoid infinite loops
    degradation_threshold = 0.9

    # Keep track of all items for final testing
    all_capacity_values = []
    all_capacity_keys = []

    # Start with fresh memory
    capacity_memory = MatrixMemory(dim=dim)

    for i in range(max_capacity_items):
        # Create new value
        value = np.random.randn(dim)
        all_capacity_values.append(value)

        # Add to memory
        key = capacity_memory.insert(value)
        all_capacity_keys.append(key)

        # Test retrieval of all items added so far
        if (i + 1) % 50 == 0 or i < 10:
            similarities = []
            for j in range(i + 1):
                retrieved = capacity_memory.retrieve(all_capacity_keys[j])
                similarity = np.dot(all_capacity_values[j], retrieved) / (
                        np.linalg.norm(all_capacity_values[j]) * np.linalg.norm(retrieved)
                )
                similarities.append(similarity)

            avg_similarity = np.mean(similarities)
            capacity_values.append(i + 1)
            capacity_similarities.append(avg_similarity)

            if verbose and ((i + 1) % 50 == 0 or i < 10):
                print(f"  Items: {i + 1}, Average similarity: {avg_similarity:.6f}")

            # Check if quality has degraded significantly
            if avg_similarity < degradation_threshold:
                if verbose:
                    print(f"  Retrieval quality degraded below {degradation_threshold} at {i + 1} items")
                break

    # Compile results
    results = {
        'parameters': {
            'dim': dim,
            'num_items': num_items,
            'item_dim': dim
        },
        'accuracy': accuracy_results,
        'retrieval_times': retrieval_times,
        'capacity': {
            'values': capacity_values,
            'similarities': capacity_similarities
        },
        'orthogonality': memory.check_orthogonality()
    }

    return results


def plot_stress_test_results(results: Dict) -> None:
    """
    Plot the results from the stress test.

    Args:
        results: Dictionary of results from stress_test_matrix_memory
    """
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Similarity distributions for different noise levels
    ax1 = axes[0, 0]
    noise_levels = list(results['accuracy'].keys())

    for noise in noise_levels:
        similarities = results['accuracy'][noise]['all_similarities']
        ax1.hist(similarities, alpha=0.5, bins=20, label=f'Noise: {noise}')

    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Retrieval Similarities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Mean similarity vs noise level
    ax2 = axes[0, 1]
    mean_similarities = [results['accuracy'][noise]['mean_similarity'] for noise in noise_levels]
    min_similarities = [results['accuracy'][noise]['min_similarity'] for noise in noise_levels]

    ax2.plot(noise_levels, mean_similarities, 'o-', label='Mean Similarity')
    ax2.plot(noise_levels, min_similarities, 's--', label='Min Similarity')
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Similarity')
    ax2.set_title('Retrieval Quality vs Noise Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Memory capacity test
    ax3 = axes[1, 0]
    ax3.plot(results['capacity']['values'], results['capacity']['similarities'], 'o-')
    ax3.set_xlabel('Number of Items')
    ax3.set_ylabel('Average Retrieval Similarity')
    ax3.set_title('Memory Capacity Test')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Success rate (perfect, good, acceptable) vs noise
    ax4 = axes[1, 1]
    perfect_rates = [results['accuracy'][noise]['num_perfect'] / results['parameters']['num_items']
                     for noise in noise_levels]
    good_rates = [results['accuracy'][noise]['num_good'] / results['parameters']['num_items']
                  for noise in noise_levels]
    acceptable_rates = [results['accuracy'][noise]['num_acceptable'] / results['parameters']['num_items']
                        for noise in noise_levels]

    ax4.plot(noise_levels, perfect_rates, 'o-', label='Perfect (>0.999)')
    ax4.plot(noise_levels, good_rates, 's-', label='Good (>0.95)')
    ax4.plot(noise_levels, acceptable_rates, '^-', label='Acceptable (>0.8)')
    ax4.set_xlabel('Noise Level')
    ax4.set_ylabel('Success Rate')
    ax4.set_title('Retrieval Success Rate vs Noise Level')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_matrix_memory_benchmark():
    """Run a comprehensive benchmark of the MatrixMemory system"""
    # More challenging test with higher dimensionality
    print("\nRunning high-dimensionality test...")
    results_high_dim = stress_test_matrix_memory(
        dim=1024,
        num_items=1536,
        noise_levels=[0.0, 0.1],
        verbose=True
    )

    # Test memory update operations
    print("\nTesting memory update operations...")
    memory = MatrixMemory(dim=50)

    # Add initial values
    print("Adding 20 initial values...")
    original_values = []
    keys = []

    for i in range(20):
        value = np.random.randn(50)
        original_values.append(value)
        key = memory.insert(value)
        keys.append(key)

    # Update half the values
    print("Updating 10 values...")
    updated_values = []

    for i in range(10):
        new_value = np.random.randn(50)
        updated_values.append(new_value)
        memory[keys[i]] = new_value

    # Check retrieval accuracy
    correct_retrievals = 0

    # Check updated values
    for i in range(10):
        retrieved = memory.retrieve(keys[i])
        similarity = np.dot(updated_values[i], retrieved) / (
                np.linalg.norm(updated_values[i]) * np.linalg.norm(retrieved)
        )
        if similarity > 0.99:
            correct_retrievals += 1

    # Check untouched values
    for i in range(10, 20):
        retrieved = memory.retrieve(keys[i])
        similarity = np.dot(original_values[i], retrieved) / (
                np.linalg.norm(original_values[i]) * np.linalg.norm(retrieved)
        )
        if similarity > 0.99:
            correct_retrievals += 1

    print(f"Correct retrievals after updates: {correct_retrievals}/20")

    print("\nBenchmark completed!")


if __name__ == "__main__":
    run_matrix_memory_benchmark()
