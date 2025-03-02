import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union


class MatrixMemory:
    """
    A high-dimensional associative memory that stores values using orthogonal keys.

    This implementation uses the Gram-Schmidt process to ensure all keys are
    orthogonal to each other, enabling efficient retrieval with minimal interference.
    """

    def __init__(self, dim: int = 100):
        """
        Initialize a MatrixMemory with a given dimensionality.

        Args:
            dim: Dimension of the memory space
        """
        self.dim = dim
        self.memory_matrix = np.zeros((dim, dim))  # The memory matrix
        self.keys = []  # Store all keys for orthogonalization

    def _generate_random_key(self) -> np.ndarray:
        """Generate a random vector for key initialization."""
        return np.random.randn(self.dim)

    def _orthogonalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Apply Gram-Schmidt orthogonalization to make a vector orthogonal
        to all existing keys.

        Args:
            vector: The vector to orthogonalize

        Returns:
            The orthogonalized unit vector
        """
        v = vector.copy()

        # Subtract projections onto all existing keys
        for key in self.keys:
            projection = np.dot(v, key) * key
            v = v - projection

        # Normalize to unit length if not zero
        norm = np.linalg.norm(v)
        if norm < 1e-10:  # Avoid division by near-zero
            # If vector became near-zero after orthogonalization,
            # generate a new random one and try again
            return self._orthogonalize(self._generate_random_key())

        return v / norm

    def _dimensionalize(self, vector: np.ndarray) -> np.ndarray:
        if len(vector) != self.dim:
            # Resize or pad the value to match memory dimensions
            if len(vector) > self.dim:
                vector = vector[:self.dim]
            else:
                pad_width = self.dim - len(vector)
                vector = np.pad(vector, (0, pad_width), 'constant')
        return vector

    def add(self, value: np.ndarray) -> np.ndarray:
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
        orthogonal_key = self._orthogonalize(key)

        # Add the key to our list
        self.keys.append(orthogonal_key)

        # Update memory matrix with outer product
        self.memory_matrix += np.outer(value_array, orthogonal_key)

        return orthogonal_key

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

    def __getitem__(self, key: np.ndarray):
        return self.retrieve(key)

    def __setitem__(self, key: np.ndarray, value: np.ndarray):
        pass


if __name__ == "__main__":
    # Create a memory with 10-dimensional space
    memory = MatrixMemory(dim=10)

    # Add some values
    value1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    value2 = np.array([0.5, -1.5, 2.5, -3.5, 4.5, -5.5, 6.5, -7.5, 8.5, -9.5])
    value3 = "This is a text value"  # Non-numeric values are supported too

    # Store values in memory
    id1 = memory.add(value1)
    id2 = memory.add(value2)
    id3 = memory.add(value3)

    print(f"Value IDs: {id1}, {id2}, {id3}")

    # Retrieve values using their IDs
    retrieved1 = memory.retrieve_by_key(memory.key_map[id1])
    retrieved2 = memory.retrieve_by_key(memory.key_map[id2])
    retrieved3 = memory.retrieve_by_key(memory.key_map[id3])

    print(f"Retrieved value 1: {retrieved1}")
    print(f"Retrieved value 2: {retrieved2}")
    print(f"Retrieved value 3: {retrieved3}")

    # Check if all keys are orthogonal
    print(f"Keys are orthogonal: {memory.check_orthogonality()}")

    # Retrieve by similarity
    query = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1])
    similar_values = memory.retrieve_most_similar(query, top_k=2)

    print("\nSimilarity search results:")
    for value_id, similarity, value in similar_values:
        print(f"Value ID: {value_id}, Similarity: {similarity:.4f}")
        if isinstance(value, np.ndarray):
            print(f"Value: {value}")
        else:
            print(f"Value: {value}")

    # Direct retrieval using a key
    key = memory.key_map[id1]
    retrieved_vector = memory.retrieve_by_key(key)
    print(f"\nRetrieved using key directly: {retrieved_vector}")
