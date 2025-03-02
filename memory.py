import os.path

import numpy as np
import torch
from network import StatefulKeyOrthogonalizer


class MatrixMemory:
    """
    A high-dimensional associative memory that stores values using orthogonal keys.

    This implementation uses the Gram-Schmidt process to ensure all keys are
    orthogonal to each other, enabling efficient retrieval with minimal interference.
    """

    def __init__(self, dim: int = 384, model_name="simple_generator_384.pt"):
        """
        Initialize a MatrixMemory with a given dimensionality.

        Args:
            dim: Dimension of the memory space
        """
        self.dim = dim
        self.memory_matrix = np.zeros((dim, dim))
        model_path = os.path.join("models", model_name)
        self.model = StatefulKeyOrthogonalizer(dim)
        self.model.load_state_dict(torch.load(model_path))

    def _generate_random_keys(self, amount: int):
        self.model.eval()
        self.model.reset_state()
        with torch.no_grad():
            keys = self.model.generate_sequence(amount)
        return keys

    def _dimensionalize(self, vector) -> np.ndarray:
        if isinstance(vector, np.ndarray):
            pass
        elif isinstance(vector, torch.Tensor):
            vector = vector.numpy()
        else:
            raise ValueError

        if len(vector) != self.dim:
            # Resize or pad the value to match memory dimensions
            if len(vector) > self.dim:
                vector = vector[:self.dim]
            else:
                pad_width = self.dim - len(vector)
                vector = np.pad(vector, (0, pad_width), 'constant')
        return vector

    def insert(self, value, key=None):
        """
        Add a value to memory with a newly generated orthogonal key.

        Args:
            value: The value to store
            key: Optional self generated key

        Returns:
            The ID assigned to this value
        """
        # Ensure the value has the right dimensionality
        value_array = self._dimensionalize(value)

        # Generate new random key and orthogonalize it
        if not key:
            key = self._generate_random_keys(1)

        # Update memory matrix with outer product
        self.memory_matrix += np.outer(value_array, key)

        return key

    def replace(self, key: np.ndarray, value: np.ndarray):
        value_array = self._dimensionalize(value)
        key_array = self._dimensionalize(key)
        old_value = self.memory_matrix @ key_array
        self.memory_matrix -= np.outer(old_value, key_array)
        # Add new contribution to memory matrix
        self.memory_matrix += np.outer(value_array, key_array)

    def retrieve(self, key):
        """
        Retrieve a value directly using a key.

        Args:
            key: The key to use for retrieval

        Returns:
            The retrieved value vector
        """
        key_array = self._dimensionalize(key)
        return self.memory_matrix @ key_array

    def __getitem__(self, key):
        return self.retrieve(key)

    def __setitem__(self, key, value):
        return self.replace(key, value)
