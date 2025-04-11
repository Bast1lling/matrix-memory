import numpy as np
import torch

from memory import MatrixMemory


def generate_random_vectors(key_dim, num_keys, mean=0, std=1):
    keys = torch.zeros(num_keys, key_dim, device="cuda:0")
    for i in range(num_keys):
        vector = np.random.randn(key_dim)
        vector = std * vector + mean
        keys[i] = torch.tensor(vector, device="cuda:0")
    return keys


def compute_error(vectorA: np.ndarray, vectorB: np.ndarray) -> float:
    squared_diff = np.square(vectorA - vectorB)
    return np.mean(squared_diff)


def benchmark(num_samples: int):
    memory = MatrixMemory()
    baseline_memory = MatrixMemory()
    print("Generating values...")
    values = generate_random_vectors(memory.dim, num_samples)
    baseline_keys = generate_random_vectors(memory.dim, num_samples)
    print("Inserting values...")
    # insert values
    keys = []
    for i, value in enumerate(values):
        key = memory.insert(value)
        keys.append(key)
        baseline_memory.insert(value, key=baseline_keys[i])

    reconstruction_errors = []
    baseline_reconstruction_errors = []
    print("Reconstructing values")
    # retrieve values
    for i, value in enumerate(values):
        reconstruction = memory.retrieve(keys[i])
        baseline_reconstruction = baseline_memory.retrieve(baseline_keys[i])
        value_np = value.cpu().squeeze().numpy()
        reconstruction_errors.append(compute_error(value_np, reconstruction))
        baseline_reconstruction_errors.append(compute_error(value_np, baseline_reconstruction))

    score = np.mean(np.array(reconstruction_errors))
    baseline_score = np.mean(np.array(baseline_reconstruction_errors))

    print(f"Average reconstruction error: {score:2f}")
    print(f"Average baseline reconstruction error: {baseline_score:2f}")


if __name__ == "__main__":
    benchmark(100)
