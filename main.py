import numpy as np
import torch

from memory import MatrixMemory
from sentence_encoder import generate_batch_embeddings


def embed_ilias():
    with open("texts/ilias.txt", "r") as file:
        text = file.read()
    lines = text.split('\n')
    return generate_batch_embeddings(lines)


def generate_keys_mockup(key_dim, num_keys):
    keys = torch.zeros(num_keys, key_dim, device="cuda:0")
    for i in range(num_keys):
        keys[i] = torch.tensor(np.random.randn(key_dim), device="cuda:0")
    return keys


if __name__ == "__main__":
    memory = MatrixMemory()
    baseline_memory = MatrixMemory()
    line_embeddings = embed_ilias()
    keys = []
    baseline_keys = generate_keys_mockup(memory.dim, len(line_embeddings))
    for i, e in enumerate(line_embeddings):
        keys.append(memory.insert(e))
        baseline_memory.insert(e, key=baseline_keys[i])

    reconstruction_error = 0
    baseline_reconstruction_error = 0

    for i, e in enumerate(line_embeddings):
        reconstruction = memory.retrieve(keys[i])
        baseline_reconstruction = baseline_memory.retrieve(baseline_keys[i])
        actual: np.ndarray = e
        error = np.mean((actual - reconstruction) ** 2)
        baseline_error = np.mean((actual - baseline_reconstruction) ** 2)

        reconstruction_error += error
        baseline_reconstruction_error += baseline_error

    # Calculate average reconstruction error
    avg_reconstruction_error = reconstruction_error / len(line_embeddings)
    avg_baseline_error = baseline_reconstruction_error / len(line_embeddings)

    print(f"Average reconstruction error: {avg_reconstruction_error:.6f}")
    print(f"Average baseline reconstruction error: {avg_baseline_error:.6f}")
