import numpy as np
from sentence_transformers import SentenceTransformer


def generate_sentence_embedding(text: str, model_name: str = "all-MiniLM-L6-v2"):
    """
    Generate a vector embedding from a Python string using sentence-transformers.

    Args:
        text: The input text to embed
        model_name: The name of the sentence-transformers model to use
                    (default: "all-MiniLM-L6-v2")

    Returns:
        np.ndarray: The embedding vector for the input text
    """
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)

    # Generate embedding
    return model.encode(text)


def generate_batch_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2"):
    """
    Generate vector embeddings for a batch of strings.

    Args:
        texts: List of input texts to embed
        model_name: The name of the sentence-transformers model to use
                    (default: "all-MiniLM-L6-v2")

    Returns:
        np.ndarray: Array of embedding vectors for the input texts
    """
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)

    # Generate embeddings for all texts at once (more efficient than one by one)
    return model.encode(texts)


if __name__ == "__main__":
    # Example usage
    test_str = "def hello_world():\n    print('Hello, world!')"
    embedding = generate_sentence_embedding(test_str)
    print(f"Embedding shape: {embedding.shape}")

    # Example with multiple strings
    code_samples = [
        "def add(a, b):\n    return a + b",
        "class Person:\n    def __init__(self, name):\n        self.name = name",
        "import numpy as np\nx = np.array([1, 2, 3])"
    ]

    embeddings = generate_batch_embeddings(code_samples)
    print(f"Batch embeddings shape: {embeddings.shape}")