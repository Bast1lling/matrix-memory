# Matrix Memory Project

This project's core idea is inspired by [this research paper](https://arxiv.org/pdf/2405.04517) and leverages the exponential growth of "almost" orthogonal vectors in high-dimensional spaces.

## Project Structure

- `network.py`: Contains the main neural network implementation and training logic
- `memory.py`: Implements memory management functionality
- `sentence_encoder.py`: Handles text encoding
- `algorithms.py`: Contains various algorithms used in the project
- `test.py`: Contains test cases and evaluation code
- `models/`: Directory for storing trained models
- `texts/`: Directory containing text data

## Setup

1. Create a virtual environment with conda, uv, poetry etc.

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training a Model

To train the stateful orthogonalizer model:

```bash
python network.py
```

The training process will:
- Initialize a neural network with specified dimensions
- Train the model to generate orthogonal keys
- Save the trained model to the `models/` directory

You can customize training parameters by modifying the `train_stateful_orthogonalizer` function in `network.py`:
- `lr`: Learning rate (default: 0.001)
- `key_dim`: Dimension of key vectors (default: 1024)
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Training batch size (default: 16)
- `seq_length`: Sequence length for training (default: 2048)

## Testing the Model

To test a trained model:

```bash
python test.py
```

The test will:
- Load the trained model from the `models/` directory
- Generate a sequence of keys
- Evaluate the orthogonality of the generated keys
- Print the results
