import torch
import torch.nn as nn
import torch.nn.functional as F


class StatefulKeyOrthogonalizer(nn.Module):
    def __init__(self, key_dim, hidden_dim=256, rnn_layers=2):
        """
        Neural network that maintains a memory of previous keys and generates orthogonal keys
        without requiring explicit storage of previous keys.

        Args:
            key_dim (int): Dimension of the key vectors
            hidden_dim (int): Dimension of the hidden layers
            rnn_layers (int): Number of LSTM layers
        """
        super(StatefulKeyOrthogonalizer, self).__init__()

        self.device = "cuda:0"
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers

        # LSTM to maintain state representing all previously generated keys
        self.lstm = nn.LSTM(
            input_size=key_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True
        )

        # Key transformer to make a key orthogonal to the implicit memory
        self.key_transformer = nn.Sequential(
            nn.Linear(hidden_dim + key_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, key_dim)
        )

        # Initialize hidden state and cell state
        self.reset_state()

    def reset_state(self, batch_size=1):
        """Reset the LSTM state to forget all previous keys."""
        self.hidden_state = torch.zeros(self.rnn_layers, batch_size, self.hidden_dim, device=self.device)
        self.cell_state = torch.zeros(self.rnn_layers, batch_size, self.hidden_dim, device=self.device)

    def forward(self, candidate_key):
        """
        Transform a candidate key to make it more orthogonal to previously generated keys.
        The memory of previous keys is maintained in the LSTM state.

        Args:
            candidate_key (torch.Tensor): The input candidate key [batch_size, key_dim]

        Returns:
            torch.Tensor: Transformed key that's more orthogonal to previous keys
        """
        batch_size = candidate_key.size(0)

        # Ensure hidden state matches batch size
        if self.hidden_state.size(1) != batch_size:
            self.reset_state(batch_size)

        # Reshape candidate key for LSTM input
        lstm_input = candidate_key.unsqueeze(1)  # [batch_size, 1, key_dim]

        # Process with LSTM, using and updating the internal state
        lstm_out, (self.hidden_state, self.cell_state) = self.lstm(
            lstm_input,
            (self.hidden_state, self.cell_state)
        )

        # Get LSTM output for orthogonalization (contains information about previous keys)
        lstm_context = lstm_out.squeeze(1)  # [batch_size, hidden_dim]

        # Combine candidate key with LSTM context
        combined = torch.cat([candidate_key, lstm_context], dim=1)  # [batch_size, key_dim + hidden_dim]

        # Transform the key to be orthogonal to previous keys
        transformed_key = self.key_transformer(combined)  # [batch_size, key_dim]

        # Normalize the key
        transformed_key = F.normalize(transformed_key, p=2, dim=1)

        return transformed_key

    def generate_sequence(self, num_keys, initial_key=None):
        """
        Generate a sequence of orthogonal keys.

        Args:
            num_keys (int): Number of keys to generate
            initial_key (torch.Tensor, optional): Initial key to start with, or random if None

        Returns:
            torch.Tensor: Sequence of orthogonal keys [num_keys, key_dim]
        """
        device = self.device

        # Reset the LSTM state
        self.reset_state()

        # Initialize sequence storage
        keys = torch.zeros(num_keys, self.key_dim, device=device)

        # Generate first key or use provided initial key
        if initial_key is None:
            key = torch.randn(1, self.key_dim, device=device)
            key = F.normalize(key, p=2, dim=1)
        else:
            key = initial_key.unsqueeze(0) if initial_key.dim() == 1 else initial_key

        # Store first key
        keys[0] = key.squeeze(0)

        # Generate remaining keys one by one
        for i in range(1, num_keys):
            # Generate random candidate
            candidate = torch.randn(1, self.key_dim, device=device)
            candidate = F.normalize(candidate, p=2, dim=1)

            # Transform to be orthogonal to previous keys
            denoiser = self.to(self.device)
            key = denoiser(candidate.to(self.device))

            # Store the key
            keys[i] = key.squeeze(0)

        return keys


class StatefulOrthogonalizerTrainer:
    def __init__(self, model, learning_rate=0.001):
        """
        Trainer for the StatefulKeyOrthogonalizer model.

        Args:
            model (StatefulKeyOrthogonalizer): The model to train
            learning_rate (float): Learning rate for optimization
        """
        self.device = model.device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def orthogonality_loss(self, keys):
        """
        Compute loss based on the orthogonality of a batch of keys.

        Args:
            keys (torch.Tensor): Batch of keys [batch_size, seq_len, key_dim]

        Returns:
            torch.Tensor: Loss value measuring deviation from orthogonality
        """
        keys = keys.to(self.device)
        batch_size, seq_len, _ = keys.size()
        loss = torch.tensor(0.0, device=self.device)

        for b in range(batch_size):
            # Compute pairwise dot products for all keys in the sequence
            key_seq = keys[b]  # [seq_len, key_dim]
            dot_products = torch.mm(key_seq, key_seq.transpose(0, 1))  # [seq_len, seq_len]

            # Create a target matrix (identity matrix for orthogonal vectors)
            target = torch.eye(seq_len, device=self.device)

            # Compute the loss as the sum of squared differences
            seq_loss = torch.sum((dot_products - target) ** 2)
            loss += seq_loss / (seq_len * (seq_len - 1))  # Normalize by number of off-diagonal elements

        return loss / batch_size

    def train_step(self, seq_length=20, batch_size=16):
        """
        Perform a single training step by generating sequences of keys.

        Args:
            seq_length (int): Length of key sequences to generate
            batch_size (int): Number of sequences to generate in parallel

        Returns:
            dict: Dictionary containing loss information
        """
        self.optimizer.zero_grad()

        # Reset model state for each batch
        self.model.reset_state(batch_size)

        # Storage for generated keys
        all_keys = []

        # Generate first keys randomly
        keys = torch.randn(batch_size, 1, self.model.key_dim, device=self.device)
        keys = F.normalize(keys, p=2, dim=2)
        all_keys.append(keys)

        # Generate remaining keys with the model
        for i in range(1, seq_length):
            # Generate random candidates
            candidates = torch.randn(batch_size, self.model.key_dim, device=self.device)
            candidates = F.normalize(candidates, p=2, dim=1)

            # Get orthogonalized keys
            new_keys = self.model(candidates).unsqueeze(1)  # [batch_size, 1, key_dim]
            all_keys.append(new_keys)

        # Combine all keys into a single tensor
        key_sequences = torch.cat(all_keys, dim=1)  # [batch_size, seq_length, key_dim]

        # Compute orthogonality loss
        loss = self.orthogonality_loss(key_sequences)

        # Backpropagation and optimization
        loss.backward()
        self.optimizer.step()

        return {'total_loss': loss.item()}


def train_stateful_orthogonalizer(key_dim=1024, epochs=50, batch_size=16, seq_length=2048,
                                  save_path="stateful_orthogonalizer.pt"):
    """
    Train the stateful neural orthogonalizer.

    Args:
        key_dim: Dimension of the keys
        epochs: Number of training epochs
        batch_size: Number of batches to process in parallel
        seq_length: Length of key sequences to generate
        save_path: Where to save the trained model
    """
    print(f"Training stateful orthogonalizer (key_dim={key_dim}, seq_length={seq_length})...")

    # Create model and trainer
    model = StatefulKeyOrthogonalizer(key_dim=key_dim)
    trainer = StatefulOrthogonalizerTrainer(model)

    # Training loop
    for epoch in range(epochs):
        loss_info = trainer.train_step(seq_length=seq_length, batch_size=batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_info['total_loss']:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to '{save_path}'")

    return model


def evaluate_orthogonality(model, num_keys=100):
    """
    Evaluate the orthogonality of keys generated by the model.

    Args:
        model: The StatefulKeyOrthogonalizer model
        num_keys: Number of keys to generate

    Returns:
        float: Average absolute dot product between key pairs
    """
    model.eval()
    model.reset_state()

    with torch.no_grad():
        keys = model.generate_sequence(num_keys)

    # Calculate dot products
    dot_products = []
    for i in range(num_keys):
        for j in range(i + 1, num_keys):
            dot = torch.abs(torch.dot(keys[i], keys[j]))
            dot_products.append(dot.item())

    return sum(dot_products) / len(dot_products) if dot_products else 0


def test_stateful_orthogonalizer(model_path="stateful_orthogonalizer.pt", key_dim=100, num_keys=100):
    """
    Test the stateful orthogonalizer.

    Args:
        model_path: Path to the trained model
        key_dim: Dimension of the keys
        num_keys: Number of keys to generate
    """
    # Load model
    model = StatefulKeyOrthogonalizer(key_dim=key_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate keys
    print(f"Generating {num_keys} orthogonal keys...")
    model.reset_state()

    with torch.no_grad():
        keys = model.generate_sequence(num_keys)

    # Evaluate orthogonality
    avg_dot_product = evaluate_orthogonality(model, num_keys)
    print(f"Average absolute dot product between key pairs: {avg_dot_product:.6f}")

    # Check a few specific dot products
    print("\nSample dot products between generated keys:")
    for i in range(min(5, num_keys)):
        for j in range(i + 1, min(i + 3, num_keys)):
            dot = torch.dot(keys[i], keys[j]).item()
            print(f"  Keys {i + 1} and {j + 1}: {dot:.6f}")


train_stateful_orthogonalizer()
