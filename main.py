import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalKeyGenerator(nn.Module):
    def __init__(self, input_dim, key_dim, hidden_dim=128):
        """
        Neural network that generates orthogonal keys based on input values.

        Args:
            input_dim (int): Dimension of the input value vectors
            key_dim (int): Dimension of the output key vectors
            hidden_dim (int): Dimension of the hidden layers
        """
        super(OrthogonalKeyGenerator, self).__init__()

        self.input_dim = input_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim

        # Feature extraction from input values
        self.value_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # RNN to maintain history of previously generated keys
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Key generator that produces the output vector
        self.key_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, key_dim)
        )

    def forward(self, value, prev_keys=None, hidden_state=None):
        """
        Generate a new key that is orthogonal to previously generated keys.

        Args:
            value (torch.Tensor): Input value vector [batch_size, input_dim]
            prev_keys (torch.Tensor, optional): Previously generated keys [batch_size, num_prev_keys, key_dim]
            hidden_state (torch.Tensor, optional): Hidden state from previous step

        Returns:
            tuple: (new_key, hidden_state)
                new_key (torch.Tensor): Generated key that's orthogonal to prev_keys
                hidden_state (torch.Tensor): Updated hidden state
        """
        batch_size = value.size(0)

        # Encode the input value
        encoded_value = self.value_encoder(value)  # [batch_size, hidden_dim]

        # Reshape for RNN input (expects [batch_size, seq_len, input_size])
        rnn_input = encoded_value.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Process with RNN
        if hidden_state is None:
            # Initialize hidden state if not provided
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim, device=value.device)

        rnn_out, hidden_state = self.rnn(rnn_input, hidden_state)

        # Generate the initial key
        key = self.key_generator(rnn_out.squeeze(1))  # [batch_size, key_dim]

        # If there are previous keys, ensure orthogonality
        if prev_keys is not None and prev_keys.size(1) > 0:
            # Normalize the key
            key = F.normalize(key, p=2, dim=1)

            # Iteratively make the key orthogonal to all previous keys
            for i in range(prev_keys.size(1)):
                prev_key = prev_keys[:, i, :]

                # Calculate dot product
                dot_product = torch.sum(key * prev_key, dim=1, keepdim=True)

                # Subtract the projection
                key = key - dot_product * prev_key

                # Renormalize after each projection
                key = F.normalize(key, p=2, dim=1)
        else:
            # Just normalize if no previous keys
            key = F.normalize(key, p=2, dim=1)

        return key, hidden_state

    def generate_sequence(self, values):
        """
        Generate a sequence of orthogonal keys for a sequence of values.

        Args:
            values (torch.Tensor): Sequence of value vectors [batch_size, seq_len, input_dim]

        Returns:
            torch.Tensor: Sequence of orthogonal keys [batch_size, seq_len, key_dim]
        """
        batch_size, seq_len, _ = values.size()
        keys = torch.zeros(batch_size, seq_len, self.key_dim, device=values.device)
        hidden = None

        for t in range(seq_len):
            value_t = values[:, t, :]
            prev_keys = keys[:, :t, :] if t > 0 else None

            key_t, hidden = self.forward(value_t, prev_keys, hidden)
            keys[:, t, :] = key_t

        return keys


def orthogonality_loss(keys, epsilon=1e-8):
    """
    Compute loss based on orthogonality of keys.

    Args:
        keys (torch.Tensor): Batch of sequences of keys [batch_size, seq_len, key_dim]
        epsilon (float): Small value to prevent numerical issues

    Returns:
        torch.Tensor: Loss value measuring deviation from orthogonality
    """
    batch_size, seq_len, _ = keys.size()
    loss = torch.tensor(0.0, device=keys.device)

    for b in range(batch_size):
        # Compute pairwise dot products for all keys in the sequence
        key_seq = keys[b]  # [seq_len, key_dim]
        dot_products = torch.mm(key_seq, key_seq.transpose(0, 1))  # [seq_len, seq_len]

        # Create a target matrix (identity matrix for orthogonal vectors)
        target = torch.eye(seq_len, device=keys.device)

        # Compute the loss as the sum of squared differences
        seq_loss = torch.sum((dot_products - target) ** 2)
        loss += seq_loss / (seq_len ** 2)

    return loss / batch_size


class OrthogonalKeyTrainer:
    def __init__(self, model, learning_rate=0.001):
        """
        Trainer for the OrthogonalKeyGenerator model.

        Args:
            model (OrthogonalKeyGenerator): The model to train
            learning_rate (float): Learning rate for optimization
        """
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_step(self, values, orthogonality_weight=1.0):
        """
        Perform a single training step.

        Args:
            values (torch.Tensor): Batch of value sequences [batch_size, seq_len, input_dim]
            orthogonality_weight (float): Weight for the orthogonality loss

        Returns:
            dict: Dictionary containing loss information
        """
        self.optimizer.zero_grad()

        # Generate keys from values
        keys = self.model.generate_sequence(values)

        # Compute orthogonality loss
        ortho_loss = orthogonality_loss(keys)

        # Total loss
        total_loss = orthogonality_weight * ortho_loss

        # Backpropagation and optimization
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'orthogonality_loss': ortho_loss.item()
        }


# Example usage
def example():
    # Parameters
    batch_size = 8
    seq_len = 5
    input_dim = 64
    key_dim = 32

    # Create model
    model = OrthogonalKeyGenerator(input_dim=input_dim, key_dim=key_dim)
    trainer = OrthogonalKeyTrainer(model)

    # Generate random values for demonstration
    values = torch.randn(batch_size, seq_len, input_dim)

    # Train for a few steps
    for i in range(10):
        loss_info = trainer.train_step(values)
        print(f"Step {i + 1}, Total Loss: {loss_info['total_loss']:.6f}, "
              f"Orthogonality Loss: {loss_info['orthogonality_loss']:.6f}")

    # Generate keys
    keys = model.generate_sequence(values)

    # Check orthogonality of generated keys
    for b in range(1):  # Check the first batch
        key_seq = keys[b]
        dot_products = torch.mm(key_seq, key_seq.transpose(0, 1))
        print("\nDot product matrix for generated keys:")
        print(dot_products.detach().numpy().round(3))


if __name__ == "__main__":
    example()