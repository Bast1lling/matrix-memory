# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


def setup_t5_encoder_decoder(model_name="t5-small"):
    """Setup a T5 encoder-decoder model"""
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
    return model, tokenizer


def encode_to_vector(text, model, tokenizer, return_hidden_state=True):
    """Encode text to vector representation"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True
        )

    if return_hidden_state:
        # Return the last hidden state
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    else:
        # Return the full encoder outputs for more flexibility
        return outputs


def decode_from_hidden_state(hidden_state, model, tokenizer, max_length=50):
    """Decode from hidden state back to text"""
    if isinstance(hidden_state, torch.Tensor):
        encoder_outputs = hidden_state
    else:
        # Convert numpy array back to tensor if needed
        encoder_outputs = torch.tensor(hidden_state)

    # Create the encoder outputs object expected by the decoder
    encoder_outputs = {"last_hidden_state": encoder_outputs.unsqueeze(0)}

    # Generate from the hidden state
    output_ids = model.generate(
        encoder_outputs=encoder_outputs,
        max_length=max_length,
    )

    # Decode the generated IDs back to text
    decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded_text


def example_usage():
    """Example of encoding and decoding"""
    model, tokenizer = setup_t5_encoder_decoder()

    original_text = "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    print(f"Original: {original_text}")

    # Encode
    hidden_state = encode_to_vector(original_text, model, tokenizer)
    print(f"Vector len: {len(hidden_state)}")

    # Decode
    reconstructed_text = decode_from_hidden_state(hidden_state, model, tokenizer)
    print(f"Reconstructed: {reconstructed_text}")

    return original_text, reconstructed_text


if __name__ == "__main__":
    example_usage()
