import torch
import torch.nn as nn
from typing import Dict, Any, Tuple

# Model type constants - use strings with underscores consistently
MODEL_TYPE_ENCODER_ONLY = "encoder_only"
MODEL_TYPE_DECODER_ONLY = "decoder_only"
MODEL_TYPE_ENCODER_DECODER = "encoder_decoder"

# Task mode constants
TASK_MODE_MLM = "mlm"
TASK_MODE_NEXT_TOKEN = "next_token"
TASK_MODE_SEQ2SEQ = "seq2seq"


def prepare_batch_for_model(
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
        model_type: str,
        task_mode: str,
        device: torch.device,
        ignore_index: int = -100
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prepare a batch for the model based on model type and task mode.

    Args:
        batch: The input batch dictionary
        model: The transformer model
        model_type: Type of transformer model (encoder_only, decoder_only, encoder_decoder)
        task_mode: Training task mode (mlm, next_token, seq2seq)
        device: The device for computation
        ignore_index: Index to ignore in loss computation

    Returns:
        model_inputs: Dictionary of inputs for the model
        labels_info: Dictionary containing label tensors and metadata for loss calculation
    """
    # Move batch to device
    batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    # Default outputs
    model_inputs = {}
    labels_info = {"labels": None}

    # Generate appropriate masks and inputs based on model type and task
    if model_type == MODEL_TYPE_ENCODER_DECODER:
        # Create source mask [batch_size, 1, seq_len]
        src_mask = batch["attention_mask"].unsqueeze(1)

        # Create target mask [batch_size, seq_len, seq_len]
        tgt_mask = model.generate_square_subsequent_mask(
            batch["decoder_input_ids"].size(1),
            batch["decoder_input_ids"].device
        )

        # Create padding masks
        src_key_padding_mask = (batch["attention_mask"] == 0)
        tgt_key_padding_mask = (batch["decoder_attention_mask"] == 0)

        # Prepare model inputs
        model_inputs = {
            "src": batch["input_ids"],
            "tgt": batch["decoder_input_ids"],
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "src_key_padding_mask": src_key_padding_mask,
            "tgt_key_padding_mask": tgt_key_padding_mask
        }

        # Labels for loss calculation
        labels_info["labels"] = batch["labels"]

    elif model_type == MODEL_TYPE_DECODER_ONLY:
        if task_mode == TASK_MODE_MLM:
            # For MLM in decoder-only model, we still use causal attention
            # but the labels have masks at specific positions
            tgt_mask = model.generate_square_subsequent_mask(
                batch["input_ids"].size(1),
                batch["input_ids"].device
            )

            # Prepare model inputs - use tgt parameter for decoder-only models
            model_inputs = {
                "tgt": batch["input_ids"],
                "tgt_mask": tgt_mask,
            }

            # MLM labels - already prepared by DataLoader
            labels_info["labels"] = batch["labels"]

        elif task_mode == TASK_MODE_NEXT_TOKEN:
            # For next-token prediction, we use causal attention
            tgt_mask = model.generate_square_subsequent_mask(
                batch["input_ids"].size(1),
                batch["input_ids"].device
            )

            # Prepare model inputs
            model_inputs = {
                "tgt": batch["input_ids"],
                "tgt_mask": tgt_mask,
            }

            # Next-token labels
            labels_info["labels"] = batch["labels"]

    elif model_type == MODEL_TYPE_ENCODER_ONLY:
        # Encoder-only models - typically used for classification or feature extraction
        # For MLM specifically:
        if task_mode == TASK_MODE_MLM:
            # Create mask for self-attention
            src_mask = batch["attention_mask"].unsqueeze(1).unsqueeze(2)

            # Prepare model inputs
            model_inputs = {
                "src": batch["input_ids"],
                "src_mask": src_mask,
            }

            # MLM labels
            labels_info["labels"] = batch["labels"]

    # Return both the model inputs and labels information
    return model_inputs, labels_info


def compute_loss(
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute loss for the model outputs and labels.

    Args:
        outputs: Model outputs of shape [batch_size, seq_len, vocab_size]
        labels: Label indices of shape [batch_size, seq_len]
        criterion: Loss function
        ignore_index: Index to ignore in loss computation

    Returns:
        Loss tensor
    """
    # Ensure labels are properly formatted for cross-entropy loss
    # Reshape outputs to [batch_size * seq_len, vocab_size]
    outputs_flat = outputs.reshape(-1, outputs.size(-1))

    # Reshape labels to [batch_size * seq_len]
    labels_flat = labels.reshape(-1)

    # Apply loss function
    loss = criterion(outputs_flat, labels_flat)

    return loss


def setup_hidden_state_hooks(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Set up forward hooks to capture hidden states from all layers.

    Args:
        model: The transformer model

    Returns:
        Dictionary to store hidden states
    """
    hidden_states = {}

    def save_hidden_state(name):
        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                # Some layers return tuples (like attention layers)
                hidden_states[name] = output[0].detach()
            else:
                # Single tensor output
                hidden_states[name] = output.detach()

        return hook

    # Attach hooks based on model architecture
    if hasattr(model, 'encoder') and model.encoder is not None:
        for idx, layer in enumerate(model.encoder.layers):
            layer.register_forward_hook(save_hidden_state(f"encoder_layer_{idx}"))

    if hasattr(model, 'decoder') and model.decoder is not None:
        for idx, layer in enumerate(model.decoder.layers):
            layer.register_forward_hook(save_hidden_state(f"decoder_layer_{idx}"))

    return hidden_states


def get_device_from_string(device_str: str) -> torch.device:
    """
    Convert device string to torch.device, with auto-detection.

    Args:
        device_str: Device string ('auto', 'cpu', 'cuda', etc.)

    Returns:
        torch.device object
    """
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_str)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
        model: nn.Module,
        save_path: str,
        model_type: str,
        tokenizer,
        epoch: int,
        optimizer,
        best_val_loss: float,
        batch_size: int,
        epochs: int,
        lr: float,
        warmup_steps: int = None,
        weight_decay: float = None
):
    """
    Save model checkpoint with metadata.

    Args:
        model: The model to save
        save_path: Path to save the checkpoint
        model_type: Type of the model
        tokenizer: Tokenizer used
        epoch: Current epoch
        optimizer: Optimizer state
        best_val_loss: Best validation loss achieved
        batch_size: Training batch size
        epochs: Total epochs
        lr: Learning rate
        warmup_steps: Warmup steps used
        weight_decay: Weight decay used
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model_type': model_type,
        'hyperparameters': {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'warmup_steps': warmup_steps,
            'weight_decay': weight_decay
        }
    }

    torch.save(checkpoint, save_path)


def validate_model(
        model: nn.Module,
        val_loader,
        criterion: nn.Module,
        device: torch.device,
        model_type: str,
        task_mode: str
) -> float:
    """
    Validate the model on a validation dataset.

    Args:
        model: The transformer model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device for computation
        model_type: Type of transformer model
        task_mode: Training task mode

    Returns:
        Average validation loss
    """
    from tqdm import tqdm

    model.eval()
    total_val_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Prepare inputs and labels based on model type and task mode
            model_inputs, labels_info = prepare_batch_for_model(
                batch, model, model_type, task_mode, device
            )

            # Forward pass
            outputs = model(**model_inputs)

            # Calculate loss
            loss = compute_loss(outputs, labels_info["labels"], criterion)

            # Update running totals
            total_val_loss += loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)

    # Calculate average loss
    avg_val_loss = total_val_loss / total_samples if total_samples > 0 else float('inf')

    return avg_val_loss