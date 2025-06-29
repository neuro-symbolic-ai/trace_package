import math
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List

from tqdm import tqdm

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
) -> torch.Tensor:
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

    # Attach hooks based on model architecture - imitate run with hooks from transformerlens
    if hasattr(model, 'encoder') and model.encoder is not None:
        for idx, layer in enumerate(model.encoder.layers):
            layer.register_forward_hook(save_hidden_state(f"encoder_layer_{idx}"))

    if hasattr(model, 'decoder') and model.decoder is not None:
        for idx, layer in enumerate(model.decoder.layers):
            layer.register_forward_hook(save_hidden_state(f"decoder_layer_{idx}"))

    return hidden_states



def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
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
    print(f"Saving checkpoint to {save_path} at epoch {epoch} with best validation loss {best_val_loss:.4f}")
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


def calculate_token_accuracy(pred_tokens: List[int], true_tokens: List[int]) -> float:
    if not true_tokens:
        return 0.0

    # Truncate to the minimum length -> handle length mismatches
    min_length = min(len(pred_tokens), len(true_tokens))
    if min_length == 0:
        return 0.0

    matches = sum(p == t for p, t in zip(pred_tokens[:min_length], true_tokens[:min_length]))
    return matches / len(true_tokens)


def calculate_bleu(pred_tokens: List[int], true_tokens: List[int]) -> float:
    """Calculate BLEU score between prediction and ground truth."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        print("Warning: NLTK not available. BLEU score will be 0.")
        return 0.0

    if not true_tokens:
        return 0.0

    # Convert token IDs to strings to handle potential tokenizer peculiarities
    pred_str = [str(t) for t in pred_tokens]
    true_str = [str(t) for t in true_tokens]

    # Use smoothing to handle cases when there are no n-gram overlaps
    smoothie = SmoothingFunction().method1

    try:
        return sentence_bleu([true_str], pred_str, smoothing_function=smoothie)
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return 0.0


def evaluate_model_comprehensive(
        model: nn.Module,
        test_loader,
        criterion: nn.Module,
        device: torch.device,
        model_type: str,
        task_mode: str,
        tokenizer,
        ignore_index: int = -100,
) -> Dict[str, float]:
    """
    Comprehensive evaluation function that computes multiple metrics.

    Args:
        model: The transformer model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device for computation
        model_type: Type of transformer model
        task_mode: Training task mode
        tokenizer: Tokenizer for decoding predictions
        ignore_index: Index to ignore in loss computation
        verbose: Whether to print detailed examples
        num_examples: Number of examples to print for inspection

    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()

    model_type = model_type.replace('-', '_')

    # Metrics tracking
    total_test_loss = 0
    total_samples = 0
    metrics = {
        "exact_match": 0,
        "token_accuracy": 0,
        "bleu": 0,
    }

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            model_inputs, labels_info = prepare_batch_for_model(
                batch, model, model_type, task_mode, device, ignore_index
            )

            # Forward pass
            outputs = model(**model_inputs)
            loss = compute_loss(outputs, labels_info["labels"], criterion)
            pred_tokens = torch.argmax(outputs, dim=-1).cpu()
            true_tokens = labels_info["labels"].cpu()

            for i in range(batch["input_ids"].size(0)):
                input_text, pred_filtered, true_filtered = _process_sample_by_model_type(
                    model_type, task_mode, batch, pred_tokens, true_tokens,
                    i, tokenizer, ignore_index
                )

                # Calculate metrics
                exact_match = int(_calculate_exact_match(pred_filtered, true_filtered, tokenizer))
                token_accuracy = calculate_token_accuracy(pred_filtered, true_filtered)
                bleu_score = calculate_bleu(pred_filtered, true_filtered)

                # Update metrics
                metrics["exact_match"] += exact_match
                metrics["token_accuracy"] += token_accuracy
                metrics["bleu"] += bleu_score
                total_samples += 1
            # Update total loss
            total_test_loss += loss.item() * batch["input_ids"].size(0)

    # Calculate average metrics
    avg_test_loss = total_test_loss / total_samples if total_samples > 0 else 0
    perplexity = math.exp(min(avg_test_loss, 100))
    avg_metrics = {k: v / total_samples if total_samples > 0 else 0 for k, v in metrics.items()}

    results = {
        "test_loss": avg_test_loss,
        "exact_match": avg_metrics["exact_match"],
        "token_accuracy": avg_metrics["token_accuracy"],
        "bleu_score": avg_metrics["bleu"],
        "perplexity": perplexity
    }
    return results


def _process_sample_by_model_type(
        model_type: str,
        task_mode: str,
        batch: Dict[str, torch.Tensor],
        pred_tokens: torch.Tensor,
        true_tokens: torch.Tensor,
        sample_idx: int,
        tokenizer,
        ignore_index: int
) -> Tuple[str, List[int], List[int]]:
    if model_type == "encoder_decoder":
        # Input text from source
        input_text = tokenizer.decode(batch["input_ids"][sample_idx].cpu().tolist())

        # True and predicted sequences
        pred_seq = pred_tokens[sample_idx].tolist()
        true_seq = true_tokens[sample_idx].tolist()

        # Filter out padding and ignore tokens
        pred_filtered = [t for t in pred_seq if t != tokenizer.pad_token_id and t != ignore_index]
        true_filtered = [t for t in true_seq if t != tokenizer.pad_token_id and t != ignore_index]

    elif model_type == "decoder_only":
        # Get input and label sequences
        input_seq = batch["input_ids"][sample_idx].cpu().tolist()
        label_seq = true_tokens[sample_idx].tolist()
        pred_seq = pred_tokens[sample_idx].tolist()

        # Find where the target sequence starts (first non-ignore token in labels)
        target_start = 0
        for j, token in enumerate(label_seq):
            if token != ignore_index:
                target_start = j
                break

        # Extract input part (for display)
        input_part = input_seq[:target_start]
        input_text = tokenizer.decode(input_part)

        # Extract prediction and ground truth from target portion
        pred_part = pred_seq[target_start:]
        true_part = [token for token in label_seq[target_start:] if token != ignore_index]

        # Filter out padding
        pred_filtered = [t for t in pred_part if t != tokenizer.pad_token_id]
        true_filtered = [t for t in true_part if t != tokenizer.pad_token_id]

    elif model_type == "encoder_only":
        input_text = tokenizer.decode(batch["input_ids"][sample_idx].cpu().tolist())

        # For MLM, we compare masked positions
        pred_seq = pred_tokens[sample_idx].tolist()
        true_seq = true_tokens[sample_idx].tolist()

        # Only compare tokens that were masked (not ignore_index)
        mask_positions = [j for j, token in enumerate(true_seq) if token != ignore_index]
        pred_filtered = [pred_seq[j] for j in mask_positions if j < len(pred_seq)]
        true_filtered = [true_seq[j] for j in mask_positions]

    else:
        # Fallback
        input_text = tokenizer.decode(batch["input_ids"][sample_idx].cpu().tolist())
        pred_filtered = []
        true_filtered = []

    return input_text, pred_filtered, true_filtered


def _calculate_exact_match(pred_filtered: List[int], true_filtered: List[int], tokenizer) -> bool:
    if not pred_filtered and not true_filtered:
        return True
    if not pred_filtered or not true_filtered:
        return False

    pred_text = tokenizer.decode(pred_filtered).strip()
    true_text = tokenizer.decode(true_filtered).strip()
    return pred_text == true_text
