import torch
import torch.nn as nn
from typing import List, Optional


# def freeze_ffn(model: nn.Module, freeze: bool = True) -> None:
#     """
#     Freeze or unfreeze feed-forward network parameters in a transformer model.
#
#     Args:
#         model: Transformer model to modify
#         freeze: Whether to freeze (True) or unfreeze (False) parameters
#     """
#     for name, param in model.named_parameters():
#         if 'feed_forward' in name or 'ffn' in name:
#             param.requires_grad = not freeze


# def unfreeze_ffn_layer(model: nn.Module, layer_idx: int) -> None:
#     """
#     Unfreeze feed-forward network parameters for a specific layer.
#
#     Args:
#         model: Transformer model to modify
#         layer_idx: Index of the layer to unfreeze
#     """
#     for name, param in model.named_parameters():
#         if f'layers.{layer_idx}' in name and 'feed_forward' in name:
#             param.requires_grad = True


def attach_hooks(model: nn.Module, names: Optional[List[str]] = None) -> nn.Module:
    """
    Attach gradient monitoring hooks to model parameters.

    Args:
        model: Transformer model to attach hooks to
        names: Optional list of parameter names to hook. If None, hooks all non-embedding parameters

    Returns:
        Model with attached hooks
    """

    def create_hook(param_name: str):
        """Create a hook function with parameter name captured in closure."""
        return lambda grad: print(f"Gradient for {param_name}: {grad.norm()}")

    if names is None:
        names = [name for name, _ in model.named_parameters() if 'embedding' not in name]

    for name, param in model.named_parameters():
        if name in names:
            hook = create_hook(name)
            param.register_hook(hook)

    return model


def remove_hooks(model: nn.Module, names: Optional[List[str]] = None) -> nn.Module:
    """
    Remove gradient hooks from model parameters.

    Args:
        model: Transformer model to remove hooks from
        names: Optional list of parameter names (currently unused)

    Returns:
        Model with hooks removed
    """
    for param in model.parameters():
        param.register_hook(None)
    return model


def expand_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Expand attention mask to support different input dimensions.

    Supports broadcasting mask shapes for multi-head attention:
    - 2D masks are broadcasted over batch size and number of heads
    - 3D masks are broadcasted over number of heads
    - 4D masks are left as-is

    Args:
        mask: Input mask tensor of shape [seq_len, seq_len], [batch, seq_len, seq_len],
              or [batch, heads, seq_len, seq_len]

    Returns:
        Expanded mask tensor of shape [batch, heads, seq_len, seq_len]
    """
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)

    return mask
