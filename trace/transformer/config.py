from dataclasses import dataclass
from typing import Optional

import torch

'''
Inspired by https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py
and https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/HookedTransformerConfig.py 
but simplified for our use case. 
'''
@dataclass
class TransformerConfig:
    """
    Configuration class for transformer models.

    This dataclass contains all hyperparameters and configuration options
    for creating transformer models, following the reference style pattern.
    """

    # Model architecture
    model_type: str = "decoder_only"  # 'encoder_only', 'decoder_only', 'encoder_decoder'
    vocab_size: int = 32000
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 0
    num_decoder_layers: int = 6
    d_ff: int = 2048

    # Training parameters
    max_seq_length: int = 5000
    dropout: float = 0.1
    pad_idx: int = 0

    # Special options
    no_fnn: bool = False  # Disable feed-forward networks
    freeze_ffn: bool = False  # Freeze feed-forward parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.model_type not in ['encoder_only', 'decoder_only', 'encoder_decoder']:
            raise ValueError(f"Invalid model_type: {self.model_type}")

        if self.model_type in ['encoder_only', 'encoder_decoder'] and self.num_encoder_layers <= 0:
            raise ValueError("Encoder layers must be > 0 for encoder models")

        if self.model_type in ['decoder_only', 'encoder_decoder'] and self.num_decoder_layers <= 0:
            raise ValueError("Decoder layers must be > 0 for decoder models")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.d_ff is None:
            self.d_ff = self.d_model * 4

    @classmethod
    def encoder_only(cls, **kwargs) -> 'TransformerConfig':
        """Create configuration for encoder-only model."""
        defaults = {
            'model_type': 'encoder_only',
            'num_encoder_layers': 6,
            'num_decoder_layers': 0
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def decoder_only(cls, **kwargs) -> 'TransformerConfig':
        """Create configuration for decoder-only model."""
        defaults = {
            'model_type': 'decoder_only',
            'num_encoder_layers': 0,
            'num_decoder_layers': 6
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def encoder_decoder(cls, **kwargs) -> 'TransformerConfig':
        """Create configuration for encoder-decoder model."""
        defaults = {
            'model_type': 'encoder_decoder',
            'num_encoder_layers': 6,
            'num_decoder_layers': 6
        }
        defaults.update(kwargs)
        return cls(**defaults)