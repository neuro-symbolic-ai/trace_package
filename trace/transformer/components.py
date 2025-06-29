import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from .utils import expand_mask

'''
Transformer model components including positional encoding, multi-head attention,
the code is largely inspired by the original transformer paper and PyTorch's implementation and 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html 
'''

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models using sinusoidal patterns.
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            d_model: Model dimension
            max_seq_length: Maximum sequence length to support
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # Register buffer (persistent state)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]

        Returns:
            Output tensor with added positional encoding
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor of shape [batch_size, query_len, d_model]
            key: Key tensor of shape [batch_size, key_len, d_model]
            value: Value tensor of shape [batch_size, value_len, d_model]
            mask: Optional attention mask of shape [batch_size, 1, 1, key_len]
                  or [batch_size, query_len, key_len]

        Returns:
            Output tensor and attention weights
        """
        batch_size = query.size(0)

        # Linear projections and reshape
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            mask = expand_mask(mask)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)

        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.output_linear(context)

        return output, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (typically 4 * d_model as the original paper suggests)
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]

        Returns:
            Output tensor of shape [batch_size, seq_length, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Encoder layer with self-attention and feed-forward network."""

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            no_fnn: bool = False
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        if not no_fnn:
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.feed_forward = None
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor and attention weights
        """
        # Self-attention with residual connection and normalization
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        if self.feed_forward is None:
            return x, attn_weights
        # Feed forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class DecoderLayer(nn.Module):
    """Decoder layer with self-attention, encoder-decoder attention, and feed-forward network."""

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            no_fnn: bool = False
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        if no_fnn:
            self.feed_forward = None

        else:
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            encoder_output: torch.Tensor,
            self_mask: Optional[torch.Tensor] = None,
            cross_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            encoder_output: Encoder output for cross-attention
            self_mask: Mask for self-attention
            cross_mask: Mask for cross-attention

        Returns:
            Output tensor, self-attention weights, and cross-attention weights
        """
        # Self attention with residual connection and normalization
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # Cross attention with residual connection and normalization
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output, cross_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # if self.feed_forward is None:
        #     return x, self_attn_weights, cross_attn_weights

        # Feed forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        self.last_residual = x # we use this for debugging purposes and other mechanics interpretabilty functions

        return x, self_attn_weights, cross_attn_weights



class DecoderOnlyLayer(nn.Module):
    """
    Simplified decoder layer for decoder-only models without cross-attention.
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            no_fnn: bool = False
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            no_fnn: Whether to disable feed-forward network
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        if no_fnn:
            self.feed_forward = None
        else:
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
            self.norm2 = nn.LayerNorm(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.no_ffn = no_fnn

    def forward(
            self,
            x: torch.Tensor,
            self_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            self_mask: Mask for self-attention

        Returns:
            Output tensor and self-attention weights
        """
        # Self attention with residual connection and normalization
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        if self.no_ffn:
            return x, self_attn_weights
        else:
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))

        self.last_residual = x
        return x, self_attn_weights


class Encoder(nn.Module):
    """
    Transformer encoder consisting of multiple encoder layers.
    """

    def __init__(
            self,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            no_fnn: bool = False
    ):
        """
        Args:
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            no_fnn: Whether to disable feed-forward networks
        """
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, no_fnn)
            for _ in range(num_layers)
        ])

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """

        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor and list of attention weights from each layer
        """
        attentions = [] #todo: maybe convert to a tensor

        for layer in self.layers:
            x, attention = layer(x, mask)
            attentions.append(attention)

        return x, attentions


class Decoder(nn.Module):
    """
    Transformer decoder supporting both decoder-only and encoder-decoder modes.

    """

    def __init__(
            self,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            decoder_only: bool = False,
            no_fnn: bool = False
    ):

        super().__init__()
        self.decoder_only = decoder_only

        # Create decoder layers based on whether this is decoder-only or encoder-decoder
        if decoder_only:
            self.layers = nn.ModuleList([
                DecoderOnlyLayer(d_model, num_heads, d_ff, dropout, no_fnn)
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                DecoderLayer(d_model, num_heads, d_ff, dropout, no_fnn)
                for _ in range(num_layers)
            ])

    def forward(
            self,
            x: torch.Tensor,
            encoder_output: Optional[torch.Tensor] = None,
            self_mask: Optional[torch.Tensor] = None,
            cross_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """
        Apply decoder transformation.

        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
            encoder_output: Output from encoder (for encoder-decoder models)
            self_mask: Mask for self-attention
            cross_mask: Mask for cross-attention

        Returns:
            Output tensor, list of self-attention weights, and list of cross-attention weights
        """
        self_attentions = [] #todo: maybe convert to a tensor
        cross_attentions = [] #todo: maybe convert to a tensor

        if self.decoder_only:
            # Decoder-only forward pass (no cross-attention)
            for layer in self.layers:
                x, self_attn = layer(x, self_mask)
                self_attentions.append(self_attn)
                cross_attentions.append(None)
        else:
            # Encoder-decoder forward pass (with cross-attention)
            if encoder_output is None:
                raise ValueError("encoder_output cannot be None for encoder-decoder model")

            for layer in self.layers:
                x, self_attn, cross_attn = layer(x, encoder_output, self_mask, cross_mask)
                self_attentions.append(self_attn)
                cross_attentions.append(cross_attn)

        return x, self_attentions, cross_attentions