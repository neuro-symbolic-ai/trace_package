import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List, Mapping, Any
from .config import TransformerConfig
from .components import PositionalEncoding, Encoder, Decoder


class Transformer(nn.Module):
    """
    Flexible transformer model supporting encoder-only, decoder-only, and encoder-decoder configurations.
    """

    def __init__(self, config: TransformerConfig):
        """
        Initialize transformer model from configuration.

        Args:
            config: TransformerConfig object containing all model parameters
        """
        super().__init__()

        # Store configuration
        self.config = config
        self.model_type = config.model_type
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.pad_idx = config.pad_idx
        self.num_heads = config.num_heads
        self.d_ff = config.d_ff if config.d_ff is not None else config.d_model * 4

        # Input embedding and positional encoding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_seq_length, config.dropout)

        # Initialize encoder and/or decoder based on model_type
        if config.model_type in ['encoder_only', 'encoder_decoder']:
            assert config.num_encoder_layers > 0, "Encoder layers must be > 0 for encoder models"
            self.encoder = Encoder(
                config.d_model,
                config.num_encoder_layers,
                config.num_heads,
                config.d_ff,
                config.dropout,
                config.no_fnn
            )
        else:
            self.encoder = None

        if config.model_type in ['decoder_only', 'encoder_decoder']:
            assert config.num_decoder_layers > 0, "Decoder layers must be > 0 for decoder models"
            is_decoder_only = (config.model_type == 'decoder_only')
            self.decoder = Decoder(
                config.d_model,
                config.num_decoder_layers,
                config.num_heads,
                config.d_ff,
                config.dropout,
                decoder_only=is_decoder_only,
                no_fnn=config.no_fnn
            )
        else:
            self.decoder = None

        # Output projection for decoder-based models
        if config.model_type in ['decoder_only', 'encoder_decoder']:
            self.output_projection = nn.Linear(config.d_model, config.vocab_size)

    @classmethod
    def from_config(cls, config: TransformerConfig) -> 'Transformer':
        """
        Creating transformer model from configuration.

        Args:
            config: TransformerConfig object

        Returns:
            Initialized Transformer model
        """
        return cls(config) # return cls(**config.__dict__) #return cls(**config.__dict__)

    def generate_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate padding mask for attention.
        # Source: https://medium.com/@swarms/understanding-masking-in-pytorch-for-attention-mechanisms-e725059fd49f

        Args:
            x: Input tensor with padding tokens

        Returns:
            Padding mask tensor
        """
        return (x == self.pad_idx).unsqueeze(1).unsqueeze(2)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask for decoder self-attention.
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(
            self,
            src: Optional[torch.Tensor] = None,
            tgt: Optional[torch.Tensor] = None,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the transformer.

        Args:
            src: Source sequence tensor [batch_size, src_seq_len]
            tgt: Target sequence tensor [batch_size, tgt_seq_len]
            src_mask: Source sequence padding mask
            tgt_mask: Target sequence causal mask
            memory_mask: Cross-attention mask

        Returns:
            Output tensor(s) depending on model type
        """
        if self.model_type == 'encoder_only':
            return self._forward_encoder_only(src, src_mask)
        elif self.model_type == 'decoder_only':
            return self._forward_decoder_only(tgt, tgt_mask)
        elif self.model_type == 'encoder_decoder':
            return self._forward_encoder_decoder(src, tgt, src_mask, tgt_mask, memory_mask)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _forward_encoder_only(
            self,
            src: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for encoder-only models."""
        assert src is not None, "Source sequence required for encoder-only model"

        # Create padding mask if not provided
        if src_mask is None:
            src_mask = self.generate_padding_mask(src)

        # Embed source sequence
        src_embedded = self.embedding(src)
        src_embedded = self.positional_encoding(src_embedded)

        # Encode
        encoder_output, encoder_attentions = self.encoder(src_embedded, src_mask)

        return encoder_output

    def _forward_decoder_only(
            self,
            tgt: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for decoder-only models."""
        assert tgt is not None, "Target sequence required for decoder-only model"

        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)

        # Embed target sequence
        tgt_embedded = self.embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        # Decode (without encoder output)
        decoder_output, self_attentions, _ = self.decoder(tgt_embedded, None, tgt_mask, None)

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    def _forward_encoder_decoder(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for encoder-decoder models."""
        assert src is not None, "Source sequence required for encoder-decoder model"
        assert tgt is not None, "Target sequence required for encoder-decoder model"

        # Create masks if not provided
        if src_mask is None:
            src_mask = self.generate_padding_mask(src)
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        if memory_mask is None:
            memory_mask = self.generate_padding_mask(src)

        # Embed source and target sequences
        src_embedded = self.embedding(src)
        src_embedded = self.positional_encoding(src_embedded)

        tgt_embedded = self.embedding(tgt)
        tgt_embedded = self.positional_encoding(tgt_embedded)

        # Encode
        encoder_output, encoder_attentions = self.encoder(src_embedded, src_mask)

        # Decode
        decoder_output, self_attentions, cross_attentions = self.decoder(
            tgt_embedded, encoder_output, tgt_mask, memory_mask
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)

        return logits

    def get_grad(self, include_ffn: bool = True, layers: Optional[List[str]] = None) -> dict:
        """
        Extract gradients from model parameters.

        Args:
            include_ffn: Whether to include feed-forward network gradients
            layers: Optional list of layer names to filter gradients

        Returns:
            Dictionary mapping parameter names to flattened gradients
        """
        grad = {}
        for name, param in self.named_parameters():
            if not include_ffn and ('feed_forward' in name or 'norm2' in name):
                continue
            if layers is not None and not any(layer in name for layer in layers):
                continue
            elif param.grad is not None:
                grad[name] = param.grad.clone().detach().flatten()
        return grad

    def initialize_weights(self):
        """
        Initialize model weights using Xavier uniform initialization.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def freeze_layer(self, layer_idx: int, freeze: bool = True,
                     freezing_components: Optional[List[str]] = None) -> None:
        """
        Freeze or unfreeze parameters of a specific layer in the transformer model.

        Args:
            layer_idx: Index of the layer to freeze/unfreeze
            freeze: Whether to freeze (True) or unfreeze (False) parameters
        """
        if freezing_components:
            components = [(name, param) for name, param in self.named_parameters() if
                          any(comp in name for comp in freezing_components)]
        else:
            components = self.named_parameters()
        for name, param in components:
            if f'layers.{layer_idx}' in name:
                param.requires_grad = not freeze

    def freeze_ffn(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze (all) feed-forward network parameters in the transformer model.
        """
        for name, param in self.named_parameters():
            if 'feed_forward' in name or 'ffn' in name:
                param.requires_grad = not freeze

    def attach_hooks(self, names: Optional[List[str]] = None) -> None:
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
            names = [name for name, _ in self.named_parameters() if 'embedding' not in name]

        for name, param in self.named_parameters():
            if name in names:
                hook = create_hook(name)
                param.register_hook(hook)

    def remove_hooks(self, names: Optional[List[str]] = None) -> None:
        """
        Remove gradient hooks from model parameters.

        Args:
            names: Optional list of parameter names to remove hooks from. If None, removes all hooks
        """
        if names is None:
            names = [name for name, _ in self.named_parameters() if 'embedding' not in name]

        for name, param in self.named_parameters():
            if name in names:
                param.register_hook(None)

    def setup_tracking(self,
                       track_layers: Optional[List[str]] = None,
                       track_last_token: bool = True,
                       track_interval: int = 0,
                       max_history: int = -1,
                       ema_alpha: float = 0.9,
                       track_save_dir: Optional[str] = None):
        """
        Setup tracking for model states during training.

        Args:
            track_layers: List of layer names to track
            track_last_token: Whether to track last token
            track_interval: Interval for tracking
            max_history: Maximum history to keep
            ema_alpha: EMA alpha for tracking
            track_save_dir: Directory to save tracking data
        """
        # Placeholder for tracking functionality
        pass

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in a model.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_model_size(self) -> str:
        """
        Get an approximate describing model size - HuggingFace style.
        """
        param_count = self.count_parameters(trainable_only=False)

        if param_count >= 1e9:
            return f"{param_count / 1e9:.1f}B parameters"
        elif param_count >= 1e6:
            return f"{param_count / 1e6:.1f}M parameters"
        elif param_count >= 1e3:
            return f"{param_count / 1e3:.1f}K parameters"
        else:
            return f"{param_count} parameters"

    # def load_state_dict(
    #     self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    # ):
    #     """
    #     Load state dictionary into the model.
    #
    #     Args:
    #         state_dict: State dictionary to load
    #         strict: Whether to enforce strict loading
    #         assign: Whether to assign state_dict directly
    #
    #     Returns:
    #         None
    #     """
    #     if assign:
    #         self.__dict__.update(state_dict) # add check point and use only state_dict
    #     else:
    #         super().load_state_dict(state_dict, strict=strict)



