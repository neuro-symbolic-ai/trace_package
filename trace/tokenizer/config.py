from dataclasses import dataclass

@dataclass
class TokenizerConfig:
    """
    Configuration class for tokenizer settings.

    This dataclass contains all hyperparameters and configuration options
    for tokenization processes in transformer models.
    """

    # Tokenizer type
    tokenizer_type: str = "whitespace"
    # Vocabulary size
    vocab_size: int = 30522
    # Special tokens
    unk_token: str = "[UNK]",
    pad_token: str = "[PAD]",
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    mask_token: str = "[MASK]",

    tokenizer_save_path: str = "tokenizer_config.json"

    def __post_init__(self):
        """Validate configuration parameters."""
        if not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")

        # Ensure special tokens are strings
        for token in [self.unk_token, self.pad_token, self.cls_token, self.sep_token, self.mask_token]:
            if not isinstance(token, str):
                raise ValueError(f"Special token {token} must be a string")

