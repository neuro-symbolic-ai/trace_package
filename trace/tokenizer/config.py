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
    max_vocab_size: int = 0
    # Special tokens
    unk_token: str = "[UNK]",
    pad_token: str = "[PAD]",
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    mask_token: str = "[MASK]",

    tokenizer_save_path: str = "./tokenizers"

    def __post_init__(self):
        """Validate configuration parameters."""
        # Ensure special tokens are strings
        for token in [self.unk_token, self.pad_token, self.cls_token, self.sep_token, self.mask_token]:
            if not isinstance(token, str):
                raise ValueError(f"Special token {token} must be a string")


    @classmethod
    def default(cls) -> 'TokenizerConfig':
        """Create a default configuration for the tokenizer."""
        return cls(
            tokenizer_type="whitespace",
            max_vocab_size=0,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            tokenizer_save_path="./tokenizers/tokenizer_config.json"
        )