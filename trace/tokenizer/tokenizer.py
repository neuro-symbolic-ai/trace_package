from .config import TokenizerConfig
from transformers import PreTrainedTokenizer
import json
import codecs
import os


class LogicalFormTokenizer(PreTrainedTokenizer):
    def __init__(self, config: TokenizerConfig = None, **kwargs):
        self.config = config if config else TokenizerConfig.default()
        self.vocab = {
            config.unk_token: 1,
            config.pad_token: 0,
            config.cls_token: 2,
            config.sep_token: 3,
            config.mask_token: 4
        }
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        super().__init__(
            unk_token=config.unk_token,
            pad_token=config.pad_token,
            cls_token=config.cls_token,
            sep_token=config.sep_token,
            mask_token=config.mask_token,
            **kwargs
        )

        if os.path.exists(config.tokenizer_save_path):
            self.load_vocab_from_file(config.tokenizer_save_path)

        elif 'vocab_file' in kwargs:
            self.load_vocab_from_file(kwargs['vocab_file'])


    def load_vocab_from_file(self, vocab_file, save_if_not_exists=True):
        """Load vocabulary from a JSON file containing example sentences and semantics."""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            data = json.load(f)['corpus']

        all_text = ""
        for item in data:
            if item.get("sentence"):
                sentence = self._decode_unicode_escapes(item.get("sentence", ""))
            else:
                sentence = ""
            if item.get("semantics"):
                semantics = self._decode_unicode_escapes(item.get("semantics", ""))
            else:
                semantics = ""
            if sentence or semantics:
                # Combine sentence and semantics, ensuring they are not empty
                all_text += sentence + " " + semantics + " "

        tokens = self._tokenize(all_text)
        for token in tokens:
            if token not in self.vocab:
                index = len(self.vocab)
                self.vocab[token] = index
                self.ids_to_tokens[index] = token

        if save_if_not_exists:
            if not os.path.exists(self.config.tokenizer_save_path):
                self.save_vocabulary(os.path.dirname(self.config.tokenizer_save_path), filename_prefix="tokenizer_absynth")

    @staticmethod
    def _decode_unicode_escapes(text):
        """Properly handle Unicode escape sequences in the text."""
        if "\\u" in text:
            return codecs.decode(text.encode('utf-8'), 'unicode_escape')
        return text

    def _tokenize(self, text):
        """Tokenize the text into individual tokens."""
        # Make sure we've handled any Unicode escape sequences
        text = self._decode_unicode_escapes(text)

        # Special tokens for logical operators (ensure they remain separate)
        special_tokens = {"∃", "∧"}

        tokens = []
        for word in text.split():
            # Check if the word contains any special tokens
            contains_special = False
            for special in special_tokens:
                if special in word:
                    contains_special = True
                    parts = word.split(special)

                    # Process each part and add special token between them
                    for i, part in enumerate(parts):
                        if part:  # Only add non-empty parts
                            if i > 0:  # Add the special token before this part
                                tokens.append(special)

                            # Process this part for parentheses
                            current = ""
                            for char in part:
                                if char in "()":
                                    if current:
                                        tokens.append(current)
                                        current = ""
                                    tokens.append(char)
                                else:
                                    current += char
                            if current:
                                tokens.append(current)
                        elif i > 0:  # Handle case where split results in empty parts
                            tokens.append(special)
                    break

            # If no special tokens, process for parentheses
            if not contains_special:
                current = ""
                for char in word:
                    if char in "()":
                        if current:
                            tokens.append(current)
                            current = ""
                        tokens.append(char)
                    else:
                        current += char
                if current:
                    tokens.append(current)

        return tokens

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            # return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
            return token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def get_vocab(self):
        return dict(self.vocab)

    def save_vocabulary(self, save_directory, filename_prefix=None):

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.vocab, ensure_ascii=False))

        return (vocab_file,)

    def save_pretrained(self, save_directory, legacy_format=None, filename_prefix=None, push_to_hub=False, **kwargs):
        """
        Save the tokenizer vocabulary to a directory.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "tokenizer.json"
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.vocab, ensure_ascii=False))

        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        """
        Instantiate a tokenizer from a pre-trained model vocabulary.
        """
        vocab_files_names = {"vocab_file": "vocab.json"}
        # Look for the vocab file
        vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")

        # Create tokenizer
        tokenizer = cls(vocab_file=vocab_file, **kwargs)
        return tokenizer

    # Rename to encode and decode
    def encode(self, text, add_special_tokens=True, **kwargs):
        """Tokenize and convert to token ids."""
        tokens = self._tokenize(text)
        ids = [self._convert_token_to_id(token) for token in tokens]

        if add_special_tokens:
            return self.build_inputs_with_special_tokens(ids)
        return ids

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """Convert token ids back to string."""
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]

        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.all_special_tokens]

        return self.convert_tokens_to_string(tokens)

    def get_vocab_size(self):
        """Get the size of the vocabulary."""
        return len(self.vocab)


def create_tokenizer_from_data(vocab_file, config: TokenizerConfig = None):
    """Create and initialize a tokenizer from a JSON data file."""
    if config is None:
        config = TokenizerConfig.default()
    tokenizer = LogicalFormTokenizer(vocab_file=vocab_file, config=config)
    return tokenizer