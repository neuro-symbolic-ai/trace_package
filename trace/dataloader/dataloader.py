import json
import os

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from absynth.corpus import SyntheticCorpusGenerator


class TextDataset(Dataset):
    """
    Custom dataset for tokenized text and semantic labels using LogicalFormTokenizer.
    Designed for translating natural language sentences to formal semantics.
    """

    def __init__(self, corpus_path, tokenizer, max_length=128, model_type="encoder-decoder", task_mode="original",
                 mlm_prob=0.25, mask_prob=0.8, random_prob=0.0, keep_prob=0.1,):
        """
        Initialize dataset.

        Args:
            corpus_path: Path to the JSON file containing sentence-semantics pairs
            tokenizer: LogicalFormTokenizer instance
            max_length: Maximum sequence length for padding/truncation
            model_type: Type of model architecture ('encoder-decoder', 'decoder-only', or 'encoder-only')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        self.task_mode = task_mode
        self.mlm_prob = mlm_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        self.corpus_path = corpus_path

        # Load corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
            if not isinstance(self.corpus, list):
                if isinstance(self.corpus, dict):
                    # If it's a dictionary, assume it has a 'sentences' key
                    self.corpus = self.corpus.get('corpus')
                else:
                    raise ValueError(f"Invalid corpus format in {corpus_path}. Expected a list or dict with 'corpus' key.")

        # Get special token IDs
        self.pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 1
        self.cls_token_id = tokenizer.cls_token_id if hasattr(tokenizer, 'cls_token_id') else 2
        self.sep_token_id = tokenizer.sep_token_id if hasattr(tokenizer, 'sep_token_id') else 3
        self.masking_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else 4
        self.vocab_size = len(tokenizer.get_vocab())

        # For handling loss masking
        self.ignore_index = -100

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        """
        Tokenizes both sentence (input) and semantics (label).
        Handles different model types differently:
        - encoder-decoder: separate inputs and labels
        - decoder-only: concatenated sequence with masks for loss computation
        """
        data_item = self.corpus[idx]
        sentence = data_item['sentence']
        semantics = data_item['semantics']

        if self.task_mode == "mlm":
            input_ids = self.tokenizer.encode(
                sentence,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True
            )
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = input_ids.clone()

            # Identify special tokens
            special_token_ids = {self.pad_token_id, self.cls_token_id, self.sep_token_id}
            special_mask = torch.tensor([token in special_token_ids for token in input_ids])
            probability_matrix = torch.full(input_ids.shape, self.mlm_prob)
            probability_matrix.masked_fill_(special_mask, value=0.0)

            # Select masked indices
            masked_indices = torch.bernoulli(probability_matrix).bool()

            # Ensure at least one token is masked
            if masked_indices.sum() == 0:
                non_special_indices = (~special_mask).nonzero(as_tuple=False).squeeze()
                if non_special_indices.numel() > 0:
                    random_idx = non_special_indices[torch.randint(0, len(non_special_indices), (1,))]
                    masked_indices[random_idx] = True

            # Create labels
            labels[~masked_indices] = self.ignore_index

            # Split into 80/10/10
            indices_replaced = torch.bernoulli(torch.full(input_ids.shape, self.mask_prob)).bool() & masked_indices
            indices_random = (
                    torch.bernoulli(
                        torch.full(input_ids.shape, self.random_prob / (self.random_prob + self.keep_prob))).bool()
                    & masked_indices & ~indices_replaced
            )

            # 80% -> [MASK]
            input_ids[indices_replaced] = self.mask_token_id

            # 10% -> random token
            if indices_random.any():
                random_tokens = torch.randint(low=0, high=len(self.tokenizer.get_vocab()), size=input_ids.shape)
                input_ids[indices_random] = random_tokens[indices_random]

            # 10% -> keep original (no change needed)

            # Pad
            padding_len = self.max_length - len(input_ids)
            if padding_len > 0:
                input_ids = torch.cat([input_ids, torch.tensor([self.pad_token_id] * padding_len)])
                labels = torch.cat([labels, torch.tensor([self.ignore_index] * padding_len)])
            else:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

            return {
                "input_ids": input_ids,
                "attention_mask": (input_ids != self.pad_token_id).long(),
                "labels": labels
            }


        elif self.task_mode == "next_token":
            # Predict next token in sequence
            tokens = self.tokenizer.encode(sentence, max_length=self.max_length + 1, truncation=True, padding=False)
            input_ids = tokens[:-1]
            labels = tokens[1:]

            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

            padding_len = self.max_length - len(input_ids)
            input_ids += [self.pad_token_id] * padding_len
            labels += [self.ignore_index] * padding_len

            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor([1 if id != self.pad_token_id else 0 for id in input_ids]),
                "labels": torch.tensor(labels)
            }

        # Ensure proper Unicode decoding for logical symbols
        if hasattr(self.tokenizer, '_decode_unicode_escapes'):
            semantics = self.tokenizer._decode_unicode_escapes(semantics)

        if self.model_type == "encoder_decoder":
            # Tokenize input sentence
            sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=True)
            sentence_ids = sentence_ids[:self.max_length]

            # Create attention mask (1 for tokens, 0 for padding)
            sentence_attention_mask = [1] * len(sentence_ids)

            # Tokenize output semantics for teacher forcing
            semantics_ids = self.tokenizer.encode(semantics, add_special_tokens=True)
            semantics_ids = semantics_ids[:self.max_length]

            # For decoder input, we shift right (prep with cls_token_id and remove last token)
            decoder_input_ids = [self.cls_token_id] + semantics_ids[:-1]
            decoder_input_ids = decoder_input_ids[:self.max_length]
            decoder_attention_mask = [1] * len(decoder_input_ids)

            # For labels, we use the actual semantics (shifted left for prediction)
            labels = semantics_ids

            # Pad sequences
            input_ids = self._pad_sequence(sentence_ids, self.max_length)
            attention_mask = self._pad_sequence(sentence_attention_mask, self.max_length)
            decoder_input_ids = self._pad_sequence(decoder_input_ids, self.max_length)
            decoder_attention_mask = self._pad_sequence(decoder_attention_mask, self.max_length)
            labels = self._pad_sequence(labels, self.max_length, pad_value=self.ignore_index)

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
                "decoder_attention_mask": torch.tensor(decoder_attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }

        elif self.model_type == "decoder_only":
            # For decoder-only models, we concatenate input and output with separator
            # First encode both parts separately
            sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
            semantics_ids = self.tokenizer.encode(semantics, add_special_tokens=False)

            # Build complete sequence: [CLS] sentence [SEP] semantics [SEP]
            combined_ids = [self.cls_token_id] + sentence_ids + [self.sep_token_id] + semantics_ids + [
                self.sep_token_id]
            combined_ids = combined_ids[:self.max_length]

            # Find separator position for loss masking
            if self.sep_token_id in combined_ids:
                sep_pos = combined_ids.index(self.sep_token_id)
            else:
                # If separator got truncated, estimate position
                sep_pos = 1 + len(sentence_ids)  # Account for CLS token
                if sep_pos >= len(combined_ids):
                    sep_pos = len(combined_ids) - 1

            # Create attention mask (1 for tokens, 0 for padding)
            attention_mask = [1] * len(combined_ids)

            # Create labels: same as input but mask out the input portion for loss
            labels = combined_ids.copy()
            for i in range(sep_pos + 1):  # Mask out everything up to and including first SEP
                labels[i] = self.ignore_index

            # Pad sequences
            input_ids = self._pad_sequence(combined_ids, self.max_length)
            attention_mask = self._pad_sequence(attention_mask, self.max_length)
            labels = self._pad_sequence(labels, self.max_length, pad_value=self.ignore_index)

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }

        else:  # encoder-only
            # Simple encoder-only format (typically for classification or feature extraction)
            sentence_ids = self.tokenizer.encode(sentence, add_special_tokens=True)
            sentence_ids = sentence_ids[:self.max_length]

            # Create attention mask
            attention_mask = [1] * len(sentence_ids)

            # Pad sequences
            input_ids = self._pad_sequence(sentence_ids, self.max_length)
            attention_mask = self._pad_sequence(attention_mask, self.max_length)

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "text": sentence
            }

    def _pad_sequence(self, sequence, max_length, pad_value=None):
        """Helper method to pad sequences to desired length."""
        if pad_value is None:
            pad_value = self.pad_token_id

        padded = sequence + [pad_value] * (max_length - len(sequence))
        return padded[:max_length]  # Just in case the sequence was longer


def get_dataloader(corpus_path, tokenizer, batch_size: int = 32, max_length: int = 128,
                    model_type: str = "encoder-decoder", val_split: float = 0.1, num_sentences: int = 50000)-> tuple:
    """
    Create DataLoaders for training and validation.

    Args:
        corpus_path (str): Path to dataset JSON file.
        tokenizer: LogicalFormTokenizer instance
        batch_size (int): Batch size for training.
        max_length (int): Maximum sequence length.
        model_type (str): Model architecture type ('encoder-decoder', 'decoder-only', or 'encoder-only').
        val_split (float): Fraction of data to use for validation.

    Returns:
        tuple: (train_loader, val_loader)
    """
    if corpus_path is None or not corpus_path:
        # create dataset from absynth corpus
        if num_sentences <= 0:
            raise ValueError("num_sentences must be a positive integer")
            num_sentences = 50000

        generator = SyntheticCorpusGenerator()
        corpus = generator(10000)
        corpus_path = './data'
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
        # Save the corpus to a JSON file
        file_name = f"synthetic_corpus_{num_sentences}.json"
        corpus_path = os.path.join(corpus_path, file_name)
        corpus.save(corpus_path, format="sentences_only", indent=2)

    dataset = TextDataset(corpus_path, tokenizer, max_length=max_length, model_type=model_type)

    # Split into train and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
