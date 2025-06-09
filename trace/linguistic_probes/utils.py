import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Callable, Union
from collections import Counter

from .taggers import POSTagger, SemanticTagger
from .config import LinguisticProbesConfig


def extract_hidden_representations_with_pos_semantic(
        model, # did not specify type here to avoid circular import issues and to keep it flexible
        dataloader: DataLoader,
        device: str,
        layer_indices: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        tokenizer=None,
        config: Optional[LinguisticProbesConfig] = None,
) -> tuple:
    """
    Extract hidden representations from specified transformer layers.

    Args:
        model: The transformer model
        dataloader: DataLoader with input data
        device: Device for computation
        layer_indices: Which layers to extract from and decoder or encoder
        tokenizer: Tokenizer for decoding sequences
        config: Configuration object

    Returns:
        Tuple of (hidden_states_dict, pos_labels, semantic_labels)
    """
    model.eval()
    print('Initial layer indices:', layer_indices)
    # Set default layer indices
    if layer_indices is None:
        if model.model_type == 'encoder_only' and hasattr(model.encoder, 'layers'):
            layer_indices = {'encoder':  list(range(len(model.encoder.layers)))}  # Default to first layer
        elif model.model_type == 'decoder_only' and hasattr(model.decoder, 'layers'):
            layer_indices = {'decoder': list(range(len(model.decoder.layers)))}  # Default to first layer
        elif model.model_type == 'encoder_decoder':
            encoder_layers = list(range(len(model.encoder.layers))) if hasattr(model.encoder, 'layers') else [0]
            decoder_layers = list(range(len(model.decoder.layers))) if hasattr(model.decoder, 'layers') else [0]
            layer_indices = {'encoder': encoder_layers, 'decoder': decoder_layers}

    elif isinstance(layer_indices, list):
        if model.model_type == 'encoder_only':
            layer_indices = {'encoder': layer_indices}
        elif model.model_type == 'decoder_only':
            layer_indices = {'decoder': layer_indices}
        elif model.model_type == 'encoder_decoder':
            # For compatibility, apply to decoder only since no key is given
            layer_indices = {'decoder': layer_indices}
    elif isinstance(layer_indices, dict):
        layer_indices = layer_indices

    print(f"Extracting hidden states from updated layers: {layer_indices}")

    # Initialize hidden states dictionary
    hidden_dict = {}
    for layer_type, indices in layer_indices.items():
        for idx in indices:
            hidden_dict[(idx, layer_type)] = []

        # layer_indices = [0]
        # if model.model_type == 'encoder_only' and hasattr(model.encoder, 'layers'):
        #     # layer_indices = list(range(len(model.encoder.layers)))
        #     hidden_dict = {i: [] for i in layer_indices}
        #     hidden_dict = {(i, 'encoder'): [] for i in layer_indices}
        # elif model.model_type == 'decoder_only' and hasattr(model.decoder, 'layers'):
        #     layer_indices = list(range(len(model.decoder.layers)))
        #     # hidden_dict = {i: [] for i in layer_indices}
        #     hidden_dict = {(i, 'decoder'): [] for i in layer_indices}
        # elif model.model_type == 'encoder_decoder':
        #     layer_indices_decoder = list(range(len(model.decoder.layers)))
        #     layer_indices_encoder = list(range(len(model.encoder.layers))) # todo fix this for encoder-decoder models
        #     layer_indices = {'decoder':layer_indices_decoder, 'encoder':layer_indices_encoder} # todo: a better way to handle this is to cover layer indices to dict
        #     hidden_dict = {(i, 'decoder'): [] for i in layer_indices_decoder}
        #     hidden_dict.update({(i, 'encoder'): [] for i in layer_indices_encoder})


    pos_labels = []
    semantic_labels = []

    # Set up hooks to capture hidden states
    def save_hidden_state(index, layer_type=None):

        def hook(module, input, output):
            # hidden = output[0] if isinstance(output, tuple) else output
            # hidden_dict[key].append(hidden.detach().cpu())
            if layer_type is not None:
                # For encoder-decoder models, use tuple keys
                key = (index, layer_type)
            else:
                key = index[1] if isinstance(index, tuple) else index
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_dict[key].append(hidden.detach().cpu())
        return hook

    handles = []

    # Register hooks based on model architecture
    if model.model_type != 'encoder_decoder':
        for layer_index in layer_indices['encoder'] if model.model_type == 'encoder_only' else layer_indices['decoder']:
            print(f"Registering hook for layer {layer_index}")
            try:
                if  model.model_type == 'encoder_only' and hasattr(model.encoder, 'layers'):
                    handle = model.encoder.layers[layer_index].register_forward_hook(save_hidden_state(layer_index, 'encoder'))
                elif model.model_type == 'decoder_only' and hasattr(model.decoder, 'layers'):
                    handle = model.decoder.layers[layer_index].register_forward_hook(save_hidden_state(layer_index, 'decoder'))
                handles.append(handle)
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not register hook for layer {layer_index}: {e}")
    else:
        # For encoder-decoder models, register hooks for both encoder and decoder layers
        for layer_index in layer_indices['encoder']:
            try:
                handle = model.encoder.layers[layer_index(1)].register_forward_hook(save_hidden_state(layer_index, 'encoder'))
                handles.append(handle)
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not register hook for encoder layer {layer_index}: {e}")

        for layer_index in layer_indices['decoder']:
            try:
                handle = model.decoder.layers[layer_index(1)].register_forward_hook(save_hidden_state(layer_index, 'decoder'))
                handles.append(handle)
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not register hook for decoder layer {layer_index}: {e}")


    # Extract representations
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Hidden States"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get('attention_mask', None)
            # if attention_mask is not None:
            #     attention_mask = attention_mask.to(device)

            # Create labels if tokenizer is provided
            if tokenizer is not None:
                if config is None or config.track_pos:
                    pos_labels.append(create_one_hot_pos_labels(input_ids, tokenizer, config))
                if config is None or config.track_semantic:
                    semantic_labels.append(create_one_hot_semantic_labels(input_ids, tokenizer, config))

            if model.model_type == 'encoder_only':
                # Forward pass for encoder
                _ = model(src=input_ids)
            elif model.model_type == 'decoder_only':
                # Forward pass for decoder
                _ = model(tgt=input_ids)
            elif model.model_type == 'encoder_decoder':
                # Forward pass for encoder
                _ = model(src=input_ids, tgt=input_ids)  # Assuming src and tgt are the same for simplicity
            # # Forward pass to trigger hooks
            # if attention_mask is not None:
            #     # _ = model(input_ids, attention_mask=attention_mask)
            #     _ = model(tgt=input_ids) # todo fix this for encoder-decoder models and encoder-only models
            # else:
            #     _ = model(tgt=input_ids)

    # Clean up hooks
    for handle in handles:
        handle.remove()

    # Concatenate results
    # hidden_states = {i: torch.cat(hidden_dict[i], dim=0) for i in layer_indices if hidden_dict[i]}
    hidden_states = {}
    if model.model_type != 'encoder_decoder':
        for layer_index in layer_indices['encoder'] if model.model_type == 'encoder_only' else layer_indices['decoder']:
            key = (layer_index, 'encoder' if model.model_type == 'encoder_only' else 'decoder')
            # key = layer_index if isinstance(layer_index, int) else (layer_index, 'encoder' if model.model_type == 'encoder_only' else 'decoder')
            if key in hidden_dict:
                hidden_states[key] = torch.cat(hidden_dict[key], dim=0)
    else:
        # For encoder-decoder models, use tuple keys
        for (layer_index, layer_type) in layer_indices['encoder']:
            if hidden_dict[(layer_index, 'encoder')]:
                hidden_states[(layer_index, 'encoder')] = torch.cat(hidden_dict[(layer_index, 'encoder')], dim=0)
        for (layer_index, layer_type) in layer_indices['decoder']:
            if hidden_dict[(layer_index, 'decoder')]:
                hidden_states[(layer_index, 'decoder')] = torch.cat(hidden_dict[(layer_index, 'decoder')], dim=0)
    # Concatenate labels if available
    final_pos_labels = torch.cat(pos_labels, dim=0) if pos_labels else None
    final_semantic_labels = torch.cat(semantic_labels, dim=0) if semantic_labels else None

    return hidden_states, final_pos_labels, final_semantic_labels


def prepare_probing_dataset(
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 32
) -> DataLoader:
    """
    Prepare a DataLoader for probing experiments.
    """
    B, T, D = hidden_states.shape
    X = hidden_states.view(B * T, D)
    y = labels.repeat_interleave(T, dim=0)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_one_hot_labels(
        input_ids: torch.Tensor,
        tokenizer,
        label_type: str,
        config: Optional[LinguisticProbesConfig] = None
) -> torch.Tensor:
    """
    Create one-hot encoded labels for input sequences.

    Args:
        input_ids: Input token IDs
        tokenizer: Tokenizer for decoding
        label_type: Type of labels ('pos' or 'semantic')
        config: Configuration object

    Returns:
        One-hot encoded labels
    """
    if label_type == 'pos':
        return create_one_hot_pos_labels(input_ids, tokenizer, config)
    elif label_type == 'semantic':
        return create_one_hot_semantic_labels(input_ids, tokenizer, config)
    else:
        raise ValueError(f"Unknown label type: {label_type}")


def create_one_hot_pos_labels(
        input_ids: torch.Tensor,
        tokenizer,
        config: Optional[LinguisticProbesConfig] = None,
        ignore_index: int = -100
) -> torch.Tensor:
    """
    Create one-hot POS labels for input sequences.

    Args:
        input_ids: Input token IDs
        tokenizer: Tokenizer for decoding
        config: Configuration object

    Returns:
        One-hot encoded POS labels
    """
    if config is None: # fallback to default config
        config = LinguisticProbesConfig.default()

    # Get POS categories
    pos_to_idx = config.get_pos_categories()
    num_features = len(pos_to_idx)

    # Initialize tagger
    granularity = config.pos_granularity
    tagger = POSTagger(granularity=granularity)

    labels = []

    for sentence_ids in input_ids:
        token_ids = sentence_ids.detach().cpu().tolist()
        # Remove padding tokens
        if tokenizer.pad_token_id is not None:
            token_ids = [t for t in token_ids if t != tokenizer.pad_token_id and t != ignore_index]  # ignore_index is used in some tokenizers
        else:
            token_ids = [t for t in token_ids if t != 0] # we use 0 as the padding token in many tokenizers
        text = tokenizer.decode(token_ids)
        tagged_tokens = tagger.tag_text(text)

        label = np.zeros(num_features, dtype=np.float32)
        for _, pos in tagged_tokens:
            if pos in pos_to_idx:
                label[pos_to_idx[pos]] = 1
            else:
                # Fallback to OTHER if available
                if "OTHER" in pos_to_idx:
                    label[pos_to_idx["OTHER"]] = 1

        labels.append(label)

    return torch.tensor(labels, dtype=torch.float32)


def create_one_hot_semantic_labels(
        input_ids: torch.Tensor,
        tokenizer,
        config: Optional[LinguisticProbesConfig] = None,
        ignore_index: int = -100
) -> torch.Tensor:
    """
    Create one-hot semantic labels for input sequences.

    Args:
        input_ids: Input token IDs
        tokenizer: Tokenizer for decoding
        config: Configuration object

    Returns:
        One-hot encoded semantic labels
    """
    if config is None:
        config = LinguisticProbesConfig.default()

    # Get semantic categories
    semantic_to_idx = config.get_semantic_categories()
    # print(f"Semantic categories: {semantic_to_idx}")
    num_features = len(semantic_to_idx)

    # Initialize tagger
    granularity = config.semantic_granularity
    tagger = SemanticTagger(granularity=granularity)

    labels = []

    for sentence_ids in input_ids:
        token_ids = sentence_ids.detach().cpu().tolist()
        # Remove padding tokens
        if tokenizer.pad_token_id is not None:
            token_ids = [t for t in token_ids if t != tokenizer.pad_token_id and t != ignore_index]
        else:
            token_ids = [t for t in token_ids if t != 0]
        text = tokenizer.decode(token_ids)
        tagged_tokens = tagger.tag_text(text)

        label = np.zeros(num_features, dtype=np.float32)
        for _, role in tagged_tokens:
            if role in semantic_to_idx:
                label[semantic_to_idx[role]] = 1
            else:
                # Fallback to OTHER if available
                if "OTHER" in semantic_to_idx:
                    label[semantic_to_idx["OTHER"]] = 1

        labels.append(label)

    return torch.tensor(labels, dtype=torch.float32)


def analyze_label_distribution(
        dataloader: DataLoader,
        tokenizer,
        label_type: str,
        config: Optional[LinguisticProbesConfig] = None,
) -> Counter:
    """
    Analyze the distribution of labels in a dataset.
    """
    tag_counter = Counter()

    for batch in dataloader:
        input_ids = batch["input_ids"]
        labels = create_one_hot_labels(input_ids, tokenizer, label_type, config)

        for label in labels:
            tag_counter.update(torch.where(label > 0)[0].tolist())

    return tag_counter


def print_label_distribution(
        counter: Counter,
        label_names: List[str],
        title: str = "Label Distribution"
) -> None:
    """
    Print formatted label distribution.
    """
    print(f"\n{title}:")
    print("-" * 50)

    total = sum(counter.values())
    for idx, count in counter.most_common():
        if idx < len(label_names):
            name = label_names[idx]
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{name:<15}: {count:>6} ({percentage:>5.1f}%)")

