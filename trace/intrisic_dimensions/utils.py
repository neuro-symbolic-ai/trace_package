from typing import Optional, Union, List, Dict, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from trace.intrisic_dimensions.config import IntrinsicDimensionsConfig


def extract_hidden_representations(
        model, # did not specify type here to avoid circular import issues and to keep it flexible
        dataloader: DataLoader,
        device: str = None,
        layer_indices: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        tokenizer=None,
        config: Optional[IntrinsicDimensionsConfig] = None,
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], None, None]:
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
    if device is None:
        device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

    # print('Initial layer indices:', layer_indices)
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

    # print(f"Extracting hidden states from updated layers: {layer_indices}")

    # Initialize hidden states dictionary
    hidden_dict = {}
    for layer_type, indices in layer_indices.items():
        for idx in indices:
            hidden_dict[(idx, layer_type)] = []


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

            if model.model_type == 'encoder_only':
                # Forward pass for encoder
                _ = model(src=input_ids)
            elif model.model_type == 'decoder_only':
                # Forward pass for decoder
                _ = model(tgt=input_ids)
            elif model.model_type == 'encoder_decoder':
                # Forward pass for encoder
                _ = model(src=input_ids, tgt=input_ids)  # Assuming src and tgt are the same for simplicity

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
    # print(f"Extracted hidden states for layers: {list(hidden_states.keys())}")
    return hidden_states, None, None  # pos_labels and semantic_labels are not used in this function


def compute_intrinsic_dimensions(
        hidden_states: Dict[Tuple[int, str], torch.Tensor],
        config: IntrinsicDimensionsConfig,
        device: str = None
) -> Dict[Tuple[int, str], float]:
    """
    Compute intrinsic dimensions for given hidden states.

    Args:
        hidden_states: Dictionary of hidden states from transformer layers
        config: Configuration object for intrinsic dimensions analysis
        device: Device for computation

    Returns:
        Dictionary with intrinsic dimension results
    """
    results = {}
    for layer_name, layer_hidden in hidden_states.items():
        layer_data = layer_hidden.detach().cpu().numpy()
        layer_data = layer_data.reshape(layer_data.shape[0], -1)
        layer_data = layer_data.to(device) if device else layer_data.to('cpu')
        results[layer_name] = config.id_estimator.fit_transform(layer_data)

    # for key, states in hidden_states.items():
    #     # Move to device
    #     states = states.to(device)
    #     layer_data = layer_hidden.detach().cpu().numpy()
    #     # Compute intrinsic dimension
    #     id_result = nn_2.fit_transform(states)
    #
    #     # Store results
    #     results[str(key)] = id_result.intrinsic_dimension_

    return results


def average_intrinsic_dimension(
        intrinsic_dimensions: Dict[Tuple[int, str], float],
        layer_indices: Optional[List[Tuple[int, str]]] = None
) -> float:
    """
    Compute the average intrinsic dimension across specified layers.

    Args:
        intrinsic_dimensions: Dictionary of intrinsic dimensions for each layer
        layer_indices: Optional list or dict of layer indices to average over

    Returns:
        Average intrinsic dimension
    """
    if layer_indices is None:
        layer_indices = list(intrinsic_dimensions.keys())

    total_id = 0.0
    count = 0

    for layer in layer_indices:
        if layer in intrinsic_dimensions:
            total_id += intrinsic_dimensions[layer]
            count += 1

    if count == 0:
        return 0.0

    return total_id / count