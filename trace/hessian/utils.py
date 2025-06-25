import torch
import numpy as np
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh
from typing import List, Optional, Callable
from ..transformer import Transformer


def compute_loss(model: Transformer, loss_fn, data_batch, model_type: Optional[str] = "decoder_only",
                 ignore_index: Optional[int] = -100) -> torch.Tensor:
    """
    Compute loss for a batch of data.

    Args:
        model: The transformer model
        loss_fn: Loss function
        data_batch: A batch of data
        model_type: Type of transformer model ("encoder_decoder" or "decoder_only")
        ignore_index: Index to ignore in the loss calculation (default: -100)

    Returns:
        Loss value
    """
    # Handle different model types
    if model_type == "encoder_decoder":
        src_mask = data_batch["attention_mask"].unsqueeze(1)
        tgt_mask = model.transformer.generate_square_subsequent_mask(
            data_batch["decoder_input_ids"].size(1),
            data_batch["decoder_input_ids"].device
        ).unsqueeze(0).expand(data_batch["decoder_input_ids"].size(0), -1, -1)

        outputs = model(
            src=data_batch["input_ids"],
            tgt=data_batch["decoder_input_ids"],
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=(data_batch["attention_mask"] == 0),
            tgt_key_padding_mask=(data_batch["decoder_attention_mask"] == 0)
        )

    elif model_type == "decoder_only":
        tgt_mask = model.generate_square_subsequent_mask(
            data_batch["input_ids"].size(1),
            data_batch["input_ids"].device
        ).unsqueeze(0).expand(data_batch["input_ids"].size(0), -1, -1)

        outputs = model(
            tgt=data_batch["input_ids"],
            tgt_mask=tgt_mask,
        )
    else:
        # encoder-only or other model types
        outputs = model(
            src=data_batch["input_ids"],
            # attention_mask=data_batch["attention_mask"]
        )

    # Calculate loss - ensure labels are properly handled
    if "labels" in data_batch:
        # For decoder-only, handle padding with ignore_index
        if model_type == "decoder_only":
            target_labels = data_batch["labels"].clone()
            pad_idx = getattr(model, 'pad_idx', 0)
            target_labels[target_labels == pad_idx] = ignore_index
        else:
            target_labels = data_batch["labels"]

        loss = loss_fn(
            outputs.reshape(-1, outputs.size(-1)),
            target_labels.reshape(-1)
        )
    else:
        # If no labels, return a dummy loss (e.g., zero)
        loss = torch.tensor(0.0, device=data_batch["input_ids"].device)
    return loss


def extract_component_parameters(model: Transformer, component_name: str,
                                 include_bias: Optional[bool] = False) -> List[torch.Tensor]:
    """
    Extract parameters for specific model components.

    Args:
        model: The transformer model
        component_name: Name of the component to extract parameters for
        include_bias: Whether to include bias parameters (default: False)

    Returns:
        List of parameters for the specified component
    """
    # Define component parameter filters
    component_filters = {
        "attention": lambda name: "attn" in name,
        "attention_query": lambda name: "attention" in name and "q_linear" in name,
        "attention_key": lambda name: "attention" in name and "k_linear" in name,
        "attention_value": lambda name: "attention" in name and "v_linear" in name,
        "ffn": lambda name: any(x in name for x in ["ffn", "mlp", "feed_forward"]),
        "embeddings": lambda name: "embed" in name,
        "norm": lambda name: "norm" in name or "ln" in name,
        "hidden_states": lambda name: "norm1" in name if getattr(model, 'no_fnn', False) else "norm2" in name,
        "output_projection": lambda name: "output" in name or "classifier" in name or "lm_head" in name,
        "all": lambda name: True
    }

    if component_name not in component_filters:
        raise ValueError(f"Unknown component: {component_name}. Available components: {list(component_filters.keys())}")

    # Apply the filter to select parameters
    if include_bias:
        params = [p for name, p in model.named_parameters()
                  if component_filters[component_name](name) and p.requires_grad]
    else:
        # Exclude bias parameters
        params = [p for name, p in model.named_parameters()
                  if component_filters[component_name](name) and p.requires_grad and 'bias' not in name]

    if not params:
        raise ValueError(f"No parameters found for component: {component_name}")

    return params


def get_hessian_eigenvectors(
        model: Transformer,
        loss_fn,
        train_data_loader,
        num_batches: int,
        device,
        n_top_vectors: int,
        param_extract_fn: Optional[Callable] = None
):
    """
    Compute Hessian eigenvalues and eigenvectors using the Lanczos algorithm.
    model: a pytorch model
    loss_fn: a pytorch loss function
    train_data_loader: a pytorch data loader
    num_batches: number of batches to use for the hessian calculation
    device: the device to use for the hessian calculation
    n_top_vectors: number of top eigenvalues / eigenvectors to return
    param_extract_fn: a function that takes a model and returns a list of parameters to compute the hessian with respect to (pass None to use all parameters)
    returns: a tuple of (eigenvalues, eigenvectors)
    eigenvalues: a numpy array of the top eigenvalues, arranged in increasing order
    eigenvectors: a numpy array of the top eigenvectors, arranged in increasing order, shape (n_top_vectors, num_params)
    :note: alternative:  Computing eigenvalues using ARPACK (Arnoldi iteration)
    """
    param_extract_fn = param_extract_fn or (lambda x: x.parameters())
    # Store the selected parameters ONCE (avoid redundant function calls)
    param_list = list(param_extract_fn(model))
    num_params = sum(p.numel() for p in param_list)

    def hessian_vector_product(vector):
        """Compute Hessian-vector product."""
        model.zero_grad()
        loss = compute_loss(model, loss_fn, train_data_loader)
        grad_params = grad(loss, param_list, create_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in grad_params])
        grad_vector_product = torch.dot(flat_grad, vector)
        hvp = grad(grad_vector_product, param_list, retain_graph=True)
        return torch.cat([g.contiguous().view(-1) for g in hvp])

    def matvec(v):
        """Matrix-vector product for scipy's eigsh."""
        v_tensor = torch.tensor(v, dtype=torch.float32, device=device)
        return hessian_vector_product(v_tensor).cpu().detach().numpy()

    linear_operator = LinearOperator((num_params, num_params), matvec=matvec)
    print('linear_operator shape:', linear_operator)
    eigenvalues, eigenvectors = eigsh(
        linear_operator,
        k=n_top_vectors,
        tol=0.001,
        which='LM',
        return_eigenvectors=True
    )
    print('Computed eigenvalues:', eigenvalues)
    eigenvectors = np.transpose(eigenvectors)
    print('Eigenvectors shape:', eigenvectors.shape)
    return eigenvalues[::-1], eigenvectors[::-1]

