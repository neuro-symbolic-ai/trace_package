# Transformer Module Usage Examples

The `trace.transformer` module provides a clean, modular implementation of transformer models supporting encoder-only, decoder-only, and encoder-decoder architectures.

## Basic Usage

### Creating Models with Factory Methods

```python
from trace.transformer import Transformer, TransformerConfig
import torch



# Create an encoder-only model
config = TransformerConfig(
    model_type="encoder_only",
    vocab_size=30522,
    d_model=768,
    num_heads=12,
    num_encoder_layers=6,
    d_ff=3072,
    max_seq_length=512,
    dropout=0.1
)
encoder_model = Transformer.from_config(config)
input_ids = torch.randint(0, 30522, (2, 20))
hidden_states = encoder_model(src=input_ids)
print(f"Output shape: {hidden_states.shape}")  # [2, 20, 768]

# Create an encoder-decoder model
config = TransformerConfig(
    model_type="encoder_decoder",
    vocab_size=32128,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_length=512,
    dropout=0.1
)
encoder_decoder_model = Transformer.from_config(config)
src_ids = torch.randint(0, 32128, (2, 15))  # Source sequence
tgt_ids = torch.randint(0, 32128, (2, 10))  # Target sequence
logits = encoder_decoder_model(src=src_ids, tgt=tgt_ids)
print(f"Output shape: {logits.shape}")  # [2, 10, 32128]

# Create a decoder-only model
config = TransformerConfig(
    model_type="decoder_only",
    vocab_size=2000,
    d_model=768,
    num_heads=12,
    num_decoder_layers=1,
    d_ff=3072,
    max_seq_length=64,
    dropout=0.1,
    device="cpu"  # todo make sure this is set
    # no_fnn=True  # Disable feed-forward networks
)

decoder = Transformer.from_config(config)

input_ids = torch.randint(0, 2000, (2, 10))
logits = decoder(tgt=input_ids)
print(f"Output shape: {logits.shape}")  # [2, 10, 50257]

# We can use a causal mask for the decoder
causal_mask = decoder.generate_square_subsequent_mask(10, input_ids.device)
logits = decoder(tgt=input_ids, tgt_mask=causal_mask)
print(f"Output shape: {logits.shape}")  # [2, 10, 50257]

# Count parameters
param_count =decoder.count_parameters()
print(f"Trainable parameters: {param_count:,}")

# Get model size description
size_desc = decoder.get_model_size()
print(f"Model size: {size_desc}")

# Initialize weights
decoder.initialize_weights() #todo :fix this

# Freeze feed-forward networks
decoder.freeze_ffn(freeze=True)

# Unfreeze feed-forward networks
decoder.freeze_ffn(freeze=False)
```

### Working with Individual Components
You can also use individual components of the transformer architecture directly for more granular control.
```python
from trace.transformer.components import (
    MultiHeadAttention, 
    PositionalEncoding,
    EncoderLayer
)
import torch

# Use individual components
pos_encoding = PositionalEncoding(d_model=512, max_seq_length=1000)
attention = MultiHeadAttention(d_model=512, num_heads=8)
encoder_layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048)

# Apply components
x = torch.randn(2, 20, 512)
x_with_pos = pos_encoding(x)
attn_out, weights = attention(x_with_pos, x_with_pos, x_with_pos)
layer_out, _ = encoder_layer(x_with_pos)
```

# Hessian Trace 
```python
from trace.hessian import HessianConfig, HessianAnalyzer
import torch
# Create a Hessian configuration
config = HessianConfig(
    n_components=5,  # More eigenvalues for detailed analysis
    track_component_hessian=True,
    track_gradient_alignment=True,
    track_train_val_landscape_divergence=True,
    track_sharpness=True,
    component_list=["attention", "ffn"]
)
# or simply use the default config
# config = HessianConfig()

# Create a Hessian analyzer
analyzer = HessianAnalyzer(config=config) 
# or analyzer = HessianAnalyzer.from_config(config)
loss_fn = torch.nn.CrossEntropyLoss()

# Sample batch (you would use your actual data)
batch = {
    "input_ids": torch.randint(0, 1000, (4, 20)),
    "labels": torch.randint(0, 1000, (4, 20))
}

# Perform comprehensive analysis
results = analyzer.analyze_step(
    decoder, loss_fn, 
    train_batch=batch, val_batch=batch, 
    model_type="decoder_only", step=500
)

# Access different analysis results
print(f"Max eigenvalue: {results['hessian']['max_eigenvalue']}")
print(f"Trace estimate: {results['hessian']['hessian_trace_estimate']}")
print(f"Negative eigenvalues: {results['hessian']['negative_count']}")

from trace.hessian import HessianVisualizer

# After collecting history during training
visualizer = HessianVisualizer()

from trace.transformer import Transformer, TransformerConfig
import torch
config = TransformerConfig(
    model_type="decoder_only",
    vocab_size=2000,
    d_model=768,
    num_heads=12,
    num_decoder_layers=1,
    d_ff=3072,
    max_seq_length=64,
    dropout=0.1,
    device="cpu"  # todo make sure this is set
    # no_fnn=True  # Disable feed-forward networks
)

decoder = Transformer.from_config(config)
from trace.linguistic_probes import run_pos_probe_analysis, run_semantic_probe_analysis, LinguisticProbesConfig

results = run_pos_probe_analysis(decoder, dataloader, tokenizer, device)
results = run_semantic_probe_analysis(decoder, dataloader, tokenizer, device)

```

----

### Gradient Monitoring

```python
from trace.transformer import attach_hooks, remove_hooks

# Attach gradient monitoring hooks
model_with_hooks = attach_hooks(model)

# Train model (hooks will print gradient norms)
# ... training code ...

# Remove hooks when done
model = remove_hooks(model)
```

### Custom Training Setup

```python
# import torch.optim as optim
# 
# # Create model and optimizer
# model = TransformerFactory.create_decoder_only_transformer(vocab_size=50257)
# optimizer = optim.AdamW(model.parameters(), lr=1e-4)
# 
# # Training loop example
# model.train()
# for batch in dataloader:
#     input_ids = batch['input_ids']
#     
#     # Forward pass
#     logits = model(tgt=input_ids[:, :-1])  # All but last token
#     targets = input_ids[:, 1:]  # All but first token
#     
#     # Compute loss
#     loss = F.cross_entropy(
#         logits.reshape(-1, logits.size(-1)),
#         targets.reshape(-1)
#     )
#     
#     # Backward pass
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
```



## 🏗️ Transformer Model Creation

TRACE provides a flexible transformer implementation supporting multiple architectures with consistent configuration patterns.

### Supported Architectures

- **Encoder-Only**: BERT-style models for encoding tasks
- **Decoder-Only**: GPT-style models for autoregressive generation  
- **Encoder-Decoder**: T5/BART-style models for sequence-to-sequence tasks

### Configuration System

```python
from trace.transformer import Transformer, TransformerConfig
import torch

# All models use the same configuration class
config = TransformerConfig(
    model_type="encoder_only",        # Architecture type
    vocab_size=30522,                 # Vocabulary size
    d_model=768,                      # Model dimension
    num_heads=12,                     # Attention heads
    num_encoder_layers=6,             # Encoder layers (if applicable)
    num_decoder_layers=6,             # Decoder layers (if applicable)
    d_ff=3072,                        # Feed-forward dimension
    max_seq_length=512,               # Maximum sequence length
    dropout=0.1,                      # Dropout rate
    device="cpu"                      # Device placement
)
```

### Encoder-Only Models

Perfect for classification, encoding, and representation learning tasks:

```python
# Create an encoder-only model (BERT-style)
config = TransformerConfig(
    model_type="encoder_only",
    vocab_size=30522,
    d_model=768,
    num_heads=12,
    num_encoder_layers=6,
    d_ff=3072,
    max_seq_length=512,
    dropout=0.1
)

encoder_model = Transformer.from_config(config)

# Forward pass
input_ids = torch.randint(0, 30522, (2, 20))  # [batch_size, seq_len]
hidden_states = encoder_model(src=input_ids)
print(f"Output shape: {hidden_states.shape}")  # [2, 20, 768]
```

### Encoder-Decoder Models

Ideal for translation, summarization, and other sequence-to-sequence tasks:

```python
# Create an encoder-decoder model (T5/BART-style)
config = TransformerConfig(
    model_type="encoder_decoder",
    vocab_size=32128,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_length=512,
    dropout=0.1
)

encoder_decoder_model = Transformer.from_config(config)

# Forward pass with source and target sequences
src_ids = torch.randint(0, 32128, (2, 15))  # Source sequence
tgt_ids = torch.randint(0, 32128, (2, 10))  # Target sequence
logits = encoder_decoder_model(src=src_ids, tgt=tgt_ids)
print(f"Output shape: {logits.shape}")  # [2, 10, 32128]
```

### Decoder-Only Models

Optimized for autoregressive generation and language modeling:

```python
# Create a decoder-only model (GPT-style)
config = TransformerConfig(
    model_type="decoder_only",
    vocab_size=2000,
    d_model=768,
    num_heads=12,
    num_decoder_layers=12,
    d_ff=3072,
    max_seq_length=64,
    dropout=0.1,
    device="cpu"
)

decoder_model = Transformer.from_config(config)

# Forward pass for autoregressive generation
input_ids = torch.randint(0, 2000, (2, 32))  # [batch_size, seq_len]
logits = decoder_model(tgt=input_ids)
print(f"Output shape: {logits.shape}")  # [2, 32, 2000]
```

### Advanced Configuration Options

```python
# Specialized configurations
config = TransformerConfig(
    model_type="decoder_only",
    vocab_size=2000,
    d_model=768,
    num_heads=12,
    num_decoder_layers=1,            # Single layer for analysis
    d_ff=3072,
    max_seq_length=64,
    dropout=0.1,
    device="cpu",
    # no_ffn=True                    # Disable feed-forward networks (optional)
)

# Create model with custom settings
specialized_model = Transformer.from_config(config)
```

### Model Information and Utilities

```python
# Get model information
print(f"Model type: {decoder_model.config.model_type}")
print(f"Parameters: {sum(p.numel() for p in decoder_model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in decoder_model.parameters() if p.requires_grad):,}")

# Access model components
print(f"Number of layers: {len(decoder_model.layers)}")
print(f"Model dimension: {decoder_model.config.d_model}")
```

## 📐 Intrinsic Dimensions Analysis

TRACE provides comprehensive tools for analyzing the intrinsic dimensionality of transformer representations using advanced geometric methods. This helps understand how models compress and organize information across layers.

### Supported Methods

- **TwoNN**: Two Nearest Neighbors method for robust ID estimation
- **MLE**: Maximum Likelihood Estimation approach
- **PCA**: Principal Component Analysis for linear dimensionality

### Basic Configuration

```python
from trace.intrinsic_dimensions import IntrinsicDimensionsConfig, IntrinsicDimensionAnalyzer

# Create configuration for intrinsic dimensions analysis
config = IntrinsicDimensionsConfig(
    model_type="decoder_only",           # Model architecture
    id_method="TwoNN",                   # Dimensionality estimation method
    layers_to_analyze=None,              # None = analyze all layers
    max_samples=1000,                    # Limit samples for efficiency
    flatten_sequence=True,               # Flatten sequence dimension
    save_visualizations=True,            # Generate plots
    show_plots=False,                    # Display plots immediately
    log_dir="./plots/intrinsic_dims"     # Output directory
)

# Initialize analyzer
analyzer = IntrinsicDimensionAnalyzer(config)
```

### Layer-Specific Analysis

```python
# Analyze specific layers for encoder-only models
config = IntrinsicDimensionsConfig(
    model_type="encoder_only",
    layers_to_analyze={'encoder': [0, 3, 5]},  # Analyze layers 0, 3, and 5
    id_method="TwoNN"
)

# For encoder-decoder models
config = IntrinsicDimensionsConfig(
    model_type="encoder_decoder",
    layers_to_analyze={
        'encoder': [0, 2, 4],
        'decoder': [0, 1, 3]
    }
)

# For decoder-only models (simple list notation)
config = IntrinsicDimensionsConfig(
    model_type="decoder_only",
    layers_to_analyze=[0, 6, 11],  # First, middle, and last layers
    id_method="MLE"
)
```

### Complete Analysis Workflow

```python
from trace.intrinsic_dimensions import IntrinsicDimensionAnalyzer
from torch.utils.data import DataLoader

# Prepare your data
# data_loader = DataLoader(your_dataset, batch_size=32, shuffle=False)

# Create analyzer with default settings
analyzer = IntrinsicDimensionAnalyzer()

# Run comprehensive analysis
intrinsic_dimensions = analyzer.analyze(
    model=decoder_model,
    data_loader=data_loader,
    layers=[0, 3, 6, 9, 11],  # Specific layers to analyze
    model_name="my_transformer"
)

# Results dictionary: {(layer_idx, layer_type): intrinsic_dimension}
print("Intrinsic Dimensions Results:")
for (layer_idx, layer_type), id_value in intrinsic_dimensions.items():
    print(f"  Layer {layer_idx} ({layer_type}): {id_value:.2f}")
```

### Manual Hidden State Extraction

```python
from trace.intrinsic_dimensions import (
    extract_hidden_representations,
    compute_intrinsic_dimensions,
    average_intrinsic_dimension
)

# Extract hidden states from specific layers
hidden_states, _, _ = extract_hidden_representations(
    model=encoder_model,
    dataloader=data_loader,
    layer_indices={'encoder': [0, 2, 4]},
    device='cuda'
)

# Compute intrinsic dimensions
config = IntrinsicDimensionsConfig(id_method="TwoNN")
intrinsic_dims = compute_intrinsic_dimensions(hidden_states, config)

# Calculate average across layers
avg_id = average_intrinsic_dimension(
    intrinsic_dims,
    layer_indices=[(0, 'encoder'), (2, 'encoder'), (4, 'encoder')]
)
print(f"Average intrinsic dimension: {avg_id:.2f}")
```

### Advanced Visualization Options

```python
from trace.intrinsic_dimensions import IntrinsicDimensionsVisualizer

# Create custom visualizer
visualizer = IntrinsicDimensionsVisualizer(
    log_dir="./custom_plots",
    config=config
)

# Generate comprehensive visualizations
visualizer.generate_all_visualizations(
    intrinsic_dimensions=intrinsic_dims,
    model_name="transformer_analysis",
    show_plots=True  # Display plots immediately
)

# Individual plot types
visualizer.plot_id_by_layer(intrinsic_dims, "my_model", save_plot=True)
visualizer.plot_id_distribution(intrinsic_dims, "my_model", save_plot=True)
visualizer.plot_final_id(intrinsic_dims, "my_model", save_plot=True)

# Save metrics to CSV
visualizer.save_metrics(intrinsic_dims, "my_model")
```

### Method Comparison

```python
# Compare different ID estimation methods
methods = ["TwoNN", "MLE", "PCA"]
results_comparison = {}

for method in methods:
    config = IntrinsicDimensionsConfig(
        id_method=method,
        layers_to_analyze=[0, 6, 11]
    )
    analyzer = IntrinsicDimensionAnalyzer(config)
    results_comparison[method] = analyzer.analyze(
        model=decoder_model,
        data_loader=data_loader,
        model_name=f"model_{method.lower()}"
    )

# Compare results
for method, results in results_comparison.items():
    avg_id = average_intrinsic_dimension(results)
    print(f"{method} average ID: {avg_id:.2f}")
```

### Integration with Training Loop

```python
# Monitor intrinsic dimensions during training
def training_step_with_id_analysis(model, data_loader, step):
    # Your training code here...
    
    # Periodic intrinsic dimension analysis
    if step % 100 == 0:  # Analyze every 100 steps
        analyzer = IntrinsicDimensionAnalyzer(
            config=IntrinsicDimensionsConfig(
                layers_to_analyze=[0, -1],  # First and last layers
                save_visualizations=True,
                log_dir=f"./training_analysis/step_{step}"
            )
        )
        
        # Quick analysis on subset of data
        subset_loader = DataLoader(
            dataset=data_loader.dataset,
            batch_size=32,
            shuffle=False,
            sampler=torch.utils.data.SubsetRandomSampler(range(100))
        )
        
        id_results = analyzer.analyze(
            model=model,
            data_loader=subset_loader,
            model_name=f"training_step_{step}"
        )
        
        # Log results
        avg_id = average_intrinsic_dimension(id_results)
        print(f"Step {step}: Average ID = {avg_id:.2f}")
```

## 🌊 Hessian Analysis

TRACE provides comprehensive Hessian analysis capabilities for understanding loss landscape properties, training dynamics, and memorization patterns. This module analyzes curvature information to reveal insights about model optimization and generalization.

### Key Features

- **Eigenvalue Analysis**: Track extreme eigenvalues, trace estimates, and spectral properties
- **Component-Specific Analysis**: Analyze individual model components (attention, FFN, embeddings)
- **Gradient-Hessian Alignment**: Monitor optimization direction relative to curvature
- **Memorization Detection**: Compare train/validation landscapes to detect overfitting
- **Real-time Monitoring**: Integration with training loops for continuous analysis

### Basic Configuration

```python
from trace.hessian import HessianConfig, HessianAnalyzer

# Create configuration for Hessian analysis
config = HessianConfig(
    n_components=10,                           # Number of eigenvalues to compute
    num_batches=100,                          # Batches for Hessian estimation
    device="cuda",                            # Device for computation
    
    # Analysis toggles
    track_component_hessian=True,             # Analyze individual components
    track_gradient_alignment=True,            # Monitor gradient-Hessian alignment
    track_train_val_landscape_divergence=True, # Detect memorization signals
    
    # Component selection
    component_list=["attention", "ffn", "hidden_states"],
    
    # Output settings
    log_dir="./hessian_analysis",             # Output directory
    save_hessian_data=True,                   # Save raw data
    show_plots=False                          # Display plots immediately
)

# Initialize analyzer
analyzer = HessianAnalyzer(config)
```

### Preset Configurations

```python
# Minimal configuration for basic analysis
minimal_config = HessianConfig.minimal(
    n_components=5,
    track_component_hessian=False
)

# Comprehensive configuration for detailed research
comprehensive_config = HessianConfig.comprehensive(
    n_components=20,
    component_list=[
        "attention", "attention_query", "attention_key", "attention_value",
        "ffn", "embeddings", "norm", "hidden_states", "output_projection"
    ]
)

# Default balanced configuration
default_config = HessianConfig.default()
```

### Single-Step Analysis

```python
import torch.nn as nn

# Prepare loss function and data
loss_fn = nn.CrossEntropyLoss()
# train_batch and val_batch should be prepared with your data

# Perform comprehensive analysis at a single training step
results = analyzer.analyze_step(
    model=decoder_model,
    loss_fn=loss_fn,
    train_batch=train_batch,
    val_batch=val_batch,              # Optional for memorization analysis
    model_type="decoder_only",
    step=100
)

# Access results
print(f"Max eigenvalue: {results['hessian']['max_eigenvalue']:.2e}")
print(f"Min eigenvalue: {results['hessian']['min_eigenvalue']:.2e}")
print(f"Trace estimate: {results['hessian']['hessian_trace_estimate']:.2e}")
print(f"Negative eigenvalues: {results['hessian']['negative_count']}")
print(f"Effective rank: {results['hessian']['effective_rank_95']}")
```

### Component-Specific Analysis

```python
from trace.hessian import ComponentAnalyzer, ComponentSelector

# Initialize component analyzer
component_analyzer = ComponentAnalyzer()

# Get appropriate components for your model
no_ffn = getattr(decoder_model, 'no_ffn', False)
components = ComponentSelector.get_standard_components(no_ffn)
print(f"Analyzing components: {components}")

# Analyze all components
component_results = component_analyzer.analyze_all_components(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    component_list=components,
    n_components=10
)

# Compare component complexity
for component, metrics in component_results.items():
    if 'error' not in metrics:
        print(f"{component}:")
        print(f"  Parameters: {metrics['num_params']:,}")
        print(f"  Max eigenvalue: {metrics['max_eigenvalue']:.2e}")
        print(f"  Effective rank: {metrics['effective_rank_95']}")
```

### Advanced Component Selection

```python
# Validate components exist in your model
valid_components = ComponentSelector.validate_components(
    model=decoder_model,
    component_list=["attention", "ffn", "embeddings", "norm"]
)

# Use comprehensive component set for detailed analysis
comprehensive_components = ComponentSelector.get_comprehensive_components()

# Custom component analysis
custom_analyzer = ComponentAnalyzer()
attention_results = custom_analyzer.analyze_component(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    component_name="attention",
    n_components=15
)
```

### Gradient-Hessian Alignment Analysis

```python
# Analyze optimization dynamics through gradient-curvature relationships
alignment_results = analyzer.compute_gradient_alignment(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    eigenvalues=eigenvalues,  # From previous Hessian computation
    eigenvectors=eigenvectors
)

# Key alignment metrics
print(f"Gradient-Hessian alignment: {alignment_results['grad_Hg_alignment']:.4f}")
print(f"Weighted alignment score: {alignment_results['weighted_alignment']:.4f}")
print(f"Curvature/gradient ratio: {alignment_results['grad_Hg_ratio']:.4f}")
```

### Memorization Detection

```python
# Detect memorization through train/validation landscape comparison
memorization_signals = analyzer.compute_train_val_divergence(
    model=decoder_model,
    loss_fn=loss_fn,
    train_batch=train_batch,
    val_batch=val_batch,
    model_type="decoder_only"
)

# Memorization indicators
print(f"Landscape divergence score: {memorization_signals['train_val_landscape_divergence_score']:.4f}")
print(f"Trace ratio (train/val): {memorization_signals['trace_ratio']:.4f}")
print(f"Eigenvalue distribution overlap: {memorization_signals['eigenvalue_distribution_overlap']:.4f}")
print(f"Effective rank difference: {memorization_signals['effective_rank_diff']}")
```

### Low-Level Utilities

```python
from trace.hessian import (
    compute_loss,
    extract_component_parameters,
    get_hessian_eigenvectors
)

# Manual loss computation with proper model handling
loss = compute_loss(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    ignore_index=-100
)

# Extract parameters for specific components
attention_params = extract_component_parameters(
    model=decoder_model,
    component_name="attention",
    include_bias=False
)
print(f"Attention parameters: {sum(p.numel() for p in attention_params):,}")

# Manual Hessian eigenvalue computation
eigenvalues, eigenvectors = get_hessian_eigenvectors(
    model=decoder_model,
    loss_fn=loss_fn,
    train_data_loader=train_batch,
    num_batches=50,
    device="cuda",
    n_top_vectors=15
)
```

### Comprehensive Visualization

```python
from trace.hessian import HessianVisualizer

# Initialize visualizer
visualizer = HessianVisualizer(config)

# Track analysis over multiple steps
hessian_history = {}
for step in range(0, 1000, 100):
    # Perform analysis at each step
    results = analyzer.analyze_step(
        model=decoder_model,
        loss_fn=loss_fn,
        train_batch=train_batch,
        val_batch=val_batch,
        step=step
    )
    hessian_history[step] = results

# Generate comprehensive visualization report
visualizer.create_comprehensive_report(
    hessian_history=hessian_history,
    model_name="my_transformer"
)

# Individual plot types
visualizer.plot_eigenvalue_evolution(hessian_history, "my_transformer")
visualizer.plot_gradient_alignment(hessian_history, "my_transformer") 
visualizer.plot_component_comparison(hessian_history, "my_transformer")
visualizer.plot_memorization_metrics(hessian_history, model_name="my_transformer")
```

### Training Loop Integration

```python
def training_step_with_hessian_analysis(model, optimizer, train_batch, val_batch, step):
    # Regular training step
    optimizer.zero_grad()
    loss = compute_loss(model, loss_fn, train_batch, "decoder_only")
    loss.backward()
    optimizer.step()
    
    # Periodic Hessian analysis
    if step % 50 == 0:  # Analyze every 50 steps
        # Create step-specific analyzer
        step_analyzer = HessianAnalyzer(HessianConfig.default(
            log_dir=f"./analysis/step_{step}",
            n_components=8,  # Reduced for speed
            track_train_val_landscape_divergence=(val_batch is not None)
        ))
        
        # Perform analysis
        hessian_results = step_analyzer.analyze_step(
            model=model,
            loss_fn=loss_fn,
            train_batch=train_batch,
            val_batch=val_batch,
            step=step
        )
        
        # Log key metrics
        if 'hessian' in hessian_results:
            h = hessian_results['hessian']
            print(f"Step {step}: "
                  f"Max λ: {h['max_eigenvalue']:.2e}, "
                  f"Trace: {h['hessian_trace_estimate']:.2e}, "
                  f"Neg count: {h['negative_count']}")
        
        # Check for memorization signals
        if 'train_val_divergence' in hessian_results:
            div_score = hessian_results['train_val_divergence']['train_val_landscape_divergence_score']
            if div_score > 0.5:  # Threshold for concern
                print(f"⚠️  High memorization signal detected: {div_score:.3f}")
    
    return loss.item()
```

### Performance Optimization

```python
# Efficient configuration for large models
efficient_config = HessianConfig(
    n_components=5,                    # Fewer eigenvalues
    num_batches=50,                    # Fewer batches
    track_component_hessian=False,     # Skip component analysis
    track_gradient_alignment=False,    # Skip alignment analysis
    track_train_val_landscape_divergence=False  # Skip memorization detection
)

# Memory-efficient analysis for very large models
def memory_efficient_hessian_analysis(model, data_batch, step):
    # Use CPU for Hessian computation to save GPU memory
    cpu_config = HessianConfig.minimal(device="cpu", n_components=3)
    analyzer = HessianAnalyzer(cpu_config)
    
    # Move only necessary data to CPU
    cpu_batch = {k: v.cpu() for k, v in data_batch.items()}
    model_cpu = model.cpu()
    
    results = analyzer.analyze_step(
        model=model_cpu,
        loss_fn=nn.CrossEntropyLoss(),
        train_batch=cpu_batch,
        step=step
    )
    
    # Move model back to GPU
    model.cuda()
    return results
```

## 🏗️ Transformer Model Creation

TRACE provides a flexible transformer implementation supporting multiple architectures with consistent configuration patterns.

### Supported Architectures

- **Encoder-Only**: BERT-style models for encoding tasks
- **Decoder-Only**: GPT-style models for autoregressive generation  
- **Encoder-Decoder**: T5/BART-style models for sequence-to-sequence tasks

### Configuration System

```

## 📤 Output Monitoring

TRACE provides sophisticated output monitoring capabilities to analyze how well your model generates text with correct linguistic properties. Unlike linguistic probes that analyze internal representations, output monitoring evaluates the actual generated text for POS accuracy and semantic role correctness.

### Key Features

- **Real-time Generation Analysis**: Monitor POS and semantic accuracy of generated text during training
- **Category-specific Performance**: Track performance for different linguistic categories
- **Comparative Analysis**: Compare predicted vs. ground truth text linguistic properties
- **Evolution Tracking**: Visualize how generation quality improves during training
- **Flexible Granularity**: Basic vs detailed linguistic analysis

### Basic Usage

```python
from trace.output_monitoring import OutputMonitoringAnalyzer, OutputMonitoringConfig

# Configure output monitoring
config = OutputMonitoringConfig(
    track_pos_performance=True,       # Monitor POS accuracy in outputs
    track_semantic_roles=True,        # Monitor semantic role accuracy  
    pos_granularity='basic',          # 'basic' or 'detailed'
    semantic_granularity='basic',     # 'basic' or 'detailed'
    save_visualizations=True,         # Create performance plots
    log_dir='./output_monitoring',    # Where to save results
    device='cuda'
)

# Initialize analyzer
analyzer = OutputMonitoringAnalyzer(config)

# Monitor during training
def training_step_with_output_monitoring(model, batch, step):
    # Forward pass
    outputs = model(**batch)
    logits = outputs.logits
    
    # Analyze output quality every 50 steps
    if step % 50 == 0:
        results = analyzer.analyze(
            batch=batch,           # Contains ground truth labels
            outputs=logits,        # Model predictions
            tokenizer=tokenizer,
            step=step
        )
        
        # Print results
        if 'pos_accuracy' in results:
            print(f"Step {step} POS Accuracy:")
            for pos_tag, accuracy in results['pos_accuracy'].items():
                print(f"  {pos_tag}: {accuracy:.3f}")
        
        if 'semantic_accuracy' in results:
            print(f"Step {step} Semantic Accuracy:")
            for sem_tag, accuracy in results['semantic_accuracy'].items():
                print(f"  {sem_tag}: {accuracy:.3f}")

# After training, get comprehensive results
final_results = analyzer.get_full_results()
pos_summary = analyzer.get_pos_summary()
semantic_summary = analyzer.get_semantic_summary()

print("Hardest POS categories:", pos_summary['hardest_categories'])
print("Easiest POS categories:", pos_summary['easiest_categories'])
```

### Configuration Options

```python
# Basic monitoring - quick setup
basic_config = OutputMonitoringConfig.default()

# POS-only monitoring
pos_only_config = OutputMonitoringConfig(
    track_pos_performance=True,
    track_semantic_roles=False,
    pos_granularity='detailed'        # More fine-grained POS categories
)

# Comprehensive monitoring with detailed analysis
comprehensive_config = OutputMonitoringConfig(
    track_pos_performance=True,
    track_semantic_roles=True,
    pos_granularity='detailed',       # Detailed POS: TRANSITIVE_VERB, COMMUNICATION_VERB, etc.
    semantic_granularity='detailed',  # Detailed semantic: MOTION, COMMUNICATION, DESTINATION, etc.
    save_visualizations=True,
    log_dir='./comprehensive_output_analysis',
    show_plots=False                  # Set True to display plots during training
)

# Memory-efficient monitoring for large models
efficient_config = OutputMonitoringConfig(
    track_pos_performance=True,
    track_semantic_roles=False,       # Disable semantic to save compute
    pos_granularity='basic',          # Use basic categories
    save_visualizations=False,        # Skip visualization to save memory
    device='cpu'                      # Use CPU for monitoring
)
```

### Integration with Training Loop

Complete example showing integration with a training loop:

```python
def complete_training_with_output_monitoring():
    # Initialize monitoring
    config = OutputMonitoringConfig(
        track_pos_performance=True,
        track_semantic_roles=True,
        save_visualizations=True,
        log_dir='./training_output_monitoring'
    )
    
    analyzer = OutputMonitoringAnalyzer(config)
    
    # Training loop
    model.train()
    for step, batch in enumerate(train_dataloader):
        # Regular training step
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Output monitoring every 100 steps
        if step % 100 == 0:
            model.eval()  # Switch to eval for monitoring
            
            with torch.no_grad():
                # Get model predictions for monitoring
                eval_outputs = model(**batch)
                
                # Analyze output quality
                monitoring_results = analyzer.analyze(
                    batch=batch,
                    outputs=eval_outputs.logits,
                    tokenizer=tokenizer,
                    step=step
                )
                
                # Log key metrics
                if 'pos_accuracy' in monitoring_results:
                    avg_pos_acc = np.mean(list(monitoring_results['pos_accuracy'].values()))
                    print(f"Step {step}: Average POS accuracy = {avg_pos_acc:.3f}")
                
                if 'semantic_accuracy' in monitoring_results:
                    avg_sem_acc = np.mean(list(monitoring_results['semantic_accuracy'].values()))
                    print(f"Step {step}: Average semantic accuracy = {avg_sem_acc:.3f}")
            
            model.train()  # Switch back to training
    
    # Generate final analysis
    print("\n=== Final Output Monitoring Summary ===")
    
    # POS analysis summary
    pos_summary = analyzer.get_pos_summary()
    if pos_summary:
        print(f"\nPOS Analysis ({pos_summary['total_steps']} steps):")
        print("Hardest POS categories:")
        for category, accuracy in pos_summary['hardest_categories']:
            print(f"  {category}: {accuracy:.3f}")
        print("Easiest POS categories:")
        for category, accuracy in pos_summary['easiest_categories']:
            print(f"  {category}: {accuracy:.3f}")
    
    # Semantic analysis summary  
    semantic_summary = analyzer.get_semantic_summary()
    if semantic_summary:
        print(f"\nSemantic Analysis ({semantic_summary['total_steps']} steps):")
        print("Hardest semantic categories:")
        for category, accuracy in semantic_summary['hardest_categories']:
            print(f"  {category}: {accuracy:.3f}")
        print("Easiest semantic categories:")
        for category, accuracy in semantic_summary['easiest_categories']:
            print(f"  {category}: {accuracy:.3f}")
    
    return analyzer

# Run training with monitoring
final_analyzer = complete_training_with_output_monitoring()
```

### Advanced Analysis and Visualization

```python
from trace.output_monitoring import OutputMonitoringVisualizer

# Create visualizer
visualizer = OutputMonitoringVisualizer(
    log_dir='./output_visualizations',
    config=config
)

# Collect monitoring data over training
monitoring_history = {}
for step in range(0, 1000, 50):  # Every 50 steps
    # Get monitoring results for this step
    results = analyzer.analyze(batch, outputs, tokenizer, step)
    monitoring_history[step] = results

# Generate evolution plots
visualizer.plot_pos_performance_evolution(
    monitoring_results=monitoring_history,
    model_name="transformer_decoder",
    save_plot=True
)

visualizer.plot_semantic_role_performance_evolution(
    monitoring_results=monitoring_history,
    model_name="transformer_decoder", 
    save_plot=True
)

# Save detailed metrics to CSV
visualizer.save_metrics(
    monitoring_results=monitoring_history,
    model_name="transformer_decoder"
)
```

### Granularity Examples

The monitoring system supports different levels of linguistic analysis:

```python
# Basic granularity - simplified categories
basic_pos_categories = {
    "NOUN": 0, "VERB": 1, "ADJ": 2, "ADV": 3, 
    "PREP": 4, "DET": 5, "CONJ": 6, "OTHER": 7
}

basic_semantic_categories = {
    "AGENT": 0, "PATIENT": 1, "ACTION": 2, "LOCATION": 3,
    "RELATION": 4, "CONNECTOR": 5, "RESULT": 6, "OTHER": 7
}

# Detailed granularity - fine-grained categories
detailed_pos_categories = {
    "NOUN": 0, "TRANSITIVE_VERB": 1, "INTRANSITIVE_VERB": 2,
    "COMMUNICATION_VERB": 3, "MOTION_VERB": 4, "CHANGE_VERB": 5,
    "ADJ": 6, "ADV": 7, "LOCATION": 8, "TEMP": 9,
    "PREP": 10, "RESULT": 11, "CONJ": 12, "OTHER": 13
}

detailed_semantic_categories = {
    "AGENT": 0, "PATIENT": 1, "ACTION": 2, "MOTION": 3,
    "COMMUNICATION": 4, "CHANGE": 5, "LOCATION": 6, "DESTINATION": 7,
    "TIME": 8, "RESULT": 9, "PROPERTY": 10, "MANNER": 11,
    "RELATION": 12, "CONNECTOR": 13, "OTHER": 14
}

# Configure for specific granularity
detailed_config = OutputMonitoringConfig(
    pos_granularity='detailed',
    semantic_granularity='detailed'
)
```

### Performance Tracking

Track detailed performance metrics:

```python
# Get comprehensive performance statistics
def analyze_output_performance(analyzer):
    # Get summaries
    pos_summary = analyzer.get_pos_summary()
    semantic_summary = analyzer.get_semantic_summary()
    
    # Analyze trends
    if pos_summary:
        print("=== POS Performance Analysis ===")
        
        # Overall statistics
        total_categories = len(pos_summary['average_accuracies'])
        avg_accuracy = np.mean(list(pos_summary['average_accuracies'].values()))
        print(f"Total POS categories tracked: {total_categories}")
        print(f"Average POS accuracy: {avg_accuracy:.3f}")
        
        # Identify problematic categories
        hardest = pos_summary['hardest_categories']
        print(f"\nMost challenging POS categories:")
        for category, acc in hardest:
            sample_count = pos_summary['sample_counts'].get(category, 0)
            print(f"  {category}: {acc:.3f} (n={sample_count})")
        
        # Best performing categories
        easiest = pos_summary['easiest_categories']
        print(f"\nBest performing POS categories:")
        for category, acc in easiest:
            sample_count = pos_summary['sample_counts'].get(category, 0)
            print(f"  {category}: {acc:.3f} (n={sample_count})")
    
    # Similar analysis for semantic roles
    if semantic_summary:
        print("\n=== Semantic Role Performance Analysis ===")
        
        total_categories = len(semantic_summary['average_accuracies'])
        avg_accuracy = np.mean(list(semantic_summary['average_accuracies'].values()))
        print(f"Total semantic categories tracked: {total_categories}")
        print(f"Average semantic accuracy: {avg_accuracy:.3f}")
        
        # Analysis continues...

# Run performance analysis
analyze_output_performance(analyzer)
```

### Real-time Monitoring Dashboard

For continuous monitoring during training:

```python
def create_monitoring_dashboard(analyzer, update_frequency=50):
    """Create a simple monitoring dashboard that updates during training."""
    
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    import time
    
    def update_dashboard(step, results):
        clear_output(wait=True)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # POS accuracy plot
        if 'pos_accuracy' in results:
            pos_data = results['pos_accuracy']
            categories = list(pos_data.keys())
            accuracies = list(pos_data.values())
            
            ax1.bar(categories, accuracies)
            ax1.set_title(f'POS Accuracy at Step {step}')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
        
        # Semantic accuracy plot
        if 'semantic_accuracy' in results:
            semantic_data = results['semantic_accuracy']
            categories = list(semantic_data.keys())
            accuracies = list(semantic_data.values())
            
            ax2.bar(categories, accuracies, color='orange')
            ax2.set_title(f'Semantic Accuracy at Step {step}')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    return update_dashboard

# Usage in training loop
dashboard = create_monitoring_dashboard(analyzer)

for step, batch in enumerate(train_dataloader):
    # Training step...
    
    if step % 50 == 0:
        results = analyzer.analyze(batch, outputs, tokenizer, step)
        dashboard(step, results)
```

### Comparison with Linguistic Probes

Output monitoring complements linguistic probes by analyzing different aspects:

```python
# Comprehensive linguistic analysis combining both approaches
def comprehensive_linguistic_analysis():
    # 1. Linguistic Probes - analyze internal representations
    from trace.linguistic_probes import POSAnalyzer, SemanticAnalyzer, LinguisticProbesConfig
    
    probe_config = LinguisticProbesConfig(
        layer_indices={'decoder': [0, 6, 11]},
        save_visualizations=True
    )
    
    pos_probe_analyzer = POSAnalyzer(probe_config)
    # Load pre-trained probes...
    
    # 2. Output Monitoring - analyze generated text
    output_config = OutputMonitoringConfig(
        track_pos_performance=True,
        track_semantic_roles=True,
        save_visualizations=True
    )
    
    output_analyzer = OutputMonitoringAnalyzer(output_config)
    
    # 3. Combined analysis during training
    for step, batch in enumerate(train_dataloader):
        # Training step...
        
        if step % 100 == 0:
            # Probe analysis - what the model knows internally
            probe_results = pos_probe_analyzer.analyze(
                model, eval_dataloader, tokenizer
            )
            
            # Output analysis - how well it generates
            output_results = output_analyzer.analyze(
                batch, outputs, tokenizer, step
            )
            
            # Compare internal knowledge vs generation ability
            print(f"\nStep {step} Comparison:")
            print("Internal POS understanding (probes):")
            for layer, confidences in probe_results.items():
                avg_confidence = np.mean(list(confidences.values()))
                print(f"  Layer {layer}: {avg_confidence:.3f}")
            
            print("POS generation accuracy (output):")
            if 'pos_accuracy' in output_results:
                avg_accuracy = np.mean(list(output_results['pos_accuracy'].values()))
                print(f"  Generated text: {avg_accuracy:.3f}")

# This reveals gaps between internal understanding and generation ability
comprehensive_linguistic_analysis()
```python
from trace.transformer import Transformer, TransformerConfig
import torch

# All models use the same configuration class
config = TransformerConfig(
    model_type="encoder_only",        # Architecture type
    vocab_size=30522,                 # Vocabulary size
    d_model=768,                      # Model dimension
    num_heads=12,                     # Attention heads
    num_encoder_layers=6,             # Encoder layers (if applicable)
    num_decoder_layers=6,             # Decoder layers (if applicable)
    d_ff=3072,                        # Feed-forward dimension
    max_seq_length=512,               # Maximum sequence length
    dropout=0.1,                      # Dropout rate
    device="cpu"                      # Device placement
)
```

### Encoder-Only Models

Perfect for classification, encoding, and representation learning tasks:

```python
# Create an encoder-only model (BERT-style)
config = TransformerConfig(
    model_type="encoder_only",
    vocab_size=30522,
    d_model=768,
    num_heads=12,
    num_encoder_layers=6,
    d_ff=3072,
    max_seq_length=512,
    dropout=0.1
)

encoder_model = Transformer.from_config(config)

# Forward pass
input_ids = torch.randint(0, 30522, (2, 20))  # [batch_size, seq_len]
hidden_states = encoder_model(src=input_ids)
print(f"Output shape: {hidden_states.shape}")  # [2, 20, 768]
```

### Encoder-Decoder Models

Ideal for translation, summarization, and other sequence-to-sequence tasks:

```python
# Create an encoder-decoder model (T5/BART-style)
config = TransformerConfig(
    model_type="encoder_decoder",
    vocab_size=32128,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_length=512,
    dropout=0.1
)

encoder_decoder_model = Transformer.from_config(config)

# Forward pass with source and target sequences
src_ids = torch.randint(0, 32128, (2, 15))  # Source sequence
tgt_ids = torch.randint(0, 32128, (2, 10))  # Target sequence
logits = encoder_decoder_model(src=src_ids, tgt=tgt_ids)
print(f"Output shape: {logits.shape}")  # [2, 10, 32128]
```

### Decoder-Only Models

Optimized for autoregressive generation and language modeling:

```python
# Create a decoder-only model (GPT-style)
config = TransformerConfig(
    model_type="decoder_only",
    vocab_size=2000,
    d_model=768,
    num_heads=12,
    num_decoder_layers=12,
    d_ff=3072,
    max_seq_length=64,
    dropout=0.1,
    device="cpu"
)

decoder_model = Transformer.from_config(config)

# Forward pass for autoregressive generation
input_ids = torch.randint(0, 2000, (2, 32))  # [batch_size, seq_len]
logits = decoder_model(tgt=input_ids)
print(f"Output shape: {logits.shape}")  # [2, 32, 2000]
```

### Advanced Configuration Options

```python
# Specialized configurations
config = TransformerConfig(
    model_type="decoder_only",
    vocab_size=2000,
    d_model=768,
    num_heads=12,
    num_decoder_layers=1,            # Single layer for analysis
    d_ff=3072,
    max_seq_length=64,
    dropout=0.1,
    device="cpu",
    # no_ffn=True                    # Disable feed-forward networks (optional)
)

# Create model with custom settings
specialized_model = Transformer.from_config(config)
```

### Model Information and Utilities

```python
# Get model information
print(f"Model type: {decoder_model.config.model_type}")
print(f"Parameters: {sum(p.numel() for p in decoder_model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in decoder_model.parameters() if p.requires_grad):,}")

# Access model components
print(f"Number of layers: {len(decoder_model.layers)}")
print(f"Model dimension: {decoder_model.config.d_model}")
```

## 📐 Intrinsic Dimensions Analysis

TRACE provides comprehensive tools for analyzing the intrinsic dimensionality of transformer representations using advanced geometric methods. This helps understand how models compress and organize information across layers.

### Supported Methods

- **TwoNN**: Two Nearest Neighbors method for robust ID estimation
- **MLE**: Maximum Likelihood Estimation approach
- **PCA**: Principal Component Analysis for linear dimensionality

### Basic Configuration

```python
from trace.intrinsic_dimensions import IntrinsicDimensionsConfig, IntrinsicDimensionAnalyzer

# Create configuration for intrinsic dimensions analysis
config = IntrinsicDimensionsConfig(
    model_type="decoder_only",           # Model architecture
    id_method="TwoNN",                   # Dimensionality estimation method
    layers_to_analyze=None,              # None = analyze all layers
    max_samples=1000,                    # Limit samples for efficiency
    flatten_sequence=True,               # Flatten sequence dimension
    save_visualizations=True,            # Generate plots
    show_plots=False,                    # Display plots immediately
    log_dir="./plots/intrinsic_dims"     # Output directory
)

# Initialize analyzer
analyzer = IntrinsicDimensionAnalyzer(config)
```

### Layer-Specific Analysis

```python
# Analyze specific layers for encoder-only models
config = IntrinsicDimensionsConfig(
    model_type="encoder_only",
    layers_to_analyze={'encoder': [0, 3, 5]},  # Analyze layers 0, 3, and 5
    id_method="TwoNN"
)

# For encoder-decoder models
config = IntrinsicDimensionsConfig(
    model_type="encoder_decoder",
    layers_to_analyze={
        'encoder': [0, 2, 4],
        'decoder': [0, 1, 3]
    }
)

# For decoder-only models (simple list notation)
config = IntrinsicDimensionsConfig(
    model_type="decoder_only",
    layers_to_analyze=[0, 6, 11],  # First, middle, and last layers
    id_method="MLE"
)
```

### Complete Analysis Workflow

```python
from trace.intrinsic_dimensions import IntrinsicDimensionAnalyzer
from torch.utils.data import DataLoader

# Prepare your data
# data_loader = DataLoader(your_dataset, batch_size=32, shuffle=False)

# Create analyzer with default settings
analyzer = IntrinsicDimensionAnalyzer()

# Run comprehensive analysis
intrinsic_dimensions = analyzer.analyze(
    model=decoder_model,
    data_loader=data_loader,
    layers=[0, 3, 6, 9, 11],  # Specific layers to analyze
    model_name="my_transformer"
)

# Results dictionary: {(layer_idx, layer_type): intrinsic_dimension}
print("Intrinsic Dimensions Results:")
for (layer_idx, layer_type), id_value in intrinsic_dimensions.items():
    print(f"  Layer {layer_idx} ({layer_type}): {id_value:.2f}")
```

### Manual Hidden State Extraction

```python
from trace.intrinsic_dimensions import (
    extract_hidden_representations,
    compute_intrinsic_dimensions,
    average_intrinsic_dimension
)

# Extract hidden states from specific layers
hidden_states, _, _ = extract_hidden_representations(
    model=encoder_model,
    dataloader=data_loader,
    layer_indices={'encoder': [0, 2, 4]},
    device='cuda'
)

# Compute intrinsic dimensions
config = IntrinsicDimensionsConfig(id_method="TwoNN")
intrinsic_dims = compute_intrinsic_dimensions(hidden_states, config)

# Calculate average across layers
avg_id = average_intrinsic_dimension(
    intrinsic_dims,
    layer_indices=[(0, 'encoder'), (2, 'encoder'), (4, 'encoder')]
)
print(f"Average intrinsic dimension: {avg_id:.2f}")
```

### Advanced Visualization Options

```python
from trace.intrinsic_dimensions import IntrinsicDimensionsVisualizer

# Create custom visualizer
visualizer = IntrinsicDimensionsVisualizer(
    log_dir="./custom_plots",
    config=config
)

# Generate comprehensive visualizations
visualizer.generate_all_visualizations(
    intrinsic_dimensions=intrinsic_dims,
    model_name="transformer_analysis",
    show_plots=True  # Display plots immediately
)

# Individual plot types
visualizer.plot_id_by_layer(intrinsic_dims, "my_model", save_plot=True)
visualizer.plot_id_distribution(intrinsic_dims, "my_model", save_plot=True)
visualizer.plot_final_id(intrinsic_dims, "my_model", save_plot=True)

# Save metrics to CSV
visualizer.save_metrics(intrinsic_dims, "my_model")
```

### Method Comparison

```python
# Compare different ID estimation methods
methods = ["TwoNN", "MLE", "PCA"]
results_comparison = {}

for method in methods:
    config = IntrinsicDimensionsConfig(
        id_method=method,
        layers_to_analyze=[0, 6, 11]
    )
    analyzer = IntrinsicDimensionAnalyzer(config)
    results_comparison[method] = analyzer.analyze(
        model=decoder_model,
        data_loader=data_loader,
        model_name=f"model_{method.lower()}"
    )

# Compare results
for method, results in results_comparison.items():
    avg_id = average_intrinsic_dimension(results)
    print(f"{method} average ID: {avg_id:.2f}")
```

### Integration with Training Loop

```python
# Monitor intrinsic dimensions during training
def training_step_with_id_analysis(model, data_loader, step):
    # Your training code here...
    
    # Periodic intrinsic dimension analysis
    if step % 100 == 0:  # Analyze every 100 steps
        analyzer = IntrinsicDimensionAnalyzer(
            config=IntrinsicDimensionsConfig(
                layers_to_analyze=[0, -1],  # First and last layers
                save_visualizations=True,
                log_dir=f"./training_analysis/step_{step}"
            )
        )
        
        # Quick analysis on subset of data
        subset_loader = DataLoader(
            dataset=data_loader.dataset,
            batch_size=32,
            shuffle=False,
            sampler=torch.utils.data.SubsetRandomSampler(range(100))
        )
        
        id_results = analyzer.analyze(
            model=model,
            data_loader=subset_loader,
            model_name=f"training_step_{step}"
        )
        
        # Log results
        avg_id = average_intrinsic_dimension(id_results)
        print(f"Step {step}: Average ID = {avg_id:.2f}")
```

## 🌊 Hessian Analysis

TRACE provides comprehensive Hessian analysis capabilities for understanding loss landscape properties, training dynamics, and memorization patterns. This module analyzes curvature information to reveal insights about model optimization and generalization.

### Key Features

- **Eigenvalue Analysis**: Track extreme eigenvalues, trace estimates, and spectral properties
- **Component-Specific Analysis**: Analyze individual model components (attention, FFN, embeddings)
- **Gradient-Hessian Alignment**: Monitor optimization direction relative to curvature
- **Memorization Detection**: Compare train/validation landscapes to detect overfitting
- **Real-time Monitoring**: Integration with training loops for continuous analysis

### Basic Configuration

```python
from trace.hessian import HessianConfig, HessianAnalyzer

# Create configuration for Hessian analysis
config = HessianConfig(
    n_components=10,                           # Number of eigenvalues to compute
    num_batches=100,                          # Batches for Hessian estimation
    device="cuda",                            # Device for computation
    
    # Analysis toggles
    track_component_hessian=True,             # Analyze individual components
    track_gradient_alignment=True,            # Monitor gradient-Hessian alignment
    track_train_val_landscape_divergence=True, # Detect memorization signals
    
    # Component selection
    component_list=["attention", "ffn", "hidden_states"],
    
    # Output settings
    log_dir="./hessian_analysis",             # Output directory
    save_hessian_data=True,                   # Save raw data
    show_plots=False                          # Display plots immediately
)

# Initialize analyzer
analyzer = HessianAnalyzer(config)
```

### Preset Configurations

```python
# Minimal configuration for basic analysis
minimal_config = HessianConfig.minimal(
    n_components=5,
    track_component_hessian=False
)

# Comprehensive configuration for detailed research
comprehensive_config = HessianConfig.comprehensive(
    n_components=20,
    component_list=[
        "attention", "attention_query", "attention_key", "attention_value",
        "ffn", "embeddings", "norm", "hidden_states", "output_projection"
    ]
)

# Default balanced configuration
default_config = HessianConfig.default()
```

### Single-Step Analysis

```python
import torch.nn as nn

# Prepare loss function and data
loss_fn = nn.CrossEntropyLoss()
# train_batch and val_batch should be prepared with your data

# Perform comprehensive analysis at a single training step
results = analyzer.analyze_step(
    model=decoder_model,
    loss_fn=loss_fn,
    train_batch=train_batch,
    val_batch=val_batch,              # Optional for memorization analysis
    model_type="decoder_only",
    step=100
)

# Access results
print(f"Max eigenvalue: {results['hessian']['max_eigenvalue']:.2e}")
print(f"Min eigenvalue: {results['hessian']['min_eigenvalue']:.2e}")
print(f"Trace estimate: {results['hessian']['hessian_trace_estimate']:.2e}")
print(f"Negative eigenvalues: {results['hessian']['negative_count']}")
print(f"Effective rank: {results['hessian']['effective_rank_95']}")
```

### Component-Specific Analysis

```python
from trace.hessian import ComponentAnalyzer, ComponentSelector

# Initialize component analyzer
component_analyzer = ComponentAnalyzer()

# Get appropriate components for your model
no_ffn = getattr(decoder_model, 'no_ffn', False)
components = ComponentSelector.get_standard_components(no_ffn)
print(f"Analyzing components: {components}")

# Analyze all components
component_results = component_analyzer.analyze_all_components(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    component_list=components,
    n_components=10
)

# Compare component complexity
for component, metrics in component_results.items():
    if 'error' not in metrics:
        print(f"{component}:")
        print(f"  Parameters: {metrics['num_params']:,}")
        print(f"  Max eigenvalue: {metrics['max_eigenvalue']:.2e}")
        print(f"  Effective rank: {metrics['effective_rank_95']}")
```

### Advanced Component Selection

```python
# Validate components exist in your model
valid_components = ComponentSelector.validate_components(
    model=decoder_model,
    component_list=["attention", "ffn", "embeddings", "norm"]
)

# Use comprehensive component set for detailed analysis
comprehensive_components = ComponentSelector.get_comprehensive_components()

# Custom component analysis
custom_analyzer = ComponentAnalyzer()
attention_results = custom_analyzer.analyze_component(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    component_name="attention",
    n_components=15
)
```

### Gradient-Hessian Alignment Analysis

```python
# Analyze optimization dynamics through gradient-curvature relationships
alignment_results = analyzer.compute_gradient_alignment(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    eigenvalues=eigenvalues,  # From previous Hessian computation
    eigenvectors=eigenvectors
)

# Key alignment metrics
print(f"Gradient-Hessian alignment: {alignment_results['grad_Hg_alignment']:.4f}")
print(f"Weighted alignment score: {alignment_results['weighted_alignment']:.4f}")
print(f"Curvature/gradient ratio: {alignment_results['grad_Hg_ratio']:.4f}")
```

### Memorization Detection

```python
# Detect memorization through train/validation landscape comparison
memorization_signals = analyzer.compute_train_val_divergence(
    model=decoder_model,
    loss_fn=loss_fn,
    train_batch=train_batch,
    val_batch=val_batch,
    model_type="decoder_only"
)

# Memorization indicators
print(f"Landscape divergence score: {memorization_signals['train_val_landscape_divergence_score']:.4f}")
print(f"Trace ratio (train/val): {memorization_signals['trace_ratio']:.4f}")
print(f"Eigenvalue distribution overlap: {memorization_signals['eigenvalue_distribution_overlap']:.4f}")
print(f"Effective rank difference: {memorization_signals['effective_rank_diff']}")
```

### Low-Level Utilities

```python
from trace.hessian import (
    compute_loss,
    extract_component_parameters,
    get_hessian_eigenvectors
)

# Manual loss computation with proper model handling
loss = compute_loss(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    ignore_index=-100
)

# Extract parameters for specific components
attention_params = extract_component_parameters(
    model=decoder_model,
    component_name="attention",
    include_bias=False
)
print(f"Attention parameters: {sum(p.numel() for p in attention_params):,}")

# Manual Hessian eigenvalue computation
eigenvalues, eigenvectors = get_hessian_eigenvectors(
    model=decoder_model,
    loss_fn=loss_fn,
    train_data_loader=train_batch,
    num_batches=50,
    device="cuda",
    n_top_vectors=15
)
```

### Comprehensive Visualization

```python
from trace.hessian import HessianVisualizer

# Initialize visualizer
visualizer = HessianVisualizer(config)

# Track analysis over multiple steps
hessian_history = {}
for step in range(0, 1000, 100):
    # Perform analysis at each step
    results = analyzer.analyze_step(
        model=decoder_model,
        loss_fn=loss_fn,
        train_batch=train_batch,
        val_batch=val_batch,
        step=step
    )
    hessian_history[step] = results

# Generate comprehensive visualization report
visualizer.create_comprehensive_report(
    hessian_history=hessian_history,
    model_name="my_transformer"
)

# Individual plot types
visualizer.plot_eigenvalue_evolution(hessian_history, "my_transformer")
visualizer.plot_gradient_alignment(hessian_history, "my_transformer") 
visualizer.plot_component_comparison(hessian_history, "my_transformer")
visualizer.plot_memorization_metrics(hessian_history, model_name="my_transformer")
```

### Training Loop Integration

```python
def training_step_with_hessian_analysis(model, optimizer, train_batch, val_batch, step):
    # Regular training step
    optimizer.zero_grad()
    loss = compute_loss(model, loss_fn, train_batch, "decoder_only")
    loss.backward()
    optimizer.step()
    
    # Periodic Hessian analysis
    if step % 50 == 0:  # Analyze every 50 steps
        # Create step-specific analyzer
        step_analyzer = HessianAnalyzer(HessianConfig.default(
            log_dir=f"./analysis/step_{step}",
            n_components=8,  # Reduced for speed
            track_train_val_landscape_divergence=(val_batch is not None)
        ))
        
        # Perform analysis
        hessian_results = step_analyzer.analyze_step(
            model=model,
            loss_fn=loss_fn,
            train_batch=train_batch,
            val_batch=val_batch,
            step=step
        )
        
        # Log key metrics
        if 'hessian' in hessian_results:
            h = hessian_results['hessian']
            print(f"Step {step}: "
                  f"Max λ: {h['max_eigenvalue']:.2e}, "
                  f"Trace: {h['hessian_trace_estimate']:.2e}, "
                  f"Neg count: {h['negative_count']}")
        
        # Check for memorization signals
        if 'train_val_divergence' in hessian_results:
            div_score = hessian_results['train_val_divergence']['train_val_landscape_divergence_score']
            if div_score > 0.5:  # Threshold for concern
                print(f"⚠️  High memorization signal detected: {div_score:.3f}")
    
    return loss.item()
```

### Performance Optimization

```python
# Efficient configuration for large models
efficient_config = HessianConfig(
    n_components=5,                    # Fewer eigenvalues
    num_batches=50,                    # Fewer batches
    track_component_hessian=False,     # Skip component analysis
    track_gradient_alignment=False,    # Skip alignment analysis
    track_train_val_landscape_divergence=False  # Skip memorization detection
)

# Memory-efficient analysis for very large models
def memory_efficient_hessian_analysis(model, data_batch, step):
    # Use CPU for Hessian computation to save GPU memory
    cpu_config = HessianConfig.minimal(device="cpu", n_components=3)
    analyzer = HessianAnalyzer(cpu_config)
    
    # Move only necessary data to CPU
    cpu_batch = {k: v.cpu() for k, v in data_batch.items()}
    model_cpu = model.cpu()
    
    results = analyzer.analyze_step(
        model=model_cpu,
        loss_fn=nn.CrossEntropyLoss(),
        train_batch=cpu_batch,
        step=step
    )
    
    # Move model back to GPU
    model.cuda()
    return results
```

## 🔍 Linguistic Probes

TRACE provides sophisticated linguistic probing capabilities to analyze what linguistic knowledge models acquire during training. This module includes both probe training infrastructure and real-time monitoring systems for tracking linguistic understanding.

### Key Features

- **Multi-label Probing**: Train sophisticated probes to detect multiple linguistic features simultaneously
- **POS Analysis**: Track part-of-speech understanding (basic and detailed granularity)
- **Semantic Role Analysis**: Monitor semantic role labeling capabilities
- **Real-time Monitoring**: Use pre-trained probes to track linguistic understanding during training
- **Flexible Tagging**: Rule-based taggers for synthetic and natural text
- **Performance Tracking**: Category-specific performance monitoring

### Two-Phase Workflow

The linguistic probes module follows a two-phase workflow:

1. **Training Phase**: Train probes on your model to establish linguistic understanding baselines
2. **Monitoring Phase**: Use trained probes to monitor linguistic capabilities during model training

### Phase 1: Training Probes

First, you need to train probes on your model to establish what linguistic features it has learned:

```python
from trace.linguistic_probes import LinguisticProbesConfig, ProbeTrainer

# Configure probe training
config = LinguisticProbesConfig(
    # Probe architecture
    probe_type='multilabel',           # 'linear' or 'multilabel'
    hidden_dim=128,                    # Hidden dimension for MLPs
    
    # Training parameters
    epochs=3,                          # Training epochs for probes
    lr=0.001,                         # Learning rate
    batch_size=64,                    # Batch size for probe training
    device="cuda",                    # Device for computation
    
    # What to analyze
    track_pos=True,                   # Train POS probes
    track_semantic=True,              # Train semantic role probes
    pos_granularity='basic',          # 'basic' or 'detailed'
    semantic_granularity='basic',     # 'basic' or 'detailed'
    
    # Layer selection (None = all layers)
    layer_indices=None,
    
    # Output settings
    save_probes=True,                 # Save trained probes
    save_visualizations=False,         # Create training visualizations
    save_path='./trained_probes',     # Where to save probes
    log_dir="./probe_training_logs",  # Training logs directory
)

# Initialize trainer
trainer = ProbeTrainer(config, tokenizer)

# Train probes on your model
training_results = trainer.train_all_probes(
    model=decoder_model,
    dataloader=your_training_dataloader
)

# View training results
for layer_key, layer_results in training_results.items():
    print(f"\nLayer {layer_key}:")
    if 'pos' in layer_results:
        pos_acc = layer_results['pos']['accuracy']
        print(f"  POS accuracy: {pos_acc:.3f}")
    if 'semantic' in layer_results:
        sem_acc = layer_results['semantic']['accuracy']
        print(f"  Semantic accuracy: {sem_acc:.3f}")
```

### Granularity Options

Configure the level of linguistic analysis detail:

```python
# Basic granularity - simplified categories
basic_config = LinguisticProbesConfig(
    pos_granularity='basic',           # NOUN, VERB, ADJ, ADV, PREP, DET, CONJ, OTHER
    semantic_granularity='basic'       # AGENT, PATIENT, ACTION, LOCATION, RELATION, CONNECTOR, RESULT, OTHER
)

# Detailed granularity - fine-grained categories  
detailed_config = LinguisticProbesConfig(
    pos_granularity='detailed',        # Includes TRANSITIVE_VERB, COMMUNICATION_VERB, etc.
    semantic_granularity='detailed'    # Includes MOTION, COMMUNICATION, DESTINATION, etc.
)

# View available categories
print("POS categories:", basic_config.get_pos_categories())
print("Semantic categories:", basic_config.get_semantic_categories())
```

### Preset Training Configurations

```python
# Quick training for experimentation
quick_config = LinguisticProbesConfig.minimal(
    epochs=2,
    batch_size=32,
    hidden_dim=64,
    save_probes=True
)

# Comprehensive analysis
comprehensive_config = LinguisticProbesConfig.comprehensive(
    epochs=5,
    hidden_dim=256,
    pos_granularity='detailed',
    semantic_granularity='detailed',
    save_visualizations=True
)

# POS-only training
pos_only_config = LinguisticProbesConfig.pos_only(
    track_semantic=False,
    pos_granularity='detailed'
)
```

### Phase 2: Real-time Monitoring

Once you have trained probes, use them to monitor linguistic understanding during training:

```python
from trace.linguistic_probes import POSAnalyzer, SemanticAnalyzer

# Configure monitoring (note: different config than training)
monitoring_config = LinguisticProbesConfig(
    track_pos=True,
    track_semantic=True,
    layer_indices={'decoder': [0, 6, 11]},  # Monitor specific layers
    log_dir="./training_monitoring",
    save_visualizations=True,
    
    # Specify where to load trained probes from
    probe_load_path={
        (0, 'decoder'): './trained_probes/pos_layer0_decoder.pt',
        (6, 'decoder'): './trained_probes/pos_layer6_decoder.pt',
        (11, 'decoder'): './trained_probes/pos_layer11_decoder.pt',
    }
)

# Initialize analyzers
pos_analyzer = POSAnalyzer(monitoring_config)
semantic_analyzer = SemanticAnalyzer(monitoring_config)

# Load pre-trained probes
probe_paths = {
    (0, 'decoder'): './trained_probes/pos_layer0_decoder.pt',
    (6, 'decoder'): './trained_probes/pos_layer6_decoder.pt',
    (11, 'decoder'): './trained_probes/pos_layer11_decoder.pt',
}
pos_analyzer.load_probes(probe_paths)

# Monitor during training step
def training_step_with_linguistic_monitoring(model, batch, step):
    # Your regular training code here...
    
    if step % 50 == 0:  # Monitor every 50 steps
        # Get confidence scores from pre-trained probes
        pos_confidences = pos_analyzer.analyze(
            model=model,
            dataloader=DataLoader([batch]),  # Single batch
            tokenizer=tokenizer,
            model_name=f"step_{step}"
        )
        
        # pos_confidences structure:
        # {(layer_idx, layer_type): {'NOUN': 0.85, 'VERB': 0.72, ...}}
        
        for layer_key, confidences in pos_confidences.items():
            print(f"Step {step}, Layer {layer_key}:")
            for tag, confidence in confidences.items():
                print(f"  {tag}: {confidence:.3f}")
```

### Advanced Probe Training

For more control over the training process:

```python
# Train probes for specific model components
from trace.linguistic_probes.models import MultiLabelProbe

# Manual probe configuration
manual_config = LinguisticProbesConfig(
    input_dim=768,                    # Must match model hidden size
    num_classes=8,                    # Number of linguistic categories
    probe_type='multilabel',
    hidden_dim=128,
    dropout=0.1,
    use_class_weights=True,           # Handle class imbalance
    penalize_frequent_classes=True,   # Reduce weight of frequent classes
    penalized_classes=[0, 1],         # Typically NOUN and VERB
    criterion='cross_entropy'
)

# Create and train individual probe
probe = MultiLabelProbe(
    input_dim=768,
    config=manual_config
)

# Custom training loop
for epoch in range(manual_config.epochs):
    for batch in probe_dataloader:
        # Your custom training logic here
        pass
```

### Linguistic Taggers

TRACE includes sophisticated rule-based taggers for synthetic data:

```python
from trace.linguistic_probes import POSTagger, SemanticTagger

# Initialize taggers
pos_tagger = POSTagger(
    granularity='detailed',
    use_nltk_fallback=True  # Fallback to NLTK for unknown tokens
)

semantic_tagger = SemanticTagger(granularity='detailed')

# Tag synthetic text
text = "noun_entity transitive_verb_action noun_target preposition_to location_place"
pos_tags = pos_tagger.tag_text(text)
semantic_tags = semantic_tagger.tag_text(text)

print("POS tags:", pos_tags)
print("Semantic tags:", semantic_tags)

# Output:
# POS tags: [('noun_entity', 'NOUN'), ('transitive_verb_action', 'TRANSITIVE_VERB'), ...]
# Semantic tags: [('noun_entity', 'AGENT'), ('transitive_verb_action', 'ACTION'), ...]
```

### Performance Tracking

Track linguistic performance by category during training:

```python
from trace.linguistic_probes import POSPerformanceTracker, SemanticPerformanceTracker

# Initialize trackers
pos_tracker = POSPerformanceTracker(
    tokenizer=tokenizer,
    log_dir="./performance_tracking",
    config=LinguisticProbesConfig(pos_granularity='basic')
)

semantic_tracker = SemanticPerformanceTracker(
    tokenizer=tokenizer,
    log_dir="./performance_tracking",
    config=LinguisticProbesConfig(semantic_granularity='basic')
)

# Use during training
def training_step_with_performance_tracking(model, batch, step):
    # Forward pass
    outputs = model(**batch)
    logits = outputs.logits
    
    # Process batch for linguistic performance metrics
    pos_metrics = pos_tracker.process_batch(batch, logits)
    semantic_metrics = semantic_tracker.process_batch(batch, logits)
    
    # Update trackers
    pos_tracker.update_epoch_metrics(pos_metrics, step)
    semantic_tracker.update_epoch_metrics(semantic_metrics, step)
    
    # Print category-specific accuracies
    if step % 100 == 0:
        for category, accuracy in pos_metrics['pos_analysis_accuracy'].items():
            print(f"POS {category}: {accuracy:.3f}")
```

### Comprehensive Visualization

Generate detailed analysis visualizations:

```python
from trace.linguistic_probes import ProbesVisualizer

# Initialize visualizer
visualizer = ProbesVisualizer(
    log_dir="./probe_visualizations",
    config=monitoring_config
)

# Track confidence over training
confidence_history = {}
for step in range(0, 1000, 50):
    # Get confidence scores at this step
    pos_confidences = pos_analyzer.analyze(model, eval_dataloader, tokenizer)
    confidence_history[step] = pos_confidences

# Create comprehensive plots
visualizer.plot_probe_confidence_analysis(
    confidence_data=confidence_history,
    model_name="my_transformer",
    analysis_type="pos",
    save_plot=True,
    show_plots=False
)

# This creates:
# - Per-tag plots showing each tag's evolution across layers
# - Per-layer plots showing all tags for each layer  
# - Comprehensive plots showing all tags and layers together
```

### Legacy Compatibility Functions

For backward compatibility with existing code:

```python
from trace.linguistic_probes import (
    run_pos_probe_analysis,
    run_semantic_probe_analysis,
    run_comprehensive_probe_analysis
)

# Legacy analysis (for monitoring, not training)
pos_results = run_pos_probe_analysis(
    model=decoder_model,
    dataloader=test_dataloader,
    tokenizer=tokenizer,
    device="cuda",
    layer_indices=[0, 6, 11],
    config=monitoring_config,
    model_name="legacy_test"
)

# Comprehensive analysis
all_results = run_comprehensive_probe_analysis(
    model=decoder_model,
    dataloader=test_dataloader,
    tokenizer=tokenizer,
    device="cuda",
    layer_indices=[0, 6, 11],
    config=monitoring_config,
    model_name="comprehensive_test"
)

print("POS results:", all_results['pos'])
print("Semantic results:", all_results['semantic'])
```

### Training Loop Integration

Complete example of integrating linguistic probes into training:

```python
def complete_training_with_linguistic_analysis():
    # Step 1: Train probes on initial model
    print("Training linguistic probes...")
    probe_config = LinguisticProbesConfig.comprehensive(
        save_path='./trained_probes',
        save_probes=True
    )
    trainer = ProbeTrainer(probe_config, tokenizer)
    probe_results = trainer.train_all_probes(model, train_dataloader)
    
    # Step 2: Set up monitoring
    monitor_config = LinguisticProbesConfig(
        layer_indices={'decoder': [0, 6, 11]},
        log_dir="./linguistic_monitoring"
    )
    
    pos_analyzer = POSAnalyzer(monitor_config)
    pos_analyzer.load_probes({
        (0, 'decoder'): './trained_probes/pos_layer0_decoder.pt',
        (6, 'decoder'): './trained_probes/pos_layer6_decoder.pt', 
        (11, 'decoder'): './trained_probes/pos_layer11_decoder.pt',
    })
    
    # Step 3: Training loop with monitoring
    confidence_history = {}
    for step in range(training_steps):
        # Regular training step
        loss = train_step(model, train_batch)
        
        # Linguistic monitoring every 50 steps
        if step % 50 == 0:
            confidences = pos_analyzer.analyze(
                model, eval_dataloader, tokenizer
            )
            confidence_history[step] = confidences
            
            # Log linguistic understanding changes
            for layer_key, layer_confidences in confidences.items():
                noun_conf = layer_confidences.get('NOUN', 0)
                verb_conf = layer_confidences.get('VERB', 0)
                print(f"Step {step}, Layer {layer_key}: "
                      f"NOUN={noun_conf:.3f}, VERB={verb_conf:.3f}")
    
    # Step 4: Generate final analysis
    visualizer = ProbesVisualizer("./final_analysis", monitor_config)
    visualizer.plot_probe_confidence_analysis(
        confidence_history, "final_model", "pos"
    )
    
    return confidence_history
```

## 🏗️ Transformer Model Creation

TRACE provides a flexible transformer implementation supporting multiple architectures with consistent configuration patterns.

### Supported Architectures

- **Encoder-Only**: BERT-style models for encoding tasks
- **Decoder-Only**: GPT-style models for autoregressive generation  
- **Encoder-Decoder**: T5/BART-style models for sequence-to-sequence tasks

### Configuration System

```python
from trace.transformer import Transformer, TransformerConfig
import torch

# All models use the same configuration class
config = TransformerConfig(
    model_type="encoder_only",        # Architecture type
    vocab_size=30522,                 # Vocabulary size
    d_model=768,                      # Model dimension
    num_heads=12,                     # Attention heads
    num_encoder_layers=6,             # Encoder layers (if applicable)
    num_decoder_layers=6,             # Decoder layers (if applicable)
    d_ff=3072,                        # Feed-forward dimension
    max_seq_length=512,               # Maximum sequence length
    dropout=0.1,                      # Dropout rate
    device="cpu"                      # Device placement
)
```

### Encoder-Only Models

Perfect for classification, encoding, and representation learning tasks:

```python
# Create an encoder-only model (BERT-style)
config = TransformerConfig(
    model_type="encoder_only",
    vocab_size=30522,
    d_model=768,
    num_heads=12,
    num_encoder_layers=6,
    d_ff=3072,
    max_seq_length=512,
    dropout=0.1
)

encoder_model = Transformer.from_config(config)

# Forward pass
input_ids = torch.randint(0, 30522, (2, 20))  # [batch_size, seq_len]
hidden_states = encoder_model(src=input_ids)
print(f"Output shape: {hidden_states.shape}")  # [2, 20, 768]
```

### Encoder-Decoder Models

Ideal for translation, summarization, and other sequence-to-sequence tasks:

```python
# Create an encoder-decoder model (T5/BART-style)
config = TransformerConfig(
    model_type="encoder_decoder",
    vocab_size=32128,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_seq_length=512,
    dropout=0.1
)

encoder_decoder_model = Transformer.from_config(config)

# Forward pass with source and target sequences
src_ids = torch.randint(0, 32128, (2, 15))  # Source sequence
tgt_ids = torch.randint(0, 32128, (2, 10))  # Target sequence
logits = encoder_decoder_model(src=src_ids, tgt=tgt_ids)
print(f"Output shape: {logits.shape}")  # [2, 10, 32128]
```

### Decoder-Only Models

Optimized for autoregressive generation and language modeling:

```python
# Create a decoder-only model (GPT-style)
config = TransformerConfig(
    model_type="decoder_only",
    vocab_size=2000,
    d_model=768,
    num_heads=12,
    num_decoder_layers=12,
    d_ff=3072,
    max_seq_length=64,
    dropout=0.1,
    device="cpu"
)

decoder_model = Transformer.from_config(config)

# Forward pass for autoregressive generation
input_ids = torch.randint(0, 2000, (2, 32))  # [batch_size, seq_len]
logits = decoder_model(tgt=input_ids)
print(f"Output shape: {logits.shape}")  # [2, 32, 2000]
```

### Advanced Configuration Options

```python
# Specialized configurations
config = TransformerConfig(
    model_type="decoder_only",
    vocab_size=2000,
    d_model=768,
    num_heads=12,
    num_decoder_layers=1,            # Single layer for analysis
    d_ff=3072,
    max_seq_length=64,
    dropout=0.1,
    device="cpu",
    # no_ffn=True                    # Disable feed-forward networks (optional)
)

# Create model with custom settings
specialized_model = Transformer.from_config(config)
```

### Model Information and Utilities

```python
# Get model information
print(f"Model type: {decoder_model.config.model_type}")
print(f"Parameters: {sum(p.numel() for p in decoder_model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in decoder_model.parameters() if p.requires_grad):,}")

# Access model components
print(f"Number of layers: {len(decoder_model.layers)}")
print(f"Model dimension: {decoder_model.config.d_model}")
```

## 📐 Intrinsic Dimensions Analysis

TRACE provides comprehensive tools for analyzing the intrinsic dimensionality of transformer representations using advanced geometric methods. This helps understand how models compress and organize information across layers.

### Supported Methods

- **TwoNN**: Two Nearest Neighbors method for robust ID estimation
- **MLE**: Maximum Likelihood Estimation approach
- **PCA**: Principal Component Analysis for linear dimensionality

### Basic Configuration

```python
from trace.intrinsic_dimensions import IntrinsicDimensionsConfig, IntrinsicDimensionAnalyzer

# Create configuration for intrinsic dimensions analysis
config = IntrinsicDimensionsConfig(
    model_type="decoder_only",           # Model architecture
    id_method="TwoNN",                   # Dimensionality estimation method
    layers_to_analyze=None,              # None = analyze all layers
    max_samples=1000,                    # Limit samples for efficiency
    flatten_sequence=True,               # Flatten sequence dimension
    save_visualizations=True,            # Generate plots
    show_plots=False,                    # Display plots immediately
    log_dir="./plots/intrinsic_dims"     # Output directory
)

# Initialize analyzer
analyzer = IntrinsicDimensionAnalyzer(config)
```

### Layer-Specific Analysis

```python
# Analyze specific layers for encoder-only models
config = IntrinsicDimensionsConfig(
    model_type="encoder_only",
    layers_to_analyze={'encoder': [0, 3, 5]},  # Analyze layers 0, 3, and 5
    id_method="TwoNN"
)

# For encoder-decoder models
config = IntrinsicDimensionsConfig(
    model_type="encoder_decoder",
    layers_to_analyze={
        'encoder': [0, 2, 4],
        'decoder': [0, 1, 3]
    }
)

# For decoder-only models (simple list notation)
config = IntrinsicDimensionsConfig(
    model_type="decoder_only",
    layers_to_analyze=[0, 6, 11],  # First, middle, and last layers
    id_method="MLE"
)
```

### Complete Analysis Workflow

```python
from trace.intrinsic_dimensions import IntrinsicDimensionAnalyzer
from torch.utils.data import DataLoader

# Prepare your data
# data_loader = DataLoader(your_dataset, batch_size=32, shuffle=False)

# Create analyzer with default settings
analyzer = IntrinsicDimensionAnalyzer()

# Run comprehensive analysis
intrinsic_dimensions = analyzer.analyze(
    model=decoder_model,
    data_loader=data_loader,
    layers=[0, 3, 6, 9, 11],  # Specific layers to analyze
    model_name="my_transformer"
)

# Results dictionary: {(layer_idx, layer_type): intrinsic_dimension}
print("Intrinsic Dimensions Results:")
for (layer_idx, layer_type), id_value in intrinsic_dimensions.items():
    print(f"  Layer {layer_idx} ({layer_type}): {id_value:.2f}")
```

### Manual Hidden State Extraction

```python
from trace.intrinsic_dimensions import (
    extract_hidden_representations,
    compute_intrinsic_dimensions,
    average_intrinsic_dimension
)

# Extract hidden states from specific layers
hidden_states, _, _ = extract_hidden_representations(
    model=encoder_model,
    dataloader=data_loader,
    layer_indices={'encoder': [0, 2, 4]},
    device='cuda'
)

# Compute intrinsic dimensions
config = IntrinsicDimensionsConfig(id_method="TwoNN")
intrinsic_dims = compute_intrinsic_dimensions(hidden_states, config)

# Calculate average across layers
avg_id = average_intrinsic_dimension(
    intrinsic_dims,
    layer_indices=[(0, 'encoder'), (2, 'encoder'), (4, 'encoder')]
)
print(f"Average intrinsic dimension: {avg_id:.2f}")
```

### Advanced Visualization Options

```python
from trace.intrinsic_dimensions import IntrinsicDimensionsVisualizer

# Create custom visualizer
visualizer = IntrinsicDimensionsVisualizer(
    log_dir="./custom_plots",
    config=config
)

# Generate comprehensive visualizations
visualizer.generate_all_visualizations(
    intrinsic_dimensions=intrinsic_dims,
    model_name="transformer_analysis",
    show_plots=True  # Display plots immediately
)

# Individual plot types
visualizer.plot_id_by_layer(intrinsic_dims, "my_model", save_plot=True)
visualizer.plot_id_distribution(intrinsic_dims, "my_model", save_plot=True)
visualizer.plot_final_id(intrinsic_dims, "my_model", save_plot=True)

# Save metrics to CSV
visualizer.save_metrics(intrinsic_dims, "my_model")
```

### Method Comparison

```python
# Compare different ID estimation methods
methods = ["TwoNN", "MLE", "PCA"]
results_comparison = {}

for method in methods:
    config = IntrinsicDimensionsConfig(
        id_method=method,
        layers_to_analyze=[0, 6, 11]
    )
    analyzer = IntrinsicDimensionAnalyzer(config)
    results_comparison[method] = analyzer.analyze(
        model=decoder_model,
        data_loader=data_loader,
        model_name=f"model_{method.lower()}"
    )

# Compare results
for method, results in results_comparison.items():
    avg_id = average_intrinsic_dimension(results)
    print(f"{method} average ID: {avg_id:.2f}")
```

### Integration with Training Loop

```python
# Monitor intrinsic dimensions during training
def training_step_with_id_analysis(model, data_loader, step):
    # Your training code here...
    
    # Periodic intrinsic dimension analysis
    if step % 100 == 0:  # Analyze every 100 steps
        analyzer = IntrinsicDimensionAnalyzer(
            config=IntrinsicDimensionsConfig(
                layers_to_analyze=[0, -1],  # First and last layers
                save_visualizations=True,
                log_dir=f"./training_analysis/step_{step}"
            )
        )
        
        # Quick analysis on subset of data
        subset_loader = DataLoader(
            dataset=data_loader.dataset,
            batch_size=32,
            shuffle=False,
            sampler=torch.utils.data.SubsetRandomSampler(range(100))
        )
        
        id_results = analyzer.analyze(
            model=model,
            data_loader=subset_loader,
            model_name=f"training_step_{step}"
        )
        
        # Log results
        avg_id = average_intrinsic_dimension(id_results)
        print(f"Step {step}: Average ID = {avg_id:.2f}")
```

## 🌊 Hessian Analysis

TRACE provides comprehensive Hessian analysis capabilities for understanding loss landscape properties, training dynamics, and memorization patterns. This module analyzes curvature information to reveal insights about model optimization and generalization.

### Key Features

- **Eigenvalue Analysis**: Track extreme eigenvalues, trace estimates, and spectral properties
- **Component-Specific Analysis**: Analyze individual model components (attention, FFN, embeddings)
- **Gradient-Hessian Alignment**: Monitor optimization direction relative to curvature
- **Memorization Detection**: Compare train/validation landscapes to detect overfitting
- **Real-time Monitoring**: Integration with training loops for continuous analysis

### Basic Configuration

```python
from trace.hessian import HessianConfig, HessianAnalyzer

# Create configuration for Hessian analysis
config = HessianConfig(
    n_components=10,                           # Number of eigenvalues to compute
    num_batches=100,                          # Batches for Hessian estimation
    device="cuda",                            # Device for computation
    
    # Analysis toggles
    track_component_hessian=True,             # Analyze individual components
    track_gradient_alignment=True,            # Monitor gradient-Hessian alignment
    track_train_val_landscape_divergence=True, # Detect memorization signals
    
    # Component selection
    component_list=["attention", "ffn", "hidden_states"],
    
    # Output settings
    log_dir="./hessian_analysis",             # Output directory
    save_hessian_data=True,                   # Save raw data
    show_plots=False                          # Display plots immediately
)

# Initialize analyzer
analyzer = HessianAnalyzer(config)
```

### Preset Configurations

```python
# Minimal configuration for basic analysis
minimal_config = HessianConfig.minimal(
    n_components=5,
    track_component_hessian=False
)

# Comprehensive configuration for detailed research
comprehensive_config = HessianConfig.comprehensive(
    n_components=20,
    component_list=[
        "attention", "attention_query", "attention_key", "attention_value",
        "ffn", "embeddings", "norm", "hidden_states", "output_projection"
    ]
)

# Default balanced configuration
default_config = HessianConfig.default()
```

### Single-Step Analysis

```python
import torch.nn as nn

# Prepare loss function and data
loss_fn = nn.CrossEntropyLoss()
# train_batch and val_batch should be prepared with your data

# Perform comprehensive analysis at a single training step
results = analyzer.analyze_step(
    model=decoder_model,
    loss_fn=loss_fn,
    train_batch=train_batch,
    val_batch=val_batch,              # Optional for memorization analysis
    model_type="decoder_only",
    step=100
)

# Access results
print(f"Max eigenvalue: {results['hessian']['max_eigenvalue']:.2e}")
print(f"Min eigenvalue: {results['hessian']['min_eigenvalue']:.2e}")
print(f"Trace estimate: {results['hessian']['hessian_trace_estimate']:.2e}")
print(f"Negative eigenvalues: {results['hessian']['negative_count']}")
print(f"Effective rank: {results['hessian']['effective_rank_95']}")
```

### Component-Specific Analysis

```python
from trace.hessian import ComponentAnalyzer, ComponentSelector

# Initialize component analyzer
component_analyzer = ComponentAnalyzer()

# Get appropriate components for your model
no_ffn = getattr(decoder_model, 'no_ffn', False)
components = ComponentSelector.get_standard_components(no_ffn)
print(f"Analyzing components: {components}")

# Analyze all components
component_results = component_analyzer.analyze_all_components(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    component_list=components,
    n_components=10
)

# Compare component complexity
for component, metrics in component_results.items():
    if 'error' not in metrics:
        print(f"{component}:")
        print(f"  Parameters: {metrics['num_params']:,}")
        print(f"  Max eigenvalue: {metrics['max_eigenvalue']:.2e}")
        print(f"  Effective rank: {metrics['effective_rank_95']}")
```

### Advanced Component Selection

```python
# Validate components exist in your model
valid_components = ComponentSelector.validate_components(
    model=decoder_model,
    component_list=["attention", "ffn", "embeddings", "norm"]
)

# Use comprehensive component set for detailed analysis
comprehensive_components = ComponentSelector.get_comprehensive_components()

# Custom component analysis
custom_analyzer = ComponentAnalyzer()
attention_results = custom_analyzer.analyze_component(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    component_name="attention",
    n_components=15
)
```

### Gradient-Hessian Alignment Analysis

```python
# Analyze optimization dynamics through gradient-curvature relationships
alignment_results = analyzer.compute_gradient_alignment(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    eigenvalues=eigenvalues,  # From previous Hessian computation
    eigenvectors=eigenvectors
)

# Key alignment metrics
print(f"Gradient-Hessian alignment: {alignment_results['grad_Hg_alignment']:.4f}")
print(f"Weighted alignment score: {alignment_results['weighted_alignment']:.4f}")
print(f"Curvature/gradient ratio: {alignment_results['grad_Hg_ratio']:.4f}")
```

### Memorization Detection

```python
# Detect memorization through train/validation landscape comparison
memorization_signals = analyzer.compute_train_val_divergence(
    model=decoder_model,
    loss_fn=loss_fn,
    train_batch=train_batch,
    val_batch=val_batch,
    model_type="decoder_only"
)

# Memorization indicators
print(f"Landscape divergence score: {memorization_signals['train_val_landscape_divergence_score']:.4f}")
print(f"Trace ratio (train/val): {memorization_signals['trace_ratio']:.4f}")
print(f"Eigenvalue distribution overlap: {memorization_signals['eigenvalue_distribution_overlap']:.4f}")
print(f"Effective rank difference: {memorization_signals['effective_rank_diff']}")
```

### Low-Level Utilities

```python
from trace.hessian import (
    compute_loss,
    extract_component_parameters,
    get_hessian_eigenvectors
)

# Manual loss computation with proper model handling
loss = compute_loss(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    ignore_index=-100
)

# Extract parameters for specific components
attention_params = extract_component_parameters(
    model=decoder_model,
    component_name="attention",
    include_bias=False
)
print(f"Attention parameters: {sum(p.numel() for p in attention_params):,}")

# Manual Hessian eigenvalue computation
eigenvalues, eigenvectors = get_hessian_eigenvectors(
    model=decoder_model,
    loss_fn=loss_fn,
    train_data_loader=train_batch,
    num_batches=50,
    device="cuda",
    n_top_vectors=15
)
```

### Comprehensive Visualization

```python
from trace.hessian import HessianVisualizer

# Initialize visualizer
visualizer = HessianVisualizer(config)

# Track analysis over multiple steps
hessian_history = {}
for step in range(0, 1000, 100):
    # Perform analysis at each step
    results = analyzer.analyze_step(
        model=decoder_model,
        loss_fn=loss_fn,
        train_batch=train_batch,
        val_batch=val_batch,
        step=step
    )
    hessian_history[step] = results

# Generate comprehensive visualization report
visualizer.create_comprehensive_report(
    hessian_history=hessian_history,
    model_name="my_transformer"
)

# Individual plot types
visualizer.plot_eigenvalue_evolution(hessian_history, "my_transformer")
visualizer.plot_gradient_alignment(hessian_history, "my_transformer") 
visualizer.plot_component_comparison(hessian_history, "my_transformer")
visualizer.plot_memorization_metrics(hessian_history, model_name="my_transformer")
```

### Training Loop Integration

```python
def training_step_with_hessian_analysis(model, optimizer, train_batch, val_batch, step):
    # Regular training step
    optimizer.zero_grad()
    loss = compute_loss(model, loss_fn, train_batch, "decoder_only")
    loss.backward()
    optimizer.step()
    
    # Periodic Hessian analysis
    if step % 50 == 0:  # Analyze every 50 steps
        # Create step-specific analyzer
        step_analyzer = HessianAnalyzer(HessianConfig.default(
            log_dir=f"./analysis/step_{step}",
            n_components=8,  # Reduced for speed
            track_train_val_landscape_divergence=(val_batch is not None)
        ))
        
        # Perform analysis
        hessian_results = step_analyzer.analyze_step(
            model=model,
            loss_fn=loss_fn,
            train_batch=train_batch,
            val_batch=val_batch,
            step=step
        )
        
        # Log key metrics
        if 'hessian' in hessian_results:
            h = hessian_results['hessian']
            print(f"Step {step}: "
                  f"Max λ: {h['max_eigenvalue']:.2e}, "
                  f"Trace: {h['hessian_trace_estimate']:.2e}, "
                  f"Neg count: {h['negative_count']}")
        
        # Check for memorization signals
        if 'train_val_divergence' in hessian_results:
            div_score = hessian_results['train_val_divergence']['train_val_landscape_divergence_score']
            if div_score > 0.5:  # Threshold for concern
                print(f"⚠️  High memorization signal detected: {div_score:.3f}")
    
    return loss.item()
```

### Performance Optimization

```python
# Efficient configuration for large models
efficient_config = HessianConfig(
    n_components=5,                    # Fewer eigenvalues
    num_batches=50,                    # Fewer batches
    track_component_hessian=False,     # Skip component analysis
    track_gradient_alignment=False,    # Skip alignment analysis
    track_train_val_landscape_divergence=False  # Skip memorization detection
)

# Memory-efficient analysis for very large models
def memory_efficient_hessian_analysis(model, data_batch, step):
    # Use CPU for Hessian computation to save GPU memory
    cpu_config = HessianConfig.minimal(device="cpu", n_components=3)
    analyzer = HessianAnalyzer(cpu_config)
    
    # Move only necessary data to CPU
    cpu_batch = {k: v.cpu() for k, v in data_batch.items()}
    model_cpu = model.cpu()
    
    results = analyzer.analyze_step(
        model=model_cpu,
        loss_fn=nn.CrossEntropyLoss(),
        train_batch=cpu_batch,
        step=step
    )
    
    # Move model back to GPU
    model.cuda()
    return results
```

## 🔍 Linguistic Probes

TRACE provides sophisticated linguistic probing capabilities to analyze what linguistic knowledge models acquire during training. This module includes both probe training infrastructure and real-time monitoring systems for tracking linguistic understanding.

### Key Features

- **Multi-label Probing**: Train sophisticated probes to detect multiple linguistic features simultaneously
- **POS Analysis**: Track part-of-speech understanding (basic and detailed granularity)
- **Semantic Role Analysis**: Monitor semantic role labeling capabilities
- **Real-time Monitoring**: Use pre-trained probes to track linguistic understanding during training
- **Flexible Tagging**: Rule-based taggers for synthetic and natural text
- **Performance Tracking**: Category-specific performance monitoring

### Two-Phase Workflow

The linguistic probes module follows a two-phase workflow:

1. **Training Phase**: Train probes on your model to establish linguistic understanding baselines
2. **Monitoring Phase**: Use trained probes to monitor linguistic capabilities during model training

### Phase 1: Training Probes

First, you need to train probes on your model to establish what linguistic features it has learned:

```python
from trace.linguistic_probes import LinguisticProbesConfig, ProbeTrainer

# Configure probe training
config = LinguisticProbesConfig(
    # Probe architecture
    probe_type='multilabel',           # 'linear' or 'multilabel'
    hidden_dim=128,                    # Hidden dimension for MLPs
    
    # Training parameters
    epochs=3,                          # Training epochs for probes
    lr=0.001,                         # Learning rate
    batch_size=64,                    # Batch size for probe training
    device="cuda",                    # Device for computation
    
    # What to analyze
    track_pos=True,                   # Train POS probes
    track_semantic=True,              # Train semantic role probes
    pos_granularity='basic',          # 'basic' or 'detailed'
    semantic_granularity='basic',     # 'basic' or 'detailed'
    
    # Layer selection (None = all layers)
    layer_indices=None,
    
    # Output settings
    save_probes=True,                 # Save trained probes
    save_visualizations=False,         # Create training visualizations
    save_path='./trained_probes',     # Where to save probes
    log_dir="./probe_training_logs",  # Training logs directory
)

# Initialize trainer
trainer = ProbeTrainer(config, tokenizer)

# Train probes on your model
training_results = trainer.train_all_probes(
    model=decoder_model,
    dataloader=your_training_dataloader
)

# View training results
for layer_key, layer_results in training_results.items():
    print(f"\nLayer {layer_key}:")
    if 'pos' in layer_results:
        pos_acc = layer_results['pos']['accuracy']
        print(f"  POS accuracy: {pos_acc:.3f}")
    if 'semantic' in layer_results:
        sem_acc = layer_results['semantic']['accuracy']
        print(f"  Semantic accuracy: {sem_acc:.3f}")
```

### Granularity Options

Configure the level of linguistic analysis detail:

```python
# Basic granularity - simplified categories
basic_config = LinguisticProbesConfig(
    pos_granularity='basic',           # NOUN, VERB, ADJ, ADV, PREP, DET, CONJ, OTHER
    semantic_granularity='basic'       # AGENT, PATIENT, ACTION, LOCATION, RELATION, CONNECTOR, RESULT, OTHER
)

# Detailed granularity - fine-grained categories  
detailed_config = LinguisticProbesConfig(
    pos_granularity='detailed',        # Includes TRANSITIVE_VERB, COMMUNICATION_VERB, etc.
    semantic_granularity='detailed'    # Includes MOTION, COMMUNICATION, DESTINATION, etc.
)

# View available categories
print("POS categories:", basic_config.get_pos_categories())
print("Semantic categories:", basic_config.get_semantic_categories())
```

### Preset Training Configurations

```python
# Quick training for experimentation
quick_config = LinguisticProbesConfig.minimal(
    epochs=2,
    batch_size=32,
    hidden_dim=64,
    save_probes=True
)

# Comprehensive analysis
comprehensive_config = LinguisticProbesConfig.comprehensive(
    epochs=5,
    hidden_dim=256,
    pos_granularity='detailed',
    semantic_granularity='detailed',
    save_visualizations=True
)

# POS-only training
pos_only_config = LinguisticProbesConfig.pos_only(
    track_semantic=False,
    pos_granularity='detailed'
)
```

### Phase 2: Real-time Monitoring

Once you have trained probes, use them to monitor linguistic understanding during training:

```python
from trace.linguistic_probes import POSAnalyzer, SemanticAnalyzer

# Configure monitoring (note: different config than training)
monitoring_config = LinguisticProbesConfig(
    track_pos=True,
    track_semantic=True,
    layer_indices={'decoder': [0, 6, 11]},  # Monitor specific layers
    log_dir="./training_monitoring",
    save_visualizations=True,
    
    # Specify where to load trained probes from
    probe_load_path={
        (0, 'decoder'): './trained_probes/pos_layer0_decoder.pt',
        (6, 'decoder'): './trained_probes/pos_layer6_decoder.pt',
        (11, 'decoder'): './trained_probes/pos_layer11_decoder.pt',
    }
)

# Initialize analyzers
pos_analyzer = POSAnalyzer(monitoring_config)
semantic_analyzer = SemanticAnalyzer(monitoring_config)

# Load pre-trained probes
probe_paths = {
    (0, 'decoder'): './trained_probes/pos_layer0_decoder.pt',
    (6, 'decoder'): './trained_probes/pos_layer6_decoder.pt',
    (11, 'decoder'): './trained_probes/pos_layer11_decoder.pt',
}
pos_analyzer.load_probes(probe_paths)

# Monitor during training step
def training_step_with_linguistic_monitoring(model, batch, step):
    # Your regular training code here...
    
    if step % 50 == 0:  # Monitor every 50 steps
        # Get confidence scores from pre-trained probes
        pos_confidences = pos_analyzer.analyze(
            model=model,
            dataloader=DataLoader([batch]),  # Single batch
            tokenizer=tokenizer,
            model_name=f"step_{step}"
        )
        
        # pos_confidences structure:
        # {(layer_idx, layer_type): {'NOUN': 0.85, 'VERB': 0.72, ...}}
        
        for layer_key, confidences in pos_confidences.items():
            print(f"Step {step}, Layer {layer_key}:")
            for tag, confidence in confidences.items():
                print(f"  {tag}: {confidence:.3f}")
```

### Advanced Probe Training

For more control over the training process:

```python
# Train probes for specific model components
from trace.linguistic_probes.models import MultiLabelProbe

# Manual probe configuration
manual_config = LinguisticProbesConfig(
    input_dim=768,                    # Must match model hidden size
    num_classes=8,                    # Number of linguistic categories
    probe_type='multilabel',
    hidden_dim=128,
    dropout=0.1,
    use_class_weights=True,           # Handle class imbalance
    penalize_frequent_classes=True,   # Reduce weight of frequent classes
    penalized_classes=[0, 1],         # Typically NOUN and VERB
    criterion='cross_entropy'
)

# Create and train individual probe
probe = MultiLabelProbe(
    input_dim=768,
    config=manual_config
)

# Custom training loop
for epoch in range(manual_config.epochs):
    for batch in probe_dataloader:
        # Your custom training logic here
        pass
```

### Linguistic Taggers

TRACE includes sophisticated rule-based taggers for synthetic data:

```python
from trace.linguistic_probes import POSTagger, SemanticTagger

# Initialize taggers
pos_tagger = POSTagger(
    granularity='detailed',
    use_nltk_fallback=True  # Fallback to NLTK for unknown tokens
)

semantic_tagger = SemanticTagger(granularity='detailed')

# Tag synthetic text
text = "noun_entity transitive_verb_action noun_target preposition_to location_place"
pos_tags = pos_tagger.tag_text(text)
semantic_tags = semantic_tagger.tag_text(text)

print("POS tags:", pos_tags)
print("Semantic tags:", semantic_tags)

# Output:
# POS tags: [('noun_entity', 'NOUN'), ('transitive_verb_action', 'TRANSITIVE_VERB'), ...]
# Semantic tags: [('noun_entity', 'AGENT'), ('transitive_verb_action', 'ACTION'), ...]
```

### Performance Tracking

Track linguistic performance by category during training:

```python
from trace.linguistic_probes import POSPerformanceTracker, SemanticPerformanceTracker

# Initialize trackers
pos_tracker = POSPerformanceTracker(
    tokenizer=tokenizer,
    log_dir="./performance_tracking",
    config=LinguisticProbesConfig(pos_granularity='basic')
)

semantic_tracker = SemanticPerformanceTracker(
    tokenizer=tokenizer,
    log_dir="./performance_tracking",
    config=LinguisticProbesConfig(semantic_granularity='basic')
)

# Use during training
def training_step_with_performance_tracking(model, batch, step):
    # Forward pass
    outputs = model(**batch)
    logits = outputs.logits
    
    # Process batch for linguistic performance metrics
    pos_metrics = pos_tracker.process_batch(batch, logits)
    semantic_metrics = semantic_tracker.process_batch(batch, logits)
    
    # Update trackers
    pos_tracker.update_epoch_metrics(pos_metrics, step)
    semantic_tracker.update_epoch_metrics(semantic_metrics, step)
    
    # Print category-specific accuracies
    if step % 100 == 0:
        for category, accuracy in pos_metrics['pos_analysis_accuracy'].items():
            print(f"POS {category}: {accuracy:.3f}")
```

### Comprehensive Visualization

Generate detailed analysis visualizations:

```python
from trace.linguistic_probes import ProbesVisualizer

# Initialize visualizer
visualizer = ProbesVisualizer(
    log_dir="./probe_visualizations",
    config=monitoring_config
)

# Track confidence over training
confidence_history = {}
for step in range(0, 1000, 50):
    # Get confidence scores at this step
    pos_confidences = pos_analyzer.analyze(model, eval_dataloader, tokenizer)
    confidence_history[step] = pos_confidences

# Create comprehensive plots
visualizer.plot_probe_confidence_analysis(
    confidence_data=confidence_history,
    model_name="my_transformer",
    analysis_type="pos",
    save_plot=True,
    show_plots=False
)

# This creates:
# - Per-tag plots showing each tag's evolution across layers
# - Per-layer plots showing all tags for each layer  
# - Comprehensive plots showing all tags and layers together
```

### Legacy Compatibility Functions

For backward compatibility with existing code:

```python
from trace.linguistic_probes import (
    run_pos_probe_analysis,
    run_semantic_probe_analysis,
    run_comprehensive_probe_analysis
)

# Legacy analysis (for monitoring, not training)
pos_results = run_pos_probe_analysis(
    model=decoder_model,
    dataloader=test_dataloader,
    tokenizer=tokenizer,
    device="cuda",
    layer_indices=[0, 6, 11],
    config=monitoring_config,
    model_name="legacy_test"
)

# Comprehensive analysis
all_results = run_comprehensive_probe_analysis(
    model=decoder_model,
    dataloader=test_dataloader,
    tokenizer=tokenizer,
    device="cuda",
    layer_indices=[0, 6, 11],
    config=monitoring_config,
    model_name="comprehensive_test"
)

print("POS results:", all_results['pos'])
print("Semantic results:", all_results['semantic'])
```

### Training Loop Integration

Complete example of integrating linguistic probes into training:

```python
def complete_training_with_linguistic_analysis():
    # Step 1: Train probes on initial model
    print("Training linguistic probes...")
    probe_config = LinguisticProbesConfig.comprehensive(
        save_path='./trained_probes',
        save_probes=True
    )
    trainer = ProbeTrainer(probe_config, tokenizer)
    probe_results = trainer.train_all_probes(model, train_dataloader)
    
    # Step 2: Set up monitoring
    monitor_config = LinguisticProbesConfig(
        layer_indices={'decoder': [0, 6, 11]},
        log_dir="./linguistic_monitoring"
    )
    
    pos_analyzer = POSAnalyzer(monitor_config)
    pos_analyzer.load_probes({
        (0, 'decoder'): './trained_probes/pos_layer0_decoder.pt',
        (6, 'decoder'): './trained_probes/pos_layer6_decoder.pt', 
        (11, 'decoder'): './trained_probes/pos_layer11_decoder.pt',
    })
    
    # Step 3: Training loop with monitoring
    confidence_history = {}
    for step in range(training_steps):
        # Regular training step
        loss = train_step(model, train_batch)
        
        # Linguistic monitoring every 50 steps
        if step % 50 == 0:
            confidences = pos_analyzer.analyze(
                model, eval_dataloader, tokenizer
            )
            confidence_history[step] = confidences
            
            # Log linguistic understanding changes
            for layer_key, layer_confidences in confidences.items():
                noun_conf = layer_confidences.get('NOUN', 0)
                verb_conf = layer_confidences.get('VERB', 0)
                print(f"Step {step}, Layer {layer_key}: "
                      f"NOUN={noun_conf:.3f}, VERB={verb_conf:.3f}")
    
    # Step 4: Generate final analysis
    visualizer = ProbesVisualizer("./final_analysis", monitor_config)
    visualizer.plot_probe_confidence_analysis(
        confidence_history, "final_model", "pos"
    )
    
    return confidence_history
```

## 📊 Output Monitoring

TRACE's Output Monitoring module provides real-time analysis of model output quality by tracking linguistic accuracy across different grammatical and semantic categories during training. This helps you understand how well your model is learning to generate linguistically coherent text.

### Key Features

- **Real-time Output Analysis**: Monitor output quality during training without interrupting the process
- **POS Accuracy Tracking**: Track part-of-speech accuracy by category (nouns, verbs, adjectives, etc.)
- **Semantic Role Monitoring**: Monitor semantic role labeling accuracy (agents, patients, actions, etc.)
- **Category-specific Insights**: Identify which linguistic categories your model struggles with most
- **Training Evolution Visualization**: See how output quality improves over training steps
- **Lightweight Integration**: Minimal overhead on training performance

### Quick Start

```python
from trace.output_monitoring import OutputMonitoringAnalyzer, OutputMonitoringConfig

# Configure output monitoring
config = OutputMonitoringConfig(
    track_pos_performance=True,        # Track POS accuracy
    track_semantic_roles=True,         # Track semantic role accuracy
    pos_granularity='basic',           # 'basic' or 'detailed'
    semantic_granularity='basic',      # 'basic' or 'detailed'
    save_visualizations=True,          # Generate plots
    log_dir="./output_monitoring",     # Save results here
    show_plots=False                   # Don't display plots during training
)

# Initialize analyzer
analyzer = OutputMonitoringAnalyzer(config)

# Integrate into training loop
def training_step_with_output_monitoring(model, batch, step):
    # Forward pass
    outputs = model(**batch)
    logits = outputs.logits
    
    # Calculate loss and do backward pass
    loss = loss_function(logits, batch['labels'])
    loss.backward()
    optimizer.step()
    
    # Monitor output quality every 50 steps
    if step % 50 == 0:
        monitoring_results = analyzer.analyze(
            batch=batch,
            outputs=logits,
            tokenizer=tokenizer,
            step=step
        )
        
        # Log results
        if 'pos_accuracy' in monitoring_results:
            for category, accuracy in monitoring_results['pos_accuracy'].items():
                print(f"Step {step} - POS {category}: {accuracy:.3f}")
        
        if 'semantic_accuracy' in monitoring_results:
            for category, accuracy in monitoring_results['semantic_accuracy'].items():
                print(f"Step {step} - Semantic {category}: {accuracy:.3f}")
    
    return loss
```

### Configuration Options

```python
# Basic monitoring - lightweight and fast
basic_config = OutputMonitoringConfig(
    track_pos_performance=True,
    track_semantic_roles=False,        # Skip semantic for speed
    pos_granularity='basic',           # Fewer categories
    save_visualizations=False          # No plots for faster training
)

# Comprehensive monitoring - detailed analysis
comprehensive_config = OutputMonitoringConfig(
    track_pos_performance=True,
    track_semantic_roles=True,
    pos_granularity='detailed',        # Fine-grained POS categories
    semantic_granularity='detailed',   # Fine-grained semantic categories
    save_visualizations=True,
    log_dir="./comprehensive_monitoring"
)

# POS-only monitoring
pos_only_config = OutputMonitoringConfig(
    track_pos_performance=True,
    track_semantic_roles=False,
    pos_granularity='detailed'
)

# Semantic-only monitoring
semantic_only_config = OutputMonitoringConfig(
    track_pos_performance=False,
    track_semantic_roles=True,
    semantic_granularity='detailed'
)
```

### Granularity Levels

Choose the level of detail for linguistic analysis:

```python
# Basic granularity - simplified categories
basic_config = OutputMonitoringConfig(
    pos_granularity='basic',           # NOUN, VERB, ADJ, ADV, PREP, DET, CONJ, OTHER
    semantic_granularity='basic'       # AGENT, PATIENT, ACTION, LOCATION, RELATION, CONNECTOR, RESULT, OTHER
)

# Detailed granularity - fine-grained analysis
detailed_config = OutputMonitoringConfig(
    pos_granularity='detailed',        # TRANSITIVE_VERB, COMMUNICATION_VERB, MOTION_VERB, etc.
    semantic_granularity='detailed'    # MOTION, COMMUNICATION, DESTINATION, TIME, PROPERTY, etc.
)

print("Available POS categories:", basic_config.pos_granularity)
print("Available semantic categories:", basic_config.semantic_granularity)
```

### Complete Training Integration

```python
def complete_training_with_output_monitoring():
    # Initialize monitoring
    config = OutputMonitoringConfig(
        track_pos_performance=True,
        track_semantic_roles=True,
        pos_granularity='basic',
        save_visualizations=True,
        log_dir="./training_output_monitoring"
    )
    
    analyzer = OutputMonitoringAnalyzer(config)
    
    # Storage for all monitoring results
    all_monitoring_results = {}
    
    # Training loop
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            # Regular training
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch['labels'])
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Monitor output quality
            if step % monitoring_frequency == 0:
                monitoring_results = analyzer.analyze(
                    batch=batch,
                    outputs=outputs.logits,
                    tokenizer=tokenizer,
                    step=step
                )
                
                all_monitoring_results[step] = monitoring_results
                
                # Print progress
                print(f"Epoch {epoch}, Step {step}:")
                if 'pos_accuracy' in monitoring_results:
                    pos_avg = np.mean(list(monitoring_results['pos_accuracy'].values()))
                    print(f"  Average POS accuracy: {pos_avg:.3f}")
                
                if 'semantic_accuracy' in monitoring_results:
                    sem_avg = np.mean(list(monitoring_results['semantic_accuracy'].values()))
                    print(f"  Average semantic accuracy: {sem_avg:.3f}")
    
    # Generate final analysis
    analyzer.visualizer.plot_pos_performance_evolution(
        all_monitoring_results, "my_model", save_plot=True
    )
    
    analyzer.visualizer.plot_semantic_role_performance_evolution(
        all_monitoring_results, "my_model", save_plot=True
    )
    
    return all_monitoring_results
```

### Advanced Analysis

Get detailed insights about model performance:

```python
# After training, analyze results
analyzer = OutputMonitoringAnalyzer(config)

# Get POS performance summary
pos_summary = analyzer.get_pos_summary()
print("POS Performance Summary:")
print(f"Hardest categories: {pos_summary['hardest_categories']}")
print(f"Easiest categories: {pos_summary['easiest_categories']}")
print(f"Sample counts: {pos_summary['sample_counts']}")

# Get semantic performance summary
semantic_summary = analyzer.get_semantic_summary()
print("\nSemantic Performance Summary:")
print(f"Hardest categories: {semantic_summary['hardest_categories']}")
print(f"Easiest categories: {semantic_summary['easiest_categories']}")
print(f"Sample counts: {semantic_summary['sample_counts']}")

# Get complete results
full_results = analyzer.get_full_results()
print(f"\nTotal monitoring steps: {len(full_results['pos_history'])}")

# Example output:
# POS Performance Summary:
# Hardest categories: [('PREP', 0.65), ('CONJ', 0.72), ('ADV', 0.78)]
# Easiest categories: [('NOUN', 0.94), ('VERB', 0.91), ('ADJ', 0.88)]
# Sample counts: {'NOUN': 1250, 'VERB': 980, 'ADJ': 640, ...}
```

### Performance Optimization

For large-scale training, optimize monitoring overhead:

```python
# Lightweight monitoring for production training
production_config = OutputMonitoringConfig(
    track_pos_performance=True,
    track_semantic_roles=False,        # Skip semantic for speed
    pos_granularity='basic',           # Fewer categories
    save_visualizations=False,         # No plots during training
    device="cpu"                       # Offload to CPU
)

# Monitor less frequently for faster training
def optimized_training_step(model, batch, step):
    outputs = model(**batch)
    loss = criterion(outputs.logits, batch['labels'])
    
    # Only monitor every 100 steps instead of 50
    if step % 100 == 0:
        monitoring_results = analyzer.analyze(
            batch=batch,
            outputs=outputs.logits,
            tokenizer=tokenizer,
            step=step
        )
        
        # Log only summary statistics
        if 'pos_accuracy' in monitoring_results:
            avg_acc = np.mean(list(monitoring_results['pos_accuracy'].values()))
            print(f"Step {step} - Average POS accuracy: {avg_acc:.3f}")
    
    return loss
```

### Visualization and Export

Generate comprehensive visualizations and export data:

```python
from trace.output_monitoring import OutputMonitoringVisualizer

# Initialize visualizer
visualizer = OutputMonitoringVisualizer(
    log_dir="./monitoring_visualizations",
    config=config
)

# Plot POS accuracy evolution
visualizer.plot_pos_performance_evolution(
    monitoring_results=all_monitoring_results,
    model_name="transformer_v2",
    save_plot=True
)

# Plot semantic role accuracy evolution
visualizer.plot_semantic_role_performance_evolution(
    monitoring_results=all_monitoring_results,
    model_name="transformer_v2", 
    save_plot=True
)

# Save metrics to CSV for further analysis
visualizer.save_metrics(
    monitoring_results=all_monitoring_results,
    model_name="transformer_v2"
)

# This creates:
# - transformer_v2_pos_accuracy_evolution.png
# - transformer_v2_semantic_accuracy_evolution.png  
# - transformer_v2_output_monitoring.csv
```

### Category-Specific Analysis

Dive deep into specific linguistic categories:

```python
def analyze_specific_categories(monitoring_results):
    """Analyze performance for specific linguistic categories."""
    
    # Track noun performance over time
    noun_performance = []
    verb_performance = []
    
    for step, results in monitoring_results.items():
        if 'pos_accuracy' in results:
            noun_acc = results['pos_accuracy'].get('NOUN', 0)
            verb_acc = results['pos_accuracy'].get('VERB', 0)
            noun_performance.append((step, noun_acc))
            verb_performance.append((step, verb_acc))
    
    # Analyze trends
    noun_improvement = noun_performance[-1][1] - noun_performance[0][1] if noun_performance else 0
    verb_improvement = verb_performance[-1][1] - verb_performance[0][1] if verb_performance else 0
    
    print(f"Noun accuracy improvement: {noun_improvement:.3f}")
    print(f"Verb accuracy improvement: {verb_improvement:.3f}")
    
    # Identify problematic categories
    final_step = max(monitoring_results.keys())
    final_results = monitoring_results[final_step]
    
    if 'pos_accuracy' in final_results:
        sorted_pos = sorted(final_results['pos_accuracy'].items(), key=lambda x: x[1])
        print(f"Most challenging POS categories: {sorted_pos[:3]}")
    
    if 'semantic_accuracy' in final_results:
        sorted_semantic = sorted(final_results['semantic_accuracy'].items(), key=lambda x: x[1])
        print(f"Most challenging semantic categories: {sorted_semantic[:3]}")

# Use the analysis
analyze_specific_categories(all_monitoring_results)
```

### Error Analysis and Debugging

Use output monitoring to debug training issues:

```python
def debug_training_with_monitoring(model, val_dataloader, tokenizer):
    """Use output monitoring to debug training issues."""
    
    config = OutputMonitoringConfig(
        track_pos_performance=True,
        track_semantic_roles=True,
        pos_granularity='detailed',
        semantic_granularity='detailed'
    )
    
    analyzer = OutputMonitoringAnalyzer(config)
    
    # Analyze validation set
    for i, batch in enumerate(val_dataloader):
        if i >= 5:  # Just first 5 batches
            break
            
        outputs = model(**batch)
        results = analyzer.analyze(batch, outputs.logits, tokenizer, step=i)
        
        # Check for concerning patterns
        if 'pos_accuracy' in results:
            low_accuracy_pos = [cat for cat, acc in results['pos_accuracy'].items() if acc < 0.5]
            if low_accuracy_pos:
                print(f"Batch {i} - Low POS accuracy categories: {low_accuracy_pos}")
        
        if 'semantic_accuracy' in results:
            low_accuracy_sem = [cat for cat, acc in results['semantic_accuracy'].items() if acc < 0.5]
            if low_accuracy_sem:
                print(f"Batch {i} - Low semantic accuracy categories: {low_accuracy_sem}")
    
    # Generate diagnostic summary
    pos_summary = analyzer.get_pos_summary()
    semantic_summary = analyzer.get_semantic_summary()
    
    print("\nDiagnostic Summary:")
    print("Most problematic POS categories:", pos_summary['hardest_categories'])
    print("Most problematic semantic categories:", semantic_summary['hardest_categories'])
    
    return analyzer

# Use for debugging
debug_analyzer = debug_training_with_monitoring(model, validation_loader, tokenizer)
```

### Multi-Model Comparison

Compare output quality across different models:

```python
def compare_models_output_monitoring(models_dict, test_dataloader, tokenizer):
    """Compare output monitoring results across multiple models."""
    
    results_by_model = {}
    
    config = OutputMonitoringConfig(
        track_pos_performance=True,
        track_semantic_roles=True,
        save_visualizations=False  # We'll create custom comparisons
    )
    
    for model_name, model in models_dict.items():
        print(f"Analyzing {model_name}...")
        analyzer = OutputMonitoringAnalyzer(config)
        
        # Analyze test set
        for step, batch in enumerate(test_dataloader):
            if step >= 10:  # Limit for comparison
                break
                
            outputs = model(**batch)
            results = analyzer.analyze(batch, outputs.logits, tokenizer, step)
            
        # Store final summary
        results_by_model[model_name] = {
            'pos_summary': analyzer.get_pos_summary(),
            'semantic_summary': analyzer.get_semantic_summary()
        }
    
    # Generate comparison report
    print("\n" + "="*50)
    print("MODEL COMPARISON REPORT")
    print("="*50)
    
    for model_name, results in results_by_model.items():
        print(f"\n{model_name}:")
        pos_avg = np.mean(list(results['pos_summary']['average_accuracies'].values()))
        sem_avg = np.mean(list(results['semantic_summary']['average_accuracies'].values()))
        print(f"  Average POS accuracy: {pos_avg:.3f}")
        print(f"  Average semantic accuracy: {sem_avg:.3f}")
        print(f"  Hardest POS: {results['pos_summary']['hardest_categories'][:2]}")
        print(f"  Hardest semantic: {results['semantic_summary']['hardest_categories'][:2]}")
    
    return results_by_model

# Compare models
model_comparison = compare_models_output_monitoring(
    {'baseline': baseline_model, 'improved': improved_model},
    test_dataloader,
    tokenizer
)


# Create tokenizer from your corpus
from trace.tokenizer import create_tokenizer_from_data

CORPUS_PATH = "path/to/your/corpus_s.json"  # Your training data
tokenizer = create_tokenizer_from_data(vocab_file=CORPUS_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size()

# Create transformer model
from trace.transformer import Transformer, TransformerConfig

model_config = TransformerConfig(
    model_type="decoder_only",        # "encoder_only", "decoder_only", "encoder_decoder"
    vocab_size=VOCAB_SIZE,
    d_model=512,                      # Hidden dimension
    num_heads=8,                      # Attention heads
    num_decoder_layers=6,             # Number of layers
    d_ff=2048,                        # Feed-forward dimension
    max_seq_length=128,               # Maximum sequence length
    dropout=0.1,
    device="cuda"                     # "cpu" or "cuda"
)

model = Transformer.from_config(model_config)

# Create data loaders
from trace.dataloader import get_dataloader

train_loader, val_loader, test_loader = get_dataloader(
    corpus_path=CORPUS_PATH,
    tokenizer=tokenizer,
    batch_size=32,
    max_length=128,
    model_type="decoder_only",
    val_split=0.1,
    test_split=0.1
)

# Configure comprehensive training analysis
from trace.training import Trainer, TrainingConfig

training_config = TrainingConfig(
    # Training parameters
    epochs=10,
    learning_rate=1e-4,
    batch_size=32,
    device="cuda",
    
    # Analysis modules (enable all)
    track_hessian=True,               # Loss landscape analysis
    track_linguistic_probes=True,     # POS understanding  
    track_semantic_probes=True,       # Semantic role understanding
    track_intrinsic_dimensions=True,  # Representation dimensionality
    track_pos_performance=True,       # Output POS accuracy
    track_semantic_roles_performance=True,  # Output semantic accuracy
    
    # Analysis frequency and visualization
    track_interval=100,               # Analyze every 100 steps
    save_visualization=True,          # Generate plots
    plots_path="./analysis_results"   # Save results here
)

# Train with comprehensive analysis
trainer = Trainer(training_config, tokenizer, model)
best_loss, analysis_results = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)

# Results automatically saved to ./analysis_results/ with:
# - Hessian eigenvalue evolution plots
# - Linguistic probe confidence tracking  
# - Intrinsic dimension evolution
# - Output quality monitoring
# - Training loss curves and metrics
```


# TRACE: Tracking Representation Abstraction and Compositional Emergence

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TRACE** is a comprehensive Python package for analyzing transformer models during training. It provides modular tools for understanding model behavior through linguistic probes, intrinsic dimension analysis, Hessian landscape exploration, and real-time performance monitoring.

**Note**: TRACE is designed to work seamlessly with synthetic data from the [ABSynth dataset](https://github.com/nura-j/ABSynth_dataset), which provides controlled linguistic structures for systematic analysis of transformer learning dynamics.

## 🎯 Key Features

- **Linguistic Probes**: Monitor syntactic and semantic understanding across layers
- **Intrinsic Dimensions**: Track representation complexity using TwoNN and other methods  
- **Hessian Analysis**: Explore loss landscapes, sharpness, and training dynamics
- **Real-time Monitoring**: Comprehensive visualization during training
- **Output Quality Tracking**: Monitor linguistic accuracy by category
- **Modular Design**: Use individual components or full training integration
- **Easy Integration**: Drop-in replacement for existing training loops

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/nura-j/trace_package.git
cd trace_package

# [Optional] Create and activate the conda environment
conda env create -f environment.yml
conda activate trace

# Install the package
pip install .
```

## 🚀 Quick Start

### Complete Training Pipeline with Analysis

```python
from trace.tokenizer import create_tokenizer_from_data
from trace.transformer import Transformer, TransformerConfig
from trace.dataloader import get_dataloader
from trace.training import Trainer, TrainingConfig

# Create tokenizer from your ABSynth corpus
CORPUS_PATH = "path/to/your/absynth_corpus.json"
tokenizer = create_tokenizer_from_data(vocab_file=CORPUS_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size()

# Configure transformer model
model_config = TransformerConfig(
    model_type="decoder_only",        # "encoder_only", "decoder_only", "encoder_decoder"
    vocab_size=VOCAB_SIZE,
    d_model=512,                      # Hidden dimension
    num_heads=8,                      # Attention heads
    num_decoder_layers=6,             # Number of layers
    d_ff=2048,                        # Feed-forward dimension
    max_seq_length=128,               # Maximum sequence length
    dropout=0.1,
    device="cpu"                      # "cpu" or "cuda"
)

model = Transformer.from_config(model_config)

# Create data loaders
train_loader, val_loader, test_loader = get_dataloader(
    corpus_path=CORPUS_PATH,
    tokenizer=tokenizer,
    batch_size=32,
    max_length=128,
    model_type="decoder_only",
    val_split=0.1,
    test_split=0.1
)

# Configure comprehensive analysis
training_config = TrainingConfig(
    # Training parameters
    epochs=10,
    learning_rate=1e-4,
    batch_size=32,
    device="cpu",
    
    # Analysis modules
    track_hessian=True,                # Loss landscape analysis
    track_linguistic_probes=True,      # Syntactic understanding  
    track_semantic_probes=True,        # Semantic role understanding
    track_intrinsic_dimensions=True,   # Representation dimensionality
    track_pos_performance=True,        # Output POS accuracy
    track_semantic_roles_performance=True,  # Output semantic accuracy
    
    # Analysis settings
    track_interval=500,                # Analyze every 500 steps
    save_visualization=True,           # Generate plots
    plots_path="./analysis_results"    # Save results here
)

# Train with comprehensive analysis
trainer = Trainer(training_config, tokenizer, model)
best_loss, analysis_results = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)

# Results automatically saved with:
# - Training dynamics visualizations
# - Linguistic understanding evolution
# - Representation complexity tracking
# - Performance monitoring plots
```

## 📊 Individual Analysis Modules

### Transformer Model Creation

TRACE provides flexible transformer implementations supporting multiple architectures:

**Supported Architectures:**
- **Encoder-Only**: Perfect for classification and representation learning
- **Decoder-Only**: Optimized for autoregressive generation
- **Encoder-Decoder**: Ideal for sequence-to-sequence tasks

```python
from trace.transformer import Transformer, TransformerConfig

# Decoder-only configuration (most common for language modeling)
config = TransformerConfig(
    model_type="decoder_only",
    vocab_size=2000,
    d_model=768,
    num_heads=12,
    num_decoder_layers=12,
    d_ff=3072,
    max_seq_length=128,
    dropout=0.1,
    device="cpu"
)

model = Transformer.from_config(config)

# Forward pass
import torch
input_ids = torch.randint(0, 2000, (2, 32))  # [batch_size, seq_len]
logits = model(tgt=input_ids)
print(f"Output shape: {logits.shape}")  # [2, 32, 2000]
```

### Intrinsic Dimensions Analysis

Understand how models compress and organize information across layers:

```python
from trace.intrinsic_dimensions import IntrinsicDimensionsConfig, IntrinsicDimensionAnalyzer

# Configure analysis
config = IntrinsicDimensionsConfig(
    model_type="decoder_only",
    id_method="TwoNN",               # Robust dimensionality estimation
    layers_to_analyze=None,          # Analyze all layers
    save_visualizations=True,
    log_dir="./analysis_results"
)

# Run analysis
analyzer = IntrinsicDimensionAnalyzer(config)
intrinsic_dimensions = analyzer.analyze(
    model=model,
    data_loader=train_loader,
    model_name="my_transformer"
)

# View results
for (layer_idx, layer_type), dimension in intrinsic_dimensions.items():
    print(f"Layer {layer_idx} ({layer_type}): {dimension:.2f}D")
```

### Hessian Analysis

Explore loss landscapes and training dynamics:

```python
from trace.hessian import HessianConfig, HessianAnalyzer
import torch.nn as nn

# Configure Hessian analysis
config = HessianConfig(
    n_components=10,                  # Eigenvalues to compute
    track_component_hessian=True,     # Analyze model components
    track_gradient_alignment=True,    # Monitor optimization dynamics
    log_dir="./analysis_results"
)

analyzer = HessianAnalyzer(config)

# Analyze single training step
loss_fn = nn.CrossEntropyLoss()
results = analyzer.analyze_step(
    model=model,
    loss_fn=loss_fn,
    train_batch=next(iter(train_loader)),
    model_type="decoder_only",
    step=100
)

print(f"Max eigenvalue: {results['hessian']['max_eigenvalue']:.2e}")
print(f"Training stability: {results['hessian']['effective_rank_95']}")
```

### Linguistic Probes

Monitor what linguistic knowledge models acquire during training:

**Two-Phase Workflow:**
1. **Training Phase**: Train probes to establish linguistic baselines
2. **Monitoring Phase**: Track linguistic capabilities during model training

```python
from trace.linguistic_probes import LinguisticProbesConfig, ProbeTrainer

# Phase 1: Train probes
config = LinguisticProbesConfig(
    probe_type='multilabel',
    track_pos=True,                   # Part-of-speech analysis
    track_semantic=True,              # Semantic role analysis
    pos_granularity='basic',          # 'basic' or 'detailed'
    save_probes=True,
    save_path='./trained_probes'
)

probe_trainer = ProbeTrainer(config, tokenizer)
training_results = probe_trainer.train_all_probes(
    model=model,
    dataloader=train_loader
)

# Phase 2: Monitor during training
from trace.linguistic_probes import POSAnalyzer

pos_analyzer = POSAnalyzer(config)
probe_paths = {
    (0, 'decoder'): './trained_probes/pos_layer0_decoder.pt',
    (6, 'decoder'): './trained_probes/pos_layer6_decoder.pt'
}
pos_analyzer.load_probes(probe_paths)

# Use during training steps
confidences = pos_analyzer.analyze(model, eval_loader, tokenizer)
```

### Output Quality Monitoring

Track linguistic accuracy across categories during training:

```python
from trace.output_monitoring import OutputMonitoringAnalyzer, OutputMonitoringConfig

# Configure monitoring
config = OutputMonitoringConfig(
    track_pos_performance=True,       # POS accuracy by category
    track_semantic_roles=True,        # Semantic role accuracy
    pos_granularity='basic',
    save_visualizations=True,
    log_dir="./analysis_results"
)

output_analyzer = OutputMonitoringAnalyzer(config)

# Monitor during training
def training_step_with_monitoring(model, batch, step):
    outputs = model(**batch)
    
    if step % 50 == 0:
        results = output_analyzer.analyze(
            batch=batch,
            outputs=outputs.logits,
            tokenizer=tokenizer,
            step=step
        )
        
        # View category-specific performance
        for category, accuracy in results['pos_accuracy'].items():
            print(f"POS {category}: {accuracy:.3f}")
```

## Working with ABSynth Data

### Understanding ABSynth Structure

ABSynth generates synthetic corpora with controlled linguistic properties. Each sentence includes:

- **Semantic Frame Annotations**: Agent, Patient, Theme, Location roles
- **Linguistic Features**: POS tags, dependency parsing, constituency trees
- **Statistical Properties**: Zipfian word frequencies, entropy profiles
- **Complexity Levels**: Simple, medium, and complex sentence structures

### Loading ABSynth Corpora

```python
from trace.tokenizer import create_tokenizer_from_data

# Load ABSynth corpus
corpus_path = "path/to/absynth_corpus.json"
tokenizer = create_tokenizer_from_data(vocab_file=corpus_path)

# The tokenizer automatically handles ABSynth's synthetic vocabulary
# and maintains consistency with the corpus's linguistic annotations
```

## Creating Your Own ABSynth Dataset

If you don't have an ABSynth corpus file, you can quickly generate one using the [ABSynth library](https://github.com/nura-j/ABSynth_dataset):

### Minimal Example

```bash
# Install ABSynth
pip install git+https://github.com/nura-j/absynth.git
```

```python
from absynth.corpus import SyntheticCorpusGenerator

# Generate a basic corpus (3-line minimal example)
generator = SyntheticCorpusGenerator()
corpus = generator.generate_corpus(num_sentences=1000)
corpus.save("my_absynth_corpus.json")

# Now use with TRACE
CORPUS_PATH = "my_absynth_corpus.json"  # Use in your TRACE pipeline
```

### Custom Configuration (Optional)

For more control over your synthetic data:

```python
from absynth.corpus import SyntheticCorpusGenerator

# Generate corpus with specific properties
generator = SyntheticCorpusGenerator()
corpus = generator.generate_corpus(
    num_sentences=10000,
    complexity_distribution={
        "simple": 0.55,       # 55% simple sentences
        "medium": 0.35,       # 35% medium complexity  
        "complex": 0.1        # 10% complex sentences
    },
    semantic_frame_distribution={
        "transitive_action": 0.4,     # Subject-verb-object patterns
        "intransitive_action": 0.25,  # Subject-verb patterns
        "communication": 0.2,         # Communication verbs
        "motion": 0.15               # Movement and location
    }
)

# Save in format compatible with TRACE
corpus.save("./data/custom_absynth_corpus.json", indent=2)
```

This generates synthetic corpora optimized for transformer analysis with controlled linguistic properties, semantic annotations, and statistical characteristics ideal for TRACE's analysis modules.

## 📈 Visualization and Results

TRACE automatically generates comprehensive visualizations:

- **Training Dynamics**: Loss curves, learning rate schedules, gradient norms
- **Linguistic Evolution**: How syntactic and semantic understanding develops
- **Representation Analysis**: Intrinsic dimension evolution across layers
- **Loss Landscape**: Hessian eigenvalue evolution, sharpness metrics
- **Performance Tracking**: Category-specific accuracy improvements

All results are saved to your specified directory with organized file structure and detailed metadata.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/nura-j/trace_package.git
cd trace_package
conda env create -f environment.yml
conda activate trace
pip install -e .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use TRACE in your research, please cite:

```bibtex
@software{trace2024,
  title={TRACE: Tracking Representation Abstraction and Compositional Emergence},
  author={...},
  year={2024},
  url={https://github.com/nura-j/trace_package}
}
```

## 🔗 Related Projects

- **[ABSynth Dataset](https://github.com/nura-j/ABSynth_dataset)**: Synthetic corpus generation for controlled linguistic analysis
- **[TRACE Documentation](https://trace-docs.readthedocs.io)**: Comprehensive documentation and tutorials