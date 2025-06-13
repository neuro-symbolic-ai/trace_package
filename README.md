# TRACE: Tracking Representation Abstraction and Compositional Emergence
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TRACE** is a comprehensive Python package for analyzing transformer models during training. It provides modular tools for understanding model behavior through linguistic probes, intrinsic dimension analysis, Hessian landscape exploration, and more.

## Features

- **Linguistic Probes**: Monitor syntactic and semantic presence across layers
- **Intrinsic Dimensions**: Track representation complexity using TwoNN and other methods  
- **Hessian Analysis**: Explore loss landscapes, sharpness, and training dynamics
- **Real-time Monitoring**: Comprehensive visualization during training
- **Modular Design**: Use individual components or full training integration
- **Easy Integration**: Drop-in replacement for existing training loops

## Installation

```bash
# Clone the repository
git clone https://github.com/nura-j/trace_package.git
cd trace

# [Optional] create and activate the conda environment
conda env create -f environment.yml
conda activate trace

# Install
pip install .
```


### Basic Usage

```python
# Create tokenizer 
VOCAB_SIZE = 1000  # Example vocabulary size
CORPUS_PATH = "path/to/corpus.json"  # Path to your corpus file
CORPUS_PATH = "/Users/user/Desktop/Year_3/trace_package/data/corpus_full.json"
MODEL_TYPE = "decoder_only"  # or "encoder-decoder" or "encoder-only"
MAX_SEQ_LENGTH = 64  # Example max sequence length
from trace.tokenizer import create_tokenizer_from_data
tokenizer = create_tokenizer_from_data(vocab_file=CORPUS_PATH)
VOCAB_SIZE = tokenizer.get_vocab_size()

# Create a Transformer model
from trace.transformer import Transformer, TransformerConfig
model_config = TransformerConfig(
    model_type=MODEL_TYPE,
    vocab_size=VOCAB_SIZE,
    d_model=768,
    num_heads=12,
    num_decoder_layers=1,
    d_ff=3072,
    max_seq_length=MAX_SEQ_LENGTH,
    dropout=0.1,
    device="cpu"  
)

model = Transformer.from_config(model_config)

# Create Training config 
from trace.training import Trainer, TrainingConfig
training_config = TrainingConfig(
    epochs=1,
    learning_rate=1e-3,
    batch_size=128,
    track_linguistic_probes=True,
    track_intrinsic_dimensions=True,
    track_hessian=True,
    track_component_hessian=True,
    track_pos_performance=True,
    track_semantic_roles_performance=True,
    track_interval=100,  # Analyze every 100 steps
    plots_path="./analysis_results"
)

# Create trainer
trainer = Trainer(training_config, tokenizer, model)

# Create dataloader
from trace.dataloader import get_dataloader
train_loader, val_loader, test_loader = get_dataloader(CORPUS_PATH, tokenizer=tokenizer, batch_size=32, max_length=MAX_SEQ_LENGTH, model_type=MODEL_TYPE, val_split= 0.1,test_split= 0.1)
best_loss, results = trainer.train(train_loader, val_loader, val_loader, model)
```
----
## Transformer Model Creation

TRACE provides a flexible transformer implementation supporting multiple architectures with consistent configuration patterns.

### Supported Architectures

- **Encoder-Only**
- **Decoder-Only**
- **Encoder-Decoder**

### Configuration System

```python
from trace.transformer import Transformer, TransformerConfig
import torch

# All models use the same configuration class
config = TransformerConfig(
    model_type="decoder_only",        # Architecture type
    vocab_size=30522,                 # Vocabulary size
    d_model=768,                      # Model dimension
    num_heads=12,                     # Attention heads
    num_encoder_layers=0,             # Encoder layers (if applicable)
    num_decoder_layers=6,             # Decoder layers (if applicable)
    d_ff=3072,                        # Feed-forward dimension
    max_seq_length=512,               # Maximum sequence length
    dropout=0.1,                      # Dropout rate
    device="cpu"                      # Device placement
)
```
### Decoder-Only Models

Optimized for autoregressive generation and language modeling:

```python
# Create a decoder-only model 
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
### Encoder-Only Models

Perfect for classification, encoding, and representation learning tasks:

```python
# Create an encoder-only model 
import torch
from trace.transformer import Transformer, TransformerConfig
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

# Forward pass with source and target sequences
src_ids = torch.randint(0, 32128, (2, 15))  # Source sequence
tgt_ids = torch.randint(0, 32128, (2, 10))  # Target sequence
logits = encoder_decoder_model(src=src_ids, tgt=tgt_ids)
print(f"Output shape: {logits.shape}")  # [2, 10, 32128]
```

## Intrinsic Dimensions Analysis

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
    layers_to_analyze=None,              # None = analyze all layers, or layers_to_analyze={'decoder': [0, 3, 5]},
    save_visualizations=True,            # Generate plots
    show_plots=False,                    # Display plots immediately
    log_dir="./analysis_results"     # Output directory
)

# Initialize analyzer
analyzer = IntrinsicDimensionAnalyzer(config)

# or simply use default settings
#analyzer = IntrinsicDimensionAnalyzer()

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
```

### Intrinsic Dimensions Visualization Options


```python
from trace.intrinsic_dimensions import IntrinsicDimensionsVisualizer

# Create custom visualizer
visualizer = IntrinsicDimensionsVisualizer(
    log_dir="./analysis_results",
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