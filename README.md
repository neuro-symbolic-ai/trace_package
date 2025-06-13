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

# Using Individual Analysis Modules
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

## Hessian Analysis

TRACE provides comprehensive Hessian analysis capabilities for understanding loss landscape properties, training dynamics, and memorization patterns. This module analyzes curvature information to reveal insights about model optimization and generalization.

### Key Features

- **Eigenvalue Analysis**: Track extreme eigenvalues, trace estimates, and spectral properties
- **Component-Specific Analysis**: Analyze individual model components (attention, FFN, embeddings)
- **Gradient-Hessian Alignment**: Monitor optimization direction relative to curvature
- **Memorization Detection**: Compare train/validation landscapes to detect overfitting
- 
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
    log_dir="./analysis_results",             # Output directory
    save_hessian_data=True,                   # Save raw data
    show_plots=False                          # Display plots immediately
)

# Initialize analyzer
analyzer = HessianAnalyzer(config) # alternatively, use HessianAnalyzer.default() for default settings, HessianAnalyzer.minimal() for minimal settings, or HessianAnalyzer.comprehensive() for comprehensive settings
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

# Analyze all components
component_results = component_analyzer.analyze_all_components(
    model=decoder_model,
    loss_fn=loss_fn,
    data_batch=train_batch,
    model_type="decoder_only",
    component_list=None,  # None = analyze all components
    n_components=10
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

### Hessian Visualization

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

## Linguistic Probes

TRACE provides linguistic probing capabilities to analyze what linguistic knowledge models acquire during training. This module includes both probe training infrastructure and real-time monitoring systems for tracking linguistic understanding.

### Key Features

- **Multi-label Probing**: Train sophisticated probes to detect multiple linguistic features simultaneously
- **POS Analysis**: Track part-of-speech understanding (basic and detailed granularity)
- **Semantic Role Analysis**: Monitor semantic role labeling capabilities
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
    log_dir="./analysis_results",  # Training logs directory
)

# alternatively, use LinguisticProbesConfig.minimal() for minimal settings, LinguisticProbesConfig.comprehensive() for comprehensive settings, or LinguisticProbesConfig.default() for default settings

# Initialize trainer
probe_trainer = ProbeTrainer(config, tokenizer)

# Train probes on your model
training_results = probe_trainer.train_all_probes(
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

### Phase 2: Real-time Monitoring

Once you have trained probes, use them to monitor linguistic understanding during training:

```python
from trace.linguistic_probes import POSAnalyzer, SemanticAnalyzer

# Configure monitoring (note: different config than training)
monitoring_config = LinguisticProbesConfig(
    track_pos=True,
    track_semantic=True,
    layer_indices={'decoder': [0, 6, 11]},  # Monitor specific layers
    log_dir="./analysis_results",
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

### Probes Visualization

Generate detailed analysis visualizations:

```python
from trace.linguistic_probes import ProbesVisualizer

# Initialize visualizer
visualizer = ProbesVisualizer(
    log_dir="./analysis_results",
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
```

## Output Monitoring

TRACE's Output Monitoring module provides real-time analysis of model output quality by tracking linguistic accuracy across different grammatical and semantic categories during training. 

### Key Features
- **POS Accuracy Tracking**: Track part-of-speech accuracy by category (nouns, verbs, adjectives, etc.)
- **Semantic Role Monitoring**: Monitor semantic role labeling accuracy (agents, patients, actions, etc.)
- **Category-specific Insights**: Identify which linguistic categories your model struggles with most
- **Training Evolution Visualization**: See how output quality improves over training steps

### Basic usage

```python
from trace.output_monitoring import OutputMonitoringAnalyzer, OutputMonitoringConfig

# Configure output monitoring
config = OutputMonitoringConfig(
    track_pos_performance=True,        # Track POS accuracy
    track_semantic_roles=True,         # Track semantic role accuracy
    pos_granularity='basic',           # 'basic' or 'detailed'
    semantic_granularity='basic',      # 'basic' or 'detailed'
    save_visualizations=True,          # Generate plots
    log_dir="./analysis_results",     # Save results here
    show_plots=False                   # Don't display plots during training
)

# Initialize analyzer
output_analyzer = OutputMonitoringAnalyzer(config) # alternatively, use OutputMonitoringAnalyzer.default() for default settings

# Integrate into training loop
def training_step_with_output_monitoring(model, batch, step):
    # Forward pass
    outputs = model(**batch)
    logits = outputs.logits
    
    # Calculate loss and do backward pass
    loss = loss_function(logits, batch['labels'])
    loss.backward()
    optimizer.step()
    all_monitoring_results = {}
    # Monitor output quality every 50 steps
    if step % 50 == 0:
        monitoring_results = output_analyzer.analyze(
            batch=batch,
            outputs=logits,
            tokenizer=output_analyzer,
            step=step
        )
        
        # Log results
        if 'pos_accuracy' in monitoring_results:
            for category, accuracy in monitoring_results['pos_accuracy'].items():
                print(f"Step {step} - POS {category}: {accuracy:.3f}")
        
        
        if 'semantic_accuracy' in monitoring_results:
            for category, accuracy in monitoring_results['semantic_accuracy'].items():
                print(f"Step {step} - Semantic {category}: {accuracy:.3f}")
                
        all_monitoring_results[step] = monitoring_results

    # Generate final analysis
    output_analyzer.visualizer.plot_pos_performance_evolution(
        output_analyzer, "my_model", save_plot=True
    )
    
    output_analyzer.visualizer.plot_semantic_role_performance_evolution(
        all_monitoring_results, "my_model", save_plot=True
    )
    
    return all_monitoring_results
```

### Output Visualization

Generate comprehensive visualizations and export data:

```python
from trace.output_monitoring import OutputMonitoringVisualizer

# Initialize visualizer
output_visualizer = OutputMonitoringVisualizer(
    log_dir="./monitoring_visualizations",
    config=config
)

# Plot POS accuracy evolution
output_visualizer.plot_pos_performance_evolution(
    monitoring_results=all_monitoring_results,
    model_name="transformer_v2",
    save_plot=True
)

# Plot semantic role accuracy evolution
output_visualizer.plot_semantic_role_performance_evolution(
    monitoring_results=all_monitoring_results,
    model_name="transformer_v2", 
    save_plot=True
)

# Save metrics to CSV for further analysis
output_visualizer.save_metrics(
    monitoring_results=all_monitoring_results,
    model_name="transformer_v2"
)