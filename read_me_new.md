# TRACE: Transformer Analysis and Comprehension Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TRACE** is a comprehensive Python package for analyzing transformer models during training and inference. It provides modular tools for understanding model behavior through linguistic probes, intrinsic dimension analysis, Hessian landscape exploration, and more.

## 🚀 Features

- **🔍 Linguistic Probes**: Monitor syntactic and semantic understanding across layers
- **📐 Intrinsic Dimensions**: Track representation complexity using TwoNN and other methods  
- **🏔️ Hessian Analysis**: Explore loss landscapes, sharpness, and training dynamics
- **📊 Real-time Monitoring**: Comprehensive visualization during training
- **⚡ Modular Design**: Use individual components or full training integration
- **🔧 Easy Integration**: Drop-in replacement for existing training loops

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/nura-aj/trace.git
cd trace

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Dependencies

- PyTorch >= 1.9.0
- scikit-learn >= 1.0.0
- scikit-dimension
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- tqdm >= 4.62.0

## 🏃 Quick Start

### Basic Training with Analysis

```python
from trace.training import Trainer, TrainingConfig
from trace.linguistic_probes import LinguisticProbesConfig
from trace.intrisic_dimensions import IntrinsicDimensionsConfig

# Configure training with analysis
config = TrainingConfig(
    epochs=20,
    learning_rate=1e-3,
    batch_size=128,
    track_linguistic_probes=True,
    track_intrinsic_dimensions=True,
    track_interval=50,  # Analyze every 1000 steps
    plots_path="./analysis_results"
)

# Create trainer
trainer = Trainer(config, tokenizer, model)

# Train with comprehensive analysis
best_loss, results = trainer.train(train_loader, val_loader, test_loader)
```

### Using Individual Analysis Modules

```python
# Linguistic Probes Analysis
from trace.linguistic_probes import LinguisticProbesAnalyzer

probe_config = LinguisticProbesConfig(
    model_type="decoder_only",
    layers_to_analyze=[0, 1, 2],
    save_visualizations=True
)

analyzer = LinguisticProbesAnalyzer(probe_config)
results = analyzer.run_pos_analysis(model, dataloader, tokenizer)

# Intrinsic Dimensions Analysis  
from trace.intrinsic_dimensions import IntrinsicDimensionAnalyzer

id_analyzer = IntrinsicDimensionAnalyzer()
id_results = id_analyzer.analyze(model, dataloader, model_name="my_model")
```

### Backward Compatibility

```python
# Drop-in replacement for existing training functions
from trace.training import train_model

best_loss, results = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    lr=1e-3,
    device=device,
    save_path="model.pt",
    tokenizer=tokenizer,
    track_linguistic_probes=True,
    track_intrinsic_dimensions=True
)
```

## 📋 Analysis Modules

### 🔤 Linguistic Probes

Monitor syntactic (POS) and semantic understanding across transformer layers:

```python
from trace.linguistic_probes import LinguisticProbesAnalyzer, LinguisticProbesConfig

config = LinguisticProbesConfig(
    model_type="decoder_only",
    layers_to_analyze=[0, 1, 2, 3],
    probe_type="linear",
    save_visualizations=True,
    log_dir="./probe_analysis"
)

analyzer = LinguisticProbesAnalyzer(config)

# Analyze POS tagging capability
pos_results = analyzer.run_pos_analysis(model, dataloader, tokenizer)

# Analyze semantic role understanding  
semantic_results = analyzer.run_semantic_analysis(model, dataloader, tokenizer)
```

**Generated Visualizations:**
- Learning curves by POS/semantic category
- Layer-wise performance comparison
- Accuracy evolution heatmaps
- Category distribution plots

### 📐 Intrinsic Dimensions

Track representation complexity and dimensionality:

```python
from trace.intrinsic_dimensions import IntrinsicDimensionAnalyzer, IntrinsicDimensionsConfig

config = IntrinsicDimensionsConfig(
    model_type="decoder_only",
    id_method="TwoNN",  # or "MLE", "PCA"
    layers_to_analyze=[0, 1, 2, 3]
)

analyzer = IntrinsicDimensionAnalyzer(config)

# Single analysis
id_results = analyzer.analyze(model, dataloader)

# Evolution analysis during training
evolution_results = analyzer.analyze_evolution(
    model, 
    [dataloader_step1, dataloader_step2, dataloader_step3],
    [100, 200, 300],  # step numbers
    model_name="training_analysis"
)

# Compare multiple models
model_results = {
    "small_model": small_model_ids,
    "large_model": large_model_ids
}
analyzer.compare_models(model_results)
```

**Generated Visualizations:**
- ID progression across layers
- ID distribution histograms
- Temporal evolution during training
- Multi-model comparisons

### 🏔️ Hessian Analysis *(Coming Soon)*

Explore loss landscapes and training dynamics:

```python
# Will be available after module refactoring
from trace.hessian_analysis import HessianAnalyzer

analyzer = HessianAnalyzer(config)
hessian_results = analyzer.analyze(model, dataloader)
```

## 📊 Visualization Features

### Automatic Plot Generation

All analysis modules automatically generate publication-quality plots:

```
analysis_results/
├── linguistic_probes/
│   ├── model_pos_accuracy.png
│   ├── model_semantic_accuracy.png  
│   ├── model_pos_heatmap.png
│   └── model_layer_comparison.png
├── intrinsic_dimensions/
│   ├── model_id_by_layer.png
│   ├── model_id_distribution.png
│   ├── model_id_evolution.png
│   └── model_id_heatmap.png
└── training/
    ├── training_curves.png
    └── gradient_analysis.png
```

### Custom Visualization

```python
from trace.linguistic_probes import ProbesVisualizer
from trace.intrinsic_dimensions import IntrinsicDimensionsVisualizer

# Custom probe visualization
probe_viz = ProbesVisualizer(log_dir="./custom_plots")
probe_viz.plot_learning_curves(accuracy_data, sample_counts, "my_model")

# Custom ID visualization  
id_viz = IntrinsicDimensionsVisualizer(log_dir="./custom_plots")
id_viz.plot_id_evolution(evolution_data, "my_model")
```

## ⚙️ Configuration

### Training Configuration

```python
from trace.training import TrainingConfig

config = TrainingConfig(
    # Model architecture
    model_type="decoder_only",
    d_model=512,
    num_heads=8,
    num_decoder_layers=6,
    
    # Training parameters
    epochs=50,
    learning_rate=1e-4,
    batch_size=64,
    warmup_steps=1000,
    weight_decay=0.01,
    
    # Analysis tracking
    track_interval=500,
    track_linguistic_probes=True,
    track_intrinsic_dimensions=True,
    track_hessian=True,
    
    # Probe configuration
    probe_layers=[0, 2, 4, 5],
    probe_load_path="./trained_probes/",
    
    # Paths
    save_path="./models/model.pt",
    plots_path="./analysis_results"
)
```

### Analysis Module Configurations

```python
# Linguistic probes configuration
from trace.linguistic_probes import LinguisticProbesConfig

probe_config = LinguisticProbesConfig(
    model_type="decoder_only",
    layers_to_analyze=[0, 1, 2, 3],
    probe_type="linear",
    hidden_dim=256,
    learning_rate=1e-4,
    epochs=30,
    save_visualizations=True
)

# Intrinsic dimensions configuration
from trace.intrinsic_dimensions import IntrinsicDimensionsConfig

id_config = IntrinsicDimensionsConfig(
    model_type="decoder_only", 
    id_method="TwoNN",
    max_samples=10000,
    flatten_sequence=True
)
```

## 🔧 Advanced Usage

### Custom Analysis Integration

```python
from trace.training import TrainingCallbacks

class CustomCallbacks(TrainingCallbacks):
    def run_analysis(self, model, batch, hidden_states, step, val_loader=None):
        # Call parent analysis
        super().run_analysis(model, batch, hidden_states, step, val_loader)
        
        # Add your custom analysis
        if step % 1000 == 0:
            custom_metric = your_custom_analysis(model, hidden_states)
            print(f"Custom metric at step {step}: {custom_metric}")

# Use custom callbacks
trainer = Trainer(config, tokenizer, model)
trainer.callbacks = CustomCallbacks(config, tokenizer, device)
```

### Layer-Specific Analysis

```python
# Analyze specific layers
config = LinguisticProbesConfig(
    layers_to_analyze=[0, 3, 5],  # First, middle, last layers
    analysis_scope="targeted"
)

# Or analyze all layers
config = LinguisticProbesConfig(
    layers_to_analyze=None,  # Auto-detect all layers
    analysis_scope="comprehensive"
)
```

### Evolution Tracking

```python
# Track changes during training
evolution_data = {}

for epoch in range(num_epochs):
    # ... training code ...
    
    if epoch % 5 == 0:  # Analyze every 5 epochs
        results = analyzer.analyze(model, dataloader, f"epoch_{epoch}")
        evolution_data[epoch] = results

# Visualize evolution
analyzer.visualizer.plot_evolution(evolution_data, "training_evolution")
```

## 📖 Examples

### Complete Training Example

```python
import torch
from torch.utils.data import DataLoader
from trace.training import Trainer, TrainingConfig

# Set up data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Configure comprehensive analysis
config = TrainingConfig(
    model_type="decoder_only",
    epochs=30,
    learning_rate=1e-3,
    track_interval=500,
    
    # Enable all analysis modules
    track_linguistic_probes=True,
    track_intrinsic_dimensions=True,
    track_hessian=True,
    
    # Probe settings
    probe_load_path="./probes/",
    probe_layers=[0, 1, 2, 3],
    
    # Output settings
    plots_path="./comprehensive_analysis",
    save_path="./models/analyzed_model.pt"
)

# Train with analysis
trainer = Trainer(config, tokenizer, model)
best_loss, analysis_results = trainer.train(train_loader, val_loader)

print(f"Training completed! Best loss: {best_loss:.4f}")
print(f"Analysis results keys: {list(analysis_results.keys())}")
```

### Model Comparison Example

```python
from trace.intrinsic_dimensions import IntrinsicDimensionAnalyzer

# Analyze multiple models
models = {
    "small_transformer": small_model,
    "large_transformer": large_model,
    "distilled_model": distilled_model
}

analyzer = IntrinsicDimensionAnalyzer()
comparison_results = {}

for name, model in models.items():
    results = analyzer.analyze(model, dataloader, name)
    comparison_results[name] = results

# Generate comparison plots
analyzer.compare_models(comparison_results, "model_comparison")
```

## 📚 API Reference

### Core Classes

- **`Trainer`**: Main training orchestrator with integrated analysis
- **`TrainingConfig`**: Comprehensive training configuration
- **`LinguisticProbesAnalyzer`**: POS and semantic probe analysis
- **`IntrinsicDimensionAnalyzer`**: Representation dimensionality analysis
- **`ProbesVisualizer`**: Linguistic probe visualization tools
- **`IntrinsicDimensionsVisualizer`**: ID analysis visualization tools

### Utility Functions

- **`train_model()`**: Backward-compatible training function
- **`prepare_batch_for_model()`**: Batch preparation for different model types
- **`setup_hidden_state_hooks()`**: Automatic hidden state capture
- **`validate_model()`**: Model validation utilities

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/your-username/trace.git
cd trace
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black trace/
isort trace/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: [https://trace-analysis.readthedocs.io](https://trace-analysis.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-username/trace/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/trace/discussions)

## 🙏 Acknowledgments

- Built on PyTorch and scikit-dimension
- Inspired by research in transformer interpretability
- Thanks to all contributors and the open-source community

## 📈 Roadmap

- [ ] **Hessian Analysis Module** - Complete refactoring and integration
- [ ] **POS Performance Tracking** - Advanced syntactic analysis
- [ ] **Semantic Role Tracking** - Comprehensive semantic understanding
- [ ] **Attention Visualization** - Layer-wise attention pattern analysis
- [ ] **Interactive Dashboards** - Real-time monitoring interfaces
- [ ] **Multi-GPU Support** - Distributed analysis capabilities
- [ ] **Pre-trained Probe Library** - Ready-to-use linguistic probes

---

**TRACE** - Understanding transformers, one layer at a time. 🔍✨

# TRACE: Transformer Analysis and Comprehensive Evaluation

**TRACE** is a comprehensive Python package for analyzing transformer models during training. It provides modular tools for understanding model behavior through linguistic probes, intrinsic dimension analysis, Hessian landscape exploration, and more.

**Note**: TRACE is designed to work with synthetic data from the [ABSynth dataset](https://github.com/nura-j/ABSynth_dataset), which provides controlled linguistic structures for systematic analysis of transformer learning dynamics.

## 📖 Documentation

### Table of Contents

1. [Basic Usage](#-basic-usage)
2. [Transformer Model Creation](#transformer-model-creation)
3. [Intrinsic Dimensions Analysis](#intrinsic-dimensions-analysis)
4. [Hessian Analysis](#hessian-analysis)
5. [Linguistic Probes](#linguistic-probes)
6. [Output Monitoring](#output-monitoring)
7. [Integration Examples](#integration-examples)
8. [Configuration Reference](#configuration-reference)
9. [Contributing](#contributing)
10. [License](#license)
11. [Citation](#citation)

## 🚀 Basic Usage

Get started with TRACE in just a few lines of code:

```python
# Create tokenizer from your corpus
from trace.tokenizer import create_tokenizer_from_data

CORPUS_PATH = "path/to/your/absynth_corpus.json"  # Your ABSynth data
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

## Integration Examples

### Minimal Training Setup

For quick experimentation with minimal analysis overhead:

```python
from trace.training import TrainingConfig, Trainer

# Lightweight configuration
config = TrainingConfig(
    epochs=5,
    learning_rate=1e-4,
    batch_size=32,
    
    # Minimal analysis
    track_hessian=True,
    hessian_n_components=3,
    track_pos_performance=True,
    track_interval=200,
    save_visualization=False
)

trainer = Trainer(config, tokenizer, model)
best_loss, results = trainer.train(train_loader, val_loader)
```

### Research-Focused Setup

For comprehensive analysis and detailed research insights:

```python
# Comprehensive research configuration
research_config = TrainingConfig(
    epochs=20,
    learning_rate=5e-5,
    batch_size=16,
    
    # Full analysis suite
    track_hessian=True,
    track_component_hessian=True,
    track_gradient_alignment=True,
    track_train_val_landscape_divergence=True,
    hessian_n_components=10,
    
    track_linguistic_probes=True,
    track_semantic_probes=True,
    track_intrinsic_dimensions=True,
    track_pos_performance=True,
    track_semantic_roles_performance=True,
    
    # Frequent analysis for detailed tracking
    track_interval=50,
    save_visualization=True,
    show_plots=False,
    
    # Pre-trained probe paths
    probe_load_paths={(0, 'decoder'): './probes/pos_layer0.pt'},
    semantic_probe_load_path={(0, 'decoder'): './probes/semantic_layer0.pt'}
)
```

### Custom Analysis Pipeline

For advanced users who want to use individual analysis modules:

```python
from trace.hessian import HessianAnalyzer
from trace.linguistic_probes import POSAnalyzer
from trace.intrinsic_dimensions import IntrinsicDimensionAnalyzer

# Initialize individual analyzers
hessian_analyzer = HessianAnalyzer.comprehensive()
pos_analyzer = POSAnalyzer()
id_analyzer = IntrinsicDimensionAnalyzer()

# Custom training loop with selective analysis
for epoch in range(epochs):
    for step, batch in enumerate(train_loader):
        # Regular training step
        outputs = model(**batch)
        loss = criterion(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
        
        # Custom analysis scheduling
        if step % 100 == 0:
            # Hessian analysis every 100 steps
            hessian_results = hessian_analyzer.analyze_step(
                model, criterion, batch, step=step
            )
            
        if step % 200 == 0:
            # Linguistic analysis every 200 steps
            pos_results = pos_analyzer.analyze(
                model, [batch], tokenizer, f"step_{step}"
            )
            
        if step % 500 == 0:
            # Intrinsic dimensions every 500 steps
            id_results = id_analyzer.analyze(
                model, [batch], f"step_{step}"
            )
```

## Configuration Reference

### TrainingConfig Parameters

**Core Training**
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimizer
- `batch_size`: Training batch size
- `device`: Training device ("cpu" or "cuda")

**Analysis Modules**
- `track_hessian`: Enable Hessian analysis
- `track_linguistic_probes`: Enable POS probe analysis
- `track_semantic_probes`: Enable semantic probe analysis
- `track_intrinsic_dimensions`: Enable intrinsic dimension analysis
- `track_pos_performance`: Enable output POS monitoring
- `track_semantic_roles_performance`: Enable output semantic monitoring

**Analysis Frequency**
- `track_interval`: Steps between analyses (default: 100)
- `save_visualization`: Generate plots (default: True)
- `show_plots`: Display plots during training (default: False)

**Output Paths**
- `plots_path`: Directory for analysis results
- `save_path`: Model checkpoint save path
- `log_dir`: Analysis logging directory

### Model Configuration Options

**Architecture Types**
- `"encoder_only"`: BERT-style models
- `"decoder_only"`: GPT-style models  
- `"encoder_decoder"`: T5-style models

**Key Parameters**
- `vocab_size`: Vocabulary size
- `d_model`: Hidden dimension
- `num_heads`: Attention heads
- `num_encoder_layers`/`num_decoder_layers`: Layer counts
- `d_ff`: Feed-forward dimension
- `max_seq_length`: Maximum sequence length

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Code style and testing requirements
- Submitting pull requests
- Reporting issues

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TRACE in your research, please cite:

```bibtex
@software{trace2024,
  title={TRACE: Transformer Analysis and Comprehensive Evaluation},
  author={Your Name and Contributors},
  year={2024},
  url={https://github.com/nura-j/trace_package},
  version={1.0.0}
}