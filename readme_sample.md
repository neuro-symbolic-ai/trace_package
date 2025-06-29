# Hessian Module Usage Examples

The `project.hessian` module provides comprehensive Hessian analysis capabilities for transformer models, enabling deep insights into optimization landscapes, memorization, and component-specific behavior.

## Basic Usage

### Quick Analysis with Default Settings

```python
from project.hessian import HessianAnalyzer, HessianConfig
from project.transformer import TransformerFactory
import torch

# Create model and sample data
model = TransformerFactory.create_decoder_only_transformer(vocab_size=1000)
loss_fn = torch.nn.CrossEntropyLoss()

# Sample batch (you would use your actual data)
batch = {
    "input_ids": torch.randint(0, 1000, (4, 20)),
    "labels": torch.randint(0, 1000, (4, 20))
}

# Create analyzer with default configuration
analyzer = HessianAnalyzer()

# Perform analysis for a training step
results = analyzer.analyze_step(
    model=model,
    loss_fn=loss_fn,
    train_batch=batch,
    model_type="decoder_only",
    step=100
)

print(f"Max eigenvalue: {results['hessian']['max_eigenvalue']}")
print(f"Trace estimate: {results['hessian']['hessian_trace_estimate']}")
print(f"Negative eigenvalues: {results['hessian']['negative_count']}")

# Get human-readable summary
summary = analyzer.get_analysis_summary(results)
print(summary)
```

### Configuration-Based Analysis

```python
from project.hessian import HessianConfig, HessianAnalyzer

# Create custom configuration
config = HessianConfig(
    n_components=20,  # More eigenvalues for detailed analysis
    track_component_hessian=True,
    track_gradient_alignment=True,
    track_train_val_landscape_divergence=True,
    track_sharpness=True,
    component_list=["attention", "ffn", "embeddings"]
)

# Create analyzer with custom config
analyzer = HessianAnalyzer.from_config(config)

# Perform comprehensive analysis
results = analyzer.analyze_step(
    model, loss_fn, train_batch, val_batch, "decoder_only", step=500
)

# Access different analysis results
hessian_metrics = results["hessian"]
component_results = results["components"]
alignment_metrics = results["alignment"]
memorization_signals = results["train_val_divergence"]
sharpness_metrics = results["sharpness"]
```

## Configuration Options

### Predefined Configurations

```python
# Minimal analysis (fast)
minimal_config = HessianConfig.minimal(n_components=5)

# Default analysis (balanced)
default_config = HessianConfig.default()

# Comprehensive analysis (detailed)
comprehensive_config = HessianConfig.comprehensive(
    n_components=50,
    component_list=[
        "attention", "attention_query", "attention_key", "attention_value",
        "ffn", "embeddings", "norm", "hidden_states"
    ]
)
```

### Custom Configuration

```python
custom_config = HessianConfig(
    n_components=15,
    num_batches=200,
    device="cuda",
    track_component_hessian=True,
    component_list=["attention", "ffn"],
    track_gradient_alignment=True,
    track_sharpness=True,
    track_train_val_landscape_divergence=True,
    tolerance=0.0001,
    save_hessian_data=True,
    create_plots=True
)
```

## Component Analysis

### Analyze Specific Components

```python
from project.hessian import ComponentAnalyzer, ComponentSelector

# Initialize component analyzer
component_analyzer = ComponentAnalyzer()

# Get available components for your model
standard_components = ComponentSelector.get_standard_components(no_fnn=False)
print(f"Standard components: {standard_components}")

# Validate components exist in model
valid_components = ComponentSelector.validate_components(model, standard_components)

# Analyze all valid components
component_results = component_analyzer.analyze_all_components(
    model=model,
    loss_fn=loss_fn,
    data_batch=batch,
    model_type="decoder_only",
    component_list=valid_components,
    n_components=10
)

# Compare components
comparison = component_analyzer.compare_components(
    component_results, 
    metric_name="max_eigenvalue"
)
print(f"Component with highest max eigenvalue: {comparison['max_component']}")

# Get summary
summary = component_analyzer.get_component_summary(component_results)
print(f"Total parameters analyzed: {summary['total_parameters']}")
```

### Component Selection

```python
# Different component selection strategies
minimal_components = ComponentSelector.get_minimal_components()
comprehensive_components = ComponentSelector.get_comprehensive_components()

# Custom component selection
custom_components = ["attention_query", "attention_key", "ffn"]
valid_custom = ComponentSelector.validate_components(model, custom_components)
```

## Advanced Analysis

### Multi-Batch Analysis

```python
# Analyze across multiple batches for robust estimates
batches = [batch1, batch2, batch3, batch4, batch5]

batch_analysis = analyzer.compute_batch_analysis(
    model=model,
    loss_fn=loss_fn,
    data_batches=batches,
    model_type="decoder_only"
)

print(f"Mean max eigenvalue: {batch_analysis['max_eigenvalue_mean']:.2e}")
print(f"Std max eigenvalue: {batch_analysis['max_eigenvalue_std']:.2e}")
```

### Gradient-Hessian Alignment

```python
# Analyze how gradients align with curvature
analyzer = HessianAnalyzer(HessianConfig(track_gradient_alignment=True))

results = analyzer.analyze_step(model, loss_fn, batch, step=1000)
alignment = results["alignment"]

print(f"Gradient-Hessian alignment: {alignment['grad_Hg_alignment']}")
print(f"Weighted alignment: {alignment['weighted_alignment']}")
print(f"Curvature-to-gradient ratio: {alignment['grad_Hg_ratio']}")
```

### Sharpness Analysis

```python
# Analyze loss landscape sharpness
config = HessianConfig(track_sharpness=True)
analyzer = HessianAnalyzer(config)

results = analyzer.analyze_step(model, loss_fn, batch, step=1500)
sharpness = results["sharpness"]

print(f"Max sharpness: {sharpness['max_sharpness']}")
print(f"Mean sharpness: {sharpness['mean_sharpness']}")
print(f"Spectral norm: {sharpness['spectral_norm']}")
```

### Memorization Detection

```python
# Compare training vs validation Hessian properties
config = HessianConfig(track_train_val_landscape_divergence=True)
analyzer = HessianAnalyzer(config)

results = analyzer.analyze_step(
    model=model,
    loss_fn=loss_fn,
    train_batch=train_batch,
    val_batch=val_batch,
    step=2000
)

memorization = results["train_val_divergence"]
print(f"Memorization score: {memorization['train_val_landscape_divergence_score']}")
print(f"Trace ratio (train/val): {memorization['trace_ratio']}")
print(f"Distribution overlap: {memorization['eigenvalue_distribution_overlap']}")
```

### Training Loop Integration

```python
# Example integration into training loop
analyzer = HessianAnalyzer(HessianConfig.default())
hessian_history = {}

for epoch in range(num_epochs):
    for step, (train_batch, val_batch) in enumerate(dataloader):
        # ... normal training code ...
        
        # Perform Hessian analysis every N steps
        if step % analysis_interval == 0:
            results = analyzer.analyze_step(
                model, loss_fn, train_batch, val_batch, 
                model_type="decoder_only", step=step
            )
            hessian_history[step] = results
            
            # Print summary
            summary = analyzer.get_analysis_summary(results)
            print(summary)
```

## Comprehensive Visualization

### Create Full Analysis Report

```python
from project.hessian import HessianVisualizer

# After collecting history during training
visualizer = HessianVisualizer()

# Create comprehensive report with all plots
visualizer.create_comprehensive_report(
    hessian_history={step: results["hessian"] for step, results in hessian_history.items()},
    # component_history={step: results["components"] for step, results in hessian_history.items()}, #todo there is an error 
    alignment_history={step: results["alignment"] for step, results in hessian_history.items()},
    memorization_history={step: results["train_val_divergence"] for step, results in hessian_history.items()},
    save_path="./hessian_plots",
    model_name="my_transformer"
)
```

### Individual Plot Types

```python
# Plot eigenvalue evolution
HessianVisualizer.plot_eigenvalue_evolution(
    hessian_history, 
    save_path="./plots", 
    model_name="model_v1"
)

# Plot complexity metrics (new!)
HessianVisualizer.plot_complexity_metrics(
    hessian_history,
    save_path="./plots",
    model_name="model_v1"
)

# Plot component comparison
HessianVisualizer.plot_component_comparison(
    component_history,
    save_path="./plots",
    model_name="model_v1"
)

# Plot gradient alignment
HessianVisualizer.plot_gradient_alignment(
    alignment_history,
    save_path="./plots", 
    model_name="model_v1"
)

# Plot memorization metrics
HessianVisualizer.plot_memorization_metrics(
    memorization_history,
    save_path="./plots",
    model_name="model_v1"
)

# Save text summary (new!)
HessianVisualizer.save_analysis_summary(
    results, "./plots", "model_v1"
)
```

## Working with Legacy Code

### Backward Compatibility Functions

```python
# The module provides legacy functions for existing code
from project.hessian import (
    compute_component_hessians,
    compute_hessian_gradient_alignment,
    measure_train_val_landscape_divergence,
    compute_detailed_hessian_metrics
)

# These work exactly like the original functions
component_results = compute_component_hessians(
    model, loss_fn, batch, "decoder_only", 
    components=["attention", "ffn"], n_components=10
)

alignment_results = compute_hessian_gradient_alignment(
    model, loss_fn, batch, "decoder_only", n_components=10
)
```

## Configuration Management

### Dynamic Configuration Updates

```python
# Update analyzer configuration at runtime
analyzer = HessianAnalyzer()

# Update specific parameters
analyzer.update_config(
    n_components=25,
    track_sharpness=True,
    component_list=["attention", "ffn", "embeddings", "norm"]
)

# Perform analysis with updated config
results = analyzer.analyze_step(model, loss_fn, batch)
```

## Utility Functions

### Direct Eigenvalue Computation

```python
from project.hessian import get_hessian_eigenvectors, HessianMetrics

# Compute eigenvalues directly
eigenvalues, eigenvectors = get_hessian_eigenvectors(
    model=model,
    loss_fn=loss_fn,
    train_data_loader=batch,
    num_batches=100,
    device="cuda",
    n_top_vectors=10
)

# Compute detailed metrics
metrics = HessianMetrics.compute_detailed_hessian_metrics(eigenvalues)
print(f"Condition number: {metrics['condition_number']}")
print(f"Effective rank: {metrics['effective_rank_95']}")
print(f"Complexity score: {metrics['complexity_score']}")
```

### Component Parameter Extraction

```python
from project.hessian import extract_component_parameters

# Extract parameters for specific components
attention_params = extract_component_parameters(model, "attention")
ffn_params = extract_component_parameters(model, "ffn")

print(f"Attention parameters: {sum(p.numel() for p in attention_params)}")
print(f"FFN parameters: {sum(p.numel() for p in ffn_params)}")
```

## Error Handling and Debugging

```python
try:
    analyzer = HessianAnalyzer(config)
    results = analyzer.analyze_step(model, loss_fn, batch, step=100)
    
    # Check for errors in results
    if "error" in results:
        print(f"Analysis failed: {results['error']}")
    else:
        print("Analysis completed successfully")
        
        # Get detailed summary
        summary = analyzer.get_analysis_summary(results)
        print(summary)
        
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Validate configuration
try:
    config = HessianConfig(n_components=-1)  # Will raise ValueError
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

## Performance Tips

```python
# For large models, use fewer components for faster analysis
fast_config = HessianConfig(
    n_components=5,  # Fewer eigenvalues
    num_batches=50,  # Fewer batches
    track_component_hessian=False,  # Skip component analysis
    track_train_val_landscape_divergence=False  # Skip train/val comparison
)

# For detailed analysis, use more components but less frequently
detailed_config = HessianConfig(
    n_components=20,
    track_component_hessian=True,
    component_list=["attention", "ffn"]  # Limit to key components
)

# Analyze every 1000 steps instead of every 100
if step % 1000 == 0:
    results = analyzer.analyze_step(...)

# Use batch analysis for more robust estimates
if step % 5000 == 0:  # Less frequent but more comprehensive
    batches = collect_recent_batches(5)  # Your function to get batches
    batch_results = analyzer.compute_batch_analysis(model, loss_fn, batches)
```

## Advanced Use Cases

### Research Applications

```python
# Study optimization landscape evolution
analyzer = HessianAnalyzer(HessianConfig.comprehensive())

# Collect data throughout training
landscape_evolution = {}
for epoch in range(num_epochs):
    results = analyzer.analyze_step(model, loss_fn, batch, val_batch, step=epoch)
    landscape_evolution[epoch] = results
    
    # Print key insights
    if "train_val_divergence" in results:
        memo_score = results["train_val_divergence"]["train_val_landscape_divergence_score"]
        print(f"Epoch {epoch}: Memorization score = {memo_score:.4f}")

# Create research plots
HessianVisualizer.create_comprehensive_report(
    **landscape_evolution,
    save_path="./research_plots",
    model_name="research_model"
)
```ian import HessianConfig, HessianAnalyzer

# Create custom configuration
config = HessianConfig(
    n_components=20,  # More eigenvalues for detailed analysis
    track_component_hessian=True,
    track_gradient_alignment=True,
    track_train_val_landscape_divergence=True,
    component_list=["attention", "ffn", "embeddings"]
)

# Create analyzer with custom config
analyzer = HessianAnalyzer.from_config(config)

# Perform comprehensive analysis
results = analyzer.analyze_step(
    model, loss_fn, train_batch, val_batch, "decoder_only", step=500
)

# Access different analysis results
hessian_metrics = results["hessian"]
component_results = results["components"]
alignment_metrics = results["alignment"]
memorization_signals = results["train_val_divergence"]
```

## Configuration Options

### Predefined Configurations

```python
# Minimal analysis (fast)
minimal_config = HessianConfig.minimal(n_components=5)

# Default analysis (balanced)
default_config = HessianConfig.default()

# Comprehensive analysis (detailed)
comprehensive_config = HessianConfig.comprehensive(
    n_components=50,
    component_list=[
        "attention", "attention_query", "attention_key", "attention_value",
        "ffn", "embeddings", "norm", "hidden_states"
    ]
)
```

### Custom Configuration

```python
custom_config = HessianConfig(
    n_components=15,
    num_batches=200,
    device="cuda",
    track_component_hessian=True,
    component_list=["attention", "ffn"],
    track_gradient_alignment=True,
    track_sharpness=False,
    track_train_val_landscape_divergence=True,
    tolerance=0.0001,
    save_hessian_data=True,
    create_plots=True
)
```

## Component Analysis

### Analyze Specific Components

```python
from project.hessian import ComponentAnalyzer, ComponentSelector

# Initialize component analyzer
component_analyzer = ComponentAnalyzer()

# Get available components for your model
standard_components = ComponentSelector.get_standard_components(no_fnn=False)
print(f"Standard components: {standard_components}")

# Validate components exist in model
valid_components = ComponentSelector.validate_components(model, standard_components)

# Analyze all valid components
component_results = component_analyzer.analyze_all_components(
    model=model,
    loss_fn=loss_fn,
    data_batch=batch,
    model_type="decoder_only",
    component_list=valid_components,
    n_components=10
)

# Compare components
comparison = component_analyzer.compare_components(
    component_results, 
    metric_name="max_eigenvalue"
)
print(f"Component with highest max eigenvalue: {comparison['max_component']}")

# Get summary
summary = component_analyzer.get_component_summary(component_results)
print(f"Total parameters analyzed: {summary['total_parameters']}")
```

### Component Selection

```python
# Different component selection strategies
minimal_components = ComponentSelector.get_minimal_components()
comprehensive_components = ComponentSelector.get_comprehensive_components()

# Custom component selection
custom_components = ["attention_query", "attention_key", "ffn"]
valid_custom = ComponentSelector.validate_components(model, custom_components)
```

## Advanced Analysis

### Gradient-Hessian Alignment

```python
# Analyze how gradients align with curvature
analyzer = HessianAnalyzer(HessianConfig(track_gradient_alignment=True))

results = analyzer.analyze_step(model, loss_fn, batch, step=1000)
alignment = results["alignment"]

print(f"Gradient-Hessian alignment: {alignment['grad_Hg_alignment']}")
print(f"Weighted alignment: {alignment['weighted_alignment']}")
print(f"Curvature-to-gradient ratio: {alignment['grad_Hg_ratio']}")
```

### Memorization Detection

```python
# Compare training vs validation Hessian properties
config = HessianConfig(track_train_val_landscape_divergence=True)
analyzer = HessianAnalyzer(config)

results = analyzer.analyze_step(
    model=model,
    loss_fn=loss_fn,
    train_batch=train_batch,
    val_batch=val_batch,
    step=2000
)

memorization = results["train_val_divergence"]
print(f"Memorization score: {memorization['train_val_landscape_divergence_score']}")
print(f"Trace ratio (train/val): {memorization['trace_ratio']}")
print(f"Distribution overlap: {memorization['eigenvalue_distribution_overlap']}")
```

### Training Loop Integration

```python
# Example integration into training loop
analyzer = HessianAnalyzer(HessianConfig.default())
hessian_history = {}

for epoch in range(num_epochs):
    for step, (train_batch, val_batch) in enumerate(dataloader):
        # ... normal training code ...
        
        # Perform Hessian analysis every N steps
        if step % analysis_interval == 0:
            results = analyzer.analyze_step(
                model, loss_fn, train_batch, val_batch, 
                model_type="decoder_only", step=step
            )
            hessian_history[step] = results
            
            # Print key metrics
            metrics = results["hessian"]
            print(f"Step {step}: Max eigenvalue: {metrics['max_eigenvalue']:.2e}, "
                  f"Trace: {metrics['hessian_trace_estimate']:.2e}")
```

## Visualization

### Create Comprehensive Plots

```python
from project.hessian import HessianVisualizer

# After collecting history during training
visualizer = HessianVisualizer()

# Create all plots
visualizer.create_comprehensive_report(
    hessian_history={step: results["hessian"] for step, results in hessian_history.items()},
    component_history={step: results["components"] for step, results in hessian_history.items()},
    alignment_history={step: results["alignment"] for step, results in hessian_history.items()},
    memorization_history={step: results["train_val_divergence"] for step, results in hessian_history.items()},
    save_path="./hessian_plots",
    model_name="my_transformer"
)
```

### Individual Plot Types

```python
# Plot eigenvalue evolution
HessianVisualizer.plot_eigenvalue_evolution(
    hessian_history, 
    save_path="./plots", 
    model_name="model_v1"
)

# Plot component comparison
HessianVisualizer.plot_component_comparison(
    component_history,
    save_path="./plots",
    model_name="model_v1"
)

# Plot gradient alignment
HessianVisualizer.plot_gradient_alignment(
    alignment_history,
    save_path="./plots", 
    model_name="model_v1"
)

# Plot memorization metrics
HessianVisualizer.plot_memorization_metrics(
    memorization_history,
    save_path="./plots",
    model_name="model_v1"
)
```

## Working with Legacy Code

### Backward Compatibility Functions

```python
# The module provides legacy functions for existing code
from project.hessian import (
    compute_component_hessians,
    compute_hessian_gradient_alignment,
    measure_train_val_landscape_divergence,
    compute_detailed_hessian_metrics
)

# These work exactly like the original functions
component_results = compute_component_hessians(
    model, loss_fn, batch, "decoder_only", 
    components=["attention", "ffn"], n_components=10
)

alignment_results = compute_hessian_gradient_alignment(
    model, loss_fn, batch, "decoder_only", n_components=10
)
```

## Utility Functions

### Direct Eigenvalue Computation

```python
from project.hessian import get_hessian_eigenvectors, HessianMetrics

# Compute eigenvalues directly
eigenvalues, eigenvectors = get_hessian_eigenvectors(
    model=model,
    loss_fn=loss_fn,
    train_data_loader=batch,
    num_batches=100,
    device="cuda",
    n_top_vectors=10
)

# Compute detailed metrics
metrics = HessianMetrics.compute_detailed_hessian_metrics(eigenvalues)
print(f"Condition number: {metrics['condition_number']}")
print(f"Effective rank: {metrics['effective_rank_95']}")
```

### Component Parameter Extraction

```python
from project.hessian import extract_component_parameters

# Extract parameters for specific components
attention_params = extract_component_parameters(model, "attention")
ffn_params = extract_component_parameters(model, "ffn")

print(f"Attention parameters: {sum(p.numel() for p in attention_params)}")
print(f"FFN parameters: {sum(p.numel() for p in ffn_params)}")
```

## Error Handling and Debugging

```python
try:
    analyzer = HessianAnalyzer(config)
    results = analyzer.analyze_step(model, loss_fn, batch, step=100)
    
    # Check for errors in results
    if "error" in results:
        print(f"Analysis failed: {results['error']}")
    else:
        print("Analysis completed successfully")
        
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Validate configuration
try:
    config = HessianConfig(n_components=-1)  # Will raise ValueError
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

## Performance Tips

```python
# For large models, use fewer components for faster analysis
fast_config = HessianConfig(
    n_components=5,  # Fewer eigenvalues
    num_batches=50,  # Fewer batches
    track_component_hessian=False,  # Skip component analysis
    track_train_val_landscape_divergence=False  # Skip train/val comparison
)

# For detailed analysis, use more components but less frequently
detailed_config = HessianConfig(
    n_components=20,
    track_component_hessian=True,
    component_list=["attention", "ffn"]  # Limit to key components
)

# Analyze every 1000 steps instead of every 100
if step % 1000 == 0:
    results = analyzer.analyze_step(...)
```