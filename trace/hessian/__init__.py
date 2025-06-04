from .analysis import HessianAnalyzer
from .config import HessianConfig
from .components import ComponentAnalyzer, ComponentSelector
from .metrics import HessianMetrics
from .visualization import HessianVisualizer
from .utils import (
    compute_loss,
    extract_component_parameters,
    get_hessian_eigenvectors
)

# Legacy function imports for backward compatibility with pre_training.py
from .analysis import (
    compute_hessian_gradient_alignment,
    measure_train_val_landscape_divergence,
    compute_detailed_hessian_metrics
)
from .visualization import (
    plot_hessian_evolution,
    plot_component_comparison,
    plot_train_val_landscape_divergence_metrics
)
from .components import (compute_component_hessians)

__all__ = [
    # Main classes
    'HessianAnalyzer',
    'HessianConfig',
    'ComponentAnalyzer',
    'ComponentSelector',
    'HessianMetrics',
    'HessianVisualizer',

    # Utility functions
    'compute_loss',
    'extract_component_parameters',
    'get_hessian_eigenvectors',

    # Detailed functions
    'compute_component_hessians',
    'compute_hessian_gradient_alignment',
    'measure_train_val_landscape_divergence',
    'compute_detailed_hessian_metrics',

    # Visualization functions
    'plot_hessian_evolution',
    'plot_component_comparison',
    'plot_train_val_landscape_divergence_metrics'
]