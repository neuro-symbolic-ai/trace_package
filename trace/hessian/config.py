from dataclasses import dataclass
from typing import Optional, List


@dataclass
class HessianConfig:
    """
    Configuration class for Hessian analysis.
    """

    # Basic analysis parameters
    n_components: int = 10
    num_batches: int = 100
    device: Optional[str] = None

    # Component analysis settings
    track_component_hessian: bool = True
    component_list: Optional[List[str]] = None

    # Analysis toggles
    track_gradient_alignment: bool = True
    track_sharpness: bool = True
    track_train_val_landscape_divergence: bool = True

    # Computation settings
    tol: float = 0.001 # tol for eigenvalue convergence
    max_iterations: int = 1000

    # Output settings
    save_hessian_data: bool = True
    create_plots: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")

        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive")

        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")

        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        # Set default component list if not provided
        if self.component_list is None and self.track_component_hessian:
            self.component_list = ["attention", "ffn", "hidden_states"]

    @classmethod
    def default(cls, **kwargs) -> 'HessianConfig':
        """Create configuration with default settings."""
        defaults = {
            'n_components': 10,
            'track_component_hessian': True,
            'track_gradient_alignment': True,
            'track_sharpness': True,
            'track_train_val_landscape_divergence': True
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def minimal(cls, **kwargs) -> 'HessianConfig':
        """Create minimal configuration for basic analysis."""
        defaults = {
            'n_components': 5,
            'track_component_hessian': False,
            'track_gradient_alignment': False,
            'track_sharpness': False,
            'track_train_val_landscape_divergence': False
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def comprehensive(cls, **kwargs) -> 'HessianConfig':
        """Create comprehensive configuration for detailed analysis."""
        defaults = {
            'n_components': 20,
            'track_component_hessian': True,
            'track_gradient_alignment': True,
            'track_sharpness': True,
            'track_train_val_landscape_divergence': True,
            'component_list': ["attention", "attention_query", "attention_key",
                               "attention_value", "ffn", "embeddings", "norm"]
        }
        defaults.update(kwargs)
        return cls(**defaults)