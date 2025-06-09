from dataclasses import dataclass
from typing import Optional, List, Dict, Union
import skdim


@dataclass
class IntrinsicDimensionsConfig:
    """
    Configuration class for intrinsic dimensions analysis.

    This dataclass contains all hyperparameters and configuration options
    for performing intrinsic dimensions analysis on transformer models.
    """

    # Model architecture
    model_type: str = "decoder_only"
    layers_to_analyze: Optional[Union[List[int], Dict[str, List[int]]]] = None

    # Intrinsic dimension method
    id_method: str = "TwoNN"  # Can be "TwoNN", "MLE", "PCA", etc.

    # in case needed
    n_neighbors: int = 10

    # Data processing
    max_samples: Optional[int] = None  # Limit samples for memory efficiency
    flatten_sequence: bool = True  # Whether to flatten sequence dimension

    save_visualizations: bool = True  # Whether to save visualizations
    log_dir: Optional[str] = None

    def __post_init__(self):
        """Validate configuration parameters and create ID estimator."""
        if self.model_type not in ['encoder_only', 'decoder_only', 'encoder_decoder']:
            raise ValueError(f"Invalid model_type: {self.model_type}")

        if isinstance(self.layers_to_analyze, list):
            if not all(isinstance(layer, int) for layer in self.layers_to_analyze):
                raise ValueError("layers_to_analyze must be a list of integers")
        elif isinstance(self.layers_to_analyze, dict):
            if not all(isinstance(k, str) and isinstance(v, list) and all(isinstance(i, int) for i in v)
                       for k, v in self.layers_to_analyze.items()):
                raise ValueError("layers_to_analyze must be a dict with string keys and list of integers as values")

        # Create ID estimator based on method
        self.id_estimator = self._create_id_estimator()

    def _create_id_estimator(self):
        """Create the intrinsic dimension estimator."""
        if self.id_method == "TwoNN":
            return skdim.id.TwoNN()
        elif self.id_method == "MLE":
            return skdim.id.MLE()
        elif self.id_method == "PCA":
            return skdim.id.lPCA()
        else:
            raise ValueError(f"Unknown ID method: {self.id_method}")

    @classmethod
    def default(cls) -> 'IntrinsicDimensionsConfig':
        """Create a default configuration for intrinsic dimensions analysis."""
        return cls(
            model_type="decoder_only",
            layers_to_analyze=None,
            id_method="TwoNN"
        )