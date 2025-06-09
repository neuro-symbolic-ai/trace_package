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
    nn_2 = skdim.id.TwoNN()

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.model_type not in ['encoder_only', 'decoder_only', 'encoder_decoder']:
            raise ValueError(f"Invalid model_type: {self.model_type}")

        if isinstance(self.layers_to_analyze, list):
            if not all(isinstance(layer, int) for layer in self.layers_to_analyze):
                raise ValueError("layers_to_analyze must be a list of integers")
        elif isinstance(self.layers_to_analyze, dict):
            if not all(isinstance(k, str) and isinstance(v, list) and all(isinstance(i, int) for i in v)
                       for k, v in self.layers_to_analyze.items()):
                raise ValueError("layers_to_analyze must be a dict with string keys and list of integers as values")

    @classmethod
    def default(cls) -> 'IntrinsicDimensionsConfig':
        """Create a default configuration for intrinsic dimensions analysis."""
        return cls(
            model_type="decoder_only",
            layers_to_analyze=None,
            nn_2=skdim.id.TwoNN()
        )
