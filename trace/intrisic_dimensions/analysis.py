
# extract_hidden_states

# compute_intrinsic_dimension
from typing import Optional, List
from dataclasses import dataclass

from trace.intrisic_dimensions.config import IntrinsicDimensionsConfig


from trace.intrisic_dimensions.utils import (
    extract_hidden_representations,
    compute_intrinsic_dimensions
)


class IntrinsicDimensionAnalyzer:
    """
    Class for analyzing intrinsic dimensions of transformer models.

    This class provides methods to extract hidden states and compute intrinsic dimensions
    using the TwoNN method from skdim.
    """

    def __init__(self, config: Optional['IntrinsicDimensionsConfig'] = None):
        self.config = config or IntrinsicDimensionsConfig()

    def analyze(self, model, data_loader, layers: Optional[List[int]] = None) -> dict:
        """
        Analyze intrinsic dimensions of the model on the provided data loader.

        Args:
            model: The transformer model to analyze.
            data_loader: List of data batches to extract hidden states from.
            layers: Optional list of layer indices to analyze. If None, all layers are analyzed.

        Returns:
            Dictionary containing intrinsic dimensions for each layer.
        """

        hidden_states = extract_hidden_representations(model, data_loader, layers)[0]
        intrinsic_dimensions = compute_intrinsic_dimensions(hidden_states, self.config)

        return {
            "intrinsic_dimensions": intrinsic_dimensions,
            "layers": layers or list(range(len(hidden_states)))
        }