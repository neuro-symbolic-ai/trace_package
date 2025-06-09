
from typing import Optional, List
from dataclasses import dataclass

from .config import IntrinsicDimensionsConfig
from .visualization import IntrinsicDimensionsVisualizer
from .utils import (
    extract_hidden_representations,
    compute_intrinsic_dimensions
)


class IntrinsicDimensionAnalyzer:
    """
    Class for analyzing intrinsic dimensions of transformer models.

    This class provides methods to extract hidden states and compute intrinsic dimensions
    using the TwoNN method from skdim.
    """

    def __init__(self, config: Optional[IntrinsicDimensionsConfig] = None):
        self.config = config or IntrinsicDimensionsConfig.default()
        if hasattr(self.config, 'log_dir') and self.config.log_dir is None:
            self.config.log_dir = "./plots/intrinsic_dimensions"
        else:
            log_dir = self.config.log_dir if hasattr(self.config, 'log_dir') else None
        self.visualizer = IntrinsicDimensionsVisualizer(
            self.config.log_dir if hasattr(self.config, 'log_dir') else log_dir,
            self.config
        ) if getattr(self.config, 'save_visualizations', True) else None

    def analyze(self,
                model,
                data_loader,
                layers: Optional[List[int]] = None,
                model_name: str = "") -> dict:
        """
        Analyze intrinsic dimensions of the model on the provided data loader.

        Args:
            model: The transformer model to analyze.
            data_loader: List of data batches to extract hidden states from.
            layers: Optional list of layer indices to analyze. If None, all layers are analyzed.

        Returns:
            Dictionary containing intrinsic dimensions for each layer.
        """
        print("Starting intrinsic dimension analysis...")
        if layers is not None:
            self.config.layers_to_analyze = layers
        print("Extracting hidden representations...")
        hidden_states, _, _ = extract_hidden_representations(model, data_loader, layers)
        print("Computing intrinsic dimensions...")
        intrinsic_dimensions = compute_intrinsic_dimensions(hidden_states, self.config)

        if self.visualizer:
            print("Visualizing intrinsic dimensions...")
            self.visualizer.generate_all_visualizations(intrinsic_dimensions, model_name)
        return intrinsic_dimensions



