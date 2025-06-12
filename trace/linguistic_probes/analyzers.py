# import torch
# from torch.utils.data import DataLoader
# from typing import Dict, List, Optional, Any, Union
# from tqdm import tqdm
#
# from .config import LinguisticProbesConfig
# from .models import MultiLabelProbe, LinearProbe
# from .utils import extract_hidden_representations_with_pos_semantic, prepare_probing_dataset
# from .visualization import ProbesVisualizer
#
#
# class BaseAnalyzer:
#     """
#     Base class for probe analyzers.
#     """
#
#     def __init__(self, config: Optional[LinguisticProbesConfig] = None,
#                  multilinear_probe: Optional[MultiLabelProbe] = None,
#                  linear_probe: Optional[LinearProbe] = None):
#
#         self.config = config or LinguisticProbesConfig.default()
#         self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.model_trained = False  # Flag to track if the model has been trained
#         # I should set the probe model here
#         try:
#             if multilinear_probe is not None:
#                 self.probe_model = multilinear_probe
#             elif linear_probe is not None:
#                 self.probe_model = linear_probe
#
#         except ValueError as e:
#             print(f"Error initializing probe model: {e}")
#             raise
#
#         self.visualizer = ProbesVisualizer(self.config.log_dir,
#                                            self.config) if self.config.save_visualizations else None
#
#     def get_analysis_type(self) -> str:
#         """Get the type of analysis (to be overridden by subclasses)."""
#         raise NotImplementedError
#
#     def get_label_names(self) -> List[str]:
#         """Get label names for this analysis type (to be overridden by subclasses)."""
#         raise NotImplementedError
#
#     def run_probe_analysis(
#             self,
#             model,
#             dataloader: DataLoader,
#             tokenizer,
#             layer_indices: Optional[Union[List[int], Dict[str, List[int]]]] = None,
#             model_name: str = ""
#     ) -> Dict[int, Any]:
#         """
#         Run probing analysis on specified layers.
#
#         Args:
#             model: The transformer model to analyze
#             dataloader: DataLoader with input data
#             tokenizer: Tokenizer for decoding sequences
#             layer_indices: Which layers to probe (None for default)
#             model_name: Name of the model for saving/logging
#             model_name: Name of the model for saving/logging
#
#         Returns:
#            this should return the confidence scores of tags existance in the hidden states
#         """
#         print(f"Starting {self.get_analysis_type()} probe analysis...")
#
#         # Extract representations and labels
#         hidden_states, pos_labels, semantic_labels = extract_hidden_representations_with_pos_semantic(
#             model, dataloader, self.device, layer_indices, tokenizer, self.config
#         )
#
#         # Select appropriate labels based on analysis type
#         if self.get_analysis_type() == 'pos':
#             labels = pos_labels
#         elif self.get_analysis_type() == 'semantic':
#             labels = semantic_labels
#         else:
#             raise ValueError(f"Unknown analysis type: {self.get_analysis_type()}")
#
#         if labels is None:
#             raise ValueError(f"No {self.get_analysis_type()} labels available")
#
#         results = {}
#
#         # Probe for each layer
#         for layer_idx, layer_hidden in hidden_states.items():
#             print(f"\nProbing layer {layer_idx} for {self.get_analysis_type()}:")
#
#             # Prepare dataset
#             probe_loader = prepare_probing_dataset(
#                 layer_hidden, labels, self.config.batch_size
#             )
#
#             # Initialize and train probe
#             input_dim = layer_hidden.shape[2]  # Hidden dimension size
#             num_features = labels.shape[1]  # Number of labels
#
#             # probe = MultiLabelProbe(
#             #     input_dim=input_dim,
#             #     num_features=num_features,
#             #     hidden_dim=self.config.hidden_dim,
#             #     lr=self.config.lr,
#             #     epochs=self.config.epochs,
#             #     device=self.device,
#             #     dropout=self.config.dropout
#             # )
#
#             # Train the probe
#             # training_losses = probe.train_probe(
#             # training_losses=self.probe_model.train_probe(
#             #     probe_loader,
#             #     use_class_weights=self.config.use_class_weights,
#             #     penalize_classes=self.config.penalized_classes if self.config.penalize_frequent_classes else None
#             # )
#
#             # Evaluate the probe
#             # evaluation_results = probe.evaluate_probe(
#             #     probe_loader,
#             #     label_names=self.get_label_names()
#             # )
#
#             # Print evaluation report
#             probe.print_evaluation_report(evaluation_results)
#
#             # Save probe if requested
#             if self.config.save_probes and self.config.save_path:
#                 name_layer_idx = str(layer_idx) if isinstance(layer_idx, int) else f"{layer_idx[0]}_{layer_idx[1]}"
#                 name_layer_idx = name_layer_idx.replace("(", "").replace(")", "")
#                 name_layer_idx = name_layer_idx.replace(",", "_")
#                 name_layer_idx = name_layer_idx.replace(" ", "")
#                 name_layer_idx = name_layer_idx.replace("'", "")
#                 layer_save_path = f"{self.config.save_path}_{self.get_analysis_type()}_layer{name_layer_idx}.pt"
#                 probe.save(layer_save_path)
#                 print(f"Saved {self.get_analysis_type()} probe for layer {layer_idx} to {layer_save_path}")
#
#             # Store results
#             results[layer_idx] = {
#                 'probe': probe,
#                 'training_losses': training_losses,
#                 'evaluation_results': evaluation_results,
#                 'overall_accuracy': evaluation_results['overall_accuracy']
#             }
#
#         # Generate visualizations if requested
#         if self.visualizer:
#             self.visualizer.plot_probe_comparison(
#                 {k: v['evaluation_results'] for k, v in results.items()},
#                 model_name,
#                 self.get_analysis_type()
#             )
#
#         return results
#
#
# class POSAnalyzer(BaseAnalyzer):
#     """
#     Analyzer for Part-of-Speech probing experiments.
#
#     This class handles POS-specific probe analysis including training,
#     evaluation, and visualization.
#     """
#
#     def get_analysis_type(self) -> str:
#         """Get the type of analysis."""
#         return 'pos'
#
#     def get_label_names(self) -> List[str]:
#         """Get POS label names based on configuration."""
#         pos_categories = self.config.get_pos_categories()
#         # Sort by index to get correct order
#         sorted_categories = sorted(pos_categories.items(), key=lambda x: x[1])
#         return [name for name, _ in sorted_categories]
#
#
# class SemanticAnalyzer(BaseAnalyzer):
#     """
#     Analyzer for semantic role probing experiments.
#
#     This class handles semantic role-specific probe analysis including
#     training, evaluation, and visualization.
#     """
#
#     def get_analysis_type(self) -> str:
#         """Get the type of analysis."""
#         return 'semantic'
#
#     def get_label_names(self) -> List[str]:
#         """Get semantic label names based on configuration."""
#         semantic_categories = self.config.get_semantic_categories()
#         # Sort by index to get correct order
#         sorted_categories = sorted(semantic_categories.items(), key=lambda x: x[1])
#         return [name for name, _ in sorted_categories]
#
#
# # High-level analysis functions
# def run_pos_probe_analysis(
#         model,
#         dataloader: DataLoader,
#         tokenizer,
#         device: str,
#         layer_indices: Optional[List[int]] = None,
#         config: Optional[LinguisticProbesConfig] = None,
#         model_name: str = ""
# ) -> Dict[int, Any]:
#     """
#     Run POS probing analysis on a model.
#
#     Args:
#         model: The transformer model to analyze
#         dataloader: DataLoader with input data
#         tokenizer: Tokenizer for decoding sequences
#         device: Device for computation
#         layer_indices: Which layers to probe
#         config: Configuration object
#         model_name: Name of the model for saving/logging
#
#     Returns:
#         Dictionary mapping layer indices to probe results
#     """
#     if config is None:
#         config = LinguisticProbesConfig.pos_only()
#
#     config.device = device
#     analyzer = POSAnalyzer(config)
#     return analyzer.run_probe_analysis(model, dataloader, tokenizer, layer_indices, model_name)
#
#
# def run_semantic_probe_analysis(
#         model,
#         dataloader: DataLoader,
#         tokenizer,
#         device: str,
#         layer_indices: Optional[List[int]] = None,
#         config: Optional[LinguisticProbesConfig] = None,
#         model_name: str = ""
# ) -> Dict[int, Any]:
#     """
#     Run semantic role probing analysis on a model.
#
#     Args:
#         model: The transformer model to analyze
#         dataloader: DataLoader with input data
#         tokenizer: Tokenizer for decoding sequences
#         device: Device for computation
#         layer_indices: Which layers to probe
#         config: Configuration object
#         model_name: Name of the model for saving/logging
#
#     Returns:
#         Dictionary mapping layer indices to probe results
#     """
#     if config is None:
#         config = LinguisticProbesConfig.semantic_only()
#
#     config.device = device
#     analyzer = SemanticAnalyzer(config)
#     return analyzer.run_probe_analysis(model, dataloader, tokenizer, layer_indices, model_name)
#
#
# def run_comprehensive_probe_analysis(
#         model,
#         dataloader: DataLoader,
#         tokenizer,
#         device: str,
#         layer_indices: Optional[List[int]] = None,
#         config: Optional[LinguisticProbesConfig] = None,
#         model_name: str = ""
# ) -> Dict[str, Dict[int, Any]]:
#     """
#     Run both POS and semantic probing analysis on a model.
#
#     Args:
#         model: The transformer model to analyze
#         dataloader: DataLoader with input data
#         tokenizer: Tokenizer for decoding sequences
#         device: Device for computation
#         layer_indices: Which layers to probe
#         config: Configuration object
#         model_name: Name of the model for saving/logging
#
#     Returns:
#         Dictionary with 'pos' and 'semantic' keys containing probe results
#     """
#     if config is None:
#         config = LinguisticProbesConfig.comprehensive()
#
#     config.device = device
#
#     results = {}
#
#     # Run POS analysis
#     if config.track_pos:
#         print("Running POS probe analysis...")
#         pos_analyzer = POSAnalyzer(config)
#         results['pos'] = pos_analyzer.run_probe_analysis(
#             model, dataloader, tokenizer, layer_indices, model_name
#         )
#
#     # Run semantic analysis
#     if config.track_semantic:
#         print("\nRunning semantic probe analysis...")
#         semantic_analyzer = SemanticAnalyzer(config)
#         results['semantic'] = semantic_analyzer.run_probe_analysis(
#             model, dataloader, tokenizer, layer_indices, model_name
#         )
#
#     return results
#
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm

from .config import LinguisticProbesConfig
from .models import MultiLabelProbe, LinearProbe
from .utils import extract_hidden_representations_with_pos_semantic, prepare_probing_dataset
from .visualization import ProbesVisualizer


class BaseAnalyzer:
    """
    Base class for probe analyzers.

    This class uses PRE-TRAINED probes to monitor confidence during training,
    NOT to train new probes.
    """

    def __init__(self, config: Optional[LinguisticProbesConfig] = None):
        self.config = config or LinguisticProbesConfig.default()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Store loaded probes for each layer
        self.loaded_probes = {}

        self.visualizer = ProbesVisualizer(
            self.config.log_dir, self.config
        ) if self.config.save_visualizations else None

    def get_analysis_type(self) -> str:
        """Get the type of analysis (to be overridden by subclasses)."""
        raise NotImplementedError

    def get_label_names(self) -> List[str]:
        """Get label names for this analysis type (to be overridden by subclasses)."""
        raise NotImplementedError

    def load_probes(self, probe_paths: Dict[Union[int, tuple], str]) -> None:
        """
        Load pre-trained probes for specified layers.

        Args:
            probe_paths: Dictionary mapping layer keys to probe file paths
        """
        print(f"Loading {self.get_analysis_type()} probes...")

        for layer_key, probe_path in probe_paths.items():
            try:
                # Determine probe type from config
                if self.config.probe_type == "multilabel":
                    # We need to know input_dim - will be set when we first use the probe
                    probe = None  # Will load when we have hidden states
                else:
                    probe = None  # Will load when we have hidden states

                # Store the path for later loading
                self.loaded_probes[layer_key] = {'path': probe_path, 'probe': None}
                print(f"Registered probe path for layer {layer_key}: {probe_path}")

            except Exception as e:
                print(f"Failed to register probe for layer {layer_key}: {e}")

    def _load_probe_for_layer(self, layer_key: Union[int, tuple], input_dim: int) -> Optional[torch.nn.Module]:
        """
        Load a specific probe for a layer when we know the input dimension.

        Args:
            layer_key: Layer identifier
            input_dim: Hidden state dimension

        Returns:
            Loaded probe model
        """
        if layer_key not in self.loaded_probes:
            print(f"No probe registered for layer {layer_key}")
            return None

        probe_info = self.loaded_probes[layer_key]

        # If probe already loaded, return it
        if probe_info['probe'] is not None:
            return probe_info['probe']

        try:
            # Create probe model
            if self.config.probe_type == "multilabel":
                probe = MultiLabelProbe(input_dim=input_dim, config=self.config)
            else:
                probe = LinearProbe(input_dim=input_dim, config=self.config)

            # Load trained weights

            probe.load(probe_info['path'])
            probe.eval()  # Set to evaluation mode

            # Cache the loaded probe
            self.loaded_probes[layer_key]['probe'] = probe
            print(f"Loaded {self.get_analysis_type()} probe for layer {layer_key}")

            return probe

        except Exception as e:
            print(f"Failed to load probe for layer {layer_key}: {e}")
            return None

    def analyze(
            self,
            model,
            dataloader: DataLoader,
            tokenizer,
            model_name: str = ""
    ) -> Dict[Union[int, tuple], Dict[str, float]]:
        """
        Analyze model using pre-trained probes.

        This method extracts hidden states and gets confidence scores from pre-trained probes.

        Args:
            model: The transformer model to analyze
            dataloader: DataLoader with input data
            model_name: Name of the model for logging

        Returns:
            Dictionary mapping layer keys to confidence scores
        """
        print(f"Starting {self.get_analysis_type()} probe analysis...")

        # Extract representations and labels
        hidden_states, pos_labels, semantic_labels = extract_hidden_representations_with_pos_semantic(
            model, dataloader, self.device, self.config.layer_indices,
            tokenizer, self.config
        )

        # If no hidden states, return empty results
        if not hidden_states:
            print(f"Warning: No hidden states extracted for {self.get_analysis_type()} analysis")

        # Select appropriate labels based on analysis type
        if self.get_analysis_type() == 'pos':
            labels = pos_labels
        elif self.get_analysis_type() == 'semantic':
            labels = semantic_labels
        else:
            raise ValueError(f"Unknown analysis type: {self.get_analysis_type()}")

        if labels is None:
            print(f"Warning: No {self.get_analysis_type()} labels available")
            return {}

        results = {}

        # Analyze each layer with its probe
        for layer_key, layer_hidden in hidden_states.items():
            # Get input dimension
            input_dim = layer_hidden.shape[2]

            # Load probe for this layer
            probe = self._load_probe_for_layer(layer_key, input_dim)
            if probe is None:
                print(f"Skipping layer {layer_key} - no probe available")
                continue

            # Get confidence scores from the probe
            confidence_scores = self._get_confidence_scores(
                probe, layer_hidden, labels
            )

            results[layer_key] = confidence_scores

        return results

    def _get_confidence_scores(
            self,
            probe: torch.nn.Module,
            hidden_states: torch.Tensor,
            labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get confidence scores from a probe on hidden states.

        Args:
            probe: Pre-trained probe model
            hidden_states: Hidden states [batch, seq_len, hidden_dim]
            labels: Ground truth labels [batch, num_features]

        Returns:
            Dictionary with confidence scores per feature
        """
        probe.eval()

        with torch.no_grad():
            # Prepare data
            B, T, D = hidden_states.shape
            X = hidden_states.view(B * T, D).to(self.device)

            # Get predictions
            predictions = probe(X)  # [B*T, num_features]

            # Average predictions across all tokens
            avg_predictions = predictions.mean(dim=0).cpu().numpy()

            # Create confidence scores dictionary
            label_names = self.get_label_names()
            confidence_scores = {}

            for i, label_name in enumerate(label_names):
                if i < len(avg_predictions):
                    confidence_scores[label_name] = float(avg_predictions[i])

        return confidence_scores

    @classmethod
    def load_probe(cls, probe_path: str, layer_index: Union[int, tuple], probe_type: str) -> torch.nn.Module:
        """
        Class method to load a single probe (for backward compatibility).

        Args:
            probe_path: Path to the probe file
            layer_index: Layer identifier
            probe_type: Type of probe ('linear' or 'multilabel')

        Returns:
            Loaded probe model
        """
        try:
            # This is a simplified version - in practice you'd need input_dim
            # For now, return None and print a message
            print(f"To load probe for layer {layer_index}, use the analyze() method with proper configuration")
            return None
        except Exception as e:
            print(f"Failed to load probe: {e}")
            return None


class POSAnalyzer(BaseAnalyzer):
    """
    Analyzer for Part-of-Speech probing experiments.

    This class loads pre-trained POS probes and monitors their confidence
    during training to track linguistic understanding.
    """

    def get_analysis_type(self) -> str:
        """Get the type of analysis."""
        return 'pos'

    def get_label_names(self) -> List[str]:
        """Get POS label names based on configuration."""
        pos_categories = self.config.get_pos_categories()
        # Sort by index to get correct order
        sorted_categories = sorted(pos_categories.items(), key=lambda x: x[1])
        return [name for name, _ in sorted_categories]


class SemanticAnalyzer(BaseAnalyzer):
    """
    Analyzer for semantic role probing experiments.

    This class loads pre-trained semantic probes and monitors their confidence
    during training to track semantic understanding.
    """

    def get_analysis_type(self) -> str:
        """Get the type of analysis."""
        return 'semantic'

    def get_label_names(self) -> List[str]:
        """Get semantic label names based on configuration."""
        semantic_categories = self.config.get_semantic_categories()
        # Sort by index to get correct order
        sorted_categories = sorted(semantic_categories.items(), key=lambda x: x[1])
        return [name for name, _ in sorted_categories]


# Legacy compatibility functions (updated to use proper analyzer pattern)
def run_pos_probe_analysis(
        model,
        dataloader: DataLoader,
        tokenizer,
        device: str,
        layer_indices: Optional[List[int]] = None,
        config: Optional[LinguisticProbesConfig] = None,
        model_name: str = ""
) -> Dict[int, Any]:
    """
    Run POS probing analysis on a model using pre-trained probes.
    """
    if config is None:
        config = LinguisticProbesConfig.pos_only()

    config.device = device
    config.layer_indices = layer_indices

    analyzer = POSAnalyzer(config)

    # If probe paths are provided in config, load them
    if hasattr(config, 'probe_load_path') and config.probe_load_path:
        analyzer.load_probes(config.probe_load_path)

    return analyzer.analyze(model, dataloader, model_name)


def run_semantic_probe_analysis(
        model,
        dataloader: DataLoader,
        tokenizer,
        device: str,
        layer_indices: Optional[List[int]] = None,
        config: Optional[LinguisticProbesConfig] = None,
        model_name: str = ""
) -> Dict[int, Any]:
    """
    Run semantic role probing analysis on a model using pre-trained probes.
    """
    if config is None:
        config = LinguisticProbesConfig.semantic_only()

    config.device = device
    config.layer_indices = layer_indices

    analyzer = SemanticAnalyzer(config)

    # If probe paths are provided in config, load them
    if hasattr(config, 'probe_load_path') and config.probe_load_path:
        analyzer.load_probes(config.probe_load_path)

    return analyzer.analyze(model, dataloader, model_name)


def run_comprehensive_probe_analysis(
        model,
        dataloader: DataLoader,
        tokenizer,
        device: str,
        layer_indices: Optional[List[int]] = None,
        config: Optional[LinguisticProbesConfig] = None,
        model_name: str = ""
) -> Dict[str, Dict[int, Any]]:
    """
    Run both POS and semantic probing analysis using pre-trained probes.
    """
    if config is None:
        config = LinguisticProbesConfig.comprehensive()

    config.device = device
    config.layer_indices = layer_indices

    results = {}

    # Run POS analysis
    if config.track_pos:
        print("Running POS probe analysis...")
        results['pos'] = run_pos_probe_analysis(
            model, dataloader, tokenizer, device, layer_indices, config, model_name
        )

    # Run semantic analysis
    if config.track_semantic:
        print("\nRunning semantic probe analysis...")
        results['semantic'] = run_semantic_probe_analysis(
            model, dataloader, tokenizer, device, layer_indices, config, model_name
        )

    return results