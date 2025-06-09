import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm

from .config import LinguisticProbesConfig
from .models import MultiLabelProbe, LinearProbe
from .utils import extract_hidden_representations, prepare_probing_dataset
from .visualization import ProbesVisualizer


class BaseAnalyzer:
    """
    Base class for probe analyzers.
    """

    def __init__(self, config: Optional[LinguisticProbesConfig] = None):

        self.config = config or LinguisticProbesConfig.default()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.visualizer = ProbesVisualizer(self.config.log_dir,
                                           self.config) if self.config.save_visualizations else None

    def get_analysis_type(self) -> str:
        """Get the type of analysis (to be overridden by subclasses)."""
        raise NotImplementedError

    def get_label_names(self) -> List[str]:
        """Get label names for this analysis type (to be overridden by subclasses)."""
        raise NotImplementedError

    def run_probe_analysis(
            self,
            model,
            dataloader: DataLoader,
            tokenizer,
            layer_indices: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            model_name: str = ""
    ) -> Dict[int, Any]:
        """
        Run probing analysis on specified layers.

        Args:
            model: The transformer model to analyze
            dataloader: DataLoader with input data
            tokenizer: Tokenizer for decoding sequences
            layer_indices: Which layers to probe (None for default)
            model_name: Name of the model for saving/logging
            model_name: Name of the model for saving/logging

        Returns:
            Dictionary mapping layer indices to probe results
        """
        print(f"Starting {self.get_analysis_type()} probe analysis...")

        # Extract representations and labels
        hidden_states, pos_labels, semantic_labels = extract_hidden_representations(
            model, dataloader, self.device, layer_indices, tokenizer, self.config
        )

        # Select appropriate labels based on analysis type
        if self.get_analysis_type() == 'pos':
            labels = pos_labels
        elif self.get_analysis_type() == 'semantic':
            labels = semantic_labels
        else:
            raise ValueError(f"Unknown analysis type: {self.get_analysis_type()}")

        if labels is None:
            raise ValueError(f"No {self.get_analysis_type()} labels available")

        results = {}

        # Probe for each layer
        for layer_idx, layer_hidden in hidden_states.items():
            print(f"\nProbing layer {layer_idx} for {self.get_analysis_type()}:")

            # Prepare dataset
            probe_loader = prepare_probing_dataset(
                layer_hidden, labels, self.config.batch_size
            )

            # Initialize and train probe
            input_dim = layer_hidden.shape[2]  # Hidden dimension size
            num_features = labels.shape[1]  # Number of labels

            probe = MultiLabelProbe(
                input_dim=input_dim,
                num_features=num_features,
                hidden_dim=self.config.hidden_dim,
                lr=self.config.lr,
                epochs=self.config.epochs,
                device=self.device,
                dropout=self.config.dropout
            )

            # Train the probe
            training_losses = probe.train_probe(
                probe_loader,
                use_class_weights=self.config.use_class_weights,
                penalize_classes=self.config.penalized_classes if self.config.penalize_frequent_classes else None
            )

            # Evaluate the probe
            evaluation_results = probe.evaluate_probe(
                probe_loader,
                label_names=self.get_label_names()
            )

            # Print evaluation report
            probe.print_evaluation_report(evaluation_results)

            # Save probe if requested
            if self.config.save_probes and self.config.save_path:
                layer_save_path = f"{self.config.save_path}_{self.get_analysis_type()}_layer{layer_idx}.pt"
                probe.save(layer_save_path)
                print(f"Saved {self.get_analysis_type()} probe for layer {layer_idx} to {layer_save_path}")

            # Store results
            results[layer_idx] = {
                'probe': probe,
                'training_losses': training_losses,
                'evaluation_results': evaluation_results,
                'overall_accuracy': evaluation_results['overall_accuracy']
            }

        # Generate visualizations if requested
        if self.visualizer:
            self.visualizer.plot_probe_comparison(
                {k: v['evaluation_results'] for k, v in results.items()},
                model_name,
                self.get_analysis_type()
            )

        return results


class POSAnalyzer(BaseAnalyzer):
    """
    Analyzer for Part-of-Speech probing experiments.

    This class handles POS-specific probe analysis including training,
    evaluation, and visualization.
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

    This class handles semantic role-specific probe analysis including
    training, evaluation, and visualization.
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


# High-level analysis functions
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
    Run POS probing analysis on a model.

    Args:
        model: The transformer model to analyze
        dataloader: DataLoader with input data
        tokenizer: Tokenizer for decoding sequences
        device: Device for computation
        layer_indices: Which layers to probe
        config: Configuration object
        model_name: Name of the model for saving/logging

    Returns:
        Dictionary mapping layer indices to probe results
    """
    if config is None:
        config = LinguisticProbesConfig.pos_only()

    config.device = device
    analyzer = POSAnalyzer(config)
    return analyzer.run_probe_analysis(model, dataloader, tokenizer, layer_indices, model_name)


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
    Run semantic role probing analysis on a model.

    Args:
        model: The transformer model to analyze
        dataloader: DataLoader with input data
        tokenizer: Tokenizer for decoding sequences
        device: Device for computation
        layer_indices: Which layers to probe
        config: Configuration object
        model_name: Name of the model for saving/logging

    Returns:
        Dictionary mapping layer indices to probe results
    """
    if config is None:
        config = LinguisticProbesConfig.semantic_only()

    config.device = device
    analyzer = SemanticAnalyzer(config)
    return analyzer.run_probe_analysis(model, dataloader, tokenizer, layer_indices, model_name)


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
    Run both POS and semantic probing analysis on a model.

    Args:
        model: The transformer model to analyze
        dataloader: DataLoader with input data
        tokenizer: Tokenizer for decoding sequences
        device: Device for computation
        layer_indices: Which layers to probe
        config: Configuration object
        model_name: Name of the model for saving/logging

    Returns:
        Dictionary with 'pos' and 'semantic' keys containing probe results
    """
    if config is None:
        config = LinguisticProbesConfig.comprehensive()

    config.device = device

    results = {}

    # Run POS analysis
    if config.track_pos:
        print("Running POS probe analysis...")
        pos_analyzer = POSAnalyzer(config)
        results['pos'] = pos_analyzer.run_probe_analysis(
            model, dataloader, tokenizer, layer_indices, model_name
        )

    # Run semantic analysis
    if config.track_semantic:
        print("\nRunning semantic probe analysis...")
        semantic_analyzer = SemanticAnalyzer(config)
        results['semantic'] = semantic_analyzer.run_probe_analysis(
            model, dataloader, tokenizer, layer_indices, model_name
        )

    return results


# Legacy compatibility functions
def run_semantic_role_probe(
        model,
        dataloader,
        device,
        layer_indices,
        tokenizer,
        hidden_dim=128,
        lr=1e-3,
        epochs=3,
        batch_size=32,
        save_path=None
):
    """Legacy function for semantic role probing."""
    config = LinguisticProbesConfig(
        hidden_dim=hidden_dim,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        track_pos=False,
        track_semantic=True,
        semantic_granularity='reduced'
    )

    return run_semantic_probe_analysis(
        model, dataloader, tokenizer, device, layer_indices, config
    )