import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Union
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
        self.use_random_probes = False

        self.visualizer = ProbesVisualizer(
            self.config.log_dir, self.config
        ) if self.config.save_visualizations else None

    def get_analysis_type(self) -> str:
        """Get the type of analysis (to be overridden by subclasses)."""
        raise NotImplementedError

    def get_label_names(self) -> List[str]:
        """Get label names for this analysis type (to be overridden by subclasses)."""
        raise NotImplementedError

    def load_probes(self, probe_paths: Dict[Union[int, tuple], str] = None) -> None:
        """
        Load pre-trained probes for specified layers.

        Args:
            probe_paths: Dictionary mapping layer keys to probe file paths
        """
        print(f"Loading {self.get_analysis_type()} probes...")
        if probe_paths is None:
            self.use_random_probes = True
            print("No probe paths provided, using random probes.")
            self.loaded_probes = {}
            return

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
        self.use_random_probes = False

    def _create_random_probe(self, layer_key: Union[int, tuple], input_dim: int, linguistic_target: Optional[str] = 'pos') -> torch.nn.Module:
        """
        Create a random probe for a layer when no pre-trained probe is available.

        Args:
            layer_key: Layer identifier
            input_dim: Hidden state dimension

        Returns:
            Randomly initialized probe model
        """
        print(f"Creating random probe for layer {layer_key} with input dim {input_dim}")
        if layer_key in self.loaded_probes and self.loaded_probes[layer_key]['probe'] is not None:
            return self.loaded_probes[layer_key]['probe']
        try:
            if self.config.probe_type == "multilabel":
                probe = MultiLabelProbe(input_dim=input_dim, config=self.config, linguistic_target=self.get_analysis_type())
            else:
                probe = LinearProbe(input_dim=input_dim, config=self.config, linguistic_target=self.get_analysis_type())

            # Move to device and set to evaluation mode
            probe.to(self.device)
            probe.eval()
            self.loaded_probes[layer_key] = {'probe': probe, 'path': None, 'random': True}

            print(f"Created random {self.get_analysis_type()} probe for layer {layer_key}")

            return probe
        except Exception as e:
            print(f"Failed to create random probe for layer {layer_key}: {e}")
            return None

    def _load_probe_for_layer(self, layer_key: Union[int, tuple], input_dim: int, linguistic_target: Optional[str] = 'pos') -> Optional[torch.nn.Module]:
        """
        Load a specific probe for a layer when we know the input dimension.

        Args:
            layer_key: Layer identifier
            input_dim: Hidden state dimension

        Returns:
            Loaded probe model
        """
        # If using random probes, return rand probe
        if self.use_random_probes:
            return self._create_random_probe(layer_key, input_dim, linguistic_target=linguistic_target)

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
                probe = MultiLabelProbe(input_dim=input_dim, config=self.config, linguistic_target=self.get_analysis_type())
            else:
                probe = LinearProbe(input_dim=input_dim, config=self.config, linguistic_target=self.get_analysis_type())
            print(f"Loading {self.get_analysis_type()} probe for layer {layer_key} from {probe_info['path']}")
            print(f'Config: {self.config}')
            print(f'Probe {probe}')
            input("Press Enter to continue...")  # Debug pause
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