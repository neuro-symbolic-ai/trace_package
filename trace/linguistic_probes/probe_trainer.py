import torch
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Any, Optional, Union

from trace.linguistic_probes import LinguisticProbesConfig
from trace.linguistic_probes.models import MultiLabelProbe, LinearProbe
from trace.linguistic_probes.utils import (
    extract_hidden_representations_with_pos_semantic,
    prepare_probing_dataset
)
from trace.linguistic_probes.visualization import ProbesVisualizer


class ProbeTrainer:
    def __init__(self, config: LinguisticProbesConfig, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        # updating the number of classes based on the config
        if config.track_pos:
            config.num_pos_classes = len(config.get_pos_categories())
        if config.track_semantic:
            config.num_semantic_classes = len(config.get_semantic_categories())


        # Create output directories
        if config.save_path:
            os.makedirs(config.save_path, exist_ok=True)
        if config.log_dir:
            os.makedirs(config.log_dir, exist_ok=True)

        # Initialize visualizer
        self.visualizer = ProbesVisualizer(config.log_dir, config) if config.save_visualizations else None

        # Store training results
        self.training_results = {}

    def train_all_probes(
            self,
            model,
            dataloader: DataLoader,
            layer_indices: Optional[Union[List[int], Dict[str, List[int]]]] = None
    ) -> Dict[Union[int, tuple], Dict[str, Any]]:
        """
        Train probes for all specified layers of the model.
        """
        print("Starting probe training for all layers...")

        # Use config layer indices if not provided

        if self.config.layer_indices is None:
            if model.model_type == 'encoder_only' and hasattr(model.encoder, 'layers'):
                layer_indices = {'encoder': list(range(len(model.encoder.layers)))}  # Default to first layer
            elif model.model_type == 'decoder_only' and hasattr(model.decoder, 'layers'):
                layer_indices = {'decoder': list(range(len(model.decoder.layers)))}  # Default to first layer
            elif model.model_type == 'encoder_decoder':
                encoder_layers = list(range(len(model.encoder.layers))) if hasattr(model.encoder, 'layers') else [0]
                decoder_layers = list(range(len(model.decoder.layers))) if hasattr(model.decoder, 'layers') else [0]
                layer_indices = {'encoder': encoder_layers, 'decoder': decoder_layers}

            # Set layer indices in config for extraction
            self.config.layer_indices = layer_indices

        # Extract hidden representations and labels for all layers
        print("Extracting hidden representations...")
        hidden_states, pos_labels, semantic_labels = extract_hidden_representations_with_pos_semantic(
            model, dataloader, self.device, self.config.layer_indices, self.tokenizer, self.config
        )

        results = {}

        # Train probes for each layer
        for layer_key, layer_hidden in hidden_states.items():
            print(f"\nTraining probes for layer {layer_key}...")

            layer_results = {}

            # Train POS probe if enabled
            if self.config.track_pos and pos_labels is not None:
                pos_probe, pos_metrics = self._train_single_probe(
                    layer_hidden, pos_labels, layer_key, 'pos'
                )
                layer_results['pos'] = {
                    'probe': pos_probe,
                    'metrics': pos_metrics,
                    'accuracy': pos_metrics['overall_accuracy']
                }

            # Train semantic probe if enabled
            if self.config.track_semantic and semantic_labels is not None:
                semantic_probe, semantic_metrics = self._train_single_probe(
                    layer_hidden, semantic_labels, layer_key, 'semantic'
                )
                layer_results['semantic'] = {
                    'probe': semantic_probe,
                    'metrics': semantic_metrics,
                    'accuracy': semantic_metrics['overall_accuracy']
                }

            results[layer_key] = layer_results

        # Save overall results
        self._save_training_results(results)

        print(f"\nProbe training completed! Results saved to {self.config.save_path}")
        return results

    def _train_single_probe(
            self,
            hidden_states: torch.Tensor,
            labels: torch.Tensor,
            layer_key: Union[int, tuple],
            probe_type: str
    ) -> tuple:
        """
        Train a single probe for one layer.

        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_dim]
            labels: One-hot labels [batch, num_features]
            layer_key: Layer identifier
            probe_type: 'pos' or 'semantic'

        Returns:
            Tuple of (trained_probe, evaluation_metrics)
        """
        print(f"  Training {probe_type} probe...")

        # Get probe configuration
        input_dim = hidden_states.shape[2]
        if probe_type == 'pos':
            num_features = self.config.num_pos_classes
        elif probe_type == 'semantic':
            num_features = self.config.num_semantic_classes
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")


        # Create probe model
        if self.config.probe_type == "multilabel":
            probe = MultiLabelProbe(input_dim=input_dim, config=self.config, num_features=num_features)
        else:
            probe = LinearProbe(input_dim=input_dim, config=self.config, num_features=num_features)

        # Prepare dataset
        probe_loader = prepare_probing_dataset(
            hidden_states, labels, self.config.batch_size
        )

        # Train the probe
        training_losses = probe.train_probe(
            probe_loader,
            use_class_weights=self.config.use_class_weights,
            penalize_classes=self.config.penalized_classes if self.config.penalize_frequent_classes else None
        )

        # Evaluate the probe
        label_names = self._get_label_names(probe_type)
        evaluation_results = probe.evaluate_probe(probe_loader, label_names)

        # Print results
        probe.print_evaluation_report(evaluation_results)

        # Save probe if requested
        if self.config.save_probes and self.config.save_path:
            save_path = self._get_probe_save_path(layer_key, probe_type)
            probe.save(save_path)
            print(f"  Saved {probe_type} probe to {save_path}")

        return probe, evaluation_results

    def _get_label_names(self, probe_type: str) -> List[str]:
        """Get label names for the probe type."""
        if probe_type == 'pos':
            pos_categories = self.config.get_pos_categories()
            sorted_categories = sorted(pos_categories.items(), key=lambda x: x[1])
            return [name for name, _ in sorted_categories]
        elif probe_type == 'semantic':
            semantic_categories = self.config.get_semantic_categories()
            sorted_categories = sorted(semantic_categories.items(), key=lambda x: x[1])
            return [name for name, _ in sorted_categories]
        else:
            raise ValueError(f"Unknown probe type: {probe_type}")

    def _get_probe_save_path(self, layer_key: Union[int, tuple], probe_type: str) -> str:
        """Generate save path for a probe."""
        if isinstance(layer_key, tuple):
            layer_idx, layer_type = layer_key
            filename = f"{probe_type}_layer{layer_idx}_{layer_type}.pt"
        else:
            filename = f"{probe_type}_layer{layer_key}.pt"

        return os.path.join(self.config.save_path, filename)

    def _save_training_results(self, results: Dict[Union[int, tuple], Dict[str, Any]]):
        """Save training results to JSON file."""
        if not self.config.save_path:
            return

        # Convert results to JSON-serializable format
        json_results = {}
        for layer_key, layer_results in results.items():
            layer_str = str(layer_key)
            json_results[layer_str] = {}

            for probe_type, probe_results in layer_results.items():
                metrics = probe_results['metrics']
                json_results[layer_str][probe_type] = {
                    'overall_accuracy': float(metrics['overall_accuracy']),
                    'per_label_metrics': {
                        label: {
                            'count': int(label_metrics['count']),
                            'accuracy': float(label_metrics['accuracy']),
                            'precision': float(label_metrics['precision']),
                            'recall': float(label_metrics['recall']),
                            'f1': float(label_metrics['f1'])
                        }
                        for label, label_metrics in metrics['per_label_metrics'].items()
                    }
                }

        # Save to file
        results_file = os.path.join(self.config.save_path, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Training results saved to {results_file}")