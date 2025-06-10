import os

import numpy as np
import torch
import copy
from typing import Dict, Any, Optional, List
from collections import defaultdict

# Import the refactored modules
from ..linguistic_probes import LinguisticProbesConfig, POSAnalyzer, SemanticAnalyzer
from ..intrisic_dimensions import IntrinsicDimensionAnalyzer, IntrinsicDimensionsConfig


# from ..hessian_analysis import HessianAnalyzer, HessianConfig  # TODO: Add when refactored


class TrainingCallbacks:
    """
    Handles all analysis callbacks during training.
    """

    def __init__(self, config, tokenizer, device):
        """
        Initialize training callbacks.

        Args:
            config: Training configuration
            tokenizer: Model tokenizer
            device: Training device
        """
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        # Initialize analysis modules
        self._setup_analysis_modules()

        # Initialize tracking data
        self._setup_tracking_data()

    @staticmethod
    def save_analysis_results(analysis_results, analysis_results_path):
        """
        Save analysis results to a specified path.

        Args:
            analysis_results: Dictionary containing analysis results
            analysis_results_path: Path to save the results
        """
        import json
        # with open(analysis_results_path, 'w') as f:
        #     json.dump(analysis_results, f, indent=4)
        print(f"Analysis results saved to {analysis_results_path}")

        def save_json_data(data, filename, save_dir="./logs"):
            if not data:
                return

            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)

            try:
                def make_serializable(obj):
                    if isinstance(obj, (np.generic, np.ndarray)):
                        return obj.tolist()
                    elif isinstance(obj, torch.Tensor):
                        return obj.detach().cpu().tolist()
                    elif isinstance(obj, tuple):
                        return str(obj)
                    elif isinstance(obj, dict):
                        return {str(k): make_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [make_serializable(x) for x in obj]
                    elif isinstance(obj, (int, float, str, bool)) or obj is None:
                        return obj
                    else:
                        return str(obj)  # Fallback: stringify unknown types

                # Top-level step keys must be str
                serializable_history = {str(step): make_serializable(metrics) for step, metrics in data.items()}

                with open(save_path, 'w') as f:
                    json.dump(serializable_history, f, indent=2)

                print(f"Data successfully saved to {save_path}")

            except Exception as e:
                print(f"Error saving data to {filename}: {e}")

        if analysis_results.get('results').get('intrinsic_dimensions'):
            intrinsic_dim_log_dir = os.path.join(analysis_results_path, 'intrinsic_dimensions')
            os.makedirs(intrinsic_dim_log_dir, exist_ok=True)
            save_json_data(analysis_results.get('results').get('intrinsic_dimensions'), 'intrinsic_dimension_history.json', intrinsic_dim_log_dir)


        if analysis_results.get('linguistic_probes'):
            print("Linguistic probes results:", analysis_results['linguistic_probes'])
        if analysis_results.get('semantic_probes'):
            print("Semantic probes results:", analysis_results['semantic_probes'])

        if analysis_results.get('hessian'):
            print("Hessian analysis results:", analysis_results['hessian'])
        if analysis_results.get('pos_performance'):
            print("POS performance results:", analysis_results['pos_performance'])
        if analysis_results.get('semantic_roles'):
            print("Semantic roles results:", analysis_results['semantic_roles'])
        if analysis_results.get('gradients'):
            print("Gradient analysis results:", analysis_results['gradients'])
        print("Analysis results summary:")
        print(f"Total steps analyzed: {len(analysis_results.get('steps_analyzed', []))}")
        print(f"Analysis modules enabled: {analysis_results.get('analysis_modules', {})}")

    def _setup_analysis_modules(self):
        """Initialize the analysis modules based on configuration."""

        # Linguistic (POS) Probes Analyzer
        if self.config.track_linguistic_probes:
            print("Setting up linguistic probes analyzer...")
            probe_config = LinguisticProbesConfig(
                probe_type=self.config.probe_type,
                layer_indices=self.config.probe_layers,
                probe_load_path=self.config.probe_load_path,
                num_classes=self.config.probe_num_features,
                hidden_dim=self.config.probe_hidden_dim,
                lr=self.config.probe_lr,
                epochs=self.config.probe_epochs,
                log_dir=self.config.plots_path,
                save_visualizations=True
            )
            self.pos_linguistic_analyzer = POSAnalyzer(probe_config)
        else:
            self.pos_linguistic_analyzer = None

        # Semantic Probes Analyzer (same structure as linguistic)
        if self.config.track_semantic_probes:
            print("Setting up semantic probes analyzer...")
            semantic_config = LinguisticProbesConfig(
                probe_type=self.config.semantic_probe_type,
                layer_indices=self.config.semantic_probe_layers,
                probe_load_path=self.config.semantic_probe_load_path,
                num_classes=self.config.semantic_probe_num_features,
                hidden_dim=self.config.semantic_probe_hidden_dim,
                lr=self.config.semantic_probe_lr,
                epochs=self.config.semantic_probe_epochs,
                log_dir=self.config.plots_path,
                save_visualizations=True
            )
            self.semantic_analyzer = SemanticAnalyzer(semantic_config)
        else:
            self.semantic_analyzer = None

        # Intrinsic Dimensions Analyzer
        if self.config.track_intrinsic_dimensions:
            print("Setting up intrinsic dimensions analyzer...")
            id_config = IntrinsicDimensionsConfig(
                model_type=self.config.model_type,
                layers_to_analyze=self.config.id_selected_layers,  # Use same layers as probes
                id_method=self.config.id_method,
                log_dir=self.config.plots_path
            )
            self.intrinsic_analyzer = IntrinsicDimensionAnalyzer(id_config)
        else:
            self.intrinsic_analyzer = None

        # TODO: Add other analyzers when refactored
        # Hessian Analyzer
        if self.config.track_hessian:
            # hessian_config = HessianConfig(...)
            # self.hessian_analyzer = HessianAnalyzer(hessian_config)
            print("Hessian analysis will be added when module is refactored")
            self.hessian_analyzer = None
        else:
            self.hessian_analyzer = None

        # POS Performance Tracker (placeholder)
        if self.config.track_pos_performance:
            print("POS performance tracking will be added later")
            self.pos_tracker = None
        else:
            self.pos_tracker = None

        # Semantic Role Tracker (placeholder)
        if self.config.track_semantic_roles:
            print("Semantic role tracking will be added later")
            self.semantic_role_tracker = None
        else:
            self.semantic_role_tracker = None

    def _setup_tracking_data(self):
        """Initialize data structures for tracking results."""
        # Gradient tracking
        if self.config.track_gradients:
            self.grad_history = defaultdict(list)
            self.grad_similarities_history = defaultdict(list)

        # Analysis results storage
        self.analysis_results = {
            'linguistic_probes': {},
            'semantic_probes': {},
            'intrinsic_dimensions': {},
            'hessian': {},
            'pos_performance': {},
            'semantic_roles': {}
        }

        # Step tracking
        self.tracked_steps = []

    def should_track(self, step: int) -> bool:
        """
        Determine if analysis should be performed at this step.

        Args:
            step: Current training step

        Returns:
            Whether to perform analysis
        """
        return step % self.config.track_interval == 0 and step > 0

    def run_analysis(self, model, batch, hidden_states, step: int, val_loader=None):
        """
        Run all enabled analysis modules.

        Args:
            model: The training model
            batch: Current training batch
            hidden_states: Captured hidden states
            step: Current training step
            val_loader: Validation loader for some analyses
        """
        if not self.should_track(step):
            return

        print(f"Running analysis at step {step}...")
        self.tracked_steps.append(step)

        # Create a simple data loader from the current batch for analysis
        batch_loader = [batch]  # Simple wrapper for single batch

        # Run linguistic probes analysis
        if self.pos_linguistic_analyzer:
            try:
                print("Running linguistic probes analysis...")
                # TODO: Implement probe monitoring logic from original code
                print("Monitoring linguistic probes...TODO")
                # This should monitor confidence scores over time
                self._monitor_linguistic_probes(hidden_states, step)
            except Exception as e:
                print(f"Linguistic probes analysis failed: {e}")

        # Run semantic probes analysis
        if self.semantic_analyzer:
            try:
                print("Running semantic probes analysis...")
                # TODO: Implement semantic probe monitoring
                self._monitor_semantic_probes(hidden_states, step)
            except Exception as e:
                print(f"Semantic probes analysis failed: {e}")

        # Run intrinsic dimensions analysis
        if self.intrinsic_analyzer:
            try:
                print("Running intrinsic dimensions analysis...")
                id_results = self.intrinsic_analyzer.analyze(
                    model, batch_loader, model_name=f"step_{step}"
                )
                self.analysis_results['intrinsic_dimensions'][step] = id_results
                print(f"Intrinsic dimensions computed for {len(id_results)} layers")
            except Exception as e:
                print(f"Intrinsic dimensions analysis failed: {e}")

        # Run Hessian analysis (placeholder)
        if self.hessian_analyzer:
            try:
                print("Running Hessian analysis...")
                # TODO: Implement when Hessian module is refactored
                self._run_hessian_analysis(model, batch, step, val_loader)
            except Exception as e:
                print(f"Hessian analysis failed: {e}")

        # Run POS performance tracking (placeholder)
        if self.pos_tracker:
            try:
                print("Running POS performance tracking...")
                # TODO: Implement POS tracking
                pass
            except Exception as e:
                print(f"POS performance tracking failed: {e}")

        # Run semantic role tracking (placeholder)
        if self.semantic_role_tracker:
            try:
                print("Running semantic role tracking...")
                # TODO: Implement semantic role tracking
                pass
            except Exception as e:
                print(f"Semantic role tracking failed: {e}")

        # Run gradient analysis
        if self.config.track_gradients:
            try:
                print("Running gradient analysis...")
                self._track_gradients(model, step)
            except Exception as e:
                print(f"Gradient analysis failed: {e}")

    def _monitor_linguistic_probes(self, hidden_states: Dict[str, torch.Tensor], step: int):
        """
        Monitor linguistic probe predictions (from original code logic).

        Args:
            hidden_states: Dictionary of layer hidden states
            step: Current training step
        """
        # TODO: Implement the probe monitoring logic from original train_model
        # This should load pre-trained probes and monitor their confidence scores

        # Placeholder for the monitoring logic that was in the original code:
        # with torch.no_grad():
        #     for name, probe in probes.items():
        #         if name not in hidden_states:
        #             continue
        #         hidden = hidden_states[name]
        #         hidden_mean = hidden.mean(dim=1)
        #         preds = probe(hidden_mean)
        #         avg_conf = preds.mean(dim=0).cpu().numpy()
        #         for tag_idx, score in enumerate(avg_conf):
        #             probe_predictions[name][tag_idx].append((step, score))

        print(f"Linguistic probe monitoring at step {step} (placeholder)")
        pass

    def _monitor_semantic_probes(self, hidden_states: Dict[str, torch.Tensor], step: int):
        """
        Monitor semantic probe predictions.

        Args:
            hidden_states: Dictionary of layer hidden states
            step: Current training step
        """
        # TODO: Similar to linguistic probes but for semantic roles
        print(f"Semantic probe monitoring at step {step} (placeholder)")
        pass

    def _run_hessian_analysis(self, model, batch, step: int, val_loader=None):
        """
        Run Hessian analysis (placeholder for original Hessian logic).

        Args:
            model: Training model
            batch: Current batch
            step: Current training step
            val_loader: Validation loader
        """
        # TODO: Implement the Hessian analysis from original code
        # This includes:
        # - Basic Hessian eigenvalues
        # - Component-specific Hessian metrics
        # - Gradient-Hessian alignment
        # - Train-val landscape divergence

        print(f"Hessian analysis at step {step} (placeholder)")

        # Placeholder for original Hessian logic:
        # try:
        #     with torch.no_grad():
        #         original_state = copy.deepcopy(model.state_dict())
        #
        #     eigenvalues, _ = get_hessian_eigenvectors(...)
        #     hessian_metrics = compute_detailed_hessian_metrics(eigenvalues)
        #     self.analysis_results['hessian'][step] = hessian_metrics
        #
        #     if self.config.track_component_hessian:
        #         component_metrics = compute_component_hessians(...)
        #         # Store component results
        #
        #     if self.config.track_gradient_alignment:
        #         alignment_metrics = compute_hessian_gradient_alignment(...)
        #         # Store alignment results
        #
        #     # Restore model state
        #     model.load_state_dict(original_state)
        # except Exception as e:
        #     print(f"Hessian computation error: {e}")

        pass

    def _track_gradients(self, model, step: int):
        """
        Track gradient statistics and similarities.

        Args:
            model: Training model
            step: Current training step
        """
        # TODO: Implement gradient tracking from original code
        # This should compute gradient similarities and track changes

        # Placeholder for original gradient logic:
        # grads = model.get_grad()  # Need to implement this method
        #
        # for key, value in grads.items():
        #     if key in self.grad_history:
        #         self.grad_history[key].append(grads[key])
        #         if len(self.grad_history[key]) > 2:
        #             self.grad_history[key].pop(0)
        #     else:
        #         self.grad_history[key] = [grads[key]]
        #
        #     if len(self.grad_history[key]) > 1:
        #         cos_sim = torch.nn.functional.cosine_similarity(
        #             grads[key], self.grad_history[key][-2], dim=-1
        #         ).cpu().detach().numpy()
        #         self.grad_similarities_history[key].append((step, cos_sim))

        print(f"Gradient tracking at step {step} (placeholder)")
        pass

    def generate_final_visualizations(self, model_name: str):
        """
        Generate all final visualizations after training.

        Args:
            model_name: Name of the model for saving plots
        """
        print("Generating final visualizations...")

        # Generate linguistic probe visualizations
        if self.pos_linguistic_analyzer and self.analysis_results['linguistic_probes']:
            try:
                print("Generating linguistic probe visualizations...")
                # The visualizations should be generated automatically by the analyzer
                pass
            except Exception as e:
                print(f"Failed to generate linguistic probe visualizations: {e}")

        # Generate semantic probe visualizations
        if self.semantic_analyzer and self.analysis_results['semantic_probes']:
            try:
                print("Generating semantic probe visualizations...")
                pass
            except Exception as e:
                print(f"Failed to generate semantic probe visualizations: {e}")

        # Generate intrinsic dimensions visualizations
        if self.intrinsic_analyzer and self.analysis_results['intrinsic_dimensions']:
            try:
                print("Generating intrinsic dimensions evolution plots...")
                # Convert step-wise results to evolution format

                self.intrinsic_analyzer.visualizer.plot_final_id(
                        self.analysis_results['intrinsic_dimensions'], model_name
                )

            except Exception as e:
                print(f"Failed to generate intrinsic dimensions visualizations: {e}")

        # Generate Hessian visualizations (placeholder)
        if self.hessian_analyzer and self.analysis_results['hessian']:
            try:
                print("Generating Hessian visualizations...")
                # TODO: Generate Hessian plots when module is refactored
                pass
            except Exception as e:
                print(f"Failed to generate Hessian visualizations: {e}")

        # Generate gradient visualizations
        if self.config.track_gradients and self.grad_similarities_history:
            try:
                print("Generating gradient visualizations...")
                # TODO: Implement gradient plotting
                pass
            except Exception as e:
                print(f"Failed to generate gradient visualizations: {e}")

        print("Final visualizations completed!")

    def _convert_to_evolution_format(self, step_results: Dict[int, Dict]) -> Dict:
        """
        Convert step-wise results to evolution format for plotting.

        Args:
            step_results: Dictionary mapping steps to analysis results

        Returns:
            Dictionary in evolution format for visualization
        """
        evolution_data = defaultdict(list)

        for step, results in step_results.items():
            for layer_key, value in results.items():
                evolution_data[layer_key].append((step, value))

        return dict(evolution_data)

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all analysis results.

        Returns:
            Dictionary with analysis summary
        """
        summary = {
            'steps_analyzed': self.tracked_steps,
            'num_steps': len(self.tracked_steps),
            'analysis_modules': {
                'linguistic_probes': self.pos_linguistic_analyzer is not None,
                'semantic_probes': self.semantic_analyzer is not None,
                'intrinsic_dimensions': self.intrinsic_analyzer is not None,
                'hessian': self.hessian_analyzer is not None,
                'pos_performance': self.pos_tracker is not None,
                'semantic_roles': self.semantic_role_tracker is not None,
                'gradients': self.config.track_gradients
            },
            'results': self.analysis_results
        }

        return summary