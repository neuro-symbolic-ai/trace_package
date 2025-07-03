import os
import numpy as np
import torch
import copy
from typing import Dict, Any, Optional, List
from collections import defaultdict

from ..hessian import HessianAnalyzer, HessianConfig
from ..linguistic_probes import LinguisticProbesConfig, POSAnalyzer, SemanticAnalyzer, MultiLabelProbe
from ..intrisic_dimensions import IntrinsicDimensionAnalyzer, IntrinsicDimensionsConfig
from ..output_monitoring import OutputMonitoringConfig, OutputMonitoringAnalyzer


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
        print(analysis_results.keys())
        print(analysis_results.get('results', {}).keys())
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

        if analysis_results.get('results').get('linguistic_probes'):
            pos_probe_log_dir = os.path.join(analysis_results_path, 'pos_probe_analysis')
            os.makedirs(pos_probe_log_dir, exist_ok=True)
            save_json_data(analysis_results.get('results').get('linguistic_probes'), 'pos_probe_history.json', pos_probe_log_dir)

        if analysis_results.get('results').get('semantic_probes'):
            semantic_probe_log_dir = os.path.join(analysis_results_path, 'semantic_probe_analysis')
            os.makedirs(semantic_probe_log_dir, exist_ok=True)
            save_json_data(analysis_results.get('results').get('semantic_probes'), 'semantic_probe_history.json', semantic_probe_log_dir)

        if analysis_results.get('results').get('hessian'):
            hessian_log_dir = os.path.join(analysis_results_path, 'hessian')
            os.makedirs(hessian_log_dir, exist_ok=True)
            save_json_data(analysis_results.get('results').get('hessian'), 'hessian_history.json', hessian_log_dir)

        if analysis_results.get('results').get('output_pos_performance'):
            # print("POS performance results:", analysis_results.get('results').get('output_pos_performance'))
            pos_log_dir = os.path.join(analysis_results_path, 'output_pos_performance')
            os.makedirs(pos_log_dir, exist_ok=True)
            save_json_data(analysis_results.get('results').get('output_pos_performance'), 'output_pos_performance.json', pos_log_dir)
        if analysis_results.get('results').get('output_semantic_roles_performance'):
            # print("Semantic roles results:", analysis_results.get('results').get('output_semantic_roles_performance'))
            semantic_log_dir = os.path.join(analysis_results_path, 'output_semantic_roles_performance')
            os.makedirs(semantic_log_dir, exist_ok=True)
            save_json_data(analysis_results.get('results').get('output_semantic_roles_performance'), 'output_semantic_roles_performance.json', semantic_log_dir)

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
            # Determine number of POS categories
            self.config.probe_num_features = len(self.config.get_pos_categories())
            probe_config = LinguisticProbesConfig(
                probe_type=self.config.probe_type,
                layer_indices=self.config.probe_layers,
                num_classes=self.config.probe_num_features,
                hidden_dim=self.config.probe_hidden_dim,
                lr=self.config.probe_lr,
                epochs=self.config.probe_epochs,
                log_dir=self.config.log_dir,
                save_visualizations=True,
                device=self.device,
                pos_granularity= self.config.pos_granularity,
            )
            self.probe_pos_linguistic_analyzer = POSAnalyzer(probe_config)
            # Load pre-trained probes if paths provided
            if hasattr(self.config, 'probe_load_paths') and self.config.probe_load_paths:
                self.probe_pos_linguistic_analyzer.load_probes(self.config.probe_load_paths)
                print(f"Loaded POS probes from: {self.config.probe_load_paths}")
            else:
                print("Warning: No probe paths provided - analysis will skip layers without probes")
                self.probe_pos_linguistic_analyzer.load_probes(self.config.probe_load_paths)
                # self.track_linguistic_probe = False  # Disable semantic tracking if no probes are loaded
                # self.probe_pos_linguistic_analyzer = None  # Set to None if no probes are loaded
        else:
            self.probe_pos_linguistic_analyzer = None

        # Semantic Probes Analyzer
        if self.config.track_semantic_probes:
            print("Setting up semantic probes analyzer...")
            self.config.semantic_probe_num_features = len(self.config.get_semantic_categories())
            semantic_config = LinguisticProbesConfig(
                probe_type=self.config.semantic_probe_type,
                layer_indices=self.config.semantic_probe_layers,
                num_classes=self.config.semantic_probe_num_features,
                hidden_dim=self.config.semantic_probe_hidden_dim,
                lr=self.config.semantic_probe_lr,
                epochs=self.config.semantic_probe_epochs,
                log_dir=self.config.log_dir,
                save_visualizations=self.config.save_visualization,
                device=self.device,
                semantic_granularity=self.config.semantic_granularity,
            )
            self.probe_semantic_analyzer = SemanticAnalyzer(semantic_config)
            # Load pre-trained probes if paths provided
            if hasattr(self.config, 'semantic_probe_load_path') and self.config.semantic_probe_load_path:
                self.probe_semantic_analyzer.load_probes(self.config.semantic_probe_load_path)
                print(f"Loaded semantic probes from: {self.config.semantic_probe_load_path}")
            else:
                print("Warning: No semantic probe paths provided - analysis will skip layers without probes")
                # self.track_semantic_probes = False  # Disable semantic tracking if no probes are loaded
                self.probe_semantic_analyzer.load_probes(self.config.semantic_probe_load_path)
                # self.probe_semantic_analyzer = None  # Set to None if no probes are loaded
        else:
            self.probe_semantic_analyzer = None

        # Intrinsic Dimensions Analyzer
        if self.config.track_intrinsic_dimensions:
            print("Setting up intrinsic dimensions analyzer...")
            id_config = IntrinsicDimensionsConfig(
                model_type=self.config.model_type,
                layers_to_analyze=self.config.id_selected_layers,  # Use same layers as probes
                id_method=self.config.id_method,
                log_dir=self.config.log_dir
            )
            self.intrinsic_analyzer = IntrinsicDimensionAnalyzer(id_config)
        else:
            self.intrinsic_analyzer = None

        # Hessian Analyzer
        if self.config.track_hessian:
            hessian_config = HessianConfig(
                n_components=self.config.hessian_n_components,
                track_component_hessian=self.config.track_component_hessian,
                track_gradient_alignment=self.config.track_gradient_alignment,
                component_list=self.config.component_list,
                track_train_val_landscape_divergence=self.config.track_train_val_landscape_divergence,
                save_hessian_data=self.config.save_hessian_data,
                loss_fn=self.config.hessian_loss_fn,
                log_dir=self.config.log_dir,
            )
            self.hessian_analyzer = HessianAnalyzer(hessian_config)
            print("Hessian analysis will be added when module is refactored")
        else:
            self.hessian_analyzer = None

        # POS Performance Tracker (placeholder)
        if self.config.track_pos_performance:
            pos_performance_config = OutputMonitoringConfig(
                model_type=self.config.model_type,
                track_pos_performance=self.config.track_pos_performance,
                pos_granularity=self.config.pos_granularity,
                track_semantic_roles=False,  # Not used here
                save_visualizations=self.config.save_visualization,
                device=self.device
            )
            self.pos_tracker = OutputMonitoringAnalyzer(pos_performance_config)
            print("POS performance tracking will be added later")
        else:
            self.pos_tracker = None

        # Semantic Role Tracker
        if self.config.track_semantic_roles_performance:
            print("Semantic role tracking will be added later")
            semantic_performance_config = OutputMonitoringConfig(
                model_type=self.config.model_type,
                track_pos_performance=False,  # Not used here
                track_semantic_roles=self.config.track_semantic_roles_performance,
                semantic_granularity=self.config.semantic_granularity,
                save_visualizations=self.config.save_visualization,
                device=self.device
            )
            self.semantic_role_tracker = OutputMonitoringAnalyzer(semantic_performance_config)
        else:
            self.semantic_role_tracker = None

            # TODO: Add other analyzers when refactored

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
            'output_pos_performance': {},
            'output_semantic_roles_performance': {}
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

    def run_analysis(self, model, batch, hidden_states, step: int, val_loader=None, tokenizer=None, predictions=None):
        """
        Run all enabled analysis modules.

        Args:
            model: The training model
            batch: Current training batch
            hidden_states: Captured hidden states
            step: Current training step
            val_loader: Validation loader for some analyses
            tokenizer: Tokenizer for decoding (if needed)
        """
        if not self.should_track(step):
            return

        print(f"Running analysis at step {step}...")
        self.tracked_steps.append(step)

        # Create a simple data loader from the current batch for analysis
        batch_loader = [batch]  # Simple wrapper for single batch

        # Run linguistic probes analysis
        if self.probe_pos_linguistic_analyzer:
            try:
                print("Running linguistic probes analysis...")
                # Use the updated analyzer that works with pre-trained probes
                pos_results = self.probe_pos_linguistic_analyzer.analyze(
                    model, batch_loader, tokenizer=self.tokenizer, model_name=f"step_{step}"
                )
                self.analysis_results['linguistic_probes'][step] = pos_results
                print(f"POS analysis completed - {len(pos_results)} layers analyzed")
            except Exception as e:
                print(f"Linguistic probes analysis failed: {e}")

        # Run semantic probes analysis
        if self.probe_semantic_analyzer:
            try:
                print("Running semantic probes analysis...")
                semantic_results = self.probe_semantic_analyzer.analyze(
                    model, batch_loader, model_name=f"step_{step}", tokenizer=self.tokenizer,
                )
                self.analysis_results['semantic_probes'][step] = semantic_results
                print(f"Semantic analysis completed - {len(semantic_results)} layers analyzed")
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
                # Use the HessianAnalyzer to perform analysis
                hessian_results = self.hessian_analyzer.analyze_step(
                    model, self.config.hessian_loss_fn,
                    train_batch=batch, val_batch=batch,
                    model_type=model.model_type, step=step
                )
                self.analysis_results['hessian'][step] = hessian_results
            except Exception as e:
                print(f"Hessian analysis failed: {e}")

        # Run POS performance tracking
        if self.pos_tracker:
            try:
                print("Running POS performance tracking...")
                pos_performance = self.pos_tracker.analyze(batch=batch, outputs=predictions, tokenizer=tokenizer, step=step)
                self.analysis_results['output_pos_performance'][step] = pos_performance

            except Exception as e:
                print(f"POS performance tracking failed: {e}")

        # Run semantic role tracking (placeholder)
        if self.semantic_role_tracker:
            try:
                print("Running semantic role tracking...")
                semantic_performance = self.semantic_role_tracker.analyze(
                    batch=batch, outputs=predictions, tokenizer=tokenizer, step=step
                )
                self.analysis_results['output_semantic_roles_performance'][step] = semantic_performance
            except Exception as e:
                print(f"Semantic role tracking failed: {e}")

        # Run gradient analysis
        if self.config.track_gradients:
            try:
                print("Running gradient analysis...")
                # self._track_gradients(model, step)
            except Exception as e:
                print(f"Gradient analysis failed: {e}")


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
        if self.probe_pos_linguistic_analyzer and self.analysis_results['linguistic_probes']:
            try:
                print("Generating linguistic probe visualizations...")
                self.probe_pos_linguistic_analyzer.visualizer.plot_probe_confidence_analysis(
                    confidence_data= self.analysis_results['linguistic_probes'],
                    model_name=model_name,
                    analysis_type=self.probe_pos_linguistic_analyzer.get_analysis_type(),
                    show_plots=self.config.show_plots,
                )
                # The visualizations should be generated automatically by the analyzer
            except Exception as e:
                print(f"Failed to generate linguistic probe visualizations: {e}")

        # Generate semantic probe visualizations
        if self.probe_semantic_analyzer and self.analysis_results['semantic_probes']:
            try:
                print("Generating semantic probe visualizations...")
                self.probe_semantic_analyzer.visualizer.plot_probe_confidence_analysis(
                    confidence_data=self.analysis_results['semantic_probes'],
                    model_name=model_name,
                    analysis_type=self.probe_semantic_analyzer.get_analysis_type(),
                    show_plots= self.config.show_plots,
                )
            except Exception as e:
                print(f"Failed to generate semantic probe visualizations: {e}")

        # Generate intrinsic dimensions visualizations
        if self.intrinsic_analyzer and self.analysis_results['intrinsic_dimensions']:
            try:
                print("Generating intrinsic dimensions evolution plots...")
                # Convert step-wise results to evolution format
                self.intrinsic_analyzer.visualizer.plot_final_id(
                        self.analysis_results['intrinsic_dimensions'], model_name,
                )

            except Exception as e:
                print(f"Failed to generate intrinsic dimensions visualizations: {e}")

        # Generate POS performance visualizations
        if self.pos_tracker and self.analysis_results['output_pos_performance']:
            try:
                print("Generating POS performance visualizations...")
                self.pos_tracker.visualizer.plot_pos_performance_evolution(
                    monitoring_results=self.analysis_results['output_pos_performance'],
                    model_name=model_name,
                )
            except Exception as e:
                print(f"Failed to generate POS performance visualizations: {e}")
        # Generate semantic role visualizations
        if self.semantic_role_tracker and self.analysis_results['output_semantic_roles_performance']:
            try:
                print("Generating semantic role visualizations...")
                self.semantic_role_tracker.visualizer.plot_semantic_role_performance_evolution(
                    monitoring_results=self.analysis_results['output_semantic_roles_performance'],
                    model_name=model_name,
                )
            except Exception as e:
                print(f"Failed to generate semantic role visualizations: {e}")


        # Generate Hessian visualizations
        if self.hessian_analyzer and self.analysis_results['hessian']:
            try:
                print("Generating Hessian visualizations...")
                self.hessian_analyzer.visualizer.plot_eigenvalue_evolution(
                    hessian_history=self.analysis_results['hessian'], model_name= model_name,
                )
                self.hessian_analyzer.visualizer.plot_eigenvalue_heatmap(
                    hessian_history=self.analysis_results['hessian'], model_name= model_name,
                )

                # We check one by one for the hessian visualizations
                if self.hessian_analyzer.config.track_component_hessian: #components
                    self.hessian_analyzer.visualizer.plot_component_comparison( #results["components"]
                        hessian_history=self.analysis_results['hessian'], model_name=model_name,
                    )
                if self.hessian_analyzer.config.track_gradient_alignment: #alignment
                    self.hessian_analyzer.visualizer.plot_gradient_alignment(
                        hessian_history=self.analysis_results['hessian'], model_name=model_name,
                    )
                # memorization
                if self.hessian_analyzer.config.track_train_val_landscape_divergence: #train-val divergence
                    self.hessian_analyzer.visualizer.plot_memorization_metrics(
                        hessian_history=self.analysis_results['hessian'], model_name=model_name,
                )
            except Exception as e:
                print(f"Failed to generate Hessian visualizations: {e}")



        # Generate gradient visualizations
        if self.config.track_gradients and self.analysis_results['grad_similarities_history']:
            try:
                print("Generating gradient visualizations...")
                # TODO: Implement gradient plotting
                pass
            except Exception as e:
                print(f"Failed to generate gradient visualizations: {e}")

        print("Final visualizations completed!")

    # def _convert_to_evolution_format(self, step_results: Dict[int, Dict]) -> Dict:
    #     """
    #     Convert step-wise results to evolution format for plotting.
    #
    #     Args:
    #         step_results: Dictionary mapping steps to analysis results
    #
    #     Returns:
    #         Dictionary in evolution format for visualization
    #     """
    #     evolution_data = defaultdict(list)
    #
    #     for step, results in step_results.items():
    #         for layer_key, value in results.items():
    #             evolution_data[layer_key].append((step, value))
    #
    #     return dict(evolution_data)

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
                'linguistic_probes': self.probe_pos_linguistic_analyzer is not None,
                'semantic_probes': self.probe_semantic_analyzer is not None,
                'intrinsic_dimensions': self.intrinsic_analyzer is not None,
                'hessian': self.hessian_analyzer is not None,
                'output_pos_performance': self.pos_tracker is not None,
                'output_semantic_roles_performance': self.semantic_role_tracker is not None,
                'gradients': self.config.track_gradients
            },
            'results': self.analysis_results
        }

        return summary