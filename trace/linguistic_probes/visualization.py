import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
from matplotlib.lines import Line2D
from .config import LinguisticProbesConfig


class ProbesVisualizer:
    """
    Fixed visualizer that matches the original plotting functionality.
    """

    def __init__(self, log_dir: Optional[str] = None, config: Optional[LinguisticProbesConfig] = None):
        self.log_dir = log_dir
        self.config = config or LinguisticProbesConfig.default()

        # Set up matplotlib style to match original
        plt.style.use('seaborn-v0_8-whitegrid')

        # # Create output directories
        # if not log_dir:
        #     log_dir = './plots/probe_analysis/probe_monitor_plots'
        #
        # self.save_dir = log_dir
        # os.makedirs(self.save_dir, exist_ok=True)
        # Create output directories
        if not log_dir:
            log_dir = './analysis_results'
        self.pos_plots_dir = os.path.join(log_dir, 'pos_probe_analysis')
        self.semantic_plots_dir = os.path.join(log_dir, 'semantic_probe_analysis')
        os.makedirs(self.pos_plots_dir, exist_ok=True)
        os.makedirs(self.semantic_plots_dir, exist_ok=True)
    def convert_confidence_data_to_original_format(
            self,
            confidence_data: Dict[int, Dict[Union[int, tuple], Dict[str, float]]],
            label_names: List[str]
    ) -> Dict[str, Dict[int, List[Tuple[int, float]]]]:
        """
        Convert the new confidence data format to the original format expected by plotting functions.

        Args:
            confidence_data: {step: {layer_key: {'TAG_NAME': confidence, ...}}}
            label_names: List of label names

        Returns:
            Original format: {layer_name: {tag_idx: [(step, score), ...]}}
        """
        # Create tag name to index mapping
        tag_to_idx = {name: idx for idx, name in enumerate(label_names)}

        # Initialize result structure
        probe_predictions = {}

        # Process each step
        for step, step_data in confidence_data.items():
            for layer_key, tag_confidences in step_data.items():
                # Convert layer key to string for consistency
                if isinstance(layer_key, tuple):
                    layer_name = f"layer_{layer_key[0]}_{layer_key[1]}"
                else:
                    layer_name = f"layer_{layer_key}"

                # Initialize layer if not exists
                if layer_name not in probe_predictions:
                    probe_predictions[layer_name] = {}

                # Process each tag
                for tag_name, confidence in tag_confidences.items():
                    if tag_name in tag_to_idx:
                        tag_idx = tag_to_idx[tag_name]

                        # Initialize tag list if not exists
                        if tag_idx not in probe_predictions[layer_name]:
                            probe_predictions[layer_name][tag_idx] = []

                        # Add data point
                        probe_predictions[layer_name][tag_idx].append((step, confidence))

        # Sort all data points by step
        for layer_name in probe_predictions:
            for tag_idx in probe_predictions[layer_name]:
                probe_predictions[layer_name][tag_idx].sort(key=lambda x: x[0])

        return probe_predictions

    def plot_probe_predictions_enhanced(
            self,
            probe_predictions: Dict[str, Dict[int, List[Tuple[int, float]]]],
            label_names: Optional[List[str]] = None,
            model_name: Optional[str] = None,
            number_classes: int = 8,
            save_dir: Optional[str] = None
    ):
        if save_dir is None:
            save_dir = self.save_dir

        os.makedirs(save_dir, exist_ok=True)

        if label_names is None:
            label_names = [f"POS {i}" for i in range(number_classes)]

        # Per tag, per layer plots
        for layer_name, tag_data in probe_predictions.items():
            for tag_idx, history in tag_data.items():
                if not history:  # Skip empty histories
                    continue

                steps, scores = zip(*history)
                plt.figure(figsize=(8, 3))
                plt.plot(steps, scores, label=f"{label_names[tag_idx]}")
                plt.title(f"Probe Confidence Over Time - {layer_name} - POS: {label_names[tag_idx]}")
                plt.xlabel("Step")
                plt.ylabel("Confidence")
                plt.legend(loc="lower left")
                plt.tight_layout()

                # Safe filename
                safe_tag_name = label_names[tag_idx].replace('/', '_').replace(' ', '_')
                filename = f"{model_name}_{layer_name}_tag{tag_idx}_{safe_tag_name}.png"
                plt.savefig(os.path.join(save_dir, filename), dpi=300)
                plt.close()

        # Combined plot: all tags in one graph per layer
        for layer_name, tag_data in probe_predictions.items():
            plt.figure(figsize=(10, 5))
            for tag_idx, history in tag_data.items():
                if not history:  # Skip empty histories
                    continue

                steps, scores = zip(*history)
                plt.plot(steps, scores, label=label_names[tag_idx])

            plt.title(f"All POS Tags - Confidence Over Time - {layer_name}")
            plt.xlabel("Step")
            plt.ylabel("Confidence")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{model_name}_{layer_name}_all_tags_combined.png"), dpi=300)
            plt.close()

        # One plot: each tag, each line = layer
        num_tags = len(label_names)
        for tag_idx in range(num_tags):
            plt.figure(figsize=(10, 5))
            for layer_name, tag_data in probe_predictions.items():
                if tag_idx in tag_data and tag_data[tag_idx]:
                    steps, scores = zip(*tag_data[tag_idx])
                    plt.plot(steps, scores, label=f"{layer_name}")

            plt.title(f"POS Tag: {label_names[tag_idx]} - Confidence Across Layers")
            plt.xlabel("Step")
            plt.ylabel("Confidence")
            plt.legend(loc="lower left")
            plt.tight_layout()

            safe_tag_name = label_names[tag_idx].replace('/', '_').replace(' ', '_')
            filename = f"{model_name}_all_layers_tag{tag_idx}_{safe_tag_name}.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
            plt.close()

    def plot_all_tags_all_layers(
            self,
            probe_predictions: Dict[str, Dict[int, List[Tuple[int, float]]]],
            label_names: Optional[List[str]] = None,
            model_name: Optional[str] = None,
            number_of_classes: int = 8,
            save_path: Optional[str] = None,
            show_plot: bool = False
    ):
        """
        Comprehensive plot showing all tags across all layers .
        """
        if label_names is None:
            label_names = [f"POS {i}" for i in range(number_of_classes)]

        if save_path is None:
            save_path = self.save_dir

        plt.style.use('seaborn-v0_8-whitegrid')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 7))

        styles = ["-", "--", "-.", ":"]
        color_map = plt.get_cmap("tab10")
        layer_names = sorted(probe_predictions.keys())

        # For POS legend (colors only)
        pos_handles = {}
        # For Layer legend (line styles only)
        layer_handles = {}

        for tag_idx, pos_name in enumerate(label_names):
            color = color_map(tag_idx % 10)
            for i, layer_name in enumerate(layer_names):
                if tag_idx not in probe_predictions[layer_name]:
                    continue

                history = probe_predictions[layer_name][tag_idx]
                if not history:
                    continue

                steps, scores = zip(*history)
                linestyle = styles[i % len(styles)]

                line = ax.plot(
                    steps,
                    scores,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    label=None  # suppress default label
                )

                if pos_name not in pos_handles:
                    pos_handles[pos_name] = Line2D([0], [0], color=color, lw=3)

                if layer_name not in layer_handles:
                    layer_handles[layer_name] = Line2D([0], [0], color="black", lw=2, linestyle=linestyle)

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Confidence", fontsize=12)
        ax.set_title("All POS Tags Across All Layers (Color = POS, Line Style = Layer)", fontsize=14)
        ax.grid(True, linestyle="-", alpha=0.3)

        # POS tag legend
        if pos_handles:
            legend1 = ax.legend(pos_handles.values(), pos_handles.keys(), title="POS Tags", loc="lower left")

        # Layer line style legend
        if layer_handles:
            legend2 = ax.legend(layer_handles.values(), layer_handles.keys(), title="Layer Styles", loc="lower right")

            # Add the first legend back if both exist
            if pos_handles:
                ax.add_artist(legend1)

        plt.tight_layout()

        if show_plot:
            plt.show()
        else:
            plt.savefig(os.path.join(save_path, f"{model_name}_all_tags_all_layers.png"), dpi=300)
            plt.close()

    def plot_probe_confidence_analysis(
            self,
            confidence_data: Dict[int, Dict[Union[int, tuple], Dict[str, float]]],
            model_name: str = '',
            analysis_type: str = 'pos',
            save_plot: bool = True,
            show_plots: bool = False,
    ) -> None:
        """
        Main plotting function that converts data and calls original plotting functions.
        """
        if not confidence_data:
            print(f"No {analysis_type} confidence data available for plotting")
            return

        # Get label names based on analysis type
        if analysis_type == 'pos':
            label_names = self._get_pos_label_names()
            save_dir = self.pos_plots_dir
            number_classes = len(label_names)
        elif analysis_type == 'semantic':
            label_names = self._get_semantic_label_names()
            save_dir = self.semantic_plots_dir
            number_classes = len(label_names)
        else:
            print(f"Unknown analysis type: {analysis_type}")
            return

        # Convert to original format
        probe_predictions = self.convert_confidence_data_to_original_format(
            confidence_data, label_names
        )

        if not probe_predictions:
            print("No valid probe predictions data for plotting")
            return

        print(f"Creating {analysis_type} plots with {len(probe_predictions)} layers and {len(label_names)} tags")

        self.plot_probe_predictions_enhanced(
            probe_predictions,
            label_names,
            model_name,
            number_classes,
            save_dir
        )

        self.plot_all_tags_all_layers(
            probe_predictions,
            label_names,
            model_name,
            number_classes,
            save_dir,
            show_plots
        )

        if save_plot:
            print(f"Plots saved to: {self.save_dir}")

    def _get_pos_label_names(self) -> List[str]:
        """Get POS label names from config."""
        if self.config:
            pos_categories = self.config.get_pos_categories()
            sorted_categories = sorted(pos_categories.items(), key=lambda x: x[1])
            return [name for name, _ in sorted_categories]
        else:
            return ["NOUN", "VERB", "ADJ", "ADV", "PREP", "CONJ", "OTHER"]

    def _get_semantic_label_names(self) -> List[str]:
        """Get semantic label names from config."""
        if self.config:
            semantic_categories = self.config.get_semantic_categories()
            sorted_categories = sorted(semantic_categories.items(), key=lambda x: x[1])
            return [name for name, _ in sorted_categories]
        else:
            return ["AGENT", "PATIENT", "ACTION", "LOCATION", "RELATION", "CONNECTOR", "RESULT", "OTHER"]

