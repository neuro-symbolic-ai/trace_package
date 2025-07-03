import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from matplotlib.lines import Line2D

from .config import LinguisticProbesConfig

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

        # Create output directories
        if not log_dir:
            log_dir = './plots/probe_analysis/probe_monitor_plots'

        self.save_dir = log_dir
        os.makedirs(self.save_dir, exist_ok=True)

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
        """
        Enhanced plotting function that matches the original implementation.
        """
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
        Comprehensive plot showing all tags across all layers (matches original).
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
            number_classes = len(label_names)
        elif analysis_type == 'semantic':
            label_names = self._get_semantic_label_names()
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

        # Use original plotting functions
        self.plot_probe_predictions_enhanced(
            probe_predictions,
            label_names,
            model_name,
            number_classes,
            self.save_dir
        )

        self.plot_all_tags_all_layers(
            probe_predictions,
            label_names,
            model_name,
            number_classes,
            self.save_dir,
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

    # Legacy compatibility methods
    def save_metrics(self, accuracy_data, sample_counts, model_name='', analysis_type='pos'):
        """Save metrics to CSV (compatibility method)."""
        pass  # Implement if needed

# class ProbesVisualizer:
#     """
#     Visualizer for probe analysis results.
#
#     This class handles all visualization tasks for both POS and semantic
#     role analysis, following the reference style pattern.
#     """
#
#     def __init__(self, log_dir: Optional[str] = None, config: Optional[LinguisticProbesConfig] = None):
#         """
#         Initialize the visualizer.
#
#         Args:
#             log_dir: Directory to save visualizations
#             config: Configuration object
#         """
#         self.log_dir = log_dir
#         self.config = config or LinguisticProbesConfig.default()
#         self.confidence_data = {}  # Will hold the confidence data for plotting
#
#         # Set up matplotlib style
#         plt.style.use('seaborn-v0_8-whitegrid')
#
#         # Create output directories
#         if not log_dir:
#             log_dir = './analysis_results'
#         self.pos_plots_dir = os.path.join(log_dir, 'pos_probe_analysis')
#         self.semantic_plots_dir = os.path.join(log_dir, 'semantic_probe_analysis')
#         os.makedirs(self.pos_plots_dir, exist_ok=True)
#         os.makedirs(self.semantic_plots_dir, exist_ok=True)
#
#     def save_metrics(
#             self,
#             accuracy_data: Dict[str, List[tuple]],
#             sample_counts: Dict[str, int],
#             model_name: str = '',
#             analysis_type: str = 'pos'
#     ) -> None:
#         """
#         Save metrics to CSV files.
#
#         Args:
#             accuracy_data: Accuracy data by category
#             sample_counts: Sample counts per category
#             model_name: Name of the model
#             analysis_type: Type of analysis ('pos' or 'semantic')
#         """
#         if not self.log_dir:
#             return
#
#         plots_dir = getattr(self, f'{analysis_type}_plots_dir', self.log_dir)
#
#         # Create dataframes for each category
#         for category, data_points in accuracy_data.items():
#             if data_points:
#                 # Convert list of tuples to dataframe
#                 steps, accuracies = zip(*data_points)
#                 category_df = pd.DataFrame({
#                     'step': steps,
#                     'accuracy': accuracies
#                 })
#                 category_df.to_csv(
#                     os.path.join(plots_dir, f'{model_name}_{category}_accuracy.csv'),
#                     index=False
#                 )
#
#         # Save combined file with all categories
#         all_data = []
#         for category, data_points in accuracy_data.items():
#             for step, acc in data_points:
#                 all_data.append({
#                     'step': step,
#                     'category': category,
#                     'accuracy': acc
#                 })
#
#         if all_data:
#             combined_df = pd.DataFrame(all_data)
#
#             # Handle potential duplicate step values by taking mean
#             grouped_df = combined_df.groupby(['step', 'category']).mean().reset_index()
#
#             # Pivot without duplicates
#             pivot_df = grouped_df.pivot(index='step', columns='category', values='accuracy')
#             pivot_df.to_csv(os.path.join(plots_dir, f'{model_name}_all_{analysis_type}_accuracy.csv'))
#
#         # Save sample counts
#         counts_df = pd.DataFrame({
#             f'{analysis_type}_category': list(sample_counts.keys()),
#             'count': list(sample_counts.values())
#         })
#         counts_df.to_csv(
#             os.path.join(plots_dir, f'{model_name}_{analysis_type}_counts.csv'),
#             index=False
#         )
#
#
#     def _organize_confidence_data_fixed(self) -> None:
#         """
#         Organize the confidence data for easier plotting (FIXED VERSION).
#
#         Returns:
#             Dictionary with organized data structure for plotting
#         """
#         self.organized = {
#             'steps': [],
#             'layers': set(),
#             'tags': set(),
#             'data': {}  # Will be: data[tag][layer] = [(step, confidence), ...]
#         }
#
#         # Extract unique steps, layers, and tags
#         for step, step_data in self.confidence_data.items():
#             self.organized['steps'].append(step)
#
#             for layer_key, tag_confidences in step_data.items():
#                 self.organized['layers'].add(layer_key)
#
#                 for tag_name, confidence in tag_confidences.items():
#                     self.organized['tags'].add(tag_name)
#
#                     # Initialize nested structure if needed
#                     if tag_name not in self.organized['data']:
#                         self.organized['data'][tag_name] = {}
#                     if layer_key not in self.organized['data'][tag_name]:
#                         self.organized['data'][tag_name][layer_key] = []
#
#                     # Add the data point
#                     self.organized['data'][tag_name][layer_key].append((step, confidence))
#
#         # Sort steps and convert sets to sorted lists
#         self.organized['steps'] = sorted(self.organized['steps'])
#         self.organized['layers'] = sorted(list(self.organized['layers']))
#         self.organized['tags'] = sorted(list(self.organized['tags']))
#
#         # Sort data points by step for each tag-layer combination
#         for tag_name in self.organized['data']:
#             for layer_key in self.organized['data'][tag_name]:
#                 self.organized['data'][tag_name][layer_key].sort(key=lambda x: x[0])
#
#     def _plot_per_tag_analysis_fixed(self, model_name: str, analysis_type: str, show_plot: bool = False) -> None:
#         """
#         Create plots showing each tag's confidence across layers and steps.
#         """
#         print("Creating per-tag analysis plots...")
#
#         for tag_name in self.organized['tags']:
#             plt.figure(figsize=(10, 5))
#
#             # Plot each layer as a separate line
#             for layer_key in self.organized['layers']:
#                 if tag_name in self.organized['data'] and layer_key in self.organized['data'][tag_name]:
#                     data_points = self.organized['data'][tag_name][layer_key]
#                     if data_points:
#                         steps, confidences = zip(*data_points)
#                         layer_label = f"Layer {layer_key[0]} ({layer_key[1]})" if isinstance(layer_key,
#                                                                                              tuple) else f"Layer {layer_key}"
#                         plt.plot(steps, confidences, marker='o', linewidth=2, label=layer_label, alpha=0.8)
#
#             plt.title(f'{analysis_type.upper()} Tag: {tag_name} - Confidence Across Layers ({model_name})', fontsize=14)
#             plt.xlabel('Training Step', fontsize=14)
#             plt.ylabel('Confidence Score', fontsize=14)
#             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#             plt.grid(True, alpha=0.3)
#             plt.ylim(0, 1.1)
#             plt.tight_layout()
#
#             if analysis_type == 'pos':
#                 save_dir = self.pos_plots_dir
#             elif analysis_type == 'semantic':
#                 save_dir = self.semantic_plots_dir
#             else:
#                 print(f"Unknown analysis type: {analysis_type}. Use 'pos' or 'semantic'.")
#                 return
#
#             safe_tag_name = tag_name.replace('/', '_').replace(' ', '_')
#             plt.savefig(
#                     os.path.join(save_dir, f"{model_name}_{analysis_type}_tag_{safe_tag_name}_across_layers.png"),
#                     dpi=300, bbox_inches='tight'
#             )
#             if show_plot:
#                 plt.show()
#             plt.close()
#
#     def _plot_per_layer_analysis_fixed(self, model_name: str, analysis_type: str, show_plot: bool = False) -> None:
#         """
#         Create plots showing all tags for each layer across steps.
#         """
#         print("Creating per-layer analysis plots...")
#
#         for layer_key in self.organized['layers']:
#             plt.figure(figsize=(10, 5))
#
#             # Plot each tag as a separate line for this layer
#             for tag_name in self.organized['tags']:
#                 if tag_name in self.organized['data'] and layer_key in self.organized['data'][tag_name]:
#                     data_points = self.organized['data'][tag_name][layer_key]
#                     if data_points:
#                         steps, confidences = zip(*data_points)
#                         plt.plot(steps, confidences, marker='o', linewidth=2, label=tag_name, alpha=0.8)
#
#             layer_label = f"Layer {layer_key[0]} ({layer_key[1]})" if isinstance(layer_key,
#                                                                                  tuple) else f"Layer {layer_key}"
#             plt.title(f'{analysis_type.upper()} Analysis: All Tags - {layer_label} ({model_name})', fontsize=14)
#             plt.xlabel('Training Step', fontsize=14)
#             plt.ylabel('Confidence Score', fontsize=14)
#             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#             plt.grid(True, alpha=0.3)
#             plt.ylim(0, 1.1)
#             plt.tight_layout()
#
#             if analysis_type == 'pos':
#                 save_dir = self.pos_plots_dir
#             elif analysis_type == 'semantic':
#                 save_dir = self.semantic_plots_dir
#             else:
#                 print(f"Unknown analysis type: {analysis_type}. Use 'pos' or 'semantic'.")
#                 return
#
#             safe_layer_name = str(layer_key).replace('(', '').replace(')', '').replace(',', '_').replace("'",
#                                                                                                              '').replace(
#                     ' ', '')
#             plt.savefig(
#                     os.path.join(save_dir, f"{model_name}_{analysis_type}_layer_{safe_layer_name}_all_tags.png"),
#                     dpi=300, bbox_inches='tight'
#             )
#
#             if show_plot:
#                 plt.show()
#             plt.close()
#
#     def _plot_all_tags_all_layers_fixed(self, model_name: str, analysis_type: str, show_plot: bool = False) -> None:
#         """
#         Create a comprehensive plot showing all tags across all layers.
#         """
#         print("Creating comprehensive all-tags-all-layers plot...")
#
#         # Use different colors for tags and different line styles for layers
#         colors = plt.cm.tab20(np.linspace(0, 1, len(self.organized['tags'])))
#         line_styles = ['-', '--', '-.', ':'] * (len(self.organized['layers']) // 4 + 1)
#
#         plt.figure(figsize=(10, 5))
#
#         # Create legend handles
#         tag_handles = []
#         layer_handles = []
#
#         for tag_idx, tag_name in enumerate(self.organized['tags']):
#             color = colors[tag_idx]
#
#             # Track if we've added handles for this tag
#             tag_handle_added = False
#
#             for layer_idx, layer_key in enumerate(self.organized['layers']):
#                 line_style = line_styles[layer_idx]
#
#                 if tag_name in self.organized['data'] and layer_key in self.organized['data'][tag_name]:
#                     data_points = self.organized['data'][tag_name][layer_key]
#                     if data_points:
#                         steps, confidences = zip(*data_points)
#
#                         # Plot the line
#                         plt.plot(steps, confidences,
#                                  color=color,
#                                  linestyle=line_style,
#                                  linewidth=3,
#                                  alpha=0.7,
#                                  label=None)  # No automatic labels
#
#                         # Add to legend handles if not already added
#                         if not tag_handle_added:
#                             tag_handles.append((tag_name, color))
#                             tag_handle_added = True
#
#         # Add layer style handles
#         for layer_idx, layer_key in enumerate(self.organized['layers']):
#             layer_label = f"Layer {layer_key[0]} ({layer_key[1]})" if isinstance(layer_key,
#                                                                                  tuple) else f"Layer {layer_key}"
#             line_style = line_styles[layer_idx]
#             layer_handles.append((layer_label, line_style))
#
#         plt.title(f'{analysis_type.upper()} Analysis: All Tags Across All Layers\n'
#                   f'(Color = Tag, Line Style = Layer) - {model_name}', fontsize=14)
#         plt.xlabel('Training Step', fontsize=16)
#         plt.ylabel('Confidence Score', fontsize=16)
#         plt.grid(True, alpha=0.3)
#         plt.ylim(-0.001, 1.1)
#
#
#         # Tag legend (colors)
#         tag_legend_elements = [Line2D([0], [0], color=color, lw=3, label=tag)
#                                for tag, color in tag_handles]
#
#         # Layer legend (line styles)
#         layer_legend_elements = [Line2D([0], [0], color='black', lw=2, linestyle=style, label=layer)
#                                  for layer, style in layer_handles]
#
#         # Add legends
#         if tag_legend_elements:
#             tag_legend = plt.legend(handles=tag_legend_elements, title=f'{analysis_type.upper()} Tags',
#                                     loc='lower left', bbox_to_anchor=(1.05, .7))
#
#         if layer_legend_elements:
#             layer_legend = plt.legend(handles=layer_legend_elements, title='Layers',
#                                       loc='upper left', bbox_to_anchor=(1.05, .3))
#
#             # Add the first legend back if both exist
#             if tag_legend_elements:
#                 plt.gca().add_artist(tag_legend)
#
#         plt.tight_layout()
#
#         if analysis_type == 'pos':
#             save_dir = self.pos_plots_dir
#         elif analysis_type == 'semantic':
#             save_dir = self.semantic_plots_dir
#         else:
#             print(f"Unknown analysis type: {analysis_type}. Use 'pos' or 'semantic'.")
#             return
#         plt.savefig(
#                 os.path.join(save_dir, f"{model_name}_{analysis_type}_all_tags_all_layers_comprehensive.png"),
#                 dpi=300, bbox_inches='tight'
#         )
#
#         if show_plot:
#             plt.show()
#         plt.close()
#
#     def plot_probe_confidence_analysis(
#             self,
#             confidence_data: Dict[int, Dict[tuple, Dict[str, float]]],
#             model_name: str = '',
#             analysis_type: str = 'pos',
#             save_plot: bool = True,
#             show_plots: bool = False,
#     ) -> None:
#         """
#         Plot probe confidence analysis from training data.
#
#         Args:
#             confidence_data: Dictionary with structure:
#                 {step: {(layer_idx, layer_type): {'TAG_NAME': confidence, ...}, ...}, ...}
#             model_name: Name of the model for plot titles
#             analysis_type: Type of analysis ('pos' or 'semantic')
#             save_dir: Directory to save plots
#             save_plot: Whether to save the plots
#         """
#
#         if not confidence_data:
#             print(f"No {analysis_type} confidence data available for plotting")
#             return
#         self.confidence_data = confidence_data
#         if analysis_type == 'pos':
#             plot_dir = self.pos_plots_dir
#         elif analysis_type == 'semantic':
#             plot_dir = self.semantic_plots_dir
#         else:
#             print(f"Unknown analysis type: {analysis_type}. Use 'pos' or 'semantic'.")
#             return
#         # Extract and organize data
#         self._organize_confidence_data_fixed()
#
#         if not self.organized['steps'] or not self.organized['layers'] or not self.organized['tags']:
#             print("No valid data found for plotting")
#             return
#
#         print(f"Creating {analysis_type} confidence plots...")
#         print(
#             f"Found {len(self.organized['steps'])} steps, {len(self.organized['layers'])} layers, {len(self.organized['tags'])} tags")
#
#         # 1. Per tag plots - each tag across layers and steps
#         self._plot_per_tag_analysis_fixed(model_name, analysis_type, show_plots)
#
#         # 2. Per layer plots - all tags for each layer across steps
#         self._plot_per_layer_analysis_fixed(model_name, analysis_type, show_plots)
#
#         # 3. All tags, all layers combined plot
#         self._plot_all_tags_all_layers_fixed(model_name, analysis_type, show_plots)
#
#         if save_plot and plot_dir:
#             print(f"Plots saved to: {plot_dir}")

