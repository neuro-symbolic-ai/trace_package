import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict

from matplotlib.lines import Line2D

from .config import LinguisticProbesConfig


class ProbesVisualizer:
    """
    Visualizer for probe analysis results.

    This class handles all visualization tasks for both POS and semantic
    role analysis, following the reference style pattern.
    """

    def __init__(self, log_dir: Optional[str] = None, config: Optional[LinguisticProbesConfig] = None):
        """
        Initialize the visualizer.

        Args:
            log_dir: Directory to save visualizations
            config: Configuration object
        """
        self.log_dir = log_dir
        self.config = config or LinguisticProbesConfig.default()
        self.confidence_data = {}  # Will hold the confidence data for plotting

        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Create output directories
        if log_dir:
            self.pos_plots_dir = os.path.join(log_dir, 'pos_probe_analysis')
            self.semantic_plots_dir = os.path.join(log_dir, 'semantic_probe_analysis')
            os.makedirs(self.pos_plots_dir, exist_ok=True)
            os.makedirs(self.semantic_plots_dir, exist_ok=True)

    def save_metrics(
            self,
            accuracy_data: Dict[str, List[tuple]],
            sample_counts: Dict[str, int],
            model_name: str = '',
            analysis_type: str = 'pos'
    ) -> None:
        """
        Save metrics to CSV files.

        Args:
            accuracy_data: Accuracy data by category
            sample_counts: Sample counts per category
            model_name: Name of the model
            analysis_type: Type of analysis ('pos' or 'semantic')
        """
        if not self.log_dir:
            return

        plots_dir = getattr(self, f'{analysis_type}_plots_dir', self.log_dir)

        # Create dataframes for each category
        for category, data_points in accuracy_data.items():
            if data_points:
                # Convert list of tuples to dataframe
                steps, accuracies = zip(*data_points)
                category_df = pd.DataFrame({
                    'step': steps,
                    'accuracy': accuracies
                })
                category_df.to_csv(
                    os.path.join(plots_dir, f'{model_name}_{category}_accuracy.csv'),
                    index=False
                )

        # Save combined file with all categories
        all_data = []
        for category, data_points in accuracy_data.items():
            for step, acc in data_points:
                all_data.append({
                    'step': step,
                    'category': category,
                    'accuracy': acc
                })

        if all_data:
            combined_df = pd.DataFrame(all_data)

            # Handle potential duplicate step values by taking mean
            grouped_df = combined_df.groupby(['step', 'category']).mean().reset_index()

            # Pivot without duplicates
            pivot_df = grouped_df.pivot(index='step', columns='category', values='accuracy')
            pivot_df.to_csv(os.path.join(plots_dir, f'{model_name}_all_{analysis_type}_accuracy.csv'))

        # Save sample counts
        counts_df = pd.DataFrame({
            f'{analysis_type}_category': list(sample_counts.keys()),
            'count': list(sample_counts.values())
        })
        counts_df.to_csv(
            os.path.join(plots_dir, f'{model_name}_{analysis_type}_counts.csv'),
            index=False
        )


    def _organize_confidence_data_fixed(self) -> None:
        """
        Organize the confidence data for easier plotting (FIXED VERSION).

        Returns:
            Dictionary with organized data structure for plotting
        """
        # Initialize organized structure with proper defaultdict nesting
        self.organized = {
            'steps': [],
            'layers': set(),
            'tags': set(),
            'data': {}  # Will be: data[tag][layer] = [(step, confidence), ...]
        }

        # Extract unique steps, layers, and tags
        for step, step_data in self.confidence_data.items():
            self.organized['steps'].append(step)

            for layer_key, tag_confidences in step_data.items():
                self.organized['layers'].add(layer_key)

                for tag_name, confidence in tag_confidences.items():
                    self.organized['tags'].add(tag_name)

                    # Initialize nested structure if needed
                    if tag_name not in self.organized['data']:
                        self.organized['data'][tag_name] = {}
                    if layer_key not in self.organized['data'][tag_name]:
                        self.organized['data'][tag_name][layer_key] = []

                    # Add the data point
                    self.organized['data'][tag_name][layer_key].append((step, confidence))

        # Sort steps and convert sets to sorted lists
        self.organized['steps'] = sorted(self.organized['steps'])
        self.organized['layers'] = sorted(list(self.organized['layers']))
        self.organized['tags'] = sorted(list(self.organized['tags']))

        # Sort data points by step for each tag-layer combination
        for tag_name in self.organized['data']:
            for layer_key in self.organized['data'][tag_name]:
                self.organized['data'][tag_name][layer_key].sort(key=lambda x: x[0])

    def _plot_per_tag_analysis_fixed(self, model_name: str, analysis_type: str, show_plot: bool = False) -> None:
        """
        Create plots showing each tag's confidence across layers and steps.
        """
        print("Creating per-tag analysis plots...")

        for tag_name in self.organized['tags']:
            plt.figure(figsize=(12, 8))

            # Plot each layer as a separate line
            for layer_key in self.organized['layers']:
                if tag_name in self.organized['data'] and layer_key in self.organized['data'][tag_name]:
                    data_points = self.organized['data'][tag_name][layer_key]
                    if data_points:
                        steps, confidences = zip(*data_points)
                        layer_label = f"Layer {layer_key[0]} ({layer_key[1]})" if isinstance(layer_key,
                                                                                             tuple) else f"Layer {layer_key}"
                        plt.plot(steps, confidences, marker='o', linewidth=2, label=layer_label, alpha=0.8)

            plt.title(f'{analysis_type.upper()} Tag: {tag_name} - Confidence Across Layers ({model_name})', fontsize=14)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Confidence Score', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()

            if analysis_type == 'pos':
                save_dir = self.pos_plots_dir
            elif analysis_type == 'semantic':
                save_dir = self.semantic_plots_dir
            else:
                print(f"Unknown analysis type: {analysis_type}. Use 'pos' or 'semantic'.")
                return

            safe_tag_name = tag_name.replace('/', '_').replace(' ', '_')
            plt.savefig(
                    os.path.join(save_dir, f"{model_name}_{analysis_type}_tag_{safe_tag_name}_across_layers.png"),
                    dpi=300, bbox_inches='tight'
            )
            if show_plot:
                plt.show()
            plt.close()

    def _plot_per_layer_analysis_fixed(self, model_name: str, analysis_type: str, show_plot: bool = False) -> None:
        """
        Create plots showing all tags for each layer across steps.
        """
        print("Creating per-layer analysis plots...")

        for layer_key in self.organized['layers']:
            plt.figure(figsize=(12, 8))

            # Plot each tag as a separate line for this layer
            for tag_name in self.organized['tags']:
                if tag_name in self.organized['data'] and layer_key in self.organized['data'][tag_name]:
                    data_points = self.organized['data'][tag_name][layer_key]
                    if data_points:
                        steps, confidences = zip(*data_points)
                        plt.plot(steps, confidences, marker='o', linewidth=2, label=tag_name, alpha=0.8)

            layer_label = f"Layer {layer_key[0]} ({layer_key[1]})" if isinstance(layer_key,
                                                                                 tuple) else f"Layer {layer_key}"
            plt.title(f'{analysis_type.upper()} Analysis: All Tags - {layer_label} ({model_name})', fontsize=14)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Confidence Score', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()

            if analysis_type == 'pos':
                save_dir = self.pos_plots_dir
            elif analysis_type == 'semantic':
                save_dir = self.semantic_plots_dir
            else:
                print(f"Unknown analysis type: {analysis_type}. Use 'pos' or 'semantic'.")
                return

            safe_layer_name = str(layer_key).replace('(', '').replace(')', '').replace(',', '_').replace("'",
                                                                                                             '').replace(
                    ' ', '')
            plt.savefig(
                    os.path.join(save_dir, f"{model_name}_{analysis_type}_layer_{safe_layer_name}_all_tags.png"),
                    dpi=300, bbox_inches='tight'
            )

            if show_plot:
                plt.show()
            plt.close()

    def _plot_all_tags_all_layers_fixed(self, model_name: str, analysis_type: str, show_plot: bool = False) -> None:
        """
        Create a comprehensive plot showing all tags across all layers.
        """
        print("Creating comprehensive all-tags-all-layers plot...")

        # Use different colors for tags and different line styles for layers
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.organized['tags'])))
        line_styles = ['-', '--', '-.', ':'] * (len(self.organized['layers']) // 4 + 1)

        plt.figure(figsize=(16, 10))

        # Create legend handles
        tag_handles = []
        layer_handles = []

        for tag_idx, tag_name in enumerate(self.organized['tags']):
            color = colors[tag_idx]

            # Track if we've added handles for this tag
            tag_handle_added = False

            for layer_idx, layer_key in enumerate(self.organized['layers']):
                line_style = line_styles[layer_idx]

                if tag_name in self.organized['data'] and layer_key in self.organized['data'][tag_name]:
                    data_points = self.organized['data'][tag_name][layer_key]
                    if data_points:
                        steps, confidences = zip(*data_points)

                        # Plot the line
                        plt.plot(steps, confidences,
                                 color=color,
                                 linestyle=line_style,
                                 linewidth=2,
                                 alpha=0.7,
                                 label=None)  # No automatic labels

                        # Add to legend handles if not already added
                        if not tag_handle_added:
                            tag_handles.append((tag_name, color))
                            tag_handle_added = True

        # Add layer style handles
        for layer_idx, layer_key in enumerate(self.organized['layers']):
            layer_label = f"Layer {layer_key[0]} ({layer_key[1]})" if isinstance(layer_key,
                                                                                 tuple) else f"Layer {layer_key}"
            line_style = line_styles[layer_idx]
            layer_handles.append((layer_label, line_style))

        plt.title(f'{analysis_type.upper()} Analysis: All Tags Across All Layers\n'
                  f'(Color = Tag, Line Style = Layer) - {model_name}', fontsize=14)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Confidence Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        # Create custom legends
        from matplotlib.lines import Line2D

        # Tag legend (colors)
        tag_legend_elements = [Line2D([0], [0], color=color, lw=3, label=tag)
                               for tag, color in tag_handles]

        # Layer legend (line styles)
        layer_legend_elements = [Line2D([0], [0], color='black', lw=2, linestyle=style, label=layer)
                                 for layer, style in layer_handles]

        # Add legends
        if tag_legend_elements:
            tag_legend = plt.legend(handles=tag_legend_elements, title=f'{analysis_type.upper()} Tags',
                                    loc='upper left', bbox_to_anchor=(1.02, 1))

        if layer_legend_elements:
            layer_legend = plt.legend(handles=layer_legend_elements, title='Layers',
                                      loc='upper left', bbox_to_anchor=(1.02, 0.6))

            # Add the first legend back if both exist
            if tag_legend_elements:
                plt.gca().add_artist(tag_legend)

        plt.tight_layout()

        if analysis_type == 'pos':
            save_dir = self.pos_plots_dir
        elif analysis_type == 'semantic':
            save_dir = self.semantic_plots_dir
        else:
            print(f"Unknown analysis type: {analysis_type}. Use 'pos' or 'semantic'.")
            return
        plt.savefig(
                os.path.join(save_dir, f"{model_name}_{analysis_type}_all_tags_all_layers_comprehensive.png"),
                dpi=300, bbox_inches='tight'
        )

        if show_plot:
            plt.show()
        plt.close()

    def plot_probe_confidence_analysis(
            self,
            confidence_data: Dict[int, Dict[tuple, Dict[str, float]]],
            model_name: str = '',
            analysis_type: str = 'pos',
            save_plot: bool = True
    ) -> None:
        """
        Plot probe confidence analysis from training data.

        Args:
            confidence_data: Dictionary with structure:
                {step: {(layer_idx, layer_type): {'TAG_NAME': confidence, ...}, ...}, ...}
            model_name: Name of the model for plot titles
            analysis_type: Type of analysis ('pos' or 'semantic')
            save_dir: Directory to save plots
            save_plot: Whether to save the plots
        """

        if not confidence_data:
            print(f"No {analysis_type} confidence data available for plotting")
            return
        self.confidence_data = confidence_data
        if analysis_type == 'pos':
            plot_dir = self.pos_plots_dir
        elif analysis_type == 'semantic':
            plot_dir = self.semantic_plots_dir
        else:
            print(f"Unknown analysis type: {analysis_type}. Use 'pos' or 'semantic'.")
            return
        # Extract and organize data
        self._organize_confidence_data_fixed()

        if not self.organized['steps'] or not self.organized['layers'] or not self.organized['tags']:
            print("No valid data found for plotting")
            return

        print(f"Creating {analysis_type} confidence plots...")
        print(
            f"Found {len(self.organized['steps'])} steps, {len(self.organized['layers'])} layers, {len(self.organized['tags'])} tags")

        # 1. Per tag plots - each tag across layers and steps
        self._plot_per_tag_analysis_fixed(model_name, analysis_type)

        # 2. Per layer plots - all tags for each layer across steps
        self._plot_per_layer_analysis_fixed(model_name, analysis_type)

        # 3. All tags, all layers combined plot
        self._plot_all_tags_all_layers_fixed(model_name, analysis_type)

        if save_plot and plot_dir:
            print(f"Plots saved to: {plot_dir}")

