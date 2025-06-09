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

        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Create output directories
        if log_dir:
            self.pos_plots_dir = os.path.join(log_dir, 'pos_analysis')
            self.semantic_plots_dir = os.path.join(log_dir, 'semantic_analysis')
            os.makedirs(self.pos_plots_dir, exist_ok=True)
            os.makedirs(self.semantic_plots_dir, exist_ok=True)

    def plot_learning_curves(
            self,
            accuracy_data: Dict[str, List[tuple]],
            sample_counts: Dict[str, int],
            model_name: str = '',
            analysis_type: str = 'pos',
            save_plot: bool = True
    ) -> None:
        """
        Generate learning curves by category.

        Args:
            accuracy_data: Dictionary mapping categories to (step, accuracy) tuples
            sample_counts: Sample counts per category
            model_name: Name of the model for plot titles
            analysis_type: Type of analysis ('pos' or 'semantic')
            save_plot: Whether to save the plot
        """
        if not accuracy_data:
            print(f"No {analysis_type} accuracy data available for plotting")
            return

        # Prepare the data
        categories = sorted(accuracy_data.keys())

        # Create accuracy plot
        plt.figure(figsize=(12, 8))
        print(f"Plotting {analysis_type} accuracy for {len(categories)} categories")
        print(f"{analysis_type.capitalize()} categories: {categories}")

        for category in categories:
            data_points = accuracy_data[category]
            if len(data_points) > 0:
                # Unzip the list of tuples into separate lists for steps and accuracies
                steps, accuracies = zip(*data_points)
                # Filter out NaN values for plotting
                valid_data = [(s, a) for s, a in zip(steps, accuracies) if not math.isnan(a)]
                if valid_data:
                    valid_steps, valid_accuracies = zip(*valid_data)
                    plt.plot(
                        valid_steps,
                        valid_accuracies,
                        marker='o',
                        linestyle='-',
                        label=f'{category} (n={sample_counts.get(category, 0)})',
                        alpha=0.8
                    )

        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Token Prediction Accuracy by {analysis_type.upper()} Category ({model_name})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        if save_plot and self.log_dir:
            plots_dir = getattr(self, f'{analysis_type}_plots_dir', self.log_dir)
            plt.savefig(
                os.path.join(plots_dir, f'{model_name}_{analysis_type}_accuracy.png'),
                dpi=300,
                bbox_inches='tight'
            )

        plt.close()

    def plot_heatmap(
            self,
            accuracy_data: Dict[str, List[tuple]],
            model_name: str = '',
            analysis_type: str = 'pos',
            save_plot: bool = True
    ) -> None:
        """
        Generate a heatmap of accuracies across training steps.

        Args:
            accuracy_data: Dictionary mapping categories to (step, accuracy) tuples
            model_name: Name of the model for plot titles
            analysis_type: Type of analysis ('pos' or 'semantic')
            save_plot: Whether to save the plot
        """
        if not accuracy_data:
            print(f"No {analysis_type} accuracy data available for heatmap")
            return

        # Get all unique steps across all categories
        all_steps = set()
        for category, data_points in accuracy_data.items():
            steps, _ = zip(*data_points) if data_points else ([], [])
            all_steps.update(steps)

        all_steps = sorted(all_steps)

        # Prepare data for heatmap
        heatmap_data = {}
        categories = sorted(accuracy_data.keys())

        for category in categories:
            heatmap_data[category] = {}
            # Initialize with NaN for all steps
            for step in all_steps:
                heatmap_data[category][step] = float('nan')

            # Fill in actual values
            for step, acc in accuracy_data[category]:
                heatmap_data[category][step] = acc

        # Convert to DataFrame
        df = pd.DataFrame(heatmap_data, index=all_steps)

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            df,
            cmap='viridis',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Accuracy'},
            annot=True,
            fmt='.2f',
            mask=df.isna()
        )

        plt.xlabel(f'{analysis_type.upper()} Category', fontsize=12)
        plt.ylabel('Training Step', fontsize=12)
        plt.title(f'Token Prediction Accuracy Evolution by {analysis_type.upper()} ({model_name})', fontsize=14)
        plt.tight_layout()

        if save_plot and self.log_dir:
            plots_dir = getattr(self, f'{analysis_type}_plots_dir', self.log_dir)
            plt.savefig(
                os.path.join(plots_dir, f'{model_name}_{analysis_type}_heatmap.png'),
                dpi=300,
                bbox_inches='tight'
            )

        plt.close()

    def plot_probe_comparison(
            self,
            probe_results: Dict[Union[int, str, tuple], Dict],
            model_name: str = '',
            analysis_type: str = 'pos'
    ) -> None:
        """
        Plot comparison of probe performance across layers.

        Args:
            probe_results: Results from multiple layer probes
            model_name: Name of the model
            analysis_type: Type of analysis ('pos' or 'semantic')
        """
        if not probe_results:
            print(f"No {analysis_type} probe results available for comparison")
            return

        # Convert complex keys to simple string representations for plotting
        layer_labels = []
        overall_accuracies = []

        for layer_key, results in probe_results.items():
            # Convert complex keys to readable string labels
            if isinstance(layer_key, tuple):
                # Handle tuple keys like (layer_index, layer_type)
                if len(layer_key) == 2:
                    layer_idx, layer_type = layer_key
                    layer_label = f"{layer_type}_{layer_idx}"
                else:
                    layer_label = str(layer_key)
            elif isinstance(layer_key, (int, str)):
                layer_label = str(layer_key)
            else:
                # For any other complex types, convert to string
                layer_label = str(layer_key)

            layer_labels.append(layer_label)

            # Extract overall accuracy
            if 'overall_accuracy' in results:
                overall_accuracies.append(results['overall_accuracy'])
            else:
                overall_accuracies.append(0.0)

        # Sort by layer index for consistent ordering
        try:
            # Try to sort numerically if possible
            sorted_data = sorted(zip(layer_labels, overall_accuracies),
                                 key=lambda x: self._extract_layer_number(x[0]))
            layer_labels, overall_accuracies = zip(*sorted_data)
        except:
            # If sorting fails, keep original order
            pass

        # Plot layer comparison
        plt.figure(figsize=(10, 6))
        x_positions = range(len(layer_labels))
        plt.plot(x_positions, overall_accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel('Overall Accuracy', fontsize=12)
        plt.title(f'{analysis_type.upper()} Probe Accuracy by Layer ({model_name})', fontsize=14)
        plt.xticks(x_positions, layer_labels, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if self.log_dir:
            plots_dir = getattr(self, f'{analysis_type}_plots_dir', self.log_dir)
            plt.savefig(
                os.path.join(plots_dir, f'{model_name}_{analysis_type}_layer_comparison.png'),
                dpi=300,
                bbox_inches='tight'
            )

        plt.close()

    def _extract_layer_number(self, layer_label: str) -> int:
        """
        Extract numeric layer index from layer label for sorting.

        Args:
            layer_label: String representation of layer

        Returns:
            Numeric layer index for sorting
        """
        import re

        # Look for numbers in the layer label
        numbers = re.findall(r'\d+', layer_label)
        if numbers:
            return int(numbers[0])  # Return first number found
        else:
            return 0  # Default to 0 if no number found

    # def plot_category_distribution(
    #         self,
    #         sample_counts: Dict[str, int],
    #         model_name: str = '',
    #         analysis_type: str = 'pos'
    # ) -> None:
    #     """
    #     Plot distribution of categories in the dataset.
    #
    #     Args:
    #         sample_counts: Count of samples per category
    #         model_name: Name of the model
    #         analysis_type: Type of analysis ('pos' or 'semantic')
    #     """
    #     if not sample_counts:
    #         print(f"No {analysis_type} sample counts available")
    #         return
    #
    #     categories = list(sample_counts.keys())
    #     counts = list(sample_counts.values())
    #
    #     plt.figure(figsize=(12, 6))
    #     bars = plt.bar(categories, counts)
    #     plt.xlabel(f'{analysis_type.upper()} Category', fontsize=12)
    #     plt.ylabel('Sample Count', fontsize=12)
    #     plt.title(f'{analysis_type.upper()} Category Distribution ({model_name})', fontsize=14)
    #     plt.xticks(rotation=45, ha='right')
    #
    #     # Add value labels on bars
    #     for bar, count in zip(bars, counts):
    #         plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
    #                  str(count), ha='center', va='bottom')
    #
    #     plt.tight_layout()
    #
    #     if self.log_dir:
    #         plots_dir = getattr(self, f'{analysis_type}_plots_dir', self.log_dir)
    #         plt.savefig(
    #             os.path.join(plots_dir, f'{model_name}_{analysis_type}_distribution.png'),
    #             dpi=300,
    #             bbox_inches='tight'
    #         )
    #
    #     plt.close()

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

    def plot_probe_predictions_enhanced(
            self,
            probe_predictions: Dict[str, Dict[int, List[tuple]]],
            label_names: Optional[List[str]] = None,
            model_name: str = '',
            analysis_type: str = 'pos',
            save_plot: bool = True
    ) -> None:
        """
        Enhanced probe predictions visualization with multiple plot types.

        Args:
            probe_predictions: Dictionary with layer_name -> {tag_idx -> [(step, score), ...]}
            label_names: List of label names for tags
            model_name: Name of the model
            analysis_type: Type of analysis ('pos' or 'semantic')
            save_plot: Whether to save the plots
        """
        if not probe_predictions:
            print(f"No {analysis_type} probe predictions available for plotting")
            return

        if not self.log_dir:
            print("No log directory specified, skipping save")
            return

        plots_dir = getattr(self, f'{analysis_type}_plots_dir', self.log_dir)
        os.makedirs(plots_dir, exist_ok=True)

        # Determine number of classes from the data
        all_tag_indices = set()
        for layer_data in probe_predictions.values():
            all_tag_indices.update(layer_data.keys())
        number_classes = len(all_tag_indices)

        if label_names is None:
            if analysis_type == 'pos':
                label_names = [f"POS {i}" for i in range(number_classes)]
            else:
                label_names = [f"SEM {i}" for i in range(number_classes)]

        # Per tag, per layer plots
        for layer_name, tag_data in probe_predictions.items():
            # Convert layer_name to string if it's a complex type
            layer_str = str(layer_name) if not isinstance(layer_name, str) else layer_name

            for tag_idx, history in tag_data.items():
                if not history:  # Skip empty histories
                    continue

                steps, scores = zip(*history)
                plt.figure(figsize=(8, 3))
                plt.plot(steps, scores, label=f"{label_names[tag_idx]}")
                plt.title(f"Probe Confidence Over Time - {layer_str} - {analysis_type.upper()}: {label_names[tag_idx]}")
                plt.xlabel("Step")
                plt.ylabel("Confidence")
                plt.legend(loc="lower left")
                plt.tight_layout()

                if save_plot:
                    name_layer_idx = str(layer_str) if isinstance(layer_str, int) else f"{layer_str[0]}_{layer_str[1]}"
                    name_layer_idx = name_layer_idx.replace("(", "").replace(")", "")
                    name_layer_idx = name_layer_idx.replace(",", "_")
                    name_layer_idx = name_layer_idx.replace(" ", "")
                    name_layer_idx = name_layer_idx.replace("'", "")
                    plt.savefig(
                        os.path.join(plots_dir, f"{model_name}_{name_layer_idx}_tag{tag_idx}_{label_names[tag_idx]}.png"),
                        dpi=300, bbox_inches='tight'
                    )
                plt.show()
                plt.close()

        # Combined plot: all tags in one graph per layer
        for layer_name, tag_data in probe_predictions.items():
            layer_str = str(layer_name) if not isinstance(layer_name, str) else layer_name

            plt.figure(figsize=(10, 5))
            for tag_idx, history in tag_data.items():
                if not history:  # Skip empty histories
                    continue
                steps, scores = zip(*history)
                plt.plot(steps, scores, label=label_names[tag_idx])

            plt.title(f"All {analysis_type.upper()} Tags - Confidence Over Time - {layer_str}")
            plt.xlabel("Step")
            plt.ylabel("Confidence")
            plt.legend(loc="lower left")
            plt.tight_layout()

            if save_plot:
                name_layer_idx = str(layer_str) if isinstance(layer_str, int) else f"{layer_str[0]}_{layer_str[1]}"
                name_layer_idx = name_layer_idx.replace("(", "").replace(")", "")
                name_layer_idx = name_layer_idx.replace(",", "_")
                name_layer_idx = name_layer_idx.replace(" ", "")
                name_layer_idx = name_layer_idx.replace("'", "")
                plt.savefig(
                    os.path.join(plots_dir, f"{model_name}_{name_layer_idx}_all_tags_combined.png"),
                    dpi=300, bbox_inches='tight'
                )
            plt.show()
            plt.close()

        # One plot: each tag, each line = layer
        num_tags = len(label_names)
        for tag_idx in range(num_tags):
            plt.figure(figsize=(10, 5))
            for layer_name, tag_data in probe_predictions.items():
                layer_str = str(layer_name) if not isinstance(layer_name, str) else layer_name

                if tag_idx in tag_data and tag_data[tag_idx]:  # Check if history exists and is not empty
                    steps, scores = zip(*tag_data[tag_idx])
                    plt.plot(steps, scores, label=f"{layer_str}")

            plt.title(f"{analysis_type.upper()} Tag: {label_names[tag_idx]} - Confidence Across Layers")
            plt.xlabel("Step")
            plt.ylabel("Confidence")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.show()
            if save_plot:
                plt.savefig(
                    os.path.join(plots_dir, f"{model_name}_all_layers_tag{tag_idx}_{label_names[tag_idx]}.png"),
                    dpi=300, bbox_inches='tight'
                )
            plt.close()

    def plot_all_tags_all_layers(
            self,
            probe_predictions: Dict[str, Dict[int, List[tuple]]],
            label_names: Optional[List[str]] = None,
            model_name: str = '',
            analysis_type: str = 'pos',
            save_plot: bool = True
    ) -> None:
        """
        Create a comprehensive plot showing all tags across all layers.

        Args:
            probe_predictions: Dictionary with layer_name -> {tag_idx -> [(step, score), ...]}
            label_names: List of label names for tags
            model_name: Name of the model
            analysis_type: Type of analysis ('pos' or 'semantic')
            save_plot: Whether to save the plot
        """
        if not probe_predictions:
            print(f"No {analysis_type} probe predictions available for plotting")
            return

        # Determine number of classes from the data
        all_tag_indices = set()
        for layer_data in probe_predictions.values():
            all_tag_indices.update(layer_data.keys())
        number_of_classes = len(all_tag_indices)

        if label_names is None:
            if analysis_type == 'pos':
                label_names = [f"POS {i}" for i in range(number_of_classes)]
            else:
                label_names = [f"SEM {i}" for i in range(number_of_classes)]

        plt.style.use('seaborn-v0_8-whitegrid')

        if self.log_dir:
            plots_dir = getattr(self, f'{analysis_type}_plots_dir', self.log_dir)
            os.makedirs(plots_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 7))

        styles = ["-", "--", "-.", ":"]
        color_map = plt.get_cmap("tab10")

        # Convert layer names to strings and sort them
        layer_names = [str(k) for k in probe_predictions.keys()]
        layer_names = sorted(layer_names, key=lambda x: self._extract_layer_number(x))

        # For legends
        pos_handles = {}
        layer_handles = {}

        for tag_idx, pos_name in enumerate(label_names):
            if tag_idx >= len(label_names):
                continue

            color = color_map(tag_idx % 10)

            for i, layer_name in enumerate(layer_names):
                # Find the original layer key that matches this string representation
                original_layer_key = None
                for orig_key in probe_predictions.keys():
                    if str(orig_key) == layer_name:
                        original_layer_key = orig_key
                        break

                if original_layer_key is None:
                    continue

                if tag_idx not in probe_predictions[original_layer_key]:
                    continue

                history = probe_predictions[original_layer_key][tag_idx]
                if not history:  # Skip empty histories
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
        ax.set_title(f"All {analysis_type.upper()} Tags Across All Layers (Color = Tag, Line Style = Layer)",
                     fontsize=14)
        ax.grid(True, linestyle="-", alpha=0.3)

        # Create legends only if we have data
        if pos_handles:
            legend1 = ax.legend(pos_handles.values(), pos_handles.keys(),
                                title=f"{analysis_type.upper()} Tags", loc="lower left")

        if layer_handles:
            legend2 = ax.legend(layer_handles.values(), layer_handles.keys(),
                                title="Layer Styles", loc="lower right")

            # Add the first legend back if both exist
            if pos_handles:
                ax.add_artist(legend1)

        plt.tight_layout()

        if save_plot and self.log_dir:
            plt.savefig(
                os.path.join(plots_dir, f"{model_name}_all_tags_all_layers.png"),
                dpi=300, bbox_inches='tight'
            )
        plt.close()
    # @staticmethod
    # def plot_probe_predictions_enhanced(probe_predictions, label_names=None, model_name=None, number_classes=8,
    #                                     save_dir="../plots/probe_analysis/probe_monitor_plots"):
    #
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     if label_names is None:
    #         label_names = [f"POS {i}" for i in range(number_classes)]
    #
    #     # Per tag, per layer plots
    #     for layer_name, tag_data in probe_predictions.items():
    #         for tag_idx, history in tag_data.items():
    #             steps, scores = zip(*history)
    #             plt.figure(figsize=(8, 3))
    #             plt.plot(steps, scores, label=f"{label_names[tag_idx]}")
    #             plt.title(f"Probe Confidence Over Time - {layer_name} - POS: {label_names[tag_idx]}")
    #             plt.xlabel("Step")
    #             plt.ylabel("Confidence")
    #             plt.legend(loc="lower left")
    #             plt.tight_layout()
    #             plt.savefig(
    #                 os.path.join(save_dir, f"{model_name}_{layer_name}_tag{tag_idx}_{label_names[tag_idx]}.png"),
    #                 dpi=300)
    #             plt.close()
    #
    #     # Combined plot: all tags in one graph per layer
    #     for layer_name, tag_data in probe_predictions.items():
    #         plt.figure(figsize=(10, 5))
    #         for tag_idx, history in tag_data.items():
    #             steps, scores = zip(*history)
    #             plt.plot(steps, scores, label=label_names[tag_idx])
    #         plt.title(f"All POS Tags - Confidence Over Time - {layer_name}")
    #         plt.xlabel("Step")
    #         plt.ylabel("Confidence")
    #         plt.legend(loc="lower left")
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(save_dir, f"{model_name}_{layer_name}_all_tags_combined.png"), dpi=300)
    #         plt.close()
    #
    #     # One plot: each tag, each line = layer
    #     num_tags = len(label_names)
    #     for tag_idx in range(num_tags):
    #         plt.figure(figsize=(10, 5))
    #         for layer_name, tag_data in probe_predictions.items():
    #             if tag_idx in tag_data:
    #                 steps, scores = zip(*tag_data[tag_idx])
    #                 plt.plot(steps, scores, label=f"{layer_name}")
    #         plt.title(f"POS Tag: {label_names[tag_idx]} - Confidence Across Layers")
    #         plt.xlabel("Step")
    #         plt.ylabel("Confidence")
    #         plt.legend(loc="lower left")
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(save_dir, f"{model_name}_all_layers_tag{tag_idx}_{label_names[tag_idx]}.png"),
    #                     dpi=300)
    #         plt.close()
    #
    # @staticmethod
    # def plot_all_tags_all_layers(probe_predictions, label_names=None, model_name=None, number_of_classes=8,
    #                              save_path="../plots/probe_analysis/probe_monitor_plots"):
    #
    #     if label_names is None:
    #         label_names = [f"POS {i}" for i in range(number_of_classes)]
    #
    #     plt.style.use('seaborn-v0_8-whitegrid')
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #
    #     fig, ax = plt.subplots(figsize=(14, 7))
    #
    #     styles = ["-", "--", "-.", ":"]
    #     color_map = plt.get_cmap("tab10")
    #     layer_names = sorted(probe_predictions.keys())
    #
    #     # For POS legend (colors only)
    #     pos_handles = {}
    #     # For Layer legend (line styles only)
    #     layer_handles = {}
    #
    #     for tag_idx, pos_name in enumerate(label_names):
    #         color = color_map(tag_idx % 10)
    #         for i, layer_name in enumerate(layer_names):
    #             if tag_idx not in probe_predictions[layer_name]:
    #                 continue
    #             steps, scores = zip(*probe_predictions[layer_name][tag_idx])
    #             linestyle = styles[i % len(styles)]
    #
    #             line = ax.plot(
    #                 steps,
    #                 scores,
    #                 color=color,
    #                 linestyle=linestyle,
    #                 linewidth=2,
    #                 label=None  # suppress default label
    #             )
    #
    #             if pos_name not in pos_handles:
    #                 pos_handles[pos_name] = Line2D([0], [0], color=color, lw=3)
    #
    #             if layer_name not in layer_handles:
    #                 layer_handles[layer_name] = Line2D([0], [0], color="black", lw=2, linestyle=linestyle)
    #
    #     ax.set_xlabel("Training Step", fontsize=12)
    #     ax.set_ylabel("Confidence", fontsize=12)
    #     ax.set_title("All POS Tags Across All Layers (Color = POS, Line Style = Layer)", fontsize=14)
    #     ax.grid(True, linestyle="-", alpha=0.3)
    #
    #     # POS tag legend
    #     legend1 = ax.legend(pos_handles.values(), pos_handles.keys(), title="POS Tags", loc="lower left")
    #
    #     # Layer line style legend
    #     legend2 = ax.legend(layer_handles.values(), layer_handles.keys(), title="Layer Styles", loc="lower right")
    #
    #     ax.add_artist(legend1)  # Add the first legend manually
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(save_path, f"{model_name}_all_tags_all_layers.png"), dpi=300)
    #     plt.close()


# old compatibility functions
# def plot_pos_learning_curves_legacy(tracker, model_name: str = ''):
#     """Legacy function for plotting POS learning curves."""
#     visualizer = ProbesVisualizer(tracker.log_dir)
#     visualizer.plot_learning_curves(
#         tracker.accuracy,
#         tracker.sample_counts,
#         model_name,
#         'pos'
#     )
#
#
# def plot_semantic_learning_curves_legacy(tracker, model_name: str = ''):
#     """Legacy function for plotting semantic learning curves."""
#     visualizer = ProbesVisualizer(tracker.log_dir)
#     visualizer.plot_learning_curves(
#         tracker.accuracy,
#         tracker.sample_counts,
#         model_name,
#         'semantic'
#     )