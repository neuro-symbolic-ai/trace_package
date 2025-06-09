# # visualization.py
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# from typing import Dict, List, Optional, Union, Tuple
# import os
#
#
# class IntrinsicDimensionVisualizer:
#     """
#     Visualizer for intrinsic dimension analysis results.
#
#     This class handles all visualization tasks for intrinsic dimension analysis,
#     providing various plots and charts to understand ID patterns across layers.
#     """
#
#     def __init__(self, save_dir: Optional[str] = None, style: str = 'seaborn-v0_8-whitegrid'):
#         """
#         Initialize the visualizer.
#
#         Args:
#             save_dir: Directory to save plots
#             style: Matplotlib style to use
#         """
#         self.save_dir = save_dir
#         if save_dir:
#             os.makedirs(save_dir, exist_ok=True)
#
#         plt.style.use(style)
#
#
#     def plot_layer_progression(
#             self,
#             progression_results: Dict[str, List[float]],
#             model_name: str = "",
#             save_plot: bool = True,
#             figsize: Tuple[int, int] = (12, 6)
#     ) -> None:
#         """
#         Plot intrinsic dimension progression across layers.
#
#         Args:
#             progression_results: Results from analyze_layer_progression
#             model_name: Name of the model for plot title
#             save_plot: Whether to save the plot
#             figsize: Figure size
#         """
#         plt.figure(figsize=figsize)
#
#         for i, (layer_type, id_values) in enumerate(progression_results.items()):
#             layer_indices = list(range(len(id_values)))
#
#             # Filter out NaN values for plotting
#             valid_data = [(idx, val) for idx, val in zip(layer_indices, id_values) if not np.isnan(val)]
#             if valid_data:
#                 valid_indices, valid_values = zip(*valid_data)
#                 plt.plot(
#                     valid_indices,
#                     valid_values,
#                     marker='o',
#                     linewidth=2.5,
#                     markersize=8,
#                     label=f'{layer_type.capitalize()} Stack',
#                     color=self.colors[i % len(self.colors)],
#                     alpha=0.8
#                 )
#
#         plt.xlabel('Layer Index', fontsize=12)
#         plt.ylabel('Intrinsic Dimension', fontsize=12)
#         plt.title(f'Intrinsic Dimension Progression Across Layers ({model_name})', fontsize=14)
#         plt.grid(True, alpha=0.3)
#         plt.legend(fontsize=11)
#
#         # Add average line if multiple layer types
#         if len(progression_results) > 1:
#             all_values = []
#             for id_values in progression_results.values():
#                 all_values.extend([v for v in id_values if not np.isnan(v)])
#             if all_values:
#                 avg_id = np.mean(all_values)
#                 plt.axhline(y=avg_id, color='red', linestyle='--', alpha=0.7,
#                             label=f'Overall Average: {avg_id:.2f}')
#                 plt.legend(fontsize=11)
#
#         plt.tight_layout()
#
#         if save_plot and self.save_dir:
#             plt.savefig(
#                 os.path.join(self.save_dir, f'{model_name}_id_progression.png'),
#                 dpi=300, bbox_inches='tight'
#             )
#         plt.show()
#         plt.close()
#
#     def plot_id_distribution(
#             self,
#             results: Dict[str, Union[float, List]],
#             model_name: str = "",
#             save_plot: bool = True,
#             figsize: Tuple[int, int] = (10, 6)
#     ) -> None:
#         """
#         Plot distribution of intrinsic dimensions.
#
#         Args:
#             results: Results from analyzer.analyze()
#             model_name: Name of the model
#             save_plot: Whether to save the plot
#             figsize: Figure size
#         """
#         intrinsic_dims = results.get("intrinsic_dimensions", {})
#
#         # Extract ID values and layer info
#         id_values = []
#         layer_types = []
#         layer_indices = []
#
#         for layer_name, id_value in intrinsic_dims.items():
#             if not np.isnan(id_value):
#                 id_values.append(id_value)
#                 # Parse layer name: "layer_X_type"
#                 parts = layer_name.split('_')
#                 if len(parts) >= 3:
#                     layer_indices.append(int(parts[1]))
#                     layer_types.append(parts[2])
#                 else:
#                     layer_indices.append(0)
#                     layer_types.append('unknown')
#
#         if not id_values:
#             print("No valid intrinsic dimension values to plot")
#             return
#
#         # Create subplots
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
#
#         # Histogram
#         ax1.hist(id_values, bins=min(15, len(id_values)), alpha=0.7, color='skyblue', edgecolor='black')
#         ax1.set_xlabel('Intrinsic Dimension', fontsize=12)
#         ax1.set_ylabel('Frequency', fontsize=12)
#         ax1.set_title(f'ID Distribution ({model_name})', fontsize=13)
#         ax1.grid(True, alpha=0.3)
#
#         # Add statistics
#         mean_id = np.mean(id_values)
#         std_id = np.std(id_values)
#         ax1.axvline(mean_id, color='red', linestyle='--', label=f'Mean: {mean_id:.2f}')
#         ax1.axvline(mean_id + std_id, color='orange', linestyle=':', alpha=0.7, label=f'±1σ: {std_id:.2f}')
#         ax1.axvline(mean_id - std_id, color='orange', linestyle=':', alpha=0.7)
#         ax1.legend()
#
#         # Box plot by layer type
#         unique_types = list(set(layer_types))
#         if len(unique_types) > 1:
#             data_by_type = []
#             labels = []
#             for layer_type in unique_types:
#                 type_values = [id_values[i] for i, t in enumerate(layer_types) if t == layer_type]
#                 if type_values:
#                     data_by_type.append(type_values)
#                     labels.append(f'{layer_type.capitalize()}\n(n={len(type_values)})')
#
#             if data_by_type:
#                 ax2.boxplot(data_by_type, labels=labels)
#                 ax2.set_ylabel('Intrinsic Dimension', fontsize=12)
#                 ax2.set_title('ID by Layer Type', fontsize=13)
#                 ax2.grid(True, alpha=0.3)
#         else:
#             # Single layer type - show layer-wise variation
#             ax2.scatter(layer_indices, id_values, alpha=0.7, s=60)
#             ax2.set_xlabel('Layer Index', fontsize=12)
#             ax2.set_ylabel('Intrinsic Dimension', fontsize=12)
#             ax2.set_title('ID by Layer Index', fontsize=13)
#             ax2.grid(True, alpha=0.3)
#
#         plt.tight_layout()
#
#         if save_plot and self.save_dir:
#             plt.savefig(
#                 os.path.join(self.save_dir, f'{model_name}_id_distribution.png'),
#                 dpi=300, bbox_inches='tight'
#             )
#         plt.show()
#         plt.close()
#
#     def plot_id_heatmap(
#             self,
#             results: Dict[str, Union[float, List]],
#             model_name: str = "",
#             save_plot: bool = True,
#             figsize: Tuple[int, int] = (12, 8)
#     ) -> None:
#         """
#         Plot heatmap of intrinsic dimensions across layers.
#
#         Args:
#             results: Results from analyzer.analyze()
#             model_name: Name of the model
#             save_plot: Whether to save the plot
#             figsize: Figure size
#         """
#         intrinsic_dims = results.get("intrinsic_dimensions", {})
#
#         # Parse layer information
#         layer_data = []
#         for layer_name, id_value in intrinsic_dims.items():
#             parts = layer_name.split('_')
#             if len(parts) >= 3:
#                 layer_idx = int(parts[1])
#                 layer_type = parts[2]
#                 layer_data.append({
#                     'layer_idx': layer_idx,
#                     'layer_type': layer_type,
#                     'id_value': id_value if not np.isnan(id_value) else None
#                 })
#
#         if not layer_data:
#             print("No valid layer data for heatmap")
#             return
#
#         # Create DataFrame
#         df = pd.DataFrame(layer_data)
#
#         # Pivot to create heatmap format
#         pivot_df = df.pivot(index='layer_type', columns='layer_idx', values='id_value')
#
#         # Create heatmap
#         plt.figure(figsize=figsize)
#         mask = pivot_df.isna()
#
#         sns.heatmap(
#             pivot_df,
#             annot=True,
#             fmt='.2f',
#             cmap='viridis',
#             mask=mask,
#             cbar_kws={'label': 'Intrinsic Dimension'},
#             linewidths=0.5
#         )
#
#         plt.title(f'Intrinsic Dimensions Heatmap ({model_name})', fontsize=14)
#         plt.xlabel('Layer Index', fontsize=12)
#         plt.ylabel('Layer Type', fontsize=12)
#         plt.tight_layout()
#
#         if save_plot and self.save_dir:
#             plt.savefig(
#                 os.path.join(self.save_dir, f'{model_name}_id_heatmap.png'),
#                 dpi=300, bbox_inches='tight'
#             )
#         plt.show()
#         plt.close()
#
#     def plot_comparison(
#             self,
#             multiple_results: Dict[str, Dict[str, Union[float, List]]],
#             save_plot: bool = True,
#             figsize: Tuple[int, int] = (14, 8)
#     ) -> None:
#         """
#         Compare intrinsic dimensions across multiple models.
#
#         Args:
#             multiple_results: Dict mapping model names to their results
#             save_plot: Whether to save the plot
#             figsize: Figure size
#         """
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
#
#         # Collect data for comparison
#         all_model_data = {}
#         all_values = []
#
#         for model_name, results in multiple_results.items():
#             intrinsic_dims = results.get("intrinsic_dimensions", {})
#             model_values = [v for v in intrinsic_dims.values() if not np.isnan(v)]
#             if model_values:
#                 all_model_data[model_name] = model_values
#                 all_values.extend(model_values)
#
#         if not all_model_data:
#             print("No valid data for comparison")
#             return
#
#         # Box plot comparison
#         data_for_boxplot = list(all_model_data.values())
#         labels = list(all_model_data.keys())
#
#         box_plot = ax1.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
#
#         # Color the boxes
#         for patch, color in zip(box_plot['boxes'], self.colors):
#             patch.set_facecolor(color)
#             patch.set_alpha(0.7)
#
#         ax1.set_ylabel('Intrinsic Dimension', fontsize=12)
#         ax1.set_title('ID Comparison Across Models', fontsize=13)
#         ax1.grid(True, alpha=0.3)
#         ax1.tick_params(axis='x', rotation=45)
#
#         # Average ID comparison
#         model_names = []
#         mean_ids = []
#         std_ids = []
#
#         for model_name, values in all_model_data.items():
#             model_names.append(model_name)
#             mean_ids.append(np.mean(values))
#             std_ids.append(np.std(values))
#
#         bars = ax2.bar(model_names, mean_ids, yerr=std_ids, capsize=5,
#                        color=self.colors[:len(model_names)], alpha=0.7)
#
#         ax2.set_ylabel('Mean Intrinsic Dimension', fontsize=12)
#         ax2.set_title('Average ID by Model', fontsize=13)
#         ax2.grid(True, alpha=0.3)
#         ax2.tick_params(axis='x', rotation=45)
#
#         # Add value labels on bars
#         for bar, mean_val in zip(bars, mean_ids):
#             height = bar.get_height()
#             ax2.text(bar.get_x() + bar.get_width() / 2., height + max(mean_ids) * 0.01,
#                      f'{mean_val:.2f}', ha='center', va='bottom')
#
#         plt.tight_layout()
#
#         if save_plot and self.save_dir:
#             plt.savefig(
#                 os.path.join(self.save_dir, 'id_model_comparison.png'),
#                 dpi=300, bbox_inches='tight'
#             )
#         plt.show()
#         plt.close()
#
#     def generate_comprehensive_report(
#             self,
#             results: Dict[str, Union[float, List]],
#             model_name: str = "",
#             progression_results: Optional[Dict[str, List[float]]] = None
#     ) -> None:
#         """
#         Generate a comprehensive visualization report.
#
#         Args:
#             results: Results from analyzer.analyze()
#             model_name: Name of the model
#             progression_results: Optional progression analysis results
#         """
#         print(f"Generating comprehensive ID analysis report for {model_name}...")
#
#         # Generate all individual plots
#         self.plot_id_distribution(results, model_name)
#         self.plot_id_heatmap(results, model_name)
#
#         if progression_results:
#             self.plot_layer_progression(progression_results, model_name)
#
#         # Print summary statistics
#         intrinsic_dims = results.get("intrinsic_dimensions", {})
#         valid_values = [v for v in intrinsic_dims.values() if not np.isnan(v)]
#
#         if valid_values:
#             print(f"\n=== Intrinsic Dimension Summary for {model_name} ===")
#             print(f"Number of layers analyzed: {len(valid_values)}")
#             print(f"Mean ID: {np.mean(valid_values):.3f}")
#             print(f"Std ID: {np.std(valid_values):.3f}")
#             print(f"Min ID: {np.min(valid_values):.3f}")
#             print(f"Max ID: {np.max(valid_values):.3f}")
#             print(f"Median ID: {np.median(valid_values):.3f}")
#
#             if self.save_dir:
#                 print(f"Plots saved to: {self.save_dir}")
#
#
# # utils.py (additional utility functions)
# def average_intrinsic_dimension(results: Dict[str, Union[float, List]]) -> float:
#     """
#     Calculate the average intrinsic dimension from analysis results.
#
#     Args:
#         results: Results dictionary from IntrinsicDimensionAnalyzer.analyze()
#
#     Returns:
#         Average intrinsic dimension across all analyzed layers
#     """
#     intrinsic_dims = results.get("intrinsic_dimensions", {})
#
#     # Filter out NaN values
#     valid_values = [v for v in intrinsic_dims.values() if not np.isnan(v)]
#
#     if not valid_values:
#         print("Warning: No valid intrinsic dimension values found")
#         return np.nan
#
#     avg_id = np.mean(valid_values)
#     print(f"Average Intrinsic Dimension: {avg_id:.3f}")
#     print(f"Computed from {len(valid_values)} layers")
#
#     return avg_id
#
#
# def extract_id_statistics(results: Dict[str, Union[float, List]]) -> Dict[str, float]:
#     """
#     Extract comprehensive statistics from ID analysis results.
#
#     Args:
#         results: Results dictionary from IntrinsicDimensionAnalyzer.analyze()
#
#     Returns:
#         Dictionary with statistical measures
#     """
#     intrinsic_dims = results.get("intrinsic_dimensions", {})
#     valid_values = [v for v in intrinsic_dims.values() if not np.isnan(v)]
#
#     if not valid_values:
#         return {"error": "No valid values"}
#
#     stats = {
#         "mean": np.mean(valid_values),
#         "std": np.std(valid_values),
#         "min": np.min(valid_values),
#         "max": np.max(valid_values),
#         "median": np.median(valid_values),
#         "q25": np.percentile(valid_values, 25),
#         "q75": np.percentile(valid_values, 75),
#         "count": len(valid_values)
#     }
#
#     return stats
#
#
# def save_results_to_csv(
#         results: Dict[str, Union[float, List]],
#         filename: str,
#         model_name: str = ""
# ) -> None:
#     """
#     Save ID analysis results to CSV file.
#
#     Args:
#         results: Results dictionary from IntrinsicDimensionAnalyzer.analyze()
#         filename: Output CSV filename
#         model_name: Model name for the CSV
#     """
#     intrinsic_dims = results.get("intrinsic_dimensions", {})
#
#     # Prepare data for CSV
#     data = []
#     for layer_name, id_value in intrinsic_dims.items():
#         parts = layer_name.split('_')
#         layer_idx = int(parts[1]) if len(parts) >= 2 else 0
#         layer_type = parts[2] if len(parts) >= 3 else 'unknown'
#
#         data.append({
#             'model_name': model_name,
#             'layer_name': layer_name,
#             'layer_index': layer_idx,
#             'layer_type': layer_type,
#             'intrinsic_dimension': id_value
#         })
#
#     # Save to CSV
#     df = pd.DataFrame(data)
#     df.to_csv(filename, index=False)
#     print(f"Results saved to {filename}")
#
#
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from collections import defaultdict

from .config import IntrinsicDimensionsConfig


class IntrinsicDimensionsVisualizer:
    """
    Visualizer for intrinsic dimensions analysis results.

    This class handles all visualization tasks for intrinsic dimension analysis,
    including layer-wise plots, evolution tracking, and comparative analysis.
    """

    def __init__(self, log_dir: Optional[str] = None, config: Optional[IntrinsicDimensionsConfig] = None):
        """
        Initialize the visualizer.

        Args:
            log_dir: Directory to save visualizations
            config: Configuration object
        """
        self.log_dir = log_dir
        self.config = config or IntrinsicDimensionsConfig.default()

        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Create output directories
        if log_dir:
            self.plots_dir = os.path.join(log_dir, 'intrinsic_dimensions')
        else:
            log_dir = './plots'
            self.plots_dir = os.path.join(log_dir, 'intrinsic_dimensions')
        os.makedirs(self.plots_dir, exist_ok=True)


    def plot_id_by_layer(
            self,
            intrinsic_dimensions: Dict[Union[Tuple[int, str], str, int], float],
            model_name: str = '',
            save_plot: bool = True
    ) -> None:
        """
        Plot intrinsic dimensions across layers.

        Args:
            intrinsic_dimensions: Dictionary mapping layer keys to ID values
            model_name: Name of the model for plot titles
            save_plot: Whether to save the plot
        """
        if not intrinsic_dimensions:
            print("No intrinsic dimensions data available for plotting")
            return

        # Convert complex keys to readable labels and extract layer information
        layer_data = []
        for layer_key, id_value in intrinsic_dimensions.items():
            layer_info = self._parse_layer_key(layer_key)
            layer_data.append({
                'layer_key': layer_key,
                'layer_label': layer_info['label'],
                'layer_index': layer_info['index'],
                'layer_type': layer_info['type'],
                'id_value': id_value
            })

        # Sort by layer index for consistent ordering
        layer_data.sort(key=lambda x: (x['layer_type'], x['layer_index']))

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Group by layer type for different colors/styles
        layer_types = {}
        for data in layer_data:
            layer_type = data['layer_type']
            if layer_type not in layer_types:
                layer_types[layer_type] = {'labels': [], 'ids': [], 'indices': []}
            layer_types[layer_type]['labels'].append(data['layer_label'])
            layer_types[layer_type]['ids'].append(data['id_value'])
            layer_types[layer_type]['indices'].append(data['layer_index'])

        # Plot each layer type
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']

        for i, (layer_type, data) in enumerate(layer_types.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            plt.plot(data['indices'], data['ids'],
                     marker=marker, linestyle='-', linewidth=2, markersize=8,
                     color=color, label=f'{layer_type.capitalize()} Layers')

        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Intrinsic Dimension', fontsize=12)
        plt.title(f'Intrinsic Dimensions by Layer ({model_name})', fontsize=14)
        plt.grid(True, alpha=0.3)

        if len(layer_types) > 1:
            plt.legend()

        # Add value annotations
        for data in layer_data:
            plt.annotate(f'{data["id_value"]:.1f}',
                         (data['layer_index'], data['id_value']),
                         textcoords="offset points", xytext=(0, 10), ha='center',
                         fontsize=9, alpha=0.8)

        plt.tight_layout()
        print('Plotting intrinsic dimensions by layer...')
        print('Savve pltot:', save_plot)
        print('Model name:', model_name)
        print('Log directory:', self.log_dir)
        if save_plot and self.log_dir:
            plt.savefig(
                os.path.join(self.plots_dir, f'{model_name}_id_by_layer.png'),
                dpi=300, bbox_inches='tight'
            )
        plt.show()
        plt.close()

    def plot_id_distribution(
            self,
            intrinsic_dimensions: Dict[Union[Tuple[int, str], str, int], float],
            model_name: str = '',
            save_plot: bool = True
    ) -> None:
        """
        Plot distribution of intrinsic dimensions.

        Args:
            intrinsic_dimensions: Dictionary mapping layer keys to ID values
            model_name: Name of the model for plot titles
            save_plot: Whether to save the plot
        """
        if not intrinsic_dimensions:
            print("No intrinsic dimensions data available for distribution plot")
            return

        id_values = list(intrinsic_dimensions.values())

        plt.figure(figsize=(10, 6))

        # Create histogram
        plt.hist(id_values, bins=min(15, len(id_values)), alpha=0.7, color='skyblue', edgecolor='black')

        # Add statistics
        mean_id = np.mean(id_values)
        median_id = np.median(id_values)
        std_id = np.std(id_values)

        plt.axvline(mean_id, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_id:.1f}')
        plt.axvline(median_id, color='green', linestyle='--', linewidth=2, label=f'Median: {median_id:.1f}')

        plt.xlabel('Intrinsic Dimension', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution of Intrinsic Dimensions ({model_name})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add text box with statistics
        stats_text = f'Mean: {mean_id:.2f}\nMedian: {median_id:.2f}\nStd: {std_id:.2f}\nMin: {min(id_values):.2f}\nMax: {max(id_values):.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_plot and self.log_dir:
            plt.savefig(
                os.path.join(self.plots_dir, f'{model_name}_id_distribution.png'),
                dpi=300, bbox_inches='tight'
            )
        plt.show()
        plt.close()



    def _parse_layer_key(self, layer_key: Union[Tuple[int, str], str, int]) -> Dict[str, Any]:
        """
        Parse layer key into standardized information.
        """
        if isinstance(layer_key, tuple) and len(layer_key) == 2:
            # (layer_index, layer_type) format
            layer_index, layer_type = layer_key
            return {
                'index': layer_index,
                'type': layer_type,
                'label': f"{layer_type}_{layer_index}"
            }
        elif isinstance(layer_key, str):
            # Try to extract layer information from string
            import re
            numbers = re.findall(r'\d+', layer_key)
            layer_index = int(numbers[0]) if numbers else 0

            if 'encoder' in layer_key.lower():
                layer_type = 'encoder'
            elif 'decoder' in layer_key.lower():
                layer_type = 'decoder'
            else:
                layer_type = 'layer'

            return {
                'index': layer_index,
                'type': layer_type,
                'label': layer_key
            }
        elif isinstance(layer_key, int):
            # Simple integer layer index
            return {
                'index': layer_key,
                'type': 'layer',
                'label': f"layer_{layer_key}"
            }
        else:
            # Fallback for any other type
            return {
                'index': 0,
                'type': 'unknown',
                'label': str(layer_key)
            }

    def save_metrics(
            self,
            intrinsic_dimensions: Dict[Union[Tuple[int, str], str, int], float],
            model_name: str = ''
    ) -> None:
        """
        Save intrinsic dimensions metrics to CSV.

        Args:
            intrinsic_dimensions: Dictionary mapping layer keys to ID values
            model_name: Name of the model
        """
        if not self.log_dir or not intrinsic_dimensions:
            return

        # Prepare data for CSV
        csv_data = []
        for layer_key, id_value in intrinsic_dimensions.items():
            layer_info = self._parse_layer_key(layer_key)
            csv_data.append({
                'layer_key': str(layer_key),
                'layer_index': layer_info['index'],
                'layer_type': layer_info['type'],
                'layer_label': layer_info['label'],
                'intrinsic_dimension': id_value
            })

        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df = df.sort_values(['layer_type', 'layer_index'])

        csv_path = os.path.join(self.plots_dir, f'{model_name}_intrinsic_dimensions.csv')
        df.to_csv(csv_path, index=False)
        print(f"Intrinsic dimensions metrics saved to {csv_path}")

    def generate_all_visualizations(
            self,
            intrinsic_dimensions: Dict[Union[Tuple[int, str], str, int], float],
            model_name: str = '',
            id_evolution: Optional[Dict[Union[Tuple[int, str], str, int], List[Tuple[int, float]]]] = None
    ) -> None:
        """
        Generate all visualizations for intrinsic dimensions analysis.

        Args:
            intrinsic_dimensions: Dictionary mapping layer keys to ID values
            model_name: Name of the model
            id_evolution: Optional evolution data for temporal plots
        """
        print("Generating intrinsic dimensions visualizations...")

        # Generate all plots
        self.plot_id_by_layer(intrinsic_dimensions, model_name)
        self.plot_id_distribution(intrinsic_dimensions, model_name)

        # Save metrics
        self.save_metrics(intrinsic_dimensions, model_name)

        print(f"Intrinsic dimensions visualizations saved to {self.plots_dir}")


# legacy function for backward compatibility
def plot_intrinsic_dimensions(
        intrinsic_dimensions: Dict[Union[Tuple[int, str], str, int], float],
        model_name: str = '',
        save_dir: str = '../plots/intrinsic_dimensions'
) -> None:
    """
    Legacy function for plotting intrinsic dimensions.

    Args:
        intrinsic_dimensions: Dictionary mapping layer keys to ID values
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    visualizer = IntrinsicDimensionsVisualizer(save_dir)
    visualizer.generate_all_visualizations(intrinsic_dimensions, model_name)