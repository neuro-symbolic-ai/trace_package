import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
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
            save_plot: bool = True,
            show_plots: bool = False
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
        colors = plt.cm.tab20.colors
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
        print('Save plot:', save_plot)
        print('Model name:', model_name)
        print('Log directory:', self.log_dir)
        if save_plot and self.log_dir:
            plt.savefig(
                os.path.join(self.plots_dir, f'{model_name}_id_by_layer.png'),
                dpi=300, bbox_inches='tight'
            )
        if show_plots or self.config.show_plots:
            plt.show()
        plt.close()

    def plot_final_id(self,
            intrinsic_dimensions: Dict[Union[Tuple[int, str], str, int], float],
            model_name: str = '',
            save_plot: bool = True,
            averaged: bool = True,
            show_plots: bool = False
    ) -> None:
        """
        Plot final intrinsic dimensions for each layer.

        Args:
            intrinsic_dimensions: Dictionary mapping layer keys to ID values
            model_name: Name of the model for plot titles
            save_plot: Whether to save the plot
        """
        if not intrinsic_dimensions:
            print("No intrinsic dimensions data available for final ID plot")
            return
        # print('Plotting final intrinsic dimensions...', intrinsic_dimensions)
        steps = list(intrinsic_dimensions.keys())
        IDs = []
        for step, layers in intrinsic_dimensions.items():
            avg = 0
            for _, ID in layers.items():
                avg += ID
            avg /= len(layers)
            IDs.append(avg)

        # ids = list(intrinsic_dimensions.values())

        plt.figure(figsize=(10, 6))
        plt.plot(steps, IDs, marker='o', linestyle='-', linewidth=2, markersize=8, color='red', alpha=0.8)

        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Intrinsic Dimension', fontsize=12)
        plt.title(f'Final Intrinsic Dimensions ({model_name})', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        if save_plot and self.log_dir:
            plt.savefig(
                os.path.join(self.plots_dir, f'{model_name}_final_id.png'),
                dpi=300, bbox_inches='tight'
            )
        if show_plots or self.config.show_plots:
            plt.show()
        plt.close()

    def plot_id_distribution(
            self,
            intrinsic_dimensions: Dict[Union[Tuple[int, str], str, int], float],
            model_name: str = '',
            save_plot: bool = True,
            show_plots: bool = False
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
        if show_plots or self.config.show_plots:
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
            id_evolution: Optional[Dict[Union[Tuple[int, str], str, int], List[Tuple[int, float]]]] = None,
            show_plots: bool = False,
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
        self.plot_id_by_layer(intrinsic_dimensions, model_name, show_plots=show_plots)
        self.plot_id_distribution(intrinsic_dimensions, model_name, show_plots=show_plots)

        # Save metrics
        self.save_metrics(intrinsic_dimensions, model_name)

        print(f"Intrinsic dimensions visualizations saved to {self.plots_dir}")


# legacy function for backward compatibility
def plot_intrinsic_dimensions(
        intrinsic_dimensions: Dict[Union[Tuple[int, str], str, int], float],
        model_name: str = '',
        save_dir: str = '../plots/intrinsic_dimensions',
        show_plots: bool = False
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