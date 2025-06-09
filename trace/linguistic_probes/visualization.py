import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from collections import defaultdict

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
            probe_results: Dict[str, Dict],
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

        # Extract overall accuracies by layer
        layers = sorted(probe_results.keys())
        overall_accuracies = []

        for layer in layers:
            results = probe_results[layer]
            if 'overall_accuracy' in results:
                overall_accuracies.append(results['overall_accuracy'])
            else:
                overall_accuracies.append(0.0)

        # Plot layer comparison
        plt.figure(figsize=(10, 6))
        plt.plot(layers, overall_accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Overall Accuracy', fontsize=12)
        plt.title(f'{analysis_type.upper()} Probe Accuracy by Layer ({model_name})', fontsize=14)
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

    def plot_category_distribution(
            self,
            sample_counts: Dict[str, int],
            model_name: str = '',
            analysis_type: str = 'pos'
    ) -> None:
        """
        Plot distribution of categories in the dataset.

        Args:
            sample_counts: Count of samples per category
            model_name: Name of the model
            analysis_type: Type of analysis ('pos' or 'semantic')
        """
        if not sample_counts:
            print(f"No {analysis_type} sample counts available")
            return

        categories = list(sample_counts.keys())
        counts = list(sample_counts.values())

        plt.figure(figsize=(12, 6))
        bars = plt.bar(categories, counts)
        plt.xlabel(f'{analysis_type.upper()} Category', fontsize=12)
        plt.ylabel('Sample Count', fontsize=12)
        plt.title(f'{analysis_type.upper()} Category Distribution ({model_name})', fontsize=14)
        plt.xticks(rotation=45, ha='right')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                     str(count), ha='center', va='bottom')

        plt.tight_layout()

        if self.log_dir:
            plots_dir = getattr(self, f'{analysis_type}_plots_dir', self.log_dir)
            plt.savefig(
                os.path.join(plots_dir, f'{model_name}_{analysis_type}_distribution.png'),
                dpi=300,
                bbox_inches='tight'
            )

        plt.close()

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

    def generate_all_visualizations(
            self,
            accuracy_data: Dict[str, List[tuple]],
            sample_counts: Dict[str, int],
            model_name: str = '',
            analysis_type: str = 'pos'
    ) -> None:
        """
        Generate all visualizations for the given data.

        Args:
            accuracy_data: Accuracy data by category
            sample_counts: Sample counts per category
            model_name: Name of the model
            analysis_type: Type of analysis ('pos' or 'semantic')
        """
        if not self.config.save_visualizations:
            return

        print(f"Generating {analysis_type} visualizations...")

        # Generate all plots
        self.plot_learning_curves(accuracy_data, sample_counts, model_name, analysis_type)
        self.plot_heatmap(accuracy_data, model_name, analysis_type)
        self.plot_category_distribution(sample_counts, model_name, analysis_type)

        # Save metrics
        self.save_metrics(accuracy_data, sample_counts, model_name, analysis_type)

        plots_dir = getattr(self, f'{analysis_type}_plots_dir', self.log_dir)
        print(f"{analysis_type.upper()} analysis visualizations saved to {plots_dir}")


# old compatibility functions
def plot_pos_learning_curves_legacy(tracker, model_name: str = ''):
    """Legacy function for plotting POS learning curves."""
    visualizer = ProbesVisualizer(tracker.log_dir)
    visualizer.plot_learning_curves(
        tracker.accuracy,
        tracker.sample_counts,
        model_name,
        'pos'
    )


def plot_semantic_learning_curves_legacy(tracker, model_name: str = ''):
    """Legacy function for plotting semantic learning curves."""
    visualizer = ProbesVisualizer(tracker.log_dir)
    visualizer.plot_learning_curves(
        tracker.accuracy,
        tracker.sample_counts,
        model_name,
        'semantic'
    )