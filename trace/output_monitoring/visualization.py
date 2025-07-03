import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Optional, Any
from .config import OutputMonitoringConfig


class OutputMonitoringVisualizer:
    """
    Visualizer for output monitoring analysis results.

    This class handles all visualization tasks for output monitoring analysis,
    including POS accuracy evolution, semantic role performance, and comparative analysis.
    """

    def __init__(self, log_dir: Optional[str] = None, config: Optional[OutputMonitoringConfig] = None):
        """
        Initialize the visualizer.

        Args:
            log_dir: Directory to save visualizations
            config: Configuration object
        """
        self.log_dir = log_dir
        self.config = config or OutputMonitoringConfig.default()

        # Set up matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')

        # Color palette for consistency
        self.colors = plt.cm.tab20.colors

        # Create output directories
        if not self.log_dir:
            self.log_dir = './analysis_results'
        # self.plots_dir = os.path.join(log_dir, 'output_monitoring')
        self.pos_performance_dir = os.path.join(self.log_dir, 'output_pos_performance')
        self.semantic_performance_dir = os.path.join(self.log_dir, 'output_semantic_roles_performance')
        # os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.pos_performance_dir, exist_ok=True)
        os.makedirs(self.semantic_performance_dir, exist_ok=True)

    def plot_pos_performance_evolution(
            self,
            monitoring_results: Dict[int, Dict[str, Any]],
            model_name: str = '',
            save_plot: bool = True
    ) -> None:
        """
        Plot POS accuracy evolution over training steps.

        Args:
            monitoring_results: Dictionary mapping steps to monitoring results
            model_name: Name of the model for plot titles
            save_plot: Whether to save the plot
        """
        # Extract POS data
        pos_data = self._extract_pos_data(monitoring_results)

        if not pos_data:
            print("No POS accuracy data available for plotting")
            return

        plt.figure(figsize=(8, 4))

        # Get all POS categories
        all_pos_tags = set()
        for step_data in pos_data.values():
            all_pos_tags.update(step_data.keys())

        all_pos_tags = sorted(list(all_pos_tags))
        steps = sorted(pos_data.keys())

        # Plot each POS category
        for i, pos_tag in enumerate(all_pos_tags):
            accuracies = []
            valid_steps = []

            for step in steps:
                if pos_tag in pos_data[step]:
                    accuracies.append(pos_data[step][pos_tag])
                    valid_steps.append(step)

            if accuracies:  # Only plot if we have data
                color = self.colors[i % len(self.colors)]
                plt.plot(valid_steps, accuracies,
                         marker='o', linestyle='-', linewidth=2, markersize=6,
                         color=color, label=pos_tag, alpha=0.8)

        plt.xlabel('Training Step', fontsize=16)
        plt.ylabel('POS Accuracy', fontsize=16)
        plt.title(f'POS Accuracy Evolution During Training', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 1.1)

        plt.tight_layout()

        if save_plot and self.pos_performance_dir :
            plt.savefig(
                os.path.join(self.pos_performance_dir , f'{model_name}_pos_accuracy_evolution.png'),
                dpi=300, bbox_inches='tight'
            )
        if self.config.show_plots:
            plt.show()
        plt.close()

    def plot_semantic_role_performance_evolution(
            self,
            monitoring_results: Dict[int, Dict[str, Any]],
            model_name: str = '',
            save_plot: bool = True
    ) -> None:
        """
        Plot semantic role accuracy evolution over training steps.

        Args:
            monitoring_results: Dictionary mapping steps to monitoring results
            model_name: Name of the model for plot titles
            save_plot: Whether to save the plot
        """
        # Extract semantic data
        semantic_data = self._extract_semantic_data(monitoring_results)

        if not semantic_data:
            print("No semantic accuracy data available for plotting")
            return

        plt.figure(figsize=(8, 4))

        # Get all semantic categories
        all_semantic_tags = set()
        for step_data in semantic_data.values():
            all_semantic_tags.update(step_data.keys())

        all_semantic_tags = sorted(list(all_semantic_tags))
        steps = sorted(semantic_data.keys())

        # Plot each semantic category
        for i, semantic_tag in enumerate(all_semantic_tags):
            accuracies = []
            valid_steps = []

            for step in steps:
                if semantic_tag in semantic_data[step]:
                    accuracies.append(semantic_data[step][semantic_tag])
                    valid_steps.append(step)

            if accuracies:  # Only plot if we have data
                color = self.colors[i % len(self.colors)]
                plt.plot(valid_steps, accuracies,
                         marker='s', linestyle='-', linewidth=2, markersize=6,
                         color=color, label=semantic_tag, alpha=0.8)

        plt.xlabel('Training Step', fontsize=16)
        plt.ylabel('Semantic Role Accuracy', fontsize=16)
        plt.title(f'Semantic Role Accuracy Evolution During Training ', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 1.1)

        plt.tight_layout()

        if save_plot and self.semantic_performance_dir:
            plt.savefig(
                os.path.join(self.semantic_performance_dir, f'{model_name}_semantic_accuracy_evolution.png'),
                dpi=300, bbox_inches='tight'
            )
        if self.config.show_plots:
            plt.show()
        plt.close()


    def save_metrics(
            self,
            monitoring_results: Dict[int, Dict[str, Any]],
            model_name: str = ''
    ) -> None:
        """
        Save output monitoring metrics to CSV.

        Args:
            monitoring_results: Dictionary mapping steps to monitoring results
            model_name: Name of the model
        """
        if not self.log_dir or not monitoring_results:
            return

        # Prepare data for CSV
        csv_data = []
        for step, step_data in monitoring_results.items():
            # POS data
            if 'pos_accuracy' in step_data:
                for category, accuracy in step_data['pos_accuracy'].items():
                    csv_data.append({
                        'step': step,
                        'analysis_type': 'pos',
                        'category': category,
                        'accuracy': accuracy
                    })

            # Semantic data
            if 'semantic_accuracy' in step_data:
                for category, accuracy in step_data['semantic_accuracy'].items():
                    csv_data.append({
                        'step': step,
                        'analysis_type': 'semantic',
                        'category': category,
                        'accuracy': accuracy
                    })

        # Create DataFrame and save
        if csv_data:
            df = pd.DataFrame(csv_data)
            df = df.sort_values(['step', 'analysis_type', 'category'])

            csv_path = os.path.join(self.plots_dir, f'{model_name}_output_monitoring.csv')
            df.to_csv(csv_path, index=False)
            print(f"Output monitoring metrics saved to {csv_path}")

    def _extract_pos_data(self, monitoring_results: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        """Extract POS accuracy data from monitoring results."""
        pos_data = {}
        for step, step_data in monitoring_results.items():
            if 'pos_accuracy' in step_data:
                pos_data[step] = step_data['pos_accuracy']
        return pos_data

    def _extract_semantic_data(self, monitoring_results: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
        """Extract semantic accuracy data from monitoring results."""
        semantic_data = {}
        for step, step_data in monitoring_results.items():
            if 'semantic_accuracy' in step_data:
                semantic_data[step] = step_data['semantic_accuracy']
        return semantic_data