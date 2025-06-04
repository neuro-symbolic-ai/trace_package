import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List


class HessianVisualizer:
    """
    Class for creating visualizations of Hessian analysis results.
    """

    @staticmethod
    def plot_eigenvalue_evolution(
            hessian_history: Dict[int, Dict[str, Any]],
            save_path: Optional[str] = None,
            model_name: str = ""
    ) -> None:
        """
        Plot evolution of Hessian eigenvalues over training.

        Args:
            hessian_history: Dictionary of Hessian metrics at each step
            save_path: Path to save plots
            model_name: Model name for plot titles and filenames

            Creates multiple visualizations:
            1. Evolution of extreme eigenvalues (max and min)
            2. Number of negative eigenvalues over time
            3. Hessian trace over time
            4. Heatmap of top eigenvalues throughout training
        """
        if not hessian_history:
            print("No Hessian history to plot")
            return

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        # Extract and sort steps
        steps = sorted([int(step) for step in hessian_history.keys()])

        # Extract metrics
        max_eigs = [hessian_history[step]["max_eigenvalue"] for step in steps]
        min_eigs = [hessian_history[step]["min_eigenvalue"] for step in steps]
        traces = [hessian_history[step]["hessian_trace_estimate"] for step in steps]
        negative_counts = [hessian_history[step]["negative_count"] for step in steps]

        # Set up plotting style
        plt.style.use('seaborn-v0_8-whitegrid') # tried others and this one was clearest
        fig_size = (12, 8)

        # Plot extreme eigenvalues
        plt.figure(figsize=fig_size)
        plt.semilogy(steps, max_eigs, 'b-', label='Max Eigenvalue', linewidth=2)
        plt.semilogy(steps, [abs(v) for v in min_eigs], 'r-', label='|Min Eigenvalue|', linewidth=2)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Eigenvalue (log scale)', fontsize=12)
        plt.title(f'Evolution of Extreme Eigenvalues - {model_name}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, f'{model_name}_extreme_eigenvalues.png'), dpi=300)
        plt.close()

        # Plot trace evolution
        plt.figure(figsize=fig_size)
        plt.plot(steps, traces, 'g-', linewidth=2)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Hessian Trace Estimate', fontsize=12)
        plt.title(f'Evolution of Hessian Trace - {model_name}', fontsize=14)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, f'{model_name}_hessian_trace.png'), dpi=300)
        plt.close()

        # Plot negative eigenvalue count
        plt.figure(figsize=fig_size)
        plt.plot(steps, negative_counts, 'm-', linewidth=2)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Number of Negative Eigenvalues', fontsize=12)
        plt.title(f'Evolution of Negative Eigenvalues - {model_name}', fontsize=14)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, f'{model_name}_negative_eigenvalues.png'), dpi=300)
        plt.close()

        # Create eigenvalue heatmap
        HessianVisualizer._plot_eigenvalue_heatmap(hessian_history, steps, save_path, model_name)

    @staticmethod
    def _plot_eigenvalue_heatmap(
            hessian_history: Dict[int, Dict[str, Any]],
            steps: List[int],
            save_path: Optional[str],
            model_name: str
    ) -> None:
        """Create heatmap of top eigenvalues throughout training."""
        # Sample steps to avoid overcrowding
        steps_to_plot = sorted(steps)[::max(1, len(steps) // 20)]
        top_n = min(8, len(hessian_history[steps[0]]["top_eigenvalues"]))

        eigenvalue_matrix = np.zeros((len(steps_to_plot), top_n))

        for i, step in enumerate(steps_to_plot):
            eigenvalue_matrix[i, :] = hessian_history[step]["top_eigenvalues"][:top_n]

        plt.figure(figsize=(12, 8))
        im = plt.imshow(eigenvalue_matrix.T, aspect='auto', cmap='viridis')
        plt.colorbar(im, label='Eigenvalue')
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Eigenvalue Index', fontsize=12)
        plt.title(f'Top Eigenvalues Throughout Training - {model_name}', fontsize=14)
        plt.xticks(range(len(steps_to_plot)), [str(s) for s in steps_to_plot], rotation=45)
        plt.yticks(range(top_n), range(1, top_n + 1))
        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, f'{model_name}_eigenvalue_heatmap.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_gradient_alignment(
            alignment_history: Dict[int, Dict[str, Any]],
            save_path: Optional[str] = None,
            model_name: str = ""
    ) -> None:
        """
        Plot gradient-Hessian alignment metrics.

        Args:
            alignment_history: Dictionary of alignment metrics at each step
            save_path: Path to save plots
            model_name: Model name for plot titles and filenames
        """
        if not alignment_history:
            print("No alignment history to plot")
            return

        steps = sorted([int(step) for step in alignment_history.keys()])
        plt.style.use('seaborn-v0_8-whitegrid')
        fig_size = (12, 8)

        # Plot gradient-Hessian alignment
        if all("grad_Hg_alignment" in alignment_history[step] for step in steps):
            alignments = [alignment_history[step]["grad_Hg_alignment"] for step in steps]

            plt.figure(figsize=fig_size)
            plt.plot(steps, alignments, 'b-', linewidth=2)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Gradient-Hessian Alignment', fontsize=12)
            plt.title(f'Evolution of Gradient-Hessian Alignment - {model_name}', fontsize=14)
            plt.grid(True, alpha=0.2)
            plt.ylim(0, 1.05)
            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, f'{model_name}_grad_hessian_alignment.png'), dpi=300)
            plt.close()

        # Plot gradient and Hg norms
        HessianVisualizer._plot_gradient_norms(alignment_history, steps, save_path, model_name)

        # Plot curvature-to-gradient ratio
        HessianVisualizer._plot_curvature_gradient_ratio(alignment_history, steps, save_path, model_name)

    @staticmethod
    def _plot_gradient_norms(
            alignment_history: Dict[int, Dict[str, Any]],
            steps: List[int],
            save_path: Optional[str],
            model_name: str
    ) -> None:
        """Plot gradient norm and Hessian-gradient norm comparison."""
        if not all("grad_norm" in alignment_history[step] and "Hg_norm" in alignment_history[step]
                   for step in steps):
            return

        grad_norms = [alignment_history[step]["grad_norm"] for step in steps]
        Hg_norms = [alignment_history[step]["Hg_norm"] for step in steps]

        fig, ax1 = plt.subplots(figsize=(12, 8))

        color1 = 'tab:blue'
        ax1.set_xlabel('Training Step', fontsize=12)
        ax1.set_ylabel('Gradient Norm', color=color1, fontsize=12)
        ax1.semilogy(steps, grad_norms, color=color1, linewidth=2, label='Gradient Norm')
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Hessian-Gradient Norm', color=color2, fontsize=12)
        ax2.semilogy(steps, Hg_norms, color=color2, linewidth=2, label='Hessian-Gradient Norm')
        ax2.tick_params(axis='y', labelcolor=color2)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title(f'Gradient and Hessian-Gradient Norm Comparison - {model_name}', fontsize=14)
        fig.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, f'{model_name}_grad_hg_norms.png'), dpi=300)
        plt.close()

    @staticmethod
    def _plot_curvature_gradient_ratio(
            alignment_history: Dict[int, Dict[str, Any]],
            steps: List[int],
            save_path: Optional[str],
            model_name: str
    ) -> None:
        """Plot curvature-to-gradient ratio."""
        if not all("grad_Hg_ratio" in alignment_history[step] for step in steps):
            return

        ratios = [alignment_history[step]["grad_Hg_ratio"] for step in steps]

        plt.figure(figsize=(12, 8))
        plt.semilogy(steps, ratios, 'g-', linewidth=2)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Hg/Gradient Norm Ratio (log scale)', fontsize=12)
        plt.title(f'Evolution of Curvature-to-Gradient Ratio - {model_name}', fontsize=14)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, f'{model_name}_curvature_gradient_ratio.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_component_comparison(
            component_history: Dict[str, Dict[int, Dict[str, Any]]],
            save_path: Optional[str] = None,
            model_name: str = ""
    ) -> None:
        """
        Plot comparison of Hessian metrics across different model components.

        Args:
            component_history: Dictionary of component-specific Hessian metrics
            save_path: Path to save plots
            model_name: Model name for plot titles and filenames
        """
        if not component_history:
            print("No component history to plot")
            return

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        components = list(component_history.keys())
        common_steps = HessianVisualizer._find_common_steps(component_history)

        if not common_steps:
            print("No common steps found across components")
            return

        colors = plt.cm.tab10(np.linspace(0, 1, len(components)))
        plt.style.use('seaborn-v0_8-whitegrid')

        # Plot different metrics
        metrics_to_plot = {
            "max_eigenvalue": ("Max Eigenvalue (log scale)", True),
            "hessian_trace_estimate": ("Hessian Trace Estimate", False),
            "negative_count": ("Number of Negative Eigenvalues", False),
            "effective_rank_95": ("Effective Rank (95% energy)", False),
            "effective_rank_entropy": ("Effective Rank (entropy)", False)
        }

        for metric, (ylabel, use_log) in metrics_to_plot.items():
            HessianVisualizer._plot_metric_comparison(
                component_history, common_steps, components, colors,
                metric, ylabel, use_log, save_path, model_name
            )

    @staticmethod
    def _find_common_steps(component_history: Dict[str, Dict[int, Dict[str, Any]]]) -> List[int]: # todo: this is not working
        """Find common steps across all components."""
        common_steps = set()
        print('component_history:', component_history)
        for component, history in component_history.items():
            if not common_steps:
                common_steps = set(int(step) for step in history.keys())
            else:
                common_steps &= set(int(step) for step in history.keys())
        return sorted(common_steps)

    @staticmethod
    def _plot_metric_comparison(
            component_history: Dict[str, Dict[int, Dict[str, Any]]],
            common_steps: List[int],
            components: List[str],
            colors: np.ndarray,
            metric: str,
            ylabel: str,
            use_log: bool,
            save_path: Optional[str],
            model_name: str
    ) -> None:
        """Plot a specific metric comparison across components."""
        # Check if metric exists in all components
        if not all(all(metric in component_history[comp][step]
                       for step in common_steps if step in component_history[comp])
                   for comp in components):
            return

        plt.figure(figsize=(14, 10))

        for i, component in enumerate(components):
            values = []
            for step in common_steps:
                if step in component_history[component]:
                    val = component_history[component][step].get(metric, np.nan)
                    if use_log and not np.isnan(val):
                        val = abs(val) if val != 0 else 1e-10
                    values.append(val)
                else:
                    values.append(np.nan)

            if use_log:
                plt.semilogy(common_steps, values, linewidth=2, label=component, color=colors[i])
            else:
                plt.plot(common_steps, values, linewidth=2, label=component, color=colors[i])

        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f'{ylabel} by Component - {model_name}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        if save_path:
            filename = f'{model_name}_component_{metric}.png'
            plt.savefig(os.path.join(save_path, filename), dpi=300)
        plt.close()

    @staticmethod
    def plot_memorization_metrics(
            memorization_history: Dict[int, Dict[str, Any]],
            save_path: Optional[str] = None,
            model_name: str = ""
    ) -> None:
        """
        Plot metrics related to memorization detection.

        Args:
            memorization_history: Dictionary of memorization metrics
            save_path: Path to save plots
            model_name: Model name for plot titles and filenames
        """
        if not memorization_history:
            print("No memorization history to plot")
            return

        if save_path:
            os.makedirs(save_path, exist_ok=True)

        steps = sorted([int(step) for step in memorization_history.keys()])
        plt.style.use('seaborn-v0_8-whitegrid')
        fig_size = (12, 8)

        # Plot memorization score - changed this from original code (memorization_score for simplicity - check hessian_ana;ysis.py for original)
        if all("train_val_landscape_divergence_score" in memorization_history[step] for step in steps):
            scores = [memorization_history[step]["train_val_landscape_divergence_score"] for step in steps]

            plt.figure(figsize=fig_size)
            plt.plot(steps, scores, 'r-', linewidth=2)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Train-Val Landscape Divergence Score', fontsize=12)
            plt.title(f'Evolution of Train-Val Landscape Divergence - {model_name}', fontsize=14)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, f'{model_name}_memorization_score.png'), dpi=300)
            plt.close()

        # Plot trace ratio
        if all("trace_ratio" in memorization_history[step] for step in steps):
            trace_ratios = [memorization_history[step]["trace_ratio"] for step in steps]

            plt.figure(figsize=fig_size)
            plt.plot(steps, trace_ratios, 'b-', linewidth=2)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Train/Val Trace Ratio', fontsize=12)
            plt.title(f'Evolution of Hessian Trace Ratio - {model_name}', fontsize=14)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, f'{model_name}_trace_ratio.png'), dpi=300)
            plt.close()

        # Plot eigenvalue distribution overlap
        if all("eigenvalue_distribution_overlap" in memorization_history[step] for step in steps):
            overlaps = [memorization_history[step]["eigenvalue_distribution_overlap"] for step in steps]

            plt.figure(figsize=fig_size)
            plt.plot(steps, overlaps, 'g-', linewidth=2)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Distribution Overlap', fontsize=12)
            plt.title(f'Train/Val Eigenvalue Distribution Overlap - {model_name}', fontsize=14)
            plt.grid(True, alpha=0.2)
            plt.ylim(0, 1.05) # Overlap is between 0 and 1
            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, f'{model_name}_eigenvalue_overlap.png'), dpi=300)
            plt.close()

        # Plot comprehensive comparison
        HessianVisualizer._plot_memorization_comparison(memorization_history, steps, save_path, model_name)

    @staticmethod
    def _plot_memorization_comparison(
            memorization_history: Dict[int, Dict[str, Any]],
            steps: List[int],
            save_path: Optional[str],
            model_name: str
    ) -> None:
        """Plot multiple memorization metrics in one figure for comparison."""
        metrics_to_plot = {
            "trace_ratio": "Train/Val Trace Ratio",
            "max_eigenvalue_ratio": "Train/Val Max Eigenvalue Ratio",
            "effective_rank_diff": "Effective Rank Difference",
            "eigenvalue_distribution_overlap": "Eigenvalue Distribution Overlap",
            "complexity_score": "Curvature Complexity Score",
        }

        available_metrics = {
            metric: label for metric, label in metrics_to_plot.items()
            if all(metric in memorization_history[step] for step in steps)
        }

        if not available_metrics:
            return

        plt.figure(figsize=(12, 8))

        for metric, label in available_metrics.items():
            values = [memorization_history[step][metric] for step in steps]

            # Normalize to 0-1 range for comparison if not already in that range
            if metric != "eigenvalue_distribution_overlap":
                min_val = min(values)
                max_val = max(values)
                if max_val > min_val:
                    values = [(v - min_val) / (max_val - min_val) for v in values]

            plt.plot(steps, values, linewidth=2, label=label)

        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Normalized Value', fontsize=12)
        plt.title(f'Memorization Metrics Comparison - {model_name}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()

        if save_path:
            plt.savefig(os.path.join(save_path, f'{model_name}_memorization_metrics.png'), dpi=300)
        plt.close()

    @staticmethod
    def plot_complexity_metrics(
            hessian_history: Dict[int, Dict[str, Any]],
            save_path: Optional[str] = None,
            model_name: str = ""
    ) -> None:
        """
        Plot complexity-related metrics from Hessian analysis.

        Args:
            hessian_history: Dictionary of Hessian metrics
            save_path: Path to save plots
            model_name: Model name for plot titles and filenames
        """
        if not hessian_history:
            print("No Hessian history to plot")
            return

        steps = sorted([int(step) for step in hessian_history.keys()])
        plt.style.use('seaborn-v0_8-whitegrid')
        fig_size = (12, 8)

        # Plot complexity score
        if all("complexity_score" in hessian_history[step] for step in steps):
            complexity_scores = [hessian_history[step]["complexity_score"] for step in steps]

            plt.figure(figsize=fig_size)
            plt.plot(steps, complexity_scores, 'm-', linewidth=2)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Complexity Score', fontsize=12)
            plt.title(f'Evolution of Complexity Score - {model_name}', fontsize=14)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, f'{model_name}_complexity_score.png'), dpi=300)
            plt.close()

        # Plot effective rank metrics
        if all("effective_rank_95" in hessian_history[step] for step in steps):
            effective_rank_95 = [hessian_history[step]["effective_rank_95"] for step in steps]

            plt.figure(figsize=fig_size)
            plt.plot(steps, effective_rank_95, 'c-', linewidth=2)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Effective Rank (95% energy)', fontsize=12)
            plt.title(f'Evolution of Effective Rank (95%) - {model_name}', fontsize=14)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, f'{model_name}_effective_rank_95.png'), dpi=300)
            plt.close()

        # Plot effective rank entropy
        if all("effective_rank_entropy" in hessian_history[step] for step in steps):
            effective_rank_entropy = [hessian_history[step]["effective_rank_entropy"] for step in steps]

            plt.figure(figsize=fig_size)
            plt.plot(steps, effective_rank_entropy, 'y-', linewidth=2)
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Effective Rank (entropy)', fontsize=12)
            plt.title(f'Evolution of Effective Rank (entropy) - {model_name}', fontsize=14)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()

            if save_path:
                plt.savefig(os.path.join(save_path, f'{model_name}_effective_rank_entropy.png'), dpi=300)
            plt.close()

        # Plot condition number
        if all("condition_number" in hessian_history[step] for step in steps):
            condition_numbers = [hessian_history[step]["condition_number"] for step in steps]
            # Filter out infinite values for plotting
            finite_steps = []
            finite_conditions = []
            for step, cond in zip(steps, condition_numbers):
                if np.isfinite(cond):
                    finite_steps.append(step)
                    finite_conditions.append(cond)

            if finite_conditions:
                plt.figure(figsize=fig_size)
                plt.semilogy(finite_steps, finite_conditions, 'orange', linewidth=2)
                plt.xlabel('Training Step', fontsize=12)
                plt.ylabel('Condition Number (log scale)', fontsize=12)
                plt.title(f'Evolution of Condition Number - {model_name}', fontsize=14)
                plt.grid(True, which="both", ls="-", alpha=0.2)
                plt.tight_layout()

                if save_path:
                    plt.savefig(os.path.join(save_path, f'{model_name}_condition_number.png'), dpi=300)
                plt.close()

    @staticmethod
    def create_comprehensive_report(
            hessian_history: Dict[int, Dict[str, Any]],
            component_history: Optional[Dict[str, Dict[int, Dict[str, Any]]]] = None,
            alignment_history: Optional[Dict[int, Dict[str, Any]]] = None,
            memorization_history: Optional[Dict[int, Dict[str, Any]]] = None,
            save_path: Optional[str] = None,
            model_name: str = ""
    ) -> None:
        """
        Create a comprehensive visualization report of all Hessian analysis results.

        Args:
            hessian_history: Basic Hessian metrics history
            component_history: Component-specific metrics history
            alignment_history: Gradient alignment metrics history
            memorization_history: Memorization detection metrics history
            save_path: Path to save plots
            model_name: Model name for plot titles and filenames
        """
        if save_path:
            os.makedirs(save_path, exist_ok=True)

        print(f"Creating comprehensive Hessian analysis report for {model_name}")

        # Plot basic Hessian evolution
        if hessian_history:
            print("Plotting eigenvalue evolution...")
            HessianVisualizer.plot_eigenvalue_evolution(hessian_history, save_path, model_name)

            print("Plotting complexity metrics...")
            HessianVisualizer.plot_complexity_metrics(hessian_history, save_path, model_name)

        # Plot gradient alignment metrics
        if alignment_history:
            print("Plotting gradient alignment metrics...")
            HessianVisualizer.plot_gradient_alignment(alignment_history, save_path, model_name)

        # Plot component comparisons
        if component_history:
            print("Plotting component comparisons...")
            HessianVisualizer.plot_component_comparison(component_history, save_path, model_name)

        # Plot memorization metrics
        if memorization_history:
            print("Plotting memorization metrics...")
            HessianVisualizer.plot_memorization_metrics(memorization_history, save_path, model_name)

        print(f"Comprehensive Hessian analysis report saved to {save_path}")

    @staticmethod
    def save_analysis_summary(
            analysis_results: Dict[str, Any],
            save_path: str,
            model_name: str
    ) -> None:
        """
        Save a text summary of analysis results.

        Args:
            analysis_results: Results from HessianAnalyzer
            save_path: Path to save summary
            model_name: Model name
        """
        summary_path = os.path.join(save_path, f"{model_name}_hessian_summary.txt")

        with open(summary_path, 'w') as f:
            f.write(f"Hessian Analysis Summary for {model_name}\n")
            f.write("=" * 50 + "\n\n")

            if "hessian" in analysis_results:
                hessian = analysis_results["hessian"]
                f.write("Basic Hessian Metrics:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Max Eigenvalue: {hessian.get('max_eigenvalue', 'N/A'):.2e}\n")
                f.write(f"Min Eigenvalue: {hessian.get('min_eigenvalue', 'N/A'):.2e}\n")
                f.write(f"Trace Estimate: {hessian.get('hessian_trace_estimate', 'N/A'):.2e}\n")
                f.write(f"Negative Count: {hessian.get('negative_count', 'N/A')}\n")
                f.write(f"Effective Rank (95%): {hessian.get('effective_rank_95', 'N/A')}\n")
                f.write(f"Complexity Score: {hessian.get('complexity_score', 'N/A'):.2e}\n\n")

            if "alignment" in analysis_results:
                alignment = analysis_results["alignment"]
                f.write("Gradient Alignment Metrics:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Grad-Hessian Alignment: {alignment.get('grad_Hg_alignment', 'N/A'):.4f}\n")
                f.write(f"Weighted Alignment: {alignment.get('weighted_alignment', 'N/A'):.4f}\n")
                f.write(f"Curvature-Gradient Ratio: {alignment.get('grad_Hg_ratio', 'N/A'):.4f}\n\n")

            if "train_val_divergence" in analysis_results:
                divergence = analysis_results["train_val_divergence"]
                f.write("Memorization Signals:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Divergence Score: {divergence.get('train_val_landscape_divergence_score', 'N/A'):.4f}\n")
                f.write(f"Trace Ratio: {divergence.get('trace_ratio', 'N/A'):.4f}\n")
                f.write(f"Distribution Overlap: {divergence.get('eigenvalue_distribution_overlap', 'N/A'):.4f}\n\n")

            if "components" in analysis_results:
                components = analysis_results["components"]
                f.write("Component Analysis:\n")
                f.write("-" * 20 + "\n")
                for comp_name, comp_data in components.items():
                    if "error" not in comp_data:
                        f.write(f"{comp_name}:\n")
                        f.write(f"  Max Eigenvalue: {comp_data.get('max_eigenvalue', 'N/A'):.2e}\n")
                        f.write(f"  Parameters: {comp_data.get('num_params', 'N/A')}\n")
                f.write("\n")


# Legacy compatibility functions for pre_training.py
def plot_hessian_evolution(hessian_history, alignment_history=None, plots_path=None, model_name=''):
    """Legacy wrapper for hessian evolution plotting."""
    HessianVisualizer.plot_eigenvalue_evolution(hessian_history, plots_path, model_name)

    if alignment_history:
        HessianVisualizer.plot_gradient_alignment(alignment_history, plots_path, model_name)


def plot_component_comparison(component_history, plots_path=None, model_name=""):
    """Legacy wrapper for component comparison plotting."""
    HessianVisualizer.plot_component_comparison(component_history, plots_path, model_name)


def plot_train_val_landscape_divergence_metrics(memorization_history, plots_path=None, model_name=""):
    """Legacy wrapper for memorization metrics plotting."""
    HessianVisualizer.plot_memorization_metrics(memorization_history, plots_path, model_name)