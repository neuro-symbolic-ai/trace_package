import torch
from typing import List, Dict, Any, Optional
from .utils import extract_component_parameters, get_hessian_eigenvectors
from .metrics import HessianMetrics
from .config import HessianConfig
from .. import Transformer


class ComponentSelector:
    """
    Utility class for selecting and managing model components for Hessian analysis.

    Provides methods to define and filter model components following
    the reference style pattern with static methods.
    """

    @staticmethod
    def get_standard_components(no_fnn: bool = False) -> List[str]:
        """
        Get standard list of components for analysis.

        Args:
            no_fnn: Whether the model has feed-forward networks disabled

        Returns:
            List of standard component names
        """
        if no_fnn:
            return ["attention", "hidden_states"]
        else:
            return ["attention", "ffn", "hidden_states"]

    @staticmethod
    def get_comprehensive_components() -> List[str]:
        """Get comprehensive list of components for detailed analysis."""
        return [
            "attention", "attention_query", "attention_key", "attention_value",
            "ffn", "embeddings", "norm", "hidden_states", "output_projection"
        ]

    @staticmethod
    def get_minimal_components() -> List[str]:
        """Get minimal list of components for basic analysis."""
        return ["attention", "ffn"]

    @staticmethod
    def validate_components(model, component_list: List[str]) -> List[str]:
        """
        Validate that components exist in the model and return valid ones.

        Args:
            model: The transformer model
            component_list: List of component names to validate

        Returns:
            List of valid component names
        """
        valid_components = []

        for component in component_list:
            try:
                params = extract_component_parameters(model, component)
                if params:  # Component has parameters
                    valid_components.append(component)
            except ValueError:
                print(f"Warning: Component '{component}' not found in model")
                print('Removing from analysis list')

        return valid_components


class ComponentAnalyzer:
    """
    Class for performing component-specific Hessian analysis.

    This class handles the analysis of individual model components,
    following the reference style pattern with clear separation of concerns.
    """

    def __init__(self, config: Optional[HessianConfig] = None):
        """
        Initialize component analyzer.

        Args:
            config: HessianConfig object with analysis parameters
        """
        self.config = config

    def analyze_component(
            self,
            model,
            loss_fn,
            data_batch,
            component_name: str,
            model_type: str = "decoder_only",
            n_components: int = 10,
            device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a specific model component.

        Args:
            model: The transformer model
            loss_fn: Loss function
            data_batch: Batch of data
            component_name: Name of component to analyze
            model_type: Type of model
            n_components: Number of eigenvalues to compute
            device: Device for computation

        Returns:
            Dict of component analysis results
        """
        try:
            # Extract parameters for this component
            params = extract_component_parameters(model, component_name)

            # Define parameter filter for Hessian computation
            param_set = set(params)
            params_filter = lambda p: p in param_set

            # Compute component-specific Hessian eigenvalues
            eigenvalues, _ = get_hessian_eigenvectors(
                model, loss_fn, data_batch,
                device=device,
                num_batches=100,
                n_top_vectors=n_components,
                param_extract_fn=lambda m: [p for p in m.parameters() if p in param_set]
            )

            # Compute detailed metrics
            metrics = HessianMetrics.compute_detailed_hessian_metrics(eigenvalues)

            # Add component-specific info
            metrics["num_params"] = sum(p.numel() for p in params)
            metrics["component_name"] = component_name

            return metrics

        except Exception as e:
            print(f"Error computing Hessian for component {component_name}: {e}")
            return {
                "component_name": component_name,
                "error": str(e),
                "num_params": 0
            }

    def analyze_all_components(
            self,
            model: Transformer,
            loss_fn,
            data_batch,
            model_type: str = "decoder_only",
            component_list: Optional[List[str]] = None,
            n_components: int = 10,
            device: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple model components.

        Args:
            model: The transformer model
            loss_fn: Loss function
            data_batch: Batch of data
            model_type: Type of model
            component_list: List of components to analyze
            n_components: Number of eigenvalues to compute
            device: Device for computation

        Returns:
            Dict mapping component names to their analysis results
        """
        if component_list is None:
            no_fnn = getattr(model, 'no_fnn', False)
            component_list = ComponentSelector.get_standard_components(no_fnn)

        # Validate components exist in model
        valid_components = ComponentSelector.validate_components(model, component_list)

        if not valid_components:
            print("Warning: No valid components found for analysis")
            return {}

        component_results = {}

        for component in valid_components:
            print(f"Analyzing component: {component}")
            result = self.analyze_component(
                model, loss_fn, data_batch, component,
                model_type, n_components, device
            )
            component_results[component] = result

        return component_results

    # def compare_components(
    #         self,
    #         component_results: Dict[str, Dict[str, Any]],
    #         metric_name: str = "max_eigenvalue"
    # ) -> Dict[str, Any]:
    #     """
    #     Compare a specific metric across components.
    #
    #     Args:
    #         component_results: Results from analyze_all_components
    #         metric_name: Name of metric to compare
    #
    #     Returns:
    #         Dict with comparison statistics
    #     """
    #     if not component_results:
    #         return {}
    #
    #     values = {}
    #     for component, results in component_results.items():
    #         if metric_name in results and not isinstance(results[metric_name], str):
    #             values[component] = results[metric_name]
    #
    #     if not values:
    #         return {"error": f"Metric '{metric_name}' not found in results"}
    #
    #     comparison = {
    #         "metric": metric_name,
    #         "values": values,
    #         "max_component": max(values, key=values.get),
    #         "min_component": min(values, key=values.get),
    #         "max_value": max(values.values()),
    #         "min_value": min(values.values()),
    #         "mean_value": sum(values.values()) / len(values),
    #         "component_ranking": sorted(values.items(), key=lambda x: x[1], reverse=True)
    #     }
    #
    #     return comparison
    #
    # def get_component_summary(
    #         self,
    #         component_results: Dict[str, Dict[str, Any]]
    # ) -> Dict[str, Any]:
    #     """
    #     Generate a summary of component analysis results.
    #
    #     Args:
    #         component_results: Results from analyze_all_components
    #
    #     Returns:
    #         Dict with summary statistics
    #     """
    #     if not component_results:
    #         return {}
    #
    #     summary = {
    #         "num_components_analyzed": len(component_results),
    #         "components": list(component_results.keys()),
    #         "total_parameters": sum(
    #             r.get("num_params", 0) for r in component_results.values()
    #         ),
    #         "failed_components": [
    #             comp for comp, results in component_results.items()
    #             if "error" in results
    #         ]
    #     }
    #
    #     # Add metric comparisons for key metrics
    #     key_metrics = [
    #         "max_eigenvalue", "hessian_trace_estimate",
    #         "negative_count", "effective_rank_95"
    #     ]
    #
    #     for metric in key_metrics:
    #         comparison = self.compare_components(component_results, metric)
    #         if "error" not in comparison:
    #             summary[f"{metric}_comparison"] = comparison
    #
    #     return summary


# Legacy compatibility functions
def compute_component_hessians(
        model: Transformer, loss_fn, data_batch, model_type,
        components=None, n_components=10, device=None
):
    """Legacy wrapper for backward compatibility with pre_training.py."""
    analyzer = ComponentAnalyzer()

    if components is None:
        no_fnn = getattr(model, 'no_fnn', False)
        components = ComponentSelector.get_standard_components(no_fnn)

    return analyzer.analyze_all_components(
        model, loss_fn, data_batch, model_type,
        components, n_components, device
    )