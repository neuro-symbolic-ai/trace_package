import torch
import numpy as np
from typing import Dict, Any, Optional
from torch.autograd import grad
from .config import HessianConfig
from .utils import compute_loss, get_hessian_eigenvectors
from .metrics import HessianMetrics
from .components import ComponentAnalyzer
from .visualization import HessianVisualizer


class HessianAnalyzer:
    """
    Main class for performing comprehensive Hessian analysis on transformer models.

    This class provides a high-level interface for all Hessian analysis functionality,
    following the reference style pattern with clear method organization.
    """

    def __init__(self, config: Optional[HessianConfig] = None):
        self.config = config or HessianConfig.default()
        self.component_analyzer = ComponentAnalyzer(config)
        self.visualizer = HessianVisualizer(
            self.config
        ) if getattr(self.config, 'save_visualizations', True) else None

    def analyze_step(
            self,
            model,
            loss_fn,
            train_batch,
            val_batch=None,
            model_type: str = "decoder_only",
            step: int = 0
    ) -> Dict[str, Any]:
        """
        Perform comprehensive Hessian analysis for a `single` training step.
        Typically called during training loops to analyze model behavior.
        """
        results = {"step": step}
        print(f"Analyzing Hessian at step {step}...")
        try:
            # Basic Hessian eigenvalue analysis
            print("Computing Hessian eigenvalues and eigenvectors...")
            # moving the batch to the correct device
            device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
            train_batch = {k: v.to(device) for k, v in train_batch.items()}
            eigenvalues, eigenvectors = get_hessian_eigenvectors(
                model, loss_fn, train_batch,
                device=self.config.device,
                num_batches=self.config.num_batches,
                n_top_vectors=self.config.n_components
            )
            # Compute detailed metrics
            hessian_metrics = HessianMetrics.compute_detailed_hessian_metrics(eigenvalues)
            results["hessian"] = hessian_metrics

            # Component-specific analysis
            if self.config.track_component_hessian:
                component_results = self.component_analyzer.analyze_all_components(
                    model, loss_fn, train_batch, model_type,
                    self.config.component_list, self.config.n_components,
                    self.config.device
                )
                results["components"] = component_results
                # print("Component-specific Hessian analysis completed.")
                # print(f"Component results: {component_results}")

            # Gradient-Hessian alignment analysis
            if self.config.track_gradient_alignment:
                alignment_results = self.compute_gradient_alignment(
                    model, loss_fn, train_batch, model_type,
                    eigenvalues, eigenvectors
                )
                results["alignment"] = alignment_results

            # Train-val landscape divergence
            if self.config.track_train_val_landscape_divergence and val_batch is not None:
                divergence_results = self.compute_train_val_divergence(
                    model, loss_fn, train_batch, val_batch, model_type
                )
                results["train_val_divergence"] = divergence_results

        except Exception as e:
            print(f"Error in Hessian analysis at step {step}: {e}")
            results["error"] = str(e)

        return results

    def compute_gradient_alignment(
            self,
            model,
            loss_fn,
            data_batch,
            model_type: str,
            eigenvalues: np.ndarray,
            eigenvectors: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute alignment between gradient and Hessian eigenvectors.
        """
        # Get model parameters
        model_params = [p for p in model.parameters() if p.requires_grad]

        # Compute loss and gradient
        loss = compute_loss(model, loss_fn, data_batch, model_type)
        grad_params = torch.autograd.grad(loss, model_params, create_graph=True)
        flat_grad = torch.cat([g.flatten() for g in grad_params])
        grad_norm = torch.norm(flat_grad)
        normalized_grad = flat_grad / (grad_norm + 1e-10)

        # Compute Hessian-gradient product
        Hg = torch.autograd.grad(grad_params[0].sum(), model_params, retain_graph=True)
        flat_Hg = torch.cat([g.flatten() for g in Hg])
        Hg_norm = torch.norm(flat_Hg)
        normalized_Hg = flat_Hg / (Hg_norm + 1e-10)

        # Alignment between gradient and Hg
        grad_Hg_alignment = torch.abs(torch.dot(normalized_grad, normalized_Hg)).item()

        # Compute alignment metrics using HessianMetrics
        flat_grad_np = normalized_grad.cpu().detach().numpy()
        alignment_metrics = HessianMetrics.compute_alignment_metrics(
            flat_grad_np, eigenvalues, eigenvectors,
            grad_norm.item(), Hg_norm.item()
        )

        # Add the grad-Hg alignment
        alignment_metrics["grad_Hg_alignment"] = float(grad_Hg_alignment)

        return alignment_metrics

    def compute_train_val_divergence(
            self,
            model,
            loss_fn,
            train_batch,
            val_batch,
            model_type: str
    ) -> Dict[str, Any]:
        """
        Compare Hessian properties on training vs validation data.

        Args:
            model: The transformer model
            loss_fn: Loss function
            train_batch: Training data batch
            val_batch: Validation data batch
            model_type: Type of model

        Returns:
            Dict with memorization signals
        """
        # Compute Hessian metrics on training data
        train_eigenvalues, _ = get_hessian_eigenvectors(
            model, loss_fn, train_batch,
            device=self.config.device,
            num_batches=self.config.num_batches,
            n_top_vectors=self.config.n_components
        )
        train_metrics = HessianMetrics.compute_detailed_hessian_metrics(train_eigenvalues)

        # Compute Hessian metrics on validation data
        val_eigenvalues, _ = get_hessian_eigenvectors(
            model, loss_fn, val_batch,
            device=self.config.device,
            num_batches=self.config.num_batches,
            n_top_vectors=self.config.n_components
        )
        val_metrics = HessianMetrics.compute_detailed_hessian_metrics(val_eigenvalues)

        # Compute memorization signals
        memorization_signals = HessianMetrics.compute_memorization_signals(
            train_metrics, val_metrics, train_eigenvalues, val_eigenvalues
        )

        return memorization_signals

    @classmethod
    def from_config(cls, config: HessianConfig) -> 'HessianAnalyzer':
        """
        Create analyzer from configuration.

        Args:
            config: HessianConfig object

        Returns:
            Initialized HessianAnalyzer
        """
        return cls(config)




