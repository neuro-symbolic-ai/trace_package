import numpy as np
import scipy.stats
from typing import Dict, Any


class HessianMetrics:
    """
    Class for computing detailed metrics from Hessian eigenvalues.
    """

    @staticmethod
    def compute_detailed_hessian_metrics(eigenvalues) -> Dict[str, Any]:
        """
        Compute detailed metrics from Hessian eigenvalues.

        Args:
            eigenvalues: Array of eigenvalues

        Returns:
            Dict of metrics derived from eigenvalues
        """
        # Convert to numpy for analysis
        eigenvalues = np.array(eigenvalues)
        abs_eigenvalues = np.abs(eigenvalues)

        # Calculate basic metrics
        metrics = {
            "top_eigenvalues": eigenvalues.tolist(),
            "max_eigenvalue": float(np.max(eigenvalues)),
            "min_eigenvalue": float(np.min(eigenvalues)),
            "mean_eigenvalue": float(np.mean(eigenvalues)),
            "median_eigenvalue": float(np.median(eigenvalues)),
            "negative_count": int(np.sum(eigenvalues < 0)),
            "positive_count": int(np.sum(eigenvalues > 0)),
            "zero_count": int(np.sum(np.abs(eigenvalues) < 1e-6)),
            "hessian_trace_estimate": float(np.sum(eigenvalues)),  # Overall curvature
        }

        # Add advanced metrics
        metrics.update(HessianMetrics._compute_condition_number(abs_eigenvalues))
        metrics.update(HessianMetrics._compute_eigenvalue_statistics(eigenvalues))
        metrics.update(HessianMetrics._compute_decay_rate(abs_eigenvalues))
        metrics.update(HessianMetrics._compute_effective_rank(abs_eigenvalues))
        metrics.update(HessianMetrics._compute_complexity_score(metrics))

        return metrics

    @staticmethod
    def _compute_condition_number(abs_eigenvalues: np.ndarray) -> Dict[str, float]:
        """Compute condition number from absolute eigenvalues."""
        if np.min(abs_eigenvalues) > 0:
            condition_number = float(np.max(abs_eigenvalues) / np.min(abs_eigenvalues))
        else:
            condition_number = float('inf')

        return {"condition_number": condition_number}

    @staticmethod
    def _compute_eigenvalue_statistics(eigenvalues: np.ndarray) -> Dict[str, float]:
        """Compute statistical properties of eigenvalues."""
        metrics = {}

        if len(eigenvalues) > 1:
            metrics["eigenvalue_std"] = float(np.std(eigenvalues))
            try:
                metrics["eigenvalue_skew"] = float(scipy.stats.skew(eigenvalues))
            except:
                metrics["eigenvalue_skew"] = 0.0
        else:
            metrics["eigenvalue_std"] = 0.0
            metrics["eigenvalue_skew"] = 0.0

        return metrics

    @staticmethod
    def _compute_decay_rate(abs_eigenvalues: np.ndarray) -> Dict[str, float]:
        """Compute eigenvalue decay rate through exponential fitting."""
        if len(abs_eigenvalues) <= 2:
            return {"eigenvalue_decay_rate": 0.0}

        sorted_eigs = np.sort(abs_eigenvalues)[::-1]
        log_eigs = np.log(sorted_eigs + 1e-10) # we add a small constant to avoid log(0)
        indices = np.arange(len(sorted_eigs))

        # Linear fit to log values gives exponential decay rate
        if not np.all(log_eigs == log_eigs[0]):  # Check if all eigenvalues are identical
            try:
                slope, _, _, _, _ = scipy.stats.linregress(indices, log_eigs)
                decay_rate = float(-slope)  # Negative because we want decay rate
            except:
                decay_rate = 0.0
        else:
            decay_rate = 0.0

        return {"eigenvalue_decay_rate": decay_rate}

    @staticmethod
    def _compute_effective_rank(abs_eigenvalues: np.ndarray) -> Dict[str, float]:
        """Compute effective rank metrics."""
        metrics = {}

        if np.sum(abs_eigenvalues) > 0:
            # Effective rank (95% energy)
            sorted_eigs = np.sort(abs_eigenvalues)[::-1]
            cumulative_energy = np.cumsum(sorted_eigs) / np.sum(sorted_eigs)
            effective_rank_95 = int(np.searchsorted(cumulative_energy, 0.95) + 1)

            # Effective rank (entropy-based)
            # r_eff = exp(-∑ p_i * log(p_i)) where p_i = |λ_i| / ∑|λ_j|
            p_i = abs_eigenvalues / np.sum(abs_eigenvalues)
            effective_rank_entropy = float(np.exp(-np.sum(p_i * np.log(p_i + 1e-10))))

            metrics["effective_rank_95"] = effective_rank_95
            metrics["effective_rank_entropy"] = effective_rank_entropy
        else:
            metrics["effective_rank_95"] = 0
            metrics["effective_rank_entropy"] = 0.0

        return metrics

    @staticmethod
    def _compute_complexity_score(metrics: Dict[str, Any]) -> Dict[str, float]:
        """Compute complexity score from existing metrics."""
        if metrics["effective_rank_entropy"] > 0:
            complexity_score = float(
                metrics["hessian_trace_estimate"] / metrics["effective_rank_entropy"]
            )
        else:
            complexity_score = float('inf')

        return {"complexity_score": complexity_score}

    @staticmethod
    def compute_alignment_metrics(
            normalized_grad: np.ndarray,
            eigenvalues: np.ndarray,
            eigenvectors: np.ndarray,
            grad_norm: float,
            hg_norm: float
    ) -> Dict[str, Any]:
        """
        Compute gradient-Hessian alignment metrics.

        Args:
            normalized_grad: Normalized gradient vector
            eigenvalues: Hessian eigenvalues
            eigenvectors: Hessian eigenvectors
            grad_norm: L2 norm of gradient
            hg_norm: L2 norm of Hessian-gradient product

        Returns:
            Dict of alignment metrics
        """
        # Calculate alignment with each eigenvector
        alignments = []
        for i, eigenvector in enumerate(eigenvectors):
            # Normalize the eigenvector (should be normalized already, but ensuring)
            normalized_eigenvector = eigenvector / np.linalg.norm(eigenvector)
            # Compute alignment as absolute dot product (cosine similarity)
            alignment = abs(np.dot(normalized_grad, normalized_eigenvector))
            alignments.append(float(alignment))

        # Compute weighted alignment score
        eigenvalue_weights = np.abs(eigenvalues) / np.sum(np.abs(eigenvalues))
        weighted_alignment = np.sum(np.array(alignments) * eigenvalue_weights)

        return {
            # "individual_alignments": alignments,
            "weighted_alignment": float(weighted_alignment),
            "grad_norm": float(grad_norm),
            "Hg_norm": float(hg_norm),
            "grad_Hg_ratio": float(hg_norm / (grad_norm + 1e-10)),
            "top_eigenvalues": eigenvalues.tolist()
        }

    @staticmethod
    def compute_memorization_signals(
            train_metrics: Dict[str, Any],
            val_metrics: Dict[str, Any],
            train_eigenvalues: np.ndarray,
            val_eigenvalues: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute memorization detection signals from train/val Hessian comparison.

        Args:
            train_metrics: Hessian metrics on training data
            val_metrics: Hessian metrics on validation data
            train_eigenvalues: Training eigenvalues for distribution comparison
            val_eigenvalues: Validation eigenvalues for distribution comparison

        Returns:
            Dict of memorization signals
        """
        # Calculate divergence metrics
        memorization_signals = {
            "trace_ratio": train_metrics["hessian_trace_estimate"] / (val_metrics["hessian_trace_estimate"] + 1e-10),
            "max_eigenvalue_ratio": train_metrics["max_eigenvalue"] / (val_metrics["max_eigenvalue"] + 1e-10),
            "negative_eigenvalue_diff": train_metrics["negative_count"] - val_metrics["negative_count"],
            "effective_rank_diff": train_metrics["effective_rank_95"] - val_metrics["effective_rank_95"],
            "effective_rank_entropy_diff": train_metrics["effective_rank_entropy"] - val_metrics[
                "effective_rank_entropy"],
            "train_decay_rate": train_metrics["eigenvalue_decay_rate"],
            "val_decay_rate": val_metrics["eigenvalue_decay_rate"],
            "decay_rate_ratio": train_metrics["eigenvalue_decay_rate"] / (val_metrics["eigenvalue_decay_rate"] + 1e-10),
            "train_complexity_score": train_metrics["complexity_score"],
            "val_complexity_score": val_metrics["complexity_score"],
        }

        # Compute eigenvalue distribution overlap
        overlap = HessianMetrics._compute_distribution_overlap(train_eigenvalues, val_eigenvalues)
        memorization_signals["eigenvalue_distribution_overlap"] = overlap

        # Compute interpretable memorization score
        train_val_score = HessianMetrics._compute_train_val_divergence_score(memorization_signals)
        memorization_signals["train_val_landscape_divergence_score"] = train_val_score

        return memorization_signals

    @staticmethod
    def _compute_distribution_overlap(train_eigenvalues: np.ndarray, val_eigenvalues: np.ndarray) -> float:
        """Compute eigenvalue distribution overlap coefficient."""
        train_sorted = np.sort(np.abs(train_eigenvalues))
        val_sorted = np.sort(np.abs(val_eigenvalues))

        # Normalize to [0,1] range for comparison
        if np.max(train_sorted) > 0:
            train_normalized = train_sorted / np.max(train_sorted)
        else:
            train_normalized = train_sorted

        if np.max(val_sorted) > 0:
            val_normalized = val_sorted / np.max(val_sorted)
        else:
            val_normalized = val_sorted

        # Calculate overlap as histogram intersection
        bin_edges = np.linspace(0, 1, 20)
        train_hist, _ = np.histogram(train_normalized, bins=bin_edges, density=True)
        val_hist, _ = np.histogram(val_normalized, bins=bin_edges, density=True)
        overlap = np.sum(np.minimum(train_hist, val_hist)) / np.sum(train_hist)

        return float(overlap)

    @staticmethod
    def _compute_train_val_divergence_score(memorization_signals: Dict[str, float]) -> float:
        """Compute interpretable train-val landscape divergence score."""
        components = [
            np.log1p(memorization_signals["trace_ratio"]),
            np.log1p(memorization_signals["max_eigenvalue_ratio"]),
            np.abs(memorization_signals["negative_eigenvalue_diff"]),
            (1 - memorization_signals["eigenvalue_distribution_overlap"])
        ]

        score = sum(components) / len(components)
        return float(score)


# Legacy compatibility function
def compute_detailed_hessian_metrics(eigenvalues):
    """Legacy wrapper for backward compatibility."""
    return HessianMetrics.compute_detailed_hessian_metrics(eigenvalues)