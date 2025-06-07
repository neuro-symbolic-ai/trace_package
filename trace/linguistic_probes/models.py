import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List, Dict, Any
import numpy as np


class LinearProbe(nn.Module):
    """
    Simple linear probe for classification tasks.
    This is a basic linear classifier that can be used for POS tagging and semantic role labeling tasks.
    """

    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        self.classifier = nn.Linear(input_dim, num_classes)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the probe."""
        return self.classifier(x)


class MultiLabelProbe(nn.Module):
    """
    Multi-label probe for detecting multiple properties simultaneously.

    In our original paper we named it: MultiLabelSemanticProbe
    """

    def __init__(
            self,
            input_dim: int,
            num_features: int = 12,
            hidden_dim: int = 128,
            lr: float = 1e-3,
            epochs: int = 3,
            device: str = "cpu",
            dropout: float = 0.5,

    ):
        """
        Initialize multi-label probe.

        Args:
            input_dim: Dimension of input features
            num_features: Number of output features/labels
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            epochs: Training epochs
            device: Device for computation
            dropout: Dropout probability
        """
        super().__init__()
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.num_features = num_features
        self.class_weights = None

        # Build the probe network
        self.probe = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid()
        )

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.BCELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the probe."""
        return self.probe(x)

    def compute_class_weights(
            self,
            train_loader: DataLoader,
            penalize_classes: Optional[List[int]] = None,
            penalization_scale: float = 0.1
    ) -> torch.Tensor:
        """
        Compute class weights to handle class imbalance.
        Inspired by https://stackoverflow.com/questions/69783897/compute-class-weight-function-issue-in-sklearn-library-when-used-in-keras-cl.
        Args:
            train_loader: Training data loader
            penalize_classes: Classes to penalize (reduce weight)
            penalization_scale: Scale factor for penalized classes

        Returns:
            Computed class weights
        """
        # Calculate class frequencies
        label_counts = torch.zeros(self.num_features)
        for _, y_batch in train_loader:
            label_counts += y_batch.sum(dim=0)

        # Standard inverse frequency weighting
        weights = 1.0 / (label_counts + 1e-6)
        weights = weights / weights.sum()

        # Apply penalization to frequent classes if specified
        if penalize_classes:
            for class_idx in penalize_classes:
                if class_idx < len(weights):
                    weights[class_idx] *= penalization_scale

        # Renormalize
        weights = weights / weights.sum() * len(weights)

        self.class_weights = weights.to(self.device)
        return weights

    def train_probe(
            self,
            train_loader: DataLoader,
            use_class_weights: bool = True,
            penalize_classes: Optional[List[int]] = None
    ) -> List[float]:
        """
        Train the probe on the provided data.

        Args:
            train_loader: Training data loader
            use_class_weights: Whether to use class weights
            penalize_classes: Classes to penalize during training

        Returns:
            List of training losses per epoch
        """
        self.train()

        # Compute class weights if requested
        if use_class_weights:
            self.compute_class_weights(train_loader, penalize_classes)
            self.criterion = nn.BCELoss(weight=self.class_weights)

        losses = []

        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0

            for x_batch, y_batch in tqdm(train_loader, desc=f"Probe Epoch {epoch + 1}"):
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                self.optimizer.zero_grad()
                preds = self.forward(x_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            epoch_loss = total_loss / num_batches
            losses.append(epoch_loss)
            print(f"[Epoch {epoch + 1}] Probe Loss: {epoch_loss:.4f}")

        return losses

    def evaluate_probe(
            self,
            eval_loader: DataLoader,
            label_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the probe on the provided data.
        note: we added a small value to avoid division by zero in precision/recall calculations.
        we implemented
        Args:
            eval_loader: Evaluation data loader
            label_names: Optional list of label names for reporting
        """
        self.eval()

        if label_names is None:
            label_names = [f"Label {i}" for i in range(self.num_features)]

        total_correct = 0
        total_tokens = 0
        num_labels = len(label_names)

        # Initialize metric trackers
        true_positives = torch.zeros(num_labels)
        false_positives = torch.zeros(num_labels)
        false_negatives = torch.zeros(num_labels)
        label_counts = torch.zeros(num_labels)

        with torch.no_grad():
            for x_batch, y_batch in eval_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                preds = self.forward(x_batch)
                predicted = (preds > 0.5).float()

                # Overall accuracy
                total_correct += (predicted == y_batch).sum().item()
                total_tokens += y_batch.numel()

                # Per-label metrics
                label_counts += y_batch.sum(dim=0).cpu()
                true_positives += ((predicted == 1) & (y_batch == 1)).sum(dim=0).cpu()
                false_positives += ((predicted == 1) & (y_batch == 0)).sum(dim=0).cpu()
                false_negatives += ((predicted == 0) & (y_batch == 1)).sum(dim=0).cpu()

        # Calculate metrics
        overall_accuracy = total_correct / total_tokens

        per_label_metrics = {}
        for i in range(num_labels):
            precision = true_positives[i] / (true_positives[i] + false_positives[i] + 1e-6)
            recall = true_positives[i] / (true_positives[i] + false_negatives[i] + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            accuracy = true_positives[i] / (label_counts[i] + 1e-6)

            per_label_metrics[label_names[i]] = {
                'count': int(label_counts[i]),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            }

        results = {
            'overall_accuracy': overall_accuracy,
            'per_label_metrics': per_label_metrics,
            'label_names': label_names
        }

        return results

    def print_evaluation_report(self, results: Dict[str, Any]) -> None:
        """Print a formatted evaluation report."""
        print(f"\nProbe Evaluation Report:")
        print(f"Overall accuracy: {results['overall_accuracy']:.4f}\n")

        for label_name, metrics in results['per_label_metrics'].items():
            print(
                f"{label_name:<15} | "
                f"Count: {metrics['count']} | "
                f"Acc: {metrics['accuracy']:.3f} | "
                f"Precision: {metrics['precision']:.3f} | "
                f"Recall: {metrics['recall']:.3f} | "
                f"F1: {metrics['f1']:.3f}"
            )

    def save(self, path: str) -> None:
        """Save the probe state."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the probe state."""
        self.load_state_dict(torch.load(path, map_location=self.device))


