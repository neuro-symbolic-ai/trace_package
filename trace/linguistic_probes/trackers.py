import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

from .taggers import POSTagger, SemanticTagger
from .config import LinguisticProbesConfig


class BasePerformanceTracker:
    """
    Base class for tracking model performance metrics by categories.
    """

    def __init__(self, tokenizer, log_dir: Optional[str] = None, config: Optional[LinguisticProbesConfig] = None):
        """
        Initialize performance tracker.

        Args:
            tokenizer: The tokenizer used by the model
            log_dir: Directory to save analysis logs and visualizations
            config: Configuration object
        """
        self.tokenizer = tokenizer
        self.log_dir = log_dir
        self.config = config or LinguisticProbesConfig.default()

        # Metrics storage - using (step, value) tuples for consistency
        self.accuracy = defaultdict(list)
        self.loss = defaultdict(list)
        self.sample_counts = defaultdict(int)

        # Create log directory if specified
        if log_dir:
            import os
            self.plots_dir = os.path.join(log_dir, self.get_analysis_type())
            os.makedirs(self.plots_dir, exist_ok=True)

    def get_analysis_type(self) -> str:
        """Get the type of analysis (to be overridden by subclasses)."""
        return "base_analysis"

    def get_tagger(self):
        """Get the appropriate tagger (to be overridden by subclasses)."""
        raise NotImplementedError

    def get_categories(self) -> Dict[str, int]:
        """Get category mappings (to be overridden by subclasses)."""
        raise NotImplementedError

    def process_batch(self, batch: Dict[str, torch.Tensor], logits: torch.Tensor) -> Dict[str, Any]:
        """
        Process a batch to collect category-specific performance metrics.

        Args:
            batch: The input batch dictionary
            logits: Model output logits

        Returns:
            Dictionary with category-specific metrics
        """
        # Extract relevant batch info
        input_ids = batch['input_ids'].cpu()
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).cpu()
        labels = batch.get('labels', input_ids).cpu()

        # Get predictions
        predictions = torch.argmax(logits, dim=-1).cpu()

        # Initialize metrics
        metrics = {
            'accuracy_by_category': defaultdict(list),
            'count_by_category': defaultdict(int)
        }

        # Get tagger
        tagger = self.get_tagger()

        # Process each sequence in the batch
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            # Decode input sequence to text
            input_sequence = input_ids[i].tolist()
            input_text = self.tokenizer.decode(input_sequence)

            # Get category tags for the sequence
            token_category_pairs = tagger.tag_text(input_text)

            # Create a mapping from position to category
            category_by_position = {}
            token_idx = 0
            for j, token_id in enumerate(input_sequence):
                if token_id == getattr(self.tokenizer, 'pad_token_id', 0):
                    continue

                # Skip special tokens or assign them SPECIAL category
                special_token_ids = [
                    getattr(self.tokenizer, 'cls_token_id', None),
                    getattr(self.tokenizer, 'sep_token_id', None),
                    getattr(self.tokenizer, 'bos_token_id', None),
                    getattr(self.tokenizer, 'eos_token_id', None)
                ]
                if token_id in [tid for tid in special_token_ids if tid is not None]:
                    category_by_position[j] = 'SPECIAL'
                else:
                    # Try to match with category tags
                    if token_idx < len(token_category_pairs):
                        category_by_position[j] = token_category_pairs[token_idx][1]
                        token_idx += 1
                    else:
                        category_by_position[j] = 'UNKNOWN'

            # Gather metrics by category
            for pos in range(len(input_sequence)):
                # Skip padded positions
                if attention_mask[i, pos] == 0:
                    continue

                # Skip positions where labels are ignored (-100)
                if labels[i, pos] == -100:
                    continue

                # Get category for this position
                category = category_by_position.get(pos, 'UNKNOWN')

                # Calculate accuracy
                correct = (predictions[i, pos] == labels[i, pos]).item()
                metrics['accuracy_by_category'][category].append(float(correct))

                # Increment count
                metrics['count_by_category'][category] += 1

        # Calculate aggregate metrics
        results = {
            f'{self.get_analysis_type()}_accuracy': {},
            f'{self.get_analysis_type()}_counts': dict(metrics['count_by_category'])
        }

        for category, acc_list in metrics['accuracy_by_category'].items():
            if acc_list:
                results[f'{self.get_analysis_type()}_accuracy'][category] = sum(acc_list) / len(acc_list)

        return results

    def update_epoch_metrics(self, epoch_metrics: Dict[str, Dict[str, float]], step_number: int):
        """
        Update metrics with the current step number.
        """
        accuracy_key = f'{self.get_analysis_type()}_accuracy'
        counts_key = f'{self.get_analysis_type()}_counts'

        # Get all categories we've seen so far
        all_categories = set(self.accuracy.keys())
        all_categories.update(epoch_metrics.get(accuracy_key, {}).keys())

        # Update accuracy for all known categories
        for category in all_categories:
            # If we have this category in the current batch, add its accuracy
            if category in epoch_metrics.get(accuracy_key, {}):
                acc = epoch_metrics[accuracy_key][category]
                self.accuracy[category].append((step_number, acc))
            # Otherwise, if we've seen this category before, add NaN for this step
            elif category in self.accuracy and len(self.accuracy[category]) > 0:
                self.accuracy[category].append((step_number, float('nan')))

        # Update sample counts
        for category, count in epoch_metrics.get(counts_key, {}).items():
            self.sample_counts[category] += count

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a summary report of category-specific performance.

        Returns:
            Dictionary with summary statistics
        """
        report = {
            'final_accuracy_by_category': {},
            'improvement_by_category': {},
            'hardest_categories': [],
            'easiest_categories': [],
            'sample_counts': dict(self.sample_counts)
        }

        # Calculate final accuracy and improvement
        for category, data_points in self.accuracy.items():
            if len(data_points) >= 2:
                # Extract accuracy values for first and last points
                first_acc = data_points[0][1]
                last_acc = data_points[-1][1]

                report['final_accuracy_by_category'][category] = last_acc
                report['improvement_by_category'][category] = last_acc - first_acc

        # Identify hardest and easiest categories
        if report['final_accuracy_by_category']:
            sorted_acc = sorted(report['final_accuracy_by_category'].items(), key=lambda x: x[1])
            report['hardest_categories'] = sorted_acc[:3]
            report['easiest_categories'] = sorted_acc[-3:]

        return report


class POSPerformanceTracker(BasePerformanceTracker):
    """
    Tracks model performance metrics separated by part-of-speech categories.
    """

    def __init__(self, tokenizer, log_dir: Optional[str] = None, config: Optional[LinguisticProbesConfig] = None):
        """
        Initialize POS performance tracker.

        Args:
            tokenizer: The tokenizer used by the model
            log_dir: Directory to save POS analysis logs and visualizations
            config: Configuration object
        """
        super().__init__(tokenizer, log_dir, config)
        self.tagger = POSTagger(granularity=self.config.pos_granularity)

    def get_analysis_type(self) -> str:
        """Get the type of analysis."""
        return "pos_analysis"

    def get_tagger(self):
        """Get the POS tagger."""
        return self.tagger

    def get_categories(self) -> Dict[str, int]:
        """Get POS category mappings."""
        return self.config.get_pos_categories()


class SemanticPerformanceTracker(BasePerformanceTracker):
    """
    Tracks model performance metrics separated by semantic role categories.
    """

    def __init__(self, tokenizer, log_dir: Optional[str] = None, config: Optional[LinguisticProbesConfig] = None):
        """
        Initialize semantic performance tracker.

        Args:
            tokenizer: The tokenizer used by the model
            log_dir: Directory to save semantic analysis logs and visualizations
            config: Configuration object
        """
        super().__init__(tokenizer, log_dir, config)
        self.tagger = SemanticTagger(granularity=self.config.semantic_granularity)

    def get_analysis_type(self) -> str:
        """Get the type of analysis."""
        return "semantic_analysis"

    def get_tagger(self):
        """Get the semantic tagger."""
        return self.tagger

    def get_categories(self) -> Dict[str, int]:
        """Get semantic category mappings."""
        return self.config.get_semantic_categories()


# Legacy compatibility aliases and wrappers
class POSPerformanceTracker_Legacy(POSPerformanceTracker):
    """Legacy compatibility wrapper for POSPerformanceTracker."""

    def __init__(self, tokenizer, log_dir: str = None, pos_granularity: str = 'basic'):
        config = LinguisticProbesConfig.default()
        config.pos_granularity = pos_granularity
        config.log_dir = log_dir
        super().__init__(tokenizer, log_dir, config)

        # Legacy attributes for compatibility
        self.pos_accuracy = self.accuracy
        self.pos_sample_counts = self.sample_counts
        self.pos_plots_dir = self.plots_dir if hasattr(self, 'plots_dir') else None

    def _map_nltk_tag(self, nltk_tag: str) -> str:
        """Legacy method for NLTK tag mapping."""
        return self.tagger._map_nltk_tag(nltk_tag)

    def _tokenize_and_tag(self, text: str) -> List[Tuple[str, str]]:
        """Legacy method for tokenization and tagging."""
        return self.tagger.tag_text(text)


class SemanticRolePerformanceTracker(SemanticPerformanceTracker):
    """Legacy compatibility wrapper for SemanticPerformanceTracker."""

    def __init__(self, tokenizer, log_dir: str = None):
        config = LinguisticProbesConfig.default()
        config.semantic_granularity = 'reduced'
        config.log_dir = log_dir
        super().__init__(tokenizer, log_dir, config)

        # Legacy attributes for compatibility
        self.semantic_accuracy = self.accuracy
        self.semantic_sample_counts = self.sample_counts
        self.semantic_plots_dir = self.plots_dir if hasattr(self, 'plots_dir') else None

    def _tokenize_and_tag_semantic(self, text: str) -> List[Tuple[str, str]]:
        """Legacy method for semantic tagging."""
        return self.tagger.tag_text(text)