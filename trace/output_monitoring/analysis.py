# output_monitoring/analysis.py

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import torch

from .config import OutputMonitoringConfig
from .utils import extract_output_monitoring_data
from ..linguistic_probes import POSTagger, SemanticTagger

class OutputMonitoringAnalyzer:
    """
    Class for analyzing output monitoring of transformer models.
    todo: should I change this to an interface? like probes?
    """

    def __init__(self, config: Optional[OutputMonitoringConfig] = None):
        self.config = config or OutputMonitoringConfig.default()

        # Initialize tracking storage
        self.pos_history = {}
        self.semantic_history = {}

        # Initialize sample counters
        self.pos_sample_counts = {}
        self.semantic_sample_counts = {}

        if self.config.track_pos_performance:
            self.pos_tagger = POSTagger(
                granularity=self.config.pos_granularity,
            )
        else:
            self.pos_tagger = None
        if self.config.track_semantic_roles:
            self.semantic_tagger = SemanticTagger(
                granularity=self.config.semantic_granularity,
            )
        else:
            self.semantic_tagger = None

        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

    def analyze(self,
                batch: Dict[str, Any],
                outputs,
                tokenizer,
                step: int = 0) -> Dict[str, Any]:
        """
        Analyze output monitoring for the given batch.

        Args:
            model: The transformer model
            batch: Input batch dictionary - # contains true labels
            outputs: Model outputs (logits)
            tokenizer: Tokenizer for decoding
            step: Current training step

        Returns:
            Dictionary containing monitoring results
        """
        if not (self.config.track_pos_performance or self.config.track_semantic_roles):
            return {}

        # Extract monitoring data
        monitoring_data = extract_output_monitoring_data(
            batch, outputs, tokenizer, self.config,
            pos_tagger=self.pos_tagger,
            semantic_tagger=self.semantic_tagger
        )
        print(f"Monitoring data at step {step}: {monitoring_data}")

        results = {'step': step}

        # Process POS results
        if 'pos_accuracy' in monitoring_data:
            pos_results = monitoring_data['pos_accuracy']
            results['pos_accuracy'] = pos_results

            # Update history
            if step not in self.pos_history:
                self.pos_history[step] = {}
            self.pos_history[step].update(pos_results)

            # Update sample counts
            for pos_tag in pos_results:
                if pos_tag not in self.pos_sample_counts:
                    self.pos_sample_counts[pos_tag] = 0
                self.pos_sample_counts[pos_tag] += 1

        # Process semantic results
        if 'semantic_accuracy' in monitoring_data:
            semantic_results = monitoring_data['semantic_accuracy']
            results['semantic_accuracy'] = semantic_results

            # Update history
            if step not in self.semantic_history:
                self.semantic_history[step] = {}
            self.semantic_history[step].update(semantic_results)

            # Update sample counts
            for semantic_tag in semantic_results:
                if semantic_tag not in self.semantic_sample_counts:
                    self.semantic_sample_counts[semantic_tag] = 0
                self.semantic_sample_counts[semantic_tag] += 1

        return results

    def get_pos_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for POS monitoring.

        Returns:
            Dictionary with POS summary statistics
        """
        if not self.pos_history:
            return {}

        # Calculate average accuracy per POS tag
        pos_averages = {}
        for step_data in self.pos_history.values():
            for pos_tag, accuracy in step_data.items():
                if pos_tag not in pos_averages:
                    pos_averages[pos_tag] = []
                pos_averages[pos_tag].append(accuracy)

        # Compute final averages
        final_averages = {
            pos_tag: sum(accuracies) / len(accuracies)
            for pos_tag, accuracies in pos_averages.items()
        }

        # Sort by performance
        sorted_pos = sorted(final_averages.items(), key=lambda x: x[1])

        return {
            'average_accuracies': final_averages,
            'hardest_categories': sorted_pos[:3],  # Bottom 3
            'easiest_categories': sorted_pos[-3:],  # Top 3
            'sample_counts': self.pos_sample_counts,
            'total_steps': len(self.pos_history)
        }

    def get_semantic_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for semantic role monitoring.

        Returns:
            Dictionary with semantic summary statistics
        """
        if not self.semantic_history:
            return {}

        # Calculate average accuracy per semantic tag
        semantic_averages = {}
        for step_data in self.semantic_history.values():
            for semantic_tag, accuracy in step_data.items():
                if semantic_tag not in semantic_averages:
                    semantic_averages[semantic_tag] = []
                semantic_averages[semantic_tag].append(accuracy)

        # Compute final averages
        final_averages = {
            semantic_tag: sum(accuracies) / len(accuracies)
            for semantic_tag, accuracies in semantic_averages.items()
        }

        # Sort by performance
        sorted_semantic = sorted(final_averages.items(), key=lambda x: x[1])

        return {
            'average_accuracies': final_averages,
            'hardest_categories': sorted_semantic[:3],  # Bottom 3
            'easiest_categories': sorted_semantic[-3:],  # Top 3
            'sample_counts': self.semantic_sample_counts,
            'total_steps': len(self.semantic_history)
        }

    def get_full_results(self) -> Dict[str, Any]:
        """
        Get all monitoring results.

        Returns:
            Dictionary with complete monitoring data
        """
        results = {
            'config': self.config,
            'pos_history': self.pos_history,
            'semantic_history': self.semantic_history
        }

        if self.config.track_pos_performance:
            results['pos_summary'] = self.get_pos_summary()

        if self.config.track_semantic_roles:
            results['semantic_summary'] = self.get_semantic_summary()

        return results