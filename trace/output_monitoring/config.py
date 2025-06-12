# output_monitoring/config.py

from dataclasses import dataclass
from typing import Optional, List, Dict, Union


@dataclass
class OutputMonitoringConfig:
    """
    Configuration class for output monitoring analysis.

    This dataclass contains all hyperparameters and configuration options
    for performing POS and semantic role monitoring on transformer models.
    """

    # Model architecture
    model_type: str = "decoder_only"

    # POS monitoring settings
    track_pos_performance: bool = True
    pos_granularity: str = 'basic'  # 'basic' (8 categories) or 'detailed' (14 categories)

    # Semantic role monitoring settings
    track_semantic_roles: bool = False
    semantic_granularity: str = 'basic'  # 'basic' (8 roles) or 'detailed' (15 roles)

    # Visualization and saving
    save_visualizations: bool = True  # Whether to save visualizations
    log_dir: Optional[str] = None

    device: str = "cpu"  # Device to run monitoring on (e.g., 'cuda' or 'cpu')


    def __post_init__(self):
        """Validate configuration parameters."""
        if self.model_type not in ['encoder_only', 'decoder_only', 'encoder_decoder']:
            raise ValueError(f"Invalid model_type: {self.model_type}")

        if self.pos_granularity not in ['basic', 'detailed']:
            raise ValueError(f"Invalid pos_granularity: {self.pos_granularity}")

        if self.semantic_granularity not in ['basic', 'detailed']:
            raise ValueError(f"Invalid semantic_granularity: {self.semantic_granularity}")

    @classmethod
    def default(cls) -> 'OutputMonitoringConfig':
        """Create a default configuration for output monitoring."""
        return cls(
            model_type="decoder_only",
            track_pos_performance=True,
            pos_granularity='basic',
            track_semantic_roles=True,
            semantic_granularity='basic'
        )