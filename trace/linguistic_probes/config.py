from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple


@dataclass
class LinguisticProbesConfig:
    """
    Configuration class for linguistic probes.
    """

    # Probe architecture
    input_dim: int = None  # Input dimension for the probe (e.g., BERT hidden size)
    probe_type: str = "multilabel"  # 'linear' or 'multilabel'
    num_classes: int = 8  # Number of classes for classification probes
    num_pos_classes: int = 8  # Number of POS classes (for basic granularity)
    num_semantic_classes: int = 8  # Number of semantic classes (for basic granularity)
    hidden_dim: int = 128  # Hidden dimension for MLP probes
    lr: float = 0.001  # Learning rate for training probes
    dropout: float = 0.1  # Dropout rate for MLP probes
    criterion: str = "cross_entropy"

    # Training parameters
    batch_size: int = 32
    device: str = "cpu"  # Device to run probes on (e.g., 'cuda' or 'cpu')
    epochs: int = 10

    # Analysis toggles
    track_pos: bool = True
    track_semantic: bool = True

    # Special options - which layers to probe
    layer_indices: Optional[Union[List[int], Dict[str, List[int]]]] = None
    probe_all_layers: bool = False

    # Data processing settings
    pos_granularity: str = 'basic'  # 'basic' or 'detailed' (similar to original paper testing - based on our retsults this must be checked according to the tags distribution)
    semantic_granularity: str = 'basic'  # 'basic' or 'detailed'

    # Class weighting
    use_class_weights: bool = True
    penalize_frequent_classes: bool = True
    penalization_scale: float = 0.1
    penalized_classes: Optional[List[int]] = None

    # Output settings
    save_probes: bool = True
    save_visualizations: bool = True
    save_path: Optional[str] = None
    log_dir: Optional[str] = None
    probe_load_path: Union[Optional[str], Dict[Tuple[int, str], str]] = None  # Path to load existing probes if needed
    show_plots: bool = False  # Whether to show plots during training


    def __post_init__(self):
        """Validate configuration parameters."""
        if self.probe_type not in ['linear', 'multilabel']:
            raise ValueError(f"Unknown probe_type: {self.probe_type}")

        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("Dropout rate must be in [0.0, 1.0)")

            # Set default penalized classes for POS (NOUN is typically class 0)
        if self.penalized_classes is None and self.penalize_frequent_classes:
            self.penalized_classes = [0, 1]  # Typically NOUN and VERB

    @classmethod
    def default(cls) -> 'LinguisticProbesConfig':
        """Create a default configuration for linguistic probes."""
        return cls(
            probe_type="multilabel",
            # num_classes=2,
            hidden_dim=128,
            lr=0.001,
            dropout=0.1,
            criterion="cross_entropy",
            batch_size=32,
            device="cpu",
            epochs=10,
            pos_granularity="basic",
            semantic_granularity="basic",
            track_pos=True,
            track_semantic=True,
            layer_indices=None,
            probe_all_layers=True,
            use_class_weights=False,
            penalize_frequent_classes=True,
            penalization_scale=0.1,
            penalized_classes=None,
            save_probes=True,
            save_visualizations=True,
            show_plots=False,
            save_path=None,
            log_dir=None
        )

    @classmethod
    def minimal(cls) -> 'LinguisticProbesConfig':
        """Create a minimal configuration for basic linguistic probes."""
        return cls(
            probe_type="linear",
            hidden_dim=64,
            lr=0.001,
            dropout=0.1,
            pos_granularity="basic",
            semantic_granularity="basic",
            criterion="cross_entropy",
            batch_size=16,
            device="cpu",
            epochs=5,
            track_pos=False,
            track_semantic=False,
            layer_indices=None,
            probe_all_layers=False,
            use_class_weights=False,
            penalize_frequent_classes=False,
            penalization_scale=0.1,
            penalized_classes=None,
            save_probes=False,
            save_visualizations=False,
            save_path=None,
            log_dir=None
        )



    @classmethod
    def pos_only(cls, **kwargs) -> 'LinguisticProbesConfig':
        """Create configuration for POS probing only."""
        defaults = {
            'track_pos': True,
            'track_semantic': False,
            'pos_granularity': 'detailed'
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def comprehensive(cls, **kwargs) -> 'LinguisticProbesConfig':
        """Create a comprehensive configuration for both POS and semantic probing."""
        defaults = {
            'track_pos': True,
            'track_semantic': True,
            'pos_granularity': 'detailed',
            'semantic_granularity': 'detailed'
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def semantic_only(cls, **kwargs) -> 'LinguisticProbesConfig':
        """Create configuration for semantic probing only."""
        defaults = {
            'track_pos': False,
            'track_semantic': True,
            'semantic_granularity': 'detailed'
        }
        defaults.update(kwargs)
        return cls(**defaults)

    def get_pos_categories(self) -> Dict[str, int]:
        """Get POS categories based on granularity setting."""
        if self.pos_granularity == 'basic':
            return {
                "NOUN": 0,
                "VERB": 1,
                "ADJ": 2,
                "ADV": 3,
                "PREP": 4,
                "CONJ": 5,
                "OTHER": 6,
                # "DET": 5,
            }
        elif self.pos_granularity == 'detailed':  # detailed
            return {
                "NOUN": 0,
                "TRANSITIVE_VERB": 1,
                "INTRANSITIVE_VERB": 2,
                "COMMUNICATION_VERB": 3,
                "MOTION_VERB": 4,
                "CHANGE_VERB": 5,
                "ADJ": 6,
                "ADV": 7,
                "LOCATION": 8,
                "TEMP": 9,
                "PREP": 10,
                "RESULT": 11,
                "CONJ": 12,
                # "OTHER": 13
            }
        # todo: elif 'nltk:

    def get_semantic_categories(self) -> Dict[str, int]:
        """Get semantic categories based on granularity setting."""
        if self.semantic_granularity == 'basic':
            return {
                "AGENT": 0,
                "PATIENT": 1,
                "ACTION": 2,
                "LOCATION": 3,
                "RELATION": 4,
                "CONNECTOR": 5,
                "RESULT": 6,
                "OTHER": 7
            }
        elif self.semantic_granularity == 'detailed':
            return {
                "AGENT": 0,
                "PATIENT": 1,
                "ACTION": 2,
                "MOTION": 3,
                "COMMUNICATION": 4,
                "CHANGE": 5,
                "LOCATION": 6,
                "DESTINATION": 7,
                "TIME": 8,
                "RESULT": 9,
                "PROPERTY": 10,
                "MANNER": 11,
                "RELATION": 12,
                "CONNECTOR": 13,
                "OTHER": 14
            }
        # todo: elif 'nltk:


