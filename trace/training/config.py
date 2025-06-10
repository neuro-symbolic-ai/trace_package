from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import os

import torch


@dataclass
class TrainingConfig:
    """
    Configuration class for training transformer models.

    This dataclass contains all parameters for training, including
    model settings, optimization parameters, and analysis tracking options.
    """

    # Model architecture
    model_type: str = "decoder_only"
    vocab_size: int = 7000
    d_model: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 6
    num_decoder_layers: int = 2
    max_seq_length: int = 16
    dropout: float = 0.1
    pad_idx: int = 0
    d_ff: int = 128
    no_fnn: bool = False

    # Training parameters
    epochs: int = 30
    learning_rate: float = 1e-3
    batch_size: int = 128
    warmup_steps: Optional[int] = 1000
    weight_decay: Optional[float] = 0.01
    task_mode: str = "next_token"  # mlm, next_token, seq2seq
    ignore_index: int = -100

    # Logging and saving
    log_steps: int = 100
    log_only_at_epoch_end: bool = False
    save_path: str = "../models/model.pt"
    plots_path: Optional[str] = "../plots"

    # Analysis tracking intervals
    track_interval: int = 1000
    max_history: int = 10

    # Gradient analysis
    track_gradients: bool = True
    track_layers: Optional[List[str]] = None
    track_last_token: bool = False
    gradient_save_dir: Optional[str] = None

    # Hessian analysis
    track_hessian: bool = True
    hessian_n_components: int = 5
    track_component_hessian: bool = True
    component_list: Optional[List[str]] = None
    track_gradient_alignment: bool = True
    track_sharpness: bool = True
    track_train_val_landscape_divergence: bool = True
    save_hessian_data: bool = True

    # Visualization options
    create_training_plots: bool = True
    analyze_embeddings: bool = True
    analyze_attention: bool = False
    analyze_token_importance: bool = False

    # Linguistic probes tracking
    track_linguistic_probes: bool = True
    probe_layers: Optional[Union[List[int], Dict[str, List[int]]]] = None
    probe_load_path: Optional[str] = None
    probe_num_features: int = 8
    probe_hidden_dim: int = 128
    probe_lr: float = 0.001
    probe_epochs: int = 10
    probe_type: str = "multilabel"  # 'linear' or 'multilabel'


    # Semantic probes tracking
    track_semantic_probes: bool = True
    semantic_probe_layers: Optional[Union[List[int], Dict[str, List[int]]]] = None
    semantic_probe_load_path: Optional[str] = None
    semantic_probe_num_features: int = 8
    semantic_probe_hidden_dim: int = 128
    semantic_probe_lr: float = 0.001
    semantic_probe_epochs: int = 10
    semantic_probe_type: str = "multilabel"  # 'linear' or 'multilabel'

    # Intrinsic dimensions tracking
    track_intrinsic_dimensions: bool = True
    id_method: str = "TwoNN"
    id_selected_layers: Optional[Union[List[int], Dict[str, List[int]]]] = None

    # POS tracking (placeholder for now)
    track_pos_performance: bool = True
    pos_granularity: str = 'basic'

    # Semantic role tracking (placeholder for now)
    track_semantic_roles: bool = False
    semantic_granularity: str = 'basic'

    # Device and reproducibility
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42

    def __post_init__(self):
        """Validate and set up derived configurations."""
        # Standardize model type
        self.model_type = self.model_type.replace('-', '_')

        # Set up default paths if not provided
        if self.plots_path and not self.gradient_save_dir:
            self.gradient_save_dir = os.path.join(self.plots_path, 'gradient_data')

        # Set up default component list for Hessian analysis
        if self.component_list is None and self.track_component_hessian:
            if self.no_fnn:
                self.component_list = ["attention", "hidden_states"]
            else:
                self.component_list = ["ffn", "attention", "hidden_states"]

        # Validate model type
        if self.model_type not in ['encoder_only', 'decoder_only', 'encoder_decoder']:
            raise ValueError(f"Invalid model_type: {self.model_type}")

        # Validate task mode
        if self.task_mode not in ['mlm', 'next_token', 'seq2seq']:
            raise ValueError(f"Invalid task_mode: {self.task_mode}")

    @classmethod
    def from_args(cls, args) -> 'TrainingConfig':
        """
        Create a TrainingConfig from command line arguments.

        Args:
            args: Parsed command line arguments

        Returns:
            TrainingConfig instance
        """
        # Map argument names to config field names
        config_dict = {}

        # Model architecture
        config_dict['model_type'] = getattr(args, 'model_type', 'decoder_only')
        config_dict['vocab_size'] = getattr(args, 'vocab_size', 7000)
        config_dict['d_model'] = getattr(args, 'd_model', 64)
        config_dict['num_heads'] = getattr(args, 'num_heads', 4)
        config_dict['num_encoder_layers'] = getattr(args, 'num_encoder_layers', 6)
        config_dict['num_decoder_layers'] = getattr(args, 'num_decoder_layers', 2)
        config_dict['max_seq_length'] = getattr(args, 'max_seq_length', 16)
        config_dict['dropout'] = getattr(args, 'dropout', 0.1)
        config_dict['pad_idx'] = getattr(args, 'pad_idx', 0)
        config_dict['d_ff'] = getattr(args, 'd_ff', 128)
        config_dict['no_fnn'] = getattr(args, 'no_fnn', False)

        # Training parameters
        config_dict['epochs'] = getattr(args, 'num_epochs', 30)
        config_dict['learning_rate'] = getattr(args, 'learning_rate', 1e-3)
        config_dict['batch_size'] = getattr(args, 'batch_size', 128)
        config_dict['warmup_steps'] = getattr(args, 'warmup_steps', None)
        config_dict['weight_decay'] = getattr(args, 'weight_decay', None)
        config_dict['task_mode'] = getattr(args, 'task_mode', 'next_token')

        # Paths
        model_save_dir = getattr(args, 'model_save_dir', '../models')
        model_name = getattr(args, 'model_name', 'model.pt')
        config_dict['save_path'] = f"{model_save_dir}/{model_name}"
        config_dict['plots_path'] = getattr(args, 'plots_path', '../plots')

        # Analysis tracking
        config_dict['track_interval'] = getattr(args, 'track_interval', 1000)
        config_dict['track_gradients'] = getattr(args, 'track_gradients', False)
        config_dict['track_hessian'] = getattr(args, 'track_hessian', False)
        config_dict['hessian_n_components'] = getattr(args, 'hessian_n_components', 10)
        config_dict['track_component_hessian'] = getattr(args, 'track_component_hessian', False)
        config_dict['track_gradient_alignment'] = getattr(args, 'track_gradient_alignment', False)
        config_dict['track_train_val_landscape_divergence'] = getattr(args, 'track_train_val_landscape_divergence',
                                                                      False)

        # Probe tracking
        config_dict['track_linguistic_probes'] = getattr(args, 'track_probe', False)
        config_dict['probe_load_path'] = getattr(args, 'probe_load_path', None)
        config_dict['probe_layers'] = getattr(args, 'probe_layer_indices', None)
        config_dict['probe_num_features'] = getattr(args, 'probe_num_features', 8)
        config_dict['probe_hidden_dim'] = getattr(args, 'probe_hidden_dim', 128)
        config_dict['probe_lr'] = getattr(args, 'probe_lr', 0.001)
        config_dict['probe_epochs'] = getattr(args, 'probe_epochs', 10)
        config_dict['probe_type'] = getattr(args, 'probe_type', 'multilabel')  # 'linear' or 'multilabel'

        # Semantic probe tracking
        config_dict['track_semantic_probes'] = getattr(args, 'semantic_track_probe', False)
        config_dict['semantic_probe_load_path'] = getattr(args, 'semantic_probe_load_path', None)
        config_dict['semantic_probe_num_features'] = getattr(args, 'semantic_probe_num_features', 8)
        config_dict['semantic_probe_hidden_dim'] = getattr(args, 'semantic_probe_hidden_dim', 128)
        config_dict['semantic_probe_lr'] = getattr(args, 'semantic_probe_lr', 0.001)
        config_dict['semantic_probe_epochs'] = getattr(args, 'semantic_probe_epochs', 10)
        config_dict['semantic_probe_type'] = getattr(args, 'semantic_probe_type', 'multilabel')  # 'linear' or 'multilabel'

        # Intrinsic dimensions
        config_dict['track_intrinsic_dimensions'] = getattr(args, 'track_intrinsic_dimension', False)

        # POS and semantic role tracking (placeholders)
        config_dict['track_pos_performance'] = getattr(args, 'track_POS', False)
        config_dict['track_semantic_roles'] = getattr(args, 'track_semantic_roles', False)

        # Visualization
        config_dict['create_training_plots'] = getattr(args, 'create_training_plots', False)
        config_dict['analyze_embeddings'] = getattr(args, 'analyze_embeddings', False)
        config_dict['log_only_at_epoch_end'] = getattr(args, 'log_only_at_epoch_end', False)

        # Device and seed
        config_dict['device'] = getattr(args, 'device', 'auto')
        config_dict['seed'] = getattr(args, 'seed', 42)

        return cls(**config_dict)

    def create_directories(self):
        """Create necessary directories for saving results."""
        if self.plots_path:
            os.makedirs(self.plots_path, exist_ok=True)

        if self.gradient_save_dir:
            os.makedirs(self.gradient_save_dir, exist_ok=True)

        # Create save directory
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)