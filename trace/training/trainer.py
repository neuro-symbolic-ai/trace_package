import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple

from .config import TrainingConfig
from .utils import (
    prepare_batch_for_model, compute_loss, setup_hidden_state_hooks,
    set_seed, save_checkpoint, validate_model, evaluate_model_comprehensive
)
from .callbacks import TrainingCallbacks


class Trainer:
    """
    Main trainer class for transformer models with comprehensive analysis.

    This class orchestrates the training process while integrating various
    analysis modules at specified intervals, preserving the original training
    logic in a modular structure.
    """

    def __init__(self, config: TrainingConfig, tokenizer, model=None):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            tokenizer: Model tokenizer
            model: Optional pre-initialized model
        """
        self.config = config
        self.tokenizer = tokenizer
        self.model = model

        # Set up device and reproducibility
        set_seed(config.seed)
        self.device = config.device

        # Create necessary directories
        config.create_directories()

        # Initialize training components
        self.criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index)
        self.optimizer = None
        self.scheduler = None

        # Initialize callbacks for analysis
        self.callbacks = TrainingCallbacks(config, tokenizer, self.device)

        # Training state
        self.step_counter = 0
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "token_accuracy": [],
            "exact_match": [],
            "bleu_score": [],
            "perplexity": [],
            "epochs": []
        }

        # Hidden states storage
        self.hidden_states = {}

        # WandB integration
        self.use_wandb = self._setup_wandb()

    def _setup_wandb(self) -> bool:
        """Set up WandB logging if available."""
        try:
            import wandb
            return wandb.run is not None
        except ImportError:
            return False

    def _setup_optimizer(self):
        """Set up optimizer and learning rate scheduler."""
        optimizer_kwargs = {"lr": self.config.learning_rate}
        if self.config.weight_decay:
            optimizer_kwargs["weight_decay"] = self.config.weight_decay

        self.optimizer = optim.Adam(self.model.parameters(), **optimizer_kwargs)
        print(f"Number of model parameters: {sum(p.numel() for p in self.model.parameters())}")
        print(
            f"Number of optimizer parameters: {sum(p.numel() for g in self.optimizer.param_groups for p in g['params'])}")
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(param is p for g in self.optimizer.param_groups for p in g['params']):
                print(f"[!] Parameter {name} missing from optimizer.")



        # Set up learning rate scheduler if warmup is requested
        if self.config.warmup_steps:
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(1.0, step / self.config.warmup_steps)
            )

    def _setup_model_hooks(self):
        """Set up forward hooks to capture hidden states."""
        self.hidden_states = setup_hidden_state_hooks(self.model)

    def train(
            self,
            train_loader,
            val_loader,
            test_loader: Optional = None,
            # model=None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Train the model with comprehensive analysis.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            model: Optional model to train (if not provided in __init__)

        Returns:
            Tuple of (best_validation_loss, analysis_results)
        """
        # Set model if provided
        # if model is not None:
        #     self.model = model
        # else:
        #     raise ValueError("No model provided for training")

        # Move model to device
        self.model.to(self.device)

        # Set up training components
        self._setup_optimizer()
        self._setup_model_hooks()

        print(f"Training {self.config.model_type} model for {self.config.epochs} epochs")
        print(f"Task mode: {self.config.task_mode}")
        print(f"Device: {self.device}")
        print(f"Model: {self.model}")

        # Main training loop
        for epoch in range(self.config.epochs):
            epoch_loss = self._train_epoch(train_loader, epoch)
            val_loss = self._validate_epoch(val_loader)

            # Update training history
            self.training_history["train_loss"].append(epoch_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["epochs"].append(epoch + 1)

            # Log epoch results
            print(f"Epoch {epoch + 1} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

            # Log to wandb
            if self.use_wandb:
                import wandb
                wandb.log({
                    "train/epoch_loss": epoch_loss,
                    "val/loss": val_loss,
                    "epoch": epoch + 1
                })

            # Save checkpoint if best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch)
                print(f"Best model saved (Val Loss: {val_loss:.4f})")

            # Run test if provided
            if test_loader is not None:
                test_metrics = self._test_epoch(test_loader, epoch)
                self._log_test_metrics(test_metrics, epoch)

        # Generate final visualizations
        model_name = os.path.basename(self.config.save_path).split(".")[0]
        self.callbacks.generate_final_visualizations(model_name)

        # Get analysis summary
        analysis_results = self.callbacks.get_analysis_summary()
        analysis_results["training_history"] = self.training_history

        print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        print(f"All visualizations saved to {self.config.plots_path}")
        # saving the analysis results

        analysis_results_path = os.path.join(self.config.plots_path)
        print(f"Saving analysis results to {analysis_results_path}")
        self.callbacks.save_analysis_results(analysis_results, analysis_results_path)

        return self.best_val_loss, analysis_results

    def _train_epoch(self, train_loader, epoch: int) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_train_loss = 0
        epoch_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            # Reset gradients
            self.optimizer.zero_grad()

            # Prepare inputs and labels
            model_inputs, labels_info = prepare_batch_for_model(
                batch, self.model, self.config.model_type,
                self.config.task_mode, self.device, self.config.ignore_index
            )
            print("Labels stats:", labels_info["labels"].min(), labels_info["labels"].max())
            print("Example input IDs:", batch["input_ids"][0])
            print("Example labels:", labels_info["labels"][0])

            print("Ignore index used:", self.criterion.ignore_index)
            exit(1)
            # Forward pass
            outputs = self.model(**model_inputs)

            # Calculate loss
            loss = compute_loss(
                outputs, labels_info["labels"],
                self.criterion,
            )
            # Backward pass
            loss.backward()
            # predictions = torch.argmax(outputs, dim=-1).cpu()
            # Run analysis if needed (before optimizer step to capture gradients)
            if self.callbacks.should_track(self.step_counter):
                self.callbacks.run_analysis(
                    self.model, batch, self.hidden_states,
                    self.step_counter,
                    val_loader=None,
                    tokenizer=self.tokenizer,
                    predictions=outputs
                    # Could pass val_loader here
                )

            # Apply gradients
            self.optimizer.step()

            # Update learning rate if using scheduler
            if self.scheduler:
                self.scheduler.step()

            # Update counters
            batch_size = batch["input_ids"].size(0)
            total_train_loss += loss.item() * batch_size
            epoch_steps += batch_size
            self.step_counter += 1

            # Log per step (if not log_only_at_epoch_end)
            if not self.config.log_only_at_epoch_end and self.step_counter % self.config.log_steps == 0:
                progress_bar.set_postfix({"loss": loss.item()})
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/step_loss": loss.item(),
                        "train/step": self.step_counter,
                    })

        # Calculate average loss for epoch
        avg_train_loss = total_train_loss / epoch_steps if epoch_steps > 0 else float('inf')
        return avg_train_loss

    def _validate_epoch(self, val_loader) -> float:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        return validate_model(
            self.model, val_loader, self.criterion,
            self.device, self.config.model_type, self.config.task_mode
        )

    def _test_epoch(self, test_loader, epoch: int) -> Dict[str, float]:
        """
        Test for one epoch and compute metrics.

        Args:
            test_loader: Test data loader
            epoch: Current epoch

        Returns:
            Dictionary of test metrics
        """

        return evaluate_model_comprehensive(
            model=self.model,
            test_loader=test_loader,
            criterion=self.criterion,
            device=self.device,
            model_type=self.config.model_type,
            task_mode=self.config.task_mode,
            tokenizer=self.tokenizer,
            ignore_index=self.config.ignore_index,
        )
        #
        # test_loss = validate_model(
        #     self.model, test_loader, self.criterion,
        #     self.device, self.config.model_type, self.config.task_mode
        # )
        #
        # # Placeholder for other metrics
        # metrics = {
        #     "test_loss": test_loss,
        #     "exact_match": 0.0,  # TODO: Implement
        #     "token_accuracy": 0.0,  # TODO: Implement
        #     "bleu_score": 0.0,  # TODO: Implement
        #     "perplexity": math.exp(test_loss)
        # }
        #
        # return metrics

    def _log_test_metrics(self, test_metrics: Dict[str, float], epoch: int):
        """
        Log test metrics to history and wandb.

        Args:
            test_metrics: Dictionary of test metrics
            epoch: Current epoch
        """
        # Update training history
        self.training_history["test_loss"].append(test_metrics["test_loss"])
        self.training_history["exact_match"].append(test_metrics["exact_match"])
        self.training_history["token_accuracy"].append(test_metrics["token_accuracy"])
        self.training_history["bleu_score"].append(test_metrics["bleu_score"])
        self.training_history["perplexity"].append(test_metrics["perplexity"])

        # Print test results
        print(
            f"Test metrics | Loss: {test_metrics['test_loss']:.4f} | "
            f"EM: {test_metrics['exact_match']:.4f} | "
            f"Token Acc: {test_metrics['token_accuracy']:.4f} | "
            f"BLEU: {test_metrics['bleu_score']:.4f}"
        )

        # Log to wandb
        if self.use_wandb:
            import wandb
            wandb_metrics = {f"test/{k}": v for k, v in test_metrics.items()}
            wandb_metrics["epoch"] = epoch + 1
            wandb.log(wandb_metrics)

    def _save_checkpoint(self, epoch: int):
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
        """
        print(f"Saving checkpoint for epoch {epoch + 1}...")
        save_checkpoint(
            self.model,
            self.config.save_path,
            self.config.model_type,
            self.tokenizer,
            epoch,
            self.optimizer,
            self.best_val_loss,
            self.config.batch_size,
            self.config.epochs,
            self.config.learning_rate,
            self.config.warmup_steps,
            self.config.weight_decay
        )

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if self.model is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return {
            'epoch': checkpoint.get('epoch', 0),
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'hyperparameters': checkpoint.get('hyperparameters', {})
        }


# Convenience function to maintain compatibility with original interface
def train_model(
        model,
        train_loader,
        val_loader,
        epochs: int,
        lr: float,
        device: str,
        save_path: str,
        tokenizer,
        test_loader=None,
        **kwargs
) -> Tuple[float, Dict[str, Any]]:
    """
    Compatibility function that mimics the original train_model interface.

    Args:
        model: The transformer model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Training device
        save_path: Model save path
        tokenizer: Model tokenizer
        test_loader: Optional test loader
        **kwargs: Additional training arguments

    Returns:
        Tuple of (best_val_loss, analysis_results)
    """
    # Create config from parameters
    config = TrainingConfig(
        epochs=epochs,
        learning_rate=lr,
        device=device,
        save_path=save_path,
        **kwargs
    )

    # Create trainer
    trainer = Trainer(config, tokenizer, model)

    # Train the model
    return trainer.train(train_loader, val_loader, test_loader)