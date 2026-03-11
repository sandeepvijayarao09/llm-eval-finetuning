"""
Direct Preference Optimization (DPO) fine-tuning implementation.
Implements the DPO loss function with LoRA support via PEFT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm
import json
import logging

from .data_utils import PreferenceDataset, TokenizationHelper

logger = logging.getLogger(__name__)


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss.

    Paper: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
    https://arxiv.org/abs/2305.18290

    The DPO loss encourages the model to prefer chosen outputs over rejected outputs
    by optimizing the log-likelihood ratio directly.
    """

    def __init__(self, beta: float = 0.5, label_smoothing: float = 0.0):
        """
        Initialize DPO loss.

        Args:
            beta: Temperature parameter controlling preference strength
            label_smoothing: Label smoothing for numerical stability
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute DPO loss.

        Args:
            chosen_logps: Log probabilities of chosen sequences
            rejected_logps: Log probabilities of rejected sequences
            reference_chosen_logps: Reference model log probs for chosen
            reference_rejected_logps: Reference model log probs for rejected

        Returns:
            Dictionary with loss components
        """
        # Compute log probability ratios
        log_probs_chosen = chosen_logps - reference_chosen_logps
        log_probs_rejected = rejected_logps - reference_rejected_logps

        # DPO objective: maximize log(sigmoid(beta * (log_probs_chosen - log_probs_rejected)))
        log_odds = log_probs_chosen - log_probs_rejected
        log_odds = self.beta * log_odds

        # Compute loss (binary cross entropy style)
        loss = -F.logsigmoid(log_odds)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            loss = loss * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        return {
            'loss': loss.mean(),
            'chosen_logps': chosen_logps.mean(),
            'rejected_logps': rejected_logps.mean(),
            'log_odds': log_odds.mean(),
        }


class DPOTrainer:
    """
    Trainer for Direct Preference Optimization (DPO).
    Supports LoRA via PEFT and efficient training.
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        beta: float = 0.5,
        max_length: int = 512,
        label_smoothing: float = 0.0
    ):
        """
        Initialize DPO trainer.

        Args:
            model: Main model to train
            ref_model: Reference model (frozen)
            tokenizer: Tokenizer
            device: Training device
            learning_rate: Learning rate
            beta: DPO beta parameter
            max_length: Max sequence length
            label_smoothing: Label smoothing
        """
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.beta = beta
        self.max_length = max_length

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.dpo_loss = DPOLoss(beta=beta, label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        self.training_history = {
            'loss': [],
            'chosen_logps': [],
            'rejected_logps': [],
            'log_odds': []
        }

    def compute_logps(
        self,
        sequences: List[str],
        labels: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Compute log probabilities for sequences.

        Args:
            sequences: List of text sequences
            labels: Optional labels (for masked loss)

        Returns:
            Tensor of log probabilities
        """
        logps = []

        for seq in sequences:
            # Tokenize
            encoded = self.tokenizer(
                seq,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )

            # Extract log probabilities
            # For a next-token prediction task
            logp = outputs.logits.log_softmax(dim=-1)

            # Get log prob of actual tokens
            shifted_logp = logp[..., :-1, :]
            shifted_ids = input_ids[..., 1:]

            token_logps = shifted_logp.gather(-1, shifted_ids.unsqueeze(-1)).squeeze(-1)
            masked_logps = token_logps * attention_mask[..., 1:]
            seq_logp = masked_logps.sum() / attention_mask[..., 1:].sum().clamp(min=1)

            logps.append(seq_logp)

        return torch.stack(logps)

    def train_step(
        self,
        batch: List[PreferenceDataset]
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Batch of preference samples

        Returns:
            Dictionary with loss metrics
        """
        prompts = [sample.prompt for sample in batch]
        chosen = [sample.chosen for sample in batch]
        rejected = [sample.rejected for sample in batch]

        # Compute log probabilities
        self.model.train()

        with torch.no_grad():
            ref_chosen_logps = self.compute_logps(chosen)
            ref_rejected_logps = self.compute_logps(rejected)

        # Model forward pass
        chosen_logps = self.compute_logps(chosen)
        rejected_logps = self.compute_logps(rejected)

        # Compute DPO loss
        loss_dict = self.dpo_loss(
            chosen_logps=chosen_logps,
            rejected_logps=rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps
        )

        loss = loss_dict['loss']

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Record metrics
        metrics = {
            'loss': loss.item(),
            'chosen_logps': loss_dict['chosen_logps'].item(),
            'rejected_logps': loss_dict['rejected_logps'].item(),
            'log_odds': loss_dict['log_odds'].item(),
        }

        for key, value in metrics.items():
            self.training_history[key].append(value)

        return metrics

    def train(
        self,
        train_dataset: List[PreferenceDataset],
        num_epochs: int = 3,
        batch_size: int = 8,
        val_dataset: Optional[List[PreferenceDataset]] = None,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Train the model using DPO.

        Args:
            train_dataset: Training preference pairs
            num_epochs: Number of training epochs
            batch_size: Batch size
            val_dataset: Optional validation dataset
            save_dir: Optional directory to save checkpoints

        Returns:
            Training results dictionary
        """
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        total_steps = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            # Training
            epoch_losses = []
            for i in tqdm(
                range(0, len(train_dataset), batch_size),
                desc=f"Training Epoch {epoch+1}"
            ):
                batch = train_dataset[i:i+batch_size]
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])
                total_steps += 1

            avg_train_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")

            # Validation
            if val_dataset:
                val_loss = self.evaluate(val_dataset, batch_size)
                logger.info(f"Validation loss: {val_loss:.4f}")

                # Save best checkpoint
                if save_dir and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(save_dir)

        return {
            'total_steps': total_steps,
            'final_train_loss': avg_train_loss,
            'best_val_loss': best_val_loss,
            'training_history': self.training_history
        }

    def evaluate(
        self,
        dataset: List[PreferenceDataset],
        batch_size: int = 8
    ) -> float:
        """
        Evaluate on a dataset.

        Args:
            dataset: Evaluation dataset
            batch_size: Batch size

        Returns:
            Average loss
        """
        self.model.eval()
        losses = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(dataset), batch_size),
                desc="Evaluating"
            ):
                batch = dataset[i:i+batch_size]
                metrics = self.train_step(batch)
                losses.append(metrics['loss'])

        return sum(losses) / len(losses)

    def save_model(self, save_dir: str) -> None:
        """
        Save model and training state.

        Args:
            save_dir: Directory to save to
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_path / "model")
        self.tokenizer.save_pretrained(save_path / "tokenizer")

        # Save training state
        state = {
            'optimizer': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': {
                'learning_rate': self.learning_rate,
                'beta': self.beta,
                'max_length': self.max_length
            }
        }

        with open(save_path / "training_state.json", 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_history = {
                k: v for k, v in self.training_history.items()
            }
            json.dump(state['config'], f, indent=2)

        logger.info(f"Model saved to {save_dir}")

    def load_model(self, model_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            model_path: Path to model directory
        """
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path
        ).to(self.device)

        logger.info(f"Model loaded from {model_path}")
