#!/usr/bin/env python3
"""
CLI runner for DPO fine-tuning.

Usage:
    python run_finetune.py --config configs/dpo_config.yaml
"""

import argparse
import sys
import logging
from pathlib import Path
import yaml

from src.finetuning import DPOTrainer, DPODatasetProcessor
from src.finetuning.data_utils import PreferenceDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FineTuneRunner:
    """Runner for fine-tuning tasks."""

    def __init__(self, config_path: str):
        """
        Initialize runner.

        Args:
            config_path: Path to DPO config
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load YAML configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {self.config_path}")
            return {}

    def _init_models(self):
        """Initialize base and reference models."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            model_config = self.config.get('model', {})
            base_model_id = model_config.get('base_model')
            ref_model_id = model_config.get('reference_model')
            device = model_config.get('device', 'cuda')

            logger.info(f"Loading base model: {base_model_id}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dtype = torch.float16 if model_config.get('dtype') == 'float16' else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=dtype,
                device_map=device,
            )

            logger.info(f"Loading reference model: {ref_model_id}")
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_model_id,
                torch_dtype=dtype,
                device_map=device,
            )

            return model, ref_model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _apply_lora(self, model):
        """Apply LoRA to model."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType

            lora_config = self.config.get('lora', {})

            if not lora_config.get('enabled', False):
                logger.info("LoRA is disabled")
                return model

            logger.info("Applying LoRA configuration")

            config = LoraConfig(
                r=lora_config.get('r', 8),
                lora_alpha=lora_config.get('lora_alpha', 16),
                target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
                lora_dropout=lora_config.get('lora_dropout', 0.05),
                bias=lora_config.get('bias', 'none'),
                task_type=TaskType.CAUSAL_LM,
            )

            model = get_peft_model(model, config)
            logger.info("LoRA applied successfully")

            return model

        except ImportError:
            logger.warning("peft not available. Install with: pip install peft")
            return model

    def prepare_datasets(self):
        """Prepare training datasets."""
        processor = DPODatasetProcessor(
            template_type=self.config.get('data', {}).get('template_type', 'alpaca')
        )

        # Create sample dataset for demo
        logger.info("Creating sample preference dataset")
        full_dataset = processor.create_sample_dataset()

        # Split into train/val/test
        train_set, val_set, test_set = processor.split_dataset(
            full_dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            seed=self.config.get('seed', 42)
        )

        logger.info(f"Dataset split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

        return train_set, val_set, test_set

    def run_training(self):
        """Run DPO fine-tuning."""
        logger.info("Starting DPO fine-tuning")

        # Load models
        logger.info("Loading models...")
        model, ref_model, tokenizer = self._init_models()

        # Apply LoRA
        model = self._apply_lora(model)

        # Prepare data
        logger.info("Preparing datasets...")
        train_set, val_set, test_set = self.prepare_datasets()

        # Initialize trainer
        train_config = self.config.get('training', {})
        logger.info("Initializing DPO trainer")

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            device=self.config.get('model', {}).get('device', 'cuda'),
            learning_rate=train_config.get('learning_rate', 1e-4),
            beta=train_config.get('beta', 0.5),
            max_length=train_config.get('max_length', 512),
            label_smoothing=train_config.get('label_smoothing', 0.0)
        )

        # Run training
        output_dir = self.config.get('output', {}).get('output_dir', './checkpoints/dpo')

        logger.info(f"Starting training (output: {output_dir})")
        results = trainer.train(
            train_dataset=train_set,
            num_epochs=train_config.get('num_epochs', 3),
            batch_size=train_config.get('batch_size', 8),
            val_dataset=val_set,
            save_dir=output_dir
        )

        # Print summary
        self._print_training_summary(results)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_loss = trainer.evaluate(test_set, batch_size=train_config.get('batch_size', 8))
        logger.info(f"Test loss: {test_loss:.4f}")

        # Save final model
        logger.info(f"Saving final model to {output_dir}")
        trainer.save_model(output_dir)

        return results

    def _print_training_summary(self, results: dict):
        """Print training summary."""
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Total steps: {results.get('total_steps', 'N/A')}")
        print(f"Final training loss: {results.get('final_train_loss', 'N/A'):.4f}")
        print(f"Best validation loss: {results.get('best_val_loss', 'N/A'):.4f}")
        print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DPO Fine-Tuning Runner"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/dpo_config.yaml",
        help="Path to DPO config file"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with minimal data"
    )

    args = parser.parse_args()

    try:
        runner = FineTuneRunner(config_path=args.config)
        runner.run_training()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
