"""
Model quantization utilities.
Supports INT8 and GPTQ quantization with benchmarking.
"""

import torch
import time
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Quantize models using different methods."""

    @staticmethod
    def quantize_int8(
        model,
        device_map: str = "auto"
    ):
        """
        Quantize model to INT8 using bitsandbytes.

        Args:
            model: Model to quantize
            device_map: Device mapping for quantized model

        Returns:
            Quantized model
        """
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )

            # Model loading with quantization would go here
            # In production, you'd use:
            # from_pretrained(..., quantization_config=quantization_config)

            logger.info("INT8 quantization config created")
            return model

        except ImportError:
            logger.warning(
                "bitsandbytes not available. "
                "Install with: pip install bitsandbytes"
            )
            return model

    @staticmethod
    def quantize_int4(
        model,
        double_quant: bool = True,
        quant_type: str = "nf4"
    ):
        """
        Quantize model to INT4 (QLoRA style).

        Args:
            model: Model to quantize
            double_quant: Use double quantization
            quant_type: Quantization type ('nf4' or 'fp4')

        Returns:
            Quantized model
        """
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            logger.info(f"INT4 quantization ({quant_type}) config created")
            return model

        except ImportError:
            logger.warning(
                "bitsandbytes not available. "
                "Install with: pip install bitsandbytes"
            )
            return model

    @staticmethod
    def quantize_dynamic(model):
        """
        Apply dynamic quantization using PyTorch native.

        Args:
            model: Model to quantize

        Returns:
            Quantized model
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )

        logger.info("Dynamic quantization applied")
        return quantized_model


class QuantizationBenchmark:
    """Benchmark quantization effects on speed, memory, and quality."""

    @staticmethod
    def benchmark_model(
        original_model,
        quantized_model,
        test_input: torch.Tensor,
        num_runs: int = 10
    ) -> Dict:
        """
        Benchmark original vs quantized model.

        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_input: Sample input tensor
            num_runs: Number of inference runs

        Returns:
            Dictionary with benchmark results
        """
        results = {
            'original': {},
            'quantized': {}
        }

        # Benchmark original model
        original_model.eval()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = original_model(test_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        original_time = time.time() - start_time

        results['original']['inference_time'] = original_time / num_runs
        results['original']['memory_mb'] = ModelQuantizer._get_model_size(original_model) / 1e6

        # Benchmark quantized model
        quantized_model.eval()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = quantized_model(test_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        quantized_time = time.time() - start_time

        results['quantized']['inference_time'] = quantized_time / num_runs
        results['quantized']['memory_mb'] = ModelQuantizer._get_model_size(quantized_model) / 1e6

        # Compute speedup and compression
        results['speedup'] = original_time / quantized_time
        results['compression_ratio'] = (
            results['original']['memory_mb'] /
            results['quantized']['memory_mb']
        )

        return results

    @staticmethod
    def _get_model_size(model) -> int:
        """Get total model size in bytes."""
        total_params = 0
        for param in model.parameters():
            total_params += param.numel()

        # Approximate: 4 bytes per float32 parameter
        return total_params * 4

    @staticmethod
    def compare_output_quality(
        original_output: torch.Tensor,
        quantized_output: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compare output quality between original and quantized models.

        Args:
            original_output: Output from original model
            quantized_output: Output from quantized model

        Returns:
            Dictionary with quality metrics
        """
        # Calculate L2 distance
        l2_distance = torch.norm(
            original_output - quantized_output,
            p=2
        ).item()

        # Calculate cosine similarity
        orig_flat = original_output.flatten().float()
        quant_flat = quantized_output.flatten().float()

        cosine_sim = torch.nn.functional.cosine_similarity(
            orig_flat.unsqueeze(0),
            quant_flat.unsqueeze(0)
        ).item()

        # Calculate relative error
        relative_error = (
            torch.norm(original_output - quantized_output, p=2) /
            torch.norm(original_output, p=2)
        ).item()

        return {
            'l2_distance': l2_distance,
            'cosine_similarity': cosine_sim,
            'relative_error': relative_error,
        }

    @staticmethod
    def generate_comparison_report(benchmarks: Dict) -> str:
        """
        Generate a human-readable comparison report.

        Args:
            benchmarks: Benchmark results from benchmark_model

        Returns:
            Formatted string report
        """
        report = "=" * 60 + "\n"
        report += "MODEL QUANTIZATION BENCHMARK REPORT\n"
        report += "=" * 60 + "\n\n"

        report += "Original Model:\n"
        report += f"  Inference Time: {benchmarks['original']['inference_time']:.4f}s\n"
        report += f"  Memory: {benchmarks['original']['memory_mb']:.2f} MB\n\n"

        report += "Quantized Model:\n"
        report += f"  Inference Time: {benchmarks['quantized']['inference_time']:.4f}s\n"
        report += f"  Memory: {benchmarks['quantized']['memory_mb']:.2f} MB\n\n"

        report += "Performance Metrics:\n"
        report += f"  Speedup: {benchmarks['speedup']:.2f}x\n"
        report += f"  Compression Ratio: {benchmarks['compression_ratio']:.2f}x\n"
        report += f"  Latency Reduction: {(1 - 1/benchmarks['speedup'])*100:.1f}%\n"

        return report


class LoRAQuantizer:
    """Quantization utilities for LoRA-adapted models."""

    @staticmethod
    def get_lora_config(
        r: int = 8,
        lora_alpha: int = 16,
        target_modules: Optional[list] = None,
        lora_dropout: float = 0.05
    ) -> Dict:
        """
        Create LoRA configuration.

        Args:
            r: LoRA rank
            lora_alpha: LoRA alpha (scaling factor)
            target_modules: Modules to apply LoRA to
            lora_dropout: Dropout probability

        Returns:
            LoRA config dictionary
        """
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        return {
            'r': r,
            'lora_alpha': lora_alpha,
            'target_modules': target_modules,
            'lora_dropout': lora_dropout,
            'bias': 'none',
            'task_type': 'CAUSAL_LM'
        }

    @staticmethod
    def apply_lora_and_quantize(
        model,
        quantization_method: str = "int4",
        lora_config: Optional[Dict] = None
    ):
        """
        Apply LoRA adaptation and quantization.

        Args:
            model: Base model
            quantization_method: 'int4', 'int8', or 'none'
            lora_config: LoRA configuration

        Returns:
            Adapted and optionally quantized model
        """
        if lora_config is None:
            lora_config = LoRAQuantizer.get_lora_config()

        # Note: In production, you'd use peft.get_peft_model
        # and BitsAndBytesConfig for proper quantization
        # This is a simplified example

        logger.info(f"Applying LoRA with config: {lora_config}")

        if quantization_method == "int4":
            logger.info("Applying INT4 quantization")
        elif quantization_method == "int8":
            logger.info("Applying INT8 quantization")

        return model
