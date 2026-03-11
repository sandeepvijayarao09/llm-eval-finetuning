#!/usr/bin/env python3
"""
CLI runner for LLM evaluation.
Supports multiple benchmarks and output formats.

Usage:
    python run_eval.py --model meta-llama/Llama-2-7b-hf --benchmark truthful_qa
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List
import yaml

from src.evaluation import LLMEvaluator, BenchmarkDatasets
from src.inference import OptimizedInferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Runner for evaluation tasks."""

    def __init__(self, config_path: str = "configs/eval_config.yaml"):
        """
        Initialize runner.

        Args:
            config_path: Path to evaluation config
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.evaluator = None
        self.inference_engine = None

    def _load_config(self) -> dict:
        """Load YAML configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _init_model_inference(self, model_name: str):
        """
        Initialize model and inference engine.

        Args:
            model_name: HuggingFace model ID
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info(f"Loading model: {model_name}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            device = self.config.get('model', {}).get('device', 'cuda')

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device,
            )

            self.inference_engine = OptimizedInferenceEngine(
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_batch_size=self.config.get('inference', {}).get('batch_size', 8)
            )

            logger.info(f"Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _create_model_generator(self):
        """Create a text generation function."""
        def generate_fn(prompt: str) -> str:
            try:
                output = self.inference_engine.generate_single(
                    prompt,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9
                )
                # Remove prompt from output
                return output.replace(prompt, '').strip()
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return f"[Error during generation: {str(e)}]"

        return generate_fn

    def run_benchmark(
        self,
        model_name: str,
        benchmark_name: str,
        num_samples: int = None,
        output_dir: str = "results"
    ):
        """
        Run evaluation on a benchmark.

        Args:
            model_name: HuggingFace model ID
            benchmark_name: Name of benchmark
            num_samples: Number of samples to evaluate
            output_dir: Output directory for results
        """
        # Validate benchmark
        available = BenchmarkDatasets.list_benchmarks()
        if benchmark_name not in available:
            logger.error(
                f"Unknown benchmark: {benchmark_name}. "
                f"Available: {', '.join(available)}"
            )
            return

        # Initialize model
        try:
            self._init_model_inference(model_name)
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run evaluation
        logger.info(f"Starting evaluation on {benchmark_name}")
        model_generator = self._create_model_generator()

        evaluator = LLMEvaluator(
            batch_size=self.config.get('inference', {}).get('batch_size', 8)
        )

        result = evaluator.evaluate_model_on_benchmark(
            model_generator=model_generator,
            benchmark_name=benchmark_name,
            num_samples=num_samples
        )

        # Save results
        logger.info("Saving results")
        json_path = output_path / f"{benchmark_name}_results.json"
        csv_path = output_path / f"{benchmark_name}_results.csv"
        html_path = output_path / f"{benchmark_name}_report.html"

        evaluator.export_results_json(result, str(json_path))
        logger.info(f"JSON results: {json_path}")

        evaluator.export_results_csv(result, str(csv_path))
        logger.info(f"CSV results: {csv_path}")

        if self.config.get('output', {}).get('generate_html_report', True):
            evaluator.generate_html_report(result, str(html_path))
            logger.info(f"HTML report: {html_path}")

        # Print summary
        self._print_summary(result)

    def _print_summary(self, result):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Model: {result.model_name}")
        print(f"Benchmark: {result.benchmark_name}")
        print(f"Samples: {result.total_samples}")
        print(f"Time: {result.inference_time_seconds:.2f}s")
        print(f"Time per sample: {result.inference_time_seconds/result.total_samples:.2f}s")
        print("\nMetrics:")
        for metric, value in sorted(result.summary.items()):
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        print("=" * 70 + "\n")

    def run_all_benchmarks(
        self,
        model_name: str,
        output_dir: str = "results"
    ):
        """
        Run evaluation on all benchmarks.

        Args:
            model_name: HuggingFace model ID
            output_dir: Output directory
        """
        benchmarks = BenchmarkDatasets.list_benchmarks()

        for benchmark in benchmarks:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"Running benchmark: {benchmark}")
            logger.info(f"{'=' * 70}\n")

            self.run_benchmark(
                model_name=model_name,
                benchmark_name=benchmark,
                output_dir=output_dir
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model ID"
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=BenchmarkDatasets.list_benchmarks() + ["all"],
        default="truthful_qa",
        help="Benchmark to evaluate on"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    # Create runner
    runner = EvaluationRunner(config_path=args.config)

    # Run evaluation
    if args.benchmark == "all":
        runner.run_all_benchmarks(
            model_name=args.model,
            output_dir=args.output_dir
        )
    else:
        runner.run_benchmark(
            model_name=args.model,
            benchmark_name=args.benchmark,
            num_samples=args.num_samples,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
