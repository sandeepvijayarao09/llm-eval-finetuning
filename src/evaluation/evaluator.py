"""
Main LLM evaluation harness.
Supports multi-model evaluation with various benchmark suites.
"""

import json
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import csv
from tqdm import tqdm
import numpy as np

from .metrics import MetricsComputer
from .benchmarks import BenchmarkDatasets, BenchmarkQuestion, InstructionFollowingTask


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    model_name: str
    benchmark_name: str
    timestamp: str
    total_samples: int
    results: List[Dict]
    summary: Dict[str, float]
    inference_time_seconds: float


class LLMEvaluator:
    """Main evaluation harness for LLM benchmarking."""

    def __init__(self, batch_size: int = 8, device: str = 'cpu'):
        """
        Initialize the evaluator.

        Args:
            batch_size: Batch size for inference
            device: Device to use ('cuda' or 'cpu')
        """
        self.batch_size = batch_size
        self.device = device
        self.metrics = MetricsComputer()

    def evaluate_model_on_benchmark(
        self,
        model_generator,
        benchmark_name: str,
        num_samples: Optional[int] = None
    ) -> EvaluationResult:
        """
        Evaluate a model on a specific benchmark.

        Args:
            model_generator: Callable that takes prompt and returns generated text
            benchmark_name: Name of benchmark to use
            num_samples: Number of samples to evaluate (None = all)

        Returns:
            EvaluationResult object
        """
        benchmark_data = BenchmarkDatasets.get_benchmark(benchmark_name)

        if num_samples:
            benchmark_data = benchmark_data[:num_samples]

        results = []
        start_time = time.time()

        # Get model name from generator if possible
        model_name = getattr(model_generator, '__name__', 'unknown')

        # Evaluate based on benchmark type
        if benchmark_name == 'truthful_qa':
            results = self._evaluate_truthful_qa(model_generator, benchmark_data)
        elif benchmark_name == 'instruction_following':
            results = self._evaluate_instruction_following(model_generator, benchmark_data)
        elif benchmark_name == 'commonsense':
            results = self._evaluate_commonsense(model_generator, benchmark_data)
        elif benchmark_name == 'hallucination':
            results = self._evaluate_hallucination(model_generator, benchmark_data)

        inference_time = time.time() - start_time

        # Compute summary statistics
        summary = self._compute_summary_stats(results, benchmark_name)

        return EvaluationResult(
            model_name=model_name,
            benchmark_name=benchmark_name,
            timestamp=datetime.now().isoformat(),
            total_samples=len(benchmark_data),
            results=results,
            summary=summary,
            inference_time_seconds=inference_time
        )

    def _evaluate_truthful_qa(
        self,
        model_generator,
        questions: List[BenchmarkQuestion]
    ) -> List[Dict]:
        """Evaluate on factual question-answering."""
        results = []

        for q in tqdm(questions, desc="TruthfulQA"):
            prompt = f"Q: {q.question}\nA:"
            try:
                generated = model_generator(prompt)
            except Exception as e:
                generated = f"[Error: {str(e)}]"

            # Compute metrics
            metrics = {}

            # ROUGE score
            if q.reference_answer:
                rouge = self.metrics.compute_rouge(
                    generated,
                    q.reference_answer
                )
                metrics.update(rouge)

                # BLEU score
                bleu = self.metrics.compute_bleu(
                    generated,
                    [q.reference_answer]
                )
                metrics['bleu'] = bleu

                # BERTScore-like similarity
                bertscore = self.metrics.compute_bertscore_simple(
                    generated,
                    q.reference_answer
                )
                metrics['bertscore'] = bertscore

            results.append({
                'question': q.question,
                'generated': generated,
                'reference': q.reference_answer,
                'category': q.category,
                **metrics
            })

        return results

    def _evaluate_instruction_following(
        self,
        model_generator,
        tasks: List[InstructionFollowingTask]
    ) -> List[Dict]:
        """Evaluate on instruction-following tasks."""
        results = []

        for task in tqdm(tasks, desc="Instruction Following"):
            # Build prompt
            if task.input_text:
                prompt = f"{task.instruction}\n\nInput: {task.input_text}\n\nOutput:"
            else:
                prompt = f"{task.instruction}\n\nOutput:"

            try:
                generated = model_generator(prompt)
            except Exception as e:
                generated = f"[Error: {str(e)}]"

            # Compute metrics
            metrics = {}

            # Instruction following score
            if_score = self.metrics.instruction_following_score(
                generated,
                task.instruction
            )
            metrics['instruction_following_score'] = if_score

            # ROUGE score
            if task.reference_output:
                rouge = self.metrics.compute_rouge(
                    generated,
                    task.reference_output
                )
                metrics.update(rouge)

                # BERTScore
                bertscore = self.metrics.compute_bertscore_simple(
                    generated,
                    task.reference_output
                )
                metrics['bertscore'] = bertscore

            results.append({
                'instruction': task.instruction,
                'generated': generated,
                'reference': task.reference_output,
                'category': task.category,
                **metrics
            })

        return results

    def _evaluate_commonsense(
        self,
        model_generator,
        questions: List[BenchmarkQuestion]
    ) -> List[Dict]:
        """Evaluate on commonsense reasoning."""
        results = []

        for q in tqdm(questions, desc="Commonsense"):
            prompt = f"Q: {q.question}\nA:"
            try:
                generated = model_generator(prompt)
            except Exception as e:
                generated = f"[Error: {str(e)}]"

            metrics = {}

            if q.reference_answer:
                rouge = self.metrics.compute_rouge(
                    generated,
                    q.reference_answer
                )
                metrics.update(rouge)

                bertscore = self.metrics.compute_bertscore_simple(
                    generated,
                    q.reference_answer
                )
                metrics['bertscore'] = bertscore

            results.append({
                'question': q.question,
                'generated': generated,
                'reference': q.reference_answer,
                'category': q.category,
                **metrics
            })

        return results

    def _evaluate_hallucination(
        self,
        model_generator,
        test_cases: List
    ) -> List[Dict]:
        """Evaluate hallucination detection."""
        results = []

        for tc in tqdm(test_cases, desc="Hallucination"):
            prompt = f"Context: {tc.context}\n\nQuestion: {tc.question}\n\nAnswer:"
            try:
                generated = model_generator(prompt)
            except Exception as e:
                generated = f"[Error: {str(e)}]"

            # Hallucination detection
            hall_metrics = self.metrics.hallucination_detection(
                generated,
                tc.context
            )

            # Factual accuracy
            accuracy = self.metrics.factual_accuracy_check(
                generated,
                tc.reference_facts
            )

            results.append({
                'question': tc.question,
                'context': tc.context,
                'generated': generated,
                'category': tc.category,
                **hall_metrics,
                **accuracy
            })

        return results

    def _compute_summary_stats(
        self,
        results: List[Dict],
        benchmark_name: str
    ) -> Dict[str, float]:
        """Compute summary statistics across all results."""
        if not results:
            return {}

        summary = {}

        # Common metrics
        for metric in ['rouge1_f1', 'rougeL_f1', 'bleu', 'bertscore',
                       'instruction_following_score']:
            values = [r.get(metric) for r in results if metric in r]
            if values:
                summary[f'{metric}_mean'] = float(np.mean(values))
                summary[f'{metric}_std'] = float(np.std(values))

        # Hallucination metrics
        if benchmark_name == 'hallucination':
            hall_scores = [r.get('hallucination_score', 0) for r in results]
            summary['hallucination_rate'] = float(np.mean(hall_scores))
            summary['grounding_score_mean'] = float(
                np.mean([r.get('grounding_score', 0) for r in results])
            )
            accuracy_scores = [r.get('accuracy', 0) for r in results]
            summary['factual_accuracy_mean'] = float(np.mean(accuracy_scores))

        return summary

    def export_results_json(
        self,
        result: EvaluationResult,
        output_path: str
    ) -> None:
        """Export results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'model_name': result.model_name,
            'benchmark_name': result.benchmark_name,
            'timestamp': result.timestamp,
            'total_samples': result.total_samples,
            'inference_time_seconds': result.inference_time_seconds,
            'summary': result.summary,
            'results': result.results
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def export_results_csv(
        self,
        result: EvaluationResult,
        output_path: str
    ) -> None:
        """Export results to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not result.results:
            return

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=result.results[0].keys())
            writer.writeheader()
            writer.writerows(result.results)

    def generate_html_report(
        self,
        result: EvaluationResult,
        output_path: str
    ) -> None:
        """Generate an HTML report of results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self._build_html_report(result)

        with open(output_path, 'w') as f:
            f.write(html)

    def _build_html_report(self, result: EvaluationResult) -> str:
        """Build HTML content for report."""
        summary_rows = ''.join([
            f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
            for k, v in result.summary.items()
        ])

        sample_results = result.results[:5]  # Show first 5 samples
        detail_rows = ''.join([
            self._build_result_row(r) for r in sample_results
        ])

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>LLM Evaluation Report</h1>
            <div class="summary">
                <p><strong>Model:</strong> {result.model_name}</p>
                <p><strong>Benchmark:</strong> {result.benchmark_name}</p>
                <p><strong>Samples Evaluated:</strong> {result.total_samples}</p>
                <p><strong>Time Taken:</strong> {result.inference_time_seconds:.2f}s</p>
                <p><strong>Timestamp:</strong> {result.timestamp}</p>
            </div>

            <h2>Summary Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {summary_rows}
            </table>

            <h2>Sample Results (First 5)</h2>
            <table>
                <tr><th>Input</th><th>Generated</th><th>Reference</th></tr>
                {detail_rows}
            </table>
        </body>
        </html>
        """

        return html

    def _build_result_row(self, result: Dict) -> str:
        """Build HTML row for a single result."""
        input_key = next(
            (k for k in result.keys()
             if k in ['question', 'instruction']),
            'N/A'
        )
        input_text = result.get(input_key, 'N/A')[:100]
        generated = result.get('generated', '')[:100]
        reference = result.get('reference', '')[:100]

        return f"""
        <tr>
            <td>{input_text}</td>
            <td>{generated}</td>
            <td>{reference}</td>
        </tr>
        """


class BatchInferenceHelper:
    """Helper for efficient batch inference."""

    @staticmethod
    def batch_generate(
        texts: List[str],
        model_generator,
        batch_size: int = 8
    ) -> List[str]:
        """
        Perform batch inference on multiple texts.

        Args:
            texts: List of input texts
            model_generator: Model generation function
            batch_size: Batch size for processing

        Returns:
            List of generated outputs
        """
        outputs = []

        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            batch_outputs = [model_generator(text) for text in batch]
            outputs.extend(batch_outputs)

        return outputs
