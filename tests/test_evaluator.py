"""
Unit tests for the LLM evaluation framework.
Uses mocked models to avoid downloading dependencies.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import List

from src.evaluation import (
    MetricsComputer,
    BenchmarkDatasets,
    LLMEvaluator,
    EvaluationResult
)


class TestMetricsComputer:
    """Test metrics computation."""

    @pytest.fixture
    def metrics(self):
        """Create metrics computer instance."""
        return MetricsComputer()

    def test_compute_rouge(self, metrics):
        """Test ROUGE score computation."""
        prediction = "The capital of France is Paris"
        reference = "Paris is the capital of France"

        result = metrics.compute_rouge(prediction, reference)

        assert 'rouge1_f1' in result
        assert 'rougeL_f1' in result
        assert 0 <= result['rouge1_f1'] <= 1
        assert 0 <= result['rougeL_f1'] <= 1

    def test_compute_bleu(self, metrics):
        """Test BLEU score computation."""
        prediction = "The cat sat on the mat"
        references = ["The cat sat on the mat", "A cat was on a mat"]

        result = metrics.compute_bleu(prediction, references)

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_bertscore_simple(self, metrics):
        """Test simplified BERTScore."""
        text1 = "The quick brown fox"
        text2 = "The quick fox is brown"

        result = metrics.compute_bertscore_simple(text1, text2)

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_instruction_following_score(self, metrics):
        """Test instruction following scoring."""
        prediction = "• Item 1\n• Item 2\n• Item 3"
        instruction = "List three items in bullet points"

        result = metrics.instruction_following_score(prediction, instruction)

        assert isinstance(result, float)
        assert 0 <= result <= 1
        assert result > 0.5  # Should score well for bullet points

    def test_hallucination_detection(self, metrics):
        """Test hallucination detection."""
        prediction = "Paris is the capital of France"
        context = "France is located in Europe. Paris is the capital."

        result = metrics.hallucination_detection(prediction, context)

        assert 'hallucination_score' in result
        assert 'grounding_score' in result
        assert 'confidence' in result
        assert 0 <= result['hallucination_score'] <= 1

    def test_factual_accuracy_check(self, metrics):
        """Test factual accuracy checking."""
        prediction = "Einstein developed the theory of relativity"
        facts = ["Einstein", "relativity", "physics"]

        result = metrics.factual_accuracy_check(prediction, facts)

        assert 'accuracy' in result
        assert 'matches' in result
        assert result['accuracy'] >= 0.5

    def test_perplexity_from_logits(self, metrics):
        """Test perplexity computation."""
        log_probs = np.array([-1.0, -1.2, -0.8, -1.5])

        result = metrics.perplexity_from_logits(log_probs)

        assert isinstance(result, float)
        assert result > 0
        assert result < 10  # Reasonable perplexity range

    def test_semantic_similarity(self, metrics):
        """Test semantic similarity."""
        text1 = "The cat is sleeping"
        text2 = "The cat is sleeping"

        result = metrics.semantic_similarity(text1, text2)

        assert result == 1.0  # Identical texts

        text3 = "The dog is running"
        result2 = metrics.semantic_similarity(text1, text3)

        assert 0 <= result2 < 1  # Different texts


class TestBenchmarkDatasets:
    """Test benchmark dataset loading."""

    def test_get_truthful_qa_sample(self):
        """Test TruthfulQA dataset."""
        data = BenchmarkDatasets.get_truthful_qa_sample()

        assert len(data) > 0
        assert hasattr(data[0], 'question')
        assert hasattr(data[0], 'reference_answer')

    def test_get_instruction_following_sample(self):
        """Test instruction following dataset."""
        data = BenchmarkDatasets.get_instruction_following_sample()

        assert len(data) > 0
        assert hasattr(data[0], 'instruction')
        assert hasattr(data[0], 'reference_output')

    def test_get_commonsense_sample(self):
        """Test commonsense dataset."""
        data = BenchmarkDatasets.get_commonsense_sample()

        assert len(data) > 0
        assert all(hasattr(item, 'question') for item in data)

    def test_get_hallucination_testcases(self):
        """Test hallucination test cases."""
        data = BenchmarkDatasets.get_hallucination_testcases()

        assert len(data) > 0
        assert all(hasattr(item, 'question') for item in data)
        assert all(hasattr(item, 'context') for item in data)

    def test_get_benchmark(self):
        """Test generic benchmark loader."""
        for benchmark_name in BenchmarkDatasets.list_benchmarks():
            data = BenchmarkDatasets.get_benchmark(benchmark_name)
            assert len(data) > 0

    def test_invalid_benchmark(self):
        """Test invalid benchmark raises error."""
        with pytest.raises(ValueError):
            BenchmarkDatasets.get_benchmark('invalid_benchmark')

    def test_list_benchmarks(self):
        """Test listing available benchmarks."""
        benchmarks = BenchmarkDatasets.list_benchmarks()

        assert isinstance(benchmarks, list)
        assert len(benchmarks) > 0
        assert all(isinstance(b, str) for b in benchmarks)


class TestLLMEvaluator:
    """Test LLM evaluator."""

    @pytest.fixture
    def mock_model_generator(self):
        """Create mock model generator."""
        def generator(prompt: str) -> str:
            # Simple mock: return a fixed response
            if "question" in prompt.lower() or "q:" in prompt.lower():
                return "Paris"
            elif "haiku" in prompt.lower():
                return "Morning dew falls\nBirds sing their songs\nWinter fades"
            else:
                return "Generated text response"

        generator.__name__ = "mock_model"
        return generator

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return LLMEvaluator(batch_size=4, device='cpu')

    def test_evaluate_truthful_qa(self, evaluator, mock_model_generator):
        """Test TruthfulQA evaluation."""
        result = evaluator.evaluate_model_on_benchmark(
            model_generator=mock_model_generator,
            benchmark_name='truthful_qa',
            num_samples=5
        )

        assert isinstance(result, EvaluationResult)
        assert result.benchmark_name == 'truthful_qa'
        assert len(result.results) == 5
        assert result.total_samples == 5
        assert result.inference_time_seconds > 0

    def test_evaluate_instruction_following(self, evaluator, mock_model_generator):
        """Test instruction following evaluation."""
        result = evaluator.evaluate_model_on_benchmark(
            model_generator=mock_model_generator,
            benchmark_name='instruction_following',
            num_samples=3
        )

        assert result.benchmark_name == 'instruction_following'
        assert len(result.results) == 3

    def test_evaluate_commonsense(self, evaluator, mock_model_generator):
        """Test commonsense evaluation."""
        result = evaluator.evaluate_model_on_benchmark(
            model_generator=mock_model_generator,
            benchmark_name='commonsense',
            num_samples=3
        )

        assert result.benchmark_name == 'commonsense'
        assert len(result.results) == 3

    def test_evaluate_hallucination(self, evaluator, mock_model_generator):
        """Test hallucination evaluation."""
        result = evaluator.evaluate_model_on_benchmark(
            model_generator=mock_model_generator,
            benchmark_name='hallucination',
            num_samples=2
        )

        assert result.benchmark_name == 'hallucination'
        assert len(result.results) == 2

    def test_summary_statistics(self, evaluator, mock_model_generator):
        """Test summary statistics computation."""
        result = evaluator.evaluate_model_on_benchmark(
            model_generator=mock_model_generator,
            benchmark_name='truthful_qa',
            num_samples=5
        )

        assert 'rouge1_f1_mean' in result.summary or len(result.summary) == 0
        for key, value in result.summary.items():
            assert isinstance(value, (int, float))

    def test_export_json(self, evaluator, mock_model_generator, tmp_path):
        """Test JSON export."""
        result = evaluator.evaluate_model_on_benchmark(
            model_generator=mock_model_generator,
            benchmark_name='truthful_qa',
            num_samples=3
        )

        output_file = tmp_path / "results.json"
        evaluator.export_results_json(result, str(output_file))

        assert output_file.exists()

    def test_export_csv(self, evaluator, mock_model_generator, tmp_path):
        """Test CSV export."""
        result = evaluator.evaluate_model_on_benchmark(
            model_generator=mock_model_generator,
            benchmark_name='instruction_following',
            num_samples=2
        )

        output_file = tmp_path / "results.csv"
        evaluator.export_results_csv(result, str(output_file))

        assert output_file.exists()

    def test_html_report(self, evaluator, mock_model_generator, tmp_path):
        """Test HTML report generation."""
        result = evaluator.evaluate_model_on_benchmark(
            model_generator=mock_model_generator,
            benchmark_name='truthful_qa',
            num_samples=3
        )

        output_file = tmp_path / "report.html"
        evaluator.generate_html_report(result, str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert 'Evaluation Report' in content or '>' in content


class TestBatchInferenceHelper:
    """Test batch inference helper."""

    def test_batch_generate(self):
        """Test batch generation."""
        from src.evaluation import BatchInferenceHelper

        prompts = ["prompt1", "prompt2", "prompt3"]

        def mock_generator(prompt: str) -> str:
            return f"Response to {prompt}"

        outputs = BatchInferenceHelper.batch_generate(
            prompts,
            mock_generator,
            batch_size=2
        )

        assert len(outputs) == 3
        assert all(isinstance(o, str) for o in outputs)


@pytest.mark.integration
class TestEndToEndEvaluation:
    """End-to-end evaluation tests."""

    def test_full_evaluation_pipeline(self, tmp_path):
        """Test complete evaluation pipeline."""
        from src.evaluation import LLMEvaluator

        # Create mock generator
        def generator(prompt):
            return "Test response"

        generator.__name__ = "test_model"

        # Run evaluation
        evaluator = LLMEvaluator()
        result = evaluator.evaluate_model_on_benchmark(
            model_generator=generator,
            benchmark_name='truthful_qa',
            num_samples=2
        )

        # Export results
        evaluator.export_results_json(result, str(tmp_path / "results.json"))
        evaluator.export_results_csv(result, str(tmp_path / "results.csv"))
        evaluator.generate_html_report(result, str(tmp_path / "report.html"))

        # Verify files exist
        assert (tmp_path / "results.json").exists()
        assert (tmp_path / "results.csv").exists()
        assert (tmp_path / "report.html").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
