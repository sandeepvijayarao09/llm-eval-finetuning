"""
Optimized inference utilities.
Includes batched generation, KV cache management, and performance profiling.
"""

import torch
import time
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OptimizedInferenceEngine:
    """
    High-performance inference engine with caching and batching.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda',
        max_batch_size: int = 32,
        enable_kv_cache: bool = True
    ):
        """
        Initialize inference engine.

        Args:
            model: Model for inference
            tokenizer: Tokenizer
            device: Device to use
            max_batch_size: Maximum batch size
            enable_kv_cache: Enable KV cache
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.enable_kv_cache = enable_kv_cache
        self.model.eval()

        self.stats = {
            'total_inference_time': 0.0,
            'total_tokens_generated': 0,
            'num_inferences': 0,
        }

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 1
    ) -> List[str]:
        """
        Generate text for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_beams: Number of beams for beam search

        Returns:
            List of generated texts
        """
        outputs = []
        start_time = time.time()

        for i in range(0, len(prompts), self.max_batch_size):
            batch_prompts = prompts[i:i+self.max_batch_size]
            batch_outputs = self._generate_batch_internal(
                batch_prompts,
                max_new_tokens,
                temperature,
                top_p,
                num_beams
            )
            outputs.extend(batch_outputs)

        inference_time = time.time() - start_time
        self.stats['total_inference_time'] += inference_time
        self.stats['num_inferences'] += len(prompts)

        return outputs

    def _generate_batch_internal(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        num_beams: int
    ) -> List[str]:
        """Internal batch generation implementation."""
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=num_beams == 1,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=self.enable_kv_cache,
            )

        # Decode
        outputs = self.tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )

        # Track tokens
        num_new_tokens = output_ids.shape[1] - inputs['input_ids'].shape[1]
        self.stats['total_tokens_generated'] += num_new_tokens * len(prompts)

        return outputs

    def generate_single(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text for a single prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling

        Returns:
            Generated text
        """
        outputs = self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return outputs[0]

    def benchmark_throughput(
        self,
        prompts: List[str],
        num_runs: int = 3,
        max_new_tokens: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference throughput.

        Args:
            prompts: List of test prompts
            num_runs: Number of benchmark runs
            max_new_tokens: Tokens per generation

        Returns:
            Dictionary with throughput metrics
        """
        times = []
        total_tokens = 0

        for _ in range(num_runs):
            start_time = time.time()
            outputs = self.generate_batch(
                prompts,
                max_new_tokens=max_new_tokens
            )
            elapsed = time.time() - start_time
            times.append(elapsed)

            # Estimate tokens generated
            total_tokens = sum(
                len(self.tokenizer.encode(text))
                for text in outputs
            )

        avg_time = sum(times) / len(times)
        tokens_per_second = total_tokens / avg_time

        return {
            'avg_inference_time_s': avg_time,
            'tokens_per_second': tokens_per_second,
            'samples_per_second': len(prompts) / avg_time,
            'latency_per_sample_ms': (avg_time / len(prompts)) * 1000,
        }

    def get_stats(self) -> Dict:
        """Get inference statistics."""
        if self.stats['num_inferences'] == 0:
            return self.stats

        return {
            **self.stats,
            'avg_time_per_inference': (
                self.stats['total_inference_time'] /
                self.stats['num_inferences']
            ),
            'avg_tokens_per_inference': (
                self.stats['total_tokens_generated'] /
                self.stats['num_inferences']
            ),
        }


class KVCacheManager:
    """Manage KV cache for efficient inference."""

    def __init__(self, max_cache_size_mb: int = 1024):
        """
        Initialize cache manager.

        Args:
            max_cache_size_mb: Maximum cache size in MB
        """
        self.max_cache_size = max_cache_size_mb * 1e6  # Convert to bytes
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_key(self, prompt: str) -> str:
        """Generate cache key for a prompt."""
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve cached KV states."""
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        return None

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store KV states in cache."""
        # Simple LRU-like eviction
        current_size = sum(
            v.numel() * 4 for v in self.cache.values()
        )  # Assume float32

        if current_size + value.numel() * 4 > self.max_cache_size:
            # Evict oldest entry
            if self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

        self.cache[key] = value

    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class LatencyProfiler:
    """Profile latency of model components."""

    def __init__(self):
        """Initialize profiler."""
        self.profiles = {}

    def profile_forward_pass(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Profile forward pass latency.

        Args:
            model: Model to profile
            input_ids: Input tensor
            attention_mask: Attention mask
            num_runs: Number of runs for averaging

        Returns:
            Latency metrics
        """
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = model(input_ids, attention_mask)

        # Measure
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_ids, attention_mask)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start

        return {
            'forward_pass_ms': (elapsed / num_runs) * 1000,
            'total_time_s': elapsed,
            'runs': num_runs
        }

    def compare_latency(
        self,
        model1,
        model2,
        test_input: torch.Tensor,
        test_mask: torch.Tensor,
        num_runs: int = 10
    ) -> Dict:
        """Compare latency between two models."""
        profile1 = self.profile_forward_pass(
            model1, test_input, test_mask, num_runs
        )
        profile2 = self.profile_forward_pass(
            model2, test_input, test_mask, num_runs
        )

        return {
            'model1': profile1,
            'model2': profile2,
            'speedup': profile1['forward_pass_ms'] / profile2['forward_pass_ms']
        }
