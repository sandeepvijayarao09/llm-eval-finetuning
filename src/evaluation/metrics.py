"""
Evaluation metrics for LLM outputs.
Includes ROUGE, BLEU, BERTScore, hallucination detection, and factual accuracy.
"""

import re
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class MetricsComputer:
    """Compute various evaluation metrics for LLM outputs."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rougeL'],
            use_stemmer=True
        )

    def compute_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Compute ROUGE scores (ROUGE-1 and ROUGE-L).

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            Dictionary with rouge1_f1, rougeL_f1 scores
        """
        rouge1 = self.rouge_scorer.score(reference, prediction)['rouge1']
        rougeL = self.rouge_scorer.score(reference, prediction)['rougeL']

        return {
            'rouge1_f1': rouge1.fmeasure,
            'rougeL_f1': rougeL.fmeasure,
        }

    def compute_bleu(self, prediction: str, references: List[str],
                     max_n: int = 4) -> float:
        """
        Compute BLEU score (corpus-level).

        Args:
            prediction: Generated text
            references: List of reference texts
            max_n: Maximum n-gram order

        Returns:
            BLEU score
        """
        pred_tokens = word_tokenize(prediction.lower())
        ref_tokens_list = [word_tokenize(ref.lower()) for ref in references]

        weights = tuple([1/max_n] * max_n)
        smooth_fn = SmoothingFunction().method1

        try:
            return sentence_bleu(
                ref_tokens_list,
                pred_tokens,
                weights=weights,
                smoothing_function=smooth_fn
            )
        except Exception:
            return 0.0

    def compute_bertscore_simple(self, prediction: str, reference: str) -> float:
        """
        Simplified BERTScore using token overlap (F1).
        Production code should use bert-score package.

        Args:
            prediction: Generated text
            reference: Reference text

        Returns:
            Simple token-based F1 score
        """
        pred_tokens = set(word_tokenize(prediction.lower()))
        ref_tokens = set(word_tokenize(reference.lower()))

        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0

        intersection = len(pred_tokens & ref_tokens)
        recall = intersection / len(ref_tokens) if ref_tokens else 0.0
        precision = intersection / len(pred_tokens) if pred_tokens else 0.0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def instruction_following_score(self,
                                   prediction: str,
                                   instruction: str) -> float:
        """
        Score instruction following based on format adherence.

        Checks for:
        - Non-empty output
        - Reasonable length
        - Format markers if present in instruction

        Args:
            prediction: Generated text
            instruction: Original instruction

        Returns:
            Score between 0 and 1
        """
        score = 0.0
        checks = []

        # Check 1: Non-empty
        if prediction.strip():
            checks.append(1.0)
        else:
            checks.append(0.0)

        # Check 2: Reasonable length (not too short, not obviously truncated)
        words = len(prediction.split())
        if 5 <= words <= 500:
            checks.append(1.0)
        elif words > 500:
            checks.append(0.9)  # Penalize verbosity
        else:
            checks.append(0.5)

        # Check 3: Format compliance
        format_keywords = ['list', 'table', 'json', 'bullet', 'numbered', 'step']
        has_format = any(kw in instruction.lower() for kw in format_keywords)

        if has_format:
            # Check for list/bullet formatting
            has_bullets = any(c in prediction for c in ['•', '-', '*', '1.', '2.'])
            checks.append(1.0 if has_bullets else 0.6)
        else:
            checks.append(1.0)  # No format requirement

        # Check 4: No obvious cut-offs
        if prediction.rstrip().endswith(('.', '!', '?', ')', ']')):
            checks.append(1.0)
        elif len(prediction.split()) > 20:
            checks.append(0.8)  # Might be complete
        else:
            checks.append(0.5)

        return np.mean(checks)

    def hallucination_detection(self,
                               prediction: str,
                               context: str,
                               nli_threshold: float = 0.5) -> Dict[str, float]:
        """
        Detect hallucinations using simple NLI-inspired logic.
        Production should use real NLI model.

        Args:
            prediction: Generated text
            context: Reference context/facts
            nli_threshold: Threshold for marking as hallucination

        Returns:
            Dictionary with hallucination_score, confidence
        """
        if not context:
            return {'hallucination_score': 0.0, 'confidence': 0.0}

        # Extract key entities/facts from prediction and context
        pred_facts = self._extract_key_phrases(prediction)
        context_facts = self._extract_key_phrases(context)

        if not pred_facts:
            return {'hallucination_score': 0.0, 'confidence': 0.0}

        # Check overlap
        overlap = len(set(pred_facts) & set(context_facts))
        total = len(pred_facts)

        grounding_score = overlap / total if total > 0 else 1.0
        hallucination_score = 1.0 - grounding_score

        # Confidence based on context length
        confidence = min(1.0, len(context.split()) / 50.0)

        return {
            'hallucination_score': float(hallucination_score),
            'grounding_score': float(grounding_score),
            'confidence': float(confidence)
        }

    def _extract_key_phrases(self, text: str, n: int = 2) -> List[str]:
        """Extract n-grams as key phrases."""
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if len(t) > 2]  # Filter short tokens

        phrases = []
        for i in range(len(tokens) - n + 1):
            phrase = ' '.join(tokens[i:i+n])
            phrases.append(phrase)

        return phrases

    def factual_accuracy_check(self,
                              prediction: str,
                              facts: List[str]) -> Dict[str, float]:
        """
        Check factual accuracy against known facts.

        Args:
            prediction: Generated text
            facts: List of true facts to check against

        Returns:
            Dictionary with accuracy score and details
        """
        if not facts:
            return {'accuracy': 0.5, 'matches': 0, 'total': 0}

        pred_lower = prediction.lower()
        matches = 0

        for fact in facts:
            # Simple substring matching (production should use better NLI)
            fact_lower = fact.lower()
            # Check for fact or key parts of it
            key_words = [w for w in fact_lower.split() if len(w) > 3]
            if key_words and any(word in pred_lower for word in key_words):
                matches += 1

        accuracy = matches / len(facts) if facts else 0.0

        return {
            'accuracy': float(accuracy),
            'matches': int(matches),
            'total': int(len(facts))
        }

    def perplexity_from_logits(self, log_probs: np.ndarray) -> float:
        """
        Compute perplexity from log probabilities.

        Args:
            log_probs: Array of log probabilities (tokens)

        Returns:
            Perplexity score
        """
        if len(log_probs) == 0:
            return float('inf')

        # Avoid division by zero
        log_probs = np.array(log_probs)
        cross_entropy = -np.mean(log_probs)
        perplexity = np.exp(cross_entropy)

        return float(perplexity)

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity using token overlap.
        Production should use sentence transformers.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        tokens1 = set(word_tokenize(text1.lower()))
        tokens2 = set(word_tokenize(text2.lower()))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0
