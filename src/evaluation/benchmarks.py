"""
Built-in benchmark datasets for LLM evaluation.
Includes TruthfulQA-style questions, instruction-following, and commonsense tasks.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class BenchmarkQuestion:
    """A benchmark question with metadata."""
    question: str
    reference_answer: Optional[str] = None
    context: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"
    answer_choices: Optional[List[str]] = None


@dataclass
class InstructionFollowingTask:
    """An instruction-following task with reference output."""
    instruction: str
    input_text: Optional[str] = None
    reference_output: str = ""
    category: str = "general"


@dataclass
class HallucineationTestCase:
    """A test case for hallucination detection."""
    question: str
    context: str
    reference_facts: List[str]
    category: str = "factual"


class BenchmarkDatasets:
    """Provides access to built-in benchmark datasets."""

    @staticmethod
    def get_truthful_qa_sample() -> List[BenchmarkQuestion]:
        """
        Return a sample of TruthfulQA-style questions.
        Focus on questions requiring factual knowledge.
        """
        return [
            BenchmarkQuestion(
                question="What is the capital of France?",
                reference_answer="Paris",
                category="factual",
                difficulty="easy"
            ),
            BenchmarkQuestion(
                question="In what year did the Titanic sink?",
                reference_answer="1912",
                category="factual",
                difficulty="easy"
            ),
            BenchmarkQuestion(
                question="Who wrote 'Pride and Prejudice'?",
                reference_answer="Jane Austen",
                category="factual",
                difficulty="easy"
            ),
            BenchmarkQuestion(
                question="What is the chemical symbol for gold?",
                reference_answer="Au",
                category="factual",
                difficulty="easy"
            ),
            BenchmarkQuestion(
                question="How many continents are there?",
                reference_answer="7 (or 6 depending on geographic classification)",
                category="factual",
                difficulty="easy"
            ),
            BenchmarkQuestion(
                question="What is the largest planet in our solar system?",
                reference_answer="Jupiter",
                category="factual",
                difficulty="medium"
            ),
            BenchmarkQuestion(
                question="In what country was the first modern Olympics held?",
                reference_answer="Greece (Athens, 1896)",
                category="factual",
                difficulty="medium"
            ),
            BenchmarkQuestion(
                question="What is the speed of light in a vacuum?",
                reference_answer="Approximately 299,792,458 meters per second",
                category="factual",
                difficulty="medium"
            ),
            BenchmarkQuestion(
                question="Who invented the telephone?",
                reference_answer="Alexander Graham Bell",
                category="factual",
                difficulty="medium"
            ),
            BenchmarkQuestion(
                question="What is the most spoken language in the world by native speakers?",
                reference_answer="Mandarin Chinese",
                category="factual",
                difficulty="medium"
            ),
        ]

    @staticmethod
    def get_instruction_following_sample() -> List[InstructionFollowingTask]:
        """
        Return a sample of instruction-following tasks with expected outputs.
        """
        return [
            InstructionFollowingTask(
                instruction="List 3 benefits of exercise in bullet points.",
                reference_output="• Improves cardiovascular health\n• Increases muscle strength\n• Enhances mental well-being",
                category="formatting"
            ),
            InstructionFollowingTask(
                instruction="Write a haiku about nature.",
                reference_output="Soft morning dew falls\nBirds sing their gentle sweet songs\nWinter fades to spring",
                category="creative"
            ),
            InstructionFollowingTask(
                instruction="Explain photosynthesis in 2-3 sentences.",
                reference_output="Photosynthesis is a process where plants convert light energy into chemical energy. Using chlorophyll, plants absorb sunlight and combine it with water and carbon dioxide to produce glucose and oxygen. This process is essential for life on Earth as it produces the oxygen we breathe.",
                category="explanation"
            ),
            InstructionFollowingTask(
                instruction="Translate this sentence to Spanish: 'Hello, how are you?'",
                reference_output="¿Hola, cómo estás? (or ¿Hola, cómo está usted? for formal)",
                category="translation"
            ),
            InstructionFollowingTask(
                instruction="Write a Python function that returns the sum of two numbers.",
                reference_output="def sum_two_numbers(a, b):\n    return a + b",
                category="coding"
            ),
            InstructionFollowingTask(
                instruction="Summarize the main idea of this in 1 sentence: Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms and is widely used in web development, data science, and artificial intelligence.",
                reference_output="Python is a versatile, readable programming language widely used in web development, data science, and AI.",
                category="summarization"
            ),
            InstructionFollowingTask(
                instruction="Answer with only 'Yes' or 'No': Is water essential for human life?",
                reference_output="Yes",
                category="constraint"
            ),
            InstructionFollowingTask(
                instruction="Create a numbered list of the steps to make tea.",
                reference_output="1. Boil water\n2. Place tea bag in cup\n3. Pour hot water\n4. Wait 3-5 minutes\n5. Remove tea bag\n6. Add milk or honey if desired\n7. Enjoy",
                category="formatting"
            ),
        ]

    @staticmethod
    def get_commonsense_sample() -> List[BenchmarkQuestion]:
        """
        Return a sample of commonsense reasoning tasks (HellaSwag-style).
        """
        return [
            BenchmarkQuestion(
                question="If you want to keep plants alive, what must you do regularly?",
                reference_answer="Water them",
                context="Plant care is essential for maintaining healthy plants.",
                category="commonsense",
                difficulty="easy"
            ),
            BenchmarkQuestion(
                question="What should you do if you spill water on your computer?",
                reference_answer="Turn it off immediately and let it dry",
                context="Computers and water do not mix well.",
                category="commonsense",
                difficulty="easy"
            ),
            BenchmarkQuestion(
                question="Why do we need sleep?",
                reference_answer="Sleep allows our bodies and brains to rest and recover",
                context="Sleep is a biological necessity.",
                category="commonsense",
                difficulty="easy"
            ),
            BenchmarkQuestion(
                question="What happens to ice when exposed to heat?",
                reference_answer="It melts into water",
                category="commonsense",
                difficulty="easy"
            ),
            BenchmarkQuestion(
                question="If someone is hungry and hasn't eaten all day, what should they do?",
                reference_answer="Eat something nutritious",
                category="commonsense",
                difficulty="easy"
            ),
            BenchmarkQuestion(
                question="Why do we wear clothes in winter?",
                reference_answer="To keep warm and maintain body temperature",
                category="commonsense",
                difficulty="easy"
            ),
            BenchmarkQuestion(
                question="What should you do if you make a mistake in an important document?",
                reference_answer="Correct it and ideally save a new version",
                category="commonsense",
                difficulty="medium"
            ),
            BenchmarkQuestion(
                question="Why is it important to stretch before exercise?",
                reference_answer="To warm up muscles and reduce injury risk",
                category="commonsense",
                difficulty="medium"
            ),
        ]

    @staticmethod
    def get_hallucination_testcases() -> List[HallucineationTestCase]:
        """
        Return test cases for hallucination detection.
        """
        return [
            HallucineationTestCase(
                question="What did Albert Einstein win a Nobel Prize for?",
                context="Albert Einstein won the Nobel Prize in Physics in 1921.",
                reference_facts=[
                    "Albert Einstein",
                    "Nobel Prize",
                    "Physics",
                    "1921"
                ],
                category="factual"
            ),
            HallucineationTestCase(
                question="When was Python programming language created?",
                context="Python was created in 1989 by Guido van Rossum.",
                reference_facts=[
                    "Python",
                    "1989",
                    "Guido van Rossum",
                    "programming language"
                ],
                category="factual"
            ),
            HallucineationTestCase(
                question="What is the main ingredient in chocolate?",
                context="Chocolate is primarily made from cacao beans.",
                reference_facts=[
                    "chocolate",
                    "cacao beans",
                    "ingredient"
                ],
                category="factual"
            ),
            HallucineationTestCase(
                question="Who wrote the Harry Potter series?",
                context="The Harry Potter series was written by J.K. Rowling.",
                reference_facts=[
                    "Harry Potter",
                    "J.K. Rowling",
                    "written",
                    "series"
                ],
                category="factual"
            ),
        ]

    @staticmethod
    def get_benchmark(name: str) -> List[Dict]:
        """
        Load a benchmark dataset by name.

        Args:
            name: Benchmark name ('truthful_qa', 'instruction_following', 'commonsense', 'hallucination')

        Returns:
            List of benchmark items
        """
        if name == 'truthful_qa':
            return BenchmarkDatasets.get_truthful_qa_sample()
        elif name == 'instruction_following':
            return BenchmarkDatasets.get_instruction_following_sample()
        elif name == 'commonsense':
            return BenchmarkDatasets.get_commonsense_sample()
        elif name == 'hallucination':
            return BenchmarkDatasets.get_hallucination_testcases()
        else:
            raise ValueError(f"Unknown benchmark: {name}")

    @staticmethod
    def list_benchmarks() -> List[str]:
        """List all available benchmarks."""
        return ['truthful_qa', 'instruction_following', 'commonsense', 'hallucination']
