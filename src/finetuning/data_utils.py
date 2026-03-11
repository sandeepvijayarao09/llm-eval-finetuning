"""
Data utilities for fine-tuning.
Handles tokenization, dataset preprocessing, and prompt template formatting.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass
class PreferenceDataset:
    """A preference pair for DPO training."""
    prompt: str
    chosen: str
    rejected: str
    metadata: Optional[Dict] = None


class PromptTemplate:
    """Prompt template formatter for different model families."""

    @staticmethod
    def alpaca_template(instruction: str, input_text: str = "", output: str = "") -> str:
        """
        Format prompt using Alpaca template.
        """
        if input_text:
            template = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            template = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
        return template

    @staticmethod
    def chatml_template(instruction: str, input_text: str = "", output: str = "") -> str:
        """
        Format prompt using ChatML template.
        """
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        template = f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
        return template

    @staticmethod
    def mistral_template(instruction: str, input_text: str = "", output: str = "") -> str:
        """
        Format prompt using Mistral template.
        """
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        template = f"""[INST] {prompt} [/INST] {output}"""
        return template

    @staticmethod
    def format_prompt(
        template_type: str,
        instruction: str,
        input_text: str = "",
        output: str = ""
    ) -> str:
        """
        Format prompt using specified template.

        Args:
            template_type: 'alpaca', 'chatml', or 'mistral'
            instruction: Instruction text
            input_text: Optional input text
            output: Optional output text

        Returns:
            Formatted prompt
        """
        if template_type == 'alpaca':
            return PromptTemplate.alpaca_template(instruction, input_text, output)
        elif template_type == 'chatml':
            return PromptTemplate.chatml_template(instruction, input_text, output)
        elif template_type == 'mistral':
            return PromptTemplate.mistral_template(instruction, input_text, output)
        else:
            raise ValueError(f"Unknown template type: {template_type}")


class TokenizationHelper:
    """Helper for tokenization and padding."""

    def __init__(self, tokenizer, max_length: int = 512):
        """
        Initialize tokenization helper.

        Args:
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenize_text(self, text: str) -> Dict:
        """
        Tokenize text with padding and truncation.

        Args:
            text: Input text

        Returns:
            Dictionary with input_ids and attention_mask
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded

    def tokenize_pair(self, text1: str, text2: str) -> Tuple[Dict, Dict]:
        """
        Tokenize a pair of texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Tuple of tokenized outputs
        """
        enc1 = self.tokenize_text(text1)
        enc2 = self.tokenize_text(text2)
        return enc1, enc2

    def pad_batch(self, batch: List[Dict], pad_token_id: int = 0) -> Dict:
        """
        Pad a batch of sequences.

        Args:
            batch: List of tokenized sequences
            pad_token_id: ID to use for padding

        Returns:
            Padded batch
        """
        max_len = max(seq['input_ids'].shape[1] for seq in batch)

        padded_batch = {
            'input_ids': [],
            'attention_mask': []
        }

        for seq in batch:
            current_len = seq['input_ids'].shape[1]
            if current_len < max_len:
                pad_length = max_len - current_len
                padded_ids = np.pad(
                    seq['input_ids'].numpy()[0],
                    (0, pad_length),
                    constant_values=pad_token_id
                )
                padded_mask = np.pad(
                    seq['attention_mask'].numpy()[0],
                    (0, pad_length),
                    constant_values=0
                )
            else:
                padded_ids = seq['input_ids'].numpy()[0]
                padded_mask = seq['attention_mask'].numpy()[0]

            padded_batch['input_ids'].append(padded_ids)
            padded_batch['attention_mask'].append(padded_mask)

        padded_batch['input_ids'] = np.array(padded_batch['input_ids'])
        padded_batch['attention_mask'] = np.array(padded_batch['attention_mask'])

        return padded_batch


class DPODatasetProcessor:
    """Process preference datasets for DPO training."""

    def __init__(self, template_type: str = 'alpaca'):
        """
        Initialize processor.

        Args:
            template_type: Prompt template to use
        """
        self.template_type = template_type

    def create_preference_pairs(
        self,
        instruction: str,
        input_text: str,
        chosen_output: str,
        rejected_output: str
    ) -> PreferenceDataset:
        """
        Create a preference pair for DPO training.

        Args:
            instruction: Task instruction
            input_text: Input text (can be empty)
            chosen_output: Preferred output
            rejected_output: Non-preferred output

        Returns:
            PreferenceDataset object
        """
        prompt = PromptTemplate.format_prompt(
            self.template_type,
            instruction,
            input_text,
            ""  # No output in prompt for DPO
        )

        return PreferenceDataset(
            prompt=prompt,
            chosen=chosen_output,
            rejected=rejected_output,
            metadata={
                'instruction': instruction,
                'input': input_text
            }
        )

    def load_preference_data(self, file_path: str) -> List[PreferenceDataset]:
        """
        Load preference pairs from JSON file.

        File format:
        [
            {
                "instruction": "...",
                "input": "...",
                "chosen": "...",
                "rejected": "..."
            },
            ...
        ]

        Args:
            file_path: Path to JSON file

        Returns:
            List of PreferenceDataset objects
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        preference_pairs = []
        for item in data:
            pair = self.create_preference_pairs(
                instruction=item.get('instruction', ''),
                input_text=item.get('input', ''),
                chosen_output=item['chosen'],
                rejected_output=item['rejected']
            )
            preference_pairs.append(pair)

        return preference_pairs

    def save_preference_data(
        self,
        preferences: List[PreferenceDataset],
        output_path: str
    ) -> None:
        """
        Save preference pairs to JSON file.

        Args:
            preferences: List of PreferenceDataset objects
            output_path: Output file path
        """
        data = []
        for pref in preferences:
            data.append({
                'instruction': pref.metadata.get('instruction', '') if pref.metadata else '',
                'input': pref.metadata.get('input', '') if pref.metadata else '',
                'chosen': pref.chosen,
                'rejected': pref.rejected,
                'prompt': pref.prompt
            })

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def split_dataset(
        self,
        dataset: List[PreferenceDataset],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[PreferenceDataset], List[PreferenceDataset], List[PreferenceDataset]]:
        """
        Split dataset into train/val/test.

        Args:
            dataset: Full dataset
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            seed: Random seed

        Returns:
            Tuple of (train, val, test) datasets
        """
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))

        train_size = int(len(dataset) * train_ratio)
        val_size = int(len(dataset) * val_ratio)

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        train_set = [dataset[i] for i in train_idx]
        val_set = [dataset[i] for i in val_idx]
        test_set = [dataset[i] for i in test_idx]

        return train_set, val_set, test_set

    def create_sample_dataset(self) -> List[PreferenceDataset]:
        """
        Create a sample preference dataset for testing.

        Returns:
            List of sample PreferenceDataset objects
        """
        samples = [
            {
                'instruction': 'Write a haiku about nature',
                'input': '',
                'chosen': 'Morning dew glistens\nBirds sing their ancient sweet songs\nWinter fades to spring',
                'rejected': 'nature is good'
            },
            {
                'instruction': 'Explain photosynthesis',
                'input': '',
                'chosen': 'Photosynthesis is a process where plants convert light energy into chemical energy using chlorophyll. Water and CO2 combine with light to produce glucose and oxygen.',
                'rejected': 'plants do photosynthesis'
            },
            {
                'instruction': 'List three benefits of exercise',
                'input': '',
                'chosen': '1. Improves cardiovascular health\n2. Increases muscle strength and endurance\n3. Enhances mental well-being and mood',
                'rejected': 'exercise is good'
            },
            {
                'instruction': 'Write a Python function to calculate factorial',
                'input': '',
                'chosen': 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)',
                'rejected': 'def factorial(n): return n * factorial(n)'
            },
            {
                'instruction': 'Describe the water cycle',
                'input': '',
                'chosen': 'The water cycle involves evaporation (water becomes vapor), condensation (vapor becomes liquid), precipitation (water falls), and collection in bodies of water.',
                'rejected': 'water goes around'
            },
        ]

        preference_pairs = []
        for item in samples:
            pair = self.create_preference_pairs(
                instruction=item['instruction'],
                input_text=item['input'],
                chosen_output=item['chosen'],
                rejected_output=item['rejected']
            )
            preference_pairs.append(pair)

        return preference_pairs
