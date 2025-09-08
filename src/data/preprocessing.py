import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 512

    def load_customer_support_data(self) -> Dataset:
        """Load and preprocess customer support dataset."""
        try:
            # Using a public customer support dataset
            dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

            # Convert to pandas for easier manipulation
            df = pd.DataFrame(dataset['train'])

            # Create instruction-response pairs for fine-tuning
            df['input_text'] = df['instruction'].apply(
                lambda x: f"Customer query: {x}\nProvide a helpful response:"
            )
            df['target_text'] = df['response']

            # Filter and clean data
            df = self._clean_data(df)

            # Convert back to Hugging Face dataset
            processed_dataset = Dataset.from_pandas(df)

            logger.info(f"Loaded {len(processed_dataset)} examples")
            return processed_dataset

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter the dataset."""
        # Remove duplicates
        df = df.drop_duplicates(subset=['instruction', 'response'])

        # Filter by length
        df = df[
            (df['instruction'].str.len() > 10) &
            (df['instruction'].str.len() < 500) &
            (df['response'].str.len() > 10) &
            (df['response'].str.len() < 500)
        ]

        # Remove null values
        df = df.dropna(subset=['instruction', 'response'])

        return df.reset_index(drop=True)

    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize the input and target texts."""
        # Tokenize inputs
        model_inputs = self.tokenizer(
            examples['input_text'],
            max_length=self.max_length,
            truncation=True,
            padding=True
        )

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['target_text'],
                max_length=self.max_length,
                truncation=True,
                padding=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets."""
        dataset = self.load_customer_support_data()

        # Split the dataset
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']

        # Tokenize datasets
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        eval_dataset = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )

        return train_dataset, eval_dataset