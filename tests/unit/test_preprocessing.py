# Unit tests
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.preprocessing import DataPreprocessor
from datasets import Dataset
import pandas as pd

class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.preprocessor = DataPreprocessor()

    def test_init(self):
        """Test DataPreprocessor initialization."""
        assert self.preprocessor.max_length == 512
        assert hasattr(self.preprocessor, 'simple_tokenize')

    def test_simple_tokenize_basic(self):
        """Test basic tokenization functionality."""
        text = "Hello world"
        tokens = self.preprocessor.simple_tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) == 2  # "hello" and "world"
        assert all(isinstance(token, int) for token in tokens)

    def test_simple_tokenize_empty_string(self):
        """Test tokenization with empty string."""
        tokens = self.preprocessor.simple_tokenize("")
        assert tokens == []

    def test_simple_tokenize_long_text(self):
        """Test tokenization with text longer than max_length."""
        long_text = " ".join([f"word{i}" for i in range(600)])
        tokens = self.preprocessor.simple_tokenize(long_text)

        assert len(tokens) <= self.preprocessor.max_length
        assert len(tokens) == 512  # Should be truncated to max_length

    def test_simple_tokenize_case_insensitive(self):
        """Test that tokenization is case insensitive."""
        tokens1 = self.preprocessor.simple_tokenize("Hello World")
        tokens2 = self.preprocessor.simple_tokenize("hello world")

        assert tokens1 == tokens2

    def test_load_customer_support_data(self):
        """Test customer support data loading."""
        dataset = self.preprocessor.load_customer_support_data()

        assert isinstance(dataset, Dataset)
        assert len(dataset) == 5  # We have 5 predefined examples

        # Check required columns
        assert 'instruction' in dataset.column_names
        assert 'response' in dataset.column_names
        assert 'input_text' in dataset.column_names
        assert 'target_text' in dataset.column_names

    def test_load_customer_support_data_content(self):
        """Test the content of loaded customer support data."""
        dataset = self.preprocessor.load_customer_support_data()

        # Check first example
        first_example = dataset[0]
        assert 'order' in first_example['instruction'].lower()
        assert 'Customer query:' in first_example['input_text']
        assert 'Provide a helpful response:' in first_example['input_text']

    def test_tokenize_function(self):
        """Test the tokenize function with sample data."""
        sample_data = {
            'input_text': ["Customer query: Test\nProvide a helpful response:"],
            'target_text': ["This is a test response."]
        }

        result = self.preprocessor.tokenize_function(sample_data)

        assert 'input_ids' in result
        assert 'labels' in result
        assert len(result['input_ids']) == 1
        assert len(result['labels']) == 1
        assert isinstance(result['input_ids'][0], list)
        assert isinstance(result['labels'][0], list)

    def test_tokenize_function_batch(self):
        """Test tokenize function with multiple examples."""
        sample_data = {
            'input_text': [
                "Customer query: Test 1\nProvide a helpful response:",
                "Customer query: Test 2\nProvide a helpful response:"
            ],
            'target_text': [
                "Response 1",
                "Response 2"
            ]
        }

        result = self.preprocessor.tokenize_function(sample_data)

        assert len(result['input_ids']) == 2
        assert len(result['labels']) == 2

    def test_prepare_datasets(self):
        """Test dataset preparation and splitting."""
        train_dataset, eval_dataset = self.preprocessor.prepare_datasets()

        # Check dataset types
        assert isinstance(train_dataset, Dataset)
        assert isinstance(eval_dataset, Dataset)

        # Check dataset sizes (80/20 split of 5 examples = 4/1)
        assert len(train_dataset) == 4
        assert len(eval_dataset) == 1

        # Check that datasets contain tokenized data
        assert 'input_ids' in train_dataset.column_names
        assert 'labels' in train_dataset.column_names

        # Original columns should be removed
        assert 'instruction' not in train_dataset.column_names
        assert 'response' not in train_dataset.column_names

    def test_prepare_datasets_reproducibility(self):
        """Test that dataset preparation is reproducible."""
        train1, eval1 = self.preprocessor.prepare_datasets()
        train2, eval2 = self.preprocessor.prepare_datasets()

        # Should be identical due to fixed seed
        assert len(train1) == len(train2)
        assert len(eval1) == len(eval2)

        # Compare first example
        assert train1[0]['input_ids'] == train2[0]['input_ids']
        assert train1[0]['labels'] == train2[0]['labels']

    def test_tokenize_function_empty_input(self):
        """Test tokenize function with empty inputs."""
        sample_data = {
            'input_text': [""],
            'target_text': [""]
        }

        result = self.preprocessor.tokenize_function(sample_data)

        assert result['input_ids'] == [[]]
        assert result['labels'] == [[]]

# Integration test
class TestDataPreprocessorIntegration:
    """Integration tests for the complete preprocessing pipeline."""

    def test_end_to_end_preprocessing(self):
        """Test the complete preprocessing pipeline."""
        preprocessor = DataPreprocessor()

        # Load and prepare datasets
        train_dataset, eval_dataset = preprocessor.prepare_datasets()

        # Verify the pipeline produces valid training data
        assert len(train_dataset) > 0
        assert len(eval_dataset) > 0

        # Check that we can access tokenized examples
        train_example = train_dataset[0]
        assert 'input_ids' in train_example
        assert 'labels' in train_example
        assert len(train_example['input_ids']) > 0
        assert len(train_example['labels']) > 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])