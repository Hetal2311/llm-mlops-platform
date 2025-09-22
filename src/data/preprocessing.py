import pandas as pd
from datasets import Dataset
from typing import Dict, List, Tuple
import logging
import os
import ssl
import urllib3
import certifi

# Configure SSL for corporate environments
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Disable SSL warnings and verification
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Monkey patch requests to skip SSL verification
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

class NoSSLVerifyHTTPAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

session = requests.Session()
session.mount('https://', NoSSLVerifyHTTPAdapter())

# Patch the requests module
requests.Session.request = lambda self, *args, **kwargs: session.request(*args, **kwargs)

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        # Simple tokenizer approach - no external downloads
        self.max_length = 512

    def simple_tokenize(self, text: str) -> List[int]:
        """Simple tokenization for demonstration purposes."""
        # Convert text to simple token IDs based on character/word mapping
        tokens = text.lower().split()
        # Simple hash-based token mapping
        token_ids = [hash(token) % 10000 for token in tokens]
        return token_ids[:self.max_length]

    def load_customer_support_data(self) -> Dataset:
        """Load and preprocess customer support dataset."""
        try:
            # Simple local dataset
            data = [
                {"instruction": "My order is delayed, what should I do?",
                 "response": "I apologize for the delay. Let me check your order status."},
                {"instruction": "How do I return an item?",
                 "response": "You can return items within 30 days."},
                {"instruction": "I forgot my password",
                 "response": "Click 'Forgot Password' on the login page."},
                {"instruction": "What are your business hours?",
                 "response": "We're open Monday-Friday 9AM-5PM EST."},
                {"instruction": "My payment was declined",
                 "response": "Please check with your bank or try a different payment method."},
            ]

            df = pd.DataFrame(data)
            df['input_text'] = df['instruction'].apply(
                lambda x: f"Customer query: {x}\nProvide a helpful response:"
            )
            df['target_text'] = df['response']

            processed_dataset = Dataset.from_pandas(df)
            logger.info(f"Loaded {len(processed_dataset)} examples")
            return processed_dataset

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def tokenize_function(self, examples: Dict) -> Dict:
        """Simple tokenization function."""
        input_ids = [self.simple_tokenize(text) for text in examples['input_text']]
        labels = [self.simple_tokenize(text) for text in examples['target_text']]

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets."""
        dataset = self.load_customer_support_data()

        # Split the dataset
        train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']

        # Apply tokenization
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['instruction', 'response', 'input_text', 'target_text']
        )

        eval_dataset = eval_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=['instruction', 'response', 'input_text', 'target_text']
        )

        return train_dataset, eval_dataset