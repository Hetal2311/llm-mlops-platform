from datasets import Dataset
import pandas as pd

def create_simple_dataset():
    """Create a simple customer support dataset for testing."""
    data = [
        {"instruction": "My order is delayed, what should I do?",
         "response": "I apologize for the delay. Let me check your order status and provide an update."},
        {"instruction": "How do I return an item?",
         "response": "You can return items within 30 days. Please visit our returns page for instructions."},
        {"instruction": "I forgot my password",
         "response": "No problem! Click 'Forgot Password' on the login page to reset it."},
        {"instruction": "What are your business hours?",
         "response": "We're open Monday-Friday 9AM-5PM EST. Online support is available 24/7."},
        {"instruction": "My payment was declined",
         "response": "Payment declines can happen for various reasons. Please check with your bank or try a different payment method."},
    ]

    df = pd.DataFrame(data)
    df['input_text'] = df['instruction'].apply(
        lambda x: f"Customer query: {x}\nProvide a helpful response:"
    )
    df['target_text'] = df['response']

    return Dataset.from_pandas(df)