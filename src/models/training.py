import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import mlflow
import mlflow.pytorch
from datetime import datetime
import os
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class LLMTrainer:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup_model(self):
        """Initialize model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(self, train_dataset, eval_dataset, hyperparams: Dict[str, Any] = None):
        """Train the model with MLflow tracking."""

        # Default hyperparameters
        default_params = {
            "learning_rate": 5e-5,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "num_train_epochs": 3,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 2,
        }

        if hyperparams:
            default_params.update(hyperparams)

        # Start MLflow run
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(default_params)
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("dataset_size", len(train_dataset))

            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=f"./models/flan-t5-finetuned-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                evaluation_strategy="steps",
                eval_steps=500,
                logging_steps=100,
                save_steps=500,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to="none",  # We'll use MLflow for tracking
                **default_params
            )

            # Setup data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )

            # Initialize trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=[MLflowCallback()]
            )

            # Train the model
            logger.info("Starting model training...")
            train_result = self.trainer.train()

            # Log training metrics
            mlflow.log_metrics({
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics['train_runtime'],
                "train_samples_per_second": train_result.metrics['train_samples_per_second']
            })

            # Evaluate the model
            eval_result = self.trainer.evaluate()
            mlflow.log_metrics({
                "eval_loss": eval_result['eval_loss'],
                "eval_runtime": eval_result['eval_runtime']
            })

            # Save the model to MLflow
            mlflow.pytorch.log_model(
                self.model,
                "model",
                registered_model_name="flan-t5-customer-support"
            )

            # Save tokenizer
            tokenizer_path = "tokenizer"
            self.tokenizer.save_pretrained(tokenizer_path)
            mlflow.log_artifacts(tokenizer_path, "tokenizer")

            logger.info("Training completed successfully!")
            return train_result, eval_result

class MLflowCallback:
    """Custom callback to log metrics to MLflow during training."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            # Filter out non-metric logs
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            mlflow.log_metrics(metrics, step=state.global_step)


# Main execution script
if __name__ == "__main__":
    import sys
    sys.path.append(".")

    from src.data.preprocessing import DataPreprocessor

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize components
    preprocessor = DataPreprocessor()
    trainer = LLMTrainer()

    # Setup model
    trainer.setup_model()

    # Prepare data
    train_dataset, eval_dataset = preprocessor.prepare_datasets()

    # Train model
    train_result, eval_result = trainer.train(train_dataset, eval_dataset)

    print(f"Training completed! Final eval loss: {eval_result['eval_loss']:.4f}")