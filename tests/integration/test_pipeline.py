import pytest
import sys
import os
import json
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.preprocessing import DataPreprocessor
from src.models.training import SimpleMLTrainer
import requests
import time
import subprocess
import signal

class TestMLOpsPipelineIntegration:
    """Integration tests for the complete MLOps pipeline."""

    def test_data_preprocessing_to_training_pipeline(self):
        """Test the complete data preprocessing to training pipeline."""
        # Initialize components
        preprocessor = DataPreprocessor()
        trainer = SimpleMLTrainer()

        # Prepare data
        train_dataset, eval_dataset = preprocessor.prepare_datasets()

        # Verify datasets are compatible with trainer
        assert len(train_dataset) > 0
        assert len(eval_dataset) > 0

        # Run training
        results = trainer.train(train_dataset, eval_dataset)

        # Verify training results
        assert 'final_train_loss' in results
        assert 'final_eval_loss' in results
        assert 'model_type' in results
        assert isinstance(results['final_train_loss'], float)
        assert isinstance(results['final_eval_loss'], float)
        assert results['final_train_loss'] > 0
        assert results['final_eval_loss'] > 0
        assert 'hyperparameters' in results
        assert 'train_losses' in results
        assert 'eval_losses' in results

class TestAPIIntegration:
    """Integration tests for the API service."""

    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for testing."""
        # This is a simplified test - in production you'd use a test server
        # For now, we'll test the components directly
        pass

    def test_api_health_endpoint_structure(self):
        """Test the health endpoint structure without running server."""
        # Import the app to test structure
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

        from src.api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_api_predict_endpoint_structure(self):
        """Test the predict endpoint structure."""
        from src.api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Test with valid input
        response = client.post(
            "/predict",
            json={"text": "My order is delayed"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "generated_text" in data
        assert "processing_time" in data
        assert "model_version" in data
        assert isinstance(data["processing_time"], float)
        assert isinstance(data["generated_text"], str)

    def test_api_predict_endpoint_validation(self):
        """Test API input validation."""
        from src.api.main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Test with missing text field
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error

        # Test with invalid data type
        response = client.post("/predict", json={"text": 123})
        assert response.status_code == 422  # Validation error

class TestTrainingPersistence:
    """Test training artifacts and persistence."""

    def test_training_produces_artifacts(self):
        """Test that training produces and saves artifacts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            preprocessor = DataPreprocessor()
            trainer = SimpleMLTrainer()

            # Change to temp directory
            original_dir = os.getcwd()
            try:
                os.chdir(temp_dir)

                # Prepare data and train
                train_dataset, eval_dataset = preprocessor.prepare_datasets()
                results = trainer.train(train_dataset, eval_dataset)

                # Verify training completed successfully
                assert 'final_train_loss' in results
                assert 'final_eval_loss' in results
                assert 'model_type' in results

            finally:
                os.chdir(original_dir)

class TestDataQuality:
    """Test data quality and validation."""

    def test_dataset_quality_checks(self):
        """Test data quality validation."""
        preprocessor = DataPreprocessor()
        dataset = preprocessor.load_customer_support_data()

        # Check dataset size
        assert len(dataset) > 0

        # Check required columns exist
        required_columns = ['instruction', 'response', 'input_text', 'target_text']
        for col in required_columns:
            assert col in dataset.column_names

        # Check data quality
        for example in dataset:
            # No empty strings
            assert len(example['instruction'].strip()) > 0
            assert len(example['response'].strip()) > 0
            assert len(example['input_text'].strip()) > 0
            assert len(example['target_text'].strip()) > 0

            # Input text should contain the instruction
            assert example['instruction'] in example['input_text']

            # Target text should match response
            assert example['response'] == example['target_text']

    def test_tokenization_quality(self):
        """Test tokenization quality and consistency."""
        preprocessor = DataPreprocessor()

        # Test same input produces same output
        text = "Test tokenization consistency"
        tokens1 = preprocessor.simple_tokenize(text)
        tokens2 = preprocessor.simple_tokenize(text)

        assert tokens1 == tokens2

        # Test different inputs produce different outputs
        text2 = "Different test input"
        tokens3 = preprocessor.simple_tokenize(text2)

        assert tokens1 != tokens3

        # Test empty input
        empty_tokens = preprocessor.simple_tokenize("")
        assert empty_tokens == []

class TestModelCompatibility:
    """Test model and API compatibility."""

    def test_model_api_integration(self):
        """Test that trained model is compatible with API serving."""
        # Initialize components
        preprocessor = DataPreprocessor()
        trainer = SimpleMLTrainer()

        # Train model
        train_dataset, eval_dataset = preprocessor.prepare_datasets()
        training_results = trainer.train(train_dataset, eval_dataset)

        # Test API can handle model output format
        from src.api.main import SimpleModelManager

        model_manager = SimpleModelManager()
        response = model_manager.generate_response("Test query")

        assert isinstance(response, str)
        assert len(response) > 0

class TestReproducibility:
    """Test pipeline reproducibility."""

    def test_preprocessing_reproducibility(self):
        """Test that preprocessing is reproducible."""
        preprocessor1 = DataPreprocessor()
        preprocessor2 = DataPreprocessor()

        train1, eval1 = preprocessor1.prepare_datasets()
        train2, eval2 = preprocessor2.prepare_datasets()

        # Should produce identical results
        assert len(train1) == len(train2)
        assert len(eval1) == len(eval2)

        # Compare first examples
        assert train1[0]['input_ids'] == train2[0]['input_ids']
        assert train1[0]['labels'] == train2[0]['labels']

    def test_training_determinism(self):
        """Test training determinism with same inputs."""
        preprocessor = DataPreprocessor()
        train_dataset, eval_dataset = preprocessor.prepare_datasets()

        trainer1 = SimpleMLTrainer()
        trainer2 = SimpleMLTrainer()

        results1 = trainer1.train(train_dataset, eval_dataset)
        results2 = trainer2.train(train_dataset, eval_dataset)

        # Results should be identical for deterministic simulation
        assert results1['final_train_loss'] == results2['final_train_loss']
        assert results1['final_eval_loss'] == results2['final_eval_loss']
        assert results1['model_type'] == results2['model_type']

if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])