
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import logging
import sys
sys.path.append(".")

from src.data.preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class InferenceRequest(BaseModel):
    text: str

class InferenceResponse(BaseModel):
    generated_text: str
    processing_time: float
    model_version: str

# Initialize FastAPI app
app = FastAPI(
    title="LLM MLOps Platform API",
    description="Production-ready API for LLM inference",
    version="1.0.0"
)

# Simple response generator (simulates your trained model)
class SimpleModelManager:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model_version = "1.0.0"

    def generate_response(self, text: str) -> str:
        """Simple response generation based on preprocessing."""
        # Simulate trained model response
        responses = {
            "order": "I understand your order concern. Let me help you track it.",
            "return": "You can return items within 30 days with original receipt.",
            "password": "Click 'Forgot Password' and check your email for reset link.",
            "payment": "Please verify your payment method or contact your bank.",
            "default": "Thank you for contacting support. How can I assist you today?"
        }

        text_lower = text.lower()
        for key, response in responses.items():
            if key in text_lower:
                return response
        return responses["default"]

model_manager = SimpleModelManager()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": model_manager.model_version
    }

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    start_time = time.time()

    try:
        generated_text = model_manager.generate_response(request.text)
        processing_time = time.time() - start_time

        return InferenceResponse(
            generated_text=generated_text,
            processing_time=processing_time,
            model_version=model_manager.model_version
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)