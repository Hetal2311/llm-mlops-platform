from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import mlflow.pytorch
import uvicorn
import time
import logging
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class InferenceRequest(BaseModel):
    text: str
    max_length: Optional[int] = 150
    temperature: Optional[float] = 0.7

class InferenceResponse(BaseModel):
    generated_text: str
    processing_time: float
    model_version: str

class BatchInferenceRequest(BaseModel):
    texts: List[str]
    max_length: Optional[int] = 150
    temperature: Optional[float] = 0.7

class BatchInferenceResponse(BaseModel):
    results: List[InferenceResponse]
    total_processing_time: float

# Initialize FastAPI app
app = FastAPI(
    title="LLM MLOps Platform API",
    description="Production-ready API for LLM inference with monitoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_version = "1.0.0"

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_path: str = None):
        """Load model from MLflow or local path."""
        try:
            if model_path:
                # Load from local path
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                # For now, load base model (later integrate with MLflow)
                model_name = "google/flan-t5-small"
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)

                # Uncomment below when MLflow model is available
                # client = mlflow.tracking.MlflowClient()
                # latest_version = client.get_latest_versions(
                #     "flan-t5-customer-support",
                #     stages=["Production", "Staging"]
                # )[0]
                # model_uri = f"models:/flan-t5-customer-support/{latest_version.version}"
                # self.model = mlflow.pytorch.load_model(model_uri)

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def generate_response(self, text: str, max_length: int = 150, temperature: float = 0.7) -> str:
        """Generate response for input text."""
        if not self.model or not self.tokenizer:
            raise HTTPException(status_code=500, detail="Model not loaded")

        try:
            # Prepare input
            input_text = f"Customer query: {text}\nProvide a helpful response:"
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail="Error generating response")

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Loading model...")
    model_manager.load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "device": str(model_manager.device),
        "version": model_version
    }

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Single text inference endpoint."""
    start_time = time.time()

    try:
        generated_text = model_manager.generate_response(
            request.text,
            request.max_length,
            request.temperature
        )

        processing_time = time.time() - start_time

        return InferenceResponse(
            generated_text=generated_text,
            processing_time=processing_time,
            model_version=model_version
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchInferenceResponse)
async def predict_batch(request: BatchInferenceRequest):
    """Batch inference endpoint."""
    start_time = time.time()

    try:
        results = []
        for text in request.texts:
            text_start_time = time.time()
            generated_text = model_manager.generate_response(
                text,
                request.max_length,
                request.temperature
            )
            text_processing_time = time.time() - text_start_time

            results.append(InferenceResponse(
                generated_text=generated_text,
                processing_time=text_processing_time,
                model_version=model_version
            ))

        total_processing_time = time.time() - start_time

        return BatchInferenceResponse(
            results=results,
            total_processing_time=total_processing_time
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/custom")
async def custom_metrics():
    """Custom metrics endpoint."""
    return {
        "model_version": model_version,
        "device": str(model_manager.device),
        "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)