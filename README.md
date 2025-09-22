# LLM MLOps Platform

A production-ready MLOps platform for Large Language Model fine-tuning, deployment, and monitoring. This project demonstrates end-to-end machine learning operations capabilities including automated data pipelines, model training, API serving, and deployment orchestration.

## Architecture Overview

```
Data Pipeline → Model Training → Model Registry → API Serving → Monitoring
     ↓              ↓              ↓              ↓            ↓
 preprocessing   experiment     versioning    FastAPI      Prometheus
  validation     tracking      artifacts     Docker       Grafana
  tokenization   MLflow        storage       K8s          alerts
```

## Features

- **Data Pipeline**: Automated preprocessing, validation, and tokenization
- **Model Training**: Experiment tracking with MLflow, hyperparameter optimization
- **Model Registry**: Versioned model artifacts with metadata storage
- **API Serving**: Production-ready FastAPI service with health checks
- **Containerization**: Docker containers with security best practices
- **Orchestration**: Kubernetes manifests for scalable deployment
- **CI/CD**: Automated testing and deployment pipeline
- **Monitoring**: Comprehensive logging and metrics collection

## Technology Stack

- **ML Framework**: PyTorch, Transformers (Hugging Face)
- **API Framework**: FastAPI, Uvicorn
- **Experiment Tracking**: MLflow
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **Testing**: Pytest
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana

## Project Structure

```
llm-mlops-platform/
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # Data pipeline and tokenization
│   │   └── __init__.py
│   ├── models/
│   │   ├── training.py           # Model training with MLflow tracking
│   │   └── __init__.py
│   ├── api/
│   │   ├── main.py              # FastAPI service
│   │   └── __init__.py
│   └── monitoring/
│       └── metrics.py           # Custom metrics and monitoring
├── tests/
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── load/                    # Performance tests
├── deployment/
│   ├── kubernetes/              # K8s manifests
│   ├── helm/                    # Helm charts
│   └── terraform/               # Infrastructure as Code
├── .github/
│   └── workflows/               # CI/CD pipelines
├── docs/
│   └── architecture.md         # System architecture
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, for containerization)
- Git

### 1. Clone Repository

```bash
git clone https://github.com/Hetal2311/llm-mlops-platform.git
cd llm-mlops-platform
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Training Pipeline

```bash
# Train the model with experiment tracking
python src/models/training.py
```

### 4. Start API Service

```bash
# Start the FastAPI server
python src/api/main.py
```

The API will be available at `http://localhost:8000`

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "My order is delayed, what should I do?"}'
```

## API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/predict` | POST | Single text inference |
| `/predict/batch` | POST | Batch text inference |
| `/metrics/custom` | GET | Custom metrics endpoint |

## Development

### Running Tests

```bash
# Run unit tests
python -m pytest tests/unit/ -v

# Run integration tests
python -m pytest tests/integration/ -v

# Run all tests with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Docker Development

```bash
# Build Docker image
docker build -t llm-mlops-platform:latest .

# Run container
docker run -p 8000:8000 llm-mlops-platform:latest

# Run with volume mounting for development
docker run -p 8000:8000 -v $(pwd)/src:/app/src llm-mlops-platform:latest
```

## Deployment

### Local Deployment

```bash
# Using Docker Compose (if available)
docker-compose up

# Direct Python execution
python src/api/main.py
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get pods -l app=llm-mlops-api

# View logs
kubectl logs -l app=llm-mlops-api
```

### Scaling

```bash
# Scale deployment
kubectl scale deployment llm-mlops-api --replicas=5

# Enable horizontal pod autoscaling
kubectl apply -f deployment/kubernetes/hpa.yaml
```

## Monitoring and Observability

### Metrics

The platform exposes metrics at `/metrics` for Prometheus scraping:
- Request latency and throughput
- Model inference time
- Error rates and types
- Resource utilization

### Logging

Structured logging is implemented throughout:
- Request/response logging
- Model training progress
- Error tracking and debugging
- Performance monitoring

### Health Checks

- **Liveness probe**: `/health` endpoint
- **Readiness probe**: Model loading verification
- **Startup probe**: Container initialization

## MLOps Capabilities

### Experiment Tracking

- MLflow integration for experiment management
- Hyperparameter logging and comparison
- Model performance metrics tracking
- Artifact versioning and storage

### Model Registry

- Automated model versioning
- Model metadata and lineage tracking
- Stage-based model promotion (staging → production)
- Model performance monitoring

### Data Pipeline

- Automated data validation and quality checks
- Reproducible preprocessing workflows
- Data versioning and lineage tracking
- Schema validation and drift detection

## CI/CD Pipeline

GitHub Actions workflow includes:
- Automated testing on pull requests
- Docker image building and pushing
- Security scanning and code quality checks
- Automated deployment to staging environments

## Security Considerations

- Non-root container execution
- Dependency vulnerability scanning
- Input validation and sanitization
- Rate limiting and authentication ready
- Secrets management best practices

## Performance Optimization

- Async API endpoints for better throughput
- Connection pooling and resource management
- Model optimization and quantization ready
- Horizontal scaling with load balancing
- Caching strategies for repeated requests

## Configuration

Key configuration parameters:

```python
# Model Configuration
MODEL_NAME = "simple-local-model"
MAX_LENGTH = 512
BATCH_SIZE = 32

# API Configuration
HOST = "0.0.0.0"
PORT = 8000
WORKERS = 4

# Training Configuration
LEARNING_RATE = 0.001
EPOCHS = 10
VALIDATION_SPLIT = 0.2
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Write tests for new features
- Follow PEP 8 style guidelines
- Update documentation for changes
- Ensure Docker builds successfully
- Run the full test suite before submitting

## Troubleshooting

### Common Issues

**SSL Certificate Errors (Corporate Networks)**
```bash
export PYTHONHTTPSVERIFY=0
export CURL_CA_BUNDLE=""
```

**Docker Permission Issues**
```bash
sudo docker build -t llm-mlops-platform:latest .
```

**Port Already in Use**
```bash
# Kill process using port 8000
lsof -ti:8000 | xargs kill -9
```

**Memory Issues During Training**
```bash
# Reduce batch size in training configuration
BATCH_SIZE = 16  # or smaller
```

### Getting Help

- Check the [Issues](https://github.com/Hetal2311/llm-mlops-platform/issues) page
- Review logs: `docker logs <container-name>`
- Enable debug logging: `export LOG_LEVEL=DEBUG`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Hetal Patel**
- Email: hetal2311@gmail.com
- LinkedIn: [linkedin.com/in/hetal2311](https://linkedin.com/in/hetal2311)
- GitHub: [github.com/Hetal2311](https://github.com/Hetal2311)

## Acknowledgments

- Built with modern MLOps best practices
- Inspired by production ML systems at scale
- Designed for enterprise deployment patterns
- Optimized for team collaboration and maintainability

---

**Note**: This platform demonstrates MLOps engineering capabilities with a simplified ML model due to network restrictions. The infrastructure patterns and deployment practices are production-ready and can be easily adapted for real transformer models and datasets.