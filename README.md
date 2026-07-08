# Telco Customer Churn Prediction: End-to-End MLOps Pipeline

Developed in collaboration with Andrés Malón as the final project for the University Master's Degree in Machine Learning at the Public University of Navarra (UPNA).

## How to Run the Project

This project uses `uv` for fast dependency management and lockfile resolution (`uv.lock`). 

### Prerequisites
* **Python:** 3.10
* **Package Manager:** [uv](https://github.com/astral-sh/uv) (`pip install uv`)
* **Containerization:** Docker & Docker Compose (for the monitoring stack)

### 1. Environment Setup
Clone the repository and install the dependencies. Using `uv sync` ensures you are running the exact same environment used in production.

```bash
# Clone the repository
git clone https://github.com/raldanondo/customer_churn_mlops.git
cd customer_churn_mlops

# Install dependencies and create the virtual environment
uv sync --frozen
```

### 2. Run the Training Pipeline

To retrain the XGBoost and TabNet (Shadow Deployment strategy) models, execute the training master script. This will process the raw data, tune hyperparameters, log metrics to MLflow, and save the serialized `.joblib` artifacts to `api/models_local/`.

```bash
uv run python -m mylib.train_pipeline
```

---

### 3. Run Quality Assurance Tests

The project includes a set of tests tha acts as the CI/CD quality gate. They validate the generated artifacts, preprocessing pipeline, and API functionality.

```bash
uv run pytest tests/
```

---

### 4. Run the Inference API (Backend)

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API documentation (Swagger UI) is available at:

```text
http://localhost:8000/docs
```

---

### 5. Run the User Interface (Frontend)

```bash
uv run streamlit run gui/app.py
```

The UI is available at:

```text
http://localhost:8501
```

---

### 6. Run the Monitoring Stack

To observe API metrics and model drift, start the monitoring stack:

```bash
cd monitoring
docker compose up -d
```

Then generate traffic:

```bash
uv run python stream_test_data.py
```

Grafana: `http://localhost:3000`
Prometheus: `http://localhost:9090`

> **Note:** The monitoring stack relies on Docker Compose. It was not validated on the development machine because my installed Windows version does not meet the minimum requirements for the current Docker Desktop release. The configuration files (`docker-compose.yml`, Prometheus configuration, and Grafana dashboards) are included in the repository, but local execution requires a supported Docker installation.