

---

# **Voyage Analytics: Travel Price Prediction with MLOps**

## **Project Overview**

This project demonstrates a **full MLOps pipeline** for predicting flight prices using historical travel data. It integrates **data preprocessing, machine learning modeling, REST API deployment, workflow orchestration, and CI/CD automation**.

Key features:

* **Regression modeling** with Decision Tree, Random Forest, and XGBoost
* **Preprocessing and feature engineering**, including target encoding and ordinal mapping
* **Model tracking** with MLflow for experiments and versioning
* **Automated workflows** using Apache Airflow DAGs
* **Deployment** via Docker and Kubernetes for scalability
* **CI/CD** pipeline with Jenkins for automated build, test, and deployment

---

## **Project Structure**

```
.
├── app/                       # Flask API code
│   └── api_price_predictor.py
├── airflow/                    # Apache Airflow DAGs
│   └── dags/travel_price_regression_dag.py
├── data/                       # Sample datasets
│   ├── flights.csv
│   ├── hotels.csv
│   └── users.csv
├── k8s/                        # Kubernetes deployment files
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
├── models/                     # Pretrained model & encoders
│   ├── xgb_regressor.pkl
│   ├── from_encoder.pkl
│   ├── to_encoder.pkl
│   ├── agency_encoder.pkl
│   ├── flight_type_map.pkl
│   └── feature_columns.pkl
├── notebooks/                  # Exploratory analysis & model development
│   └── Price_predictor.ipynb
├── src/                        # Core modules
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   └── train.py
├── Dockerfile                  # Docker configuration
├── Jenkinsfile                 # CI/CD pipeline configuration
├── requirements.txt            # Python dependencies
├── .gitignore
├── .dockerignore
├── README.md
└── LICENSE
```

---

## **Getting Started**

### **1. Clone the repository**

```bash
git clone https://github.com/bsshewale/Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-.git
cd Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the training script**

```bash
python src/train.py
```

* Trains the XGBoost regression model
* Logs parameters, metrics, and artifacts to MLflow

---

## **MLflow Model Tracking**

* Start MLflow UI:

```bash
mlflow ui
```

* Access at [http://localhost:5000](http://localhost:5000)
* Track experiments, compare runs, and register model versions

---

## **Flask API for Real-Time Predictions**

* Start the API:

```bash
python app/api_price_predictor.py
```
![Price_predictor](https://github.com/user-attachments/assets/d443aae2-3288-4685-9c7c-1a791450dc4b)

* Example request:

```http
POST /predict
Content-Type: application/json

{
    "from": "Recife (PE)",
    "to": "Florianopolis (SC)",
    "flightType": "firstClass",
    "time": 1.76,
    "distance": 676.53,
    "agency": "FlyingDrops",
    "day": 26,
    "month": 9,
    "year": 2019
}
```

* Returns the **predicted flight price** in real time

---

## **Workflow Automation with Airflow**

* DAG: `travel_price_regression_dag.py` orchestrates:

  * Data ingestion
  * Preprocessing
  * Model training
  * MLflow logging

* Start Airflow:

```bash
airflow db init
airflow webserver --port 8080
airflow scheduler
```

* Access DAGs at [http://localhost:8080](http://localhost:8080)

---

## **Docker & Kubernetes Deployment**

* Build Docker image:

```bash
docker build -t bsshewale/flight-price-predictor:latest .
```

* Push to Docker Hub:

```bash
docker push bsshewale/flight-price-predictor:latest
```

* Deploy to Kubernetes:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

* Scalable deployment with **Horizontal Pod Autoscaling (HPA)**

---

## **CI/CD with Jenkins**

* Jenkins pipeline defined in `Jenkinsfile`:

  1. Checkout repository
  2. Install dependencies
  3. Run basic tests
  4. Build and push Docker image
  5. Deploy to Kubernetes cluster

* Credentials (`dockerhub-creds`, `kubeconfig`) securely stored in Jenkins

---

## **Project Highlights**

* **Full MLOps pipeline** from data ingestion → modeling → deployment
* **Automated workflows** via Airflow
* **Model versioning & tracking** via MLflow
* **CI/CD automation** with Jenkins
* **Scalable deployment** using Docker + Kubernetes

---

## **Requirements**

* Python ≥ 3.10
* Docker
* Kubernetes cluster (minikube or cloud)
* Jenkins (optional for CI/CD)
* Apache Airflow
* MLflow

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---


