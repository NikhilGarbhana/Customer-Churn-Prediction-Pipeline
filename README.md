# Customer-Churn-Prediction-Pipeline
A scalable data engineering **Pipeline to Predict Customer Churn** for a subscription-based business. Provide actionable insights and real-time dashboards for stakeholders

---

### Steps to Implement
### 1. Data Collection (Simulated Data Source)
We’ll generate synthetic customer and transaction data using the Faker library.

```python
from faker import Faker
import pandas as pd
import random

# Initialize Faker
fake = Faker()

# Generate synthetic customer data
def generate_customer_data(num_records=1000):
    data = []
    for _ in range(num_records):
        data.append({
            "customer_id": fake.uuid4(),
            "name": fake.name(),
            "age": random.randint(18, 70),
            "signup_date": fake.date_between(start_date='-2y', end_date='today'),
            "subscription_type": random.choice(["Basic", "Standard", "Premium"]),
            "last_active_date": fake.date_between(start_date='-1y', end_date='today'),
            "is_churned": random.choice([0, 1])
        })
    return pd.DataFrame(data)

# Save to CSV
customer_data = generate_customer_data()
customer_data.to_csv("customers.csv", index=False)
```
---
### 2. Data Ingestion
Ingest the data into a cloud storage bucket (e.g., AWS S3) or local database.

```python
# AWS Configuration
AWS_ACCESS_KEY = "<your_access_key>"
AWS_SECRET_KEY = "<your_secret_key>"
RAW_BUCKET = "ng-test-csv"
PROCESSED_BUCKET = "ng-test-csv-processed"

CSV_FILE_PATH = "customers.csv"  # Local CSV file path
S3_OBJECT_NAME = "customers.csv"  # Desired S3 path
FILE_KEY = "transformed/customers.csv"  # Path to the cleaned data in S3

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name='eu-north-1'
)

# Function to Upload File
def upload_to_s3(file_path, bucket_name, object_name):
    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"File uploaded successfully to s3://{bucket_name}/{object_name}")
    except FileNotFoundError:
        print("File not found. Check the file path.")
    except NoCredentialsError:
        print("AWS credentials not available.")

# Upload the file
upload_to_s3(CSV_FILE_PATH, RAW_BUCKET, S3_OBJECT_NAME)
```
Alternatively, use Python to load the data into a SQL database:
```python
from sqlalchemy import create_engine

# Load data into SQLite (for demo purposes)
engine = create_engine("sqlite:///customer_data.db")
customer_data.to_sql("customers", engine, if_exists="replace", index=False)
```
---
### 3. ETL Pipeline
Use Apache Airflow to create a pipeline that:

1. Extracts data from the database.
2. Transforms it into clean, analytics-ready data.
3. Loads it into a data warehouse like Snowflake or BigQuery.
   
**Airflow DAG:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

# Define default args
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 11, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define ETL functions
def extract():
    customer_data = pd.read_sql("SELECT * FROM customers", con=engine)
    customer_data.to_csv("/tmp/extracted_data.csv", index=False)

def transform():
    df = pd.read_csv("/tmp/extracted_data.csv")
    df["days_since_last_active"] = (datetime.now() - pd.to_datetime(df["last_active_date"])).dt.days
    df.to_csv("/tmp/transformed_data.csv", index=False)

def load():
    df = pd.read_csv("/tmp/transformed_data.csv")
    df.to_sql("cleaned_customers", engine, if_exists="replace", index=False)

# Define DAG
with DAG(
    "customer_churn_pipeline",
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    task_extract = PythonOperator(task_id="extract", python_callable=extract)
    task_transform = PythonOperator(task_id="transform", python_callable=transform)
    task_load = PythonOperator(task_id="load", python_callable=load)

    task_extract >> task_transform >> task_load
```
### 4. Feature Engineering and Modeling
Prepare features and train a machine learning model for churn prediction.
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load cleaned data
df = pd.read_sql("SELECT * FROM cleaned_customers", con=engine)

# Feature selection
X = df[["age", "days_since_last_active"]]
y = df["is_churned"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```
Save the trained model:
```python
import joblib
joblib.dump(clf, "churn_model.pkl")
```
---
### 5. Model Deployment
Deploy the churn prediction model using FastAPI.
```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model
model = joblib.load("churn_model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"churn_prediction": int(prediction[0])}
```
Run the API server:
```bash
uvicorn main:app --reload
```
Testing FastAPI:

![image](https://github.com/user-attachments/assets/0701ec7e-4a5d-4d7c-8f10-d9c5f565e669)

---

### 6. Visualization
Build a real-time dashboard using Plotly Dash to display churn insights.
```python
import dash
from dash import dcc, html
import pandas as pd

app = dash.Dash(__name__)

# Load data
df = pd.read_sql("SELECT * FROM cleaned_customers", con=engine)

app.layout = html.Div([
    html.H1("Customer Churn Insights"),
    dcc.Graph(
        figure={
            "data": [
                {"x": df["subscription_type"], "y": df["is_churned"], "type": "bar", "name": "Churned"}
            ],
            "layout": {"title": "Churn by Subscription Type"}
        }
    )
])

if __name__ == "__main__":
    app.run_server(debug=True)
```

![image](https://github.com/user-attachments/assets/957ada01-6899-4667-b01a-7e1d9c37e0da)

