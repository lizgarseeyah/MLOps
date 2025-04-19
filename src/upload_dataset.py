import requests
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# Step 1: Dataset URL and local file
url = 'https://mlopsliz.blob.core.windows.net/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv'
local_path = 'Telco-Customer-Churn.csv'

# Step 2: Check if file exists locally
if not os.path.exists(local_path):
    print(f"Downloading dataset from {url}...")
    response = requests.get(url)
    with open(local_path, 'wb') as file:
        file.write(response.content)
    print("Download complete!")
else:
    print(f"Dataset already exists at {local_path}, skipping download.")

# Step 3: Connect to Azure ML Workspace (SDK v2 way)
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential)

# Step 4: Upload and register dataset
my_data = Data(
    path=local_path,
    type=AssetTypes.URI_FILE,
    description="Telco Customer Churn dataset",
    name="telco-customer-churn",
    version="1"
)

ml_client.data.create_or_update(my_data)

print("Dataset uploaded and registered successfully.")
