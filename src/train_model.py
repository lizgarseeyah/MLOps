import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from azureml.core import Workspace
from azureml.core.model import Model
import joblib

# 1. Load your dataset
data = pd.read_csv('https://mlopsliz.blob.core.windows.net/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')  # Change this to your dataset's location

# Best Practice: Check your data for any missing values
print(data.isnull().sum())

# Handle missing values (if any) - here we'll just drop them (you can also fill them with a mean/median if preferred)
data = data.dropna()

# 2. Feature engineering: Drop non-predictive columns or any unnecessary data
# For example, drop a customer ID or any columns that don't affect the churn prediction
X = data.drop(['Churn', 'CustomerID'], axis=1)  # 'Churn' is your target, 'CustomerID' is just an identifier
y = data['Churn']  # This is the target you're trying to predict

# Best Practice: Standardize the features (e.g., using MinMaxScaler, StandardScaler) for most ML models
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split the dataset into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Best Practice: Set a random_state for reproducibility of results


# 4. Train the model
model = LogisticRegression(max_iter=1000)  # Best practice: increase max_iter if needed
model.fit(X_train, y_train)

# 5. Make predictions on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression Model: ", accuracy)

# 7. Register the trained model
ws = Workspace.from_config()  # Ensure you have the correct Azure workspace configuration

# Save the model locally
joblib.dump(model, 'customer_churn_model.pkl')

# Register the model in Azure ML
model = Model.register(workspace=ws,
                       model_path="customer_churn_model.pkl",  # The saved model file
                       model_name="customer_churn_model")  # Model name in Azure ML

print(f"Model registered: {model.name} (Version {model.version})")