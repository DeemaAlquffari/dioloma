import joblib
import pandas as pd
import xgboost as xgb

# Load the trained XGBoost model using joblib
try:
    model = joblib.load("model.joblib")  # Assumes the model is saved as 'model.joblib'
    print("Loaded model.joblib")
except FileNotFoundError:
    raise FileNotFoundError("model.joblib not found. Please make sure the model file exists.")

# Load the test data
try:
    X_test_sample = pd.read_csv("X_test_sample.csv")  # Assumes the test data is saved as 'X_test_sample.csv'
    print("Loaded X_test_sample.csv")
except FileNotFoundError:
    raise FileNotFoundError("X_test_sample.csv not found. Please make sure the test data file exists.")

# Make predictions
y_pred = model.predict(xgb.DMatrix(X_test_sample))  # Convert the test data to DMatrix for XGBoost predictions

# Save predictions to a CSV file
pd.DataFrame(y_pred, columns=["predicted"]).to_csv("y_pred.csv", index=False)
print("Predictions saved to y_pred.csv")

# Print a few sample predictions
print("\nSample Predictions:")
print(y_pred[:5])
