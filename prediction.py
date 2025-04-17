import joblib
import pandas as pd
import xgboost as xgb

model = joblib.load("model.joblib")
print("Loaded model.joblib")

scaler = joblib.load("scalar.joblib")
print("Loaded scalar.joblib")

test_data = pd.read_csv("X_test.csv")  # Replace 'X_test.csv' with your actual test data file
print("Loaded test data from X_test.csv")

test_data_scaled = scaler.transform(test_data)

predictions = model.predict(xgb.DMatrix(test_data_scaled))

pd.DataFrame(predictions, columns=["predicted"]).to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv")

print("Sample Predictions (first 5):")
print(predictions[:5])
