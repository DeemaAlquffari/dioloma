import joblib
import pandas as pd
import xgboost as xgb

try:
    model = joblib.load("model.joblib")  
    print("Loaded model.joblib")
except FileNotFoundError:
    raise FileNotFoundError("model.joblib not found. Please make sure the model file exists.")

try:
    X_test_sample = pd.read_csv("X_test_sample.csv")  # Assumes the test data is saved as 'X_test_sample.csv'
    print("Loaded X_test_sample.csv")
except FileNotFoundError:
    raise FileNotFoundError("X_test_sample.csv not found. Please make sure the test data file exists.")

y_pred = model.predict(xgb.DMatrix(X_test_sample)) 

pd.DataFrame(y_pred, columns=["predicted"]).to_csv("y_pred.csv", index=False)
print("Predictions saved to y_pred.csv")

print("\nSample Predictions:")
print(y_pred[:5])
