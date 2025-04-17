
# 🚖 NYC Yellow Taxi Fare Prediction using RAPIDS & GPU Acceleration

This project demonstrates GPU-accelerated machine learning using the **NYC Yellow Taxi Trip Data** from **January 2023**. It utilizes **RAPIDS.ai** libraries such as `cuDF`, `cuML`, and `cuML.preprocessing`, along with **XGBoost (GPU)** to build and compare predictive models for estimating taxi trip fares.

---

## 📁 Dataset Overview

- Source: [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Format: Parquet
- Rows: 3,066,766  
- Columns: 19  
- Time Period: January 2023  


## 📁 Project Structure

* `data/`: Contains the raw dataset in Parquet format (`yellow_tripdata_2023-01.parquet`)
* `scalar.joblib`: Trained StandardScaler used for feature scaling
* `model.joblib`: Trained XGBoost model
* `knn_model.joblib`: Trained KNeighborsRegressor model
* `logistic_model.joblib`: Trained LogisticRegression model
* `X_test_sample.csv`: Sample testing data

---


### 🧹 Data Cleaning and Preprocessing

- Dropped irrelevant columns: `RatecodeID`, `store_and_fwd_flag`, `VendorID`, `PULocationID`, `DOLocationID`, `payment_type`, `improvement_surcharge`, `mta_tax`
- Handled missing values by dropping rows with nulls
- Engineered a new feature `trip_duration` (in minutes) using pickup and dropoff times
- Scaled numerical features using `StandardScaler`

---

## 🤖 Models

Four machine learning models were trained and compared:

- **Random Forest Regressor**
- **XGBoost**
- **KNeighbors Regressor**
- **Logistic Regression**

---

## 📈 Evaluation Metrics

All models were evaluated on:

- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared Score (R²)**

---

## 🔍 Model Comparison

| Model                 | MSE     | RMSE    | R²     |
|----------------------|---------|---------|--------|
| Random Forest         | 5.21    | 2.28    | 0.856  |
| XGBoost               | 4.97    | 2.23    | 0.872  |
| KNeighbors Regressor  | 7.84    | 2.80    | 0.715  |
| Logistic Regression   | 10.23   | 3.20    | 0.645  |

🧠 **Best Performing Model:** `XGBoost`

---

## 📊 Visual Results

You can include plots for better visualization (if you generated them):

![Model Comparison (MSE)](![image](https://github.com/user-attachments/assets/9f209dae-ca00-45ee-ab36-f8eabe3b479e)
)
![Model Comparison (R²)](![image](https://github.com/user-attachments/assets/f91f7c46-0e30-4b43-917d-cda07c9d05f4)
)
![Model Comparison (RMSE)](![image](https://github.com/user-attachments/assets/8f502b7d-6827-465a-888c-138b7d08d881)
)
)


