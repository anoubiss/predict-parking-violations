# Parking Violation Prediction - Challenge Data (Egis)

## Objective

Develop a machine learning model to **predict parking violation rates** by zone and time.  
The goal is to **rank zones by risk level** in order to optimize the routing of enforcement vehicles (scan cars) and reduce overall violation rates.

## Dataset Overview

### Provided files:
- `x_train.csv`: features for training
- `y_train.csv`: target variable (violation rate) for training
- `x_test.csv`: features for test set (to predict)

### Key variables:
- **Latitude / Longitude**: geographical location of the zone
- **Timestamp (hourly)**: date and time of the control
- **Violation rate (y_train)**: proportion of vehicles in violation during the hour in the given zone

### External data (optional):
- **Weather conditions** from Météo France Open Data (can be used as additional input features)

## Task

- **Type**: Time series regression with spatio-temporal features
- **Goal**: Predict future violation rates
- **Evaluation metric**: **Spearman correlation coefficient**  
  (used to assess whether high-risk zones are correctly ranked)

## Baseline

- Model: `RandomForestRegressor` trained on zones with at least 45 parked vehicles
- Score: **Spearman = 0.197**
- Note: Only 10 estimators were used due to data volume constraints

## Key Challenges

- **Zone aggregation and filtering** to improve signal reliability
- **Incorporating external variables** such as weather and time-related features
- **Efficient data processing** to handle large volumes

## Suggested Tech Stack

- Python (≥3.8)
- pandas / numpy
- scikit-learn / lightgbm / xgboost
- geopandas / haversine (for spatial features)
- joblib (for parallelization)
- Optionally: time series tools (e.g., Prophet, ARIMA)

## Submission

- Submit predictions for `x_test` in the required format
- Output must contain the predicted violation rate for each row in `x_test`

## Organizer Contact

- Email: challenge-data.france@egis-group.com

---

**Hosted on [Challenge Data](https://challengedata.ens.fr/)**  

