import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from scipy.stats import uniform, randint

app = Flask(__name__)

def load_data():
    # Download the CSV from Google Sheets
    sheet_id = '1SbZesknAc9GyWsjFC2GfEdeRb-Z9AiWhG1WMRd-Ejb0'
    sheet_name = 'FilteredData'  # Specify the sheet name
    csv_url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}'
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        print(f"Error fetching or reading the CSV: {e}")
        return None

    # Update the column names based on the actual data
    df = df[['Pitch', 'Swing']]  # Replace 'Pitch' and 'Swing' with the correct column names

    # Convert the data to the correct types if necessary
    df['Pitch'] = df['Pitch'].astype(int)
    df['Swing'] = df['Swing'].astype(int)

    # Feature engineering
    df['Prev_Pitch'] = df['Pitch'].shift(1)
    df['Prev_Swing'] = df['Swing'].shift(1)
    df['Pitch_Swing_Diff'] = df['Swing'] - df['Pitch']
    df['Prev_Pitch_Swing_Diff'] = df['Prev_Swing'] - df['Prev_Pitch']
    df['Pitch_Moving_Avg'] = df['Pitch'].rolling(window=3).mean()
    df['Swing_Moving_Avg'] = df['Swing'].rolling(window=3).mean()
    df['Pitch_Moving_Std'] = df['Pitch'].rolling(window=3).std()
    df['Swing_Moving_Std'] = df['Swing'].rolling(window=3).std()
    df.fillna(0, inplace=True)

    return df

@app.route('/predict', methods=['GET'])
def predict():
    df = load_data()
    if df is None:
        return jsonify({"error": "Failed to load data"}), 500

    # Prepare features and target
    X = df[['Prev_Pitch', 'Prev_Swing', 'Pitch_Swing_Diff', 'Prev_Pitch_Swing_Diff', 'Pitch_Moving_Avg', 'Swing_Moving_Avg', 'Pitch_Moving_Std', 'Swing_Moving_Std']]
    y = df['Pitch']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models to compare
    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }

    # Hyperparameter grid for Randomized Search
    param_grids = {
        'RandomForest': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(5, 30),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 10)
        },
        'GradientBoosting': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.2),
            'subsample': uniform(0.6, 0.4)
        },
        'XGBoost': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.2),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        }
    }

    # Find the best model and hyperparameters using Randomized Search
    best_models = {}
    for model_name in models:
        random_search = RandomizedSearchCV(
            estimator=models[model_name], 
            param_distributions=param_grids[model_name], 
            n_iter=100, 
            cv=5, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1, 
            random_state=42
        )
        random_search.fit(X_train, y_train)
        best_models[model_name] = random_search.best_estimator_

    # Evaluate models using cross-validation
    cv_results = {}
    for model_name, model in best_models.items():
        cv_score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_results[model_name] = np.sqrt(-cv_score.mean())

    # Choose the best model based on CV RMSE
    best_model_name = min(cv_results, key=cv_results.get)
    best_model = best_models[best_model_name]

    # Predict the next pitch using the best model
    last_pitch, last_swing = df.iloc[-1][['Pitch', 'Swing']]
    prev_pitch = df['Pitch'].iloc[-2]
    prev_swing = df['Swing'].iloc[-2]
    pitch_swing_diff = last_swing - last_pitch
    prev_pitch_swing_diff = prev_swing - prev_pitch
    pitch_moving_avg = df['Pitch'].rolling(window=3).mean().iloc[-1]
    swing_moving_avg = df['Swing'].rolling(window=3).mean().iloc[-1]
    pitch_moving_std = df['Pitch'].rolling(window=3).std().iloc[-1]
    swing_moving_std = df['Swing'].rolling(window=3).std().iloc[-1]

    # Create a DataFrame for the new data point
    new_data = pd.DataFrame({
        'Prev_Pitch': [last_pitch],
        'Prev_Swing': [last_swing],
        'Pitch_Swing_Diff': [pitch_swing_diff],
        'Prev_Pitch_Swing_Diff': [prev_pitch_swing_diff],
        'Pitch_Moving_Avg': [pitch_moving_avg],
        'Swing_Moving_Avg': [swing_moving_avg],
        'Pitch_Moving_Std': [pitch_moving_std],
        'Swing_Moving_Std': [swing_moving_std]
    })

    predicted_next_pitch = best_model.predict(new_data)[0]

    # Determine the predicted pitch bucket
    def get_pitch_bucket(pitch):
        if pitch < 101:
            return "1"
        elif pitch < 201:
            return "101"
        elif pitch < 301:
            return "201"
        elif pitch < 401:
            return "301"
        elif pitch < 501:
            return "401"
        elif pitch < 601:
            return "501"
        elif pitch < 701:
            return "601"
        elif pitch < 801:
            return "701"
        elif pitch < 901:
            return "801"
        else:
            return "901"

    predicted_next_pitch_bucket = get_pitch_bucket(predicted_next_pitch)

    return jsonify({
        'Predicted Next Pitch': predicted_next_pitch,
        'Predicted Next Pitch Bucket': predicted_next_pitch_bucket
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
