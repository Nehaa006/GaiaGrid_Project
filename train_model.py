import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from src.data_loader import load_gaia_data

def train_gaia_brain():
    df = load_gaia_data()
    
    # Selecting the exact columns available in the Spanish dataset
    # 'price actual' is our target (what we want to predict)
    # 'temp', 'clouds_all', and 'total load actual' are our features
    features = ['temp', 'clouds_all', 'total load actual', 'Global_active_power']
    target = 'price actual'

    print("🧠 Training Random Forest Regressor...")
    X = df[features]
    y = df[target]

    # Create model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Ensure the models directory exists
    os.makedirs('src/models', exist_ok=True)
    
    # Save the brain
    joblib.dump(model, 'src/models/price_predictor.pkl')
    print("✅ Model saved successfully to src/models/price_predictor.pkl")

if __name__ == "__main__":
    train_gaia_brain()


#git config --global user.name Nehaa006       git config --global user.email nehaalakshmi2006@gmail.com
