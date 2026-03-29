import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
from src.data_loader import load_gaia_data

def evaluate_gaia_brain():
    print("🚀 Evaluation Started...") # <--- Added this to confirm it's running
    df = load_gaia_data()
    
    features = ['temp', 'clouds_all', 'total load actual', 'Global_active_power']
    target = 'price actual'
    
    test_size = int(len(df) * 0.2)
    test_df = df.tail(test_size)
    
    X_test = test_df[features]
    y_actual = test_df[target]

    model = joblib.load('src/models/price_predictor.pkl')
    y_pred = model.predict(X_test)

    r2 = r2_score(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)

    print("\n" + "="*30)
    print(f"✅ ACCURACY RATE (R2): {r2 * 100:.2f}%") # This gives you the % you wanted!
    print(f"📉 ERROR MARGIN (MAE): €{mae:.2f}")
    print("="*30)

# THIS PART BELOW IS MANDATORY - WITHOUT IT, THE PROGRAM WON'T RUN
if __name__ == "__main__":
    evaluate_gaia_brain()