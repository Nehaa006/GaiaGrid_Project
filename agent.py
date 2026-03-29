import joblib
import numpy as np
import os

class GaiaGridAgent:
    def __init__(self, model_path='src/models/price_predictor.pkl'):
        self.model = None
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print("🤖 Gaia Agent: Neural weights loaded.")
        else:
            print("⚠️ Gaia Agent: No model found! Run train_model.py first.")

        self.battery_level = 7.0
        self.capacity = 13.5
        self.total_savings = 0.0

    def compute_action(self, cons, prc, tmp, cld):
        # Use the model to see 1 hour into the future
        features = np.array([[cons, prc, tmp, cld]])
        
        if self.model:
            predicted_price = self.model.predict(features)[0]
        else:
            predicted_price = prc # Fallback

        # AGENTIC REASONING
        # If predicted price is much higher than now -> CHARGE
        if predicted_price > prc * 1.15 and self.battery_level < self.capacity:
            return "PREEMPTIVE_CHARGE"
        
        # If current price is high and it's expected to drop -> DISCHARGE
        elif prc > 65 and predicted_price < prc and self.battery_level > 1.0:
            self.total_savings += (cons * (prc/1000))
            self.battery_level -= 1.0
            return "SMART_DISCHARGE"
            
        return "HOLD"