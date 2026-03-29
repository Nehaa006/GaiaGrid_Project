import pandas as pd
from src.data_loader import load_gaia_data
from src.agent import GaiaGridAgent
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def run_gaia_simulation():
    # 1. Initialize
    data = load_gaia_data()
    agent = GaiaGridAgent()
    log = []

    print("🚀 Running Agentic Edge AI Simulation...")
    
    # 2. Process hourly steps
    for ts, row in data.iterrows():
        # Input features
        cons = row['Global_active_power']
        prc = row['price actual']
        tmp = row['temp']
        cld = row['clouds_all']
        
        # Get Agent decision
        act = agent.compute_action(cons, prc, tmp, cld)
        
        log.append({
            'timestamp': ts,
            'price': prc,
            'action': act,
            'battery_soc': agent.battery_level,
            'cumulative_savings': agent.total_savings
        })

    # 3. Export Results
    results_df = pd.DataFrame(log)
    results_df.to_csv('gaia_simulation_results.csv', index=False)
    print(f"🏁 Simulation Complete! Total Savings: ${agent.total_savings:.2f}")
    print("📊 Results saved to 'gaia_simulation_results.csv'")

if __name__ == "__main__":
    run_gaia_simulation()


def run_simulation_with_metrics(df, agent):
    y_true = []
    y_pred = []
    y_scores = [] # For ROC Curve (the 'confidence' or price delta)

    for i in range(len(df)-1):
        row = df.iloc[i]
        next_row = df.iloc[i+1]
        
        # 1. THE GROUND TRUTH (The "Oracle")
        # If price goes up significantly in the next hour, the CORRECT action was 'CHARGE' now.
        if next_row['price actual'] > row['price actual'] * 1.1:
            true_action = 1 # Charge
        elif next_row['price actual'] < row['price actual'] * 0.9:
            true_action = 0 # Discharge/Hold
        else:
            true_action = 0 # Hold
            
        # 2. THE AGENT'S PREDICTION
        agent_action_str = agent.compute_action(row['Global_active_power'], row['price actual'], row['temp'], row['clouds_all'])
        agent_action = 1 if "CHARGE" in agent_action_str else 0
        
        y_true.append(true_action)
        y_pred.append(agent_action)
        # Use price difference as a proxy for 'probability score' for ROC
        y_scores.append(next_row['price actual'] - row['price actual'])

    return y_true, y_pred, y_scores