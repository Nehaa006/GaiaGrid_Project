import pandas as pd
import numpy as np
import warnings
import os
from dotenv import load_dotenv

# Internal Imports
from src.data_loader import load_gaia_data
from src.agent import GaiaGridAgent
from src.rag_agent import GaiaConsultant
from src.vector_store import build_knowledge_base

# Setup
warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

def run_gaia_simulation():
    """Runs the battery simulation based on CSV data"""
    data = load_gaia_data()
    agent = GaiaGridAgent()
    log = []

    print("🚀 Running Agentic Edge AI Simulation...")
    
    for ts, row in data.iterrows():
        cons = row['Global_active_power']
        prc = row['price actual']
        tmp = row['temp']
        cld = row['clouds_all']
        
        act = agent.compute_action(cons, prc, tmp, cld)
        
        log.append({
            'timestamp': ts, 
            'price': prc, 
            'action': act, 
            'battery_soc': agent.battery_level, 
            'cumulative_savings': agent.total_savings
        })

    results_df = pd.DataFrame(log)
    results_df.to_csv('gaia_simulation_results.csv', index=False)
    
    print(f"🏁 Simulation Complete! Total Savings: €{agent.total_savings:.2f}")
    return results_df, agent

def start_chatbot(results, battery_agent, consultant):
    """The interactive loop for talking to Gaia and controlling the battery"""
    print("\n" + "="*50)
    print("💬 GAIA INTERACTIVE CONTROL CENTER")
    print("Commands: 'Charge', 'Discharge', 'Status', or any question.")
    print("Type 'Exit' to stop.")
    print("="*50)

    while True:
        user_query = input("\n👤 YOU: ").strip().lower()
        
        if user_query in ['exit', 'quit', 'bye']:
            print("👋 Gaia signing off. Stay optimized!")
            break
            
        if user_query == 'status':
            print(f"🔋 Current Battery: {battery_agent.battery_level:.1f}%")
            print(f"💰 Total Savings: €{battery_agent.total_savings:.2f}")
            continue

        # --- AGENTIC ACTION LOGIC ---
        # We manually trigger the battery state and change the context for the AI
        current_action = results.iloc[-1]['action'] # Default to last simulated action
        
        if "charge" in user_query:
            battery_agent.battery_level = min(100.0, battery_agent.battery_level + 10.0)
            current_action = "MANUAL_CHARGE"
            print(f"⚡ [ACTION]: Increasing battery SOC...")
            
        elif "discharge" in user_query:
            battery_agent.battery_level = max(0.0, battery_agent.battery_level - 10.0)
            current_action = "MANUAL_DISCHARGE"
            print(f"📉 [ACTION]: Decreasing battery SOC...")

        # Get the latest price context
        last_price = results.iloc[-1]['price']
        
        print("🤔 Gaia is thinking...")
        
        # Call the RAG Agent (Groq + Vector Store)
        response = consultant.explain_action(
            price=last_price,
            action=current_action,
            context_query=user_query
        )
        
        print(f"\n🤖 GAIA: {response}")
        print(f"🔋 [SYSTEM UPDATE]: Battery SOC is now {battery_agent.battery_level:.1f}%")

if __name__ == "__main__":
    # 1. Run the simulation
    results, battery_agent = run_gaia_simulation()

    # 2. Initialize RAG Agent (Memory + Voice)
    print("\n--- INITIALIZING GAIA VOICE SYSTEM ---")
    # This will load the existing './gaia_memory' folder automatically
    vector_db = build_knowledge_base("./knowledge")
    consultant = GaiaConsultant(vector_db)

    # 3. Enter Chat Mode
    start_chatbot(results, battery_agent, consultant)
