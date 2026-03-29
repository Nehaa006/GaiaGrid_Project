import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

def plot_importance():
    # 1. Load the model
    model = joblib.load('src/models/price_predictor.pkl')
    
    # 2. Define our feature names
    features = ['Temperature', 'Cloud Cover', 'Total Grid Load', 'Household Power']
    importances = model.feature_importances_

    # 3. Create the Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features, palette='viridis')
    
    plt.title('Gaia-Grid AI: Which Factors Influence Price the Most?', fontsize=14)
    plt.xlabel('Importance Score (0.0 to 1.0)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("✅ Slide-ready chart saved as 'feature_importance.png'!")
    plt.show()

if __name__ == "__main__":
    plot_importance()