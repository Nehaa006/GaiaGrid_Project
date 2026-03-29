import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def plot_gaia_performance():
    results_path = 'gaia_simulation_results.csv'
    if not os.path.exists(results_path):
        print("❌ Error: Run 'python main.py' first!")
        return

    # 1. Load and Parse
    df = pd.read_csv(results_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # 2. SLICE THE DATA (Crucial for clarity)
    # Showing the first 7 days (168 hours) so the graph is actually readable
    df_plot = df.head(168).copy() 

    # 3. Setup Plot
    plt.style.use('bmh') # Clean, professional grid style
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # 4. Plot Energy Price (Left Axis)
    color_price = '#d62728' # Professional Crimson
    ax1.set_xlabel('Timeline (Showing First 7 Days of Simulation)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Energy Price (EUR/MWh)', color=color_price, fontsize=12, fontweight='bold')
    ax1.plot(df_plot['timestamp'], df_plot['price'], color=color_price, linewidth=2, label='Grid Price', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_price)

    # 5. Plot Battery SoC (Right Axis)
    ax2 = ax1.twinx()
    color_battery = '#1f77b4' # Professional Steel Blue
    ax2.set_ylabel('Battery Level (kWh)', color=color_battery, fontsize=12, fontweight='bold')
    ax2.fill_between(df_plot['timestamp'], df_plot['battery_soc'], color=color_battery, alpha=0.2, label='Battery State of Charge')
    ax2.set_ylim(0, 15) # Focused on the battery capacity
    ax2.tick_params(axis='y', labelcolor=color_battery)

    # 6. Formatting the X-Axis (The "No More Black Bar" fix)
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1)) # One label per day
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d')) # Format: Jan 01
    plt.gcf().autofmt_xdate() # Tilted for elegance

    # 7. Final Polish
    plt.title('Gaia-Grid Agent Performance: Predictive Arbitrage Results', fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    output_image = 'gaia_clean_results.png'
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"✅ Neat and clean graph saved as '{output_image}'")
    plt.show()

if __name__ == "__main__":
    plot_gaia_performance()