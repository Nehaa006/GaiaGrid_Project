# GaiaGrid 🌍⚡

> **An Intelligent Hybrid Predictive-Generative Framework for Short-Term Electricity Price Forecasting and Decision Support Analytics**

GaiaGrid is an end-to-end artificial intelligence pipeline designed to forecast short-term electricity prices and provide advanced market analytics. The platform utilizes an optimized **Random Forest Regressor** to model highly volatile, deregulated electricity market prices by fusing real-time grid load profiles with regional meteorological data. 

To bridge the gap between technical forecasting and human operations, GaiaGrid integrates a **Retrieval-Augmented Generation (RAG)** pipeline. This allows system operators to query the framework and instantly receive context-aware, text-based market analysis based on historical patterns and localized domain knowledge.

---

## 🚀 Key Features

* **Multimodal Feature Fusion:** Dynamically combines exogenous weather features (`temp`, `humidity`, `wind_speed`) with endogenous grid metrics (`total_load_actual`) to capture environmental and demand-driven market shocks.
* **Ensemble Predictive Core:** Implements a tuned Random Forest Regressor yielding high-precision modeling with a Coefficient of Determination ($R^2 = 82.60\%$) and a Mean Absolute Error ($MAE = \text{€}3.92$).
* **Generative Knowledge Base (RAG):** Couples a localized vector database with domain-specific market logic and textual energy documentation (`energy_facts.txt`) to explain the economic "why" behind sudden price movements.

---

## 📊 Evaluation & Validation Metrics

The ensemble regression architecture successfully captures complex non-linear pricing phenomena common in highly volatile, pool-based energy systems with deep renewable penetration:

* **$R^2$ Score (Coefficient of Determination):** **82.60%** (Proving high reliability against variance)
* **Mean Absolute Error (MAE):** **€3.92**

---

## 🔮 Future Enhancements

Planned upgrades to extend the system's operational accessibility and interactive capabilities include:

* **Voice-Activated Interaction (NLI):** Implementing a Natural Language Interface layer featuring integrated Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) drivers to allow hands-free, vocal model queries.
* **Real-time API Ingestion:** Connecting live weather and grid APIs to shift the system from static validation sets to real-time stream inferencing.
