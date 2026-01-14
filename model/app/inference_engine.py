import joblib
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

MODEL_FILE = os.path.join(os.path.dirname(__file__), "../models/k_means_model.joblib")

class InferenceEngine:
    def __init__(self):
        self.stats_store = {}
        self.load_model()

    def load_model(self):
        """Loads the pre-trained stats from the pickle file."""
        if not os.path.exists(MODEL_FILE):
            print(f"Warning: {MODEL_FILE} not found. Please run train_model.py.")
            self.stats_store = {}
            return

        self.stats_store = joblib.load(MODEL_FILE)
        print("Model loaded successfully.")

    def get_heatmap(self, rental_type: str):
        data = self.stats_store.get(rental_type, {})
        results = []
        for name, stats in data.items():
            results.append({
                "neighborhood": name,
                "latitude": stats['lat'],
                "longitude": stats['lon'],
                "current_avg_price": round(stats['current_price'], 4),
                "trend_status": stats['status'],
                "trend_description": f"{rental_type} market is {stats['status']}"
            })
        return results

    def get_forecast(self, neighborhood: str, rental_type: str, months=6):
        data = self.stats_store.get(rental_type, {})
        if neighborhood not in data: return None

        stats = data[neighborhood]
        history = stats['history']
        dates = stats['dates']

        # Run linear projection (Fast)
        X = np.arange(len(history)).reshape(-1, 1)
        y = history.reshape(-1, 1)
        model = LinearRegression().fit(X, y)

        last_date = dates[-1]
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(months)]
        future_X = np.arange(len(history), len(history) + months).reshape(-1, 1)
        predictions = model.predict(future_X).flatten()

        chart_data = []
        for d, p in zip(dates, history):
            chart_data.append({"date": d.strftime("%Y-%m"), "price": round(p, 4), "type": "historical"})
        for d, p in zip(future_dates, predictions):
            chart_data.append({"date": d.strftime("%Y-%m"), "price": round(p, 4), "type": "forecast"})

        growth = ((predictions[-1] - history[-1]) / history[-1]) * 100
        
        return {
            "neighborhood": neighborhood,
            "rental_type": rental_type,
            "trend_status": stats['status'],
            "growth_rate_projection": round(growth, 2),
            "chart_data": chart_data
        }

# Create a singleton instance
inference_engine = InferenceEngine()