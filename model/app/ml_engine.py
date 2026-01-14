import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

class MarketTrendEngine:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.stats_store = {
            "MONTHLY": {},
            "DAILY": {}
        }
        self.model_ready = False

    def load_and_train(self):
        try:
            print("Loading and segregating data...")
            raw_df = pd.read_csv(self.data_path)
            raw_df['Date'] = pd.to_datetime(raw_df['Date'])
            
            # 1. Split Data by propertyType
            monthly_data = raw_df[raw_df['propertyType'] == 'MONTHLY']
            daily_data = raw_df[raw_df['propertyType'] == 'DAILY']

            # 2. Process independently
            self.stats_store["MONTHLY"] = self._process_dataset(monthly_data)
            self.stats_store["DAILY"] = self._process_dataset(daily_data)
            
            self.model_ready = True
            print("Models trained for both MONTHLY and DAILY markets.")

        except Exception as e:
            print(f"Error initializing ML Engine: {e}")
            self.model_ready = False

    def _process_dataset(self, df: pd.DataFrame):
        if df.empty: return {}

        # Aggregate: Average price per neighborhood per month
        monthly_df = df.groupby([
            'neighberhood', 
            pd.Grouper(key='Date', freq='ME')
        ])['rentPerMonth'].mean().reset_index()

        coords = df.groupby('neighberhood')[['latitude', 'longitude']].mean()

        # Pivot for Time-Series Matrix
        pivot_df = monthly_df.pivot(index='neighberhood', columns='Date', values='rentPerMonth')
        
        # Interpolation for missing months
        pivot_df = pivot_df.interpolate(axis=1).fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
        
        # Fallback if insufficient data
        if pivot_df.shape[1] < 2 or len(pivot_df) < 2: 
            return self._build_simple_stats(pivot_df, coords)

        # --- NORMALIZE BY GROWTH (Index to First Month = 1.0) ---
        growth_matrix = pivot_df.div(pivot_df.iloc[:, 0], axis=0)

        # Clustering (K-Means on GROWTH patterns)
        n_clusters = min(3, len(pivot_df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(growth_matrix)

        # Determine "HOT" vs "COOL" based on Growth Slope
        cluster_map = {}
        for i, center in enumerate(kmeans.cluster_centers_):
            # Calculate total change percentage of the cluster centroid
            total_change = center[-1] - center[0]
            
            if total_change > 0.02: status = "HOT"      # > 2% Growth
            elif total_change < -0.02: status = "COOL"  # > 2% Drop
            else: status = "STABLE"
            cluster_map[i] = status

        neighborhood_stats = {}
        for neighborhood in pivot_df.index:
            cluster_id = labels[pivot_df.index.get_loc(neighborhood)]
            neighborhood_stats[neighborhood] = {
                "lat": coords.loc[neighborhood, 'latitude'],
                "lon": coords.loc[neighborhood, 'longitude'],
                "status": cluster_map[cluster_id],
                "current_price": pivot_df.loc[neighborhood].iloc[-1],
                "history": pivot_df.loc[neighborhood].values,
                "dates": pivot_df.columns
            }
        return neighborhood_stats

    def _build_simple_stats(self, pivot_df, coords):
        stats = {}
        for neighborhood in pivot_df.index:
            history = pivot_df.loc[neighborhood].values
            slope = np.polyfit(range(len(history)), history, 1)[0]
            status = "HOT" if slope > 0 else "COOL"
            stats[neighborhood] = {
                "lat": coords.loc[neighborhood, 'latitude'],
                "lon": coords.loc[neighborhood, 'longitude'],
                "status": status,
                "current_price": history[-1],
                "history": history,
                "dates": pivot_df.columns
            }
        return stats

    def get_heatmap_data(self, rental_type: str):
        if not self.model_ready: return []
        
        target_stats = self.stats_store.get(rental_type, {})
        results = []
        
        for name, stats in target_stats.items():
            # NO DIVISION NEEDED: Data is already ~9.284
            price = stats['current_price']
            
            results.append({
                "neighborhood": name,
                "latitude": stats['lat'],
                "longitude": stats['lon'],
                "current_avg_price": round(price, 4),
                "trend_status": stats['status'],
                "trend_description": f"{rental_type} market is {stats['status']}"
            })
        return results

    def get_forecast(self, neighborhood_name: str, rental_type: str, months_ahead: int = 6):
        if not self.model_ready: return None
        
        target_stats = self.stats_store.get(rental_type, {})
        if neighborhood_name not in target_stats: return None

        stats = target_stats[neighborhood_name]
        history = stats['history']
        dates = stats['dates']
        
        # Forecast Logic
        X = np.arange(len(history)).reshape(-1, 1)
        y = history.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(X, y)
        
        last_date = dates[-1]
        future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(months_ahead)]
        future_X = np.arange(len(history), len(history) + months_ahead).reshape(-1, 1)
        predictions = model.predict(future_X).flatten()

        chart_data = []
        for d, p in zip(dates, history):
            chart_data.append({
                "date": d.strftime("%Y-%m"), 
                "price": round(p, 4), # NO DIVISION
                "type": "historical"
            })
        for d, p in zip(future_dates, predictions):
            chart_data.append({
                "date": d.strftime("%Y-%m"), 
                "price": round(p, 4), # NO DIVISION
                "type": "forecast"
            })

        growth_rate = ((predictions[-1] - history[-1]) / history[-1]) * 100

        return {
            "neighborhood": neighborhood_name,
            "rental_type": rental_type,
            "trend_status": stats['status'],
            "growth_rate_projection": round(growth_rate, 2),
            "chart_data": chart_data
        }

ml_engine = MarketTrendEngine("properties_data.csv")