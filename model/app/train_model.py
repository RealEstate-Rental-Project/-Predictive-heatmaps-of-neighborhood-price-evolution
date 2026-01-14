import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans

# Defines where to save the trained brain

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../models/k_means_model.joblib")
DATA_FILE = os.path.join(os.path.dirname(__file__), "../data/properties_data.csv")

def train():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run generate_data.py first.")
        return

    print("Loading data...")
    raw_df = pd.read_csv(DATA_FILE)
    raw_df['Date'] = pd.to_datetime(raw_df['Date'])

    # We will store the processed stats here
    final_output = {
        "MONTHLY": process_dataset(raw_df[raw_df['propertyType'] == 'MONTHLY']),
        "DAILY": process_dataset(raw_df[raw_df['propertyType'] == 'DAILY'])
    }

    # Save to disk
    joblib.dump(final_output, OUTPUT_FILE)
    
    # Save feature matrix and preprocessor for MONTHLY
    monthly_df = pd.read_csv(DATA_FILE)
    monthly_df['Date'] = pd.to_datetime(monthly_df['Date'])
    monthly_data = monthly_df[monthly_df['propertyType'] == 'MONTHLY']
    monthly_df_agg = monthly_data.groupby(['neighberhood', pd.Grouper(key='Date', freq='ME')])['rentPerMonth'].mean().reset_index()
    pivot_df = monthly_df_agg.pivot(index='neighberhood', columns='Date', values='rentPerMonth')
    pivot_df = pivot_df.interpolate(axis=1).fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
    growth_matrix = pivot_df.div(pivot_df.iloc[:, 0], axis=0)
    np.save(os.path.join(os.path.dirname(__file__), "../data/property_feature_matrix.npy"), growth_matrix.values)
    n_clusters = min(3, len(pivot_df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(growth_matrix)
    joblib.dump(kmeans, os.path.join(os.path.dirname(__file__), "../models/preprocessor.joblib"))
    
    print(f"Training Complete. Model saved to {OUTPUT_FILE}")

def process_dataset(df):
    if df.empty: return {}
    # 1. Aggregate Time Series
    monthly_df = df.groupby(['neighberhood', pd.Grouper(key='Date', freq='ME')])['rentPerMonth'].mean().reset_index()
    coords = df.groupby('neighberhood')[['latitude', 'longitude']].mean()
    pivot_df = monthly_df.pivot(index='neighberhood', columns='Date', values='rentPerMonth')
    pivot_df = pivot_df.interpolate(axis=1).fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
    # 2. Normalize by Growth (Start = 1.0)
    growth_matrix = pivot_df.div(pivot_df.iloc[:, 0], axis=0)
    # 3. Clustering
    n_clusters = min(3, len(pivot_df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(growth_matrix)
    # 4. Define Status based on Cluster Centroid Slope
    cluster_status = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        total_change = center[-1] - center[0]
        if total_change > 0.02: status = "HOT"
        elif total_change < -0.02: status = "COOL"
        else: status = "STABLE"
        cluster_status[i] = status
    # 5. Build Stats Dictionary
    stats = {}
    for neighborhood in pivot_df.index:
        cluster_id = labels[pivot_df.index.get_loc(neighborhood)]
        stats[neighborhood] = {
            "lat": coords.loc[neighborhood, 'latitude'],
            "lon": coords.loc[neighborhood, 'longitude'],
            "status": cluster_status[cluster_id],
            "current_price": pivot_df.loc[neighborhood].iloc[-1],
            "history": pivot_df.loc[neighborhood].values,
            "dates": pivot_df.columns
        }
    return stats

if __name__ == "__main__":
    train()