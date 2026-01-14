import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
np.random.seed(42)

# Specific Neighborhoods with Coordinates and assigned Trends
# Trend > 0.02 = HOT
# Trend < -0.02 = COOL
# Trend between -0.02 and 0.02 = STABLE
neighborhoods = [
    # --- RABAT ---
    {"name": "Hassan", "lat": 34.024, "lon": -6.840, "trend": 0.05},      # HOT
    {"name": "Agdal (Rabat)", "lat": 34.00125, "lon": -6.85751, "trend": 0.01}, # STABLE
    {"name": "Hay Riad", "lat": 33.95595, "lon": -6.87263, "trend": 0.08}, # HOT
    {"name": "Souissi", "lat": 34.036, "lon": -6.819, "trend": 0.03},     # HOT
    {"name": "Médina (Rabat)", "lat": 34.017, "lon": -6.832, "trend": -0.03}, # COOL
    {"name": "Océan", "lat": 34.023, "lon": -6.821, "trend": -0.01},      # STABLE
    {"name": "Yacoub El Mansour", "lat": 34.018, "lon": -6.863, "trend": -0.04}, # COOL

    # --- CASABLANCA ---
    {"name": "Maarif", "lat": 33.590, "lon": -7.620, "trend": 0.06},      # HOT
    {"name": "Gauthier", "lat": 33.585, "lon": -7.620, "trend": 0.04},    # HOT
    {"name": "Racine", "lat": 33.584, "lon": -7.618, "trend": 0.05},      # HOT
    {"name": "Anfa", "lat": 33.596, "lon": -7.633, "trend": 0.07},        # HOT
    {"name": "Bourgogne", "lat": 33.585, "lon": -7.607, "trend": -0.02},  # STABLE
    {"name": "Ain Diab", "lat": 33.580, "lon": -7.701, "trend": 0.03},    # HOT
    {"name": "Sidi Maarouf", "lat": 33.562, "lon": -7.652, "trend": 0.02}, # STABLE
    {"name": "Belvédère", "lat": 33.587, "lon": -7.623, "trend": -0.01},  # STABLE
    {"name": "Hay Mohammadi", "lat": 33.607, "lon": -7.577, "trend": -0.05}, # COOL

    # --- MARRAKECH ---
    {"name": "Guéliz", "lat": 31.634, "lon": -7.995, "trend": 0.05},      # HOT
    {"name": "Hivernage", "lat": 31.629, "lon": -7.980, "trend": 0.04},    # HOT
    {"name": "Médina (Marrakech)", "lat": 31.629, "lon": -7.981, "trend": -0.02}, # STABLE
    {"name": "Palmeraie", "lat": 31.683, "lon": -7.987, "trend": 0.06},    # HOT
    {"name": "Agdal (Marrakech)", "lat": 31.630, "lon": -7.987, "trend": 0.01}, # STABLE
    {"name": "Targa", "lat": 31.649, "lon": -7.972, "trend": 0.02},      # STABLE

    # --- TANGIER ---
    {"name": "Malabata", "lat": 35.783, "lon": -5.741, "trend": 0.09},    # HOT
    {"name": "Iberia", "lat": 35.767, "lon": -5.798, "trend": 0.04},     # HOT
    {"name": "Marshan", "lat": 35.780, "lon": -5.807, "trend": -0.01},   # STABLE
    {"name": "Centre-ville (Tangier)", "lat": 35.767, "lon": -5.799, "trend": 0.03}, # HOT
    {"name": "Branes", "lat": 35.773, "lon": -5.805, "trend": -0.03},    # COOL
    {"name": "Achakar", "lat": 35.766, "lon": -5.805, "trend": 0.05},    # HOT

    # --- FES ---
    {"name": "Fès El Bali", "lat": 34.059, "lon": -5.003, "trend": -0.02}, # STABLE
    {"name": "Fès El Jedid", "lat": 34.046, "lon": -5.007, "trend": -0.03}, # COOL
    {"name": "Ville Nouvelle (Fès)", "lat": 34.041, "lon": -5.002, "trend": 0.01}, # STABLE
    {"name": "Route d’Imouzzer", "lat": 34.043, "lon": -5.010, "trend": 0.02}, # STABLE
    {"name": "Narjiss", "lat": 34.054, "lon": -5.000, "trend": -0.01},   # STABLE

    # --- AGADIR ---
    {"name": "Talborjt", "lat": 30.429, "lon": -9.600, "trend": 0.02},    # STABLE
    {"name": "Founty", "lat": 30.425, "lon": -9.603, "trend": 0.06},      # HOT

    # --- DAKHLA ---
    {"name": "Anza", "lat": 23.666, "lon": -15.948, "trend": 0.10},       # HOT (Emerging)

    # --- SALE ---
    {"name": "Bettana", "lat": 34.058, "lon": -6.807, "trend": 0.01},     # STABLE
    {"name": "Hay Salam", "lat": 34.055, "lon": -6.780, "trend": -0.02},  # STABLE
    {"name": "Tabriquet", "lat": 34.053, "lon": -6.820, "trend": -0.04},  # COOL
    {"name": "Médina (Salé)", "lat": 34.055, "lon": -6.797, "trend": -0.03}, # COOL
    {"name": "Sidi Moussa", "lat": 34.054, "lon": -6.785, "trend": -0.05}, # COOL

    # --- MEKNES ---
    {"name": "Hamria", "lat": 33.893, "lon": -5.540, "trend": 0.01},      # STABLE
    {"name": "Ville Nouvelle (Meknès)", "lat": 33.892, "lon": -5.544, "trend": 0.00}, # STABLE
    {"name": "Médina (Meknès)", "lat": 33.896, "lon": -5.552, "trend": -0.02}, # STABLE
    {"name": "Toulal", "lat": 33.880, "lon": -5.574, "trend": 0.02},     # STABLE
    {"name": "Marjane", "lat": 33.900, "lon": -5.550, "trend": 0.03}     # HOT
]

# Base Prices
base_price_monthly = 0.105 
base_price_daily = 0.0103

# Date Range (2 Years)
end_date = datetime.now()
dates = [end_date - timedelta(days=x) for x in range(0, 730, 15)]

data = []

print(f"Generating data for {len(neighborhoods)} neighborhoods...")

for neigh in neighborhoods:
    # 1. MONTHLY DATA (Long Term)
    for _ in range(4): # 4 Properties per neighborhood
        # Random variance in starting price (some expensive, some cheap)
        prop_base = base_price_monthly * np.random.uniform(0.85, 1.15)
        
        for d in dates:
            # Linear Trend Calculation
            months_passed = (d - dates[-1]).days / 30
            trend_factor = 1 + (neigh["trend"] * (months_passed / 24))
            
            # Noise (Random market fluctuation)
            noise = np.random.normal(0, 0.015)
            
            price = prop_base * trend_factor * (1 + noise)
            
            data.append({
                "rentPerMonth": round(price, 4),
                "latitude": neigh["lat"] + np.random.normal(0, 0.002), # Slight random offset
                "longitude": neigh["lon"] + np.random.normal(0, 0.002),
                "propertyType": "MONTHLY",
                "Date": d,
                "neighberhood": neigh["name"]
            })

    # 2. DAILY DATA (Short Term / Airbnb)
    for _ in range(4): 
        prop_base = base_price_daily * np.random.uniform(0.80, 1.20)
        
        for d in dates:
            # Seasonality (Higher in Summer: June/July/Aug)
            month = d.month
            seasonality = 1.25 if month in [6, 7, 8] else 1.0
            if neigh["name"] in ["Anza", "Malabata", "Founty", "Ain Diab"]: # Beach areas peak higher
                seasonality = 1.4 if month in [6, 7, 8] else 0.9
            
            months_passed = (d - dates[-1]).days / 30
            trend_factor = 1 + (neigh["trend"] * (months_passed / 24))
            
            price = prop_base * trend_factor * seasonality * np.random.normal(1, 0.04)
            
            data.append({
                "rentPerMonth": round(price, 4),
                "latitude": neigh["lat"] + np.random.normal(0, 0.002),
                "longitude": neigh["lon"] + np.random.normal(0, 0.002),
                "propertyType": "DAILY",
                "Date": d,
                "neighberhood": neigh["name"]
            })

df = pd.DataFrame(data)
df.to_csv("model/data/properties_data.csv", index=False)
print("Success! model/data/properties_data.csv created with Moroccan market trends.")