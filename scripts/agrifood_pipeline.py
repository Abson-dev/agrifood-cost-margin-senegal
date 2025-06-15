
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
from rasterio.transform import from_origin
import spacy
import re
from shapely.geometry import Point, LineString
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# SECTION 1: NLP Price Extraction
def extract_price_data(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    prices = []
    for sent in doc.sents:
        match = re.search(r'(\w+)\s+price.*?([A-Za-z]+)\s+is\s+KSh\s+(\d+)', sent.text, re.IGNORECASE)
        if match:
            commodity, location, price = match.groups()
            prices.append({'commodity': commodity.lower(), 'location': location, 'price': float(price)})
    return pd.DataFrame(prices)

# SECTION 2: Sample Data for Margin Modeling
def generate_margin_data():
    df = pd.DataFrame({
        'commodity': ['maize', 'maize', 'rice'],
        'price_farm': [40, 42, 60],
        'price_retail': [65, 67, 100],
        'distance_to_market_km': [80, 150, 60],
        'road_density': [2.0, 0.5, 1.2],
        'storage_availability': [1, 0, 1]
    })
    df['gross_margin'] = df['price_retail'] - df['price_farm']
    df['transaction_cost'] = (
        0.5 * df['distance_to_market_km'] +
        10 / (df['road_density'] + 1) +
        15 * (1 - df['storage_availability'])
    )
    df['net_margin'] = df['gross_margin'] - df['transaction_cost']
    return df

# SECTION 3: Train Margin Prediction Model
def train_model(df):
    X = df[['price_farm', 'price_retail', 'distance_to_market_km', 'road_density', 'storage_availability']]
    y = df['net_margin']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    joblib.dump(model, "net_margin_model.pkl")
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Model trained. RMSE: {rmse:.2f}")

# SECTION 4: Generate and Plot Synthetic Geospatial Data
def generate_geospatial_data():
    roads = gpd.GeoDataFrame({
        'name': ['Main Road A', 'Feeder Road B'],
        'road_type': ['primary', 'secondary'],
        'geometry': [LineString([(0, 0), (5, 5)]), LineString([(3, 0), (3, 5)])]
    }, crs="EPSG:4326")
    roads.to_file("sample_roads.geojson", driver="GeoJSON")

    markets = gpd.GeoDataFrame({
        'market': ['Market 1', 'Market 2'],
        'geometry': [Point(1, 1), Point(4, 4)]
    }, crs="EPSG:4326")
    markets.to_file("sample_markets.geojson", driver="GeoJSON")

    width = height = 100
    data = np.random.rand(height, width) * 100
    transform = from_origin(0, 10, 0.1, 0.1)
    with rasterio.open(
        "sample_cost_surface.tif", 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(data, 1)

    print("Synthetic geospatial data generated.")

# SECTION 5: Run All Steps
if __name__ == "__main__":
    text = "Retail maize price in Kisumu is KSh 63 per kg. Wholesale rice price in Nairobi is KSh 95."
    print("Extracted prices:")
    print(extract_price_data(text))

    df = generate_margin_data()
    print("Training margin model...")
    train_model(df)

    print("Generating synthetic geospatial files...")
    generate_geospatial_data()
