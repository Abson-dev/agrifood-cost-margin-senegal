import os
import json
import rasterio
import numpy as np
import pandas as pd
from PIL import Image
import folium
from folium.plugins import MarkerCluster, MiniMap, Fullscreen
import streamlit as st
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import uuid
import atexit
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# Configuration: File Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILES = {
    'raster': os.path.join(BASE_DIR, '201501_Global_Travel_Time_to_Cities_SEN.tiff'),
    'friction': os.path.join(BASE_DIR, '201501_Global_Travel_Speed_Friction_Surface_SEN.tiff'),
    'markets': os.path.join(BASE_DIR, 'markets_from_excel.geojson'),
    'roads': os.path.join(BASE_DIR, 'roads_filtered.geojson'),
    'prices': os.path.join(BASE_DIR, 'merged_farmgate_retail_prices_senegal.xlsx')
}

# Initialize session state
if 'map_data_updated' not in st.session_state:
    st.session_state.map_data_updated = False
if 'latest_farmgate_prices' not in st.session_state:
    st.session_state.latest_farmgate_prices = pd.DataFrame()
if 'latest_retail_prices' not in st.session_state:
    st.session_state.latest_retail_prices = pd.DataFrame()
if 'file_paths' not in st.session_state:
    st.session_state.file_paths = DEFAULT_FILES.copy()
if 'map_render_key' not in st.session_state:
    st.session_state.map_render_key = 0
if 'margin_model' not in st.session_state:
    st.session_state.margin_model = None

# -------------------------------
# Helper Functions
# -------------------------------
def validate_file(file_path, file_type):
    if not os.path.exists(file_path):
        st.error(f"{file_type} file not found: {file_path}")
        return False
    return True

def generate_colors(data, breaks, colors):
    rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    for i in range(len(breaks) - 1):
        mask = (data >= breaks[i]) & (data < breaks[i + 1])
        rgb[mask] = colors[i]
    return rgb

def cleanup_temp_files(*files):
    for file in files:
        if os.path.exists(file):
            os.remove(file)

def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0  # Earth's radius in kilometers
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def generate_margin_data(df):
    df['distance_to_market_km'] = df.apply(
        lambda row: haversine(
            row['Régions - Longitude'], 
            row['Régions - Latitude'], 
            row['longitude'], 
            row['latitude']
        ), axis=1
    )
    df['gross_margin'] = df['price_retail'] - df['price_farmgate']
    df['transaction_cost'] = 0.05 * df['distance_to_market_km']
    df['net_margin'] = df['gross_margin'] - df['transaction_cost']
    return df

@st.cache_data
def train_margin_model(df):
    features = ['price_farmgate', 'price_retail', 'distance_to_market_km', 'friction', 'travel_time', 'commodity_id']
    target = 'net_margin'
    categorical_features = ['commodity_id']
    numerical_features = ['price_farmgate', 'price_retail', 'distance_to_market_km', 'friction', 'travel_time']
    
    df = df.dropna(subset=features + [target])
    if df.empty:
        logger.warning("No valid data for model training after dropping missing values.")
        return None, None
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
    ])
    
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_path = os.path.join(BASE_DIR, 'net_margin_model.pkl')
        joblib.dump(model, model_path)
        return model, r2
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return None, None

# -------------------------------
# Load and Process Rasters
# -----------------------
@st.cache_data(hash_funcs={'numpy.ma.core.MaskedArray': lambda x: hash((x.data.tobytes(), x.mask.tobytes()))})
def load_and_process_raster(file_path, downsample_factor=2):
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1, out_shape=(1, src.height // downsample_factor, src.width // downsample_factor), resampling=rasterio.enums.Resampling.bilinear)
            nodata = src.nodata
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        return np.ma.masked_equal(data, nodata) if nodata else np.ma.masked_invalid(data), bounds
    except rasterio.errors.RasterioIOError as e:
        st.error(f"Failed to load raster {file_path}: {e}")
        return None, None

# -------------------------------
# Generate Raster Images
# -----------------------
@st.cache_data(hash_funcs={'numpy.ma.core.MaskedArray': lambda x: hash((x.data.tobytes(), x.mask.tobytes()))})
def generate_travel_image(data, bounds):
    breaks = [0, 10, 30, 60, 120, 240, 1440, np.inf]
    colors = [(255, 255, 204), (255, 237, 160), (254, 178, 76), (253, 141, 60), (240, 59, 32), (189, 0, 38), (128, 0, 38)]
    rgb = generate_colors(data, breaks, colors)
    travel_png_path = f'travel_time_{str(uuid.uuid4())}.png'
    Image.fromarray(rgb).save(travel_png_path)
    atexit.register(cleanup_temp_files, travel_png_path)
    return travel_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

@st.cache_data(hash_funcs={'numpy.ma.core.MaskedArray': lambda x: hash((x.data.tobytes(), x.mask.tobytes()))})
def generate_friction_image(data, bounds):
    friction_breaks = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, np.inf]
    friction_colors = [(0, 104, 55), (49, 163, 84), (120, 198, 121), (194, 230, 153), (253, 174, 97), (244, 109, 67), (165, 0, 38)]
    rgb = generate_colors(data, friction_breaks, friction_colors)
    friction_png_path = f'friction_surface_{str(uuid.uuid4())}.png'
    Image.fromarray(rgb).save(friction_png_path)
    atexit.register(cleanup_temp_files, friction_png_path)
    return friction_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

# -------------------------------
# Load GeoJSON Files
# -----------------------
@st.cache_data
def load_geojson(file_path, max_features=500, is_roads=False):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('type') != 'FeatureCollection':
            st.warning(f"{file_path} is not a valid FeatureCollection")
            return None
        features = data.get('features', [])
        valid_features = []
        invalid_count = 0
        for feature in features:
            try:
                geom = feature.get('geometry', {})
                if not geom or 'coordinates' not in geom:
                    invalid_count += 1
                    continue
                coords = geom['coordinates']
                if geom['type'] == 'Point':
                    if not (isinstance(coords, list) and len(coords) == 2 and all(isinstance(c, (int, float)) for c in coords)):
                        invalid_count += 1
                        continue
                    if not (-180 <= coords[0] <= 180 and -90 <= coords[1] <= 90):
                        invalid_count += 1
                        continue
                elif geom['type'] in ['LineString', 'MultiLineString']:
                    if geom['type'] == 'LineString' and not all(isinstance(c, list) and len(c) == 2 for c in coords):
                        invalid_count += 1
                        continue
                else:
                    invalid_count += 1
                    continue
                if is_roads and 'properties' in feature:
                    highway = feature['properties'].get('highway', '')
                    if highway and highway not in ['motorway', 'trunk', 'primary', 'secondary']:
                        continue
                valid_features.append(feature)
            except Exception:
                invalid_count += 1
                continue
        if invalid_count > 0:
            st.warning(f"Skipped {invalid_count} invalid features in {file_path}")
        if len(valid_features) > max_features:
            st.warning(f"GeoJSON file {file_path} has {len(valid_features)} valid features. Limiting to {max_features} for performance.")
            valid_features = valid_features[:max_features]
        data['features'] = valid_features
        st.info(f"Loaded {len(valid_features)} features from {file_path}")
        return data if valid_features else None
    except Exception as e:
        st.warning(f"Failed to load {file_path}: {e}")
        return None

# -----------------------
# Load Price Data
# -----------------------
@st.cache_data
def load_price_data(file_path):
    try:
        prices_df = pd.read_excel(file_path, sheet_name='Farmgate prices Senegal')
        if prices_df.empty:
            st.warning("Farmgate prices Excel is empty")
            return pd.DataFrame()
        prices_df['Price'] = pd.to_numeric(prices_df['Price'], errors='coerce')
        prices_df['Year'] = pd.to_numeric(prices_df['Year'], errors='coerce')
        prices_df['Month'] = pd.to_numeric(prices_df['Month'], errors='coerce')
        prices_df = prices_df.dropna(subset=['Price', 'Year', 'Month', 'Régions - Latitude', 'Régions - Longitude', 'commodity_id'])
        if not prices_df.empty and (prices_df['Year'] < 2016).any() or (prices_df['Year'] > 2025).any():
            st.warning("Farmgate data contains years outside 2016–2025. Invalid years will be filtered.")
            prices_df = prices_df[prices_df['Year'].between(2016, 2025)]
        prices_df = prices_df.rename(columns={'Price': 'price_farmgate'})
        return prices_df
    except Exception as e:
        st.error(f"Error reading farmgate prices file {file_path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_retail_data(file_path):
    try:
        retail_df = pd.read_excel(file_path, sheet_name='Retails Price Senegal')
        if retail_df.empty:
            st.warning("Retail prices Excel is empty")
            return pd.DataFrame()
        retail_df = retail_df.rename(columns={'price': 'Price'})
        retail_df['Price'] = pd.to_numeric(retail_df['Price'], errors='coerce')
        retail_df['Year'] = pd.to_numeric(retail_df['Year'], errors='coerce')
        retail_df['Month'] = pd.to_numeric(retail_df['Month'], errors='coerce')
        retail_df = retail_df.dropna(subset=['Price', 'Year', 'Month', 'latitude', 'longitude', 'commodity_id'])
        if not retail_df.empty and (retail_df['Year'] < 2016).any() or (retail_df['Year'] > 2025).any():
            st.warning("Retail data contains years outside 2016–2025. Invalid years will be filtered.")
            retail_df = retail_df[retail_df['Year'].between(2016, 2025)]
        retail_df = retail_df.rename(columns={'Price': 'price_retail'})
        return retail_df
    except Exception as e:
        st.error(f"Error reading retail prices file {file_path}: {e}")
        return pd.DataFrame()

# -----------------------
# Main Dashboard
# -----------------------
def main():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    body { font-family: 'Roboto', sans-serif; }
    .main { background-color: #f9fafb; }
    .sidebar .sidebar-content { background-color: #ffffff; }
    .stButton>button { background-color: #1e3a8a; color: white; border-radius: 8px; }
    .stSelectbox, .stMultiselect { background-color: #f3f4f6; border-radius: 8px; }
    .header { background-color: #1e3a8a; color: white; padding: 20px; border-radius: 8px; }
    .footer { background-color: #1e3a8a; color: white; padding: 10px; text-align: center; margin-top: 20px; }
    .stApp [data-testid="stMapContainer"] { 
        margin-top: 10px; 
        width: 100% !important; 
        min-height: 400px; 
        max-height: 100vh; 
        overflow: auto; 
    }
    .legend-container { background-color: white; border: 2px solid grey; padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); margin-top: 10px; }
    .legend-title { font-weight: bold; font-size: 14px; margin-bottom: 10px; }
    .legend-item { display: flex; align-items: center; margin-bottom: 5px; font-size: 14px; }
    .legend-color { width: 20px; height: 20px; margin-right: 8px; display: inline-block; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="header">
        <img src="https://www.ifpri.org/themes/custom/ifpri/logo.svg" alt="IFPRI Logo" width="150">
        <h1>Senegal Agricultural Market Dashboard</h1>
        <p>Developed in collaboration with the International Food Policy Research Institute (IFPRI)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Senegal Agricultural Market Dashboard
    This interactive tool visualizes travel time, friction surfaces, market locations, road networks, commodity prices, and margin predictions across Senegal.
    """)

    st.sidebar.header("Data Sources")
    uploaded_files = {}
    for key, default_path in DEFAULT_FILES.items():
        uploaded_file = st.sidebar.file_uploader(f"Upload {key.capitalize()} File", type=['tiff', 'geojson', 'xlsx'] if key == 'prices' else ['tiff'] if key in ['raster', 'friction'] else ['geojson'])
        if uploaded_file:
            uploaded_path = os.path.join(BASE_DIR, uploaded_file.name)
            with open(uploaded_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            uploaded_files[key] = uploaded_path
        else:
            uploaded_files[key] = default_path

    st.session_state.file_paths.update(uploaded_files)

    for key, path in st.session_state.file_paths.items():
        if not validate_file(path, key.capitalize()):
            return

    with st.spinner("Loading data..."):
        progress = st.progress(0)
        travel_time, travel_bounds = load_and_process_raster(st.session_state.file_paths['raster'])
        progress.progress(0.2)
        friction_data, friction_bounds = load_and_process_raster(st.session_state.file_paths['friction'])
        progress.progress(0.4)
        markets = load_geojson(st.session_state.file_paths['markets'])
        progress.progress(0.6)
        roads_filtered = load_geojson(st.session_state.file_paths['roads'], max_features=500, is_roads=True)
        progress.progress(0.8)
        prices_df = load_price_data(st.session_state.file_paths['prices'])
        retail_df = load_retail_data(st.session_state.file_paths['prices'])
        progress.progress(1.0)

    if travel_time is None or friction_data is None:
        st.error("Failed to load raster data. Please check the files and try again.")
        return

    travel_png_path, travel_image_bounds = generate_travel_image(travel_time, travel_bounds)
    friction_png_path, friction_image_bounds = generate_friction_image(friction_data, friction_bounds)

    farmgate_required_columns = ['Régions Name', 'Commodity', 'commodity_english', 'commodity_id', 'price_farmgate', 'Unit2', 'Régions - Latitude', 'Régions - Longitude', 'Year', 'Month']
    retail_required_columns = ['market', 'commodity', 'commodity_id', 'price_retail', 'Unit2', 'latitude', 'longitude', 'Year', 'Month']
    if not prices_df.empty and any(col not in prices_df.columns for col in farmgate_required_columns):
        st.error(f"Missing required columns in farmgate prices: {', '.join([col for col in farmgate_required_columns if col not in prices_df.columns])}")
        prices_df = pd.DataFrame()
    if not retail_df.empty and any(col not in retail_df.columns for col in retail_required_columns):
        st.error(f"Missing required columns in retail prices: {', '.join([col for col in retail_required_columns if col not in retail_df.columns])}")
        retail_df = pd.DataFrame()

    # Merge farmgate and retail data for margin calculations
    if not prices_df.empty and not retail_df.empty:
        merged_df = pd.merge(
            prices_df[['Régions Name', 'commodity_id', 'price_farmgate', 'Régions - Latitude', 'Régions - Longitude', 'Year', 'Month']],
            retail_df[['market', 'commodity_id', 'price_retail', 'latitude', 'longitude', 'Year', 'Month']],
            on=['commodity_id', 'Year', 'Month'],
            how='inner'
        )
        merged_df = generate_margin_data(merged_df)
        
        # Extract raster values
        with rasterio.open(st.session_state.file_paths['friction']) as friction_raster:
            with rasterio.open(st.session_state.file_paths['raster']) as travel_raster:
                coords = [(x, y) for x, y in zip(merged_df['longitude'], merged_df['latitude'])]
                merged_df['friction'] = [val[0] for val in friction_raster.sample(coords)]
                merged_df['travel_time'] = [val[0] for val in travel_raster.sample(coords)]
        
        friction_nodata = friction_raster.nodata if friction_raster.nodata is not None else np.nan
        travel_nodata = travel_raster.nodata if travel_raster.nodata is not None else np.nan
        merged_df['friction'] = np.where(np.isclose(merged_df['friction'], friction_nodata, equal_nan=True), merged_df['friction'].mean(), merged_df['friction'])
        merged_df['travel_time'] = np.where(np.isclose(merged_df['travel_time'], travel_nodata, equal_nan=True), merged_df['travel_time'].mean(), merged_df['travel_time'])
        merged_df['friction'] = merged_df['friction'].fillna(merged_df['friction'].mean())
        merged_df['travel_time'] = merged_df['travel_time'].fillna(merged_df['travel_time'].mean())
        merged_df['distance_to_market_km'] = merged_df['distance_to_market_km'].fillna(merged_df['distance_to_market_km'].mean())
    else:
        merged_df = pd.DataFrame()

    # Train margin model if not already trained
    if st.session_state.margin_model is None and not merged_df.empty:
        model, r2_score = train_margin_model(merged_df)
        st.session_state.margin_model = model
        st.session_state.margin_r2 = r2_score

    tab1, tab2, tab3 = st.tabs(["Interactive Map", "Data Summary", "Price Trends"])

    with tab1:
        st.subheader("Interactive Map")
        st.markdown("Explore travel time, friction surfaces, market locations, road networks, and commodity prices.")
        st.info("Map view is fixed. Pan or zoom to explore all data. Roads layer may load slowly due to data complexity.")

        st.sidebar.header("Map Filters")
        available_years = list(range(2016, 2026))
        selected_year = st.sidebar.selectbox("Select Year", available_years, index=len(available_years)-1, key="year_select", on_change=lambda: st.session_state.update({'map_data_updated': True}))

        month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                       7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
        available_months = sorted(list(set(prices_df['Month'].unique()) | set(retail_df['Month'].unique()))) if not prices_df.empty or not retail_df.empty else list(range(1, 13))
        latest_month = max(available_months) if available_months else 12
        selected_month_name = st.sidebar.selectbox("Select Month", [month_names.get(m, str(m)) for m in available_months], index=available_months.index(latest_month) if latest_month in available_months else 0, key="month_select", on_change=lambda: st.session_state.update({'map_data_updated': True}))
        selected_month = next((k for k, v in month_names.items() if v == selected_month_name), selected_month_name)

        commodity_options = []
        commodity_id_to_name = {}
        if not prices_df.empty and not retail_df.empty:
            farmgate_commodities = set(prices_df['commodity_id'].unique())
            retail_commodities = set(retail_df['commodity_id'].unique())
            common_commodity_ids = farmgate_commodities.intersection(retail_commodities)
            for cid in common_commodity_ids:
                name = prices_df[prices_df['commodity_id'] == cid]['commodity_english'].iloc[0] if cid in prices_df['commodity_id'].values else retail_df[retail_df['commodity_id'] == cid]['commodity'].iloc[0]
                commodity_options.append(cid)
                commodity_id_to_name[cid] = name
            commodity_options = sorted(commodity_options, key=lambda x: commodity_id_to_name[x])
        if not commodity_options:
            st.error("No commodities found in both farmgate and retail datasets with common commodity IDs. Please check your data.")
            return
        selected_commodity_ids = st.sidebar.multiselect(
            "Select Commodities",
            options=commodity_options,
            format_func=lambda x: commodity_id_to_name.get(x, str(x)),
            default=commodity_options,
            key="commodity_select",
            on_change=lambda: st.session_state.update({'map_data_updated': True})
        )

        st.sidebar.header("Map Layers")
        show_travel = st.sidebar.checkbox("Travel Time", value=False)
        show_friction = st.sidebar.checkbox("Friction Surface", value=False)
        show_roads = st.sidebar.checkbox("Roads", value=False)
        show_markets = st.sidebar.checkbox("Markets", value=True)
        show_farmgate = st.sidebar.checkbox("Farmgate Prices", value=False)
        show_retail = st.sidebar.checkbox("Retail Prices", value=False)

        map_height = st.sidebar.slider("Map Height (px)", 400, 1000, 800, key="map_height")

        if st.session_state.map_data_updated or st.session_state.latest_farmgate_prices.empty or st.session_state.latest_retail_prices.empty:
            latest_farmgate_prices = pd.DataFrame()
            latest_retail_prices = pd.DataFrame()
            if not prices_df.empty:
                filtered_farmgate = prices_df[prices_df['Year'] == selected_year] if selected_year else prices_df
                if selected_month:
                    filtered_farmgate = filtered_farmgate[filtered_farmgate['Month'] == selected_month]
                filtered_farmgate['Date'] = pd.to_datetime(filtered_farmgate[['Year', 'Month']].assign(day=1), errors='coerce')
                filtered_farmgate = filtered_farmgate.dropna(subset=['Date'])
                latest_farmgate_prices = filtered_farmgate.sort_values('Date').groupby(['Régions Name', 'commodity_id']).last().reset_index()
                if selected_commodity_ids:
                    latest_farmgate_prices = latest_farmgate_prices[latest_farmgate_prices['commodity_id'].isin(selected_commodity_ids)]
                if len(latest_farmgate_prices) > 500:
                    latest_farmgate_prices = latest_farmgate_prices.head(500)
                    st.warning("Limited to 500 farmgate price markers for performance.")
            if not retail_df.empty:
                filtered_retail = retail_df[retail_df['Year'] == selected_year] if selected_year else retail_df
                if selected_month:
                    filtered_retail = filtered_retail[filtered_retail['Month'] == selected_month]
                filtered_retail['Date'] = pd.to_datetime(filtered_retail[['Year', 'Month']].assign(day=1), errors='coerce')
                filtered_retail = filtered_retail.dropna(subset=['Date'])
                latest_retail_prices = filtered_retail.sort_values('Date').groupby(['market', 'commodity_id']).last().reset_index()
                if latest_retail_prices.index.duplicated().any():
                    st.warning("Duplicate indices found in retail prices. Removing duplicates.")
                    latest_retail_prices = latest_retail_prices.drop_duplicates().reset_index(drop=True)
                if selected_commodity_ids:
                    latest_retail_prices = latest_retail_prices[latest_retail_prices['commodity_id'].isin(selected_commodity_ids)]
                if len(latest_retail_prices) > 500:
                    latest_retail_prices = latest_retail_prices.head(500)
                    st.warning("Limited to 500 retail price markers for performance.")
            st.session_state.latest_farmgate_prices = latest_farmgate_prices
            st.session_state.latest_retail_prices = latest_retail_prices
            st.session_state.map_data_updated = False
            st.session_state.map_render_key += 1

        map_placeholder = st.empty()
        with map_placeholder.container():
            with st.spinner("Rendering map..."):
                try:
                    m = folium.Map(
                        location=[14.5, -14.5],
                        zoom_start=6,
                        tiles="CartoDB Positron"
                    )
                    folium.FitBounds([[12.3, -17], [16.7, -11]]).add_to(m)

                    if show_roads and roads_filtered:
                        folium.GeoJson(
                            roads_filtered,
                            name="Roads",
                            style_function=lambda x: {'color': '#3b82f6', 'weight': 1, 'opacity': 0.7}
                        ).add_to(m)

                    if show_markets and markets:
                        market_group = folium.FeatureGroup(name="Markets", show=True)
                        valid_markets = 0
                        for feature in markets.get('features', []):
                            if feature['geometry']['type'] == 'Point' and all(isinstance(c, (int, float)) for c in feature['geometry']['coordinates']):
                                coords = feature['geometry']['coordinates'][::-1]
                                popup = feature['properties'].get('market', 'Unknown Market')
                                folium.Marker(
                                    location=coords,
                                    popup=popup,
                                    icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')
                                ).add_to(market_group)
                                valid_markets += 1
                        if valid_markets == 0:
                            st.warning("No valid market locations found.")
                        else:
                            market_group.add_to(m)

                    if show_farmgate and not st.session_state.latest_farmgate_prices.empty:
                        farmgate_cluster = MarkerCluster(name="Farmgate Prices").add_to(m)
                        valid_farmgate_markers = 0
                        for _, row in st.session_state.latest_farmgate_prices.iterrows():
                            if pd.isna(row['Régions - Latitude']) or pd.isna(row['Régions - Longitude']):
                                continue
                            popup_text = f"<b>Region:</b> {row['Régions Name']}<br><b>Commodity:</b> {row['commodity_english']}<br><b>Price:</b> {row['price_farmgate']:.2f} {row['Unit2']}<br><b>Date:</b> {row['Year']}-{row['Month']:02d}"
                            folium.Marker(
                                location=[row['Régions - Latitude'], row['Régions - Longitude']],
                                popup=folium.Popup(popup_text, max_width=250),
                                icon=folium.Icon(color='green', icon='tractor', prefix='fa')
                            ).add_to(farmgate_cluster)
                            valid_farmgate_markers += 1
                        if valid_farmgate_markers == 0:
                            st.warning("No valid farmgate price locations found for the selected filters.")

                    if show_retail and not st.session_state.latest_retail_prices.empty:
                        retail_cluster = MarkerCluster(name="Retail Prices").add_to(m)
                        valid_retail_markers = 0
                        for _, row in st.session_state.latest_retail_prices.iterrows():
                            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                                continue
                            popup_text = f"<b>Market:</b> {row['market']}<br><b>Commodity:</b> {row['commodity']}<br><b>Price:</b> {row['price_retail']:.2f} {row['Unit2']}<br><b>Date:</b> {row['Year']}-{row['Month']:02d}"
                            folium.Marker(
                                location=[row['latitude'], row['longitude']],
                                popup=folium.Popup(popup_text, max_width=250),
                                icon=folium.Icon(color='purple', icon='shopping-basket', prefix='fa')
                            ).add_to(retail_cluster)
                            valid_retail_markers += 1
                        if valid_retail_markers == 0:
                            st.warning("No valid retail price locations found for the selected filters.")

                    if show_travel:
                        folium.raster_layers.ImageOverlay(
                            name="Travel Time",
                            image=travel_png_path,
                            bounds=travel_image_bounds,
                            opacity=0.6,
                            interactive=True,
                            cross_origin=False
                        ).add_to(m)
                    if show_friction:
                        folium.raster_layers.ImageOverlay(
                            name="Friction Surface (min/m)",
                            image=friction_png_path,
                            bounds=friction_image_bounds,
                            opacity=0.7,
                            interactive=True,
                            cross_origin=False
                        ).add_to(m)

                    MiniMap(tiles='OpenStreetMap', position='bottomleft', width=150, height=150).add_to(m)
                    Fullscreen(position='topright', title='Expand', title_cancel='Exit').add_to(m)

                    folium.LayerControl(collapsed=False).add_to(m)
                    st_folium(m, use_container_width=True, height=map_height, key=f"folium_map_{st.session_state.map_render_key}")

                    if show_travel or show_friction:
                        col1, col2 = st.columns(2)
                        if show_travel:
                            with col1:
                                st.markdown("""
                                <div class="legend-container">
                                    <div class="legend-title">Travel Time (min)</div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#ffffcc;"></span> 0–10
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#ffeda0;"></span> 10–30
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#feb24c;"></span> 30–60
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#fd8d3c;"></span> 60–120
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#f03b20;"></span> 120–240
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#bd0026;"></span> 240–1440
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#800026;"></span> >1440
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        if show_friction:
                            with col2:
                                st.markdown("""
                                <div class="legend-container">
                                    <div class="legend-title">Friction (min/m)</div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#006837;"></span> ≤ 0.001
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#31a354;"></span> ≤ 0.01
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#78c679;"></span> ≤ 0.1
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#c2e699;"></span> ≤ 0.5
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#fdae61;"></span> ≤ 1
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#f46d43;"></span> ≤ 2.0
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#a50026;"></span> ≤ 5.0
                                    </div>
                                    <div class="legend-item">
                                        <span class="legend-color" style="background:#800026;"></span> > 5.0
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Map rendering failed: {str(e)}. Please check data files or coordinates.")
                finally:
                    st.spinner(False)

    with tab2:
        st.subheader("Data Summary")
        st.markdown("### Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Markets", len(markets['features']) if markets else 0)
        col2.metric("Road Features", len(roads_filtered['features']) if roads_filtered else 0)
        col3.metric("Price Points", len(st.session_state.latest_farmgate_prices) + len(st.session_state.latest_retail_prices))

    with tab3:
        st.subheader("Price Trends")
        if not prices_df.empty and not retail_df.empty and commodity_options:
            st.markdown("### Farmgate, Retail, Gross Margin, and Net Margin Trends")
            selected_commodity_id = st.selectbox(
                "Select Commodity",
                options=commodity_options,
                format_func=lambda x: commodity_id_to_name.get(x, str(x)),
                key="trend_commodity_select"
            )
            show_gross_margin = st.checkbox("Show Gross Margin (Retail - Farmgate)", value=True, key="show_gross_margin")
            show_net_margin = st.checkbox("Show Net Margin (Gross Margin - Transaction Cost)", value=True, key="show_net_margin")

            @st.cache_data
            def compute_national_trends(_prices_df, _retail_df, _merged_df, commodity_id, show_gross, show_net):
                try:
                    farmgate_trend = _prices_df[_prices_df['commodity_id'] == commodity_id][['Year', 'Month', 'price_farmgate']].groupby(['Year', 'Month']).mean().reset_index()
                    if not farmgate_trend.empty:
                        farmgate_trend['Date'] = pd.to_datetime(farmgate_trend[['Year', 'Month']].assign(day=1), errors='coerce')
                        farmgate_trend = farmgate_trend.dropna(subset=['Date']).reset_index(drop=True)
                        farmgate_trend['Price Type'] = 'Farmgate'
                    else:
                        farmgate_trend = pd.DataFrame(columns=['Date', 'Price', 'Price Type'])

                    retail_trend = _retail_df[_retail_df['commodity_id'] == commodity_id][['Year', 'Month', 'price_retail']].groupby(['Year', 'Month']).mean().reset_index()
                    if not retail_trend.empty:
                        retail_trend['Date'] = pd.to_datetime(retail_trend[['Year', 'Month']].assign(day=1), errors='coerce')
                        retail_trend = retail_trend.dropna(subset=['Date']).reset_index(drop=True)
                        retail_trend['Price Type'] = 'Retail'
                    else:
                        retail_trend = pd.DataFrame(columns=['Date', 'Price', 'Price Type'])

                    gross_margin_trend = pd.DataFrame()
                    net_margin_trend = pd.DataFrame()
                    if show_gross and not farmgate_trend.empty and not retail_trend.empty:
                        margin_trend = pd.merge(
                            farmgate_trend[['Year', 'Month', 'price_farmgate', 'Date']],
                            retail_trend[['Year', 'Month', 'price_retail', 'Date']],
                            on=['Year', 'Month', 'Date'],
                            how='inner',
                            suffixes=('_farmgate', '_retail')
                        )
                        margin_trend['Price'] = margin_trend['price_retail'] - margin_trend['price_farmgate']
                        margin_trend['Price Type'] = 'Gross Margin'
                        gross_margin_trend = margin_trend[['Date', 'Price', 'Price Type']].dropna(subset=['Price'])

                    if show_net and not _merged_df.empty:
                        net_trend = _merged_df[_merged_df['commodity_id'] == commodity_id][['Year', 'Month', 'net_margin']].groupby(['Year', 'Month']).mean().reset_index()
                        if not net_trend.empty:
                            net_trend['Date'] = pd.to_datetime(net_trend[['Year', 'Month']].assign(day=1), errors='coerce')
                            net_trend = net_trend.dropna(subset=['Date']).reset_index(drop=True)
                            net_trend['Price'] = net_trend['net_margin']
                            net_trend['Price Type'] = 'Net Margin'
                            net_margin_trend = net_trend[['Date', 'Price', 'Price Type']]

                    combined_trend = pd.concat([farmgate_trend.rename(columns={'price_farmgate': 'Price'}),
                                              retail_trend.rename(columns={'price_retail': 'Price'}),
                                              gross_margin_trend,
                                              net_margin_trend], ignore_index=True)
                    return combined_trend
                except Exception as e:
                    st.error(f"Failed to compute national price trends: {e}")
                    return pd.DataFrame(columns=['Date', 'Price', 'Price Type'])

            try:
                combined_trend = compute_national_trends(prices_df, retail_df, merged_df, selected_commodity_id, show_gross_margin, show_net_margin)
                if combined_trend.empty:
                    st.warning(f"No trend data available for {commodity_id_to_name.get(selected_commodity_id, selected_commodity_id)}.")
                else:
                    fig = go.Figure()
                    if 'Farmgate' in combined_trend['Price Type'].values:
                        fig.add_trace(go.Scatter(
                            x=combined_trend[combined_trend['Price Type'] == 'Farmgate']['Date'],
                            y=combined_trend[combined_trend['Price Type'] == 'Farmgate']['Price'],
                            mode='lines+markers',
                            name='Farmgate',
                            line=dict(color='#2ca02c', width=2),
                            marker=dict(size=6),
                            hovertemplate='%{x|%b %Y}<br>Price: %{y:.2f}<br>Type: Farmgate'
                        ))
                    if 'Retail' in combined_trend['Price Type'].values:
                        fig.add_trace(go.Scatter(
                            x=combined_trend[combined_trend['Price Type'] == 'Retail']['Date'],
                            y=combined_trend[combined_trend['Price Type'] == 'Retail']['Price'],
                            mode='lines+markers',
                            name='Retail',
                            line=dict(color='#9467bd', width=2),
                            marker=dict(size=6),
                            hovertemplate='%{x|%b %Y}<br>Price: %{y:.2f}<br>Type: Retail'
                        ))
                    if show_gross_margin and 'Gross Margin' in combined_trend['Price Type'].values:
                        fig.add_trace(go.Scatter(
                            x=combined_trend[combined_trend['Price Type'] == 'Gross Margin']['Date'],
                            y=combined_trend[combined_trend['Price Type'] == 'Gross Margin']['Price'],
                            mode='lines+markers',
                            name='Gross Margin',
                            line=dict(color='#d62728', width=2, dash='dash'),
                            marker=dict(size=6),
                            hovertemplate='%{x|%b %Y}<br>Margin: %{y:.2f}<br>Type: Gross Margin'
                        ))
                    if show_net_margin and 'Net Margin' in combined_trend['Price Type'].values:
                        fig.add_trace(go.Scatter(
                            x=combined_trend[combined_trend['Price Type'] == 'Net Margin']['Date'],
                            y=combined_trend[combined_trend['Price Type'] == 'Net Margin']['Price'],
                            mode='lines+markers',
                            name='Net Margin',
                            line=dict(color='#ff7f0e', width=2, dash='dot'),
                            marker=dict(size=6),
                            hovertemplate='%{x|%b %Y}<br>Margin: %{y:.2f}<br>Type: Net Margin'
                        ))

                    fig.update_layout(
                        title=f"{commodity_id_to_name.get(selected_commodity_id, selected_commodity_id)} Price and Margin Trends",
                        xaxis_title="Date",
                        yaxis_title="Average Price/Margin (Unit2)",
                        font=dict(family="Roboto, sans-serif", size=12),
                        hovermode="x unified",
                        showlegend=True,
                        template="plotly_white",
                        xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate national price trends: {e}")

            st.markdown("### Regional Farmgate, Retail, Gross Margin, and Net Margin Trends")
            @st.cache_data
            def compute_regional_trends(_prices_df, _retail_df, _merged_df, commodity_id, show_gross, show_net):
                try:
                    required_farmgate_cols = ['Régions Name', 'Year', 'Month', 'price_farmgate', 'commodity_id']
                    required_retail_cols = ['market', 'Year', 'Month', 'price_retail', 'commodity_id']
                    if not all(col in _prices_df.columns for col in required_farmgate_cols):
                        st.warning(f"Missing farmgate columns: {', '.join([col for col in required_farmgate_cols if col not in _prices_df.columns])}")
                        return pd.DataFrame(columns=['Region', 'Date', 'Price', 'Price Type'])
                    if not all(col in _retail_df.columns for col in required_retail_cols):
                        st.warning(f"Missing retail columns: {', '.join([col for col in required_retail_cols if col not in _retail_df.columns])}")
                        return pd.DataFrame(columns=['Region', 'Date', 'Price', 'Price Type'])

                    farmgate_regional_trend = _prices_df[_prices_df['commodity_id'] == commodity_id][['Régions Name', 'Year', 'Month', 'price_farmgate']].groupby(['Régions Name', 'Year', 'Month']).mean().reset_index()
                    if not farmgate_regional_trend.empty:
                        farmgate_regional_trend['Date'] = pd.to_datetime(farmgate_regional_trend[['Year', 'Month']].assign(day=1), errors='coerce')
                        farmgate_regional_trend = farmgate_regional_trend.dropna(subset=['Date']).reset_index(drop=True)
                        farmgate_regional_trend['Price Type'] = 'Farmgate'
                        farmgate_regional_trend = farmgate_regional_trend.rename(columns={'Régions Name': 'Region', 'price_farmgate': 'Price'})
                    else:
                        farmgate_regional_trend = pd.DataFrame(columns=['Region', 'Date', 'Price', 'Price Type'])

                    retail_regional_trend = _retail_df[_retail_df['commodity_id'] == commodity_id][['market', 'Year', 'Month', 'price_retail']].groupby(['market', 'Year', 'Month']).mean().reset_index()
                    retail_regional_trend = retail_regional_trend.rename(columns={'market': 'Region', 'price_retail': 'Price'})
                    if not retail_regional_trend.empty:
                        retail_regional_trend['Date'] = pd.to_datetime(retail_regional_trend[['Year', 'Month']].assign(day=1), errors='coerce')
                        retail_regional_trend = retail_regional_trend.dropna(subset=['Date']).reset_index(drop=True)
                        retail_regional_trend['Price Type'] = 'Retail'
                    else:
                        retail_regional_trend = pd.DataFrame(columns=['Region', 'Date', 'Price', 'Price Type'])

                    gross_margin_regional_trend = pd.DataFrame()
                    net_margin_regional_trend = pd.DataFrame()
                    if show_gross and not farmgate_regional_trend.empty and not retail_regional_trend.empty:
                        farmgate_regions = set(farmgate_regional_trend['Region'])
                        retail_regions = set(retail_regional_trend['Region'])
                        common_regions = farmgate_regions.intersection(retail_regions)
                        if not common_regions:
                            st.warning("No common regions between farmgate and retail data for gross margin calculation.")
                        else:
                            margin_regional_trend = pd.merge(
                                farmgate_regional_trend[['Region', 'Year', 'Month', 'Price', 'Date']],
                                retail_regional_trend[['Region', 'Year', 'Month', 'Price', 'Date']],
                                on=['Region', 'Year', 'Month', 'Date'],
                                how='inner',
                                suffixes=('_farmgate', '_retail')
                            )
                            margin_regional_trend['Price'] = margin_regional_trend['Price_retail'] - margin_regional_trend['Price_farmgate']
                            margin_regional_trend['Price Type'] = 'Gross Margin'
                            gross_margin_regional_trend = margin_regional_trend[['Region', 'Date', 'Price', 'Price Type']].dropna(subset=['Price'])

                    if show_net and not _merged_df.empty:
                        net_trend = _merged_df[_merged_df['commodity_id'] == commodity_id][['Régions Name', 'Year', 'Month', 'net_margin']].groupby(['Régions Name', 'Year', 'Month']).mean().reset_index()
                        if not net_trend.empty:
                            net_trend['Date'] = pd.to_datetime(net_trend[['Year', 'Month']].assign(day=1), errors='coerce')
                            net_trend = net_trend.dropna(subset=['Date']).reset_index(drop=True)
                            net_trend['Price'] = net_trend['net_margin']
                            net_trend['Price Type'] = 'Net Margin'
                            net_trend = net_trend.rename(columns={'Régions Name': 'Region'})
                            net_margin_regional_trend = net_trend[['Region', 'Date', 'Price', 'Price Type']]

                    combined_regional_trend = pd.concat([farmgate_regional_trend,
                                                        retail_regional_trend,
                                                        gross_margin_regional_trend,
                                                        net_margin_regional_trend], ignore_index=True)
                    return combined_regional_trend
                except Exception as e:
                    st.error(f"Failed to compute regional price trends: {e}")
                    return pd.DataFrame(columns=['Region', 'Date', 'Price', 'Price Type'])

            try:
                combined_regional_trend = compute_regional_trends(prices_df, retail_df, merged_df, selected_commodity_id, show_gross_margin, show_net_margin)
                if combined_regional_trend.empty:
                    st.warning(f"No regional trend data available for {commodity_id_to_name.get(selected_commodity_id, selected_commodity_id)}.")
                else:
                    fig_regional = go.Figure()
                    regions = combined_regional_trend['Region'].unique()
                    for region in regions:
                        farmgate_data = combined_regional_trend[(combined_regional_trend['Region'] == region) & (combined_regional_trend['Price Type'] == 'Farmgate')]
                        if not farmgate_data.empty:
                            fig_regional.add_trace(go.Scatter(
                                x=farmgate_data['Date'],
                                y=farmgate_data['Price'],
                                mode='lines+markers',
                                name=f'{region} Farmgate',
                                line=dict(color='#2ca02c', width=1.5),
                                marker=dict(size=4),
                                hovertemplate=f'%{x|%b %Y}<br>Price: %{y:.2f}<br>Region: {region}<br>Type: Farmgate'
                            ))
                        retail_data = combined_regional_trend[(combined_regional_trend['Region'] == region) & (combined_regional_trend['Price Type'] == 'Retail')]
                        if not retail_data.empty:
                            fig_regional.add_trace(go.Scatter(
                                x=retail_data['Date'],
                                y=retail_data['Price'],
                                mode='lines+markers',
                                name=f'{region} Retail',
                                line=dict(color='#9467bd', width=1.5),
                                marker=dict(size=4),
                                hovertemplate=f'%{x|%b %Y}<br>Price: %{y:.2f}<br>Region: {region}<br>Type: Retail'
                            ))
                        if show_gross_margin:
                            gross_data = combined_regional_trend[(combined_regional_trend['Region'] == region) & (combined_regional_trend['Price Type'] == 'Gross Margin')]
                            if not gross_data.empty:
                                fig_regional.add_trace(go.Scatter(
                                    x=gross_data['Date'],
                                    y=gross_data['Price'],
                                    mode='lines+markers',
                                    name=f'{region} Gross Margin',
                                    line=dict(color='#d62728', width=1.5, dash='dash'),
                                    marker=dict(size=4),
                                    hovertemplate=f'%{x|%b %Y}<br>Margin: %{y:.2f}<br>Region: {region}<br>Type: Gross Margin'
                                ))
                        if show_net_margin:
                            net_data = combined_regional_trend[(combined_regional_trend['Region'] == region) & (combined_regional_trend['Price Type'] == 'Net Margin')]
                            if not net_data.empty:
                                fig_regional.add_trace(go.Scatter(
                                    x=net_data['Date'],
                                    y=net_data['Price'],
                                    mode='lines+markers',
                                    name=f'{region} Net Margin',
                                    line=dict(color='#ff7f0e', width=1.5, dash='dot'),
                                    marker=dict(size=4),
                                    hovertemplate=f'%{x|%b %Y}<br>Margin: %{y:.2f}<br>Region: {region}<br>Type: Net Margin'
                                ))

                    fig_regional.update_layout(
                        title=f"{commodity_id_to_name.get(selected_commodity_id, selected_commodity_id)} Regional Price and Margin Trends",
                        xaxis_title="Date",
                        yaxis_title="Average Price/Margin (Unit2)",
                        font=dict(family="Roboto", sans-serif), size=12),
                        hovermode="x unified",
                        showlegend=True,
                        template="plotly_white",
                        xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
                    )
                    st.plotly_chart(fig_regional, use_container_width=True))

                # Margin Prediction Model Results
                st.markdown("### Net Margin Prediction Model")
                if st.session_state.margin_model:
                    st.write(f"Model trained successfully with R² Score: {st.session_state.margin_r2:.2f}")
                    if not merged_df.empty and st.session_state['margin_model']:
                        sample_df = merged_df[merged_df['commodity_id'] == selected_commodity_id].sample(n=min(5, len(merged_df)), random_state=42))
                        if not sample_df.empty:
                            predictions = st.session_state.margin_model.predict(sample_df[['price_farmgate', 'price_retail', 'distance_to_market_km', 'friction', 'travel_time', 'commodity_id']])
                            st.markdown("#### Sample Predictions")
                            for i, (idx, row) in enumerate(sample_df.iterrows()):
                                st.write(f"Region: {row['Régions Name']}, Market: {row['market']}, Actual Net Margin: {row['net_margin']:.2f}, Predicted: {predictions[i]:.2f}")
                        else:
                            st.warning("No data available for predictions for the selected commodity.")
                else:
                    st.warning("Margin prediction model could not be trained due to missing or insufficient data.")

            except Exception as e:
                st.error(f"Failed to generate regional price trends: {e}")

        else:
            st.warning("No data available for price trends. Please check your data.")

    st.markdown("""
    <div class="footer">
        <p>Developed by xAI in collaboration with IFPRI | Data Sources: IFPRI, OpenStreetMap | © 2023
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()