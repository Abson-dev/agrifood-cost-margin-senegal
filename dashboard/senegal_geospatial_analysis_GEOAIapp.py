import os
import json
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
import folium
from folium.plugins import MarkerCluster, MiniMap, Fullscreen, FastMarkerCluster
import streamlit as st
from streamlit_folium import st_folium
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import atexit
from scipy.interpolate import interp1d
from shapely.geometry import Point
from rasterio.mask import mask
# New imports for geospatial AI
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import rasterio.warp
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# -------------------------------
# Configuration (unchanged)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILES = {
    'raster': os.path.join(BASE_DIR, '201501_Global_Travel_Time_to_Cities_SEN.tiff'),
    'friction': os.path.join(BASE_DIR, '201501_Global_Travel_Speed_Friction_Surface_SEN.tiff'),
    'markets': os.path.join(BASE_DIR, 'markets_from_excel.geojson'),
    'roads': os.path.join(BASE_DIR, 'roads_filtered.geojson'),
    'prices': os.path.join(BASE_DIR, 'merged_farmgate_retail_prices_senegal.xlsx'),
    'population_2016': os.path.join(BASE_DIR, 'sen_ppp_2016_UNadj.tif'),
    'population_2017': os.path.join(BASE_DIR, 'sen_ppp_2017_UNadj.tif'),
    'population_2018': os.path.join(BASE_DIR, 'sen_ppp_2018_UNadj.tif'),
    'population_2019': os.path.join(BASE_DIR, 'sen_ppp_2019_UNadj.tif'),
    'population_2020': os.path.join(BASE_DIR, 'sen_ppp_2020_UNadj.tif')
}

# Initialize session state (unchanged)
if 'map_data_updated' not in st.session_state:
    st.session_state['map_data_updated'] = False
if 'latest_farmgate_prices' not in st.session_state:
    st.session_state['latest_farmgate_prices'] = pd.DataFrame()
if 'latest_retail_prices' not in st.session_state:
    st.session_state['latest_retail_prices'] = pd.DataFrame()
if 'latest_merged_prices' not in st.session_state:
    st.session_state['latest_merged_prices'] = pd.DataFrame()
if 'file_paths' not in st.session_state:
    st.session_state['file_paths'] = DEFAULT_FILES.copy()
if 'map_render_key' not in st.session_state:
    st.session_state['map_render_key'] = 0
if 'commodity_map' not in st.session_state:
    st.session_state['commodity_map'] = {}
if 'map_height' not in st.session_state:
    st.session_state['map_height'] = 800
if 'errors' not in st.session_state:
    st.session_state['errors'] = []
if 'temp_files' not in st.session_state:
    st.session_state['temp_files'] = []
if 'show_legends' not in st.session_state:
    st.session_state['show_legends'] = True
# New session state for AI models
if 'cnn_model' not in st.session_state:
    st.session_state['cnn_model'] = None
if 'rf_model' not in st.session_state:
    st.session_state['rf_model'] = None
if 'market_clusters' not in st.session_state:
    st.session_state['market_clusters'] = None

# -------------------------------
# New Geospatial AI Functions
# -------------------------------
def preprocess_raster_for_cnn(raster_data, patch_size=64):
    """Preprocess raster data into patches for CNN input."""
    if raster_data is None:
        return None
    height, width = raster_data.shape
    patches = []
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = raster_data[i:i+patch_size, j:j+patch_size]
            if not np.ma.is_masked(patch) and np.all(np.isfinite(patch)):
                patches.append(patch)
    return np.array(patches)[..., np.newaxis]  # Add channel dimension

def build_cnn_model(input_shape=(64, 64, 1)):
    """Build a simple CNN model for accessibility classification."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 classes: low, medium, high accessibility
    ])
    modelsonian
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_model(travel_data, friction_data, patch_size=64):
    """Train CNN to classify accessibility based on travel time and friction."""
    travel_patches = preprocess_raster_for_cnn(travel_data, patch_size)
    friction_patches = preprocess_raster_for_cnn(friction_data, patch_size)
    
    if travel_patches is None or friction_patches is None or len(travel_patches) == 0:
        st.warning("Insufficient valid raster data for CNN training.")
        return None
    
    # Generate synthetic labels (example: based on travel time thresholds)
    labels = []
    for patch in travel_patches:
        mean_travel = np.mean(patch)
        if mean_travel < 30:
            labels.append(0)  # Low travel time (high accessibility)
        elif mean_travel < 120:
            labels.append(1)  # Medium
        else:
            labels.append(2)  # High (low accessibility)
    labels = np.array(labels)
    
    # Combine travel and friction patches (stack channels)
    combined_patches = np.concatenate([travel_patches, friction_patches], axis=-1)
    
    model = build_cnn_model(input_shape=(patch_size, patch_size, 2))
    model.fit(combined_patches, labels, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    return model

def predict_accessibility(cnn_model, travel_data, friction_data, bounds, patch_size=64):
    """Predict accessibility classes across the raster and generate overlay."""
    travel_patches = preprocess_raster_for_cnn(travel_data, patch_size)
    friction_patches = preprocess_raster_for_cnn(friction_data, patch_size)
    
    if travel_patches is None or friction_patches is None:
        return None, None
    
    combined_patches = np.concatenate([travel_patches, friction_patches], axis=-1)
    predictions = cnn_model.predict(combined_patches, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Reconstruct raster
    height, width = travel_data.shape
    accessibility_map = np.zeros((height, width), dtype=np.uint8)
    idx = 0
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            if idx < len(predicted_classes):
                accessibility_map[i:i+patch_size, j:j+patch_size] = predicted_classes[idx]
                idx += 1
    
    # Generate RGB image for visualization
    colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]  # Green (high), Yellow (medium), Red (low)
    breaks = [0, 1, 2, 3]
    rgb = generate_colors(accessibility_map, breaks, colors)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        accessibility_png_path = tmp.name
        Image.fromarray(rgb).save(accessibility_png_path)
        st.session_state['temp_files'].append(accessibility_png_path)
    
    return accessibility_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

def train_rf_model(merged_df, travel_data, friction_data, population_data):
    """Train Random Forest to predict gross margins."""
    if merged_df.empty or travel_data is None or friction_data is None or population_data is None:
        return None
    
    # Extract raster values at market and farmgate locations
    features = []
    for _, row in merged_df.iterrows():
        market_point = (row['latitude'], row['longitude'])
        region_point = (row['region_latitude'], row['region_longitude'])
        try:
            with rasterio.open(st.session_state.file_paths['raster']) as src:
                travel_val = next(src.sample([market_point[::-1]]))[0]
            with rasterio.open(st.session_state.file_paths['friction']) as src:
                friction_val = next(src.sample([market_point[::-1]]))[0]
            with rasterio.open(st.session_state.file_paths[f'population_{min(int(row["year"]), 2020)}']) as src:
                pop_val = next(src.sample([market_point[::-1]]))[0]
            features.append([travel_val, friction_val, pop_val, row['distance_km']])
        except Exception as e:
            st.warning(f"Error extracting raster values for point {market_point}: {e}")
            continue
    
    if not features:
        return None
    
    X = np.array(features)
    y = merged_df['gross_margin'].values[:len(features)]
    
    # Handle missing values
    X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
    y = np.nan_to_num(y, nan=np.nanmean(y))
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def cluster_markets(markets_gdf, n_clusters=3):
    """Apply K-means clustering to markets based on geospatial features."""
    if markets_gdf.empty:
        return None
    
    # Extract features: coordinates and population
    features = markets_gdf[['geometry', 'population_5km']].copy()
    features['lon'] = features['geometry'].x
    features['lat'] = features['geometry'].y
    X = features[['lon', 'lat', 'population_5km']].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    markets_gdf['cluster'] = clusters
    return markets_gdf

# -------------------------------
# Modified Helper Functions
# -------------------------------
def generate_colors(data, breaks, colors):
    """Convert raster data to RGB image based on specified colors."""
    rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    for i in range(len(breaks) - 1):
        mask = (data >= breaks[i]) & (data < breaks[i + 1])
        rgb[mask] = colors[i]
    return rgb

# Other helper functions (log_error, ensure_default_files, etc.) remain unchanged

# -------------------------------
# Modified Data Loading Functions
# -------------------------------
@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash(tuple([x.data.tobytes(), x.mask.tobytes()]))})
def load_and_process_raster(file_path, downsample_factor=2):
    try:
        with rasterio.open(file_path) as src:
            if src.count != 1:
                log_error(f"Raster {file_path} must have exactly one band.")
                return None, None
            if src.crs is None:
                st.warning(f"No CRS found for {file_path}. Assuming WGS84.")
            data = src.read(1, out_shape=(1, src.height // downsample_factor, src.width // downsample_factor), resampling=rasterio.enums.Resampling.bilinear)
            nodata = src.nodata
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        return np.ma.masked_equal(data, nodata) if nodata else np.ma.masked_invalid(data), bounds
    except rasterio.errors.RasterioIOError as e:
        log_error(f"Failed to load raster {file_path}: {e}")
        return None, None

# Other data loading functions (load_geojson, load_price_data, etc.) remain unchanged

# -------------------------------
# Modified Main Dashboard
# -------------------------------
def main():
    # Custom CSS (unchanged)
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
    .folium-map { height: calc(100vh - 200px); max-height: 1000px; width: 100% !important; }
    .stApp [data-testid="stMapContainer"] { 
        margin-top: 10px; 
        width: 100% !important; 
        max-height: 100vh; 
        overflow: auto; 
    }
    .legend-container { 
        background-color: white; 
        border: 2px solid grey; 
        padding: 10px; 
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
        margin-top: 10px; 
        display: block; 
    }
    .legend-title { font-weight: bold; font-size: 14px; margin-bottom: 10px; }
    .legend-item { display: flex; align-items: center; margin-bottom: 5px; font-size: 14px; }
    .legend-color { width: 20px; height: 20px; margin-right: 8px; display: inline-block; }
    @media (max-width: 600px) {
        .folium-map { height: 50vh; }
        .legend-container { font-size: 12px; padding: 5px; }
        .legend-color { width: 15px; height: 15px; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Header (unchanged)
    st.markdown("""
    <div class="header" role="banner" aria-label="Dashboard Header">
        <img src="https://www.ifpri.org/themes/custom/ifpri/logo.svg" alt="IFPRI Logo" width="150">
        <h1>Senegal Agricultural Market Dashboard</h1>
        <p>Developed in collaboration with the International Food Policy Research Institute (IFPRI)</p>
    </div>
    """, unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    ### Welcome to the Senegal Agricultural Market Dashboard
    This interactive tool visualizes travel time, friction surfaces, market locations, road networks, commodity prices, population density, and AI-driven accessibility predictions across Senegal. 
    Compare retail prices at specific markets with farmgate prices in production regions, and explore AI-predicted cost drivers.
    """)

    # File Upload Section (unchanged)
    st.sidebar.header("Data Sources")
    uploaded_files = {}
    for key, default_path in DEFAULT_FILES.items():
        uploaded_file = st.sidebar.file_uploader(f"Upload {key.replace('_', ' ').title()} File", type=['tiff', 'tif'] if 'population' in key or key in ['raster', 'friction'] else ['geojson', 'xlsx'])
        if uploaded_file:
            uploaded_path = os.path.join(BASE_DIR, uploaded_file.name)
            with open(uploaded_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            uploaded_files[key] = uploaded_path
        else:
            uploaded_files[key] = default_path
    st.session_state.file_paths.update(uploaded_files)

    # Validate files (unchanged)
    if not ensure_default_files():
        return
    for key, path in st.session_state.file_paths.items():
        if not validate_file(path, key.replace('_', ' ').title()):
            return

    # Load data
    with st.spinner("Loading data..."):
        progress = st.progress(0)
        travel_time, travel_bounds = load_and_process_raster(st.session_state.file_paths['raster'])
        progress.progress(0.2)
        friction_data, friction_bounds = load_and_process_raster(st.session_state.file_paths['friction'])
        progress.progress(0.4)
        
        selected_year = st.session_state.get('selected_year', 2020)
        pop_key = f'population_{min(selected_year, 2020)}'
        population_data, population_bounds = load_and_process_raster(st.session_state.file_paths[pop_key])
        progress.progress(0.6)
        
        markets = load_geojson(st.session_state.file_paths['markets'], population_file=st.session_state.file_paths[pop_key])
        progress.progress(0.8)
        roads_filtered = load_geojson(st.session_state.file_paths['roads'], max_features=500, is_roads=True)
        progress.progress(1.0)

    if travel_time is None or friction_data is None or population_data is None:
        log_error("Failed to load raster data. Please check the files and try again.")
        return

    # Generate raster images (unchanged)
    travel_png_path, travel_image_bounds, travel_breaks, travel_colors = generate_travel_image(travel_time, travel_bounds)
    friction_png_path, friction_image_bounds, friction_breaks, friction_colors = generate_friction_image(friction_data, friction_bounds)
    population_png_path, population_image_bounds, population_breaks, population_colors = generate_population_image(population_data, population_bounds)

    # Load price data (unchanged)
    prices_df = load_price_data(st.session_state.file_paths['prices'])
    retail_df = load_retail_data(st.session_state.file_paths['prices'])
    merged_df = load_merged_data(st.session_state.file_paths['prices'])
    
    commodity_map = pd.concat([
        prices_df[['commodity_id', 'commodity_english']].drop_duplicates() if not prices_df.empty else pd.DataFrame(),
        retail_df[['commodity_id', 'commodity']].drop_duplicates().rename(columns={'commodity': 'commodity_english'}) if not retail_df.empty else pd.DataFrame(),
        merged_df[['commodity_id', 'commodity_english']].drop_duplicates() if not merged_df.empty else pd.DataFrame()
    ]).drop_duplicates(subset='commodity_id').set_index('commodity_id')['commodity_english'].to_dict()
    st.session_state.commodity_map = commodity_map

    commodity_options = validate_commodity_overlap(prices_df, retail_df)
    if not commodity_options:
        log_error("No overlapping commodities found between farmgate and retail data.")
        return
    commodity_id_to_name = {cid: commodity_map.get(cid, str(cid)) for cid in commodity_options}

    # Train AI models
    with st.spinner("Training AI models..."):
        if st.session_state['cnn_model'] is None:
            st.session_state['cnn_model'] = train_cnn_model(travel_time, friction_data)
        if st.session_state['rf_model'] is None and not merged_df.empty:
            st.session_state['rf_model'] = train_rf_model(merged_df, travel_time, friction_data, population_data)
        if st.session_state['market_clusters'] is None and markets:
            markets_gdf = gpd.read_file(st.session_state.file_paths['markets'])
            st.session_state['market_clusters'] = cluster_markets(markets_gdf)

    # Generate AI-driven accessibility overlay
    accessibility_png_path, accessibility_bounds = None, None
    if st.session_state['cnn_model'] is not None:
        accessibility_png_path, accessibility_bounds = predict_accessibility(
            st.session_state['cnn_model'], travel_time, friction_data, travel_bounds
        )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Interactive Map", "Data Summary", "Price Trends", "AI Insights"])

    with tab1:
        st.subheader("Interactive Map")
        st.markdown("Explore travel time, friction surfaces, market locations, road networks, commodity prices, population density, and AI-predicted accessibility. Compare retail and farmgate prices for selected markets and commodities.")
        st.info("Map view is fixed. Pan or zoom to explore all data. Roads layer may load slowly due to data complexity.")

        # Filter controls (unchanged)
        st.sidebar.header("Map Filters")
        available_years = list(range(2016, 2026))
        selected_year = st.sidebar.selectbox("Select Year", available_years, index=len(available_years)-1, key="year_select", on_change=lambda: st.session_state.update({'map_data_updated': True, 'selected_year': st.session_state['year_select']}))
        st.session_state['selected_year'] = selected_year

        month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                       7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
        available_months = sorted(list(set(prices_df['Month'].unique()) | set(retail_df['Month'].unique()) | set(merged_df['month'].unique()))) if not prices_df.empty or not retail_df.empty or not merged_df.empty else list(range(1, 13))
        latest_month = max(available_months) if available_months else 12
        selected_month_name = st.sidebar.selectbox("Select Month", [month_names.get(m, str(m)) for m in available_months], index=available_months.index(latest_month) if latest_month in available_months else 0, key="month_select", on_change=lambda: st.session_state.update({'map_data_updated': True}))
        selected_month = next((k for k, v in month_names.items() if v == selected_month_name), selected_month_name)

        selected_commodity_ids = st.sidebar.multiselect(
            "Select Commodities",
            options=sorted(commodity_options, key=lambda x: commodity_id_to_name[x]),
            format_func=lambda x: commodity_id_to_name.get(x, str(x)),
            default=commodity_options,
            key="commodity_select",
            on_change=lambda: st.session_state.update({'map_data_updated': True})
        )

        selected_market = st.sidebar.selectbox(
            "Select Market for Price Comparison",
            options=["All"] + sorted(merged_df['market'].unique()) if not merged_df.empty else ["All"],
            index=0,
            key="market_select",
            on_change=lambda: st.session_state.update({'map_data_updated': True})
        )

        # Layer toggles
        st.sidebar.header("Map Layers")
        show_travel = st.sidebar.checkbox("Travel Time", value=False)
        show_friction = st.sidebar.checkbox("Friction Surface", value=False)
        show_roads = st.sidebar.checkbox("Roads", value=False)
        show_markets = st.sidebar.checkbox("Markets", value=True)
        show_farmgate = st.sidebar.checkbox("Farmgate Prices", value=False)
        show_retail = st.sidebar.checkbox("Retail Prices", value=False)
        show_population = st.sidebar.checkbox("Population", value=True)
        show_merged = st.sidebar.checkbox("Merged Retail-Farmgate Comparison", value=True)
        show_accessibility = st.sidebar.checkbox("AI-Predicted Accessibility", value=True)
        show_clusters = st.sidebar.checkbox("Market Clusters", value=True)

        st.sidebar.slider("Map Height (px)", 400, 1000, st.session_state['map_height'], key="map_height")
        st.sidebar.button("Clear Temporary Files", on_click=cleanup_temp_files)
        st.session_state['show_legends'] = st.sidebar.checkbox("Show Legends", value=st.session_state['show_legends'])

        # Update map data (unchanged)
        if st.session_state.map_data_updated or st.session_state.latest_farmgate_prices.empty or st.session_state.latest_retail_prices.empty or st.session_state.latest_merged_prices.empty:
            latest_farmgate_prices = pd.DataFrame()
            latest_retail_prices = pd.DataFrame()
            latest_merged_prices = pd.DataFrame()
            if not prices_df.empty:
                filtered_farmgate = prices_df[prices_df['Year'] == selected_year] if selected_year else prices_df
                if selected_month:
                    filtered_farmgate = filtered_farmgate[filtered_farmgate['Month'] == selected_month]
                filtered_farmgate['Date'] = pd.to_datetime(filtered_farmgate[['Year', 'Month']].assign(day=1), errors='coerce')
                filtered_farmgate = filtered_farmgate.dropna(subset=['Date'])
                latest_farmgate_prices = filtered_farmgate.sort_values('Date').groupby(['Régions Name', 'commodity_id']).last().reset_index()
                if selected_commodity_ids:
                    latest_farmgate_prices = latest_farmgate_prices[latest_farmgate_prices['commodity_id'].isin(selected_commodity_ids)]
                markets_gdf = gpd.read_file(st.session_state.file_paths['markets']) if markets else gpd.GeoDataFrame()
                latest_farmgate_prices = calculate_nearest_market_distance(latest_farmgate_prices, markets_gdf)
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
                if selected_commodity_ids:
                    latest_retail_prices = latest_retail_prices[latest_retail_prices['commodity_id'].isin(selected_commodity_ids)]
                if len(latest_retail_prices) > 500:
                    latest_retail_prices = latest_retail_prices.head(500)
                    st.warning("Limited to 500 retail price markers for performance.")
            if not merged_df.empty:
                filtered_merged = merged_df[merged_df['year'] == selected_year] if selected_year else merged_df
                if selected_month:
                    filtered_merged = filtered_merged[filtered_merged['month'] == selected_month]
                if selected_commodity_ids:
                    filtered_merged = filtered_merged[filtered_merged['commodity_id'].isin(selected_commodity_ids)]
                if selected_market != "All":
                    filtered_merged = filtered_merged[filtered_merged['market'] == selected_market]
                filtered_merged['date'] = pd.to_datetime(filtered_merged[['year', 'month']].assign(day=1), errors='coerce')
                filtered_merged = filtered_merged.dropna(subset=['date'])
                latest_merged_prices = filtered_merged.sort_values('date').groupby(['market', 'region_name', 'commodity_id']).last().reset_index()
                if len(latest_merged_prices) > 500:
                    latest_merged_prices = latest_merged_prices.head(500)
                    st.warning("Limited to 500 merged price markers for performance.")
            st.session_state.latest_farmgate_prices = latest_farmgate_prices
            st.session_state.latest_retail_prices = latest_retail_prices
            st.session_state.latest_merged_prices = latest_merged_prices
            st.session_state['map_data_updated'] = False
            st.session_state.map_render_key += 1

        # Render Map
        map_placeholder = st.empty()
        with map_placeholder.container():
            data_missing = False
            if show_farmgate and st.session_state.latest_farmgate_prices.empty:
                selected_commodities = ", ".join([commodity_id_to_name.get(cid, str(cid)) for cid in selected_commodity_ids]) or "any commodity"
                st.warning(f"No farmgate price data available for {selected_commodities} in {month_names.get(selected_month, selected_month)} {selected_year}. Map will not render.")
                data_missing = True
            if show_retail and st.session_state.latest_retail_prices.empty:
                selected_commodities = ", ".join([commodity_id_to_name.get(cid, str(cid)) for cid in selected_commodity_ids]) or "any commodity"
                st.warning(f"No retail price data available for {selected_commodities} in {month_names.get(selected_month, selected_month)} {selected_year}. Map will not render.")
                data_missing = True
            if show_merged and st.session_state.latest_merged_prices.empty:
                selected_commodities = ", ".join([commodity_id_to_name.get(cid, str(cid)) for cid in selected_commodity_ids]) or "any commodity"
                market_text = f"market {selected_market}" if selected_market != "All" else "any market"
                st.warning(f"No merged retail-farmgate data available for {selected_commodities} in {market_text}, {month_names.get(selected_month, selected_month)} {selected_year}. Map will not render.")
                data_missing = True

            if not data_missing or show_travel or show_friction or show_roads or show_markets or show_population or show_accessibility or show_clusters:
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
                            valid_markers = 0
                            for feature in markets.get('features', []):
                                if feature['geometry']['type'] == 'Point' and all(isinstance(c, (int, float)) for c in feature['geometry']['coordinates']):
                                    coords = feature['geometry']['coordinates'][::-1]
                                    market_name = feature['properties'].get('market', 'Unknown Market')
                                    population_5km = feature['properties'].get('population_5km', 0)
                                    popup_text = f"<b>Market:</b> {market_name}<br><b>Population (5km):</b> {population_5km:,.0f}"
                                    folium.Marker(
                                        location=coords,
                                        popup=folium.Popup(popup_text, max_width=250),
                                        icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')
                                    ).add_to(market_group)
                                    valid_markers += 1
                            if valid_markers == 0:
                                st.warning("No valid market locations found.")
                            else:
                                market_group.add_to(m)

                        if show_clusters and st.session_state['market_clusters'] is not None:
                            cluster_group = folium.FeatureGroup(name="Market Clusters", show=True)
                            cluster_colors = ['red', 'blue', 'green', 'purple', 'orange']
                            for idx, row in st.session_state['market_clusters'].iterrows():
                                if row.geometry.type == 'Point':
                                    coords = [row.geometry.y, row.geometry.x]
                                    cluster_id = row['cluster']
                                    popup_text = f"<b>Market:</b> {row.get('market', 'Unknown')}<br><b>Cluster:</b> {cluster_id}<br><b>Population (5km):</b> {row.get('population_5km', 0):,.0f}"
                                    folium.Marker(
                                        location=coords,
                                        popup=folium.Popup(popup_text, max_width=250),
                                        icon=folium.Icon(color=cluster_colors[cluster_id % len(cluster_colors)], icon='map-marker', prefix='fa')
                                    ).add_to(cluster_group)
                            cluster_group.add_to(m)

                        if show_farmgate and not st.session_state.latest_farmgate_prices.empty:
                            farmgate_data = [
                                [row['Régions - Latitude'], row['Régions - Longitude'], 
                                 f"<b>Region:</b> {row['Régions Name']}<br><b>Commodity:</b> {row['commodity_english']}<br><b>Price:</b> {row['Price']:.2f} {row['Unit2']}<br><b>Distance to Nearest Market:</b> {row['Distance_to_Nearest_Market_km']:.2f} km<br><b>Date:</b> {row['Year']}-{row['Month']:02d}"]
                                for _, row in st.session_state.latest_farmgate_prices.iterrows()
                                if not pd.isna(row['Régions - Latitude']) and not pd.isna(row['Régions - Longitude'])
                            ]
                            FastMarkerCluster(
                                data=farmgate_data,
                                name="Farmgate Prices",
                                callback="function(row) {return L.marker([row[0], row[1]], {icon: L.AwesomeMarkers.icon({icon: 'tractor', prefix: 'fa', markerColor: 'green'})}).bindPopup(row[2]);}"
                            ).add_to(m)

                        if show_retail and not st.session_state.latest_retail_prices.empty:
                            retail_data = [
                                [row['latitude'], row['longitude'], 
                                 f"<b>Market:</b> {row['market']}<br><b>Commodity:</b> {row['commodity']}<br><b>Price:</b> {row['Price']:.2f} {row['Unit2']}<br><b>Date:</b> {row['Year']}-{row['Month']:02d}"]
                                for _, row in st.session_state.latest_retail_prices.iterrows()
                                if not pd.isna(row['latitude']) and not pd.isna(row['longitude'])
                            ]
                            FastMarkerCluster(
                                data=retail_data,
                                name="Retail Prices",
                                callback="function(row) {return L.marker([row[0], row[1]], {icon: L.AwesomeMarkers.icon({icon: 'shopping-basket', prefix: 'fa', markerColor: 'purple'})}).bindPopup(row[2]);}"
                            ).add_to(m)

                        if show_merged and not st.session_state.latest_merged_prices.empty:
                            retail_merged_data = [
                                [row['latitude'], row['longitude'],
                                 f"<b>Market:</b> {row['market']}<br><b>Commodity:</b> {row['commodity_retail']}<br><b>Retail Price:</b> {row['price_retail']:.2f} {row['unit2_retail']}<br><b>Date:</b> {row['year']}-{row['month']:02d}"]
                                for _, row in st.session_state.latest_merged_prices.drop_duplicates(subset=['market', 'commodity_id']).iterrows()
                                if not pd.isna(row['latitude']) and not pd.isna(row['longitude'])
                            ]
                            farmgate_merged_data = [
                                [row['region_latitude'], row['region_longitude'],
                                 f"<b>Region:</b> {row['region_name']}<br><b>Commodity:</b> {row['commodity_farmgate']}<br><b>Farmgate Price:</b> {row['price_farmgate']:.2f} {row['unit2_farmgate']}<br><b>Retail Market:</b> {row['market']}<br><b>Retail Price:</b> {row['price_retail']:.2f} {row['unit2_retail']}<br><b>Gross Margin:</b> {row['gross_margin']:.2f} XOF/KG<br><b>Distance:</b> {row['distance_km']:.2f} km<br><b>Date:</b> {row['year']}-{row['month']:02d}"]
                                for _, row in st.session_state.latest_merged_prices.iterrows()
                                if not pd.isna(row['region_latitude']) and not pd.isna(row['region_longitude'])
                            ]
                            if retail_merged_data:
                                FastMarkerCluster(
                                    data=retail_merged_data,
                                    name="Merged Retail Prices",
                                    callback="function(row) {return L.marker([row[0], row[1]], {icon: L.AwesomeMarkers.icon({icon: 'shopping-basket', prefix: 'fa', markerColor: 'purple'})}).bindPopup(row[2]);}"
                                ).add_to(m)
                            if farmgate_merged_data:
                                FastMarkerCluster(
                                    data=farmgate_merged_data,
                                    name="Merged Farmgate Prices",
                                    callback="function(row) {return L.marker([row[0], row[1]], {icon: L.AwesomeMarkers.icon({icon: 'tractor', prefix: 'fa', markerColor: 'green'})}).bindPopup(row[2]);}"
                                ).add_to(m)

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
                        if show_population:
                            folium.raster_layers.ImageOverlay(
                                name=f"Population ({min(selected_year, 2020)})",
                                image=population_png_path,
                                bounds=population_image_bounds,
                                opacity=0.8,
                                interactive=True,
                                cross_origin=False
                            ).add_to(m)
                        if show_accessibility and accessibility_png_path:
                            folium.raster_layers.ImageOverlay(
                                name="AI-Predicted Accessibility",
                                image=accessibility_png_path,
                                bounds=accessibility_bounds,
                                opacity=0.7,
                                interactive=True,
                                cross_origin=False
                            ).add_to(m)

                        MiniMap(tiles='OpenStreetMap', position='bottomleft', width=150, height=150).add_to(m)
                        Fullscreen(position='topright', title='Expand').add_to(m)
                        folium.LayerControl(collapsed=False).add_to(m)
                        st_folium(m, use_container_width=True, height=st.session_state['map_height'], key=f"folium_map_{st.session_state.map_render_key}")

                        # Render legends
                        if st.session_state['show_legends'] and (show_travel or show_friction or show_population or show_accessibility):
                            st.markdown("### Raster Layer Legends")
                            col1, col2 = st.columns(2)
                            if show_travel and travel_breaks and travel_colors:
                                with col1:
                                    st.markdown(generate_legend_html(travel_breaks, travel_colors, "Travel Time (min)"), unsafe_allow_html=True)
                            if show_friction and friction_breaks and friction_colors:
                                with col2:
                                    st.markdown(generate_legend_html(friction_breaks, friction_colors, "Friction (min/m)"), unsafe_allow_html=True)
                            if show_population and population_breaks and population_colors:
                                st.markdown(generate_legend_html(population_breaks, population_colors, f"Population ({min(selected_year, 2020)}) (people per pixel)"), unsafe_allow_html=True)
                            if show_accessibility:
                                accessibility_breaks = [0, 1, 2, 3]
                                accessibility_colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]
                                st.markdown(generate_legend_html(accessibility_breaks, accessibility_colors, "AI Accessibility (0=High, 1=Medium, 2=Low)"), unsafe_allow_html=True)
                    except Exception as e:
                        log_error(f"Map rendering failed: {str(e)}")

    with tab2:
        # Data Summary (unchanged)
        st.subheader("Data Summary")
        st.markdown("### Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Markets", len(markets['features']) if markets else 0)
        col2.metric("Road Features", len(roads_filtered['features']) if roads_filtered else 0)
        col3.metric("Price Points", len(st.session_state.latest_farmgate_prices) + len(st.session_state.latest_retail_prices) + len(st.session_state.latest_merged_prices))
        avg_distance = st.session_state.latest_farmgate_prices['Distance_to_Nearest_Market_km'].mean() if not st.session_state.latest_farmgate_prices.empty and 'Distance_to_Nearest_Market_km' in st.session_state.latest_farmgate_prices.columns else np.nan
        col4.metric("Avg Distance to Market (km)", f"{avg_distance:.2f}" if not pd.isna(avg_distance) else "N/A")

        st.markdown("### Raw Data Preview")
        if not st.session_state.latest_farmgate_prices.empty:
            st.markdown("#### Farmgate Prices")
            st.dataframe(
                st.session_state.latest_farmgate_prices[['Régions Name', 'commodity_english', 'Price', 'Unit2', 'Distance_to_Nearest_Market_km', 'Year', 'Month']],
                use_container_width=True,
                height=300
            )
            st.download_button(
                label="Download Farmgate Prices",
                data=st.session_state.latest_farmgate_prices.to_csv(index=False),
                file_name=f"farmgate_prices_{selected_year}_{selected_month}.csv",
                mime="text/csv",
                key="download_farmgate"
            )
        if not st.session_state.latest_retail_prices.empty:
            st.markdown("#### Retail Prices")
            st.dataframe(
                st.session_state.latest_retail_prices[['market', 'commodity', 'Price', 'Unit2', 'Year', 'Month']],
                use_container_width=True,
                height=300
            )
            st.download_button(
                label="Download Retail Prices",
                data=st.session_state.latest_retail_prices.to_csv(index=False),
                file_name=f"retail_prices_{selected_year}_{selected_month}.csv",
                mime="text/csv",
                key="download_retail"
            )
        if not st.session_state.latest_merged_prices.empty:
            st.markdown("#### Merged Retail-Farmgate Prices")
            st.dataframe(
                st.session_state.latest_merged_prices[['market', 'commodity_retail', 'price_retail', 'unit2_retail', 'region_name', 'commodity_farmgate', 'price_farmgate', 'unit2_farmgate', 'gross_margin', 'distance_km', 'year', 'month']],
                use_container_width=True,
                height=300
            )
            st.download_button(
                label="Download Merged Prices",
                data=st.session_state.latest_merged_prices.to_csv(index=False),
                file_name=f"merged_prices_{selected_year}_{selected_month}.csv",
                mime="text/csv",
                key="download_merged"
            )

        if st.session_state['errors']:
            st.markdown("### Errors")
            for err in st.session_state['errors']:
                st.markdown(f"- {err}")

    with tab3:
        # Price Trends (unchanged)
        st.subheader("Price Trends")
        if not prices_df.empty and not retail_df.empty and commodity_options:
            st.markdown("### Farmgate, Retail, and Gross Margin Trends")
            selected_commodity_id = st.selectbox(
                "Select Commodity",
                options=sorted(commodity_options, key=lambda x: commodity_id_to_name[x]),
                format_func=lambda x: commodity_id_to_name.get(x, str(x)),
                key="trend_commodity_select"
            )
            show_gross_margin = st.checkbox("Show Gross Margin (Retail - Farmgate)", value=True, key="gross_margin")
            try:
                farmgate_trend = prices_df[prices_df['commodity_id'] == selected_commodity_id][['Year', 'Month', 'Price']].groupby(['Year', 'Month']).mean().reset_index()
                if not farmgate_trend.empty:
                    farmgate_trend['Date'] = pd.to_datetime(farmgate_trend[['Year', 'Month']].assign(day=1), errors='coerce')
                    farmgate_trend = farmgate_trend.dropna(subset=['Date']).reset_index(drop=True)
                    farmgate_trend = interpolate_missing(farmgate_trend)
                    farmgate_trend['Price Type'] = 'Farmgate'
                else:
                    farmgate_trend = pd.DataFrame(columns=['Date', 'Price', 'Price Type'])

                retail_trend = retail_df[retail_df['commodity_id'] == selected_commodity_id][['Year', 'Month', 'Price']].groupby(['Year', 'Month']).mean().reset_index()
                if not retail_trend.empty:
                    retail_trend['Date'] = pd.to_datetime(retail_trend[['Year', 'Month']].assign(day=1), errors='coerce')
                    retail_trend = retail_trend.dropna(subset=['Date']).reset_index(drop=True)
                    retail_trend = interpolate_missing(retail_trend)
                    retail_trend['Price Type'] = 'Retail'
                else:
                    retail_trend = pd.DataFrame(columns=['Date', 'Price', 'Price Type'])

                if show_gross_margin and not farmgate_trend.empty and not retail_trend.empty:
                    margin_trend = pd.merge(
                        farmgate_trend[['Date', 'Price']],
                        retail_trend[['Date', 'Price']],
                        on='Date',
                        how='inner',
                        suffixes=('_farmgate', '_retail')
                    )
                    margin_trend['Price'] = margin_trend['Price_retail'] - margin_trend['Price_farmgate']
                    margin_trend['Price Type'] = 'Gross Margin'
                    margin_trend = margin_trend[['Date', 'Price', 'Price Type']].dropna(subset=['Price'])
                else:
                    margin_trend = pd.DataFrame(columns=['Date', 'Price', 'Price Type'])

                combined_trend = pd.concat([farmgate_trend, retail_trend, margin_trend], ignore_index=True)
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
                            hovertemplate='%{x|%b %Y}<br>Price: %{y:.2f} XOF/KG<br>Type: Farmgate'
                        ))
                    if 'Retail' in combined_trend['Price Type'].values:
                        fig.add_trace(go.Scatter(
                            x=combined_trend[combined_trend['Price Type'] == 'Retail']['Date'],
                            y=combined_trend[combined_trend['Price Type'] == 'Retail']['Price'],
                            mode='lines+markers',
                            name='Retail',
                            line=dict(color='#9467bd', width=2),
                            marker=dict(size=6),
                            hovertemplate='%{x|%b %Y}<br>Price: %{y:.2f} XOF/KG<br>Type: Retail'
                        ))
                    if show_gross_margin and 'Gross Margin' in combined_trend['Price Type'].values:
                        fig.add_trace(go.Scatter(
                            x=combined_trend[combined_trend['Price Type'] == 'Gross Margin']['Date'],
                            y=combined_trend[combined_trend['Price Type'] == 'Gross Margin']['Price'],
                            mode='lines+markers',
                            name='Gross Margin',
                            line=dict(color='#d62728', width=2, dash='dash'),
                            marker=dict(size=6),
                            hovertemplate='%{x|%b %Y}<br>Margin: %{y:.2f} XOF/KG<br>Type: Gross Margin'
                        ))

                    fig.update_layout(
                        title=f"{commodity_id_to_name.get(selected_commodity_id, selected_commodity_id)} Price and Margin Trends",
                        title_x=0.5,
                        xaxis_title="Date",
                        yaxis_title="Average Price/Margin (XOF/KG)",
                        font=dict(family="Roboto, sans-serif", size=12),
                        hovermode="x unified",
                        showlegend=True,
                        template="plotly_white",
                        xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                log_error(f"Failed to generate price trends: {e}")
        else:
            st.warning("No data available for price trends. Please check your data.")

    with tab4:
        st.subheader("AI-Driven Insights")
        st.markdown("### Geospatial AI Analysis")
        st.markdown("This tab provides insights from AI models analyzing accessibility and price margins.")

        if st.session_state['cnn_model'] is not None:
            st.markdown("#### Accessibility Classification")
            st.markdown("The CNN model classifies regions by accessibility (High, Medium, Low) based on travel time and friction surfaces. High accessibility (green) indicates low travel times, while low accessibility (red) indicates high travel times, impacting logistics costs.")
            if accessibility_png_path:
                st.image(accessibility_png_path, caption="AI-Predicted Accessibility (Green=High, Yellow=Medium, Red=Low)", use_column_width=True)

        if st.session_state['rf_model'] is not None:
            st.markdown("#### Predicted Gross Margins")
            st.markdown("The Random Forest model predicts gross margins based on travel time, friction, population density, and distance to markets.")
            sample_data = st.session_state.latest_merged_prices.head(5)
            if not sample_data.empty:
                features = []
                for _, row in sample_data.iterrows():
                    market_point = (row['latitude'], row['longitude'])
                    try:
                        with rasterio.open(st.session_state.file_paths['raster']) as src:
                            travel_val = next(src.sample([market_point[::-1]]))[0]
                        with rasterio.open(st.session_state.file_paths['friction']) as src:
                            friction_val = next(src.sample([market_point[::-1]]))[0]
                        with rasterio.open(st.session_state.file_paths[f'population_{min(int(row["year"]), 2020)}']) as src:
                            pop_val = next(src.sample([market_point[::-1]]))[0]
                        features.append([travel_val, friction_val, pop_val, row['distance_km']])
                    except Exception as e:
                        st.warning(f"Error extracting features for prediction: {e}")
                        continue
                if features:
                    X = np.nan_to_num(np.array(features), nan=np.nanmean(features, axis=0))
                    predictions = st.session_state['rf_model'].predict(X)
                    pred_df = sample_data[['market', 'commodity_retail', 'gross_margin']].copy()
                    pred_df['Predicted Margin'] = predictions
                    st.dataframe(pred_df, use_container_width=True)
                    st.download_button(
                        label="Download Predicted Margins",
                        data=pred_df.to_csv(index=False),
                        file_name=f"predicted_margins_{selected_year}_{selected_month}.csv",
                        mime="text/csv",
                        key="download_predictions"
                    )

        if st.session_state['market_clusters'] is not None:
            st.markdown("#### Market Clusters")
            st.markdown("Markets are clustered based on location and population density, identifying groups with similar characteristics for targeted policy interventions.")
            cluster_counts = st.session_state['market_clusters']['cluster'].value_counts()
            fig = px.bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                labels={'x': 'Cluster ID', 'y': 'Number of Markets'},
                title="Market Cluster Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    # Footer (unchanged)
    st.markdown("""
    <div class="footer">
        <p>Developed by xAI in collaboration with IFPRI | Data Sources: IFPRI, OpenStreetMap, WorldPop | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()