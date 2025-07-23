import os
import json
import uuid
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from PIL import Image
import folium
from folium.plugins import MarkerCluster, MiniMap, Fullscreen
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import atexit
from scipy.interpolate import interp1d
from shapely.geometry import Point
from rasterio.mask import mask
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from datetime import datetime
import re
import streamlit as st
from streamlit_folium import st_folium

# -------------------------------
# Configuration
# -------------------------------
class Config:
    """Configuration class for file paths and model settings."""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEMP_DIR = os.path.join(BASE_DIR, "temp")
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
        'population_2020': os.path.join(BASE_DIR, 'sen_ppp_2020_UNadj.tif'),
        'merged_data_for_models': os.path.join(BASE_DIR, 'final_merged_output_senegal.xlsx')
    }
    
    COMMODITY_MODELS_CONFIG = {
        'sorghum': {
            'id': 65,
            'output_dir': os.path.join(BASE_DIR, 'sorghum_analysis_outputs'),
            'rf_model_file': 'rf_model_sorghum.pkl',
            'xgb_model_file': 'xgb_model_sorghum.pkl',
            'spoilage_rate': 0.116,
            'model_comparison_csv': 'model_comparison_sorghum.csv',
            'feature_importance_excel': 'feature_importance_sorghum.xlsx'
        },
        'maize': {
            'id': 56,
            'output_dir': os.path.join(BASE_DIR, 'maize_analysis_outputs'),
            'rf_model_file': 'rf_model_maize.pkl',
            'xgb_model_file': 'xgb_model_maize.pkl',
            'spoilage_rate': 0.1,
            'model_comparison_csv': 'model_comparison_maize.csv',
            'feature_importance_excel': 'feature_importance_maize.xlsx'
        }
    }
    TRANSPORT_RATE_XOF_PER_KG_KM = 0.3

    @staticmethod
    def ensure_temp_dir():
        """Create temporary directory if it doesn't exist."""
        if not os.path.exists(Config.TEMP_DIR):
            os.makedirs(Config.TEMP_DIR)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state with default values."""
    defaults = {
        'map_data_updated': False,
        'latest_farmgate_prices': pd.DataFrame(),
        'latest_retail_prices': pd.DataFrame(),
        'latest_merged_prices': pd.DataFrame(),
        'file_paths': Config.DEFAULT_FILES.copy(),
        'map_render_key': 0,
        'commodity_map': {},
        'map_height': 800,
        'errors': [],
        'temp_files': [],
        'show_legends': True,
        'models_and_scalers': {},
        'selected_year': 2020
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -------------------------------
# Helper Functions
# -------------------------------
def log_error(message):
    """Log errors to session state and display them."""
    st.session_state['errors'].append(message)
    st.error(message)

def validate_file(file_path, file_type):
    """Validate file existence."""
    if not os.path.exists(file_path):
        log_error(f"{file_type} file not found: {file_path}")
        return False
    return True

def generate_colors(data, breaks, colors):
    """Convert raster data to RGB image based on specified colors."""
    rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    for i in range(len(breaks) - 1):
        mask = (data >= breaks[i]) & (data < breaks[i + 1])
        rgb[mask] = colors[i]
    return rgb

def cleanup_temp_files():
    """Clean up temporary files."""
    for file in st.session_state['temp_files']:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                log_error(f"Failed to remove temporary file {file}: {e}")
    st.session_state['temp_files'] = []
    st.success("Temporary files cleaned up.")

def calculate_nearest_market_distance(farmgate_df, markets_gdf):
    """Calculate distance from each farmgate to nearest market."""
    if farmgate_df.empty or markets_gdf.empty:
        return farmgate_df
    farmgate_df['Distance_to_Nearest_Market_km'] = np.nan
    market_coords = [(row.geometry.y, row.geometry.x) for _, row in markets_gdf.iterrows() if row.geometry.type == 'Point']
    if not market_coords:
        return farmgate_df
    for idx, row in farmgate_df.iterrows():
        if pd.isna(row['Régions - Latitude']) or pd.isna(row['Régions - Longitude']):
            continue
        farmgate_point = (row['Régions - Latitude'], float(row['Régions - Longitude']))
        distances = [geodesic(farmgate_point, market).kilometers for market in market_coords]
        farmgate_df.at[idx, 'Distance_to_Nearest_Market_km'] = min(distances)
    return farmgate_df

def interpolate_missing(data, date_col='Date', value_col='Price'):
    """Interpolate missing price data."""
    if data.empty:
        return data
    try:
        x = data[date_col].map(lambda x: x.timestamp())
        y = data[value_col]
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        new_dates = pd.date_range(data[date_col].min(), data[date_col].max(), freq='M')
        new_values = f([d.timestamp() for d in new_dates])
        return pd.DataFrame({'Date': new_dates, value_col: new_values})
    except Exception as e:
        log_error(f"Error interpolating data: {e}")
        return data

def generate_legend_html(breaks, colors, title):
    """Generate HTML for map legend."""
    html = f'<div class="legend-container"><div class="legend-title">{title}</div>'
    for i, (b1, b2) in enumerate(zip(breaks[:-1], breaks[1:])):
        color = f'rgb{colors[i]}'
        label = f'{b1:.0f}–{b2:.0f}' if b2 != np.inf else f'>{b1:.0f}'
        html += f'<div class="legend-item"><span class="legend-color" style="background:{color};"></span> {label}</div>'
    html += '</div>'
    return html

def compute_population_within_buffer(markets_gdf, population_file, buffer_size_km=5):
    """Compute population within a buffer_size_km radius around each market."""
    if markets_gdf.empty or not os.path.exists(population_file):
        return {}
    utm_crs = 'EPSG:32628'
    try:
        markets_utm = markets_gdf.to_crs(utm_crs)
        markets_utm['geometry'] = markets_utm.geometry.buffer(buffer_size_km * 1000)
        markets_buffered = markets_utm.to_crs('EPSG:4326')
        population_sums = {}
        with rasterio.open(population_file) as src:
            for idx, row in markets_buffered.iterrows():
                geom = [row.geometry.__geo_interface__]
                try:
                    out_image, _ = mask(src, geom, crop=True, nodata=src.nodata)
                    out_image = np.ma.masked_equal(out_image, src.nodata)
                    population_sum = np.nansum(out_image)
                    population_sums[idx] = round(population_sum, 2) if not np.isnan(population_sum) else 0
                except Exception as e:
                    population_sums[idx] = 0
                    log_error(f"Error computing population for market {idx}: {e}")
        return population_sums
    except Exception as e:
        log_error(f"Error computing population buffers: {e}")
        return {}

def get_min_max_values(df, features_to_normalize):
    """Calculate min/max values for normalization."""
    min_max_vals = {}
    for col in features_to_normalize:
        if col in df.columns and df[col].std() > 0:
            min_max_vals[col] = {'min': df[col].min(), 'max': df[col].max()}
        elif col in df.columns:
            min_max_vals[col] = {'min': df[col].iloc[0], 'max': df[col].iloc[0]}
        else:
            log_error(f"Column '{col}' not found in DataFrame for min/max calculation.")
            min_max_vals[col] = {'min': 0, 'max': 1}
    return min_max_vals

# -------------------------------
# Data Loading Functions
# -------------------------------
@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash(tuple([x.data.tobytes(), x.mask.tobytes()]))})
def load_and_process_raster(file_path, downsample_factor=2):
    """Load and downsample raster data."""
    try:
        with rasterio.open(file_path) as src:
            if src.count != 1:
                log_error(f"Raster {file_path} must have exactly one band.")
                return None, None
            if src.crs is None:
                log_error(f"No CRS found for {file_path}. Assuming WGS84.")
            data = src.read(1, out_shape=(1, src.height // downsample_factor, src.width // downsample_factor), resampling=rasterio.enums.Resampling.bilinear)
            nodata = src.nodata
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        return np.ma.masked_equal(data, nodata) if nodata else np.ma.masked_invalid(data), bounds
    except rasterio.errors.RasterioIOError as e:
        log_error(f"Failed to load raster {file_path}: {e}")
        return None, None

@st.cache_data
def load_geojson(file_path, max_features=500, is_roads=False, population_file=None):
    """Load and process GeoJSON files."""
    try:
        gdf = gpd.read_file(file_path)
        if gdf.empty:
            log_error(f"GeoJSON {file_path} is empty.")
            return None
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        if is_roads:
            valid_highways = ['motorway', 'trunk', 'primary', 'secondary']
            gdf = gdf[gdf['highway'].isin(valid_highways)] if 'highway' in gdf.columns else gdf
        if len(gdf) > max_features:
            gdf = gdf.head(max_features)
        if not is_roads and population_file:
            population_sums = compute_population_within_buffer(gdf, population_file)
            gdf['population_5km'] = gdf.index.map(population_sums)
        return json.loads(gdf.to_json()) if gdf['geometry'].notnull().any() else None
    except Exception as e:
        log_error(f"Failed to load {file_path}: {e}")
        return None

@st.cache_data
def load_price_data(file_path):
    """Load farmgate price data from Excel."""
    try:
        prices_df = pd.read_excel(file_path, sheet_name='Farmgate prices Senegal')
        if prices_df.empty:
            return pd.DataFrame()
        required_columns = ['Régions Name', 'Commodity', 'commodity_english', 'commodity_id', 'Price', 'Unit2', 'Régions - Latitude', 'Régions - Longitude', 'Year', 'Month']
        missing_cols = [col for col in required_columns if col not in prices_df.columns]
        if missing_cols:
            log_error(f"Missing columns in farmgate prices: {missing_cols}")
        prices_df['Price'] = pd.to_numeric(prices_df['Price'], errors='coerce')
        prices_df['Year'] = pd.to_numeric(prices_df['Year'], errors='coerce')
        prices_df['Month'] = pd.to_numeric(prices_df['Month'], errors='coerce')
        prices_df = prices_df.dropna(subset=['Price', 'Year', 'Month', 'Régions - Latitude', 'Régions - Longitude', 'commodity_id'])
        if not prices_df.empty:
            prices_df = prices_df[prices_df['Year'].between(2016, 2025)]
        return prices_df
    except Exception as e:
        log_error(f"Error reading farmgate prices file {file_path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_retail_data(file_path):
    """Load retail price data from Excel."""
    try:
        retail_df = pd.read_excel(file_path, sheet_name='Retails Price Senegal')
        if retail_df.empty:
            return pd.DataFrame()
        required_columns = ['market', 'commodity', 'commodity_id', 'Price', 'Unit2', 'latitude', 'longitude', 'Year', 'Month']
        missing_cols = [col for col in required_columns if col not in retail_df.columns]
        if missing_cols:
            log_error(f"Missing columns in retail prices: {missing_cols}")
        retail_df = retail_df.rename(columns={'price': 'Price'})
        retail_df['Price'] = pd.to_numeric(retail_df['Price'], errors='coerce')
        retail_df['Year'] = pd.to_numeric(retail_df['Year'], errors='coerce')
        retail_df['Month'] = pd.to_numeric(retail_df['Month'], errors='coerce')
        retail_df = retail_df.dropna(subset=['Price', 'Year', 'Month', 'latitude', 'longitude', 'commodity_id'])
        if not retail_df.empty:
            retail_df = retail_df[retail_df['Year'].between(2016, 2025)]
        return retail_df
    except Exception as e:
        log_error(f"Error reading retail prices file {file_path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_merged_data(file_path):
    """Load merged price data from Excel."""
    try:
        merged_df = pd.read_excel(file_path, sheet_name='merged_data')
        if merged_df.empty:
            return pd.DataFrame()
        required_columns = ['date', 'market', 'market_id', 'latitude', 'longitude', 'commodity_retail', 'commodity_id',
                           'price_retail', 'unit2_retail', 'year', 'month', 'commodity_farmgate', 'region_name',
                           'region_latitude', 'region_longitude', 'price_farmgate', 'unit2_farmgate', 'commodity_english',
                           'gross_margin', 'distance_km']
        missing_cols = [col for col in required_columns if col not in merged_df.columns]
        if missing_cols:
            log_error(f"Missing columns in merged data: {missing_cols}")
        merged_df['price_retail'] = pd.to_numeric(merged_df['price_retail'], errors='coerce')
        merged_df['price_farmgate'] = pd.to_numeric(merged_df['price_farmgate'], errors='coerce')
        merged_df['gross_margin'] = pd.to_numeric(merged_df['gross_margin'], errors='coerce')
        merged_df['distance_km'] = pd.to_numeric(merged_df['distance_km'], errors='coerce')
        merged_df['year'] = pd.to_numeric(merged_df['year'], errors='coerce')
        merged_df['month'] = pd.to_numeric(merged_df['month'], errors='coerce')
        merged_df = merged_df.dropna(subset=['price_retail', 'price_farmgate', 'latitude', 'longitude', 'region_latitude', 'region_longitude', 'commodity_id', 'year', 'month'])
        if not merged_df.empty:
            merged_df = merged_df[merged_df['year'].between(2016, 2025)]
        return merged_df
    except Exception as e:
        log_error(f"Error reading merged data file {file_path}: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_all_models_and_scalers(commodity_configs, data_for_normalization_file_path):
    """Load all machine learning models and preprocessing objects."""
    all_models_and_scalers = {}
    features_for_imputation_and_normalization = [
        'travel_time_mean', 'friction_mean', 'vim', 'rfh',
        'market_population_5km', 'year', 'price_farmgate', 'distance_km'
    ]
    try:
        full_training_data_df = pd.read_excel(data_for_normalization_file_path)
        full_training_data_df['transport_cost'] = full_training_data_df['distance_km'] * Config.TRANSPORT_RATE_XOF_PER_KG_KM
    except Exception as e:
        log_error(f"Error loading training data for scalers: {e}")
        return {}
    for commodity_name, config in commodity_configs.items():
        rf_model, xgb_model, imputer = None, None, None
        min_max_values = {}
        commodity_id = config['id']
        output_dir = config['output_dir']
        rf_model_path = os.path.join(output_dir, config['rf_model_file'])
        xgb_model_path = os.path.join(output_dir, config['xgb_model_file'])
        spoilage_rate = config['spoilage_rate']
        try:
            rf_model = joblib.load(rf_model_path)
            st.success(f"Random Forest model for {commodity_name.capitalize()} loaded.")
        except Exception as e:
            log_error(f"Error loading Random Forest model for {commodity_name.capitalize()}: {e}")
        try:
            xgb_model = joblib.load(xgb_model_path)
            st.success(f"XGBoost model for {commodity_name.capitalize()} loaded.")
        except Exception as e:
            log_error(f"Error loading XGBoost model for {commodity_name.capitalize()}: {e}")
        try:
            commodity_data = full_training_data_df[full_training_data_df['commodity_id'] == commodity_id].copy()
            commodity_data['spoilage_cost'] = spoilage_rate * commodity_data['price_farmgate']
            commodity_data['transaction_cost_xof_per_kg'] = commodity_data['transport_cost'] + commodity_data['spoilage_cost']
            imputer = SimpleImputer(strategy='mean')
            cols_to_fit_imputer = [col for col in features_for_imputation_and_normalization if col in commodity_data.columns]
            if not commodity_data[cols_to_fit_imputer].empty:
                imputer.fit(commodity_data[cols_to_fit_imputer])
            else:
                log_error(f"No data for {commodity_name.capitalize()} to fit imputer.")
                imputer = None
            min_max_values = get_min_max_values(commodity_data, cols_to_fit_imputer)
            st.success(f"Preprocessing objects for {commodity_name.capitalize()} prepared.")
        except Exception as e:
            log_error(f"Error preparing preprocessing objects for {commodity_name.capitalize()}: {e}")
        all_models_and_scalers[commodity_name] = {
            'rf_model': rf_model,
            'xgb_model': xgb_model,
            'imputer': imputer,
            'min_max_values': min_max_values,
            'spoilage_rate': spoilage_rate
        }
    return all_models_and_scalers

# -------------------------------
# Raster Image Generation
# -------------------------------
@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash(tuple([x.data.tobytes(), x.mask.tobytes()]))})
def generate_travel_image(data, bounds):
    """Generate travel time raster image."""
    breaks = [0, 10, 30, 60, 120, 240, 1440, np.inf]
    colors = [(255, 255, 204), (255, 237, 160), (254, 178, 76), (253, 141, 60), (240, 59, 32), (189, 0, 38), (128, 0, 38)]
    rgb = generate_colors(data, breaks, colors)
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', dir=Config.TEMP_DIR, delete=False)
    Image.fromarray(rgb).save(temp_file.name)
    st.session_state['temp_files'].append(temp_file.name)
    atexit.register(cleanup_temp_files)
    return temp_file.name, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]], breaks, colors

@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash(tuple([x.data.tobytes(), x.mask.tobytes()]))})
def generate_friction_image(data, bounds):
    """Generate friction surface raster image."""
    breaks = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, np.inf]
    colors = [(0, 104, 55), (49, 163, 84), (120, 198, 121), (194, 230, 153), (253, 174, 97), (244, 109, 67), (165, 0, 38)]
    rgb = generate_colors(data, breaks, colors)
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', dir=Config.TEMP_DIR, delete=False)
    Image.fromarray(rgb).save(temp_file.name)
    st.session_state['temp_files'].append(temp_file.name)
    atexit.register(cleanup_temp_files)
    return temp_file.name, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]], breaks, colors

@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash(tuple([x.data.tobytes(), x.mask.tobytes()]))})
def generate_population_image(data, bounds):
    """Generate population density raster image."""
    if data is None:
        log_error("Population data is None.")
        return None, None, None, None
    breaks = [0, 1, 10, 50, 100, 500, 1000, np.inf]
    colors = [(200, 220, 255), (150, 180, 255), (100, 140, 255), (50, 100, 255), (0, 60, 200), (0, 40, 150), (0, 20, 100)]
    rgb = generate_colors(data, breaks, colors)
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', dir=Config.TEMP_DIR, delete=False)
    Image.fromarray(rgb).save(temp_file.name)
    st.session_state['temp_files'].append(temp_file.name)
    atexit.register(cleanup_temp_files)
    return temp_file.name, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]], breaks, colors

# -------------------------------
# Model Prediction Function
# -------------------------------
def predict_transaction_cost_commodity(commodity_key, input_data, models_and_scalers):
    """Predict transaction cost using Random Forest and XGBoost models."""
    if commodity_key not in models_and_scalers:
        st.error(f"Models for {commodity_key.capitalize()} not loaded.")
        return None, None, None
    commodity_info = models_and_scalers[commodity_key]
    rf_model, xgb_model, imputer, min_max_values, spoilage_rate = (
        commodity_info['rf_model'],
        commodity_info['xgb_model'],
        commodity_info['imputer'],
        commodity_info['min_max_values'],
        commodity_info['spoilage_rate']
    )
    if rf_model is None or xgb_model is None or imputer is None or not min_max_values:
        st.error(f"Models or preprocessing objects for {commodity_key.capitalize()} are incomplete.")
        return None, None, None
    input_df = pd.DataFrame([input_data])
    numerical_cols = [
        'travel_time_mean', 'friction_mean', 'vim', 'rfh',
        'market_population_5km', 'year', 'distance_km', 'price_farmgate'
    ]
    if imputer.feature_names_in_ is not None:
        for col in imputer.feature_names_in_:
            if col not in input_df.columns:
                input_df[col] = np.nan
        try:
            input_df[imputer.feature_names_in_] = imputer.transform(input_df[imputer.feature_names_in_])
        except Exception as e:
            log_error(f"Imputation error for {commodity_key.capitalize()}: {e}")
            return None, None, None
    for col in ['travel_time_mean', 'friction_mean', 'vim', 'rfh', 'market_population_5km', 'year']:
        norm_col_name = f'{col.replace("_mean", "")}_norm'
        if col in input_df.columns and col in min_max_values:
            min_val, max_val = min_max_values[col]['min'], min_max_values[col]['max']
            input_df[norm_col_name] = (input_df[col] - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0
        else:
            input_df[norm_col_name] = 0
    if 'month' in input_df.columns:
        input_df['month_sin'] = np.sin(2 * np.pi * input_df['month'] / 12)
        input_df['month_cos'] = np.cos(2 * np.pi * input_df['month'] / 12)
    else:
        input_df['month_sin'], input_df['month_cos'] = 0, 0
    features_for_prediction = [
        'travel_time_norm', 'friction_norm', 'vim_norm', 'rfh_norm',
        'market_population_5km_norm', 'year_norm', 'month_sin', 'month_cos'
    ]
    for feature in features_for_prediction:
        if feature not in input_df.columns:
            input_df[feature] = 0
    try:
        rf_prediction = rf_model.predict(input_df[features_for_prediction])[0]
        xgb_prediction = xgb_model.predict(input_df[features_for_prediction])[0]
        return rf_prediction, xgb_prediction, spoilage_rate
    except Exception as e:
        log_error(f"Prediction error for {commodity_key.capitalize()}: {e}")
        return None, None, None

# -------------------------------
# UI Components
# -------------------------------
def render_header():
    """Render the dashboard header."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    .main { background-color: #f9fafb; }
    .sidebar .sidebar-content { background-color: #ffffff; }
    .stButton>button { background-color: #1e3a8a; color: white; border-radius: 8px; padding: 10px; }
    .stSelectbox, .stMultiselect { background-color: #f3f4f6; border-radius: 8px; }
    .header { background-color: #1e3a8a; color: white; padding: 20px; border-radius: 8px; }
    .footer { background-color: #1e3a8a; color: white; padding: 10px; text-align: center; margin-top: 20px; }
    .folium-map { height: calc(100vh - 200px); max-height: 1000px; width: 100% !important; }
    .stApp [data-testid="stMapContainer"] { margin-top: 10px; width: 100% !important; max-height: 100vh; overflow: auto; }
    .legend-container { background-color: white; border: 1px solid grey; padding: 5px; box-shadow: 1px 1px 3px rgba(0,0,0,0.2); margin-top: 5px; font-size: 10px; }
    .legend-title { font-weight: bold; font-size: 10px; margin-bottom: 2px; }
    .legend-item { display: flex; align-items: center; margin-bottom: 2px; font-size: 10px; }
    .legend-color { width: 10px; height: 10px; margin-right: 4px; display: inline-block; }
    @media (max-width: 600px) { 
        .folium-map { height: 50vh; }
        .legend-container { font-size: 8px; padding: 3px; }
        .legend-color { width: 8px; height: 8px; }
        .header { padding: 10px; }
        .header img { width: 100px; }
        .header h1 { font-size: 1.5em; }
        .header p { font-size: 0.9em; }
    }
    </style>
    <div class="header" role="banner" aria-label="Dashboard Header">
        <img src="https://www.ifpri.org/themes/custom/ifpri/logo.svg" alt="IFPRI Logo" width="150">
        <h1>Senegal Agricultural Market Dashboard</h1>
        <p>Developed in collaboration with the International Food Policy Research Institute (IFPRI)</p>
    </div>
    """, unsafe_allow_html=True)

def render_geospatial_tab():
    """Render the Geospatial Analysis tab."""
    st.header("Geospatial Analysis")
    st.markdown("Explore geospatial layers including travel time, friction surfaces, market locations, and roads.")
    
    # Validate files
    for key, path in st.session_state.file_paths.items():
        if not validate_file(path, key.replace('_', ' ').title()):
            return
    
    # Load data with progress bar
    with st.spinner("Loading geospatial data..."):
        progress = st.progress(0)
        travel_time, travel_bounds = load_and_process_raster(st.session_state.file_paths['raster'])
        progress.progress(0.2)
        friction_data, friction_bounds = load_and_process_raster(st.session_state.file_paths['friction'])
        progress.progress(0.4)
        pop_key = f'population_{min(st.session_state["selected_year"], 2020)}'
        population_data, population_bounds = load_and_process_raster(st.session_state.file_paths[pop_key])
        progress.progress(0.6)
        markets = load_geojson(st.session_state.file_paths['markets'], population_file=st.session_state.file_paths[pop_key])
        progress.progress(0.8)
        roads_filtered = load_geojson(st.session_state.file_paths['roads'], max_features=500, is_roads=True)
        progress.progress(1.0)
        st.success("Geospatial data loaded successfully!")
    
    # Year selection for population data
    available_years = [2016, 2017, 2018, 2019, 2020]
    st.session_state['selected_year'] = st.selectbox("Select Population Year", available_years, index=available_years.index(st.session_state['selected_year']))
    
    geospatial_layers = {
        "Travel Time": {"data": travel_time, "bounds": travel_bounds, "generator": generate_travel_image, "title": "Travel Time (Hours)"},
        "Friction Surface": {"data": friction_data, "bounds": friction_bounds, "generator": generate_friction_image, "title": "Friction Surface"},
        "Population Density": {"data": population_data, "bounds": population_bounds, "generator": generate_population_image, "title": f"Population Density ({st.session_state['selected_year']})"}
    }
    selected_layer_name = st.selectbox("Select Geospatial Layer:", list(geospatial_layers.keys()), key='geospatial_layer_select')
    selected_layer = geospatial_layers[selected_layer_name]
    
    if selected_layer["data"] is None or selected_layer["bounds"] is None:
        st.warning(f"Failed to load data for {selected_layer_name}. Cannot display map.")
        return
    
    image_path, overlay_bounds, breaks, colors = selected_layer["generator"](selected_layer["data"], selected_layer["bounds"])
    m = folium.Map(location=[14.4974, -14.4524], zoom_start=7, control_scale=True, tiles="CartoDB positron")
    
    if image_path:
        folium.raster_layers.ImageOverlay(
            image=image_path,
            bounds=overlay_bounds,
            opacity=0.7,
            alt=selected_layer_name
        ).add_to(m)
    
    if roads_filtered:
        folium.GeoJson(
            roads_filtered,
            name='Road Network',
            style_function=lambda x: {'color': '#666666', 'weight': 1.5, 'opacity': 0.7}
        ).add_to(m)
    
    if markets:
        market_cluster = MarkerCluster(name='Markets').add_to(m)
        for feature in markets['features']:
            lat, lon = feature['geometry']['coordinates'][1], feature['geometry']['coordinates'][0]
            props = feature['properties']
            popup_html = f"<b>Market:</b> {props.get('name', 'N/A')}<br>" \
                         f"<b>Type:</b> {props.get('type', 'N/A')}<br>" \
                         f"<b>Admin Level:</b> {props.get('admin_level', 'N/A')}<br>" \
                         f"<b>Population (5km):</b> {props.get('population_5km', 'N/A'):,.0f}"
            folium.Marker([lat, lon], popup=popup_html).add_to(market_cluster)
    
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    folium.LayerControl().add_to(m)
    
    if st.session_state['show_legends'] and breaks and colors:
        st.markdown(generate_legend_html(breaks, colors, selected_layer["title"]), unsafe_allow_html=True)
    
    st_folium(m, height=st.session_state['map_height'], width=700, key=f"folium_map_{st.session_state['map_render_key']}")
    st.info("Geospatial layers are downsampled for performance. Contact support for high-resolution data.")

def render_price_trends_tab():
    """Render the Price Trends tab."""
    st.header("Commodity Price and Margin Trends")
    st.markdown("Analyze historical price data and gross margins for different commodities.")
    
    merged_data_path = st.session_state.file_paths.get('prices')
    if not merged_data_path or not os.path.exists(merged_data_path):
        st.warning("Merged data file not found. Please upload 'merged_farmgate_retail_prices_senegal.xlsx' in the Data & File Management tab.")
        return
    
    if st.session_state['latest_merged_prices'].empty:
        st.session_state['latest_merged_prices'] = load_merged_data(merged_data_path)
    
    if not st.session_state['latest_merged_prices'].empty:
        df_trends = st.session_state['latest_merged_prices'].copy()
        df_trends['date'] = pd.to_datetime(df_trends[['year', 'month']].assign(day=1))
        commodity_id_to_name = df_trends.set_index('commodity_id')['commodity_english'].to_dict()
        unique_commodity_ids = sorted(df_trends['commodity_id'].unique())
        commodity_options = {commodity_id_to_name.get(cid, f"ID: {cid}"): cid for cid in unique_commodity_ids}
        
        selected_commodity_name = st.selectbox(
            "Select Commodity for Price Trends",
            options=list(commodity_options.keys()),
            format_func=lambda x: x,
            key="commodity_trend_select"
        )
        selected_commodity_id = commodity_options[selected_commodity_name]
        
        df_filtered = df_trends[df_trends['commodity_id'] == selected_commodity_id]
        if not df_filtered.empty:
            df_agg = df_filtered.groupby('date').agg(
                avg_price_farmgate=('price_farmgate', 'mean'),
                avg_price_retail=('price_retail', 'mean'),
                avg_gross_margin=('gross_margin', 'mean')
            ).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_agg['date'],
                y=df_agg['avg_price_farmgate'],
                mode='lines+markers',
                name='Average Farmgate Price',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6),
                hovertemplate='%{x|%b %Y}<br>Farmgate Price: %{y:.2f} XOF/KG'
            ))
            fig.add_trace(go.Scatter(
                x=df_agg['date'],
                y=df_agg['avg_price_retail'],
                mode='lines+markers',
                name='Average Retail Price',
                line=dict(color='#2ca02c', width=2),
                marker=dict(size=6),
                hovertemplate='%{x|%b %Y}<br>Retail Price: %{y:.2f} XOF/KG'
            ))
            fig.add_trace(go.Scatter(
                x=df_agg['date'],
                y=df_agg['avg_gross_margin'],
                mode='lines+markers',
                name='Average Gross Margin',
                line=dict(color='#d62728', width=2, dash='dash'),
                marker=dict(size=6),
                hovertemplate='%{x|%b %Y}<br>Margin: %{y:.2f} XOF/KG'
            ))
            fig.update_layout(
                title=f"{commodity_id_to_name.get(selected_commodity_id, selected_commodity_name)} Price and Margin Trends",
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
        else:
            st.warning("No data available for the selected commodity.")
    else:
        st.warning("No price trend data available. Please upload the required Excel file.")

def render_predict_tab():
    """Render the Predict Transaction Cost tab."""
    st.header("Predict Transaction Cost")
    st.markdown("Use machine learning models to predict transaction costs based on input features.")
    
    commodity_options = list(Config.COMMODITY_MODELS_CONFIG.keys())
    selected_commodity = st.selectbox(
        "Select Commodity for Prediction",
        options=commodity_options,
        format_func=lambda x: x.capitalize(),
        key="commodity_predict_select"
    )
    
    current_config = Config.COMMODITY_MODELS_CONFIG[selected_commodity]
    spoilage_rate = current_config['spoilage_rate']
    
    if not st.session_state['models_and_scalers'] or selected_commodity not in st.session_state['models_and_scalers']:
        st.warning(f"Models for {selected_commodity.capitalize()} are not loaded. Ensure model files exist.")
        return
    
    st.subheader("Input Features")
    with st.expander("Feature Descriptions", expanded=False):
        st.markdown("""
        - **Average Travel Time**: Travel time to the market in hours.
        - **Average Friction**: Travel speed factor (hours per cell).
        - **VIM**: Vegetation Index Measure (-1 to 1).
        - **RFH**: Rainfall Index Measure (0 to 1).
        - **Market Population**: Population within 5km of the market.
        - **Year/Month**: Time period for the prediction.
        - **Distance**: Distance to the market in kilometers.
        - **Farmgate Price**: Price at the farmgate in XOF/KG.
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        travel_time_mean = st.number_input("Average Travel Time (hours)", min_value=0.0, value=5.0, step=0.1, help="Enter the average travel time to the market in hours.")
        friction_mean = st.number_input("Average Friction", min_value=0.0, value=0.5, step=0.01, help="Travel speed factor in hours per cell.")
        vim = st.number_input("VIM (Vegetation Index)", min_value=-1.0, max_value=1.0, value=0.7, step=0.01, help="Vegetation Index Measure (-1 to 1).")
        rfh = st.number_input("RFH (Rainfall Index)", min_value=0.0, max_value=1.0, value=0.6, step=0.01, help="Rainfall Index Measure (0 to 1).")
        market_population_5km = st.number_input("Market Population (5km)", min_value=0, value=10000, step=100, help="Population within 5km radius of the market.")
    with col2:
        current_year = st.number_input("Year", min_value=2016, max_value=2025, value=2024, step=1, help="Year for the prediction.")
        current_month = st.slider("Month", min_value=1, max_value=12, value=7, step=1, help="Month for the prediction.")
        distance_km = st.number_input("Distance (km)", min_value=0.0, value=50.0, step=1.0, help="Distance to the market in kilometers.")
        price_farmgate = st.number_input("Farmgate Price (XOF/KG)", min_value=0.0, value=200.0, step=10.0, help="Farmgate price in XOF per kilogram.")
    
    st.info(f"Spoilage rate for {selected_commodity.capitalize()}: {spoilage_rate * 100:.1f}%.")
    
    if st.button("Predict Transaction Cost"):
        input_data = {
            'travel_time_mean': travel_time_mean,
            'friction_mean': friction_mean,
            'vim': vim,
            'rfh': rfh,
            'market_population_5km': market_population_5km,
            'year': current_year,
            'month': current_month,
            'distance_km': distance_km,
            'price_farmgate': price_farmgate
        }
        rf_pred, xgb_pred, actual_spoilage_rate = predict_transaction_cost_commodity(selected_commodity, input_data, st.session_state['models_and_scalers'])
        
        if rf_pred is not None and xgb_pred is not None:
            st.subheader("Prediction Results")
            st.write(f"**Random Forest Prediction**: {rf_pred:.2f} XOF/KG")
            st.write(f"**XGBoost Prediction**: {xgb_pred:.2f} XOF/KG")
            estimated_transport_cost = distance_km * Config.TRANSPORT_RATE_XOF_PER_KG_KM
            estimated_spoilage_cost = actual_spoilage_rate * price_farmgate
            total_estimated_cost = estimated_transport_cost + estimated_spoilage_cost
            st.write(f"*(Formula-based cost: Transport ({estimated_transport_cost:.2f} XOF/kg) + Spoilage ({estimated_spoilage_cost:.2f} XOF/kg) = **{total_estimated_cost:.2f} XOF/kg**)*")
            st.markdown("""
                <p style='font-size:14px; color:grey;'>
                <i>Note: Model predictions incorporate learned patterns from data, which may differ from the formula-based estimate.</i>
                </p>
            """, unsafe_allow_html=True)
        else:
            st.error("Prediction failed. Check inputs and model availability.")

def render_model_insights_tab():
    """Render the Model Insights tab."""
    st.header("Model Insights and Performance")
    st.markdown("Review model performance metrics and feature importance.")
    
    commodity_options = list(Config.COMMODITY_MODELS_CONFIG.keys())
    selected_commodity = st.selectbox(
        "Select Commodity for Model Insights",
        options=commodity_options,
        format_func=lambda x: x.capitalize(),
        key="commodity_insights_select"
    )
    
    current_config = Config.COMMODITY_MODELS_CONFIG[selected_commodity]
    model_comparison_csv_path = os.path.join(current_config['output_dir'], current_config['model_comparison_csv'])
    feature_importance_excel_path = os.path.join(current_config['output_dir'], current_config['feature_importance_excel'])
    
    st.subheader(f"Model Performance ({selected_commodity.capitalize()} - Test Set)")
    try:
        model_comparison_df = pd.read_csv(model_comparison_csv_path)
        st.dataframe(model_comparison_df[['Model', 'R2_Test', 'MAE_Test', 'RMSE_Test', 'MAPE_Test', 'MedAE_Test']].round(2))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['MAE_Test', 'RMSE_Test', 'MAPE_Test']
        models = ['Random Forest', 'XGBoost']
        x = np.arange(len(metrics))
        width = 0.35
        rf_vals = [model_comparison_df[model_comparison_df['Model'] == 'Random Forest'][m].iloc[0] for m in metrics]
        xgb_vals = [model_comparison_df[model_comparison_df['Model'] == 'XGBoost'][m].iloc[0] for m in metrics]
        ax.bar(x - width/2, rf_vals, width, label='Random Forest', color='#1f77b4', alpha=0.8)
        ax.bar(x + width/2, xgb_vals, width, label='XGBoost', color='#ff7f0e', alpha=0.8)
        ax.set_ylabel('Metric Value')
        ax.set_title(f'Model Performance ({selected_commodity.capitalize()})')
        ax.set_xticks(x)
        ax.set_xticklabels(['MAE', 'RMSE', 'MAPE (%)'])
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        log_error(f"Error loading model comparison for {selected_commodity.capitalize()}: {e}")
    
    st.subheader(f"Feature Importance ({selected_commodity.capitalize()})")
    try:
        feature_importance_df = pd.read_excel(feature_importance_excel_path)
        st.dataframe(feature_importance_df)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        index = np.arange(len(feature_importance_df))
        ax.bar(index, feature_importance_df['Random_Forest_Importance'], bar_width, label='Random Forest', color='#1f77b4', alpha=0.8)
        ax.bar(index + bar_width, feature_importance_df['XGBoost_Importance'], bar_width, label='XGBoost', color='#ff7f0e', alpha=0.8)
        ax.set_xlabel('Features')
        ax.set_ylabel('Feature Importance')
        ax.set_title(f'Feature Importance ({selected_commodity.capitalize()})')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(feature_importance_df['Feature'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        log_error(f"Error loading feature importance for {selected_commodity.capitalize()}: {e}")

def render_file_management_tab():
    """Render the Data & File Management tab."""
    st.header("Data and File Management")
    st.markdown("Upload new data files to update the dashboard.")
    
    with st.expander("Current File Paths", expanded=True):
        for key, path in st.session_state.file_paths.items():
            st.write(f"- **{key.replace('_', ' ').title()}:** `{path}`")
    
    with st.expander("Upload New Files", expanded=False):
        uploaded_files = {
            'raster': st.file_uploader("Travel Time Raster (.tiff)", type=["tif", "tiff"], key="upload_raster"),
            'friction': st.file_uploader("Friction Surface Raster (.tiff)", type=["tif", "tiff"], key="upload_friction"),
            'markets': st.file_uploader("Markets GeoJSON (.geojson)", type=["geojson"], key="upload_markets"),
            'roads': st.file_uploader("Roads GeoJSON (.geojson)", type=["geojson"], key="upload_roads"),
            'prices': st.file_uploader("Merged Prices Excel (.xlsx)", type=["xlsx"], key="upload_prices"),
        }
        uploaded_population = st.file_uploader("Population Raster (2016-2020, .tif)", type=["tif", "tiff"], accept_multiple_files=True, key="upload_population")
        
        for key, uploaded_file in uploaded_files.items():
            if uploaded_file:
                temp_file_path = os.path.join(Config.TEMP_DIR, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state['file_paths'][key] = temp_file_path
                st.session_state['temp_files'].append(temp_file_path)
                st.success(f"{key.replace('_', ' ').title()} uploaded: {temp_file_path}")
                st.session_state['map_render_key'] += 1
                st.session_state['map_data_updated'] = True
                if key == 'prices':
                    st.session_state['file_paths']['merged_data_for_models'] = temp_file_path
                    st.session_state['latest_farmgate_prices'] = pd.DataFrame()
                    st.session_state['latest_retail_prices'] = pd.DataFrame()
                    st.session_state['latest_merged_prices'] = pd.DataFrame()
                    st.session_state['models_and_scalers'] = {}
                    st.rerun()
        
        if uploaded_population:
            for uploaded_file in uploaded_population:
                year_match = re.search(r'(\d{4})', uploaded_file.name)
                if year_match:
                    year = year_match.group(1)
                    pop_key = f'population_{year}'
                    temp_file_path = os.path.join(Config.TEMP_DIR, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state['file_paths'][pop_key] = temp_file_path
                    st.session_state['temp_files'].append(temp_file_path)
                    st.success(f"Population raster for {year} uploaded: {temp_file_path}")
                    st.session_state['map_render_key'] += 1
                    st.session_state['map_data_updated'] = True
                else:
                    st.warning(f"Could not extract year from: {uploaded_file.name}.")
    
    with st.expander("Utility Actions", expanded=False):
        if st.button("Clear Cache & Reload Data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state['map_data_updated'] = False
            st.session_state['latest_farmgate_prices'] = pd.DataFrame()
            st.session_state['latest_retail_prices'] = pd.DataFrame()
            st.session_state['latest_merged_prices'] = pd.DataFrame()
            st.session_state['models_and_scalers'] = {}
            st.session_state['file_paths'] = Config.DEFAULT_FILES.copy()
            cleanup_temp_files()
            st.rerun()
        st.checkbox("Show Legends on Map", value=st.session_state['show_legends'], key='show_legends_checkbox', on_change=lambda: st.session_state.update(show_legends=st.session_state['show_legends_checkbox']))

# -------------------------------
# Main Dashboard
# -------------------------------
def main():
    """Main function to run the Streamlit dashboard."""
    Config.ensure_temp_dir()
    init_session_state()
    render_header()
    st.markdown("""
    ### Welcome to the Senegal Agricultural Market Dashboard
    This tool visualizes travel time, friction surfaces, market locations, road networks, commodity prices, and population density in Senegal.
    """)
    
    if not st.session_state['models_and_scalers']:
        with st.spinner("Loading machine learning models..."):
            st.session_state['models_and_scalers'] = load_all_models_and_scalers(Config.COMMODITY_MODELS_CONFIG, Config.DEFAULT_FILES['merged_data_for_models'])
        if st.session_state['models_and_scalers']:
            st.success("Models loaded successfully!")
    
    tabs = st.tabs(["Geospatial Analysis", "Price Trends", "Predict Transaction Cost", "Model Insights", "Data & File Management"])
    with tabs[0]:
        render_geospatial_tab()
    with tabs[1]:
        render_price_trends_tab()
    with tabs[2]:
        render_predict_tab()
    with tabs[3]:
        render_model_insights_tab()
    with tabs[4]:
        render_file_management_tab()
    
    if st.session_state['errors']:
        with st.expander("Errors Encountered", expanded=True):
            for error in st.session_state['errors']:
                st.error(error)
        st.session_state['errors'] = []
    
    st.markdown("""
    <div class="footer">
        <p>Developed by xAI in collaboration with IFPRI | Data Sources: IFPRI, OpenStreetMap, WorldPop | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()