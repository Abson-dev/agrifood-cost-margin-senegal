import os
import json
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import joblib # Added for model loading
from sklearn.preprocessing import MinMaxScaler # Added for normalization
from sklearn.impute import SimpleImputer # Added for imputation
from datetime import datetime # Added for date handling
import re # Added for regex in file parsing
import sys

# -------------------------------
# Configuration
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
    'population_2020': os.path.join(BASE_DIR, 'sen_ppp_2020_UNadj.tif'),
    'merged_data_for_models': os.path.join(BASE_DIR, 'final_merged_output_senegal.xlsx') # Path to the merged data for model training
}

# Model-specific configurations
COMMODITY_MODELS_CONFIG = {
    'sorghum': {
        'id': 65, # Sorghum commodity ID
        'output_dir': os.path.join(BASE_DIR, 'sorghum_analysis_outputs'),
        'rf_model_file': 'rf_model_sorghum.pkl',
        'xgb_model_file': 'xgb_model_sorghum.pkl',
        'spoilage_rate': 0.116,
        'model_comparison_csv': 'model_comparison_sorghum.csv',
        'feature_importance_excel': 'feature_importance_sorghum.xlsx'
    },
    'maize': {
        'id': 56, # Maize commodity ID
        'output_dir': os.path.join(BASE_DIR, 'maize_analysis_outputs'),
        'rf_model_file': 'rf_model_maize.pkl',
        'xgb_model_file': 'xgb_model_maize.pkl',
        'spoilage_rate': 0.1,
        'model_comparison_csv': 'model_comparison_maize.csv',
        'feature_importance_excel': 'feature_importance_maize.xlsx'
    }
}
TRANSPORT_RATE_XOF_PER_KG_KM = 0.3

# Initialize session state
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
if 'models_and_scalers' not in st.session_state: # Store models and scalers per commodity
    st.session_state['models_and_scalers'] = {}

# -------------------------------
# Helper Functions
# -------------------------------
def log_error(message):
    """Log errors to session state for display."""
    st.session_state['errors'].append(message)
    st.error(message)

def ensure_default_files():
    """Check if default files exist."""
    for key, path in DEFAULT_FILES.items():
        if not os.path.exists(path):
            log_error(f"Default {key} file not found: {path}")
            return False
    return True

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
            os.remove(file)
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
    x = data[date_col].map(lambda x: x.timestamp())
    y = data[value_col]
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
    new_dates = pd.date_range(data[date_col].min(), data[date_col].max(), freq='M')
    new_values = f([d.timestamp() for d in new_dates])
    return pd.DataFrame({'Date': new_dates, value_col: new_values})

def generate_legend_html(breaks, colors, title):
    """Generate tiny legend HTML."""
    html = f'<div class="legend-container"><div class="legend-title">{title}</div>'
    for i, (b1, b2) in enumerate(zip(breaks[:-1], breaks[1:])):
        color = f'rgb{colors[i]}'
        label = f'{b1:.0f}–{b2:.0f}' if b2 != np.inf else f'>{b1:.0f}'
        html += f'<div class="legend-item"><span class="legend-color" style="background:{color};"></span> {label}</div>'
    html += '</div>'
    return html

def validate_commodity_overlap(farmgate_df, retail_df):
    """Validate commodity ID overlap."""
    if farmgate_df.empty or retail_df.empty:
        return set()
    common_ids = set(farmgate_df['commodity_id']).intersection(set(retail_df['commodity_id']))
    return common_ids

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
        return population_sums
    except Exception as e:
        log_error(f"Error computing population buffers: {e}")
        return {}

# Function to get min/max for normalization dynamically from the loaded data
def get_min_max_values(df, features_to_normalize):
    min_max_vals = {}
    for col in features_to_normalize:
        if col in df.columns and df[col].std() > 0: # Ensure there is variance
            min_max_vals[col] = {'min': df[col].min(), 'max': df[col].max()}
        elif col in df.columns: # If no variance, min and max are the same
             min_max_vals[col] = {'min': df[col].iloc[0], 'max': df[col].iloc[0]}
        else:
            log_error(f"Column '{col}' not found in DataFrame for min/max calculation.")
            min_max_vals[col] = {'min': 0, 'max': 1} # Default fallback
    return min_max_vals

# -------------------------------
# Data Loading Functions
# -------------------------------
@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash(tuple([x.data.tobytes(), x.mask.tobytes()]))})
def load_and_process_raster(file_path, downsample_factor=2):
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
    try:
        gdf = gpd.read_file(file_path)
        if gdf.empty:
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
        
        geojson_data = json.loads(gdf.to_json())
        return geojson_data if geojson_data['features'] else None
    except Exception as e:
        log_error(f"Failed to load {file_path}: {e}")
        return None

@st.cache_data
def load_price_data(file_path):
    try:
        prices_df = pd.read_excel(file_path, sheet_name='Farmgate prices Senegal')
        if prices_df.empty:
            return pd.DataFrame()
        farmgate_required_columns = ['Régions Name', 'Commodity', 'commodity_english', 'commodity_id', 'Price', 'Unit2', 'Régions - Latitude', 'Régions - Longitude', 'Year', 'Month']
        missing_cols = [col for col in farmgate_required_columns if col not in prices_df.columns]
        prices_df['Price'] = pd.to_numeric(prices_df['Price'], errors='coerce')
        prices_df['Year'] = pd.to_numeric(prices_df['Year'], errors='coerce')
        prices_df['Month'] = pd.to_numeric(prices_df['Month'], errors='coerce')
        prices_df = prices_df.dropna(subset=['Price', 'Year', 'Month', 'Régions - Latitude', 'Régions - Longitude', 'commodity_id'])
        if not prices_df.empty and (prices_df['Year'] < 2016).any() or (prices_df['Year'] > 2025).any():
            prices_df = prices_df[prices_df['Year'].between(2016, 2025)]
        return prices_df
    except Exception as e:
        log_error(f"Error reading farmgate prices file {file_path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_retail_data(file_path):
    try:
        retail_df = pd.read_excel(file_path, sheet_name='Retails Price Senegal')
        if retail_df.empty:
            return pd.DataFrame()
        retail_required_columns = ['market', 'commodity', 'commodity_id', 'Price', 'Unit2', 'latitude', 'longitude', 'Year', 'Month']
        missing_cols = [col for col in retail_required_columns if col not in retail_df.columns]
        retail_df = retail_df.rename(columns={'price': 'Price'})
        retail_df['Price'] = pd.to_numeric(retail_df['Price'], errors='coerce')
        retail_df['Year'] = pd.to_numeric(retail_df['Year'], errors='coerce')
        retail_df['Month'] = pd.to_numeric(retail_df['Month'], errors='coerce')
        retail_df = retail_df.dropna(subset=['Price', 'Year', 'Month', 'latitude', 'longitude', 'commodity_id'])
        if not retail_df.empty and (retail_df['Year'] < 2016).any() or (retail_df['Year'] > 2025).any():
            retail_df = retail_df[retail_df['Year'].between(2016, 2025)]
        return retail_df
    except Exception as e:
        log_error(f"Error reading retail prices file {file_path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_merged_data(file_path):
    try:
        merged_df = pd.read_excel(file_path, sheet_name='merged_data')
        if merged_df.empty:
            return pd.DataFrame()
        required_columns = ['date', 'market', 'market_id', 'latitude', 'longitude', 'commodity_retail', 'commodity_id',
                           'price_retail', 'unit2_retail', 'year', 'month', 'commodity_farmgate', 'region_name',
                           'region_latitude', 'region_longitude', 'price_farmgate', 'unit2_farmgate', 'commodity_english',
                           'gross_margin', 'distance_km']
        missing_cols = [col for col in required_columns if col not in merged_df.columns]
        merged_df['price_retail'] = pd.to_numeric(merged_df['price_retail'], errors='coerce')
        merged_df['price_farmgate'] = pd.to_numeric(merged_df['price_farmgate'], errors='coerce')
        merged_df['gross_margin'] = pd.to_numeric(merged_df['gross_margin'], errors='coerce')
        merged_df['distance_km'] = pd.to_numeric(merged_df['distance_km'], errors='coerce')
        merged_df['year'] = pd.to_numeric(merged_df['year'], errors='coerce')
        merged_df['month'] = pd.to_numeric(merged_df['month'], errors='coerce')
        merged_df = merged_df.dropna(subset=['price_retail', 'price_farmgate', 'latitude', 'longitude', 'region_latitude', 'region_longitude', 'commodity_id', 'year', 'month'])
        if not merged_df.empty and (merged_df['year'] < 2016).any() or (merged_df['year'] > 2025).any():
            merged_df = merged_df[merged_df['year'].between(2016, 2025)]
        return merged_df
    except Exception as e:
        log_error(f"Error reading merged data file {file_path}: {e}")
        return pd.DataFrame()

@st.cache_resource # Use st.cache_resource for models and scalers
def load_all_models_and_scalers(commodity_configs, data_for_normalization_path):
    all_models_and_scalers = {}
    
    # Define the numerical features that were used for training and normalization
    # These are the *original* feature names before normalization or encoding
    features_for_imputation_and_normalization = [
        'travel_time_mean', 'friction_mean', 'vim', 'rfh',
        'market_population_5km', 'year', 'price_farmgate', 'distance_km'
    ]

    try:
        # Load the full data used for training to fit the imputer and get min/max values
        full_training_data_df = pd.read_excel(data_for_normalization_path)
        
        # Pre-calculate transaction costs for the entire dataset if needed for any feature calculation
        full_training_data_df['transport_cost'] = full_training_data_df['distance_km'] * TRANSPORT_RATE_XOF_PER_KG_KM
        
        # Fit a global imputer if needed, or fit per commodity later
        # For now, we'll fit imputer and scalers per commodity to ensure consistency with analysis scripts
        
    except FileNotFoundError:
        log_error(f"Data for normalization (e.g., {data_for_normalization_path}) not found. Cannot load models/scalers.")
        return {}
    except Exception as e:
        log_error(f"Error loading full training data for scalers: {e}")
        return {}

    for commodity_name, config in commodity_configs.items():
        rf_model = None
        xgb_model = None
        imputer = None
        min_max_values = {}
        
        commodity_id = config['id']
        output_dir = config['output_dir']
        rf_model_path = os.path.join(output_dir, config['rf_model_file'])
        xgb_model_path = os.path.join(output_dir, config['xgb_model_file'])
        spoilage_rate = config['spoilage_rate']

        st.write(f"Attempting to load models for {commodity_name.capitalize()}...")
        
        try:
            rf_model = joblib.load(rf_model_path)
            st.success(f"Random Forest model for {commodity_name.capitalize()} loaded from {rf_model_path}")
        except FileNotFoundError:
            log_error(f"Random Forest model file for {commodity_name.capitalize()} not found at {rf_model_path}. Please ensure models are trained and saved.")
        except Exception as e:
            log_error(f"Error loading Random Forest model for {commodity_name.capitalize()}: {e}")

        try:
            xgb_model = joblib.load(xgb_model_path)
            st.success(f"XGBoost model for {commodity_name.capitalize()} loaded from {xgb_model_path}")
        except FileNotFoundError:
            log_error(f"XGBoost model file for {commodity_name.capitalize()} not found at {xgb_model_path}. Please ensure models are trained and saved.")
        except Exception as e:
            log_error(f"Error loading XGBoost model for {commodity_name.capitalize()}: {e}")

        # Prepare imputer and min/max values for this specific commodity
        try:
            commodity_data_for_scalers = full_training_data_df[full_training_data_df['commodity_id'] == commodity_id].copy()
            
            # Calculate spoilage cost and transaction cost for this commodity's data for scaler fitting
            commodity_data_for_scalers['spoilage_cost'] = spoilage_rate * commodity_data_for_scalers['price_farmgate']
            commodity_data_for_scalers['transaction_cost_xof_per_kg'] = commodity_data_for_scalers['transport_cost'] + commodity_data_for_scalers['spoilage_cost']

            # Imputer for this commodity
            imputer = SimpleImputer(strategy='mean')
            # Fit imputer only on the relevant numerical columns that might have NaNs
            cols_to_fit_imputer = [col for col in features_for_imputation_and_normalization if col in commodity_data_for_scalers.columns]
            if not commodity_data_for_scalers[cols_to_fit_imputer].empty:
                imputer.fit(commodity_data_for_scalers[cols_to_fit_imputer])
            else:
                log_error(f"No data available for {commodity_name.capitalize()} to fit imputer. Imputer will not be available for this commodity.")
                imputer = None # Ensure imputer is None if no data

            # Min/Max values for normalization
            min_max_values = get_min_max_values(commodity_data_for_scalers, features_to_normalize=cols_to_fit_imputer)
            
            st.success(f"Preprocessing objects (imputer, min/max values) for {commodity_name.capitalize()} prepared.")

        except Exception as e:
            log_error(f"Error preparing preprocessing objects for {commodity_name.capitalize()}: {e}")
            imputer = None
            min_max_values = {}

        all_models_and_scalers[commodity_name] = {
            'rf_model': rf_model,
            'xgb_model': xgb_model,
            'imputer': imputer,
            'min_max_values': min_max_values,
            'spoilage_rate': spoilage_rate # Store spoilage rate with the commodity config
        }
    return all_models_and_scalers

# -------------------------------
# Raster Image Generation
# -------------------------------
@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash(tuple([x.data.tobytes(), x.mask.tobytes()]))})
def generate_travel_image(data, bounds):
    breaks = [0, 10, 30, 60, 120, 240, 1440, np.inf]
    colors = [(255, 255, 204), (255, 237, 160), (254, 178, 76), (253, 141, 60), (240, 59, 32), (189, 0, 38), (128, 0, 38)]
    rgb = generate_colors(data, breaks, colors)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        travel_png_path = tmp.name
        Image.fromarray(rgb).save(travel_png_path)
        st.session_state['temp_files'].append(travel_png_path)
    atexit.register(cleanup_temp_files)
    return travel_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]], breaks, colors

@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash(tuple([x.data.tobytes(), x.mask.tobytes()]))})
def generate_friction_image(data, bounds):
    friction_breaks = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, np.inf]
    friction_colors = [(0, 104, 55), (49, 163, 84), (120, 198, 121), (194, 230, 153), (253, 174, 97), (244, 109, 67), (165, 0, 38)]
    rgb = generate_colors(data, friction_breaks, friction_colors)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        friction_png_path = tmp.name
        Image.fromarray(rgb).save(friction_png_path)
        st.session_state['temp_files'].append(friction_png_path)
    atexit.register(cleanup_temp_files)
    return friction_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]], friction_breaks, friction_colors

@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash(tuple([x.data.tobytes(), x.mask.tobytes()]))})
def generate_population_image(data, bounds):
    if data is None:
        log_error("Population data is None. Check the input file.")
        return None, None, None, None
    breaks = [0, 1, 10, 50, 100, 500, 1000, np.inf]
    colors = [(200, 220, 255), (150, 180, 255), (100, 140, 255), (50, 100, 255), (0, 60, 200), (0, 40, 150), (0, 20, 100)]
    rgb = generate_colors(data, breaks, colors)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        population_png_path = tmp.name
        Image.fromarray(rgb).save(population_png_path)
        st.session_state['temp_files'].append(population_png_path)
    atexit.register(cleanup_temp_files)
    return population_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]], breaks, colors

# -------------------------------
# Model Prediction Function
# -------------------------------
def predict_transaction_cost_commodity(commodity_key, input_data, models_and_scalers):
    if commodity_key not in models_and_scalers:
        st.error(f"Models and scalers for {commodity_key.capitalize()} not loaded.")
        return None, None

    commodity_info = models_and_scalers[commodity_key]
    rf_model = commodity_info['rf_model']
    xgb_model = commodity_info['xgb_model']
    imputer = commodity_info['imputer']
    min_max_values = commodity_info['min_max_values']
    spoilage_rate = commodity_info['spoilage_rate']

    if rf_model is None or xgb_model is None or imputer is None or not min_max_values:
        st.error(f"Models or preprocessing objects for {commodity_key.capitalize()} are not fully loaded. Cannot make predictions.")
        return None, None

    # Create a DataFrame from input data
    input_df = pd.DataFrame([input_data])

    # Apply imputation for original numerical features first
    numerical_cols_for_imputation = [
        'travel_time_mean', 'friction_mean', 'vim', 'rfh',
        'market_population_5km', 'year', 'price_farmgate', 'distance_km'
    ]
    
    # Filter to only columns that exist in input_df and are numerical
    cols_to_impute_in_input = [col for col in numerical_cols_for_imputation if col in input_df.columns]
    
    # Ensure all columns expected by imputer are present, fill with NaN if missing temporarily
    for col in imputer.feature_names_in_: # Use feature_names_in_ from fitted imputer
        if col not in input_df.columns:
            input_df[col] = np.nan # Add missing columns with NaN for imputation

    # Apply imputation
    if not input_df[imputer.feature_names_in_].empty:
        input_df[imputer.feature_names_in_] = imputer.transform(input_df[imputer.feature_names_in_])
    else:
        st.warning(f"Input DataFrame is empty for imputation for {commodity_key.capitalize()}.")
        return None, None

    # Apply normalization using the stored min/max values
    for col in ['travel_time_mean', 'friction_mean', 'vim', 'rfh', 'market_population_5km', 'year']:
        norm_col_name = f'{col.replace("_mean", "")}_norm' # Adjust for normalized column names
        if col in input_df.columns and col in min_max_values:
            min_val = min_max_values[col]['min']
            max_val = min_max_values[col]['max']
            if (max_val - min_val) > 0:
                input_df[norm_col_name] = (input_df[col] - min_val) / (max_val - min_val)
            else:
                input_df[norm_col_name] = 0 # Handle cases with no variance
        else:
            input_df[norm_col_name] = 0 # Default if column not found or min/max missing

    # Apply cyclical encoding for month
    if 'month' in input_df.columns:
        input_df['month_sin'] = np.sin(2 * np.pi * input_df['month'] / 12)
        input_df['month_cos'] = np.cos(2 * np.pi * input_df['month'] / 12)
    else:
        input_df['month_sin'] = 0
        input_df['month_cos'] = 0

    # Define features for prediction (must match training features)
    features_for_prediction = [
        'travel_time_norm',
        'friction_norm', 'vim_norm', 'rfh_norm',
        'market_population_5km_norm',
        'year_norm', 'month_sin', 'month_cos'
    ]

    # Ensure the input DataFrame has all required features for prediction
    for feature in features_for_prediction:
        if feature not in input_df.columns:
            input_df[feature] = 0 # Add missing feature with a default value

    # Make predictions
    try:
        rf_prediction = rf_model.predict(input_df[features_for_prediction])[0]
        xgb_prediction = xgb_model.predict(input_df[features_for_prediction])[0]
        return rf_prediction, xgb_prediction, spoilage_rate
    except Exception as e:
        log_error(f"Error during model prediction for {commodity_key.capitalize()}: {e}")
        return None, None, None

# -------------------------------
# Main Dashboard
# -------------------------------
def main():
    # Custom CSS
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
        border: 1px solid grey; 
        padding: 5px; 
        box-shadow: 1px 1px 3px rgba(0,0,0,0.2); 
        margin-top: 5px; 
        display: block; 
        font-size: 10px; 
    }
    .legend-title { font-weight: bold; font-size: 10px; margin-bottom: 2px; }
    .legend-item { display: flex; align-items: center; margin-bottom: 2px; font-size: 10px; }
    .legend-color { width: 10px; height: 10px; margin-right: 4px; display: inline-block; }
    @media (max-width: 600px) { 
        .folium-map { height: 50vh; }
        .legend-container { font-size: 8px; padding: 3px; }
        .legend-color { width: 8px; height: 8px; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
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
    This interactive tool visualizes travel time, friction surfaces, market locations, road networks, commodity prices, and population density across Senegal. Compare retail prices at specific markets with farmgate prices in production regions.
    """)

    # Validate files
    if not ensure_default_files():
        return
    for key, path in st.session_state.file_paths.items():
        if not validate_file(path, key.replace('_', ' ').title()):
            return

    # Load data
    with st.spinner("Loading geospatial data..."):
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
        st.success("Geospatial data loaded successfully!")

    # Load all models and scalers only once
    if not st.session_state['models_and_scalers']:
        with st.spinner("Loading machine learning models and preprocessing objects..."):
            st.session_state['models_and_scalers'] = load_all_models_and_scalers(COMMODITY_MODELS_CONFIG, DEFAULT_FILES['merged_data_for_models'])
        if st.session_state['models_and_scalers']:
            st.success("All models and preprocessing objects loaded successfully!")
        else:
            st.error("Failed to load all models and preprocessing objects. Check error messages above.")
    
    # Create tabs
    tab_titles = ["Geospatial Analysis", "Price Trends", "Predict Transaction Cost", "Model Insights", "Data & File Management"]
    tabs = st.tabs(tab_titles)

    with tabs[0]: # Geospatial Analysis
        st.header("Geospatial Analysis")
        st.markdown("Explore various geospatial layers including travel time, friction surfaces, market locations, and roads.")

        # Geospatial Layer Selection
        geospatial_layers = {
            "Travel Time": {"data": travel_time, "bounds": travel_bounds, "generator": generate_travel_image, "title": "Travel Time (Hours)"},
            "Friction Surface": {"data": friction_data, "bounds": friction_bounds, "generator": generate_friction_image, "title": "Friction Surface"},
            "Population Density": {"data": population_data, "bounds": population_bounds, "generator": generate_population_image, "title": f"Population Density ({selected_year})"}
        }
        selected_layer_name = st.selectbox("Select Geospatial Layer:", list(geospatial_layers.keys()), key='geospatial_layer_select')
        selected_layer = geospatial_layers[selected_layer_name]

        if selected_layer["data"] is None or selected_layer["bounds"] is None:
            st.warning(f"Failed to load data for {selected_layer_name}. Cannot display map.")
        else:
            image_path, overlay_bounds, breaks, colors = selected_layer["generator"](selected_layer["data"], selected_layer["bounds"])
            
            m = folium.Map(location=[14.4974, -14.4524], zoom_start=7, control_scale=True, tiles="CartoDB positron")

            # Add Image Overlay for the selected layer
            if image_path:
                folium.raster_layers.ImageOverlay(
                    image=image_path,
                    bounds=overlay_bounds,
                    opacity=0.7,
                    alt=selected_layer_name
                ).add_to(m)

            # Add roads if loaded
            if roads_filtered:
                folium.GeoJson(
                    roads_filtered,
                    name='Road Network',
                    style_function=lambda x: {'color': '#666666', 'weight': 1.5, 'opacity': 0.7}
                ).add_to(m)

            # Add markets as a MarkerCluster
            if markets:
                market_cluster = MarkerCluster(name='Markets').add_to(m)
                for feature in markets['features']:
                    lat, lon = feature['geometry']['coordinates'][1], feature['geometry']['coordinates'][0]
                    props = feature['properties']
                    popup_html = f"<b>Market:</b> {props.get('name', 'N/A')}<br>" \
                                 f"<b>Type:</b> {props.get('type', 'N/A')}<br>" \
                                 f"<b>Admin Level:</b> {props.get('admin_level', 'N/A')}<br>" \
                                 f"<b>Population (5km):</b> {props.get('population_5km', 'N/A'):,.0f}" # Display population
                    folium.Marker([lat, lon], popup=popup_html).add_to(market_cluster)

            # Add full screen and mini map
            Fullscreen().add_to(m)
            MiniMap(toggle_display=True).add_to(m)
            
            # Add Layer Control
            folium.LayerControl().add_to(m)

            # Display legends
            if st.session_state['show_legends']:
                if breaks and colors:
                    st.markdown(generate_legend_html(breaks, colors, selected_layer["title"]), unsafe_allow_html=True)

            st_folium(m, height=st.session_state['map_height'], width=700, key=f"folium_map_{st.session_state['map_render_key']}")
            
            st.info("The geospatial layers are downsampled for faster loading. Higher resolution can be provided upon request.")


    with tabs[1]: # Price Trends
        st.header("Commodity Price and Margin Trends")
        st.markdown("Analyze historical price data and gross margins for different commodities.")

        merged_data_path = st.session_state.file_paths.get('merged_data_for_models')
        if not merged_data_path or not os.path.exists(merged_data_path):
            st.warning("Merged data file not found. Please ensure 'final_merged_output_senegal.xlsx' is available for price trend analysis.")
            st.markdown("You can upload this file in the 'Data & File Management' tab.")
            st.session_state['latest_merged_prices'] = pd.DataFrame()
        else:
            if st.session_state['latest_merged_prices'].empty:
                st.session_state['latest_merged_prices'] = load_merged_data(merged_data_path)
            
            if not st.session_state['latest_merged_prices'].empty:
                df_trends = st.session_state['latest_merged_prices'].copy()
                df_trends['date'] = pd.to_datetime(df_trends[['year', 'month']].assign(day=1))
                
                # Get unique commodity_ids and map them to names for selection
                commodity_id_to_name = df_trends.set_index('commodity_id')['commodity_english'].to_dict()
                unique_commodity_ids = sorted(df_trends['commodity_id'].unique())
                commodity_options_for_trends = {commodity_id_to_name.get(cid, f"ID: {cid}"): cid for cid in unique_commodity_ids}

                selected_commodity_name_for_trends = st.selectbox(
                    "Select Commodity for Price Trends",
                    options=list(commodity_options_for_trends.keys()),
                    format_func=lambda x: x,
                    key="commodity_trend_select"
                )
                selected_commodity_id_for_trends = commodity_options_for_trends[selected_commodity_name_for_trends]

                df_filtered = df_trends[df_trends['commodity_id'] == selected_commodity_id_for_trends]
                
                if not df_filtered.empty:
                    # Aggregate by month and year
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
                        hovertemplate='%{x|%b %Y}<br>Margin: %{y:.2f} XOF/KG<br>Type: Gross Margin'
                    ))

                    fig.update_layout(
                        title=f"{commodity_id_to_name.get(selected_commodity_id_for_trends, selected_commodity_name_for_trends)} Price and Margin Trends",
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
                    log_error("No data available for price trends for the selected commodity. Please check your data.")
            else:
                log_error("No data available for price trends. Please check your data or upload 'final_merged_output_senegal.xlsx'.")

    with tabs[2]: # Predict Transaction Cost
        st.header("Predict Transaction Cost")
        st.markdown("Use the trained Random Forest and XGBoost models to predict transaction costs based on input features.")

        # Commodity selection for prediction
        commodity_options_pred = list(COMMODITY_MODELS_CONFIG.keys())
        selected_commodity_pred = st.selectbox(
            "Select Commodity for Prediction",
            options=commodity_options_pred,
            format_func=lambda x: x.capitalize(),
            key="commodity_predict_select"
        )
        
        current_commodity_config = COMMODITY_MODELS_CONFIG[selected_commodity_pred]
        current_spoilage_rate = current_commodity_config['spoilage_rate']

        if not st.session_state['models_and_scalers'] or selected_commodity_pred not in st.session_state['models_and_scalers'] or \
           st.session_state['models_and_scalers'][selected_commodity_pred]['rf_model'] is None:
            st.warning(f"Models for {selected_commodity_pred.capitalize()} are not loaded. Please ensure model files and data for normalization are in the correct directories and try reloading the app.")
            st.help(f"Expected model directory for {selected_commodity_pred.capitalize()}: {current_commodity_config['output_dir']}")
            st.help(f"Expected data for normalization: {DEFAULT_FILES['merged_data_for_models']}")
        else:
            st.subheader("Input Features for Prediction")

            col1, col2 = st.columns(2)
            with col1:
                travel_time_mean = st.number_input("Average Travel Time (hours)", min_value=0.0, value=5.0, step=0.1)
                friction_mean = st.number_input("Average Friction (e.g., travel speed in hours per cell)", min_value=0.0, value=0.5, step=0.01)
                vim = st.number_input("VIM (Vegetation Index Measure)", min_value=-1.0, max_value=1.0, value=0.7, step=0.01)
                rfh = st.number_input("RFH (Rainfall Index Measure)", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
                market_population_5km = st.number_input("Market Population (within 5km radius)", min_value=0, value=10000, step=100)
            with col2:
                current_year = st.number_input("Year", min_value=2016, max_value=2025, value=2024, step=1)
                current_month = st.slider("Month", min_value=1, max_value=12, value=7, step=1)
                distance_km = st.number_input("Distance (km)", min_value=0.0, value=50.0, step=1.0)
                price_farmgate = st.number_input("Farmgate Price (XOF/KG)", min_value=0.0, value=200.0, step=10.0)
                
                st.info(f"The prediction also implicitly considers a {selected_commodity_pred.capitalize()} spoilage rate of {current_spoilage_rate * 100:.1f}%.")


            if st.button("Predict Transaction Cost"):
                input_data = {
                    'travel_time_mean': travel_time_mean,
                    'friction_mean': friction_mean,
                    'vim': vim,
                    'rfh': rfh,
                    'market_population_5km': market_population_5km,
                    'year': current_year,
                    'month': current_month,
                    'distance_km': distance_km, # Needed for formula-based cost
                    'price_farmgate': price_farmgate # Needed for formula-based cost
                }

                rf_pred, xgb_pred, actual_spoilage_rate_used = predict_transaction_cost_commodity(selected_commodity_pred, input_data, st.session_state['models_and_scalers'])

                if rf_pred is not None and xgb_pred is not None:
                    st.subheader("Prediction Results")
                    st.write(f"**Predicted Transaction Cost (Random Forest):** {rf_pred:.2f} XOF/KG")
                    st.write(f"**Predicted Transaction Cost (XGBoost):** {xgb_pred:.2f} XOF/KG")
                    
                    # Calculate estimated transport and spoilage costs based on user input
                    estimated_transport_cost = distance_km * TRANSPORT_RATE_XOF_PER_KG_KM
                    estimated_spoilage_cost = actual_spoilage_rate_used * price_farmgate
                    total_estimated_cost_formula = estimated_transport_cost + estimated_spoilage_cost

                    st.write(f"*(Formula-based cost: Transport ({estimated_transport_cost:.2f} XOF/kg) + Spoilage ({estimated_spoilage_cost:.2f} XOF/kg) = **{total_estimated_cost_formula:.2f} XOF/kg**)*")

                    st.markdown("""
                        <p style='font-size:14px; color:grey;'>
                        <i>Note: The models learn complex relationships from the data. The formula-based cost is provided for reference, but the model predictions incorporate additional learned patterns from the features.</i>
                        </p>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Prediction could not be made. Please check the inputs and ensure models are loaded correctly.")
    
    with tabs[3]: # Model Insights
        st.header("Model Insights and Performance")
        st.markdown("Review the performance metrics and feature importance of the trained models for selected commodities.")

        # Commodity selection for insights
        commodity_options_insights = list(COMMODITY_MODELS_CONFIG.keys())
        selected_commodity_insights = st.selectbox(
            "Select Commodity for Model Insights",
            options=commodity_options_insights,
            format_func=lambda x: x.capitalize(),
            key="commodity_insights_select"
        )
        
        current_insights_config = COMMODITY_MODELS_CONFIG[selected_commodity_insights]
        model_comparison_csv_path = os.path.join(current_insights_config['output_dir'], current_insights_config['model_comparison_csv'])
        feature_importance_excel_path = os.path.join(current_insights_config['output_dir'], current_insights_config['feature_importance_excel'])

        st.subheader(f"Model Performance Comparison ({selected_commodity_insights.capitalize()} - Test Set)")
        try:
            model_comparison_df = pd.read_csv(model_comparison_csv_path)
            st.dataframe(model_comparison_df[['Model', 'R2_Test', 'MAE_Test', 'RMSE_Test', 'MAPE_Test', 'MedAE_Test']].round(2))

            # Plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = ['MAE_Test', 'RMSE_Test', 'MAPE_Test']
            models = ['Random Forest', 'XGBoost']
            x = np.arange(len(metrics))
            width = 0.35

            rf_vals = [
                model_comparison_df[model_comparison_df['Model'] == 'Random Forest']['MAE_Test'].iloc[0],
                model_comparison_df[model_comparison_df['Model'] == 'Random Forest']['RMSE_Test'].iloc[0],
                model_comparison_df[model_comparison_df['Model'] == 'Random Forest']['MAPE_Test'].iloc[0]
            ]
            xgb_vals = [
                model_comparison_df[model_comparison_df['Model'] == 'XGBoost']['MAE_Test'].iloc[0],
                model_comparison_df[model_comparison_df['Model'] == 'XGBoost']['RMSE_Test'].iloc[0],
                model_comparison_df[model_comparison_df['Model'] == 'XGBoost']['MAPE_Test'].iloc[0]
            ]

            ax.bar(x - width/2, rf_vals, width, label='Random Forest', color='#1f77b4', alpha=0.8)
            ax.bar(x + width/2, xgb_vals, width, label='XGBoost', color='#ff7f0e', alpha=0.8)

            ax.set_ylabel('Metric Value')
            ax.set_title(f'Model Performance on Test Set ({selected_commodity_insights.capitalize()})')
            ax.set_xticks(x)
            ax.set_xticklabels(['MAE', 'RMSE', 'MAPE (%)'])
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

        except FileNotFoundError:
            st.warning(f"Model comparison CSV not found for {selected_commodity_insights.capitalize()} at '{model_comparison_csv_path}'. Please ensure it's generated by the analysis script.")
        except Exception as e:
            log_error(f"Error loading/plotting model comparison for {selected_commodity_insights.capitalize()}: {e}")

        st.subheader(f"Feature Importance ({selected_commodity_insights.capitalize()})")
        try:
            feature_importance_df = pd.read_excel(feature_importance_excel_path)
            st.dataframe(feature_importance_df)

            # Plotting feature importance from the loaded dataframe
            fig_feat_imp, ax_feat_imp = plt.subplots(figsize=(10, 6))
            bar_width = 0.35
            index = np.arange(len(feature_importance_df))
            
            bars1 = ax_feat_imp.bar(index, feature_importance_df['Random_Forest_Importance'], bar_width, label='Random Forest', color='#1f77b4', alpha=0.8)
            bars2 = ax_feat_imp.bar(index + bar_width, feature_importance_df['XGBoost_Importance'], bar_width, label='XGBoost', color='#ff7f0e', alpha=0.8)

            ax_feat_imp.set_xlabel('Features')
            ax_feat_imp.set_ylabel('Feature Importance')
            ax_feat_imp.set_title(f'Feature Importance for {selected_commodity_insights.capitalize()} Transaction Cost Models')
            ax_feat_imp.set_xticks(index + bar_width / 2)
            ax_feat_imp.set_xticklabels(feature_importance_df['Feature'], rotation=45, ha='right')
            ax_feat_imp.legend(frameon=True, loc='upper right')
            ax_feat_imp.grid(True, which='major', linestyle='--', alpha=0.5)
            
            st.pyplot(fig_feat_imp)

        except FileNotFoundError:
            st.warning(f"Feature importance Excel file not found for {selected_commodity_insights.capitalize()} at '{feature_importance_excel_path}'. Please ensure it's generated by the analysis script.")
        except Exception as e:
            log_error(f"Error loading/plotting feature importance for {selected_commodity_insights.capitalize()}: {e}")

    with tabs[4]: # Data & File Management
        st.header("Data and File Management")
        st.markdown("Manage the data files used by the dashboard. You can upload new files to update the dashboard's data.")

        st.subheader("Current Default File Paths:")
        for key, path in st.session_state.file_paths.items():
            st.write(f"- **{key.replace('_', ' ').title()}:** `{path}`")

        st.subheader("Upload New Files (Optional)")

        uploaded_raster = st.file_uploader("Upload Travel Time Raster (.tiff)", type=["tif", "tiff"], key="upload_raster")
        if uploaded_raster is not None:
            temp_file_path = os.path.join(BASE_DIR, uploaded_raster.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_raster.getbuffer())
            st.session_state['file_paths']['raster'] = temp_file_path
            st.success(f"Travel time raster uploaded and set to: {temp_file_path}")
            st.session_state['map_render_key'] += 1 # Force map re-render
            st.session_state['map_data_updated'] = True

        uploaded_friction = st.file_uploader("Upload Friction Surface Raster (.tiff)", type=["tif", "tiff"], key="upload_friction")
        if uploaded_friction is not None:
            temp_file_path = os.path.join(BASE_DIR, uploaded_friction.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_friction.getbuffer())
            st.session_state['file_paths']['friction'] = temp_file_path
            st.success(f"Friction surface raster uploaded and set to: {temp_file_path}")
            st.session_state['map_render_key'] += 1 # Force map re-render
            st.session_state['map_data_updated'] = True

        uploaded_markets = st.file_uploader("Upload Markets GeoJSON (.geojson)", type=["geojson"], key="upload_markets")
        if uploaded_markets is not None:
            temp_file_path = os.path.join(BASE_DIR, uploaded_markets.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_markets.getbuffer())
            st.session_state['file_paths']['markets'] = temp_file_path
            st.success(f"Markets GeoJSON uploaded and set to: {temp_file_path}")
            st.session_state['map_render_key'] += 1 # Force map re-render
            st.session_state['map_data_updated'] = True

        uploaded_roads = st.file_uploader("Upload Roads GeoJSON (.geojson)", type=["geojson"], key="upload_roads")
        if uploaded_roads is not None:
            temp_file_path = os.path.join(BASE_DIR, uploaded_roads.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_roads.getbuffer())
            st.session_state['file_paths']['roads'] = temp_file_path
            st.success(f"Roads GeoJSON uploaded and set to: {temp_file_path}")
            st.session_state['map_render_key'] += 1 # Force map re-render
            st.session_state['map_data_updated'] = True

        uploaded_prices = st.file_uploader("Upload Merged Prices Excel (.xlsx)", type=["xlsx"], key="upload_prices")
        if uploaded_prices is not None:
            temp_file_path = os.path.join(BASE_DIR, uploaded_prices.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_prices.getbuffer())
            st.session_state['file_paths']['prices'] = temp_file_path
            st.session_state['file_paths']['merged_data_for_models'] = temp_file_path # Update path for model data
            st.success(f"Merged Prices Excel uploaded and set to: {temp_file_path}")
            # Clear cached data to force reload with new file
            st.cache_data.clear()
            st.session_state['latest_farmgate_prices'] = pd.DataFrame()
            st.session_state['latest_retail_prices'] = pd.DataFrame()
            st.session_state['latest_merged_prices'] = pd.DataFrame()
            # Also clear models if the data for normalization changes
            st.session_state['models_and_scalers'] = {} # Clear all loaded models and scalers
            st.rerun() # Rerun to trigger reload of models and data


        uploaded_population = st.file_uploader("Upload Population Raster (2016-2020, .tif)", type=["tif", "tiff"], accept_multiple_files=True, key="upload_population")
        if uploaded_population:
            for uploaded_file in uploaded_population:
                year_match = re.search(r'(\d{4})', uploaded_file.name)
                if year_match:
                    year = year_match.group(1)
                    pop_key = f'population_{year}'
                    temp_file_path = os.path.join(BASE_DIR, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state['file_paths'][pop_key] = temp_file_path
                    st.success(f"Population raster for {year} uploaded and set to: {temp_file_path}")
                    st.session_state['map_render_key'] += 1 # Force map re-render
                    st.session_state['map_data_updated'] = True
                else:
                    st.warning(f"Could not extract year from filename: {uploaded_file.name}. Please ensure the year is in the filename (e.g., sen_ppp_2020_UNadj.tif).")


        st.subheader("Utility Actions")
        if st.button("Clear Cache & Reload Data", key="clear_cache_button"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state['map_data_updated'] = False
            st.session_state['latest_farmgate_prices'] = pd.DataFrame()
            st.session_state['latest_retail_prices'] = pd.DataFrame()
            st.session_state['latest_merged_prices'] = pd.DataFrame()
            st.session_state['models_and_scalers'] = {} # Clear all loaded models and scalers
            st.session_state['file_paths'] = DEFAULT_FILES.copy() # Reset to default paths
            cleanup_temp_files()
            st.rerun()
            st.success("Cache cleared and data reloaded!")
            
        st.checkbox("Show Legends on Map", value=st.session_state['show_legends'], key='show_legends_checkbox', on_change=lambda: st.session_state.update(show_legends=st.session_state['show_legends_checkbox']))


    # Display errors at the end
    if st.session_state['errors']:
        st.subheader("Errors Encountered:")
        for error in st.session_state['errors']:
            st.error(error)
        st.session_state['errors'] = [] # Clear errors after displaying


    # Footer
    st.markdown("""
    <div class="footer">
        <p>Developed by xAI in collaboration with IFPRI | Data Sources: IFPRI, OpenStreetMap, WorldPop | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
