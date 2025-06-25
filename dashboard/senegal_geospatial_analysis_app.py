import os
import json
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
import folium
from folium.plugins import MarkerCluster, MiniMap, Fullscreen
import streamlit as st
from streamlit_folium import st_folium
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go
import uuid
import tempfile
import atexit
from scipy.interpolate import interp1d
from shapely.geometry import Point
from rasterio.mask import mask

# -------------------------------
# Configuration: File Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILES = {
    'raster': os.path.join(BASE_DIR, '201501_Global_Travel_Time_to_Cities_SEN.tiff'),
    'friction': os.path.join(BASE_DIR, '201501_Global_Travel_Speed_Friction_Surface_SEN.tiff'),
    'markets': os.path.join(BASE_DIR, 'markets_from_excel.geojson'),
    'roads': os.path.join(BASE_DIR, 'roads_filtered.geojson'),
    'prices': os.path.join(BASE_DIR, 'merged_farmgate_retail_prices_senegal.xlsx'),
    'population': os.path.join(BASE_DIR, 'sen_ppp_2020_UNadj.tif')
}

# Initialize session state
if 'map_data_updated' not in st.session_state:
    st.session_state['map_data_updated'] = False
if 'latest_farmgate_prices' not in st.session_state:
    st.session_state['latest_farmgate_prices'] = pd.DataFrame()
if 'latest_retail_prices' not in st.session_state:
    st.session_state['latest_retail_prices'] = pd.DataFrame()
if 'file_paths' not in st.session_state:
    st.session_state['file_paths'] = DEFAULT_FILES.copy()
if 'map_render_key' not in st.session_state:
    st.session_state['map_render_key'] = 0
if 'commodity_map' not in st.session_state:
    st.session_state['commodity_map'] = {}
if 'map_height' not in st.session_state:
    st.session_state['map_height'] = 800  # Default map height

# -------------------------------
# Helper Functions
# -------------------------------
def ensure_default_files():
    """Check if default files exist, prompt for uploads if missing."""
    for key, path in DEFAULT_FILES.items():
        if not os.path.exists(path):
            st.warning(f"Default {key} file not found. Please upload a {key} file.")
            return False
    return True

def validate_file(file_path, file_type):
    """Validate file existence."""
    if not os.path.exists(file_path):
        st.error(f"{file_type} file not found: {file_path}")
        return False
    return True

def generate_colors(data, breaks, colors):
    """Convert raster data to RGB image based on specified colors."""
    rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    for i in range(len(breaks) - 1):
        mask = (data >= breaks[i]) & (data < breaks[i + 1])
        rgb[mask] = colors[i]
    return rgb

def cleanup_temp_files(*files):
    """Clean up temporary files."""
    for file in files:
        if os.path.exists(file):
            os.remove(file)

def calculate_nearest_market_distance(farmgate_df, markets_gdf):
    """Calculate distance from each farmgate to nearest market."""
    if farmgate_df.empty or markets_gdf.empty:
        return farmgate_df
    
    farmgate_df['Distance_to_Nearest_Market_km'] = np.nan
    market_coords = [(row.geometry.y, row.geometry.x) for _, row in markets_gdf.iterrows() if row.geometry.type == 'Point']
    
    if not market_coords:
        st.warning("No valid market coordinates found for distance calculation.")
        return farmgate_df
    
    for idx, row in farmgate_df.iterrows():
        if pd.isna(row['Régions - Latitude']) or pd.isna(row['Régions - Longitude']):
            continue
        farmgate_point = (row['Régions - Latitude'], row['Régions - Longitude'])
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
    """Generate dynamic legend HTML."""
    html = f'<div class="legend-container"><div class="legend-title">{title}</div>'
    for i, (b1, b2) in enumerate(zip(breaks[:-1], breaks[1:])):
        color = f'rgb{colors[i]}'
        label = f'{b1:.3f}–{b2:.3f}' if b2 != np.inf else f'>{b1:.3f}'
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
    """
    Compute population within a buffer_size_km radius around each market.
    Returns a dictionary mapping market indices to population estimates.
    """
    if markets_gdf.empty:
        return {}

    # Determine UTM zone for Senegal (approx. Zone 28N for West Africa)
    utm_crs = 'EPSG:32628'  # UTM Zone 28N
    try:
        # Reproject markets to UTM for accurate buffering
        markets_utm = markets_gdf.to_crs(utm_crs)
        # Create 5km buffers (buffer distance in meters)
        markets_utm['geometry'] = markets_utm.geometry.buffer(buffer_size_km * 1000)
        # Reproject buffers back to EPSG:4326 for raster clipping
        markets_buffered = markets_utm.to_crs('EPSG:4326')

        population_sums = {}
        with rasterio.open(population_file) as src:
            for idx, row in markets_buffered.iterrows():
                geom = [row.geometry.__geo_interface__]
                try:
                    # Clip raster with buffer geometry
                    out_image, _ = mask(src, geom, crop=True, nodata=src.nodata)
                    # Mask nodata and sum valid population values
                    out_image = np.ma.masked_equal(out_image, src.nodata)
                    population_sum = np.nansum(out_image)
                    population_sums[idx] = round(population_sum, 2) if not np.isnan(population_sum) else 0
                except Exception as e:
                    st.warning(f"Failed to compute population for market at index {idx}: {e}")
                    population_sums[idx] = 0
        return population_sums
    except Exception as e:
        st.error(f"Error computing population buffers: {e}")
        return {}

# -------------------------------
# Load and Process Rasters
# -------------------------------
@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash((x.data.tobytes(), x.mask.tobytes()))})
def load_and_process_raster(file_path, downsample_factor=2):
    try:
        with rasterio.open(file_path) as src:
            if src.count != 1:
                st.error(f"Raster {file_path} must have exactly one band.")
                return None, None
            if src.crs is None:
                st.warning(f"No CRS found for {file_path}. Assuming WGS84.")
            data = src.read(1, out_shape=(1, src.height // downsample_factor, src.width // downsample_factor), resampling=rasterio.enums.Resampling.bilinear)
            nodata = src.nodata
            bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        return np.ma.masked_equal(data, nodata) if nodata else np.ma.masked_invalid(data), bounds
    except rasterio.errors.RasterioIOError as e:
        st.error(f"Failed to load raster {file_path}: {e}")
        return None, None

# -------------------------------
# Generate Raster Images
# -------------------------------
@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash((x.data.tobytes(), x.mask.tobytes()))})
def generate_travel_image(data, bounds):
    breaks = [0, 10, 30, 60, 120, 240, 1440, np.inf]
    colors = [(255, 255, 204), (255, 237, 160), (254, 178, 76), (253, 141, 60), (240, 59, 32), (189, 0, 38), (128, 0, 38)]
    rgb = generate_colors(data, breaks, colors)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        travel_png_path = tmp.name
        Image.fromarray(rgb).save(travel_png_path)
    atexit.register(cleanup_temp_files, travel_png_path)
    return travel_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]], breaks, colors

@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash((x.data.tobytes(), x.mask.tobytes()))})
def generate_friction_image(data, bounds):
    friction_breaks = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, np.inf]
    friction_colors = [(0, 104, 55), (49, 163, 84), (120, 198, 121), (194, 230, 153), (253, 174, 97), (244, 109, 67), (165, 0, 38)]
    rgb = generate_colors(data, friction_breaks, friction_colors)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        friction_png_path = tmp.name
        Image.fromarray(rgb).save(friction_png_path)
    atexit.register(cleanup_temp_files, friction_png_path)
    return friction_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]], friction_breaks, friction_colors

@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash((x.data.tobytes(), x.mask.tobytes()))})
def generate_population_image(data, bounds):
    breaks = [0, 1, 10, 50, 100, 500, 1000, np.inf]
    colors = [(200, 220, 255), (150, 180, 255), (100, 140, 255), (50, 100, 255), (0, 60, 200), (0, 40, 150), (0, 20, 100)]
    rgb = generate_colors(data, breaks, colors)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        population_png_path = tmp.name
        Image.fromarray(rgb).save(population_png_path)
    atexit.register(cleanup_temp_files, population_png_path)
    st.write(f"Population data range: min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}")
    st.write(f"Population PNG created: {os.path.exists(population_png_path)}")
    return population_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]], breaks, colors

# -------------------------------
# Load GeoJSON Files
# -------------------------------
@st.cache_data
def load_geojson(file_path, max_features=500, is_roads=False, population_file=None):
    try:
        gdf = gpd.read_file(file_path)
        if gdf.empty:
            st.warning(f"{file_path} contains no valid features.")
            return None
        if gdf.crs != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        if is_roads:
            valid_highways = ['motorway', 'trunk', 'primary', 'secondary']
            gdf = gdf[gdf['highway'].isin(valid_highways)] if 'highway' in gdf.columns else gdf
        if len(gdf) > max_features:
            st.warning(f"GeoJSON file {file_path} has {len(gdf)} features. Limiting to {max_features}.")
            gdf = gdf.head(max_features)
        
        # Compute population for markets if population_file is provided
        if not is_roads and population_file:
            population_sums = compute_population_within_buffer(gdf, population_file)
            gdf['population_5km'] = gdf.index.map(population_sums)
        
        geojson_data = json.loads(gdf.to_json())
        st.info(f"Loaded {len(gdf)} features from {file_path}")
        return geojson_data if geojson_data['features'] else None
    except Exception as e:
        st.warning(f"Failed to load {file_path}: {e}")
        return None

# -------------------------------
# Load Price Data
# -------------------------------
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
            st.warning("Farmgate data contains years outside 2016–2025. Filtering.")
            prices_df = prices_df[prices_df['Year'].between(2016, 2025)]
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
            st.warning("Retail data contains years outside 2016–2025. Filtering.")
            retail_df = retail_df[retail_df['Year'].between(2016, 2025)]
        return retail_df
    except Exception as e:
        st.error(f"Error reading retail prices file {file_path}: {e}")
        return pd.DataFrame()

# -------------------------------
# Main Dashboard
# -------------------------------
def main():
    # Custom CSS for styling and responsiveness
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    body {{ font-family: 'Roboto', sans-serif; }}
    .main {{ background-color: #f9fafb; }}
    .sidebar .sidebar-content {{ background-color: #ffffff; }}
    .stButton>button {{ background-color: #1e3a8a; color: white; border-radius: 8px; }}
    .stSelectbox, .stMultiselect {{ background-color: #f3f4f6; border-radius: 8px; }}
    .header {{ background-color: #1e3a8a; color: white; padding: 20px; border-radius: 8px; }}
    .footer {{ background-color: #1e3a8a; color: white; padding: 10px; text-align: center; margin-top: 20px; }}
    .folium-map {{ min-height: 400px; height: {st.session_state['map_height']}px; max-height: 1000px; width: 100% !important; }}
    .stApp [data-testid="stMapContainer"] {{ 
        margin-top: 10px; 
        width: 100% !important; 
        min-height: 400px; 
        max-height: 100vh; 
        overflow: auto; 
    }}
    .legend-container {{ background-color: white; border: 2px solid grey; padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); margin-top: 10px; }}
    .legend-title {{ font-weight: bold; font-size: 14px; margin-bottom: 10px; }}
    .legend-item {{ display: flex; align-items: center; margin-bottom: 5px; font-size: 14px; }}
    .legend-color {{ width: 20px; height: 20px; margin-right: 8px; display: inline-block; }}
    @media (max-width: 600px) {{
        .folium-map {{ height: 60vh; }}
        .legend-container {{ font-size: 12px; padding: 5px; }}
        .legend-color {{ width: 15px; height: 15px; }}
    }}
    </style>
    """, unsafe_allow_html=True)

    # Header with IFPRI branding
    st.markdown("""
    <div class="header">
        <img src="https://www.ifpri.org/themes/custom/ifpri/logo.svg" alt="IFPRI Logo" width="150">
        <h1>Senegal Agricultural Market Dashboard</h1>
        <p>Developed in collaboration with the International Food Policy Research Institute (IFPRI)</p>
    </div>
    """, unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    ### Welcome to the Senegal Agricultural Market Dashboard
    This interactive tool visualizes travel time, friction surfaces, market locations, road networks, commodity prices, and population density across Senegal. 
    Use the filters to explore data and download insights for policy-making.
    """)

    # File Upload Section
    st.sidebar.header("Data Sources")
    uploaded_files = {}
    for key, default_path in DEFAULT_FILES.items():
        uploaded_file = st.sidebar.file_uploader(f"Upload {key.capitalize()} File", type=['tiff', 'tif'] if key in ['raster', 'friction', 'population'] else ['geojson', 'xlsx'])
        if uploaded_file:
            uploaded_path = os.path.join(BASE_DIR, uploaded_file.name)
            with open(uploaded_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            uploaded_files[key] = uploaded_path
        else:
            uploaded_files[key] = default_path

    st.session_state.file_paths.update(uploaded_files)

    # Validate files
    if not ensure_default_files():
        return
    for key, path in st.session_state.file_paths.items():
        if not validate_file(path, key.capitalize()):
            return

    # Load data with progress bar
    with st.spinner("Loading data..."):
        progress = st.progress(0)
        travel_time, travel_bounds = load_and_process_raster(st.session_state.file_paths['raster'])
        progress.progress(0.2)
        friction_data, friction_bounds = load_and_process_raster(st.session_state.file_paths['friction'])
        progress.progress(0.4)
        population_data, population_bounds = load_and_process_raster(st.session_state.file_paths['population'])
        progress.progress(0.6)
        markets = load_geojson(st.session_state.file_paths['markets'], population_file=st.session_state.file_paths['population'])
        progress.progress(0.8)
        roads_filtered = load_geojson(st.session_state.file_paths['roads'], max_features=500, is_roads=True)
        progress.progress(1.0)

    if travel_time is None or friction_data is None or population_data is None:
        st.error("Failed to load raster data. Please check the files and try again.")
        return

    if population_data is not None:
        st.write(f"Population data loaded: shape={population_data.shape}, bounds={population_bounds}")
    else:
        st.error("Population data is None. Please check the file.")

    # Generate raster images
    travel_png_path, travel_image_bounds, travel_breaks, travel_colors = generate_travel_image(travel_time, travel_bounds)
    friction_png_path, friction_image_bounds, friction_breaks, friction_colors = generate_friction_image(friction_data, friction_bounds)
    population_png_path, population_image_bounds, population_breaks, population_colors = generate_population_image(population_data, population_bounds)

    # Validate price data columns
    farmgate_required_columns = ['Régions Name', 'Commodity', 'commodity_english', 'commodity_id', 'Price', 'Unit2', 'Régions - Latitude', 'Régions - Longitude', 'Year', 'Month']
    retail_required_columns = ['market', 'commodity', 'commodity_id', 'Price', 'Unit2', 'latitude', 'longitude', 'Year', 'Month']
    prices_df = load_price_data(st.session_state.file_paths['prices'])
    retail_df = load_retail_data(st.session_state.file_paths['prices'])
    if not prices_df.empty and any(col not in prices_df.columns for col in farmgate_required_columns):
        st.error(f"Missing columns in farmgate prices: {', '.join([col for col in farmgate_required_columns if col not in prices_df.columns])}")
        prices_df = pd.DataFrame()
    if not retail_df.empty and any(col not in retail_df.columns for col in retail_required_columns):
        st.error(f"Missing columns in retail prices: {', '.join([col for col in retail_required_columns if col not in retail_df.columns])}")
        retail_df = pd.DataFrame()

    # Create commodity name mapping
    if prices_df.empty and retail_df.empty:
        st.error("No valid price data loaded. Please check your files.")
        return
    commodity_map = pd.concat([
        prices_df[['commodity_id', 'commodity_english']].drop_duplicates() if not prices_df.empty else pd.DataFrame(),
        retail_df[['commodity_id', 'commodity']].drop_duplicates().rename(columns={'commodity': 'commodity_english'}) if not retail_df.empty else pd.DataFrame()
    ]).drop_duplicates(subset='commodity_id').set_index('commodity_id')['commodity_english'].to_dict()
    st.session_state.commodity_map = commodity_map

    # Validate commodity overlap
    commodity_options = validate_commodity_overlap(prices_df, retail_df)
    if not commodity_options:
        return
    commodity_id_to_name = {cid: commodity_map.get(cid, str(cid)) for cid in commodity_options}

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Interactive Map", "Data Summary", "Price Trends"])

    with tab1:
        st.subheader("Interactive Map")
        st.markdown("Explore travel time, friction surfaces, market locations, road networks, commodity prices, and population density.")
        st.info("Map view is fixed. Pan or zoom to explore all data. Roads layer may load slowly due to data complexity.")

        # Filter controls
        st.sidebar.header("Map Filters")
        available_years = list(range(2016, 2026))
        selected_year = st.sidebar.selectbox("Select Year", available_years, index=len(available_years)-1, key="year_select", on_change=lambda: st.session_state.update({'map_data_updated': True}))

        month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                       7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
        available_months = sorted(list(set(prices_df['Month'].unique()) | set(retail_df['Month'].unique()))) if not prices_df.empty or not retail_df.empty else list(range(1, 13))
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

        # Layer toggles
        st.sidebar.header("Map Layers")
        show_travel = st.sidebar.checkbox("Travel Time", value=False)
        show_friction = st.sidebar.checkbox("Friction Surface", value=False)
        show_roads = st.sidebar.checkbox("Roads", value=False)
        show_markets = st.sidebar.checkbox("Markets", value=True)
        show_farmgate = st.sidebar.checkbox("Farmgate Prices", value=False)
        show_retail = st.sidebar.checkbox("Retail Prices", value=False)
        show_population = st.sidebar.checkbox("Population", value=True)

        # Map height control
        st.sidebar.slider("Map Height (px)", 400, 1000, st.session_state['map_height'], key="map_height")

        # Update map data
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
                if latest_retail_prices.index.duplicated().any():
                    st.warning("Duplicate indices in retail prices. Removing duplicates.")
                    latest_retail_prices = latest_retail_prices.drop_duplicates().reset_index(drop=True)
                if selected_commodity_ids:
                    latest_retail_prices = latest_retail_prices[latest_retail_prices['commodity_id'].isin(selected_commodity_ids)]
                if len(latest_retail_prices) > 500:
                    latest_retail_prices = latest_retail_prices.head(500)
                    st.warning("Limited to 500 retail price markers for performance.")
            st.session_state.latest_farmgate_prices = latest_farmgate_prices
            st.session_state.latest_retail_prices = latest_retail_prices
            st.session_state['map_data_updated'] = False
            st.session_state.map_render_key += 1

        # Render Map
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

                    # Add Roads Layer
                    if show_roads and roads_filtered:
                        folium.GeoJson(
                            roads_filtered,
                            name="Roads",
                            style_function=lambda x: {'color': '#3b82f6', 'weight': 1, 'opacity': 0.7}
                        ).add_to(m)

                    # Add Markets Layer
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

                    # Add Farmgate Prices Layer
                    if show_farmgate and not st.session_state.latest_farmgate_prices.empty:
                        farmgate_cluster = MarkerCluster(name="Farmgate Prices").add_to(m)
                        valid_farmgate_markers = 0
                        for _, row in st.session_state.latest_farmgate_prices.iterrows():
                            if pd.isna(row['Régions - Latitude']) or pd.isna(row['Régions - Longitude']):
                                continue
                            distance_text = f"<b>Distance to Nearest Market:</b> {row['Distance_to_Nearest_Market_km']:.2f} km<br>" if not pd.isna(row.get('Distance_to_Nearest_Market_km')) else ""
                            popup_text = f"<b>Region:</b> {row['Régions Name']}<br><b>Commodity:</b> {row['commodity_english']}<br><b>Price:</b> {row['Price']:.2f} {row['Unit2']}<br>{distance_text}<b>Date:</b> {row['Year']}-{row['Month']:02d}"
                            folium.Marker(
                                location=[row['Régions - Latitude'], row['Régions - Longitude']],
                                popup=folium.Popup(popup_text, max_width=250),
                                icon=folium.Icon(color='green', icon='tractor', prefix='fa')
                            ).add_to(farmgate_cluster)
                            valid_farmgate_markers += 1
                        if valid_farmgate_markers == 0:
                            st.warning("No valid farmgate price locations found for the selected filters.")

                    # Add Retail Prices Layer
                    if show_retail and not st.session_state.latest_retail_prices.empty:
                        retail_cluster = MarkerCluster(name="Retail Prices").add_to(m)
                        valid_retail_markers = 0
                        for _, row in st.session_state.latest_retail_prices.iterrows():
                            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                                continue
                            popup_text = f"<b>Market:</b> {row['market']}<br><b>Commodity:</b> {row['commodity']}<br><b>Price:</b> {row['Price']:.2f} {row['Unit2']}<br><b>Date:</b> {row['Year']}-{row['Month']:02d}"
                            folium.Marker(
                                location=[row['latitude'], row['longitude']],
                                popup=folium.Popup(popup_text, max_width=250),
                                icon=folium.Icon(color='purple', icon='shopping-basket', prefix='fa')
                            ).add_to(retail_cluster)
                            valid_retail_markers += 1
                        if valid_retail_markers == 0:
                            st.warning("No valid retail price locations found for the selected filters.")

                    # Add Raster Overlays
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
                            name="Population",
                            image=population_png_path,
                            bounds=population_image_bounds,
                            opacity=0.8,
                            interactive=True,
                            cross_origin=False
                        ).add_to(m)

                    # Add MiniMap and Fullscreen
                    MiniMap(tiles='OpenStreetMap', position='bottomleft', width=150, height=150).add_to(m)
                    Fullscreen(position='topright', title='Expand', title_cancel='Exit').add_to(m)

                    folium.LayerControl(collapsed=False).add_to(m)
                    st_folium(m, use_container_width=True, height=st.session_state['map_height'], key=f"folium_map_{st.session_state.map_render_key}")

                    # Add Dynamic Legends
                    if show_travel or show_friction or show_population:
                        col1, col2 = st.columns(2)
                        if show_travel:
                            with col1:
                                st.markdown(generate_legend_html(travel_breaks, travel_colors, "Travel Time (min)"), unsafe_allow_html=True)
                        if show_friction:
                            with col2:
                                st.markdown(generate_legend_html(friction_breaks, friction_colors, "Friction (min/m)"), unsafe_allow_html=True)
                        if show_population:
                            st.markdown(generate_legend_html(population_breaks, population_colors, "Population (people per pixel)"), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Map rendering failed: {str(e)}. Please check data files or coordinates.")
                finally:
                    st.spinner(False)

    with tab2:
        st.subheader("Data Summary")
        st.markdown("### Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Markets", len(markets['features']) if markets else 0)
        col2.metric("Road Features", len(roads_filtered['features']) if roads_filtered else 0)
        col3.metric("Price Points", len(st.session_state.latest_farmgate_prices) + len(st.session_state.latest_retail_prices))
        if not st.session_state.latest_farmgate_prices.empty and 'Distance_to_Nearest_Market_km' in st.session_state.latest_farmgate_prices.columns:
            avg_distance = st.session_state.latest_farmgate_prices['Distance_to_Nearest_Market_km'].mean()
            col4.metric("Avg Distance to Market (km)", f"{avg_distance:.2f}" if not pd.isna(avg_distance) else "N/A")
        else:
            col4.metric("Avg Distance to Market (km)", "N/A")

    with tab3:
        st.subheader("Price Trends")
        if not prices_df.empty and not retail_df.empty and commodity_options:
            st.markdown("### Farmgate, Retail, and Gross Margin Trends")
            selected_commodity_id = st.selectbox(
                "Select Commodity",
                options=sorted(commodity_options, key=lambda x: commodity_id_to_name[x]),
                format_func=lambda x: commodity_id_to_name.get(x, str(x)),
                key="trend_commodity_select"
            )
            show_gross_margin = st.checkbox("Show Gross Margin (Retail - Farmgate)", value=True, key="show_gross_margin")
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
                st.error(f"Failed to generate price trends: {e}")
        else:
            st.warning("No data available for price trends. Please check your data.")

    # Footer
    st.markdown("""
    <div class="footer">
        <p>Developed by xAI in collaboration with IFPRI | Data Sources: IFPRI, OpenStreetMap, WorldPop | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()