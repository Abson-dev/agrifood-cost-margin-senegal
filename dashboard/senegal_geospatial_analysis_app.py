import os
import sys
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import folium
import pandas as pd
from PIL import Image
from branca.element import Template, MacroElement
import streamlit as st
from streamlit_folium import st_folium

# -------------------------------
# Configuration: File Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
raster_path = os.path.join(BASE_DIR, '201501_Global_Travel_Time_to_Cities_SEN.tiff')
friction_path = os.path.join(BASE_DIR, '201501_Global_Travel_Speed_Friction_Surface_SEN.tiff')
markets_path = os.path.join(BASE_DIR, 'markets_from_excel.geojson')
roads_filtered_path = os.path.join(BASE_DIR, 'roads_filtered.geojson')
prices_path = os.path.join(BASE_DIR, 'merged_farmgate_retail_prices_senegal.xlsx')

# -------------------------------
# Helper Function: RGB Conversion
# -------------------------------
def generate_colors(data, breaks, colors):
    """Convert raster data to RGB image based on specified colors."""
    rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    for i in range(len(breaks) - 1):
        mask = (data >= breaks[i]) & (data < breaks[i + 1])
        rgb[mask] = colors[i]
    return rgb

# -------------------------------
# Load and Process Rasters
# -------------------------------
@st.cache_data
def load_and_process_raster(file_path):
    with rasterio.open(file_path) as src:
        # Downsample to reduce computation time
        data = src.read(1, out_shape=(1, src.height // 2, src.width // 2), resampling=rasterio.enums.Resampling.bilinear)
        nodata = src.nodata
        bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)  # Convert to hashable tuple
    return np.ma.masked_equal(data, nodata) if nodata else np.ma.masked_invalid(data), bounds

travel_time, travel_bounds = load_and_process_raster(raster_path)
friction_data, friction_bounds = load_and_process_raster(friction_path)

# -------------------------------
# Generate Raster Images
# -------------------------------
def generate_travel_image(data, bounds):
    breaks = [0, 10, 30, 60, 120, 240, 1440, np.inf]
    colors = [(255, 255, 204), (255, 237, 160), (254, 178, 76), (253, 141, 60), (240, 59, 32), (189, 0, 38), (128, 0, 38)]
    rgb = generate_colors(data, breaks, colors)
    travel_png_path = 'travel_time_colored.png'
    Image.fromarray(rgb).save(travel_png_path)
    return travel_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

def generate_friction_image(data, bounds):
    friction_breaks = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, np.inf]
    friction_colors = [(0, 104, 55), (49, 163, 84), (120, 198, 121), (194, 230, 153), (253, 174, 97), (244, 109, 67), (165, 0, 38), (128, 0, 38)]
    rgb = generate_colors(data, friction_breaks, friction_colors)
    friction_png_path = 'friction_surface_colored.png'
    Image.fromarray(rgb).save(friction_png_path)
    return friction_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

travel_png_path, travel_image_bounds = generate_travel_image(travel_time, travel_bounds)
friction_png_path, friction_image_bounds = generate_friction_image(friction_data, friction_bounds)

# -------------------------------
# Load GeoJSON Files
# -------------------------------
@st.cache_data
def load_geojson(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('type') != 'FeatureCollection':
            st.warning(f"{file_path} is not a valid FeatureCollection; skipping")
            return None
        return data
    except Exception as e:
        st.warning(f"Failed to load {file_path}: {e}; skipping")
        return None

markets = load_geojson(markets_path)
roads_filtered = load_geojson(roads_filtered_path)

# -------------------------------
# Load and Process Price Data
# -------------------------------
@st.cache_data
def load_price_data(file_path):
    try:
        prices_df = pd.read_excel(file_path, sheet_name='Farmgate prices Senegal')
        if prices_df.empty:
            st.warning("Farmgate prices Excel is empty")
            return pd.DataFrame()
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
        return retail_df
    except Exception as e:
        st.error(f"Error reading retail prices file {file_path}: {e}")
        return pd.DataFrame()

prices_df = load_price_data(prices_path)
retail_df = load_retail_data(prices_path)

# Validate required columns
farmgate_required_columns = ['Régions Name', 'Commodity', 'commodity_english', 'Price', 'Unit2', 'Régions - Latitude', 'Régions - Longitude', 'Year', 'Month']
retail_required_columns = ['market', 'commodity', 'price', 'Unit2', 'latitude', 'longitude', 'Year', 'Month']
if not prices_df.empty and any(col not in prices_df.columns for col in farmgate_required_columns):
    st.error("Missing required columns in farmgate prices")
    prices_df = pd.DataFrame()
if not retail_df.empty and any(col not in retail_df.columns for col in retail_required_columns):
    st.error("Missing required columns in retail prices")
    retail_df = pd.DataFrame()

# Check for invalid coordinates
if not prices_df.empty:
    invalid_farmgate_coords = prices_df[['Régions - Latitude', 'Régions - Longitude']].isna().any(axis=1)
    if invalid_farmgate_coords.any():
        st.warning(f"Found {invalid_farmgate_coords.sum()} rows with invalid coordinates in farmgate prices")
if not retail_df.empty:
    invalid_retail_coords = retail_df[['latitude', 'longitude']].isna().any(axis=1)
    if invalid_retail_coords.any():
        st.warning(f"Found {invalid_retail_coords.sum()} rows with invalid coordinates in retail prices")

# -------------------------------
# Filter Data
# -------------------------------
st.sidebar.header("Filter Price Data")
common_years = sorted(list(set(prices_df['Year']).intersection(set(retail_df['Year'])))) if not prices_df.empty and not retail_df.empty else []
selected_year = st.sidebar.selectbox("Select Year", common_years, index=len(common_years)-1) if common_years else None
if not common_years:
    st.sidebar.warning("No common years found between farmgate and retail prices.")

available_months = sorted(list(set(prices_df['Month'].unique()) | set(retail_df['Month'].unique()))) if not prices_df.empty or not retail_df.empty else []
selected_month = st.sidebar.selectbox("Select Month", available_months, index=len(available_months)-1) if available_months else None
if not available_months:
    st.sidebar.error("No months available for the selected year(s).")

commodities = sorted(list(set(prices_df['commodity_english'].unique()) | set(retail_df['commodity'].unique()))) if not prices_df.empty or not retail_df.empty else []
selected_commodities = st.sidebar.multiselect("Select Commodities", commodities, default=commodities) if commodities else []

if st.sidebar.button("Apply Filters"):
    # Filter datasets
    latest_farmgate_prices = pd.DataFrame()
    latest_retail_prices = pd.DataFrame()
    if not prices_df.empty:
        filtered_farmgate = prices_df[prices_df['Year'] == selected_year] if selected_year else prices_df
        if selected_month:
            filtered_farmgate = filtered_farmgate[filtered_farmgate['Month'] == selected_month]
        filtered_farmgate['Date'] = pd.to_datetime(filtered_farmgate[['Year', 'Month']].assign(day=1))
        latest_farmgate_prices = filtered_farmgate.sort_values('Date').groupby(['Régions Name', 'Commodity']).last().reset_index()
        if selected_commodities:
            latest_farmgate_prices = latest_farmgate_prices[latest_farmgate_prices['commodity_english'].isin(selected_commodities)]
        if latest_farmgate_prices.empty:
            st.warning("No farmgate price data available for the selected period")

    if not retail_df.empty:
        filtered_retail = retail_df[retail_df['Year'] == selected_year] if selected_year else retail_df
        if selected_month:
            filtered_retail = filtered_retail[filtered_retail['Month'] == selected_month]
        filtered_retail['Date'] = pd.to_datetime(filtered_retail[['Year', 'Month']].assign(day=1))
        latest_retail_prices = filtered_retail.sort_values('Date').groupby(['market', 'commodity']).last().reset_index()
        if selected_commodities:
            latest_retail_prices = latest_retail_prices[latest_retail_prices['commodity'].isin(selected_commodities)]
        if latest_retail_prices.empty:
            st.warning("No retail price data available for the selected period")
else:
    latest_farmgate_prices = pd.DataFrame()
    latest_retail_prices = pd.DataFrame()
    st.warning("Click 'Apply Filters' to load and display data.")

# -------------------------------
# Create Folium Map
# -------------------------------
if st.button("Render Map"):
    st.subheader("Interactive Map")
    center_lat = (travel_bounds[1] + travel_bounds[3]) / 2
    center_lon = (travel_bounds[0] + travel_bounds[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB Positron")

    # Add Roads Layer
    if roads_filtered:
        try:
            folium.GeoJson(
                roads_filtered,
                name="Roads",
                style_function=lambda x: {'color': 'blue', 'weight': 1, 'opacity': 0.7}
            ).add_to(m)
        except Exception as e:
            st.warning(f"Failed to add roads layer to map: {e}")

    # Add Markets Layer
    if markets:
        try:
            market_group = folium.FeatureGroup(name="Markets", show=True)
            for feature in markets.get('features', []):
                if feature['geometry']['type'] == 'Point':
                    coords = feature['geometry']['coordinates'][::-1]
                    popup = feature['properties'].get('market', 'Unknown Market')
                    folium.Marker(location=coords, popup=popup, icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')).add_to(market_group)
            market_group.add_to(m)
        except Exception as e:
            st.warning(f"Failed to add markets layer to map: {e}")

    # Add Farmgate Prices Layer
    farmgate_group = folium.FeatureGroup(name="Farmgate Prices", show=True)
    for _, row in latest_farmgate_prices.iterrows():
        if pd.isna(row['Régions - Latitude']) or pd.isna(row['Régions - Longitude']):
            st.warning(f"Skipping farmgate row with invalid coordinates: {row['Régions Name']}, {row['Commodity']}")
            continue
        popup_text = f"<b>Region:</b> {row['Régions Name']}<br><b>Commodity:</b> {row['commodity_english']}<br><b>Price:</b> {row['Price']:.2f} {row['Unit2']}<br><b>Date:</b> {row['Year']}-{row['Month']:02d}"
        folium.Marker(location=[row['Régions - Latitude'], row['Régions - Longitude']], popup=folium.Popup(popup_text, max_width=250), icon=folium.Icon(color='green', icon='tractor', prefix='fa')).add_to(farmgate_group)
    farmgate_group.add_to(m)

    # Add Retail Prices Layer
    retail_group = folium.FeatureGroup(name="Retail Prices", show=True)
    for _, row in latest_retail_prices.iterrows():
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            st.warning(f"Skipping retail row with invalid coordinates: {row['market'], row['commodity']}")
            continue
        popup_text = f"<b>Market:</b> {row['market']}<br><b>Commodity:</b> {row['commodity']}<br><b>Price:</b> {row['price']:.2f} {row['Unit2']}<br><b>Date:</b> {row['Year']}-{row['Month']:02d}"
        folium.Marker(location=[row['latitude'], row['longitude']], popup=folium.Popup(popup_text, max_width=250), icon=folium.Icon(color='purple', icon='shopping-basket', prefix='fa')).add_to(retail_group)
    retail_group.add_to(m)

    # Add Raster Overlays
    folium.raster_layers.ImageOverlay(name="Travel Time", image=travel_png_path, bounds=travel_image_bounds, opacity=0.6, interactive=True, cross_origin=False).add_to(m)
    folium.raster_layers.ImageOverlay(name="Friction Surface (min/m)", image=friction_png_path, bounds=friction_image_bounds, opacity=0.7, interactive=True, cross_origin=False).add_to(m)

    # Add Legends
    travel_legend_html = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; bottom: 5%; left: 2%; width: 180px; height: 230px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
    <b>Travel Time (min)</b><br><div style="margin-top:10px;"><div style="background:#ffffcc;width:20px;height:20px;display:inline-block;"></div> 0–10<br><div style="background:#ffeda0;width:20px;height:20px;display:inline-block;"></div> 10–30<br><div style="background:#feb24c;width:20px;height:20px;display:inline-block;"></div> 30–60<br><div style="background:#fd8d3c;width:20px;height:20px;display:inline-block;"></div> 60–120<br><div style="background:#f03b20;width:20px;height:20px;display:inline-block;"></div> 120–240<br><div style="background:#bd0026;width:20px;height:20px;display:inline-block;"></div> 240–1440<br><div style="background:#800026;width:20px;height:20px;display:inline-block;"></div> >1440</div></div>
    {% endmacro %}
    """
    travel_legend = MacroElement()
    travel_legend._template = Template(travel_legend_html)
    m.get_root().add_child(travel_legend)

    friction_legend_html = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; bottom: 5%; right: 2%; width: 200px; height: 260px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
    <b>Friction (min/m)</b><br><div style="margin-top:10px;"><div style="background:#006837;width:20px;height:20px;display:inline-block;"></div> ≤ 0.001<br><div style="background:#31a354;width:20px;height:20px;display:inline-block;"></div> ≤ 0.01<br><div style="background:#78c679;width:20px;height:20px;display:inline-block;"></div> ≤ 0.1<br><div style="background:#c2e699;width:20px;height:20px;display:inline-block;"></div> ≤ 0.5<br><div style="background:#fdae61;width:20px;height:20px;display:inline-block;"></div> ≤ 1.0<br><div style="background:#f46d43;width:20px;height:20px;display:inline-block;"></div> ≤ 2.0<br><div style="background:#a50026;width:20px;height:20px;display:inline-block;"></div> ≤ 5.0<br><div style="background:#800026;width:20px;height:20px;display:inline-block;"></div> > 5.0</div></div>
    {% endmacro %}
    """
    friction_legend = MacroElement()
    friction_legend._template = Template(friction_legend_html)
    m.get_root().add_child(friction_legend)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1200, height=600)