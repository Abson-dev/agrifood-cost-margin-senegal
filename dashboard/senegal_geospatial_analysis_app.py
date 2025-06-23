"""
Streamlit application to visualize travel time, friction surfaces, farmgate prices, and retail prices in Senegal,
overlaying market locations and roads.
Inputs:
- Travel time GeoTIFF: Accessibility to cities (minutes)
- Friction GeoTIFF: Travel speed friction surface (min/m)
- Markets GeoJSON: Market locations
- Roads GeoJSON: Filtered road network
- Farmgate prices Excel: Commodity prices by region (sheet: Farmgate prices Senegal)
- Retail prices Excel: Commodity prices by market (sheet: Retails Price Senegal)
Outputs:
- Interactive Folium map
- Histogram of travel times
- Summary statistics and percentiles
"""

import os
import sys
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
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
def assign_colors(data, breaks, colors):
    """Convert raster data to RGB image based on breakpoints and colors."""
    rgb = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    for i in range(len(breaks) - 1):
        mask = (data >= breaks[i]) & (data < breaks[i + 1])
        rgb[mask] = colors[i]
    return rgb

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.set_page_config(page_title="Senegal Agrifood Geospatial Analysis", layout="wide")
st.title("Senegal Agrifood Geospatial Analysis")
st.markdown("""
This app visualizes travel time to cities, friction surfaces, farmgate prices, retail prices, markets, and roads in Senegal.
Use the sidebar to navigate and toggle map layers.
""")

# Sidebar
st.sidebar.header("Controls")
st.sidebar.markdown("Select options to customize the display.")

# -------------------------------
# 1. Load and Process Travel Time Raster
# -------------------------------
try:
    with rasterio.open(raster_path) as src:
        travel_time = src.read(1, window=rasterio.windows.Window(0, 0, src.width, src.height))
        profile = src.profile
        nodata = src.nodata
        travel_bounds = src.bounds
        transform = src.transform
except rasterio.errors.RasterioIOError as e:
    st.error(f"Error loading travel time raster {raster_path}: {e}")
    sys.exit(1)

# Mask nodata values
data = np.ma.masked_equal(travel_time, nodata) if nodata is not None else np.ma.masked_invalid(travel_time)

# -------------------------------
# 2. Summary Statistics
# -------------------------------
with st.expander("Travel Time Summary Statistics"):
    st.write(f"**Min:** {data.min():.2f} min")
    st.write(f"**Max:** {data.max():.2f} min")
    st.write(f"**Mean:** {data.mean():.2f} min")
    st.write(f"**Std Dev:** {data.std():.2f} min")

# -------------------------------
# 3. Percentiles
# -------------------------------
percentiles = np.percentile(data.compressed(), [5, 25, 50, 75, 95])
with st.expander("Travel Time Percentiles"):
    st.write(f"**5th:** {percentiles[0]:.2f} min")
    st.write(f"**25th:** {percentiles[1]:.2f} min")
    st.write(f"**50th (median):** {percentiles[2]:.2f} min")
    st.write(f"**75th:** {percentiles[3]:.2f} min")
    st.write(f"**95th:** {percentiles[4]:.2f} min")

# -------------------------------
# 4. Histogram
# -------------------------------
st.subheader("Histogram of Travel Time to Cities")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(data.compressed(), bins=50, color='skyblue', edgecolor='black')
ax.set_title('Histogram of Travel Time to Cities (min)')
ax.set_xlabel('Travel Time (minutes)')
ax.set_ylabel('Pixel Count')
ax.grid(True)
plt.tight_layout()
st.pyplot(fig)

# -------------------------------
# 5. Load and Process Friction Raster
# -------------------------------
try:
    with rasterio.open(friction_path) as src:
        friction_data = src.read(1, window=rasterio.windows.Window(0, 0, src.width, src.height))
        friction_bounds = src.bounds
        nodata = src.nodata
except rasterio.errors.RasterioIOError as e:
    st.error(f"Error loading friction raster {friction_path}: {e}")
    sys.exit(1)

friction_data = np.ma.masked_equal(friction_data, nodata) if nodata else np.ma.masked_invalid(friction_data)

# -------------------------------
# 6. Load GeoJSON Files
# -------------------------------
try:
    markets = gpd.read_file(markets_path)
    if markets.empty:
        st.warning("Markets GeoJSON is empty")
    roads_filtered = gpd.read_file(roads_filtered_path)
    if roads_filtered.empty:
        st.warning("Roads GeoJSON is empty")
except Exception as e:
    st.error(f"Error loading GeoJSON files: {e}")
    sys.exit(1)

# -------------------------------
# 7. Load and Process Farmgate Prices
# -------------------------------
try:
    prices_df = pd.read_excel(prices_path, sheet_name='Farmgate prices Senegal')
    st.info(f"Loaded farmgate prices Excel with {len(prices_df)} rows")
    with st.expander("Farmgate Column Names"):
        st.write(prices_df.columns.tolist())
    if prices_df.empty:
        st.warning("Farmgate prices Excel is empty")
except Exception as e:
    st.error(f"Error loading farmgate prices file {prices_path}: {e}")
    sys.exit(1)

# Validate required farmgate columns
farmgate_required_columns = ['Régions Name', 'Commodity', 'commodity_english', 'Price', 'Unit2', 'Régions - Latitude', 'Régions - Longitude', 'Year', 'Month']
farmgate_missing_columns = [col for col in farmgate_required_columns if col not in prices_df.columns]
if farmgate_missing_columns:
    st.error(f"Missing required columns in farmgate prices: {farmgate_missing_columns}")
    sys.exit(1)

# Check for invalid farmgate coordinates
invalid_farmgate_coords = prices_df[['Régions - Latitude', 'Régions - Longitude']].isna().any(axis=1)
if invalid_farmgate_coords.any():
    st.warning(f"Found {invalid_farmgate_coords.sum()} rows with invalid coordinates in farmgate prices")

# Filter farmgate prices to most recent year (2016 based on sample data)
farmgate_year = prices_df['Year'].max()
prices_df = prices_df[prices_df['Year'] == farmgate_year]
st.info(f"Filtered farmgate prices to year {farmgate_year} with {len(prices_df)} rows")
with st.expander("Unique Farmgate Commodities"):
    st.write(prices_df['commodity_english'].unique().tolist())

# Group by region and commodity, take the most recent price
prices_df['Date'] = pd.to_datetime(prices_df[['Year', 'Month']].assign(day=1))
latest_farmgate_prices = prices_df.sort_values('Date').groupby(['Régions Name', 'Commodity']).last().reset_index()
st.info(f"Processed {len(latest_farmgate_prices)} unique farmgate price entries")

# -------------------------------
# 8. Load and Process Retail Prices
# -------------------------------
try:
    retail_df = pd.read_excel(prices_path, sheet_name='Retails Price Senegal')
    st.info(f"Loaded retail prices Excel with {len(retail_df)} rows")
    with st.expander("Retail Column Names"):
        st.write(retail_df.columns.tolist())
    if retail_df.empty:
        st.warning("Retail prices Excel is empty")
except Exception as e:
    st.error(f"Error loading retail prices file {prices_path}: {e}")
    sys.exit(1)

# Validate required retail columns
retail_required_columns = ['market', 'commodity', 'price', 'Unit2', 'latitude', 'longitude', 'Year', 'Month']
retail_missing_columns = [col for col in retail_required_columns if col not in retail_df.columns]
if retail_missing_columns:
    st.error(f"Missing required columns in retail prices: {retail_missing_columns}")
    sys.exit(1)

# Check for invalid retail coordinates
invalid_retail_coords = retail_df[['latitude', 'longitude']].isna().any(axis=1)
if invalid_retail_coords.any():
    st.warning(f"Found {invalid_retail_coords.sum()} rows with invalid coordinates in retail prices")

# Filter retail prices to most recent year (2000 based on sample data)
retail_year = retail_df['Year'].max()
retail_df = retail_df[retail_df['Year'] == retail_year]
st.info(f"Filtered retail prices to year {retail_year} with {len(retail_df)} rows")
with st.expander("Unique Retail Commodities"):
    st.write(retail_df['commodity'].unique().tolist())

# Group by market and commodity, take the most recent price
retail_df['Date'] = pd.to_datetime(retail_df[['Year', 'Month']].assign(day=1))
latest_retail_prices = retail_df.sort_values('Date').groupby(['market', 'commodity']).last().reset_index()
st.info(f"Processed {len(latest_retail_prices)} unique retail price entries")

# -------------------------------
# 9. Color Mapping for Travel Time
# -------------------------------
breaks = [0, 10, 30, 60, 120, 240, 1440, np.inf]
colors = [
    (255, 255, 204),  # #ffffcc
    (255, 237, 160),  # #ffeda0
    (254, 178, 76),   # #feb24c
    (253, 141, 60),   # #fd8d3c
    (240, 59, 32),    # #f03b20
    (189, 0, 38),     # #bd0026
    (128, 0, 38)      # overflow
]
rgb = assign_colors(data, breaks, colors)
travel_png_path = 'travel_time_colored.png'
Image.fromarray(rgb).save(travel_png_path)
travel_image_bounds = [[travel_bounds.bottom, travel_bounds.left], [travel_bounds.top, travel_bounds.right]]

# -------------------------------
# 10. Color Mapping for Friction
# -------------------------------
friction_breaks = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, np.inf]
friction_colors = [
    (0, 104, 55),    # #006837
    (49, 163, 84),   # #31a354
    (120, 198, 121), # #78c679
    (194, 230, 153), # #c2e699
    (253, 174, 97),  # #fdae61
    (244, 109, 67),  # #f46d43
    (165, 0, 38),    # #a50026
    (128, 0, 38)     # overflow
]
rgb_friction = assign_colors(friction_data, friction_breaks, friction_colors)
friction_png_path = 'friction_surface_colored.png'
Image.fromarray(rgb_friction).save(friction_png_path)
friction_image_bounds = [[friction_bounds.bottom, friction_bounds.left], [friction_bounds.top, friction_bounds.right]]

# -------------------------------
# 11. Create Folium Map
# -------------------------------
st.subheader("Interactive Map")
center_lat = (travel_bounds.bottom + travel_bounds.top) / 2
center_lon = (travel_bounds.left + travel_bounds.right) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB Positron")

# Add Roads Layer
folium.GeoJson(
    roads_filtered,
    name="Roads",
    style_function=lambda x: {
        'color': 'blue',
        'weight': 1,
        'opacity': 0.7
    }
).add_to(m)

# Add Markets Layer
market_group = folium.FeatureGroup(name="Markets", show=True)
for _, row in markets.iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=row['market'],
        icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')
    ).add_to(market_group)
market_group.add_to(m)

# Add Farmgate Prices Layer
farmgate_group = folium.FeatureGroup(name="Farmgate Prices", show=True)
farmgate_marker_count = 0
for _, row in latest_farmgate_prices.iterrows():
    if pd.isna(row['Régions - Latitude']) or pd.isna(row['Régions - Longitude']):
        st.warning(f"Skipping farmgate row with invalid coordinates: {row['Régions Name']}, {row['Commodity']}")
        continue
    popup_text = (
        f"<b>Region:</b> {row['Régions Name']}<br>"
        f"<b>Commodity:</b> {row['commodity_english']}<br>"
        f"<b>Price:</b> {row['Price']:.2f} {row['Unit2']}<br>"
        f"<b>Date:</b> {row['Year']}-{row['Month']:02d}"
    )
    folium.Marker(
        location=[row['Régions - Latitude'], row['Régions - Longitude']],
        popup=folium.Popup(popup_text, max_width=250),
        icon=folium.Icon(color='green', icon='tractor', prefix='fa')
    ).add_to(farmgate_group)
    farmgate_marker_count += 1
st.info(f"Added {farmgate_marker_count} farmgate price markers to the map")
farmgate_group.add_to(m)

# Add Retail Prices Layer
retail_group = folium.FeatureGroup(name="Retail Prices", show=True)
retail_marker_count = 0
for _, row in latest_retail_prices.iterrows():
    if pd.isna(row['latitude']) or pd.isna(row['longitude']):
        st.warning(f"Skipping retail row with invalid coordinates: {row['market'], row['commodity']}")
        continue
    popup_text = (
        f"<b>Market:</b> {row['market']}<br>"
        f"<b>Commodity:</b> {row['commodity']}<br>"
        f"<b>Price:</b> {row['price']:.2f} {row['Unit2']}<br>"
        f"<b>Date:</b> {row['Year']}-{row['Month']:02d}"
    )
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=folium.Popup(popup_text, max_width=250),
        icon=folium.Icon(color='purple', icon='shopping-basket', prefix='fa')
    ).add_to(retail_group)
    retail_marker_count += 1
st.info(f"Added {retail_marker_count} retail price markers to the map")
retail_group.add_to(m)

# Add Raster Overlays
folium.raster_layers.ImageOverlay(
    name="Travel Time",
    image=travel_png_path,
    bounds=travel_image_bounds,
    opacity=0.6,
    interactive=True,
    cross_origin=False
).add_to(m)

folium.raster_layers.ImageOverlay(
    name="Friction Surface (min/m)",
    image=friction_png_path,
    bounds=friction_image_bounds,
    opacity=0.7,
    interactive=True,
    cross_origin=False
).add_to(m)

# -------------------------------
# 12. Add Travel Time Legend
# -------------------------------
travel_legend_html = """
{% macro html(this, kwargs) %}
<div style="
    position: fixed;
    bottom: 5%;
    left: 2%;
    width: 180px;
    height: 230px;
    background-color: white;
    border:2px solid grey;
    z-index:9999;
    font-size:14px;
    padding: 10px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">
<b>Travel Time (min)</b><br>
<div style="margin-top:10px;">
  <div style="background:#ffffcc;width:20px;height:20px;display:inline-block;"></div> 0–10<br>
  <div style="background:#ffeda0;width:20px;height:20px;display:inline-block;"></div> 10–30<br>
  <div style="background:#feb24c;width:20px;height:20px;display:inline-block;"></div> 30–60<br>
  <div style="background:#fd8d3c;width:20px;height:20px;display:inline-block;"></div> 60–120<br>
  <div style="background:#f03b20;width:20px;height:20px;display:inline-block;"></div> 120–240<br>
  <div style="background:#bd0026;width:20px;height:20px;display:inline-block;"></div> 240–1440<br>
  <div style="background:#800026;width:20px;height:20px;display:inline-block;"></div> >1440
</div>
</div>
{% endmacro %}
"""
travel_legend = MacroElement()
travel_legend._template = Template(travel_legend_html)
m.get_root().add_child(travel_legend)

# -------------------------------
# 13. Add Friction Legend
# -------------------------------
friction_legend_html = """
{% macro html(this, kwargs) %}
<div style="
    position: fixed;
    bottom: 5%;
    right: 2%;
    width: 200px;
    height: 260px;
    background-color: white;
    border:2px solid grey;
    z-index:9999;
    font-size:14px;
    padding: 10px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">
<b>Friction (min/m)</b><br>
<div style="margin-top:10px;">
  <div style="background:#006837;width:20px;height:20px;display:inline-block;"></div> ≤ 0.001<br>
  <div style="background:#31a354;width:20px;height:20px;display:inline-block;"></div> ≤ 0.01<br>
  <div style="background:#78c679;width:20px;height:20px;display:inline-block;"></div> ≤ 0.1<br>
  <div style="background:#c2e699;width:20px;height:20px;display:inline-block;"></div> ≤ 0.5<br>
  <div style="background:#fdae61;width:20px;height:20px;display:inline-block;"></div> ≤ 1.0<br>
  <div style="background:#f46d43;width:20px;height:20px;display:inline-block;"></div> ≤ 2.0<br>
  <div style="background:#a50026;width:20px;height:20px;display:inline-block;"></div> ≤ 5.0<br>
  <div style="background:#800026;width:20px;height:20px;display:inline-block;"></div> > 5.0<br>
</div>
</div>
{% endmacro %}
"""
friction_legend = MacroElement()
friction_legend._template = Template(friction_legend_html)
m.get_root().add_child(friction_legend)

# -------------------------------
# 14. Render Map in Streamlit
# -------------------------------
folium.LayerControl().add_to(m)
st_folium(m, width=1200, height=600)