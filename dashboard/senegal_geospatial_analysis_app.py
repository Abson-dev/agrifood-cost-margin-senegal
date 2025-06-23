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
from branca.element import Template, MacroElement
import plotly.express as px
import plotly.graph_objects as go
import uuid

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

# -------------------------------
# Helper Functions
# -------------------------------
def validate_file(file_path, file_type):
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

# -------------------------------
# Load and Process Rasters
# -------------------------------
@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash((x.data.tobytes(), x.mask.tobytes()))})
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
# -------------------------------
@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash((x.data.tobytes(), x.mask.tobytes()))})
def generate_travel_image(data, bounds):
    breaks = [0, 10, 30, 60, 120, 240, 1440, np.inf]
    colors = [(255, 255, 204), (255, 237, 160), (254, 178, 76), (253, 141, 60), (240, 59, 32), (189, 0, 38), (128, 0, 38)]
    rgb = generate_colors(data, breaks, colors)
    travel_png_path = f'travel_time_{str(uuid.uuid4())}.png'
    Image.fromarray(rgb).save(travel_png_path)
    return travel_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

@st.cache_data(hash_funcs={np.ma.MaskedArray: lambda x: hash((x.data.tobytes(), x.mask.tobytes()))})
def generate_friction_image(data, bounds):
    friction_breaks = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, np.inf]
    friction_colors = [(0, 104, 55), (49, 163, 84), (120, 198, 121), (194, 230, 153), (253, 174, 97), (244, 109, 67), (165, 0, 38), (128, 0, 38)]
    rgb = generate_colors(data, friction_breaks, friction_colors)
    friction_png_path = f'friction_surface_{str(uuid.uuid4())}.png'
    Image.fromarray(rgb).save(friction_png_path)
    return friction_png_path, [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

# -------------------------------
# Load GeoJSON Files
# -------------------------------
@st.cache_data
def load_geojson(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('type') != 'FeatureCollection':
            st.warning(f"{file_path} is not a valid FeatureCollection")
            return None
        return data
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
        # Ensure numeric types for critical columns
        prices_df['Price'] = pd.to_numeric(prices_df['Price'], errors='coerce')
        prices_df['Year'] = pd.to_numeric(prices_df['Year'], errors='coerce')
        prices_df['Month'] = pd.to_numeric(prices_df['Month'], errors='coerce')
        prices_df = prices_df.dropna(subset=['Price', 'Year', 'Month', 'Régions - Latitude', 'Régions - Longitude', 'commodity_id'])
        # Validate year range
        if not prices_df.empty and (prices_df['Year'] < 2016).any() or (prices_df['Year'] > 2025).any():
            st.warning("Farmgate data contains years outside 2016–2025. Invalid years will be filtered.")
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
        # Standardize column names and ensure numeric types
        retail_df = retail_df.rename(columns={'price': 'Price'})
        retail_df['Price'] = pd.to_numeric(retail_df['Price'], errors='coerce')
        retail_df['Year'] = pd.to_numeric(retail_df['Year'], errors='coerce')
        retail_df['Month'] = pd.to_numeric(retail_df['Month'], errors='coerce')
        retail_df = retail_df.dropna(subset=['Price', 'Year', 'Month', 'latitude', 'longitude', 'commodity_id'])
        # Validate year range
        if not retail_df.empty and (retail_df['Year'] < 2016).any() or (retail_df['Year'] > 2025).any():
            st.warning("Retail data contains years outside 2016–2025. Invalid years will be filtered.")
            retail_df = retail_df[retail_df['Year'].between(2016, 2025)]
        return retail_df
    except Exception as e:
        st.error(f"Error reading retail prices file {file_path}: {e}")
        return pd.DataFrame()

# -------------------------------
# Main Dashboard
# -------------------------------
def main():
    # Custom CSS for styling
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
    This interactive tool visualizes travel time, friction surfaces, market locations, road networks, and commodity prices across Senegal. 
    Use the filters to explore data and download insights for research and policy-making.
    """)

    # File Upload Section
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

    # Validate files
    for key, path in st.session_state.file_paths.items():
        if not validate_file(path, key.capitalize()):
            st.stop()

    # Load data with progress bar
    with st.spinner("Loading data..."):
        progress = st.progress(0)
        travel_time, travel_bounds = load_and_process_raster(st.session_state.file_paths['raster'])
        progress.progress(0.2)
        friction_data, friction_bounds = load_and_process_raster(st.session_state.file_paths['friction'])
        progress.progress(0.4)
        markets = load_geojson(st.session_state.file_paths['markets'])
        progress.progress(0.6)
        roads_filtered = load_geojson(st.session_state.file_paths['roads'])
        progress.progress(0.8)
        prices_df = load_price_data(st.session_state.file_paths['prices'])
        retail_df = load_retail_data(st.session_state.file_paths['prices'])
        progress.progress(1.0)

    if travel_time is None or friction_data is None:
        st.error("Failed to load raster data. Please check the files and try again.")
        st.stop()

    # Generate raster images
    travel_png_path, travel_image_bounds = generate_travel_image(travel_time, travel_bounds)
    friction_png_path, friction_image_bounds = generate_friction_image(friction_data, friction_bounds)

    # Validate price data columns
    farmgate_required_columns = ['Régions Name', 'Commodity', 'commodity_english', 'commodity_id', 'Price', 'Unit2', 'Régions - Latitude', 'Régions - Longitude', 'Year', 'Month']
    retail_required_columns = ['market', 'commodity', 'commodity_id', 'Price', 'Unit2', 'latitude', 'longitude', 'Year', 'Month']
    if not prices_df.empty and any(col not in prices_df.columns for col in farmgate_required_columns):
        st.error(f"Missing required columns in farmgate prices: {', '.join([col for col in farmgate_required_columns if col not in prices_df.columns])}")
        prices_df = pd.DataFrame()
    if not retail_df.empty and any(col not in retail_df.columns for col in retail_required_columns):
        st.error(f"Missing required columns in retail prices: {', '.join([col for col in retail_required_columns if col not in retail_df.columns])}")
        retail_df = pd.DataFrame()

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Interactive Map", "Data Summary", "Price Trends"])

    with tab1:
        st.subheader("Interactive Map")
        st.markdown("Explore travel time, friction surfaces, market locations, road networks, and commodity prices.")

        # Filter controls
        st.sidebar.header("Map Filters")
        # Hardcode years 2016–2025
        available_years = list(range(2016, 2026))
        selected_year = st.sidebar.selectbox("Select Year", available_years, index=len(available_years)-1, key="year_select", on_change=lambda: st.session_state.update({'map_data_updated': True}))

        month_names = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                       7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
        available_months = sorted(list(set(prices_df['Month'].unique()) | set(retail_df['Month'].unique()))) if not prices_df.empty or not retail_df.empty else list(range(1, 13))
        latest_month = max(available_months) if available_months else 12
        selected_month_name = st.sidebar.selectbox("Select Month", [month_names.get(m, str(m)) for m in available_months], index=available_months.index(latest_month) if latest_month in available_months else 0, key="month_select", on_change=lambda: st.session_state.update({'map_data_updated': True}))
        selected_month = next((k for k, v in month_names.items() if v == selected_month_name), selected_month_name)

        # Filter commodities to those present in both farmgate and retail based on commodity_id
        commodity_options = []
        commodity_id_to_name = {}
        if not prices_df.empty and not retail_df.empty:
            farmgate_commodities = set(prices_df['commodity_id'].unique())
            retail_commodities = set(retail_df['commodity_id'].unique())
            common_commodity_ids = farmgate_commodities.intersection(retail_commodities)
            # Map commodity_id to commodity_english for display
            for cid in common_commodity_ids:
                name = prices_df[prices_df['commodity_id'] == cid]['commodity_english'].iloc[0] if cid in prices_df['commodity_id'].values else retail_df[retail_df['commodity_id'] == cid]['commodity'].iloc[0]
                commodity_options.append(cid)
                commodity_id_to_name[cid] = name
            commodity_options = sorted(commodity_options, key=lambda x: commodity_id_to_name[x])
        if not commodity_options:
            st.error("No commodities found in both farmgate and retail datasets with common commodity IDs. Please check your data.")
            st.stop()

        # Display commodity names but store IDs
        selected_commodity_ids = st.sidebar.multiselect(
            "Select Commodities",
            options=commodity_options,
            format_func=lambda x: commodity_id_to_name.get(x, str(x)),
            default=commodity_options,
            key="commodity_select",
            on_change=lambda: st.session_state.update({'map_data_updated': True})
        )

        # Layer toggles
        st.sidebar.header("Map Layers")
        show_travel = st.sidebar.checkbox("Travel Time", value=True)
        show_friction = st.sidebar.checkbox("Friction Surface", value=True)
        show_roads = st.sidebar.checkbox("Roads", value=True)
        show_markets = st.sidebar.checkbox("Markets", value=True)
        show_farmgate = st.sidebar.checkbox("Farmgate Prices", value=True)
        show_retail = st.sidebar.checkbox("Retail Prices", value=True)

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

            st.session_state.latest_farmgate_prices = latest_farmgate_prices
            st.session_state.latest_retail_prices = latest_retail_prices
            st.session_state.map_data_updated = False

        # Render Map
        try:
            center_lat = (travel_bounds[1] + travel_bounds[3]) / 2
            center_lon = (travel_bounds[0] + travel_bounds[2]) / 2
            m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="CartoDB Positron")

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

            # Add Farmgate Prices Layer
            if show_farmgate and not st.session_state.latest_farmgate_prices.empty:
                farmgate_cluster = MarkerCluster(name="Farmgate Prices").add_to(m)
                valid_farmgate_markers = 0
                for _, row in st.session_state.latest_farmgate_prices.iterrows():
                    if pd.isna(row['Régions - Latitude']) or pd.isna(row['Régions - Longitude']):
                        continue
                    popup_text = f"<b>Region:</b> {row['Régions Name']}<br><b>Commodity:</b> {row['commodity_english']}<br><b>Price:</b> {row['Price']:.2f} {row['Unit2']}<br><b>Date:</b> {row['Year']}-{row['Month']:02d}"
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

            # Add Legends
            if show_travel:
                travel_legend_html = """
                {% macro html(this, kwargs) %}
                <div style="position: fixed; bottom: 5%; left: 2%; width: 180px; height: 230px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
                <b>Travel Time (min)</b><br><div style="margin-top:10px;">
                <div style="background:#ffffcc;width:20px;height:20px;display:inline-block;"></div> 0–10<br>
                <div style="background:#ffeda0;width:20px;height:20px;display:inline-block;"></div> 10–30<br>
                <div style="background:#feb24c;width:20px;height:20px;display:inline-block;"></div> 30–60<br>
                <div style="background:#fd8d3c;width:20px;height:20px;display:inline-block;"></div> 60–120<br>
                <div style="background:#f03b20;width:20px;height:20px;display:inline-block;"></div> 120–240<br>
                <div style="background:#bd0026;width:20px;height:20px;display:inline-block;"></div> 240–1440<br>
                <div style="background:#800026;width:20px;height:20px;display:inline-block;"></div> >1440</div></div>
                {% endmacro %}
                """
                travel_legend = MacroElement()
                travel_legend._template = Template(travel_legend_html)
                m.get_root().add_child(travel_legend)

            if show_friction:
                friction_legend_html = """
                {% macro html(this, kwargs) %}
                <div style="position: fixed; bottom: 5%; right: 2%; width: 200px; height: 260px; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
                <b>Friction (min/m)</b><br><div style="margin-top:10px;">
                <div style="background:#006837;width:20px;height:20px;display:inline-block;"></div> ≤ 0.001<br>
                <div style="background:#31a354;width:20px;height:20px;display:inline-block;"></div> ≤ 0.01<br>
                <div style="background:#78c679;width:20px;height:20px;display:inline-block;"></div> ≤ 0.1<br>
                <div style="background:#c2e699;width:20px;height:20px;display:inline-block;"></div> ≤ 0.5<br>
                <div style="background:#fdae61;width:20px;height:20px;display:inline-block;"></div> ≤ 1.0<br>
                <div style="background:#f46d43;width:20px;height:20px;display:inline-block;"></div> ≤ 2.0<br>
                <div style="background:#a50026;width:20px;height:20px;display:inline-block;"></div> ≤ 5.0<br>
                <div style="background:#800026;width:20px;height:20px;display:inline-block;"></div> > 5.0</div></div>
                {% endmacro %}
                """
                friction_legend = MacroElement()
                friction_legend._template = Template(friction_legend_html)
                m.get_root().add_child(friction_legend)

            # Add MiniMap and Fullscreen
            MiniMap(tiles='OpenStreetMap', position='bottomleft').add_to(m)
            Fullscreen(position='topright', title='Expand', title_cancel='Exit').add_to(m)

            folium.LayerControl(collapsed=False).add_to(m)
            with st.spinner("Rendering map..."):
                st_folium(m, width=1400, height=1000, key="folium_map")
        except Exception as e:
            st.error(f"Failed to render map: {e}. Please check data or coordinates.")

    with tab2:
        st.subheader("Data Summary")
        if not st.session_state.latest_farmgate_prices.empty:
            st.markdown("### Farmgate Prices")
            st.dataframe(st.session_state.latest_farmgate_prices[['Régions Name', 'commodity_english', 'Price', 'Unit2', 'Year', 'Month']])
            csv = st.session_state.latest_farmgate_prices.to_csv(index=False)
            st.download_button("Download Farmgate Prices", csv, "farmgate_prices.csv", "text/csv")
            st.markdown("#### Statistics")
            stats = st.session_state.latest_farmgate_prices.groupby('commodity_english')['Price'].agg(['mean', 'min', 'max']).reset_index()
            st.dataframe(stats)
        else:
            st.warning("No farmgate price data available for the selected filters.")

        if not st.session_state.latest_retail_prices.empty:
            st.markdown("### Retail Prices")
            st.dataframe(st.session_state.latest_retail_prices[['market', 'commodity', 'Price', 'Unit2', 'Year', 'Month']])
            csv = st.session_state.latest_retail_prices.to_csv(index=False)
            st.download_button("Download Retail Prices", csv, "retail_prices.csv", "text/csv")
            st.markdown("#### Statistics")
            stats = st.session_state.latest_retail_prices.groupby('commodity')['Price'].agg(['mean', 'min', 'max']).reset_index()
            st.dataframe(stats)
        else:
            st.warning("No retail price data available for the selected filters.")

        st.markdown("### Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Markets", len(markets['features']) if markets else 0)
        col2.metric("Road Features", len(roads_filtered['features']) if roads_filtered else 0)
        col3.metric("Price Points", len(st.session_state.latest_farmgate_prices) + len(st.session_state.latest_retail_prices))

    with tab3:
        st.subheader("Price Trends")
        if not prices_df.empty and not retail_df.empty and commodity_options:
            st.markdown("### Farmgate and Retail Price Trends")
            selected_commodity_id = st.selectbox(
                "Select Commodity",
                options=commodity_options,
                format_func=lambda x: commodity_id_to_name.get(x, str(x)),
                key="trend_commodity_select"
            )
            try:
                # Process farmgate trends
                farmgate_trend = prices_df[prices_df['commodity_id'] == selected_commodity_id][['Year', 'Month', 'Price']].groupby(['Year', 'Month']).mean().reset_index()
                if not farmgate_trend.empty:
                    farmgate_trend['Date'] = pd.to_datetime(farmgate_trend[['Year', 'Month']].assign(day=1), errors='coerce')
                    farmgate_trend = farmgate_trend.dropna(subset=['Date'])
                    farmgate_trend['Price Type'] = 'Farmgate'

                # Process retail trends
                retail_trend = retail_df[retail_df['commodity_id'] == selected_commodity_id][['Year', 'Month', 'Price']].groupby(['Year', 'Month']).mean().reset_index()
                if not retail_trend.empty:
                    retail_trend['Date'] = pd.to_datetime(retail_trend[['Year', 'Month']].assign(day=1), errors='coerce')
                    retail_trend = retail_trend.dropna(subset=['Date'])
                    retail_trend['Price Type'] = 'Retail'

                # Combine trends
                combined_trend = pd.concat([farmgate_trend, retail_trend], ignore_index=True) if not farmgate_trend.empty or not retail_trend.empty else pd.DataFrame()

                if combined_trend.empty:
                    st.warning(f"No trend data available for {commodity_id_to_name.get(selected_commodity_id, selected_commodity_id)}.")
                else:
                    # Create Plotly figure
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

                    # Customize layout
                    fig.update_layout(
                        title=f"{commodity_id_to_name.get(selected_commodity_id, selected_commodity_id)} Price Trends",
                        xaxis_title="Date",
                        yaxis_title="Average Price (Unit2)",
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
        <p>Developed by xAI in collaboration with IFPRI | Data Sources: IFPRI, OpenStreetMap | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()