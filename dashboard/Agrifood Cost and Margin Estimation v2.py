import os
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import folium_static
import rasterio
import numpy as np
import geopandas as gpd
from PIL import Image
from branca.element import Template, MacroElement
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Senegal Commodity and Geospatial Map", layout="wide")

# Cache data loading for performance
@st.cache_data
def load_commodity_data(file_path='Senegal_Merged_Food_Prices.xlsx'):
    try:
        # Load merged data
        if not os.path.exists(file_path):
            st.error(f"File '{file_path}' not found.")
            return None
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Required columns
        required_columns = [
            'date', 'admin1', 'admin2', 'market', 'market_id', 'latitude', 'longitude',
            'category', 'commodity_retail', 'commodity_id', 'unit_retail', 'priceflag',
            'pricetype', 'currency', 'price_retail', 'usdprice', 'unit2_retail',
            'year', 'month', 'commodity_farmgate_en', 'region_name', 'region_id',
            'region_latitude', 'region_longitude', 'price_farmgate', 'unit_farmgate', 'unit2_farmgate'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None

        # Validate and process data
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['region_latitude'] = pd.to_numeric(df['region_latitude'], errors='coerce')
        df['region_longitude'] = pd.to_numeric(df['region_longitude'], errors='coerce')
        df['price_retail'] = pd.to_numeric(df['price_retail'], errors='coerce')
        df['price_farmgate'] = pd.to_numeric(df['price_farmgate'], errors='coerce')
        df['unit_retail'] = df['unit_retail'].astype(str).fillna('Unknown')
        df['unit_farmgate'] = df['unit_farmgate'].astype(str).fillna('Unknown')
        df['unit2_retail'] = df['unit2_retail'].astype(str).fillna('Unknown')
        df['unit2_farmgate'] = df['unit2_farmgate'].astype(str).fillna('Unknown')

        # Normalize commodity names to prevent duplicates due to formatting
        df['commodity_retail'] = df['commodity_retail'].str.strip().str.replace(r'\s+', ' ', regex=True).str.title()
        df['commodity_farmgate_en'] = df['commodity_farmgate_en'].str.strip().str.replace(r'\s+', ' ', regex=True).str.title()

        # Deduplicate retail data
        retail_cols = [
            'market_id', 'year', 'month', 'commodity_retail', 'price_retail', 'unit_retail', 'unit2_retail',
            'latitude', 'longitude', 'market', 'admin1', 'admin2', 'category', 'commodity_id',
            'priceflag', 'pricetype', 'currency', 'usdprice'
        ]
        retail_df = df[df['commodity_retail'].notna()][retail_cols].drop_duplicates().groupby(
            ['market_id', 'year', 'month', 'commodity_retail']
        ).agg({
            'price_retail': 'mean',
            'unit_retail': 'first',
            'unit2_retail': 'first',
            'latitude': 'first',
            'longitude': 'first',
            'market': 'first',
            'admin1': 'first',
            'admin2': 'first',
            'category': 'first',
            'commodity_id': 'first',
            'priceflag': 'first',
            'pricetype': 'first',
            'currency': 'first',
            'usdprice': 'mean'
        }).reset_index()

        # Deduplicate farmgate data
        farmgate_cols = [
            'region_id', 'year', 'month', 'commodity_farmgate_en', 'price_farmgate', 'unit_farmgate',
            'unit2_farmgate', 'region_latitude', 'region_longitude', 'region_name'
        ]
        farmgate_df = df[df['commodity_farmgate_en'].notna()][farmgate_cols].drop_duplicates().groupby(
            ['region_id', 'year', 'month', 'commodity_farmgate_en']
        ).agg({
            'price_farmgate': 'mean',
            'unit_farmgate': 'first',
            'unit2_farmgate': 'first',
            'region_latitude': 'first',
            'region_longitude': 'first',
            'region_name': 'first'
        }).reset_index()

        # Combine for display, preserving all columns
        df = retail_df.merge(
            farmgate_df,
            on=['year', 'month'],
            how='outer',
            suffixes=('', '_farmgate')
        )

        # Rename columns to avoid suffix conflicts
        if 'price_retail_farmgate' in df.columns:
            df = df.rename(columns={'price_retail_farmgate': 'price_farmgate'})
        if 'unit_retail_farmgate' in df.columns:
            df = df.rename(columns={'unit_retail_farmgate': 'unit_farmgate'})
        if 'unit2_retail_farmgate' in df.columns:
            df = df.rename(columns={'unit2_retail_farmgate': 'unit2_farmgate'})

        # Check for duplicates after merging
        duplicates = df[df.duplicated(subset=['market_id', 'year', 'month', 'commodity_retail', 'region_id', 'commodity_farmgate_en'], keep=False)]
        if not duplicates.empty:
            st.warning(f"Removed {len(duplicates)} duplicate entries after merging: {duplicates[['market_id', 'region_id', 'commodity_retail', 'commodity_farmgate_en']].to_dict('records')}")
            df = df.drop_duplicates(subset=['market_id', 'year', 'month', 'commodity_retail', 'region_id', 'commodity_farmgate_en'])

        # Check for invalid coordinates
        invalid_market_coords = df[df['latitude'].isna() | df['longitude'].isna()]
        if not invalid_market_coords.empty:
            st.warning(f"Found {len(invalid_market_coords)} rows with invalid market coordinates")
        invalid_region_coords = df[df['region_latitude'].isna() | df['region_longitude'].isna()]
        if not invalid_region_coords.empty:
            st.warning(f"Found {len(invalid_region_coords)} rows with invalid region coordinates")

        # Check for invalid prices
        invalid_retail_prices = df[df['price_retail'].isna() & df['commodity_retail'].notna()]
        if not invalid_retail_prices.empty:
            st.warning(f"Found {len(invalid_retail_prices)} rows with invalid retail prices")
        invalid_farmgate_prices = df[df['price_farmgate'].isna() & df['commodity_farmgate_en'].notna()]
        if not invalid_farmgate_prices.empty:
            st.warning(f"Found {len(invalid_farmgate_prices)} rows with invalid farmgate prices")

        return df
    except Exception as e:
        st.error(f"Error loading commodity data: {str(e)}")
        return None

@st.cache_data
def load_geospatial_data(raster_path, friction_path, markets_path, roads_path):
    try:
        # Initialize variables
        travel_data, travel_bounds, friction_data, friction_bounds, markets, roads = None, None, None, None, None, None

        # Debug: Check file existence
        st.sidebar.write(f"Travel raster exists: {os.path.exists(raster_path)}")
        st.sidebar.write(f"Friction raster exists: {os.path.exists(friction_path)}")
        st.sidebar.write(f"Markets GeoJSON exists: {os.path.exists(markets_path)}")
        st.sidebar.write(f"Roads GeoJSON exists: {os.path.exists(roads_path)}")

        # Load travel time raster if file exists
        if os.path.exists(raster_path):
            with rasterio.open(raster_path) as src:
                travel_time = src.read(1)
                travel_nodata = src.nodata
                travel_bounds = src.bounds
                travel_data = np.ma.masked_equal(travel_time, travel_nodata) if travel_nodata else np.ma.masked_invalid(travel_time)
                st.sidebar.write(f"Travel raster loaded: Shape {travel_time.shape}, Nodata {travel_nodata}, Bounds {travel_bounds}")
        else:
            st.warning(f"Travel time raster file not found: {raster_path}. Continuing without travel time layer.")

        # Load friction raster if file exists
        if os.path.exists(friction_path):
            with rasterio.open(friction_path) as src:
                friction_data = src.read(1)
                friction_nodata = src.nodata
                friction_bounds = src.bounds
                friction_data = np.ma.masked_equal(friction_data, friction_nodata) if friction_nodata else np.ma.masked_invalid(friction_data)
                st.sidebar.write(f"Friction raster loaded: Shape {friction_data.shape}, Nodata {friction_nodata}, Bounds {friction_bounds}")
        else:
            st.warning(f"Friction raster file not found: {friction_path}. Continuing without friction layer.")

        # Load GeoJSON files if they exist
        if os.path.exists(markets_path):
            markets = gpd.read_file(markets_path)
            st.sidebar.write(f"Markets GeoJSON loaded: {len(markets)} features")
        else:
            st.warning(f"Markets GeoJSON file not found: {markets_path}. Continuing without markets layer.")

        if os.path.exists(roads_path):
            roads = gpd.read_file(roads_path)
            st.sidebar.write(f"Roads GeoJSON loaded: {len(roads)} features")
        else:
            st.warning(f"Roads GeoJSON file not found: {roads_path}. Continuing without roads layer.")

        return travel_data, travel_bounds, friction_data, friction_bounds, markets, roads
    except Exception as e:
        st.error(f"Error loading geospatial data: {str(e)}")
        return None, None, None, None, None, None

def generate_raster_images(travel_data, friction_data, travel_bounds, friction_bounds):
    travel_png_path, friction_png_path, image_bounds = None, None, None
    travel_breaks = [0, 10, 30, 60, 120, 240, 1440, np.inf]
    travel_colors = [
        (255, 255, 204), (255, 237, 160), (254, 178, 76), (253, 141, 60),
        (240, 59, 32), (189, 0, 38), (128, 0, 38)
    ]
    friction_breaks = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, np.inf]
    friction_colors = [
        (0, 104, 55), (49, 163, 84), (120, 198, 121), (194, 230, 153),
        (253, 174, 97), (244, 109, 67), (165, 0, 38), (128, 0, 38)
    ]
    if travel_data is not None and travel_bounds is not None:
        try:
            travel_rgb = np.zeros((travel_data.shape[0], travel_data.shape[1], 3), dtype=np.uint8)
            for i in range(len(travel_breaks) - 1):
                mask = (travel_data >= travel_breaks[i]) & (travel_data < travel_breaks[i+1])
                for j in range(3):
                    travel_rgb[:, :, j][mask] = travel_colors[i][j]
            travel_png_path = 'travel_time_colored.png'
            Image.fromarray(travel_rgb).save(travel_png_path)
            st.sidebar.write(f"Travel PNG generated: {travel_png_path}")
        except Exception as e:
            st.warning(f"Failed to generate travel PNG: {str(e)}")
    else:
        st.warning("Travel data or bounds not available. Skipping travel PNG generation.")

    if friction_data is not None and friction_bounds is not None:
        try:
            friction_rgb = np.zeros((friction_data.shape[0], friction_data.shape[1], 3), dtype=np.uint8)
            for i in range(len(friction_breaks) - 1):
                mask = (friction_data >= friction_breaks[i]) & (friction_data < friction_breaks[i+1])
                for j in range(3):
                    friction_rgb[:, :, j][mask] = friction_colors[i][j]
            friction_png_path = 'friction_surface_colored.png'
            Image.fromarray(friction_rgb).save(friction_png_path)
            st.sidebar.write(f"Friction PNG generated: {friction_png_path}")
        except Exception as e:
            st.warning(f"Failed to generate friction PNG: {str(e)}")
    else:
        st.warning("Friction data or bounds not available. Skipping friction PNG generation.")

    if travel_bounds is not None:
        image_bounds = [[travel_bounds.bottom, travel_bounds.left], [travel_bounds.top, travel_bounds.right]]
        st.sidebar.write(f"Image bounds set from travel: {image_bounds}")
    elif friction_bounds is not None:
        image_bounds = [[friction_bounds.bottom, friction_bounds.left], [friction_bounds.top, friction_bounds.right]]
        st.sidebar.write(f"Image bounds set from friction: {image_bounds}")
    else:
        st.warning("No valid bounds available for raster overlays.")

    return travel_png_path, friction_png_path, image_bounds

def generate_map(df, year, month, map_style, travel_png_path, friction_png_path, image_bounds, markets, roads, selected_commodities):
    m = folium.Map(location=[14.5, -14.5], zoom_start=7.3, tiles=map_style)
    locations_mapped = []

    # Filter data for selected year, month, and commodities
    filtered_df = df[(df['year'] == year) & (df['month'] == month) & 
                    ((df['commodity_retail'].isin(selected_commodities)) | (df['commodity_farmgate_en'].isin(selected_commodities)))]

    if filtered_df.empty:
        st.warning(f"No data found for Year {year}, Month {month}, and selected commodities.")
        return m, locations_mapped, filtered_df

    # Process market-level (retail) data
    retail_columns = ['market', 'market_id', 'latitude', 'longitude', 'commodity_retail', 'year', 'month', 'price_retail', 'unit2_retail']
    available_retail_cols = [col for col in retail_columns if col in filtered_df.columns]
    if not all(col in filtered_df.columns for col in ['market', 'market_id', 'latitude', 'longitude', 'commodity_retail']):
        st.warning("Missing required retail columns in filtered data. Skipping retail processing.")
    else:
        market_grouped = filtered_df[available_retail_cols].groupby(
            ['market', 'market_id', 'latitude', 'longitude', 'commodity_retail', 'year', 'month']
        ).agg({
            'price_retail': 'mean' if 'price_retail' in available_retail_cols else lambda x: np.nan,
            'unit2_retail': 'first' if 'unit2_retail' in available_retail_cols else lambda x: 'Unknown'
        }).reset_index()
        market_grouped = market_grouped.groupby(['market', 'market_id', 'latitude', 'longitude']).agg({
            'commodity_retail': list,
            'price_retail': list,
            'unit2_retail': list
        }).reset_index()
        market_grouped['commodity_count'] = market_grouped['commodity_retail'].apply(len)

        # Check for duplicate commodities in market_grouped
        for _, row in market_grouped.iterrows():
            unique_commodities = set(row['commodity_retail'])
            if len(unique_commodities) < len(row['commodity_retail']):
                st.warning(f"Duplicate retail commodities found for market {row['market']} (ID: {row['market_id']}): {row['commodity_retail']}")

        locations_mapped.extend(market_grouped['market'].tolist())

        for _, row in market_grouped.iterrows():
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                continue
            commodity_count = row['commodity_count']
            commodity_details = [
                f"{commodity}: {price:.2f} {unit}" if not pd.isna(price) else f"{commodity}: Price not available"
                for commodity, price, unit in zip(row['commodity_retail'], row['price_retail'], row['unit2_retail'])
            ]
            commodity_list = '<br>'.join(commodity_details)
            popup_content = f"""
            <div style='width: 250px'>
                <h4>{row['market']} (Market)</h4>
                <b>Market ID:</b> {row['market_id']}<br>
                <b>Retail Commodities ({commodity_count}):</b><br>{commodity_list}<br>
                <b>Coordinates:</b> {row['latitude']:.4f}, {row['longitude']:.4f}
            </div>
            """
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6 + (commodity_count * 1.5),
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{row['market']}: {commodity_count} retail commodities (Market)",
                fill=True,
                fill_color='green',
                color='green',
                fill_opacity=0.7
            ).add_to(folium.FeatureGroup(name="Market Retail Commodities").add_to(m))

    # Process region-level (farmgate) data
    farmgate_columns = ['region_id', 'commodity_farmgate_en', 'year', 'month', 'price_farmgate', 'unit2_farmgate', 'region_latitude', 'region_longitude', 'region_name']
    available_farmgate_cols = [col for col in farmgate_columns if col in filtered_df.columns]
    if not all(col in filtered_df.columns for col in ['region_id', 'commodity_farmgate_en', 'region_name']):
        st.warning("Missing required farmgate columns in filtered data. Skipping farmgate processing.")
    else:
        region_grouped = filtered_df[available_farmgate_cols].groupby(
            ['region_id', 'commodity_farmgate_en', 'year', 'month']
        ).agg({
            'price_farmgate': 'mean' if 'price_farmgate' in available_farmgate_cols else lambda x: np.nan,
            'unit2_farmgate': 'first' if 'unit2_farmgate' in available_farmgate_cols else lambda x: 'Unknown',
            'region_latitude': 'first' if 'region_latitude' in available_farmgate_cols else lambda x: np.nan,
            'region_longitude': 'first' if 'region_longitude' in available_farmgate_cols else lambda x: np.nan,
            'region_name': 'first'
        }).reset_index()
        region_grouped = region_grouped.groupby(['region_name', 'region_id', 'region_latitude', 'region_longitude']).agg({
            'commodity_farmgate_en': list,
            'price_farmgate': list,
            'unit2_farmgate': list
        }).reset_index()
        region_grouped['commodity_count'] = region_grouped['commodity_farmgate_en'].apply(len)

        # Check for duplicate commodities in region_grouped
        for _, row in region_grouped.iterrows():
            unique_commodities = set(row['commodity_farmgate_en'])
            if len(unique_commodities) < len(row['commodity_farmgate_en']):
                st.warning(f"Duplicate farmgate commodities found for region {row['region_name']} (ID: {row['region_id']}): {row['commodity_farmgate_en']}")

        locations_mapped.extend(region_grouped['region_name'].tolist())

        for _, row in region_grouped.iterrows():
            if pd.isna(row['region_latitude']) or pd.isna(row['region_longitude']):
                continue
            commodity_count = row['commodity_count']
            color = 'blue' if commodity_count < 5 else 'orange' if commodity_count < 10 else 'red'
            commodity_details = [
                f"{commodity}: {price:.2f} {unit}" if not pd.isna(price) else f"{commodity}: Price not available"
                for commodity, price, unit in zip(row['commodity_farmgate_en'], row['price_farmgate'], row['unit2_farmgate'])
            ]
            commodity_list = '<br>'.join(commodity_details)
            popup_content = f"""
            <div style='width: 250px'>
                <h4>{row['region_name']} (Region)</h4>
                <b>Region ID:</b> {row['region_id']}<br>
                <b>Farmgate Commodities ({commodity_count}):</b><br>{commodity_list}<br>
                <b>Coordinates:</b> {row['region_latitude']:.4f}, {row['region_longitude']:.4f}
            </div>
            """
            folium.CircleMarker(
                location=[row['region_latitude'], row['region_longitude']],
                radius=8 + (commodity_count * 2),
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{row['region_name']}: {commodity_count} farmgate commodities (Region)",
                fill=True,
                fill_color=color,
                color=color,
                fill_opacity=0.7
            ).add_to(folium.FeatureGroup(name="Region Farmgate Commodities").add_to(m))

    # Add raster overlays with validation
    if travel_png_path and image_bounds and os.path.exists(travel_png_path):
        try:
            folium.raster_layers.ImageOverlay(
                name="Travel Time",
                image=travel_png_path,
                bounds=image_bounds,
                opacity=0.6,
                interactive=True,
                cross_origin=False
            ).add_to(m)
            st.sidebar.write("Travel time raster layer added to map")
            travel_legend_html = """
            {% macro html(this, kwargs) %}
            <div style="
                position: fixed;
                bottom: 20px;
                left: 20px;
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
        except Exception as e:
            st.warning(f"Failed to add travel time raster layer: {str(e)}")
    else:
        st.warning(f"Travel raster not added: PNG exists: {os.path.exists(travel_png_path) if travel_png_path else False}, Bounds: {image_bounds is not None}")

    if friction_png_path and image_bounds and os.path.exists(friction_png_path):
        try:
            folium.raster_layers.ImageOverlay(
                name="Friction Surface (min/m)",
                image=friction_png_path,
                bounds=image_bounds,
                opacity=0.7,
                interactive=True,
                cross_origin=False
            ).add_to(m)
            st.sidebar.write("Friction surface raster layer added to map")
            friction_legend_html = """
            {% macro html(this, kwargs) %}
            <div style="
                position: fixed;
                bottom: 20px;
                right: 20px;
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
              <div style="background:#800026;width:20px;height:20px;display:inline-block;"></div> > 5.0
            </div>
            </div>
            {% endmacro %}
            """
            friction_legend = MacroElement()
            friction_legend._template = Template(friction_legend_html)
            m.get_root().add_child(friction_legend)
        except Exception as e:
            st.warning(f"Failed to add friction surface raster layer: {str(e)}")
    else:
        st.warning(f"Friction raster not added: PNG exists: {os.path.exists(friction_png_path) if friction_png_path else False}, Bounds: {image_bounds is not None}")

    # Add legends for markers
    commodity_legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        top: 20px;
        right: 20px;
        width: 180px;
        height: 110px;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
    <b>Farmgate Commodity Count</b><br>
    <div style="margin-top:10px;">
      <div style="background:#0000ff;width:20px;height:20px;display:inline-block;"></div> <5 Commodities<br>
      <div style="background:#ffa500;width:20px;height:20px;display:inline-block;"></div> 5–9 Commodities<br>
      <div style="background:#ff0000;width:20px;height:20px;display:inline-block;"></div> ≥10 Commodities
    </div>
    </div>
    {% endmacro %}
    """
    commodity_legend = MacroElement()
    commodity_legend._template = Template(commodity_legend_html)
    m.get_root().add_child(commodity_legend)

    market_legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed;
        top: 140px;
        right: 20px;
        width: 180px;
        height: 70px;
        background-color: white;
        border:2px solid grey;
        z-index:9999;
        font-size:14px;
        padding: 10px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
    <b>Retail Commodities</b><br>
    <div style="margin-top:10px;">
      <div style="background:#008000;width:20px;height:20px;display:inline-block;"></div> Markets
    </div>
    </div>
    {% endmacro %}
    """
    market_legend = MacroElement()
    market_legend._template = Template(market_legend_html)
    m.get_root().add_child(market_legend)

    # Add roads and markets layers
    if roads is not None:
        folium.GeoJson(
            roads,
            name="Roads",
            style_function=lambda x: {'color': 'blue', 'weight': 1, 'opacity': 0.7}
        ).add_to(m)
    if markets is not None:
        for _, row in markets.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=row['market'],
                icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')
            ).add_to(folium.FeatureGroup(name="Markets").add_to(m))

    folium.LayerControl().add_to(m)
    return m, locations_mapped, filtered_df

def main():
    st.title("Senegal Commodity and Geospatial Analysis Map")
    st.markdown("Explore retail and farmgate commodity prices, travel time to cities, and friction surfaces across Senegal.")

    # Sidebar controls
    st.sidebar.title("Map Controls")
    map_style = st.sidebar.selectbox("Map Style", ["OpenStreetMap", "CartoDB Positron", "Stamen Terrain"], index=1)

    # Load commodity data
    df = load_commodity_data()
    if df is None:
        return

    # Debug: Display loaded columns
    st.sidebar.write(f"Loaded columns: {df.columns.tolist()}")

    # Commodity filter
    commodities = sorted(set(df['commodity_retail'].dropna().unique()).union(set(df['commodity_farmgate_en'].dropna().unique())))
    default_commodities = commodities[:5] if len(commodities) >= 5 else commodities
    select_all = st.sidebar.checkbox("Select All Commodities", value=False)
    selected_commodities = st.sidebar.multiselect("Select Commodities", commodities, 
                                                 default=commodities if select_all else default_commodities)

    # Load geospatial data
    raster_path = '201501_Global_Travel_Time_to_Cities_SEN.tiff'
    friction_path = '201501_Global_Travel_Speed_Friction_Surface_SEN.tiff'
    markets_path = 'markets_from_excel.geojson'
    roads_path = 'roads_filtered.geojson'
    travel_data, travel_bounds, friction_data, friction_bounds, markets, roads = load_geospatial_data(
        raster_path, friction_path, markets_path, roads_path
    )

    # Generate raster images
    travel_png_path, friction_png_path, image_bounds = generate_raster_images(
        travel_data, friction_data, travel_bounds, friction_bounds
    )

    # Display travel time statistics
    if travel_data is not None:
        st.sidebar.subheader("Travel Time Statistics")
        st.sidebar.write(f"Min: {travel_data.min():.2f} min")
        st.sidebar.write(f"Max: {travel_data.max():.2f} min")
        st.sidebar.write(f"Mean: {travel_data.mean():.2f} min")
        st.sidebar.write(f"Std Dev: {travel_data.std():.2f} min")
        percentiles = np.percentile(travel_data.compressed(), [5, 25, 50, 75, 95])
        st.sidebar.write("Percentiles:")
        st.sidebar.write(f"5th: {percentiles[0]:.2f} min")
        st.sidebar.write(f"25th: {percentiles[1]:.2f} min")
        st.sidebar.write(f"50th (median): {percentiles[2]:.2f} min")
        st.sidebar.write(f"75th: {percentiles[3]:.2f} min")
        st.sidebar.write(f"95th: {percentiles[4]:.2f} min")

    # Year and month selection
    years = sorted(df['year'].dropna().unique().astype(int))
    months = sorted(df['month'].dropna().unique().astype(int))
    if not years or not months:
        st.error("No valid years or months found in the data.")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox("Select Year", years, index=len(years)-1)
    with col2:
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
            7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        month_options = [(m, month_names[m]) for m in months]
        selected_month = st.selectbox("Select Month", [m[1] for m in month_options], 
                                      index=len(month_options)-1 if month_options else 0)
        selected_month_num = next(m[0] for m in month_options if m[1] == selected_month)

    # Generate and display map
    st.subheader(f"Map for {selected_month} {selected_year}")
    map_obj, locations_mapped, filtered_df = generate_map(
        df, selected_year, selected_month_num, map_style, 
        travel_png_path, friction_png_path, image_bounds, markets, roads, selected_commodities
    )

    if map_obj:
        folium_static(map_obj, width=1000, height=600)
        if locations_mapped:
            st.write(f"Locations mapped: {', '.join(set(locations_mapped))}")
        else:
            st.write("No commodity data to display for the selected year, month, and commodities.")
        
        # Plot retail and farmgate prices for the selected year
        st.subheader(f"Price Comparison for {selected_year}")
        if not filtered_df.empty:
            # Filter data for the selected year (all months)
            year_df = df[(df['year'] == selected_year) & 
                         ((df['commodity_retail'].isin(selected_commodities)) | 
                          (df['commodity_farmgate_en'].isin(selected_commodities)))]

            if year_df.empty:
                st.warning(f"No data available for selected commodities in {selected_year}.")
            else:
                # Aggregate retail prices
                retail_price_cols = ['commodity_retail', 'price_retail', 'unit2_retail']
                retail_price_df = year_df[year_df['commodity_retail'].notna()][retail_price_cols].groupby(
                    'commodity_retail'
                ).agg({
                    'price_retail': 'mean',
                    'unit2_retail': 'first'
                }).reset_index()

                # Aggregate farmgate prices
                farmgate_price_cols = ['commodity_farmgate_en', 'price_farmgate', 'unit2_farmgate']
                farmgate_price_df = year_df[year_df['commodity_farmgate_en'].notna()][farmgate_price_cols].groupby(
                    'commodity_farmgate_en'
                ).agg({
                    'price_farmgate': 'mean',
                    'unit2_farmgate': 'first'
                }).reset_index()

                # Combine commodities
                all_commodities = sorted(set(retail_price_df['commodity_retail'].dropna()).union(set(farmgate_price_df['commodity_farmgate_en'].dropna())))

                # Prepare data for plotting
                retail_prices = []
                farmgate_prices = []
                units = []
                for commodity in all_commodities:
                    retail_row = retail_price_df[retail_price_df['commodity_retail'] == commodity]
                    farmgate_row = farmgate_price_df[farmgate_price_df['commodity_farmgate_en'] == commodity]
                    
                    retail_price = retail_row['price_retail'].iloc[0] if not retail_row.empty else np.nan
                    farmgate_price = farmgate_row['price_farmgate'].iloc[0] if not farmgate_row.empty else np.nan
                    unit = retail_row['unit2_retail'].iloc[0] if not retail_row.empty else farmgate_row['unit2_farmgate'].iloc[0] if not farmgate_row.empty else 'Unknown'
                    
                    retail_prices.append(retail_price)
                    farmgate_prices.append(farmgate_price)
                    units.append(unit)

                # Create grouped bar plot
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=all_commodities,
                    y=retail_prices,
                    name='Retail Price',
                    marker_color='green',
                    text=[f"{p:.2f}" if not pd.isna(p) else "N/A" for p in retail_prices],
                    textposition='auto'
                ))
                fig.add_trace(go.Bar(
                    x=all_commodities,
                    y=farmgate_prices,
                    name='Farmgate Price',
                    marker_color='blue',
                    text=[f"{p:.2f}" if not pd.isna(p) else "N/A" for p in farmgate_prices],
                    textposition='auto'
                ))

                fig.update_layout(
                    title=f"Average Retail and Farmgate Prices for {selected_year}",
                    xaxis_title="Commodity",
                    yaxis_title="Price (FCFA)",
                    barmode='group',
                    xaxis_tickangle=-45,
                    height=600,
                    legend=dict(x=0.01, y=0.99),
                    margin=dict(b=150)
                )

                # Add unit annotations to x-axis
                for i, commodity in enumerate(all_commodities):
                    fig.add_annotation(
                        x=commodity,
                        y=-0.1,
                        xref="x",
                        yref="paper",
                        text=units[i],
                        showarrow=False,
                        font=dict(size=10),
                        xanchor='center',
                        yanchor='top'
                    )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No commodity data available for the selected year and commodities.")
    else:
        st.write("Unable to generate map due to missing data.")

if __name__ == "__main__":
    main()