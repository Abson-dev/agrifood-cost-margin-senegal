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

# Set page configuration
st.set_page_config(page_title="Senegal Commodity and Geospatial Map", layout="wide")

# Cache data loading for performance
@st.cache_data
def load_commodity_data(input_file='commodity_prices_merged.xlsx'):
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
        required_columns = ['Year', 'Month', 'Commodity', 'Régions Name', 
                           'Régions - RegionId', 'Régions - Latitude', 'Régions - Longitude', 'Price', 'Unit']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
        # Validate data types
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
        df['Régions - Latitude'] = pd.to_numeric(df['Régions - Latitude'], errors='coerce')
        df['Régions - Longitude'] = pd.to_numeric(df['Régions - Longitude'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Unit'] = df['Unit'].astype(str).fillna('Unknown')
        # Check for invalid data
        invalid_coords = df[df['Régions - Latitude'].isna() | df['Régions - Longitude'].isna()]
        if not invalid_coords.empty:
            st.warning(f"Found {len(invalid_coords)} rows with invalid coordinates")
        invalid_prices = df[df['Price'].isna()]
        if not invalid_prices.empty:
            st.warning(f"Found {len(invalid_prices)} rows with invalid or missing prices")
        return df
    except FileNotFoundError:
        st.error(f"Input file '{input_file}' not found.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

@st.cache_data
def load_geospatial_data(raster_path, friction_path, markets_path, roads_path):
    try:
        # Initialize variables
        travel_data, travel_bounds, friction_data, friction_bounds, markets, roads = None, None, None, None, None, None

        # Load travel time raster if file exists
        if os.path.exists(raster_path):
            with rasterio.open(raster_path) as src:
                travel_time = src.read(1)
                travel_nodata = src.nodata
                travel_bounds = src.bounds
                travel_data = np.ma.masked_equal(travel_time, travel_nodata) if travel_nodata else np.ma.masked_invalid(travel_time)
        else:
            st.warning(f"Travel time raster file not found: {raster_path}. Continuing without travel time layer.")

        # Load friction raster if file exists
        if os.path.exists(friction_path):
            with rasterio.open(friction_path) as src:
                friction_data = src.read(1)
                friction_nodata = src.nodata
                friction_bounds = src.bounds
                friction_data = np.ma.masked_equal(friction_data, friction_nodata) if friction_nodata else np.ma.masked_invalid(friction_data)
        else:
            st.warning(f"Friction raster file not found: {friction_path}. Continuing without friction layer.")

        # Load GeoJSON files if they exist
        if os.path.exists(markets_path):
            markets = gpd.read_file(markets_path)
        else:
            st.warning(f"Markets GeoJSON file not found: {markets_path}. Continuing without markets layer.")

        if os.path.exists(roads_path):
            roads = gpd.read_file(roads_path)
        else:
            st.warning(f"Roads GeoJSON file not found: {roads_path}. Continuing without roads layer.")

        return travel_data, travel_bounds, friction_data, friction_bounds, markets, roads
    except Exception as e:
        st.error(f"Error loading geospatial data: {str(e)}")
        return None, None, None, None, None, None

def generate_raster_images(travel_data, friction_data, travel_bounds, friction_bounds):
    """Generate PNG images for raster layers without caching due to unhashable inputs."""
    travel_png_path, friction_png_path, image_bounds = None, None, None

    # Travel time breakpoints and colors
    travel_breaks = [0, 10, 30, 60, 120, 240, 1440, np.inf]
    travel_colors = [
        (255, 255, 204), (255, 237, 160), (254, 178, 76), (253, 141, 60),
        (240, 59, 32), (189, 0, 38), (128, 0, 38)
    ]
    # Friction breakpoints and colors
    friction_breaks = [0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, np.inf]
    friction_colors = [
        (0, 104, 55), (49, 163, 84), (120, 198, 121), (194, 230, 153),
        (253, 174, 97), (244, 109, 67), (165, 0, 38), (128, 0, 38)
    ]

    # Generate travel time RGB image if data exists
    if travel_data is not None and travel_bounds is not None:
        travel_rgb = np.zeros((travel_data.shape[0], travel_data.shape[1], 3), dtype=np.uint8)
        for i in range(len(travel_breaks) - 1):
            mask = (travel_data >= travel_breaks[i]) & (travel_data < travel_breaks[i+1])
            for j in range(3):
                travel_rgb[:, :, j][mask] = travel_colors[i][j]
        travel_png_path = 'travel_time_colored.png'
        Image.fromarray(travel_rgb).save(travel_png_path)

    # Generate friction RGB image if data exists
    if friction_data is not None and friction_bounds is not None:
        friction_rgb = np.zeros((friction_data.shape[0], friction_data.shape[1], 3), dtype=np.uint8)
        for i in range(len(friction_breaks) - 1):
            mask = (friction_data >= friction_breaks[i]) & (friction_data < friction_breaks[i+1])
            for j in range(3):
                friction_rgb[:, :, j][mask] = friction_colors[i][j]
        friction_png_path = 'friction_surface_colored.png'
        Image.fromarray(friction_rgb).save(friction_png_path)

    # Set image bounds if either raster is available
    if travel_bounds is not None:
        image_bounds = [[travel_bounds.bottom, travel_bounds.left], [travel_bounds.top, travel_bounds.right]]
    elif friction_bounds is not None:
        image_bounds = [[friction_bounds.bottom, friction_bounds.left], [friction_bounds.top, friction_bounds.right]]

    return travel_png_path, friction_png_path, image_bounds

def generate_map(df, year, month, map_style, travel_png_path, friction_png_path, image_bounds, markets, roads, selected_commodities):
    """Generate the Folium map without caching due to unhashable inputs (e.g., DataFrames)."""
    # Filter commodity data by year, month, and selected commodities
    filtered_df = df[(df['Year'] == year) & (df['Month'] == month) & (df['Commodity'].isin(selected_commodities))]
    if filtered_df.empty:
        st.warning(f"No commodity data found for Year {year}, Month {month}, and selected commodities.")
        grouped = None
    else:
        grouped = filtered_df.groupby(['Régions Name', 'Régions - RegionId', 'Régions - Latitude', 'Régions - Longitude']).agg({
            'Commodity': list,
            'Price': list,
            'Unit': list
        }).reset_index()
        grouped['commodity_count'] = grouped['Commodity'].apply(len)

    # Initialize map
    m = folium.Map(location=[14.5, -14.5], zoom_start=7.3, tiles=map_style)

    # Add raster overlays if available
    if travel_png_path and image_bounds:
        folium.raster_layers.ImageOverlay(
            name="Travel Time",
            image=travel_png_path,
            bounds=image_bounds,
            opacity=0.6,
            interactive=True,
            cross_origin=False
        ).add_to(m)
        # Add travel time legend
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

    if friction_png_path and image_bounds:
        folium.raster_layers.ImageOverlay(
            name="Friction Surface (min/m)",
            image=friction_png_path,
            bounds=image_bounds,
            opacity=0.7,
            interactive=True,
            cross_origin=False
        ).add_to(m)
        # Add friction legend
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
          <div style="background:#800026;width:20px;height:20px;display:inline-block;"></div> > 5.0<br>
        </div>
        </div>
        {% endmacro %}
        """
        friction_legend = MacroElement()
        friction_legend._template = Template(friction_legend_html)
        m.get_root().add_child(friction_legend)

    # Add roads layer if available
    if roads is not None:
        folium.GeoJson(
            roads,
            name="Roads",
            style_function=lambda x: {'color': 'blue', 'weight': 1, 'opacity': 0.7}
        ).add_to(m)

    # Add markets layer if available
    if markets is not None:
        for _, row in markets.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=row['market'],
                icon=folium.Icon(color='blue', icon='shopping-cart', prefix='fa')
            ).add_to(m)

    # Add commodity markers
    if grouped is not None:
        for _, row in grouped.iterrows():
            if pd.isna(row['Régions - Latitude']) or pd.isna(row['Régions - Longitude']):
                continue
            commodity_count = row['commodity_count']
            color = 'blue' if commodity_count < 5 else 'orange' if commodity_count < 10 else 'red'
            commodity_details = [
                f"{commodity}: {price:.2f} {unit}" if not pd.isna(price) else f"{commodity}: Price not available"
                for commodity, price, unit in zip(row['Commodity'], row['Price'], row['Unit'])
            ]
            commodity_list = '<br>'.join(commodity_details)
            popup_content = f"""
            <div style='width: 250px'>
                <h4>{row['Régions Name']}</h4>
                <b>Region ID:</b> {row['Régions - RegionId']}<br>
                <b>Commodities ({commodity_count}):</b><br>{commodity_list}<br>
                <b>Coordinates:</b> {row['Régions - Latitude']:.4f}, {row['Régions - Longitude']:.4f}
            </div>
            """
            folium.CircleMarker(
                location=[row['Régions - Latitude'], row['Régions - Longitude']],
                radius=8 + (commodity_count * 2),
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{row['Régions Name']}: {commodity_count} commodities",
                fill=True,
                fill_color=color,
                color=color,
                fill_opacity=0.7
            ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m, grouped['Régions Name'].tolist() if grouped is not None else [], filtered_df

def main():
    st.title("Senegal Commodity and Geospatial Analysis Map")
    st.markdown("Explore commodity availability, prices, travel time to cities, and friction surfaces across Senegal.")

    # Sidebar for controls
    st.sidebar.title("Map Controls")
    map_style = st.sidebar.selectbox("Map Style", ["OpenStreetMap", "CartoDB Positron", "Stamen Terrain"], index=1)

    # Load commodity data
    commodity_df = load_commodity_data()
    if commodity_df is None:
        return

    # Commodity filter
    commodities = sorted(commodity_df['Commodity'].dropna().unique())
    selected_commodities = st.sidebar.multiselect("Select Commodities", commodities, default=commodities)

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

    # Display summary statistics for travel time if available
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
    years = sorted(commodity_df['Year'].dropna().unique().astype(int))
    months = sorted(commodity_df['Month'].dropna().unique().astype(int))
    if not years or not months:
        st.error("No valid years or months found in the commodity data.")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_year = st.selectbox("Select Year", years, index=years.index(2016) if 2016 in years else 0)
    with col2:
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                       7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        month_options = [(m, month_names[m]) for m in months]
        selected_month = st.selectbox("Select Month", [m[1] for m in month_options], 
                                      index=next((i for i, m in enumerate(month_options) if m[0] == 1), 0))
        selected_month_num = next(m[0] for m in month_options if m[1] == selected_month)

    # Generate and display map
    st.subheader(f"Map for {month_names[selected_month_num]} {selected_year}")
    map_obj, regions_mapped, filtered_df = generate_map(
        commodity_df, selected_year, selected_month_num, map_style, 
        travel_png_path, friction_png_path, image_bounds, markets, roads, selected_commodities
    )
    
    if map_obj:
        folium_static(map_obj, width=1000, height=600)
        if regions_mapped:
            st.write(f"Regions mapped: {', '.join(regions_mapped)}")
        else:
            st.write("No commodity regions to display for the selected year, month, and commodities.")
        
        # Display commodity details in a table
        if not filtered_df.empty:
            st.subheader("Commodity Details")
            display_df = filtered_df[['Régions Name', 'Commodity', 'Price', 'Unit']].copy()
            display_df['Price'] = display_df['Price'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            display_df = display_df.sort_values(['Régions Name', 'Commodity'])
            st.dataframe(display_df, use_container_width=True)

            # Download button for commodity data
            csv = display_df.to_csv(index=False)
            st.download_button("Download Commodity Data", csv, "commodity_data.csv", "text/csv")
    else:
        st.write("Unable to generate map due to missing data.")

if __name__ == "__main__":
    main()