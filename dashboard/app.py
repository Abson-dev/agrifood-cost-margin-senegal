import os
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import folium_static

# Set page configuration
st.set_page_config(page_title="Senegal Commodity Map", layout="wide")

# Cache data loading for performance
@st.cache_data
def load_data(input_file='commodity_prices_merged.xlsx'):
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
        st.error(f"Input file '{input_file}' not found. Please ensure 'commodity_prices_merged.xlsx' exists.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

@st.cache_data
def generate_map(df, year, month, map_style):
    # Filter data for the selected year and month
    filtered_df = df[(df['Year'] == year) & (df['Month'] == month)]
    
    if filtered_df.empty:
        st.warning(f"No data found for Year {year}, Month {month}.")
        return None, [], None

    # Group by region to get commodities with prices and units
    grouped = filtered_df.groupby(['Régions Name', 'Régions - RegionId', 'Régions - Latitude', 'Régions - Longitude']).agg({
        'Commodity': lambda x: list(x),
        'Price': lambda x: list(x),
        'Unit': lambda x: list(x)
    }).reset_index()
    grouped['commodity_count'] = grouped['Commodity'].apply(len)

    # Initialize the map, centered on Senegal (approx. coordinates: 14.5, -14.5)
    m = folium.Map(location=[14.5, -14.5], zoom_start=6, tiles=map_style)

    # Add markers for each region
    for _, row in grouped.iterrows():
        region_name = row['Régions Name']
        region_id = row['Régions - RegionId']
        lat = row['Régions - Latitude']
        lon = row['Régions - Longitude']
        commodities = row['Commodity']
        prices = row['Price']
        units = row['Unit']
        commodity_count = row['commodity_count']

        # Skip if coordinates are invalid
        if pd.isna(lat) or pd.isna(lon):
            continue

        # Create popup content with commodities, prices, and units
        commodity_details = [
            f"{commodity}: {price:.2f} {unit}" if not pd.isna(price) else f"{commodity}: Price not available"
            for commodity, price, unit in zip(commodities, prices, units)
        ]
        commodity_list = '<br>'.join(commodity_details)
        popup_content = f"""
        <div style='width: 250px'>
            <h4>{region_name}</h4>
            <b>Region ID:</b> {region_id}<br>
            <b>Commodities ({commodity_count}):</b><br>{commodity_list}<br>
            <b>Coordinates:</b> {lat:.4f}, {lon:.4f}
        </div>
        """

        # Color based on commodity count
        color = 'blue' if commodity_count < 5 else 'orange' if commodity_count < 10 else 'red'

        # Add marker with popup
        folium.CircleMarker(
            location=[lat, lon],
            radius=8 + (commodity_count * 2),
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"{region_name}: {commodity_count} commodities",
            fill=True,
            fill_color=color,
            color=color,
            fill_opacity=0.7
        ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    return m, grouped['Régions Name'].tolist(), filtered_df

def main():
    st.title("Senegal Commodity Availability and Price Map")
    st.markdown("Select a year and month to view commodities, their prices, and units available in each region of Senegal on an interactive map.")

    # Sidebar for controls
    st.sidebar.title("Map Controls")
    map_style = st.sidebar.selectbox("Map Style", ["OpenStreetMap", "CartoDB Positron", "Stamen Terrain"], index=1)

    # Load data
    df = load_data()
    if df is None:
        return

    # Get unique years and months
    years = sorted(df['Year'].dropna().unique().astype(int))
    months = sorted(df['Month'].dropna().unique().astype(int))

    if not years or not months:
        st.error("No valid years or months found in the data.")
        return

    # Create two columns for year and month selection
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

    # Generate and display the map
    st.subheader(f"Commodities Available in {month_names[selected_month_num]} {selected_year}")
    map_obj, regions_mapped, filtered_df = generate_map(df, selected_year, selected_month_num, map_style)

    if map_obj:
        folium_static(map_obj, width=1000, height=600)
        if regions_mapped:
            st.write(f"Regions mapped: {', '.join(regions_mapped)}")
        else:
            st.write("No regions to display for the selected year and month.")
        
        # Display commodity details in a table
        if not filtered_df.empty:
            st.subheader("Commodity Details")
            display_df = filtered_df[['Régions Name', 'Commodity', 'Price', 'Unit']].copy()
            display_df['Price'] = display_df['Price'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            display_df = display_df.sort_values(['Régions Name', 'Commodity'])
            st.dataframe(display_df, use_container_width=True)
    else:
        st.write("No regions to display for the selected year and month.")

if __name__ == "__main__":
    main()