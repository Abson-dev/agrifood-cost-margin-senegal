import pandas as pd
import folium
import streamlit as st
from streamlit_folium import folium_static

# Set page configuration
st.set_page_config(page_title="Senegal Commodity Map", layout="wide")

def load_data(input_file='commodity_prices_merged.xlsx'):
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
        return df
    except FileNotFoundError:
        st.error(f"Input file '{input_file}' not found. Please ensure 'commodity_prices_merged.xlsx' exists.")
        return None
    except KeyError as e:
        st.error(f"Column {str(e)} not found in the input file. Required columns: Year, Month, Commodity, Régions Name, Régions - RegionId, Régions - Latitude, Régions - Longitude.")
        return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def generate_map(df, year, month):
    # Filter data for the selected year and month
    filtered_df = df[(df['Year'] == year) & (df['Month'] == month)]
    
    if filtered_df.empty:
        st.warning(f"No data found for Year {year}, Month {month}.")
        return None, []

    # Group by region to get unique commodities
    grouped = filtered_df.groupby(['Régions Name', 'Régions - RegionId', 'Régions - Latitude', 'Régions - Longitude'])['Commodity'].unique().reset_index()

    # Initialize the map, centered on Senegal (approx. coordinates: 14.5, -14.5)
    m = folium.Map(location=[14.5, -14.5], zoom_start=6, tiles='OpenStreetMap')

    # Add markers for each region
    for _, row in grouped.iterrows():
        region_name = row['Régions Name']
        region_id = row['Régions - RegionId']
        lat = row['Régions - Latitude']
        lon = row['Régions - Longitude']
        commodities = row['Commodity']

        # Skip if coordinates are invalid
        if pd.isna(lat) or pd.isna(lon):
            continue

        # Create popup content
        commodity_list = '<br>'.join(commodities)
        popup_content = f"""
        <b>Region:</b> {region_name} ({region_id})<br>
        <b>Commodities Available:</b><br>{commodity_list}
        """
        
        # Add marker with popup
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_content, max_width=300),
            tooltip=f"{region_name}: {len(commodities)} commodities",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    return m, grouped['Régions Name'].tolist()

def main():
    st.title("Senegal Commodity Availability Map")
    st.markdown("Select a year and month to view commodities available in each region of Senegal on an interactive map.")

    # Load data
    df = load_data()
    if df is None:
        return

    # Get unique years and months
    years = sorted(df['Year'].unique().astype(int))
    months = sorted(df['Month'].unique().astype(int))

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
    map_obj, regions_mapped = generate_map(df, selected_year, selected_month_num)

    if map_obj:
        folium_static(map_obj, width=1000, height=600)
        st.write(f"Regions mapped: {', '.join(regions_mapped)}")
    else:
        st.write("No regions to display for the selected year and month.")

if __name__ == "__main__":
    main()