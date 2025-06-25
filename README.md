<<<<<<< Updated upstream
# agrifood-cost-margin-senegal

- [ ] User Manual: Senegal Agricultural Market Dashboard

## Introduction

The **Senegal Agricultural Market Dashboard** is an interactive web application designed to visualize and analyze agricultural market data in Senegal. Developed in collaboration with the International Food Policy Research Institute (IFPRI), this tool provides insights into travel time, friction surfaces, market locations, road networks, commodity prices, and population density. It is built using Python and Streamlit, with data visualization powered by Folium and Plotly.

This manual guides users through the setup, navigation, and usage of the dashboard to explore and analyze data effectively.

---

## Table of Contents

1. System Requirements
2. Installation and Setup
3. Data Requirements
4. Using the Dashboard
   - Launching the Application
   - Navigating the Interface
   - Interactive Map Tab
   - Data Summary Tab
   - Price Trends Tab
5. Troubleshooting
6. Contact and Support

---

## System Requirements

To run the Senegal Agricultural Market Dashboard, ensure your system meets the following requirements:

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **Dependencies**: Install required Python packages listed in the `requirements.txt` file (see Installation and Setup).
- **Hardware**:
  - Minimum 4GB RAM (8GB recommended for large datasets)
  - At least 1GB free disk space for data files
- **Internet Connection**: Optional, for downloading external dependencies or accessing online map tiles.

---

## Installation and Setup

Follow these steps to set up the dashboard:

1. **Clone or Download the Code**:

   - Download the application code or clone it from the repository (if provided).
   - Ensure the Python script (e.g., `app.py`) is placed in a dedicated directory.

2. **Install Python Dependencies**:

   - Create a virtual environment (recommended):

     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Install required packages:

     ```bash
     pip install streamlit folium streamlit-folium rasterio geopandas pandas numpy scipy geopy plotly pillow
     ```
   - Alternatively, if a `requirements.txt` file is provided, run:

     ```bash
     pip install -r requirements.txt
     ```

3. **Prepare Data Files**:

   - Ensure all required data files are placed in the same directory as the script or a specified `BASE_DIR`. See Data Requirements for details.

4. **Verify Setup**:

   - Ensure all data files are accessible and correctly formatted.
   - Check that Python and all dependencies are installed correctly by running:

     ```bash
     python -c "import streamlit, folium, rasterio, geopandas, pandas, numpy, scipy, geopy, plotly, PIL"
     ```

---

## Data Requirements

The dashboard requires the following data files to function correctly. These files should be placed in the same directory as the script or updated in the `DEFAULT_FILES` dictionary in the code.

| File Type | Description | Format | Example Filename |
| --- | --- | --- | --- |
| **Raster (Travel Time)** | Travel time to cities raster | TIFF | `201501_Global_Travel_Time_to_Cities_SEN.tiff` |
| **Friction Surface** | Travel speed friction surface raster | TIFF | `201501_Global_Travel_Speed_Friction_Surface_SEN.tiff` |
| **Markets** | Market locations | GeoJSON | `markets_from_excel.geojson` |
| **Roads** | Road network data | GeoJSON | `roads_filtered.geojson` |
| **Prices** | Farmgate and retail prices | Excel (xlsx) | `merged_farmgate_retail_prices_senegal.xlsx` |
| **Population (2016-2020)** | Population density rasters for years 2016–2020 | TIFF | `sen_ppp_2016_UNadj.tif`, etc. |

### Notes:

- **Raster Files**: Must be single-band TIFF files with a valid CRS (WGS84 recommended).
- **GeoJSON Files**: Must contain valid geometries (e.g., Points for markets, LineStrings for roads).
- **Excel File**: Must include two sheets:
  - **Farmgate prices Senegal**: Columns: `Régions Name`, `Commodity`, `commodity_english`, `commodity_id`, `Price`, `Unit2`, `Régions - Latitude`, `Régions - Longitude`, `Year`, `Month`.
  - **Retails Price Senegal**: Columns: `market`, `commodity`, `commodity_id`, `Price`, `Unit2`, `latitude`, `longitude`, `Year`, `Month`.
- Ensure file paths in the `DEFAULT_FILES` dictionary match the actual file locations.

---

## Using the Dashboard

### Launching the Application

1. Open a terminal in the directory containing the script.
2. Activate the virtual environment (if used):

   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```
4. Open a web browser and navigate to the URL displayed in the terminal (typically `http://localhost:8501`).

### Navigating the Interface

The dashboard interface consists of:

- **Header**: Displays the IFPRI logo, title, and a brief description.
- **Sidebar**: Contains data upload options and map filters.
- **Main Content**: Organized into three tabs: **Interactive Map**, **Data Summary**, and **Price Trends**.
- **Footer**: Provides credits and data source information.

### Interactive Map Tab

This tab displays a map of Senegal with various data layers.

#### Features:

- **Map Layers**:
  - **Travel Time**: Visualizes travel time to cities (in minutes) as a color-coded raster.
  - **Friction Surface**: Shows travel friction (minutes per meter) as a raster layer.
  - **Population**: Displays population density (people per pixel) for the selected year (2016–2020).
  - **Markets**: Shows market locations with population estimates within a 5km radius.
  - **Roads**: Displays major road networks (motorway, trunk, primary, secondary).
  - **Farmgate Prices**: Marks locations with farmgate price data and distances to the nearest market.
  - **Retail Prices**: Marks locations with retail price data.
- **Controls**:
  - **Year and Month Filters**: Select a specific year (2016–2025) and month to filter price data.
  - **Commodity Filter**: Choose one or more commodities to display on the map.
  - **Layer Toggles**: Enable or disable specific layers via checkboxes in the sidebar.
  - **Map Height Slider**: Adjust the map height (400–1000 pixels) for better visibility.
- **Interactivity**:
  - Click on markers (markets, farmgate, retail) to view detailed information in popups.
  - Use the layer control (top-right) to toggle visibility of layers.
  - Pan and zoom to explore different regions.
  - Use the **MiniMap** (bottom-left) for context and the **Fullscreen** button (top-right) to expand the map.

#### Usage Tips:

- The map may load slowly if many layers (e.g., roads) are enabled due to data complexity.
- If no markers appear, check the filters or ensure valid data is loaded.
- Legends for raster layers (travel time, friction, population) appear below the map when the respective layers are enabled.

### Data Summary Tab

This tab provides a high-level overview of the loaded data.

#### Features:

- **Metrics**:
  - Number of markets loaded.
  - Number of road features displayed.
  - Total number of price points (farmgate + retail).
  - Average distance from farmgate locations to the nearest market (in kilometers).
- **Usage**:
  - Use this tab to quickly verify the quantity and quality of loaded data.
  - If metrics show "0" or "N/A," check the data files or filters.

### Price Trends Tab

This tab visualizes price trends for selected commodities over time.

#### Features:

- **Commodity Selection**: Choose a commodity to analyze.
- **Price Trends**:
  - **Farmgate Prices**: Average farmgate prices per month.
  - **Retail Prices**: Average retail prices per month.
  - **Gross Margin**: Difference between retail and farmgate prices (optional, enabled via checkbox).
- **Visualization**:
  - Interactive Plotly chart showing price and margin trends over time.
  - Hover over data points to see exact values and dates.
  - Toggle the **Gross Margin** checkbox to include or exclude margin data.

#### Usage Tips:

- Ensure both farmgate and retail price data are loaded for meaningful margin calculations.
- If no data appears, verify that the selected commodity has associated price data.

---

## Troubleshooting

| Issue | Possible Cause | Solution |
| --- | --- | --- |
| **Map does not load** | Missing or invalid raster/GeoJSON files | Check file paths and formats. Ensure TIFF files are single-band and GeoJSONs have valid geometries. |
| **No markers on map** | Invalid coordinates or filter mismatch | Verify latitude/longitude columns in price data. Check year, month, and commodity filters. |
| **Price trends not displaying** | Missing or incomplete price data | Ensure Excel file has required columns and data for the selected commodity. |
| **Performance issues** | Large datasets (e.g., roads) | Limit features (e.g., roads to 500) or disable complex layers. Increase system memory if possible. |
| **Error messages** | File not found or invalid format | Check file paths in `DEFAULT_FILES`. Ensure files match expected formats (TIFF, GeoJSON, Excel). |

### Common Fixes:

- **Clear Cache**: If data appears outdated, clear the Streamlit cache by adding `streamlit cache clear` to the terminal or restarting the app.
- **Check Logs**: Review terminal output for detailed error messages.
- **Validate Data**: Use tools like QGIS (for rasters/GeoJSON) or Excel to inspect data files for issues.

---

## Contact and Support

For assistance or feedback:

- **Contact**: International Food Policy Research Institute (IFPRI)
- **Website**: www.ifpri.org
- **Support**: Reach out to the IFPRI team or the application developers via the contact details provided in the dashboard footer.
- **Data Sources**: Data is sourced from IFPRI, OpenStreetMap, and WorldPop. Refer to the footer for more information.

---

This manual is designed to help users effectively utilize the Senegal Agricultural Market Dashboard. For advanced customization or additional features, consult the source code or contact the development team.
