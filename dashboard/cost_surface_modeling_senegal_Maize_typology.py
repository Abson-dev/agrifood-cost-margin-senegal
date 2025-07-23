# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 11:08:52 2025

@author: AHema
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import matplotlib.ticker as ticker
from matplotlib import font_manager

# --- Configuration: File Paths and Directories ---
EXCEL_FILE_PATH = 'final_merged_output_senegal.xlsx'
OUTPUT_DIR = 'maize_analysis_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, 'transaction_costs_maize.csv')
SHAPEFILE_PATH = 'senegal_shapefile/sen_admbnda_adm1_anat_20240520.shp'
COMMODITY_ID = 56
COMMODITY_NAME = "Maize"

print(f"Output directory created/verified: {OUTPUT_DIR}")
print(f"Original Excel file path: {EXCEL_FILE_PATH}")
print(f"Shapefile path: {SHAPEFILE_PATH}")

# --- Set Plotting Style for Publication ---
plt.style.use('seaborn-v0_8-whitegrid')  # Clean, professional style
sns.set_context("paper", font_scale=1.2)  # Adjust font scale for readability
plt.rcParams.update({
    'font.family': 'Times New Roman',  # Publication-standard font
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,  # High resolution for publication
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
})

# --- Data Loading and Initial Processing ---
df_original = pd.read_excel(EXCEL_FILE_PATH)
print(df_original.head())

df_original = df_original[df_original['commodity_id'] == COMMODITY_ID]
if df_original.empty:
    print(f"Error: No data for {COMMODITY_NAME} (commodity_id {COMMODITY_ID}) found in the dataset.")
    exit()
print(f"Filtered original data for {COMMODITY_NAME} (commodity_id {COMMODITY_ID}), DataFrame shape: {df_original.shape}")

# --- Load Model Output ---
try:
    df_model_output = pd.read_csv(OUTPUT_CSV_FILE)
    print(df_model_output.head())
except FileNotFoundError:
    print(f"Error: Model output file '{OUTPUT_CSV_FILE}' not found. Please run the model training script first.")
    exit()

# --- Merging Logic ---
merged_df = pd.merge(df_model_output, df_original, on='ID', how='left', suffixes=('_model', '_original'))
print(merged_df.head())

# --- Saving the Merged Data ---
merged_df.to_csv(OUTPUT_CSV_FILE, index=False)
print(f"Merged data saved to: {OUTPUT_CSV_FILE}")

# --- ANALYSIS ---
print(f"\n--- Starting Data Analysis for {COMMODITY_NAME} ---")

# 1. Model Performance and Comparison
print("\n1. Model Performance and Comparison:")
if 'transaction_cost_xof_per_kg' in merged_df.columns:
    comparison_df = merged_df.dropna(subset=['transaction_cost_xof_per_kg', 'C_s_rf', 'C_s_xgb']).copy()
    if not comparison_df.empty:
        y_true = comparison_df['transaction_cost_xof_per_kg']
        y_pred_rf = comparison_df['C_s_rf']
        y_pred_xgb = comparison_df['C_s_xgb']

        # Visualize actual vs. predicted
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        # Random Forest
        sns.regplot(x=y_true, y=y_pred_rf, scatter_kws={'alpha': 0.4, 'color': '#1f77b4'}, line_kws={'color': '#ff7f0e'}, ax=axes[0])
        axes[0].set_title(f'Actual vs. Predicted Transaction Costs\n({COMMODITY_NAME}, Random Forest)', pad=10)
        axes[0].set_xlabel('Actual Transaction Cost (XOF/kg)')
        axes[0].set_ylabel('Predicted Transaction Cost (XOF/kg)')
        axes[0].minorticks_on()
        axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
        # XGBoost
        sns.regplot(x=y_true, y=y_pred_xgb, scatter_kws={'alpha': 0.4, 'color': '#1f77b4'}, line_kws={'color': '#ff7f0e'}, ax=axes[1])
        axes[1].set_title(f'Actual vs. Predicted Transaction Costs\n({COMMODITY_NAME}, XGBoost)', pad=10)
        axes[1].set_xlabel('Actual Transaction Cost (XOF/kg)')
        axes[1].set_ylabel('Predicted Transaction Cost (XOF/kg)')
        axes[1].minorticks_on()
        axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(OUTPUT_DIR, f'model_performance_scatter_{COMMODITY_NAME.lower()}.png'), bbox_inches='tight')
        plt.close()
        print(f"    Model performance scatter plots saved to {OUTPUT_DIR}.")

        # Distribution of residuals
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        # Random Forest
        sns.histplot((y_true - y_pred_rf), kde=True, color='#1f77b4', ax=axes[0])
        axes[0].set_title(f'Residuals Distribution\n({COMMODITY_NAME}, Random Forest)', pad=10)
        axes[0].set_xlabel('Error (Actual - Predicted, XOF/kg)')
        axes[0].set_ylabel('Frequency')
        axes[0].minorticks_on()
        axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
        # XGBoost
        sns.histplot((y_true - y_pred_xgb), kde=True, color='#1f77b4', ax=axes[1])
        axes[1].set_title(f'Residuals Distribution\n({COMMODITY_NAME}, XGBoost)', pad=10)
        axes[1].set_xlabel('Error (Actual - Predicted, XOF/kg)')
        axes[1].set_ylabel('Frequency')
        axes[1].minorticks_on()
        axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(OUTPUT_DIR, f'model_residuals_distribution_{COMMODITY_NAME.lower()}.png'), bbox_inches='tight')
        plt.close()
        print(f"    Model residuals distribution plots saved to {OUTPUT_DIR}.")
    else:
        print("    Not enough data to compare model performance. Check 'transaction_cost_xof_per_kg_original' column for missing values.")
else:
    print("    'transaction_cost_xof_per_kg_original' column not found in merged_df. Cannot perform model performance comparison directly.")
    print("    Please ensure a column representing actual transaction costs is present in the original data and correctly merged.")

# 2. Geospatial Variation of Transaction Costs
print(f"\n2. Geospatial Variation of {COMMODITY_NAME} Transaction Costs:")
try:
    gdf_senegal = gpd.read_file(SHAPEFILE_PATH)
    print(f"    Senegal shapefile loaded from: {SHAPEFILE_PATH}")
    merged_df['admin1_retail_code'] = merged_df['admin1_retail_code'].astype(str).str.strip().str.upper()
    gdf_senegal['ADM1_PCODE'] = gdf_senegal['ADM1_PCODE'].astype(str).str.strip().str.upper()
    avg_costs_rf = merged_df.groupby('admin1_retail_code')['C_s_rf'].mean().reset_index()
    avg_costs_xgb = merged_df.groupby('admin1_retail_code')['C_s_xgb'].mean().reset_index()
    gdf_senegal_rf = gdf_senegal.merge(avg_costs_rf, left_on='ADM1_PCODE', right_on='admin1_retail_code', how='left')
    gdf_senegal_xgb = gdf_senegal.merge(avg_costs_xgb, left_on='ADM1_PCODE', right_on='admin1_retail_code', how='left')

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    # Random Forest
    gdf_senegal_rf.plot(column='C_s_rf', cmap='viridis', linewidth=0.8, ax=axes[0], edgecolor='0.2', legend=True,
                        legend_kwds={'label': 'Transaction Cost (XOF/kg)', 'orientation': 'vertical', 'shrink': 0.6})
    axes[0].set_title(f'Average Transaction Costs\n({COMMODITY_NAME}, Random Forest)', pad=10)
    axes[0].set_axis_off()
    axes[0].set_aspect('equal')
    # XGBoost
    gdf_senegal_xgb.plot(column='C_s_xgb', cmap='viridis', linewidth=0.8, ax=axes[1], edgecolor='0.2', legend=True,
                         legend_kwds={'label': 'Transaction Cost (XOF/kg)', 'orientation': 'vertical', 'shrink': 0.6})
    axes[1].set_title(f'Average Transaction Costs\n({COMMODITY_NAME}, XGBoost)', pad=10)
    axes[1].set_axis_off()
    axes[1].set_aspect('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, f'geospatial_transaction_costs_{COMMODITY_NAME.lower()}.png'), bbox_inches='tight')
    plt.close()
    print(f"    Geospatial plots of average transaction costs saved to {OUTPUT_DIR}.")
except Exception as e:
    print(f"    Error performing geospatial analysis: {e}")
    print("    Please ensure the shapefile path is correct and geopandas is installed (`pip install geopandas`).")

# 3. Factors Influencing Transaction Costs
print(f"\n3. Factors Influencing {COMMODITY_NAME} Transaction Costs:")
if 'year' in merged_df.columns and 'month' in merged_df.columns:
    avg_costs_by_year_rf = merged_df.groupby('year')['C_s_rf'].mean().reset_index()
    avg_costs_by_year_xgb = merged_df.groupby('year')['C_s_xgb'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(avg_costs_by_year_rf['year'], avg_costs_by_year_rf['C_s_rf'], marker='o', label='Random Forest', color='#1f77b4')
    ax.plot(avg_costs_by_year_xgb['year'], avg_costs_by_year_xgb['C_s_xgb'], marker='x', label='XGBoost', color='#ff7f0e')
    ax.set_title(f'Average Estimated {COMMODITY_NAME} Transaction Costs\nOver Years', pad=10)
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Transaction Cost (XOF/kg)')
    ax.minorticks_on()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    if not avg_costs_by_year_rf['year'].empty:
        ax.set_xticks(avg_costs_by_year_rf['year'].unique())
    plt.savefig(os.path.join(OUTPUT_DIR, f'costs_by_year_{COMMODITY_NAME.lower()}.png'), bbox_inches='tight')
    plt.close()
    print(f"    Plot of costs by year saved to {OUTPUT_DIR}.")

    merged_df['month'] = pd.to_numeric(merged_df['month'], errors='coerce')
    avg_costs_by_month_rf = merged_df.groupby('month')['C_s_rf'].mean().reset_index()
    avg_costs_by_month_xgb = merged_df.groupby('month')['C_s_xgb'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(avg_costs_by_month_rf['month'], avg_costs_by_month_rf['C_s_rf'], marker='o', label='Random Forest', color='#1f77b4')
    ax.plot(avg_costs_by_month_xgb['month'], avg_costs_by_month_xgb['C_s_xgb'], marker='x', label='XGBoost', color='#ff7f0e')
    ax.set_title(f'Average Estimated {COMMODITY_NAME} Transaction Costs\nby Month (Seasonality)', pad=10)
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Transaction Cost (XOF/kg)')
    ax.minorticks_on()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.savefig(os.path.join(OUTPUT_DIR, f'costs_by_month_{COMMODITY_NAME.lower()}.png'), bbox_inches='tight')
    plt.close()
    print(f"    Plot of costs by month saved to {OUTPUT_DIR}.")

    if 'price_retail' in merged_df.columns:
        print(f"\n    Correlation between Estimated {COMMODITY_NAME} Transaction Costs and Retail Price:")
        correlation_rf = merged_df['C_s_rf'].corr(merged_df['price_retail'])
        correlation_xgb = merged_df['C_s_xgb'].corr(merged_df['price_retail'])
        print(f"      Correlation (CS_rf vs Retail): {correlation_rf:.2f}")
        print(f"      Correlation (CS_xgb vs Retail): {correlation_xgb:.2f}")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        sns.scatterplot(x='price_retail', y='C_s_rf', data=merged_df, alpha=0.4, color='#1f77b4', ax=axes[0])
        axes[0].set_title(f'Retail Price vs. Estimated\n{COMMODITY_NAME} Transaction Cost\n(Random Forest)', pad=10)
        axes[0].set_xlabel('Retail Price (XOF/kg)')
        axes[0].set_ylabel('Estimated Transaction Cost (XOF/kg)')
        axes[0].minorticks_on()
        axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
        sns.scatterplot(x='price_retail', y='C_s_xgb', data=merged_df, alpha=0.4, color='#1f77b4', ax=axes[1])
        axes[1].set_title(f'Retail Price vs. Estimated\n{COMMODITY_NAME} Transaction Cost\n(XGBoost)', pad=10)
        axes[1].set_xlabel('Retail Price (XOF/kg)')
        axes[1].set_ylabel('Estimated Transaction Cost (XOF/kg)')
        axes[1].minorticks_on()
        axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(OUTPUT_DIR, f'retail_vs_transaction_costs_{COMMODITY_NAME.lower()}.png'), bbox_inches='tight')
        plt.close()
        print(f"    Scatter plot of retail price vs. transaction costs saved to {OUTPUT_DIR}.")
    else:
        print("    'price_retail_original' column not found. Skipping correlation with retail price.")
else:
    print("    'year_original' or 'month_original' column not found. Skipping temporal analysis of costs.")

# 4. Impact of Transaction Costs on Net Margins
print(f"\n4. Impact of {COMMODITY_NAME} Transaction Costs on Net Margins:")
fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
sns.scatterplot(x='C_s_rf', y='NM_t_rf', data=merged_df, alpha=0.4, color='#1f77b4', ax=axes[0])
axes[0].set_title(f'Estimated Transaction Cost vs.\nNet Margin ({COMMODITY_NAME}, Random Forest)', pad=10)
axes[0].set_xlabel('Estimated Transaction Cost (XOF/kg)')
axes[0].set_ylabel('Estimated Net Margin (XOF/kg)')
axes[0].minorticks_on()
axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
sns.scatterplot(x='C_s_xgb', y='NM_t_xgb', data=merged_df, alpha=0.4, color='#1f77b4', ax=axes[1])
axes[1].set_title(f'Estimated Transaction Cost vs.\nNet Margin ({COMMODITY_NAME}, XGBoost)', pad=10)
axes[1].set_xlabel('Estimated Transaction Cost (XOF/kg)')
axes[1].set_ylabel('Estimated Net Margin (XOF/kg)')
axes[1].minorticks_on()
axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
plt.savefig(os.path.join(OUTPUT_DIR, f'costs_vs_net_margins_{COMMODITY_NAME.lower()}.png'), bbox_inches='tight')
plt.close()
print(f"    Scatter plots of transaction costs vs. net margins saved to {OUTPUT_DIR}.")

print(f"\n    Correlation between Estimated {COMMODITY_NAME} Transaction Costs and Net Margins:")
correlation_csrf_nmtrf = merged_df['C_s_rf'].corr(merged_df['NM_t_rf'])
correlation_csxgb_nmtxgb = merged_df['C_s_xgb'].corr(merged_df['NM_t_xgb'])
print(f"      Correlation (CS_rf vs NM_t_rf): {correlation_csrf_nmtrf:.2f}")
print(f"      Correlation (CS_xgb vs NM_t_xgb): {correlation_csxgb_nmtxgb:.2f}")
print("      (Expected to be negative: higher costs, lower net margins)")

# --- Geospatial Variation of Estimated Net Margins ---
print(f"\n--- Geospatial Variation of Estimated Net Margins for {COMMODITY_NAME} ---")
try:
    gdf_senegal = gpd.read_file(SHAPEFILE_PATH)
    print(f"    Senegal shapefile loaded from: {SHAPEFILE_PATH}")
    merged_df['admin1_retail_code'] = merged_df['admin1_retail_code'].astype(str).str.strip().str.upper()
    gdf_senegal['ADM1_PCODE'] = gdf_senegal['ADM1_PCODE'].astype(str).str.strip().str.upper()
    avg_nm_rf = merged_df.groupby('admin1_retail_code')['NM_t_rf'].mean().reset_index()
    avg_nm_xgb = merged_df.groupby('admin1_retail_code')['NM_t_xgb'].mean().reset_index()
    gdf_senegal_nm_rf = gdf_senegal.merge(avg_nm_rf, left_on='ADM1_PCODE', right_on='admin1_retail_code', how='left')
    gdf_senegal_nm_xgb = gdf_senegal.merge(avg_nm_xgb, left_on='ADM1_PCODE', right_on='admin1_retail_code', how='left')

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    gdf_senegal_nm_rf.plot(column='NM_t_rf', cmap='RdYlGn', linewidth=0.8, ax=axes[0], edgecolor='0.2', legend=True,
                           legend_kwds={'label': 'Net Margin (XOF/kg)', 'orientation': 'vertical', 'shrink': 0.6})
    axes[0].set_title(f'Average Estimated Net Margins\n({COMMODITY_NAME}, Random Forest)', pad=10)
    axes[0].set_axis_off()
    axes[0].set_aspect('equal')
    gdf_senegal_nm_xgb.plot(column='NM_t_xgb', cmap='RdYlGn', linewidth=0.8, ax=axes[1], edgecolor='0.2', legend=True,
                            legend_kwds={'label': 'Net Margin (XOF/kg)', 'orientation': 'vertical', 'shrink': 0.6})
    axes[1].set_title(f'Average Estimated Net Margins\n({COMMODITY_NAME}, XGBoost)', pad=10)
    axes[1].set_axis_off()
    axes[1].set_aspect('equal')
    plt.savefig(os.path.join(OUTPUT_DIR, f'geospatial_net_margins_{COMMODITY_NAME.lower()}.png'), bbox_inches='tight')
    plt.close()
    print(f"    Geospatial plots of average estimated net margins saved to {OUTPUT_DIR}.")
except Exception as e:
    print(f"    Error performing geospatial analysis for net margins: {e}")
    print("    Please ensure the shapefile path is correct and geopandas is installed (`pip install geopandas`).")

print(f"\n--- Analysis for {COMMODITY_NAME} complete ---")