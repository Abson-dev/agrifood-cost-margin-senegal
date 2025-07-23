# -*- coding: utf-8 -*-
"""
Updated on Thu Jul 23 2025
Sorghum-only transaction cost analysis for Senegal using commodity_id 57 (assumed)
All outputs saved in sorghum_analysis_outputs folder
Incorporates time-based train-test split for time series data (2016–March 2025)
Training: Jan 2016–Mar 2024, Test: Apr 2024–Mar 2025
Fixes NameError for cross_val_score by adding it to imports
Includes hyperparameter tuning with GridSearchCV
Uses descriptive labels in feature importance plot
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, make_scorer
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set Seaborn style for publication-quality plots
sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.7})
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# --- Configuration ---
EXCEL_FILE_PATH = 'final_merged_output_senegal.xlsx'
OUTPUT_DIR = 'sorghum_analysis_outputs'  # Changed to sorghum-specific directory
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, 'transaction_costs_sorghum.csv')  # Updated file name
COST_SURFACE_RF_PLOT_FILE = os.path.join(OUTPUT_DIR, 'cost_surface_rf_sorghum.png')  # Updated file name
COST_SURFACE_XGB_PLOT_FILE = os.path.join(OUTPUT_DIR, 'cost_surface_xgb_sorghum.png')  # Updated file name
SCATTER_RF_PLOT_FILE = os.path.join(OUTPUT_DIR, 'scatter_cost_vs_distance_rf_sorghum.png')  # Updated file name
SCATTER_XGB_PLOT_FILE = os.path.join(OUTPUT_DIR, 'scatter_cost_vs_distance_xgb_sorghum.png')  # Updated file name
MODEL_COMPARISON_PLOT_FILE = os.path.join(OUTPUT_DIR, 'model_comparison_sorghum.png')  # Updated file name
MODEL_COMPARISON_CSV = os.path.join(OUTPUT_DIR, 'model_comparison_sorghum.csv')  # Updated file name
FEATURE_IMPORTANCE_EXCEL = os.path.join(OUTPUT_DIR, 'feature_importance_sorghum.xlsx')  # Updated file name
FEATURE_IMPORTANCE_PLOT_FILE = os.path.join(OUTPUT_DIR, 'feature_importance_sorghum.png')  # Updated file name
SHAPEFILE_PATH = 'senegal_shapefile/sen_admbnda_adm1_anat_20240520.shp'
RF_MODEL_FILE = os.path.join(OUTPUT_DIR, 'rf_model_sorghum.pkl')  # Updated file name
XGB_MODEL_FILE = os.path.join(OUTPUT_DIR, 'xgb_model_sorghum.pkl')  # Updated file name

REQUIRED_COLUMNS = [
    'commodity_farmgate', 'gross_margin', 'distance_km', 'friction_mean',
    'travel_time_mean', 'vim', 'rfh', 'market_latitude', 'market_longitude',
    'farmgate_latitude', 'farmgate_longitude', 'ID', 'price_farmgate', 'commodity_id',
    'market_population_5km', 'market_population_10km', 'year', 'month'
]

TRANSPORT_RATE_XOF_PER_KG_KM = 0.3
SORGHUM_SPOILAGE_RATE = 0.116  # Assumed spoilage rate for sorghum (confirm with PERISHABILITY_MAPPING)

# --- Data Loading ---
try:
    df = pd.read_excel(EXCEL_FILE_PATH)
    print(f"Successfully loaded data from '{EXCEL_FILE_PATH}'")
except FileNotFoundError:
    print(f"Error: File '{EXCEL_FILE_PATH}' not found.")
    sys.exit(1)

# Check for required columns
missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns: {', '.join(missing_columns)}")
    sys.exit(1)

# Filter for Sorghum using commodity_id 57 (assumed, replace with actual ID)
df = df[df['commodity_id'] == 65]  # Changed to sorghum commodity_id
if df.empty:
    print("Error: No data for Sorghum (commodity_id 57) found in the dataset. Please verify commodity_id.")
    sys.exit(1)
print(f"Filtered data for Sorghum (commodity_id 57), DataFrame shape: {df.shape}")

# --- Data Validation ---
df = df[REQUIRED_COLUMNS]
if df[REQUIRED_COLUMNS].isnull().any().any():
    print("Warning: Missing values detected in required columns. Dropping rows with missing data.")
    df = df.dropna(subset=REQUIRED_COLUMNS)

if (df['distance_km'] < 0).any():
    print("Error: Negative distances detected.")
    sys.exit(1)
if (df['price_farmgate'] < 0).any():
    print("Error: Negative farmgate prices detected.")
    sys.exit(1)
if (df['market_population_5km'] < 0).any() or (df['market_population_10km'] < 0).any():
    print("Error: Negative population values detected.")
    sys.exit(1)
if (df['month'] < 1).any() or (df['month'] > 12).any():
    print("Error: Invalid month values detected (must be 1–12).")
    sys.exit(1)
if (df['year'] < 1900).any() or (df['year'] > 2025).any():
    print("Error: Invalid year values detected (must be 1900–2025).")
    sys.exit(1)

print(f"Farmgate latitude range: {df['farmgate_latitude'].min():.2f} to {df['farmgate_latitude'].max():.2f}")
print(f"Farmgate longitude range: {df['farmgate_longitude'].min():.2f} to {df['farmgate_longitude'].max():.2f}")

# Assign constant spoilage rate for Sorghum
df['spoilage_rate'] = SORGHUM_SPOILAGE_RATE
print(f"Spoilage rate for Sorghum set to {SORGHUM_SPOILAGE_RATE}.")

# --- Feature Normalization ---
def normalize(series):
    if series.std() > 0:
        return (series - series.min()) / (series.max() - series.min())
    return pd.Series([0] * len(series), index=series.index)

df['travel_time_norm'] = normalize(df['travel_time_mean'])
df['friction_norm'] = normalize(df['friction_mean'])
df['vim_norm'] = normalize(df['vim'])
df['rfh_norm'] = normalize(df['rfh'])
df['market_population_5km_norm'] = normalize(df['market_population_5km'])
df['market_population_10km_norm'] = normalize(df['market_population_10km'])
df['year_norm'] = normalize(df['year'])
# Cyclical encoding for month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
# Interaction terms
df['month_spoilage_interaction'] = df['month_sin'] * SORGHUM_SPOILAGE_RATE  # Updated for sorghum spoilage rate

# Define features (no commodity dummies)
numerical_features = [
    'travel_time_norm',
    'friction_norm', 'vim_norm', 'rfh_norm',
    'market_population_5km_norm',
    'year_norm', 'month_sin', 'month_cos'
]
features = numerical_features
print("Features include numerical features only (no commodity dummies).")

# Diagnostic: Check for NaN in input and derived features
input_features = [
    'travel_time_norm',
    'travel_time_mean', 'friction_mean', 'vim', 'rfh',
    'market_population_5km', 'market_population_10km', 'year', 'month'
]
print("\nNaN counts in input features:")
for col in input_features:
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        print(f"{col}: {nan_count} NaN values")
print("\nNaN counts in derived features:")
for col in features:
    nan_count = df[col].isna().sum()
    if nan_count > 0:
        print(f"{col}: {nan_count} NaN values")

# Impute NaN values in numerical features
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(df[numerical_features]), columns=numerical_features, index=df.index)
print("Imputed NaN values in numerical features using mean strategy.")

# --- Transaction Cost Estimation ---
df['transport_cost'] = df['distance_km'] * TRANSPORT_RATE_XOF_PER_KG_KM
df['spoilage_cost'] = SORGHUM_SPOILAGE_RATE * df['price_farmgate']  # Updated for sorghum
df['transaction_cost_xof_per_kg'] = df['transport_cost'] + df['spoilage_cost']
print("Transaction costs (XOF/kg) estimated for Sorghum.")

if df['transaction_cost_xof_per_kg'].isna().any():
    print(f"Warning: {df['transaction_cost_xof_per_kg'].isna().sum()} NaN values in transaction_cost_xof_per_kg. Imputing with mean.")
    df['transaction_cost_xof_per_kg'] = df['transaction_cost_xof_per_kg'].fillna(df['transaction_cost_xof_per_kg'].mean())

# --- Time-Based Train-Test Split ---
# Create a datetime column for sorting and splitting
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df = df.sort_values('date')  # Ensure chronological order

# Split at March 2024 (training: Jan 2016–Mar 2024, test: Apr 2024–Mar 2025)
train_mask = (df['date'] <= '2024-03-01')
test_mask = (df['date'] > '2024-03-01')

df_train = df[train_mask]
df_test = df[test_mask]
X_train = X.loc[df_train.index]
X_test = X.loc[df_test.index]
y_train = df_train['transaction_cost_xof_per_kg']
y_test = df_test['transaction_cost_xof_per_kg']

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Training period: {df_train['date'].min()} to {df_train['date'].max()}")
print(f"Test period: {df_test['date'].min()} to {df_test['date'].max()}")

# --- Hyperparameter Tuning ---
# Define custom scorers for grid search
def mape_scorer(y_true, y_pred):
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

mape_scorer = make_scorer(mape_scorer, greater_is_better=False)
medae_scorer = make_scorer(median_absolute_error, greater_is_better=False)
tscv = TimeSeriesSplit(n_splits=5)

# Random Forest Grid Search
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [5, 10, 20]
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
rf = rf_grid.best_estimator_
print(f"Best Random Forest parameters: {rf_grid.best_params_}")
print(f"Best Random Forest CV MAE: {-rf_grid.best_score_:.2f}")
joblib.dump(rf, RF_MODEL_FILE)
print(f"Random Forest model saved to '{RF_MODEL_FILE}'")

# XGBoost Grid Search
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}
xgb_grid = GridSearchCV(
    XGBRegressor(random_state=42),
    xgb_param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
xgb_grid.fit(X_train, y_train)
xgb = xgb_grid.best_estimator_
print(f"Best XGBoost parameters: {xgb_grid.best_params_}")
print(f"Best XGBoost CV MAE: {-xgb_grid.best_score_:.2f}")
joblib.dump(xgb, XGB_MODEL_FILE)
print(f"XGBoost model saved to '{XGB_MODEL_FILE}'")

# --- Model Comparison ---
# Predictions
df_train['C_s_rf'] = rf.predict(X_train)
df_train['C_s_xgb'] = xgb.predict(X_train)
df_train['NM_t_rf'] = df_train['gross_margin'] - df_train['C_s_rf']
df_train['NM_t_xgb'] = df_train['gross_margin'] - df_train['C_s_xgb']

df_test['C_s_rf'] = rf.predict(X_test)
df_test['C_s_xgb'] = xgb.predict(X_test)
df_test['NM_t_rf'] = df_test['gross_margin'] - df_test['C_s_rf']
df_test['NM_t_xgb'] = df_test['gross_margin'] - df_test['C_s_xgb']

# Calculate additional metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return {'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'MedAE': medae}

# Training and Test metrics
rf_train_metrics = calculate_metrics(y_train, df_train['C_s_rf'])
xgb_train_metrics = calculate_metrics(y_train, df_train['C_s_xgb'])
rf_test_metrics = calculate_metrics(y_test, df_test['C_s_rf'])
xgb_test_metrics = calculate_metrics(y_test, df_test['C_s_xgb'])

# Cross-validation scores for multiple metrics
rf_cv_r2_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring='r2')
xgb_cv_r2_scores = cross_val_score(xgb, X_train, y_train, cv=tscv, scoring='r2')
rf_cv_mae_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error') * -1
xgb_cv_mae_scores = cross_val_score(xgb, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error') * -1
rf_cv_mse_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error') * -1
xgb_cv_mse_scores = cross_val_score(xgb, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error') * -1
rf_cv_rmse_scores = np.sqrt(rf_cv_mse_scores)
xgb_cv_rmse_scores = np.sqrt(xgb_cv_mse_scores)
rf_cv_mape_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring=mape_scorer) * -1
xgb_cv_mape_scores = cross_val_score(xgb, X_train, y_train, cv=tscv, scoring=mape_scorer) * -1
rf_cv_medae_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring=medae_scorer) * -1
xgb_cv_medae_scores = cross_val_score(xgb, X_train, y_train, cv=tscv, scoring=medae_scorer) * -1

print("\nModel Comparison for Sorghum (Time Series Cross-Validation on Training Data):")  # Updated label
print(f"Random Forest - R² (CV Mean): {rf_cv_r2_scores.mean():.2f} ± {rf_cv_r2_scores.std():.2f}")
print(f"Random Forest - MAE (CV Mean): {rf_cv_mae_scores.mean():.2f} ± {rf_cv_mae_scores.std():.2f}")
print(f"Random Forest - RMSE (CV Mean): {rf_cv_rmse_scores.mean():.2f} ± {rf_cv_rmse_scores.std():.2f}")
print(f"Random Forest - MAPE (CV Mean): {rf_cv_mape_scores.mean():.2f}% ± {rf_cv_mape_scores.std():.2f}")
print(f"Random Forest - MedAE (CV Mean): {rf_cv_medae_scores.mean():.2f} ± {rf_cv_medae_scores.std():.2f}")
print(f"XGBoost - R² (CV Mean): {xgb_cv_r2_scores.mean():.2f} ± {xgb_cv_r2_scores.std():.2f}")
print(f"XGBoost - MAE (CV Mean): {xgb_cv_mae_scores.mean():.2f} ± {xgb_cv_mae_scores.std():.2f}")
print(f"XGBoost - RMSE (CV Mean): {xgb_cv_rmse_scores.mean():.2f} ± {xgb_cv_rmse_scores.std():.2f}")
print(f"XGBoost - MAPE (CV Mean): {xgb_cv_mape_scores.mean():.2f}% ± {xgb_cv_mape_scores.std():.2f}")
print(f"XGBoost - MedAE (CV Mean): {xgb_cv_medae_scores.mean():.2f} ± {xgb_cv_medae_scores.std():.2f}")
print(f"\nTraining Metrics:")
print(f"Random Forest Metrics: R²={rf_train_metrics['R2']:.2f}, MAE={rf_train_metrics['MAE']:.2f}, MSE={rf_train_metrics['MSE']:.2f}, RMSE={rf_train_metrics['RMSE']:.2f}, MAPE={rf_train_metrics['MAPE']:.2f}%, MedAE={rf_train_metrics['MedAE']:.2f}")
print(f"XGBoost Metrics: R²={xgb_train_metrics['R2']:.2f}, MAE={xgb_train_metrics['MAE']:.2f}, MSE={xgb_train_metrics['MSE']:.2f}, RMSE={xgb_train_metrics['RMSE']:.2f}, MAPE={xgb_train_metrics['MAPE']:.2f}%, MedAE={xgb_train_metrics['MedAE']:.2f}")
print(f"\nTest Metrics:")
print(f"Random Forest Test Metrics: R²={rf_test_metrics['R2']:.2f}, MAE={rf_test_metrics['MAE']:.2f}, MSE={rf_test_metrics['MSE']:.2f}, RMSE={rf_test_metrics['RMSE']:.2f}, MAPE={rf_test_metrics['MAPE']:.2f}%, MedAE={rf_test_metrics['MedAE']:.2f}")
print(f"XGBoost Test Metrics: R²={xgb_test_metrics['R2']:.2f}, MAE={xgb_test_metrics['MAE']:.2f}, MSE={xgb_test_metrics['MSE']:.2f}, RMSE={xgb_test_metrics['RMSE']:.2f}, MAPE={xgb_test_metrics['MAPE']:.2f}%, MedAE={xgb_test_metrics['MedAE']:.2f}")

comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'R2_CV_Mean': [rf_cv_r2_scores.mean(), xgb_cv_r2_scores.mean()],
    'R2_CV_Std': [rf_cv_r2_scores.std(), xgb_cv_r2_scores.std()],
    'MAE_CV_Mean': [rf_cv_mae_scores.mean(), xgb_cv_mae_scores.mean()],
    'MAE_CV_Std': [rf_cv_mae_scores.std(), xgb_cv_mae_scores.std()],
    'RMSE_CV_Mean': [rf_cv_rmse_scores.mean(), xgb_cv_rmse_scores.mean()],
    'RMSE_CV_Std': [rf_cv_rmse_scores.std(), xgb_cv_rmse_scores.std()],
    'MAPE_CV_Mean': [rf_cv_mape_scores.mean(), xgb_cv_mape_scores.mean()],
    'MAPE_CV_Std': [rf_cv_mape_scores.std(), xgb_cv_mape_scores.std()],
    'MedAE_CV_Mean': [rf_cv_medae_scores.mean(), xgb_cv_medae_scores.mean()],
    'MedAE_CV_Std': [rf_cv_medae_scores.std(), xgb_cv_medae_scores.std()],
    'R2_Train': [rf_train_metrics['R2'], xgb_train_metrics['R2']],
    'MAE_Train': [rf_train_metrics['MAE'], xgb_train_metrics['MAE']],
    'MSE_Train': [rf_train_metrics['MSE'], xgb_train_metrics['MSE']],
    'RMSE_Train': [rf_train_metrics['RMSE'], xgb_train_metrics['RMSE']],
    'MAPE_Train': [rf_train_metrics['MAPE'], xgb_train_metrics['MAPE']],
    'MedAE_Train': [rf_train_metrics['MedAE'], xgb_train_metrics['MedAE']],
    'R2_Test': [rf_test_metrics['R2'], xgb_test_metrics['R2']],
    'MAE_Test': [rf_test_metrics['MAE'], xgb_test_metrics['MAE']],
    'MSE_Test': [rf_test_metrics['MSE'], xgb_test_metrics['MSE']],
    'RMSE_Test': [rf_test_metrics['RMSE'], xgb_test_metrics['RMSE']],
    'MAPE_Test': [rf_test_metrics['MAPE'], xgb_test_metrics['MAPE']],
    'MedAE_Test': [rf_test_metrics['MedAE'], xgb_test_metrics['MedAE']]
})
comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)
print(f"Model comparison saved to '{MODEL_COMPARISON_CSV}'")

# --- Save Feature Importance ---
feature_importance_rf = pd.Series(rf.feature_importances_, index=features)
feature_importance_xgb = pd.Series(xgb.feature_importances_, index=features)
feature_labels = {
    'travel_time_norm': 'Travel Time',
    'friction_norm': 'Surface Friction',
    'vim_norm': 'NDVI',
    'rfh_norm': 'Rainfall',
    'market_population_5km_norm': 'Market Population (5km)',
    'year_norm': 'Year',
    'month_sin': 'Month Sine Component',
    'month_cos': 'Month Cosine Component'
}
feature_importance_df = pd.DataFrame({
    'Feature': [feature_labels[f] for f in features],
    'Technical_Feature': features,  # Include technical names for reference
    'Random_Forest_Importance': feature_importance_rf.values,
    'XGBoost_Importance': feature_importance_xgb.values
})
feature_importance_df = feature_importance_df.sort_values(by='Random_Forest_Importance', ascending=False)
feature_importance_df.to_excel(FEATURE_IMPORTANCE_EXCEL, index=False)
print(f"Feature importance saved to '{FEATURE_IMPORTANCE_EXCEL}'")
print("\nRandom Forest Feature Importance (sorted):\n", feature_importance_rf.sort_values(ascending=False))
print("\nXGBoost Feature Importance (sorted):\n", feature_importance_xgb.sort_values(ascending=False))

# --- Plot Feature Importance with Descriptive Labels ---
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(features))
bars1 = ax.bar(index, feature_importance_rf, bar_width, label='Random Forest', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(index + bar_width, feature_importance_xgb, bar_width, label='XGBoost', color='#ff7f0e', alpha=0.8)
ax.set_xlabel('Features', fontsize=14)
ax.set_ylabel('Feature Importance', fontsize=14)
ax.set_title('Feature Importance for Sorghum Transaction Cost Models', fontsize=16, pad=15)  # Updated title
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels([feature_labels[f] for f in features], rotation=45, ha='right', fontsize=12)
ax.legend(frameon=True, loc='upper right', fontsize=12)
ax.grid(True, which='major', linestyle='--', alpha=0.5)
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(FEATURE_IMPORTANCE_PLOT_FILE, bbox_inches='tight')
plt.close()
print(f"Feature importance plot with descriptive labels saved to '{FEATURE_IMPORTANCE_PLOT_FILE}'")

# --- Plot Model Comparison (Cross-validation Metrics) ---
fig, ax = plt.subplots(figsize=(12, 7))
metrics_labels = ['R²', 'MAE', 'RMSE', 'MAPE (%)', 'MedAE']
rf_cv_values = [
    rf_cv_r2_scores.mean(),
    rf_cv_mae_scores.mean(),
    rf_cv_rmse_scores.mean(),
    rf_cv_mape_scores.mean(),
    rf_cv_medae_scores.mean()
]
xgb_cv_values = [
    xgb_cv_r2_scores.mean(),
    xgb_cv_mae_scores.mean(),
    xgb_cv_rmse_scores.mean(),
    xgb_cv_mape_scores.mean(),
    xgb_cv_medae_scores.mean()
]
rf_cv_stds = [
    rf_cv_r2_scores.std(),
    rf_cv_mae_scores.std(),
    rf_cv_rmse_scores.std(),
    rf_cv_mape_scores.std(),
    rf_cv_medae_scores.std()
]
xgb_cv_stds = [
    xgb_cv_r2_scores.std(),
    xgb_cv_mae_scores.std(),
    xgb_cv_rmse_scores.std(),
    xgb_cv_mape_scores.std(),
    xgb_cv_medae_scores.std()
]
x = np.arange(len(metrics_labels))
width = 0.35

bars1 = ax.bar(x - width/2, rf_cv_values, width, yerr=rf_cv_stds, capsize=5, label='Random Forest', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x + width/2, xgb_cv_values, width, yerr=xgb_cv_stds, capsize=5, label='XGBoost', color='#ff7f0e', alpha=0.8)
ax.set_ylabel('Metric Value', fontsize=14)
ax.set_title('Cross-Validation Metrics for Sorghum Transaction Cost Models', fontsize=16, pad=15)  # Updated title
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels, fontsize=12)
ax.legend(frameon=True, loc='upper right', fontsize=12)
ax.grid(True, which='major', linestyle='--', alpha=0.5)
for bars, stds in [(bars1, rf_cv_stds), (bars2, xgb_cv_stds)]:
    for bar, std in zip(bars, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02 * max(max(rf_cv_values), max(xgb_cv_values)),
                f'{height:.2f}\n±{std:.2f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(MODEL_COMPARISON_PLOT_FILE, bbox_inches='tight')
plt.close()
print(f"Model comparison plot (Cross-validation Metrics) saved to '{MODEL_COMPARISON_PLOT_FILE}'")

# --- Visualization 3: Scatter Plot (Random Forest) ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df_test, x='distance_km', y='C_s_rf', hue='C_s_rf', size='C_s_rf', sizes=(20, 200), alpha=0.7, palette='viridis', ax=ax)
ax.set_xlabel('Distance (km)', fontsize=14)
ax.set_ylabel('Transaction Cost (XOF/kg)', fontsize=14)
ax.set_title('Sorghum Transaction Cost vs. Distance (Random Forest, Test Set)', fontsize=16, pad=15)  # Updated title
ax.legend(title='Transaction Cost', frameon=True, fontsize=10)
ax.grid(True, which='major', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(SCATTER_RF_PLOT_FILE, bbox_inches='tight')
plt.close()
print(f"Scatter plot (Random Forest, Sorghum, Test Set) saved to '{SCATTER_RF_PLOT_FILE}'")

# --- Visualization 4: Scatter Plot (XGBoost) ---
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df_test, x='distance_km', y='C_s_xgb', hue='C_s_xgb', size='C_s_xgb', sizes=(20, 200), alpha=0.7, palette='viridis', ax=ax)
ax.set_xlabel('Distance (km)', fontsize=14)
ax.set_ylabel('Transaction Cost (XOF/kg)', fontsize=14)
ax.set_title('Sorghum Transaction Cost vs. Distance (XGBoost, Test Set)', fontsize=16, pad=15)  # Updated title
ax.legend(title='Transaction Cost', frameon=True, fontsize=10)
ax.grid(True, which='major', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(SCATTER_XGB_PLOT_FILE, bbox_inches='tight')
plt.close()
print(f"Scatter plot (XGBoost, Sorghum, Test Set) saved to '{SCATTER_XGB_PLOT_FILE}'")

# --- Save Results ---
df = pd.concat([df_train, df_test])
df[['ID', 'C_s_rf', 'NM_t_rf', 'C_s_xgb', 'NM_t_xgb', 'transaction_cost_xof_per_kg']].to_csv(OUTPUT_CSV_FILE, index=False)
print(f"Results saved to '{OUTPUT_CSV_FILE}'")

# --- End of Script ---
print(f"\nSorghum analysis complete. All outputs saved in '{OUTPUT_DIR}' folder.")