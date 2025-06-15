import wbdata
import pandas as pd
import matplotlib.pyplot as plt

# Fetch road density data
road_density = wbdata.get_dataframe('IS.ROD.DNST', country='all')

# Inspect the data
print("First 5 rows of WDI Road Infrastructure data:")
print(road_density.head())
print("\nSummary statistics:")
print(road_density.describe())

# Visualize: Plot road density for selected African countries
selected_countries = ['Kenya', 'Uganda', 'Tanzania']
road_density[selected_countries].plot()
plt.xlabel('Year')
plt.ylabel('Road Density (km per sq. km)')
plt.title('Road Density in Selected African Countries (WDI)')
plt.show()