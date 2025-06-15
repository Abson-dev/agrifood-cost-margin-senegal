import wbdata
import pandas as pd
import matplotlib.pyplot as plt

# Fetch LPI data
lpi = wbdata.get_dataframe('LP.LPI.OVRL.XQ', country='all')

# Inspect the data
print("First 5 rows of LPI data:")
print(lpi.head())
print("\nSummary statistics:")
print(lpi.describe())

# Visualize: Plot LPI scores for selected African countries
selected_countries = ['Kenya', 'South Africa', 'Nigeria']
lpi[selected_countries].plot()
plt.xlabel('Year')
plt.ylabel('LPI Score')
plt.title('Logistics Performance Index in Selected African Countries (LPI)')
plt.show()