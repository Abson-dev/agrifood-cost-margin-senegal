import geopandas as gpd
import matplotlib.pyplot as plt

# Load the downloaded shapefile
roads = gpd.read_file('grip_roads.shp')

# Inspect the data
print("First 5 rows of GRIP data:")
print(roads.head())
print("\nSummary statistics:")
print(roads.describe())

# Visualize the road network
roads.plot()
plt.title('Global Roads Network (GRIP)')
plt.show()

# Calculate total road length
total_length = roads.length.sum()
print(f'Total road length: {total_length} km')