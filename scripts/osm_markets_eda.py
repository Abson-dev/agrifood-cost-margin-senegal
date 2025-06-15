import osmnx as ox
import matplotlib.pyplot as plt

# Query OSM for market locations in Kenya
place = 'Kenya'
tags = {'amenity': 'market'}
markets = ox.geometries_from_place(place, tags)

# Inspect the data
print("First 5 rows of OSM Market data:")
print(markets.head())
print("\nSummary statistics:")
print(markets.describe())

# Visualize market locations
markets.plot(marker='o', color='red', markersize=5)
plt.title(f'Market Locations in {place} (OSM)')
plt.show()

# Calculate the number of markets
num_markets = len(markets)
print(f'Number of markets in {place}: {num_markets}')