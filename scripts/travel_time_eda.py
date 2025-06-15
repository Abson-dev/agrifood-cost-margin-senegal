import rasterio
import matplotlib.pyplot as plt

# Load the downloaded GeoTIFF file
with rasterio.open('travel_time.tif') as src:
    travel_time = src.read(1)

# Visualize the travel time map
plt.imshow(travel_time, cmap='viridis')
plt.colorbar(label='Travel Time (minutes)')
plt.title('Global Travel Time to Cities')
plt.show()

# Calculate mean travel time
mean_travel_time = travel_time.mean()
print(f'Mean travel time: {mean_travel_time} minutes')