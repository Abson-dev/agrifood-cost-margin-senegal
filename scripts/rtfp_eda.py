import pandas as pd
import matplotlib.pyplot as plt

# Load the downloaded CSV file
data = pd.read_csv('rtfp_data.csv')

# Inspect the data
print("First 5 rows of RTFP data:")
print(data.head())
print("\nSummary statistics:")
print(data.describe())

# Visualize: Plot rice prices over time
rice_data = data[data['commodity'] == 'Rice']
plt.plot(rice_data['date'], rice_data['price'])
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Rice Prices Over Time (RTFP)')
plt.show()