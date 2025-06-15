import pandas as pd
import matplotlib.pyplot as plt

# Load the downloaded CSV file
data = pd.read_csv('wfp_food_prices.csv')

# Inspect the data
print("First 5 rows of WFP VAM data:")
print(data.head())
print("\nSummary statistics:")
print(data.describe())

# Visualize: Plot maize prices across markets in Kenya
kenya_maize = data[(data['country'] == 'Kenya') & (data['commodity'] == 'Maize')]
plt.scatter(kenya_maize['market'], kenya_maize['price'])
plt.xlabel('Market')
plt.ylabel('Price')
plt.title('Maize Prices Across Markets in Kenya (WFP VAM)')
plt.xticks(rotation=45)
plt.show()