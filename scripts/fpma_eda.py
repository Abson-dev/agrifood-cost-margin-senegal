import pandas as pd
import matplotlib.pyplot as plt

# Load the downloaded CSV file
data = pd.read_csv('fpma_data.csv')

# Inspect the data
print("First 5 rows of FPMA data:")
print(data.head())
print("\nSummary statistics:")
print(data.describe())

# Visualize: Plot food price index for Kenya over time
kenya_data = data[data['country'] == 'Kenya']
plt.plot(kenya_data['date'], kenya_data['price_index'])
plt.xlabel('Date')
plt.ylabel('Price Index')
plt.title('Food Price Index in Kenya (FPMA)')
plt.show()