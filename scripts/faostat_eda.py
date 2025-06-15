import faostat
import pandas as pd
import matplotlib.pyplot as plt

# List datasets to find the correct code for Agricultural Producer Prices
datasets = faostat.list_datasets()
for ds in datasets:
    if 'producer' in ds['label'].lower() and 'prices' in ds['label'].lower():
        print(ds['code'], ds['label'])

# Assume the dataset code is 'PP' (adjust based on actual code)
data = faostat.get_data('PP', area='all', item='all', year='all')

# Convert to DataFrame if necessary
if isinstance(data, list):
    data = pd.DataFrame(data)

# Inspect the data
print("First 5 rows of FAOSTAT data:")
print(data.head())
print("\nSummary statistics:")
print(data.describe())

# Visualize: Plot maize prices in Kenya over time
maize_kenya = data[(data['item_code'] == 'Maize') & (data['area_code'] == 'Kenya')]
plt.plot(maize_kenya['year'], maize_kenya['value'])
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Maize Prices in Kenya (FAOSTAT)')
plt.show()