from fredapi import Fred
from datetime import datetime, timedelta
import pandas as pd

# Replace YOUR_API_KEY with your actual API key
fred = Fred(api_key=YOUR_API_KEY) 

# Define the list of series IDs to download
# series_ids = ['CPILFESL', 'PCECTrimMED', 'CPILFESLNO', 'CPIMEDSL', 'CPILFSSM', 'CPILFSEXA', 'CPILFVOTT01', 'CPILFV01', 'CPITRIM1M162N']
series_ids = ['STICKCPIM159SFRBATL',
            'STICKCPIXSHLTRM159SFRBATL',
            'CORESTICKM159SFRBATL',
            'CRESTKCPIXSLTRM159SFRBATL',
            'PCETRIM12M159SFRBDAL',
            'TRMMEANCPIM159SFRBCLE',
            'MEDCPIM159SFRBCLE',
            'FLEXCPIM159SFRBATL',
            'COREFLEXCPIM159SFRBATL']

# calculate the start and end dates
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=91)).strftime('%Y-%m-%d')
print(end_date)
print(start_date)

# Download the data for each series and store it in a dictionary
data = {}
for series_id in series_ids:
    # get the most recent observation for the series
    series_data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    data[series_id] = series_data
# print(data)

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame.from_dict(data)
print(df)

# Reset the index to be the date
df = df.reset_index()

# Rename the columns to be more descriptive
df = df.rename(columns={'index': 'Date',
                        'STICKCPIM159SFRBATL': 'Sticky_Price_CPI',
                        'STICKCPIXSHLTRM159SFRBATL': 'Sticky_Price_CPI_Less_Shelter',
                        'CORESTICKM159SFRBATL': 'Sticky_Price_CPI_Less_Food_Energy',
                        'CRESTKCPIXSLTRM159SFRBATL': 'Sticky_Price_CPI_Less_Food_Energy_Shelter',
                        'PCETRIM12M159SFRBDAL': 'Trimmed_Mean_PCE_Inflation_Rate',
                        'TRMMEANCPIM159SFRBCLE': '16_Percent_Trimmed_Mean_CPI',
                        'MEDCPIM159SFRBCLE': 'Median_CPI',
                        'FLEXCPIM159SFRBATL': 'Flexible_Price_CPI',
                        'COREFLEXCPIM159SFRBATL': 'Flexible_Price_CPI_Less_Food_Energy'})

# Convert the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the date column as the index
df = df.set_index('Date')

# # Resample the data to be monthly and calculate the monthly percentage change
# df = df.resample('M').last().pct_change()

# Drop the first row, which will have NaN values due to the percentage change calculation
# df = df.drop(df.index[0])

# Display the resulting DataFrame
print(df.head())

# df.to_excel("output.xlsx") 
print("Saving CSV...")
df.to_csv('recent_data.csv') 
print("CSV saved.")