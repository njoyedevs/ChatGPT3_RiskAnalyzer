from flask_app.config.mysqlconnection import connectToMySQL
from flask_app import DATABASE
from datetime import datetime, timedelta
import pandas as pd
import os
from fredapi import Fred
import plotly.express as px
import plotly.colors as colors
import plotly.offline as pyo
from dotenv import load_dotenv

load_dotenv()

fred_api_key = os.getenv("FRED_API_KEY")

# ALL YOU NEED TO DO IS UPDATE THE LAST FEDERAL RESERVE DATE AS THEY COME.  COULD WRITE A SCRIPT TO DO THIS USING TIMING OF THE RELEASE EACH MONTH.

class Econ_Data:
    def __init__(self,data):
        self.id = data['id']
        self.date = data['date']
        self.spcpi = data['spcpi']
        self.spcpi_m_s = data['spcpi_m_s']
        self.spcpi_m_fe = data['spcpi_m_fe']
        self.spcpi_m_fes = data['spcpi_m_fes']
        self.trim_mean_pce = data['trim_mean_pce']
        self.sixteenP_trim_mean_cpi = data['sixteenP_trim_mean_cpi']
        self.median_cpi = data['median_cpi']
        self.fpcpi = data['fpcpi']
        self.fpcpi_m_fe = data['fpcpi_m_fe']
        self.created_at = data['created_at']
        self.updated_at = data['updated_at']
        
    def __repr__(self):
        return f'(Econ Object) id: {self.id}, date: {self.date}, spcpi: {self.spcpi}, spcpi_m_s: {self.spcpi_m_s}, spcpi_m_fe: {self.spcpi_m_fe}, spcpi_m_fes: {self.spcpi_m_fes}, trim_mean_pce: {self.trim_mean_pce}, sixteenP_trim_mean_cpi: {self.sixteenP_trim_mean_cpi}, median_cpi: {self.median_cpi}, fpcpi: {self.fpcpi}, fpcpi_m_fe: {self.fpcpi_m_fe}, created_at: {self.created_at}, updated_at: {self.updated_at}'

    # Create Empty DataFrame
    @classmethod
    def create_empty_df(cls,df):
        
        # Create a DataFrame with 0 in every value and 469 rows and 9 columns
        results_df = pd.DataFrame(0, index=range(len(df)), columns=df.columns)

        # Print the first 5 rows of the DataFrame to check that it was created correctly
        # print(results_df.head())
        
        return results_df
    
    # FINAL VERSION
    @classmethod
    def create_chart_plotly(cls):
        
        # Plotly example df
        # df = px.data.stocks(indexed=True)-1
        # print(df)
        
        # Define the color scale
        color_scale = colors.sequential.Viridis
        
        # Get all econ data objects
        hist_data_objs = Econ_Data.get_all_econ_data()
        
        # Use a list comprehension to extract the data into a list of dictionaries
        extracted_hist_data = [{'date': row.date, 'spcpi': row.spcpi, 'spcpi_m_s': row.spcpi_m_s,
                    'spcpi_m_fe': row.spcpi_m_fe, 'spcpi_m_fes': row.spcpi_m_fes,
                    'trim_mean_pce': row.trim_mean_pce, 'sixteenP_trim_mean_cpi': row.sixteenP_trim_mean_cpi,
                    'median_cpi': row.median_cpi, 'fpcpi': row.fpcpi, 'fpcpi_m_fe': row.fpcpi_m_fe,
                    'created_at': row.created_at, 'updated_at': row.updated_at}
                    for row in hist_data_objs]
        
        # Convert extracted_objs into a DataFrame
        historical_df = pd.DataFrame(extracted_hist_data)
        historical_df.set_index('date', inplace=True)
        historical_df = historical_df.drop(columns=["created_at", "updated_at"])
        historical_df = historical_df.rename_axis('Component', axis=1)
        # print(historical_df)
        
        fig = px.area(historical_df, facet_col="Component", facet_col_wrap=3, color_discrete_sequence = color_scale, width = 1250, height = 1000)
        fig.update_layout(plot_bgcolor='lavender', title_text="FRED API Components**",title_font_size=20, title_pad_l=375, paper_bgcolor='black', legend_font_color='white', title_font_color='white', font_color='white')
        # fig.show()
        
        # pio.write_html(fig, file='Final_Project/flask_app/templates/plotly_chart.html', auto_open=True)
        
        # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        # print(graphJSON)
        
        # pyo.plot(fig, filename='Final_Project/flask_app/templates/plotly_chart_2.html', auto_open=True)
        return pyo.plot(fig, output_type='div')
    
    # Get all econ data from database
    @classmethod
    def get_all_econ_data(cls):
        query = "SELECT * FROM econ_data;"
        
        results = connectToMySQL(DATABASE).query_db(query)
        # print(results)
        data = []
        
        for row in results:
            data.append( cls(row) )
            
        return data

    # Get last date from database
    @classmethod
    def get_last_econ_data(cls):
        query = "SELECT date FROM econ_data ORDER BY id DESC LIMIT 1;"
        
        last_date = connectToMySQL(DATABASE).query_db(query)
        # print(last_date)

        return last_date

    # Generate Last Date Only From FRED API
    @staticmethod
    def get_last_date_from_FRED():
        
        # Replace YOUR_API_KEY with your actual API key
        fred = Fred(api_key=fred_api_key) 

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
        start_date = (datetime.now() - timedelta(days=31)).strftime('%Y-%m-%d')
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
        df.to_csv(r'recent_data.csv') 
        
        return df