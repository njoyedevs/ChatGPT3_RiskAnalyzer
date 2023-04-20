import pandas as pd
from flask_app.config.mysqlconnection import connectToMySQL
from flask_app import DATABASE
from flask_app.models import econ_data, user_data, results, chatgpt_data, users
from flask import session
import os
from dotenv import load_dotenv
import plotly.colors as colors
import plotly.offline as pyo
import plotly.graph_objects as go

load_dotenv()

model_id = os.getenv('MODEL_ID')

class Result:
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
        self.mean = data['mean']
        self.created_at = data['created_at']
        self.updated_at = data['updated_at']
        
    def __repr__(self):
        return f'(Result Object) id: {self.id}, date: {self.date}, spcpi: {self.spcpi}, spcpi_m_s: {self.spcpi_m_s}, spcpi_m_fe: {self.spcpi_m_fe}, spcpi_m_fes: {self.spcpi_m_fes}, trim_mean_pce: {self.trim_mean_pce}, sixteenP_trim_mean_cpi: {self.sixteenP_trim_mean_cpi}, median_cpi: {self.median_cpi}, fpcpi: {self.fpcpi}, fpcpi_m_fe: {self.fpcpi_m_fe}, created_at: {self.created_at}, updated_at: {self.updated_at}'

    @classmethod
    def save_results_data(cls, risk_analysis_results, user_data_id):
        
        # print(risk_analysis_results)
        
        data = {
            'date': risk_analysis_results['date'][0],
            'spcpi': risk_analysis_results['spcpi'][0],
            'spcpi_m_s': risk_analysis_results['spcpi_m_s'][0],
            'spcpi_m_fe': risk_analysis_results['spcpi_m_fe'][0],
            'spcpi_m_fes': risk_analysis_results['spcpi_m_fes'][0],
            'trim_mean_pce': risk_analysis_results['trim_mean_pce'][0],
            'sixteenP_trim_mean_cpi': risk_analysis_results['sixteenP_trim_mean_cpi'][0],
            'median_cpi': risk_analysis_results['median_cpi'][0],
            'fpcpi': risk_analysis_results['fpcpi'][0],
            'fpcpi_m_fe': risk_analysis_results['fpcpi_m_fe'][0],
            'mean': risk_analysis_results['mean'][0],
            'user_id': session['user_id'],
            'user_data_id': user_data_id,
        }
        # print(data)
        
        query = """INSERT INTO results (date, spcpi , spcpi_m_s , spcpi_m_fe, spcpi_m_fes , trim_mean_pce, sixteenP_trim_mean_cpi, median_cpi, fpcpi, fpcpi_m_fe, mean, created_at, updated_at, user_id, user_data_id)
                VALUES (%(date)s, %(spcpi)s , %(spcpi_m_s)s, %(spcpi_m_fe)s , %(spcpi_m_fes)s , %(trim_mean_pce)s , %(sixteenP_trim_mean_cpi)s, %(median_cpi)s , %(fpcpi)s , %(fpcpi_m_fe)s , %(mean)s , NOW(), NOW(), %(user_id)s , %(user_data_id)s);"""
        # print(query)
                
        # id = return connectToMySQL('risk_analysis').query_db(query, data)
        return connectToMySQL(DATABASE).query_db(query, data)
    
    # Get all results data from database
    @classmethod
    def get_all_results_data(cls):
        query = "SELECT * FROM results;"
        
        results = connectToMySQL(DATABASE).query_db(query)
        # print(results)
        data = []
        
        for row in results:
            data.append( cls(row) )
            
        return data
    
    # Get last risk analysis result
    @classmethod
    def get_last_fed_risk_result(cls):
        query = "SELECT * FROM results WHERE date = %(date)s;"
        
        # Get last date from econ database
        date = econ_data.Econ_Data.get_last_econ_data()
        # print(date[0]['date'].strftime('%Y-%m-%d'))

        results = connectToMySQL(DATABASE).query_db(query, {'date': date[0]['date'].strftime('%Y-%m-%d')})
        # print(results)
        # print(results[0]['mean'])
        last_fed_risk_result = results[0]['mean']
            
        return last_fed_risk_result
    
    # Get last risk analysis result
    @classmethod
    def get_last_risk_data(cls):
        query = "SELECT * FROM results WHERE id = %(result_id)s;"
        
        last_risk_data = connectToMySQL(DATABASE).query_db(query, {'result_id': session['result_id']})
        # print(last_risk_data)
        
        data = []
        
        for row in last_risk_data:
            data.append( cls(row) )
            
        return data
    
    # Risk Analysis 
    @classmethod
    def risk_analysis(cls, user_data_id):
        
        # Get data from FRED API
        # get_data(fred_api_key)
        
        # Get all econ data objects
        hist_data_objs = econ_data.Econ_Data.get_all_econ_data()
        
        # Use a list comprehension to extract the data into a list of dictionaries
        extracted_hist_data = [{'date': row.date, 'spcpi': row.spcpi, 'spcpi_m_s': row.spcpi_m_s,
                    'spcpi_m_fe': row.spcpi_m_fe, 'spcpi_m_fes': row.spcpi_m_fes,
                    'trim_mean_pce': row.trim_mean_pce, 'sixteenP_trim_mean_cpi': row.sixteenP_trim_mean_cpi,
                    'median_cpi': row.median_cpi, 'fpcpi': row.fpcpi, 'fpcpi_m_fe': row.fpcpi_m_fe,
                    'created_at': row.created_at, 'updated_at': row.updated_at}
                    for row in hist_data_objs]
        
        # Convert extracted_objs into a DataFrame
        historical_df = pd.DataFrame(extracted_hist_data)
        # print(historical_df)
        
        # Import User Data
        user_data_objs = user_data.User_Data.get_last_user_data(user_data_id)
        
        # Use a list comprehension to extract the data into a list of dictionaries
        extracted_user_data = [{'date': row.date, 'spcpi': row.spcpi, 'spcpi_m_s': row.spcpi_m_s,
                    'spcpi_m_fe': row.spcpi_m_fe, 'spcpi_m_fes': row.spcpi_m_fes,
                    'trim_mean_pce': row.trim_mean_pce, 'sixteenP_trim_mean_cpi': row.sixteenP_trim_mean_cpi,
                    'median_cpi': row.median_cpi, 'fpcpi': row.fpcpi, 'fpcpi_m_fe': row.fpcpi_m_fe,
                    'created_at': row.created_at, 'updated_at': row.updated_at}
                    for row in user_data_objs]
        
        # Convert extracted_objs into a DataFrame
        user_df = pd.DataFrame(extracted_user_data)
        user_df_copy = user_df.iloc[:,1:].copy()
        user_df_copy = user_df_copy.drop(columns=["created_at", "updated_at"])
        # print(user_df_copy)
        
        # Get the date from the user_df
        user_df_dates = user_df.iloc[:,0].copy()
        user_df_dates = user_df_dates.iloc[0]
        # print(user_df_dates)
        # print(user_df_dates.iloc[0])
        # print(type(user_df_dates))
        
        # Combine the historical_df and user_df
        combined_data = pd.concat([historical_df, user_df], axis=0)
        combined_data = combined_data.reset_index()
        combined_data = combined_data.drop(columns=['index'])
        # combined_data.to_csv('combined_data.csv')
        # print(combined_data)
        
        # Splice out the date column and other non necessary columns
        df = combined_data.iloc[:,1:].copy()
        df_dates = combined_data.iloc[:,0].copy()
        df = df.drop(columns=["created_at", "updated_at"])
        # print(df)
        
        # Create a dictionary of percentiles for each column
        
        # Create a list of dictionaries
        list_of_col_per = []
        
        # Loop through each column
        for column in range(len(df.columns)):
            
            percentiles = [0,.125,.25,.375,.5,.625,.75,.875,1]
            
            # get one column
            # print(df.iloc[:,column])
            
            # get percentiles for one column
            # print(df.iloc[:,column].describe([0,.125,.25,.375,.5,.625,.75,.875,1], datetime_is_numeric=True)) 
            
            # get one value from one column
            # print(df.iloc[:,column][0])
            
            # descrive one column
            describe_one = df.iloc[:,column].describe(percentiles)
            
            # Create dictionary for one column
            dict_of_col_per = {
                'zero' : describe_one.loc['0%'],
                'twelve' : describe_one.loc['12.5%'],
                'twenty_five' : describe_one.loc['25%'],
                'tirty_seven' : describe_one.loc['37.5%'],
                'fifty' : describe_one.loc['50%'],
                'sisty_two' : describe_one.loc['62.5%'],
                'seventy_five' : describe_one.loc['75%'],
                'eighty_seven' : describe_one.loc['87.5%'],
                'one_hundred' : describe_one.loc['100%']
            }
        
            list_of_col_per.append(dict_of_col_per)
        # print(list_of_col_per)
            
        # Create a DataFrame with 0 in every value and len(df) # rows and df.columns
        results_df = pd.DataFrame(0, index=range(len(user_df_copy)), columns=df.columns)

        # print(results_df.shape)
        # print(df.shape)
        # print(range(len(df)))
        # print(range(len(df.columns)))
        
        for column in range(len(user_df_copy.columns)):
            
        #   # get one column
            # print(user_df_copy.iloc[:1,column])
        
            # categorize each value in one column based on risk level bawed on normal curve percentiles
            if (user_df_copy.iloc[:,column] >= list_of_col_per[column]['zero']).any() and (user_df_copy.iloc[:,column] < list_of_col_per[column]['twelve']).any():
                # print('This is risk level 0')
                results_df.iloc[:,column] = 0
            elif (user_df_copy.iloc[:,column] >= list_of_col_per[column]['twelve']).any() and (user_df_copy.iloc[:,column] < list_of_col_per[column]['twenty_five']).any():
                # print('This is risk level 1')
                results_df.iloc[:,column] = 1
            elif (user_df_copy.iloc[:,column] >= list_of_col_per[column]['twenty_five']).any() and (user_df_copy.iloc[:,column] < list_of_col_per[column]['tirty_seven']).any():
                # print('This is risk level 2')
                results_df.iloc[:,column] = 2
            elif (user_df_copy.iloc[:,column] >= list_of_col_per[column]['tirty_seven']).any() and (user_df_copy.iloc[:,column] < list_of_col_per[column]['fifty']).any():
                # print('This is risk level 3')
                results_df.iloc[:,column] = 3
            elif (user_df_copy.iloc[:,column] >= list_of_col_per[column]['fifty']).any() and (user_df_copy.iloc[:,column] < list_of_col_per[column]['sisty_two']).any():
                # print('This is risk level 4')
                results_df.iloc[:,column] = 4
            elif (user_df_copy.iloc[:,column] >= list_of_col_per[column]['sisty_two']).any() and (user_df_copy.iloc[:,column] < list_of_col_per[column]['seventy_five']).any():
                # print('This is risk level 5')
                results_df.iloc[:,column] = 5
            elif (user_df_copy.iloc[:,column] >= list_of_col_per[column]['seventy_five']).any() and (user_df_copy.iloc[:,column] < list_of_col_per[column]['eighty_seven']).any():
                # print('This is risk level 6')
                results_df.iloc[:,column] = 6
            elif (user_df_copy.iloc[:,column] >= list_of_col_per[column]['eighty_seven']).any() and (user_df_copy.iloc[:,column] <= list_of_col_per[column]['one_hundred']).any():
                # print('This is risk level 7')
                results_df.iloc[:,column] = 7
            else:
                print('There was an error when categorizing the risk level')
        
        # Add a column of the sum of each row
        # results_df['sum'] = results_df.sum(axis=1)
        results_df['mean'] = results_df.mean(axis=1)

        # Print the updated DataFrame
        # print(results_df)
        # final_risk_result_df = pd.concat([user_df_dates, results_df], axis=0, ignore_index=True)
        results_df.insert(0, 'date', user_df_dates)
        # print(results_df)
        # results_df.to_csv(r'Final_Project/flask_app/models/recent_results.csv')
        
        return results_df

    # Get risk report (combine tables users-[user_id]-chatgpt_data-[result_id]-results-[user_data_id]-user_data]) and use to populate dashboard
    # Connect the gauge value to database and change indicator.  Input prompt below gauge
    # Data map (user input-submit, input into prompt, query chatgpt, save data, pull data to update the gauge and change indicator.)
    @classmethod
    def get_risk_report(cls, id):
        
        query = "SELECT * FROM users LEFT JOIN chatgpt_data ON chatgpt_data.user_id=users.id LEFT JOIN results ON chatgpt_data.result_id=results.id LEFT JOIN user_data ON results.user_data_id=user_data.id WHERE chatgpt_data.id = %(id)s;"
        
        db_results = connectToMySQL(DATABASE).query_db(query,{'id': id})
        
        # print(db_results)
        
        user = cls(db_results[0])
        
        for row in db_results:
            
            user_info = {
                **row,
            }
            
            user_info_inst = users.User(user_info)
            
            chatgpt_data_info = {
                **row,
                "id" : row['chatgpt_data.id'],
                "created_at" : row['chatgpt_data.created_at'],
                "updated_at" : row['chatgpt_data.updated_at']
            }
            
            chatgpt_data_inst = chatgpt_data.ChatGPT3(chatgpt_data_info)
            
            results_data_info = {
                **row,
                "result_id" : row['results.id'],
                "created_at" : row['results.created_at'],
                "updated_at" : row['results.updated_at'],
                "user_id" : row['results.user_id']
            }
            
            results_data_inst = results.Result(results_data_info)
            
            user_data_info = {
                **row,
                "id" : row['user_data.id'],
                "date" : row['user_data.date'],
                "spcpi" : row['user_data.spcpi'],
                "spcpi_m_s" : row['user_data.spcpi_m_s'],
                "spcpi_m_fe" : row['user_data.spcpi_m_fe'],
                "spcpi_m_fes" : row['user_data.spcpi_m_fes'],
                "trim_mean_pce" : row['user_data.trim_mean_pce'],
                "sixteenP_trim_mean_cpi" : row['user_data.sixteenP_trim_mean_cpi'],
                "median_cpi" : row['user_data.median_cpi'],
                "fpcpi" : row['user_data.fpcpi'],
                "fpcpi_m_fe" : row['user_data.fpcpi_m_fe'],
                "created_at" : row['user_data.created_at'],
                "updated_at" : row['user_data.updated_at'],
                "user_id" : row['user_data.user_id']
            }
            
            user_data_inst = user_data.User_Data(user_data_info)
            # print(user_data_inst)
            # print(user_data_inst.spcpi)
        
            # User Object
            user.user_data = user_data_inst
            # print(user.user_data)
            
            # ChatGPT3 Data Object
            user.chatgpt_data = chatgpt_data_inst
            # print(user.chatgpt_data)
            
            # Results Data Object
            user.chatgpt_data.results = results_data_inst
            # print(user.chatgpt_data.results)
            
            # User Data Object
            user.chatgpt_data.results.user_data = user_data_inst
            # print(user.chatgpt_data.results.user_data)
            
            # Results Data Object
            # print(user)
            
            # Save the completion percentage to the session
            session['completion'] = user.chatgpt_data.completion
            # print(session['completion'])

        return user
    
    # FINAL VERSION
    @classmethod
    def create_guage_plotly(cls):
        
        # Define the color scale
        color_scale = colors.sequential.Viridis
        
        # get previous risk level
        last_result = Result.get_last_fed_risk_result()
        # print(last_result)
        
        fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = session['completion'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Meter", 'font': {'size': 30}},
        delta = {'reference': last_result, 'increasing': {'color': "RebeccaPurple"}},
        gauge = {
            'axis': {'range': [0, 8], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "purple", 'thickness': 0.3},
            'bgcolor': "blue",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 1], 'color': color_scale[0]},
                {'range': [1, 2], 'color': color_scale[1]},
                {'range': [2, 3], 'color': color_scale[2]},
                {'range': [3, 4], 'color': color_scale[3]},
                {'range': [4, 5], 'color': color_scale[4]},
                {'range': [5, 6], 'color': color_scale[5]},
                {'range': [6, 7], 'color': color_scale[6]},
                {'range': [7, 8], 'color': color_scale[7]}
                ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 5.5}},

        ))

        fig.update_layout(paper_bgcolor = "black", font = {'color': "white", 'family': "Arial"}, height=380, width=730)
        
        # fig.show()
        # fig.write_image("Final_Project/flask_app/static/images/risk_meter_3.png")
        # fig.show()
        # Open the PNG image
        
        # fig.update_layout(plot_bgcolor='lavender', title_text="Fed Measure Components",title_font_size=40, title_pad_l=240, paper_bgcolor='black', legend_font_color='white', title_font_color='white', font_color='white')

        return pyo.plot(fig, output_type='div')
