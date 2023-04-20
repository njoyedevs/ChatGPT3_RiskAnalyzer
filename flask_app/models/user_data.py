from flask_app.config.mysqlconnection import connectToMySQL
from flask_app import DATABASE
from flask import session

class User_Data:
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
        return f'(User_Data Object) id: {self.id}, date: {self.date}, spcpi: {self.spcpi}, spcpi_m_s: {self.spcpi_m_s}, spcpi_m_fe: {self.spcpi_m_fe}, spcpi_m_fes: {self.spcpi_m_fes}, trim_mean_pce: {self.trim_mean_pce}, sixteenP_trim_mean_cpi: {self.sixteenP_trim_mean_cpi}, median_cpi: {self.median_cpi}, fpcpi: {self.fpcpi}, fpcpi_m_fe: {self.fpcpi_m_fe}, created_at: {self.created_at}, updated_at: {self.updated_at}'

    # Save historical data into database
    # Make sure to create a user before running this.
    @classmethod
    def save_user_data(cls, data):
        
        data_dict = {
            'date': data['date'],
            'spcpi': data['spcpi'],
            'spcpi_m_s': data['spcpi_m_s'],
            'spcpi_m_fe': data['spcpi_m_fe'],
            'spcpi_m_fes': data['spcpi_m_fes'],
            'trim_mean_pce': data['trim_mean_pce'],
            'sixteenP_trim_mean_cpi': data['sixteenP_trim_mean_cpi'],
            'median_cpi': data['median_cpi'],
            'fpcpi': data['fpcpi'],
            'fpcpi_m_fe': data['fpcpi_m_fe'],
            'user_id': session['user_id']
        }
        # print(data)
        
        query = """INSERT INTO user_data (date , spcpi , spcpi_m_s, spcpi_m_fe , spcpi_m_fes , trim_mean_pce, sixteenP_trim_mean_cpi, median_cpi, fpcpi, fpcpi_m_fe, created_at, updated_at, user_id)
                VALUES (%(date)s , %(spcpi)s , %(spcpi_m_s)s, %(spcpi_m_fe)s , %(spcpi_m_fes)s , %(trim_mean_pce)s , %(sixteenP_trim_mean_cpi)s, %(median_cpi)s , %(fpcpi)s , %(fpcpi_m_fe)s , NOW(), NOW(), %(user_id)s);"""
        # print(query)
                
        # id = return connectToMySQL('risk_analysis').query_db(query, data)
        return connectToMySQL(DATABASE).query_db(query, data_dict)

    # Get the last user created data set
    @classmethod
    def get_last_user_data(cls, id):
        query = "SELECT * FROM user_data WHERE id = %(id)s;"
        
        results = connectToMySQL(DATABASE).query_db(query, {'id': id})
        # print(results)
        data = []
        
        for row in results:
            data.append( cls(row) )
            
        return data
        
    
