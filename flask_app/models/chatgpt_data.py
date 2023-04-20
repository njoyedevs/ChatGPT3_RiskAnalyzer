from flask_app.config.mysqlconnection import connectToMySQL
from flask_app import DATABASE
from flask_app.models import user_data, results
from flask import session
import os
import openai
from dotenv import load_dotenv
import pandas as pd
import jsonlines

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
model_name = os.getenv('MODEL_NAME')


class ChatGPT3:
    def __init__(self, data):
        self.id = data['id']
        self.prompt = data['prompt']
        self.completion = data['completion']
        self.completion_tokens = data['completion_tokens']
        self.prompt_tokens = data['prompt_tokens']
        self.total_tokens = data['total_tokens']
        self.object = data['object']
        self.model = data['model']
        self.completion_id = data['completion_id']
        self.accuracy = data['accuracy']
        self.created_at = data['created_at']
        self.updated_at = data['updated_at']

    def __repr__(self):
        return f'(ChatGPT3 Object) id: {self.id}, prompt: {self.prompt}, completion: {self.completion}, completion_tokens: {self.completion_tokens}, prompt_tokens: {self.prompt_tokens}, total_tokens: {self.total_tokens}, object: {self.object}, model: {self.model}, completion_id: {self.completion_id}, accuracy: {self.accuracy}, created_at: {self.created_at}, updated_at: {self.updated_at}'

    # Get ChatpGPT Response query, save date,
    @classmethod
    def save_chatgpt_response(cls):

        # print(openai.FineTune.list())

        # print(session['user_data_id'])

        # Get last user_data
        data = user_data.User_Data.get_last_user_data(session['user_data_id'])
        # print(data)

        # Fine Tune Key
        fine_tune_id = model_name

        # prompt="On 6/01/2023, the Sticky Price CPI was 1.3, Sticky Price CPI Less Shelter was 2.2, Sticky Price CPI Less Food Energy was 3.2, Sticky Price CPI Less Food Energy Shelter was 1.5, Trimmed Mean PCE Inflation Rate was 1.7, 16 Percent Trimmed Mean CPI was 2.2, Median CPI was 3.6, Flexible Price CPI was 2.5, Flexible Price CPI Less Food Energy was 1.2/n/n***###***/n/n",
        prompt = f"On {data[0].date}, the Sticky Price CPI was {data[0].spcpi}, Sticky Price CPI Less Shelter was {data[0].spcpi_m_s}, Sticky Price CPI Less Food Energy was {data[0].spcpi_m_fe}, Sticky Price CPI Less Food Energy Shelter was {data[0].spcpi_m_fes}, Trimmed Mean PCE Inflation Rate was {data[0].trim_mean_pce}, 16 Percent Trimmed Mean CPI was {data[0].sixteenP_trim_mean_cpi}, Median CPI was {data[0].median_cpi}, Flexible Price CPI was {data[0].fpcpi}, Flexible Price CPI Less Food Energy was {data[0].fpcpi_m_fe}/n/n***###***/n/n",

        # Get response from ChatGPT3
        result = openai.Completion.create(
            model=fine_tune_id,
            temperature=0.0,
            max_tokens=3,
            prompt=prompt
            )

        # print(result)

        # Get last risk result
        expected_value = results.Result.get_last_risk_data()

        # Get ChatGPT3 response
        actual_value = result['choices'][0]['text']

        # Calculate accuracy
        risk_accuracy = round(float(actual_value)/float(expected_value[0].mean), 3)
        # print(f'\nExpected Value: {expected_value}, Actual Value: {actual_value}, Accuracy: {risk_accuracy}\n')

        data_dict = {
            'prompt': prompt,
            'completion': result['choices'][0]['text'],
            'completion_tokens': result['usage']['completion_tokens'],
            'prompt_tokens': result['usage']['prompt_tokens'],
            'total_tokens': result['usage']['total_tokens'],
            'object': result['object'],
            'model': result['model'],
            'completion_id': result['id'],
            'accuracy': risk_accuracy,
            'user_id': session['user_id'],
            'result_id': session['result_id']
        }
        # print(data)

        query = """INSERT INTO chatgpt_data (prompt, completion, completion_tokens, prompt_tokens, total_tokens, object, model, completion_id, accuracy, created_at, updated_at, user_id, result_id)
                VALUES (%(prompt)s, %(completion)s, %(completion_tokens)s, %(prompt_tokens)s, %(total_tokens)s, %(object)s, %(model)s, %(completion_id)s, %(accuracy)s, NOW(), NOW(), %(user_id)s , %(result_id)s);"""
        # print(query)

        # id = return connectToMySQL('risk_analysis').query_db(query, data)
        return connectToMySQL(DATABASE).query_db(query, data_dict)

    # FINAL VERSION
    @classmethod
    def retrain_chatgpt(cls):
        
        # Get last user_data
        last_user_data_obj = user_data.User_Data.get_last_user_data(session['user_data_id'])
        extracted_user_data = [{'date': row.date, 'spcpi': row.spcpi, 'spcpi_m_s': row.spcpi_m_s,
                    'spcpi_m_fe': row.spcpi_m_fe, 'spcpi_m_fes': row.spcpi_m_fes,
                    'trim_mean_pce': row.trim_mean_pce, 'sixteenP_trim_mean_cpi': row.sixteenP_trim_mean_cpi,
                    'median_cpi': row.median_cpi, 'fpcpi': row.fpcpi, 'fpcpi_m_fe': row.fpcpi_m_fe,
                    'created_at': row.created_at, 'updated_at': row.updated_at}
                    for row in last_user_data_obj]
        
        # Convert extracted_objs into a DataFrame
        user_df = pd.DataFrame(extracted_user_data)
        date = user_df['date']
        # print(date)
        user_df_copy = user_df.iloc[:,1:].copy()
        user_df_copy = user_df_copy.drop(columns=["created_at", "updated_at"])
        # print(user_df_copy)
        
        # Get last risk results
        extracted_results_data =  results.Result.get_last_risk_data()
        extracted_results_data = [{'date': row.date, 'spcpi': row.spcpi, 'spcpi_m_s': row.spcpi_m_s,
                    'spcpi_m_fe': row.spcpi_m_fe, 'spcpi_m_fes': row.spcpi_m_fes,
                    'trim_mean_pce': row.trim_mean_pce, 'sixteenP_trim_mean_cpi': row.sixteenP_trim_mean_cpi,
                    'median_cpi': row.median_cpi, 'fpcpi': row.fpcpi, 'fpcpi_m_fe': row.fpcpi_m_fe, 'mean': row.mean,
                    'created_at': row.created_at, 'updated_at': row.updated_at}
                    for row in extracted_results_data]
        
        # Convert extracted_objs into a DataFrame
        results_df = pd.DataFrame(extracted_results_data)
        results_df_copy = results_df.iloc[:,1:].copy()
        results_df_copy = results_df_copy.drop(columns=["created_at", "updated_at"])
        # print(results_df_copy)
        
        def merge_dataframes(df1, df2):
            # Get the list of common columns
            # reverse list
            # list = df1.columns.tolist()
            # reverse_list = list[::-1]
            df1_cols = list(df1.columns)
            # print(df1_cols)
            df2_cols = list(df2.columns)
            # print(df2_cols)
            # remove last 'mean' from list
            df2_cols = df2_cols[:-1]
            # print(df2_cols)
            
            df = pd.DataFrame()
            
            # Pop df2 Mean column for storage later usage in variable mean
            mean = df2.pop('mean')
            # print(mean)
            # print(df2)

            for idx in range(len(df1_cols)):
                
                df[f'{df1_cols[idx]}'] = df1.iloc[:,idx]
                # print(df[f'{df1_cols[idx]}'])
                
                df[f'{df2_cols[idx]}_risk'] = df2.iloc[:,idx]
                # print(df[f'{df2_cols[idx]} risk'])
                
                new_df = pd.concat([df[f'{df1_cols[idx]}'], df[f'{df2_cols[idx]}_risk']], axis=1)
                # print(new_df)
                # print(df)
            
            df = pd.concat([df, mean], axis=1)
            # print(df)
            
            return df
        
        merged_df = merge_dataframes(user_df_copy, results_df_copy)
        merged_df = pd.concat([date, merged_df], axis=1)
        # print(merged_df)
    
        # Convert to Jsonl
        def create_jsonl(df, filename):
            
            with jsonlines.open(filename, mode='w') as writer:
                for a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t in zip(df['date'], df['spcpi'], df['spcpi_risk'], df['spcpi_m_s'], df['spcpi_m_s_risk'], df['spcpi_m_fe'], df['spcpi_m_fe_risk'], df['spcpi_m_fes'], df['spcpi_m_fes_risk'], df['trim_mean_pce'], df['trim_mean_pce_risk'], df['sixteenP_trim_mean_cpi'], df['sixteenP_trim_mean_cpi_risk'], df['median_cpi'], df['median_cpi_risk'], df['fpcpi'], df['fpcpi_risk'], df['fpcpi_m_fe'], df['fpcpi_m_fe_risk'], df['mean']):
                    prompt = f"On {a}, the Sticky Price CPI was {b}, Sticky Price CPI Risk Score was {c}, Sticky Price CPI Less Shelter was {d}, Sticky Price CPI Less Shelter Risk Score was {e}, Sticky Price CPI Less Food Energy was {f}, Sticky Price CPI Less Food Energy Risk Score was {g}, Sticky Price CPI Less Food Energy Shelter was {h}, Sticky Price CPI Less Food Energy Shelter Risk Score was {i}, Trimmed Mean PCE Inflation Rate was {j}, Trimmed Mean PCE Inflation Rate Risk Score was {k}, 16 Percent Trimmed Mean CPI was {l}, 16 Percent Trimmed Mean CPI Risk Score was {m}, Median CPI was {n}, Median CPI Risk Score was {o}, Flexible Price CPI was {p}, Flexible Price CPI Risk Score was {q}, Flexible Price CPI Less Food Energy was {r}, Flexible Price CPI Less Food Energy Risk Score was {s} /n/n***###***/n/n"
                    # print(prompt)
                    completion = f" {t}"
                    # print(completion)
                    writer.write({'prompt': prompt, 'completion': completion})
                    
        create_jsonl(merged_df, r'training_data.jsonl')

        # Upload file - create function that will result in the file-id
        openai.File.create(
            file=open(r'training_data.jsonl', "rb"),
            purpose='fine-tune'
        )
        
        # List files
        print(openai.File.list())
        
        # Get file-id
        file_id = openai.File.list()['data'][-1]['id']
        print(f" This is the File Name: {openai.File.list()['data'][-1]['id']}")

        # Create fine-tuned model
        result = openai.FineTune.create(training_file=file_id, model="curie:ft-personal:market-risk-analyzer-2023-02-28-07-44-37", n_epochs=16, learning_rate_multiplier=0.0025)  # validation_file=validation_file,  n_epochs=, batch_size=, learning_rate_multiplier=, prompt_loss_weight=, classification_n_classes=10, suffix="market-risk-analyzer", classification_n_classes=4, compute_classification_metrics=True, n_epochs=16
        print(result)
        # print(result['events'][0]['message'])
        # print(result['status'].upper())
        result['status'] = result['status'].upper()
        # Save Results in the database in new table for Retrain Results

        # Create Class Object and send to the dashboard. Save feedback in session
        
        confirmation_statement = f"You did it! {result['events'][0]['message']}, Status: {result['status']}"

        return confirmation_statement
