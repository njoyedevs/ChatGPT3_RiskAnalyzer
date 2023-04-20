from flask import render_template, request, redirect, session, flash, url_for
from flask_app.models import user_data, results, econ_data, chatgpt_data
from flask_app import app

@app.route('/create/prompt')
def create_prompt():
    
    if 'user_id' not in session:
        flash('You must be logged in to view this page', 'login_error')
        return redirect('/')
    # Check if the user is authenticated
    
    return render_template('create_prompt.html')

@app.route('/please/wait', methods=['POST'])
def please_wait():
    
    # print(request.form)
    
    user_data_id = user_data.User_Data.save_user_data(request.form)
    session['user_data_id'] = user_data_id
    # print(session['user_data_id'])
    
    risk_analysis_results = results.Result.risk_analysis(session['user_data_id'])
    # print(risk_analysis_results)
    
    result_id = results.Result.save_results_data(risk_analysis_results, user_data_id)
    session['result_id'] = result_id
    # print(session['result_id'])
    
    return redirect('/dashboard')

@app.route('/retrain')
def retrain():
    
    result = chatgpt_data.ChatGPT3.retrain_chatgpt()
    if result:
        print(result)

    confirmation_div = '<div class="my-class">' + \
                (f'<p>{result}</p>' if result else '<p>Error in Processing</p>') + \
                '</div>'
                
    session['confirmation'] = confirmation_div
    print(session['confirmation'])
    
    return redirect('/dashboard')

@app.route('/dashboard')
def dashboard():
    
    if 'user_id' not in session:
        flash('You must be logged in to view this page', 'login_error')
        return redirect(url_for('index'))
    
    chatgpt_data_id = chatgpt_data.ChatGPT3.save_chatgpt_response()
    # print(chatgpt_data_id)
    user_report_obj = results.Result.get_risk_report(chatgpt_data_id)
    # print(user_report_obj)
    gauge_fig = results.Result.create_guage_plotly()
    chart_div = econ_data.Econ_Data.create_chart_plotly()
    
    return render_template('dashboard.html', chart_div=chart_div, gauge_fig=gauge_fig, user=user_report_obj) 

# @app.route('/table')
# def table():
    
#     if 'user_id' not in session:
#         flash('You must be logged in to view this page', 'login_error')
#         return redirect('/')
    
#     return render_template('table.html', econ_data = econ_data.Econ_Data.get_all_econ_data())  