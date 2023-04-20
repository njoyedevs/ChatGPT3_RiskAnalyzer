from flask import render_template, request, redirect, session, flash

from flask_app.models import users

from flask_app import app, BCRYPT

import re

PASSWORD_REGEX = re.compile(r"(?=^.{8,}$)(?=.*\d)(?=.*[!@#$%^&*]+)(?![.\n])(?=.*[A-Z])(?=.*[a-z]).*$")

# @app.route('/')
# def index():
#     users = User.get_all()
#     print(users)
#     return render_template("index.html", users = users)

@app.route('/')
def index():
    # return redirect('/users')
    return render_template('index.html')  

@app.route('/registration', methods=["POST"])
def create_user():

    
    if not users.User.validate_name(request.form):
        return redirect('/')

    if not users.User.is_email(request.form):
        return redirect('/')
    
    if not users.User.check_email(request.form):
        return redirect('/')
    
    if len(request.form['password']) < 8:
            flash(u"Passwords must be 8 digits!", 'password_error')
            return redirect('/')
    
    if request.form['password'] != request.form['confirm']:
            flash(u"Passwords do not match!", 'password_error')
            return redirect('/')
            
    res = re.search(PASSWORD_REGEX, request.form['password'])
        
    if not res:
        flash("""Passwords must be at least 8 characters and
            contain: 1(A-Z),1(a-z),1(0-9),1(!@#$%^&*)""", 'password_error')
    
    pw_hash = BCRYPT.generate_password_hash(request.form['password'])

    data = {
        "first_name": request.form['first_name'],
        "last_name": request.form['last_name'],
        "email": request.form['email'],
        "password":pw_hash,
    }
    
    user_id = users.User.save_user(data)
    
    session['user_id'] = user_id
    session['first_name'] =  request.form['first_name']
    
    return redirect('/create/prompt')

@app.route('/login', methods=["POST"])
def login():
    
    user = users.User.get_by_email(request.form)
    
    if not user:
        flash('User does not exist', 'login_error')
        return redirect('/')

    if not BCRYPT.check_password_hash(user.password, request.form['password']):
        # if we get False after checking the password
        flash("Invalid Password", 'password_error')
        return redirect('/')
    
    session['user_id'] = user.id
    session['first_name'] = user.first_name
    
    return redirect('/create/prompt')

@app.route('/clear_session')
def clear_session():
    
    # print('test')
    session.clear()
    return redirect('/')

# @app.route('/users')
# def users():
#     return render_template('read_all.html', users=users.User.get_all())
    
# @app.route('/users/new')
# def new():
#     return render_template('create.html')

# @app.route('/users/create', methods=["POST"])
# def create():
#     print(request.form)
#     id = users.User.save(request.form)
#     return redirect(f"/users/{id}")

# @app.route('/users/<int:id>/edit')
# def edit(id):
#     data = {
#         "id":id
#     }
#     return render_template('edit.html', users=users.User.get_one(data))

# @app.route('/users/<int:id>')
# def show(id):
#     data = {
#         "id":id
#     }
#     return render_template('show_user.html', users=users.User.get_one(data))

# @app.route('/users/update', methods=["POST"])
# def update():
#     users.User.update(request.form)
#     return redirect('/users')

# @app.route('/users/<int:id>/destroy/')
# def delete(id):
    
#     data = {
#         'id': id
#     }
    
#     users.User.delete(data)
#     return redirect('/')



