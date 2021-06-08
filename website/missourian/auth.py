import functools
from flask import Blueprint, flash, g, redirect, render_template, request, session, url_for
from flask_login import LoginManager, logout_user, login_user, login_required
from werkzeug.security import check_password_hash, generate_password_hash

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
from website.missourian.tables import Users, Base

bp = Blueprint('auth', __name__, url_prefix='/auth')
login_manager = LoginManager()


@login_manager.user_loader
def load_user(user_id):
    return Users.query.filter_by(id=user_id).first()


@bp.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        elif Users.query.filter_by(username=username).first() is not None:
            error = f"User {username} is already registered."

        if error is None:
            users = Users(username=username, password=generate_password_hash(password))
            Base.session.add(users)
            Base.session.commit()
            return redirect(url_for('auth.login'))

        flash(error)

    return render_template('auth/register.html')


@bp.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        error = None
        user = Users.query.filter_by(username=username).first()

        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user.password, password):
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            login_user(user)
            session['was_once_logged_in'] = True
            session['user_id'] = user.id
            return redirect(url_for('index'))
        if session['was_once_logged_in']:
            error = 'You have been automatically logged out.'
            del session['was_once_logged_in']

        flash(error)

    return render_template('auth/login.html')


@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = Users.query.filter_by(id=user_id).first()


@bp.route('/logout')
@login_required
def logout():
    logout_user()
    # if session['was_once_logged_in']:
    #     del session['was_once_logged_in']
    flash('You have successfully logged out')
    # return render_template('image_scores/index.html')
    return redirect(url_for('index'))


# def login_required(view):
#     @functools.wraps(view)
#     def wrapped_view(**kwargs):
#         if g.user is None:
#             return redirect(url_for('auth.login'))
#
#         return view(**kwargs)
#
#     return wrapped_view
