import os
import sys

from flask import Flask
sys.path.append(os.path.split(sys.path[0])[0])
from old_codes.config_files import db_config


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI=db_config.DB_STR,
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    from . import tables
    from . import auth
    tables.Base.init_app(app)
    auth.login_manager.init_app(app)

    app.register_blueprint(auth.bp)

    from . import image_scores
    app.register_blueprint(image_scores.bp)
    app.add_url_rule('/', endpoint='index')

    return app

