from flask import Flask
from flask_bootstrap import Bootstrap
from config import config
from flask_sqlalchemy import SQLAlchemy

bootstrap = Bootstrap()
db = SQLAlchemy()


def create_app(config_name):
	app = Flask(__name__)
	app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
	app.config.from_object(config[config_name])
	config[config_name].init_app(app)
	
	bootstrap.init_app(app)
	db.init_app(app)
	
	from app.main import main as main_blueprint
	app.register_blueprint(main_blueprint)
	
	
	return app
	
	