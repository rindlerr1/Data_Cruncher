import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
	SECRET_KEY = '123456789'
	SQL_ALCHEMY_COMMIT_ON_TEARDOWN = True
	FLASKY_MAIL_SUBJECT_PREFIX = '[Flasky]'
	FLASKY_MAIL_SENDER = 'Flasky Admin <icomapp2019@gmail.com>'
	FLASKY_ADMIN = os.environ.get('FLASKY_ADMIN')
	MAIL_SERVER = 'smtp.googlemail.com'
	MAIL_PORT = 587
	MAIL_USE_TLS = True
	MAIL_USERNAME = 'icomapp2019@gmail.com'
	MAIL_PASSWORD = 'Mindshare123'
	@staticmethod
	def init_app(app):
		pass
		
class DevelopmentConfig(Config):
	DEBUG = True
	SQLALCHEMY_DATABASE_URI = 'mysql://root:Izzie722@127.0.0.1/icom_Demo'
	
class TestingConfig(Config):
	TESTING = True
	SQLALCHEMY_DATATBASE_URI = 'mysql://root:Izzie722@127.0.0.1/icom_Demo'
	
class ProductionConfig(Config):
	SQLALCHEMY_DATABASE_URI = 'mysql://root:Izzie722@127.0.0.1/icom_Demo'
	
config = {'development' : DevelopmentConfig,
		'testing': TestingConfig,
		'production': ProductionConfig,
		'default': DevelopmentConfig}
		
		