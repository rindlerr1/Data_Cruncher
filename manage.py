import os
from app import create_app
from app.models import file_path
from flask_script import Manager
#from flask_migrate import Migrate, MigrateCommand

app = create_app('default')


manager = Manager(app)


if __name__=='__main__':
	manager.run()
	





