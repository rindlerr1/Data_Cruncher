from app import db

#database classes
class file_path(db.Model):
	__tablename__ = 'file_metadata'
	id = db.Column(db.Integer, primary_key=True)
	path = db.Column(db.String(300))
	cat_var = db.Column(db.String(300))
	num_var = db.Column(db.String(300))
	date_var = db.Column(db.String(30))
	
#database classes
class variables(db.Model):
	__tablename__ = 'model_variables'
	id = db.Column(db.Integer, primary_key=True)
	ind_var = db.Column(db.String(3000))
	dep_var = db.Column(db.String(3000))
	model_sel = db.Column(db.String(30))
	params_sel = db.Column(db.String(3000))
	