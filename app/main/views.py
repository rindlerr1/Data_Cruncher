from flask import render_template, flash, redirect, request
from . import main
from .. import db
from bokeh.embed import server_document
from .forms import file_form
from ..models import file_path
from werkzeug import secure_filename
import os
from ..file_paths import PATHS
import pandas as pd
import os

targeting_file = pd.read_csv(PATHS().target_holder)

#TODO create trigger file to read and write to another file to check for the EDA mdoule
#the file should be a single column with a zero
#the EDA module should overwrite the zero value with a 1 
#this will impact how the module figures out what to use for catgorical or numeric values


ALLOWED_EXTENSIONS = set(['csv'])

#various working pages	
@main.route('/')
def index():
	return redirect('/Ingestion')
	
@main.route('/About')
def about():
	return render_template('embed.html')
	
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@main.route('/Ingestion', methods=['GET', 'POST'])
def upload_form():
	return render_template('ingestion_2.html')

@main.route('/Ingested', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join('app/main/Data/app_data.csv'))
			targeting_file.to_csv(PATHS().target_informer, index=False)
			flash('File(s) successfully uploaded')
			return redirect('/File-Metadata')	

	
#form pages
#@main.route('/Ingestion', methods=['GET', 'POST'])
#def ingestion_form():
#	form = file_form()
#	if form.validate_on_submit():
#		file = file_path(path = form.path.data,
#					cat_var = form.cat_var.data,
#					num_var = form.num_var.data,
#					date_var = form.date_var.data)
#		db.session.add(file)
#		db.session.commit()
#		flash("File metadata now ingested!  Please continue.")
#	return render_template("ingestion.html", form=form)

#model documentation
@main.route('/CatBoost', methods=['GET'])
def bkapp_model1():
    return redirect("https://github.com/catboost/catboost")

@main.route('/XGBoost', methods=['GET'])
def bkapp_model2():
	return redirect("https://xgboost.readthedocs.io/en/latest/")

@main.route('/model3', methods=['GET'])
def bkapp_model3():
    script = server_document('http://localhost:5006/sliders')
    return render_template("embed.html", script=script)

@main.route('/Flask', methods=['GET'])
def flask_doc():
	return redirect("http://flask.pocoo.org/")

	
@main.route('/Bokeh', methods=['GET'])
def bokeh_doc():
	return redirect("https://bokeh.pydata.org/en/latest/")


#Dashboards	
@main.route('/File-Metadata', methods=['GET', 'POST'])
def bkapp_meta():
    script = server_document('http://localhost:5006/file_metadata')
    return render_template("embed_eda.html", script=script)
        
@main.route('/EDA', methods=['GET'])
def bkapp_eda():
    script = server_document('http://localhost:5006/eda')
    return render_template("embed_eda.html", script=script)

@main.route('/Model-Selection', methods=['GET'])
def bkapp_models():
    script = server_document('http://localhost:5006/model_selection')
    return render_template("embed_model_selection.html", script=script)
    
@main.route('/Model-Evaluation', methods=['GET'])
def bkapp_scenarios():
    script = server_document('http://localhost:5006/model_evaluation')
    return render_template("embed_model_evaluation.html", script=script)
    
