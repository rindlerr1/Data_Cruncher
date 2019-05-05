from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import Required
from ..models import file_path

class file_form(FlaskForm):
	path = StringField('File Path', validators = [Required()])
	cat_var = StringField('List of comma seperated Categorical Column Headers', validators = [Required()])
	num_var = StringField('List of comma seperated Numeric Column Headers', validators = [Required()])
	date_var = StringField('Primary Date Column Header', validators = [Required()])
	sumbit = SubmitField('Submit')