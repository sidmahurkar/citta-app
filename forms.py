from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length

class QueryForm(FlaskForm):
	query = TextAreaField('Query', validators = [DataRequired(), Length(min=3)])
	submit = SubmitField('Submit Query')

