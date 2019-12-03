# Flask Imports
from flask import Flask, render_template, url_for, flash, redirect
from forms import QueryForm

# Framework Imports
import nltk
import pandas as pd
import numpy as np

from nltk.cluster import KMeansClusterer
  

from sklearn import cluster
from sklearn import metrics

from utils import *


app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'



@app.route("/")
@app.route("/home")
def home():
	return render_template('home.html')


# @app.route("/inputform",methods=['GET','POST'])
# def inputform():
# 	form = QueryForm()
# 	if form.validate_on_submit():
# 		flash('Query Submitted!','success')
# 		return redirect(url_for('home'))
# 	return render_template('inputform.html', form = form)


@app.route("/inputform",methods=['GET','POST'])
def inputform():
	form = QueryForm()
	if form.validate_on_submit():

		query = form.query.data
		df = pd.read_pickle('cmu_data_phrase.pkl')

		#Get top-n entries by score ranking
		df['score'] = df.apply(lambda row: get_score(query, row.noun_phrase, row.sentiment, alpha=0.95), axis = 1) 
		df_topn = df.nlargest(500, 'score')
		df_topn.reset_index(inplace=True)


		recommendations = get_recommendation(query, df, df_topn, n=500, NUM_CLUSTERS=10)


		# Can uncomment if we want to query submission feedback
		# flash('Query Submitted!','success')
		return render_template('index.html', recommendations = recommendations)
	return render_template('inputform.html', form = form)




if __name__ == '__main__':
	app.run(debug=True)