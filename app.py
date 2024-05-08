import pandas as pd
import numpy as np
import joblib
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the model performance metrics
with open('models/model_performance.pkl', 'rb') as f:
    model_performance = pickle.load(f)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/final_output')
def final_output():
    model_performance_list = model_performance.reset_index().rename(columns={'index': 'Metric'}).to_dict(orient='records')
    return render_template('index.html', model_performance=model_performance_list)

if __name__ == "__main__":
    app.run()
