from flask import Flask,request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        try:
           return render_template('home.html')
        except Exception as e:
            return render_template('index.html', error_message=str(e))
    else:
        pass