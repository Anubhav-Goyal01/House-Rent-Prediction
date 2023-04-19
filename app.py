import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction import CustomData, PredictionPipeline

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data=CustomData(
            city = request.form.get('city'),
            furnishing_status = request.form.get('furnishing_status'),
            tenant_preferred = request.form.get('tenants_preferred'),
            area_type = request.form.get('area_type'),
            bhk = int(request.form.get('bhk')),
            size = int(request.form.get('size')),
            posted_on = (request.form.get('date')),
            bathrooms = int(request.form.get('bathrooms')),
            floor_level = int(request.form.get('total_floors')),
            total_floors = int(request.form.get('floor_level'))
        )
        pred_df = data.get_data_as_data_frame()

        prediction_pipeline = PredictionPipeline()
        results = prediction_pipeline.predict(pred_df)
        result_string = f"Predicted rent of the house is: ₹{round(results[0], 2)}"
        return render_template('index.html',results= result_string)


if __name__ == "__main__":
    app.run(debug = True)