from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os
from prophet import Prophet
from werkzeug.utils import secure_filename
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def run_forecast(csv_path):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    # Monthly aggregation by category
    monthly_cat = df.groupby([df['date'].dt.to_period('M'), 'category'])['amount'].sum().reset_index()
    monthly_cat['date'] = monthly_cat['date'].dt.to_timestamp()
    monthly_cat = monthly_cat.rename(columns={'date': 'ds', 'amount': 'y'})

    categories = monthly_cat['category'].unique()
    forecast_results = {}

    for cat in categories:
        cat_data = monthly_cat[monthly_cat['category'] == cat].copy()
        cat_data.set_index('ds', inplace=True)
        n_months = len(cat_data)
        if n_months < 12:
            try:
                X = np.arange(len(cat_data)).reshape(-1, 1)  
                y = cat_data['y'].values
                model = LinearRegression().fit(X, y)
                forecast = model.predict([[len(cat_data)]])[0]
                forecast_results[cat] = f"Rp. {int(forecast):,}".replace(",", ".")
            except Exception as e:
                forecast_results[cat] = f"Error: {str(e)}"
        else:
            forecasts = []
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 12)

            try:
                model = SARIMAX(cat_data['y'], order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit(disp=False)
                forecast = results.forecast(steps=1)
                forecasts.append(float(forecast.values[0]) if hasattr(forecast, 'values') else float(forecast))
                
                
                try:
                    if n_months >= 24:
                        ets_model = ExponentialSmoothing(cat_data['y'],
                                                       seasonal='add',
                                                       seasonal_periods=12)
                    else:
                        ets_model = ExponentialSmoothing(cat_data['y'],
                                                       trend='add',
                                                       seasonal=None)
                    forecasts.append(ets_model.fit().forecast(1)[0])
                except Exception as e:
                    print(f"ETS skipped for {cat}: {str(e)}")
                    
                    
                #prophet
                try:
                    # Prepare Prophet dataframe correctly
                    prophet_df = cat_data[['y']].reset_index()
                    prophet_df.columns = ['ds', 'y']  # Ensure only these two columns
                    
                    prophet_model = Prophet(seasonality_mode='additive')
                    prophet_model.fit(prophet_df)
                    future = prophet_model.make_future_dataframe(periods=1, freq='M')
                    prophet_forecast = prophet_model.predict(future)
                    forecasts.append(prophet_forecast['yhat'].iloc[-1])
                except Exception as e:
                    print(f"Prophet skipped for {cat}: {str(e)}")
                
                # Average successful forecasts
                if forecasts:
                    forecast_results[cat] = f"Rp. {int(np.mean(forecasts)):,}".replace(",", ".")
                else:
                    forecast_results[cat] = "Error: All models failed"
            except Exception as e:
                forecast_results[cat] = f"Error: {str(e)}"

    return forecast_results

@app.route('/', methods=['GET', 'POST'])
def index():
    forecasts = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            forecasts = run_forecast(filepath)
    return render_template('index.html', forecasts=forecasts)

if __name__ == '__main__':
    app.run(debug=True)
