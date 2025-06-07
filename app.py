from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
from werkzeug.utils import secure_filename

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

        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)

        try:
            model = SARIMAX(cat_data['y'], order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            forecast = results.forecast(steps=1)
            forecast_results[cat] = f"Rp. {int(forecast[0]):,}".replace(",", ".")
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
