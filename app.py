from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import io
import base64

app = Flask(__name__)

# Load data
file_path = "patent.xlsx"
df = pd.read_excel(file_path)

# Group data by 'Area' and 'Publication Year'
df_grouped = df.groupby(["Area", "Publication Year"]).size().reset_index(name="Count")

@app.route('/')
def index():
    areas = df['Area'].unique()
    return render_template('index.html', areas=areas)

@app.route('/predict', methods=['POST'])
def predict():
    area = request.form['area']
    year = int(request.form['year'])
    model_type = request.form['model']  # Select model type
    
    # Filter data for the selected area
    df_area = df_grouped[df_grouped['Area'] == area]
    
    # Train models
    X = df_area[['Publication Year']].values.reshape(-1, 1)  # Ensure correct shape
    y = df_area['Count']
    
    models = {
        "linear_regression": LinearRegression(),
        "svm": SVR(kernel='rbf'),
        "knn": KNeighborsRegressor(n_neighbors=3),
        "polynomial_regression": make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    }
    
    if model_type not in models:
        return jsonify({'error': 'Invalid model selection'})
    
    model = models[model_type]
    model.fit(X, y)
    prediction = model.predict(np.array([[year]]).reshape(-1, 1))[0]
    
    # Evaluate model
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    # Generate future years for long-term prediction
    future_years = np.array(range(min(df_area['Publication Year']), max(df_area['Publication Year']) + 30)).reshape(-1, 1)
    future_predictions = model.predict(future_years)
    
    # Plot graph
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df_area['Publication Year'], y=df_area['Count'], label='Actual Data', color='blue')
    sns.lineplot(x=df_area['Publication Year'], y=y_pred, label=f'{model_type} Prediction', color='red')
    sns.lineplot(x=future_years.flatten(), y=future_predictions, label='Future Prediction', color='purple', linestyle='dashed')
    plt.scatter(year, prediction, color='green', label=f'Prediction for {year}')
    plt.xlabel("Year")
    plt.ylabel("Number of Patents")
    plt.title(f"Patent Trend for {area} using {model_type}\nMSE: {mse:.2f}")
    plt.legend()
    
    # Convert plot to image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close()  # Close the figure to free memory
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return jsonify({'prediction': round(prediction, 2), 'mse': round(mse, 2), 'plot_url': plot_url})

if __name__ == '__main__':
    app.run(debug=True)