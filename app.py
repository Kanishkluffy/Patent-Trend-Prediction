from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import io
import base64

app = Flask(__name__)

# Load data
file_path = "patent.xlsx"
df = pd.read_excel(file_path)

@app.route('/')
def index():
    areas = df['Area'].unique()
    return render_template('index.html', areas=areas)

@app.route('/predict', methods=['POST'])
def predict():
    area = request.form['area']
    year = int(request.form['year'])
    
    # Filter data for the selected area
    temp = df.loc[df["Area"] == area]
    y = temp["Publication Year"].value_counts().sort_index()
    years = y.index.to_list()
    counts = y.to_list()
    
    if len(years) < 2:
        return jsonify({'error': 'Not enough data points for regression (need at least 2)'})
    
    # Perform linear regression
    slope, intercept, r, p, std_err = stats.linregress(years, counts)
    
    def predict_func(x):
        return slope * x + intercept
    
    prediction = predict_func(year)
    
    # Generate model values for plotting
    model_years = np.array(years)
    model_values = predict_func(model_years)
    
    # Generate future predictions (10 years beyond last data point)
    future_years = np.arange(min(years), max(years) + 11)
    future_predictions = predict_func(future_years)
    
    # Calculate MSE
    mse = np.mean((np.array(counts) - model_values) ** 2)
    
    # Plot graph
    plt.figure(figsize=(12, 6))
    
    # Plot actual data
    plt.scatter(years,counts,color='blue',label='Actual Data',zorder=3)
    
    # Plot regression line
    plt.plot(model_years,model_values,color='red',label='Linear Regression Fit',linewidth=2)
    
    # Plot future projections
    plt.plot(future_years,future_predictions,'purple',linestyle='dashed',label='Future Projection',linewidth=2)
    
    # Highlight the prediction point
    plt.scatter([year],[prediction],color='green',s=100,zorder=4,label=f'Prediction for {year}: {round(prediction, 1)}')
    
    # Add zero-count markers if any years are missing
    all_years = range(min(years), max(years)+1)
    missing_years = [y for y in all_years if y not in years]
    if missing_years:
        plt.scatter(missing_years,[0]*len(missing_years),color='blue',marker='o',facecolors='none',edgecolors='blue',s=60,zorder=2,label='Zero Patents')
    
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Number of Patents", fontsize=12)
    plt.title(f"Patent Trend for {area}\nMSE: {mse:.2f}, R-squared: {r**2:.2f}", fontsize=14)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Convert plot to image
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return jsonify({
        'prediction': round(prediction, 2), 
        'mse': round(mse, 2),
        'r_squared': round(r**2, 4),
        'plot_url': plot_url,
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(debug=True)