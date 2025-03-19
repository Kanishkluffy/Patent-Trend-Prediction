import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent threading issues

from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_year = None
    results = []

    try:
        df = pd.read_excel('patent.xlsx')
        df = df.dropna(subset=["Publication Date"])  # Remove NaN values
        df["Publication Date"] = pd.to_numeric(df["Publication Date"], errors='coerce')
        df = df.dropna(subset=["Publication Date"])  # Drop rows where conversion failed
        areas = df['Area'].unique()
    except Exception as e:
        return f"Error loading data: {e}"

    if request.method == 'POST':
        try:
            prediction_year = int(request.form['year'])
        except ValueError:
            return "Invalid year entered! Please enter a valid year.", 400

        for area in areas:
            temp = df[df["Area"] == area]
            y = temp["Publication Date"].value_counts()

            if len(y) <= 1:
                results.append({
                    'area': area,
                    'plot': None,
                    'predicted_lr': "Not enough data",
                    'predicted_svm': "Not enough data",
                    'predicted_knn': "Not enough data"
                })
                continue

            years = np.array(y.index).reshape(-1, 1)
            counts = np.array(y.values)

            try:
                # Linear Regression
                slope, intercept, _, _, _ = stats.linregress(years.flatten(), counts)
                predicted_lr = slope * prediction_year + intercept
                
                # SVM Model
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                years_scaled = scaler_x.fit_transform(years)
                counts_scaled = scaler_y.fit_transform(counts.reshape(-1, 1)).flatten()
                svm_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
                svm_model.fit(years_scaled, counts_scaled)
                prediction_year_scaled = scaler_x.transform([[prediction_year]])
                predicted_svm_scaled = svm_model.predict(prediction_year_scaled)[0]
                predicted_svm = scaler_y.inverse_transform([[predicted_svm_scaled]])[0][0]
                svm_predicted_values_scaled = svm_model.predict(years_scaled)
                svm_predicted_values = scaler_y.inverse_transform(svm_predicted_values_scaled.reshape(-1, 1)).flatten()

                # KNN Model
                knn_model = KNeighborsRegressor(n_neighbors=min(3, len(years)))
                knn_model.fit(years, counts)
                predicted_knn = knn_model.predict([[prediction_year]])[0]
                knn_predicted_values = knn_model.predict(years)
            except Exception as e:
                return f"Model error: {e}"

            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.scatter(years, counts, label="Actual Data", color="black")

            # Linear Regression Line
            extended_years = np.append(years.flatten(), prediction_year)
            lr_predictions = slope * extended_years + intercept
            plt.plot(extended_years, lr_predictions, 'r-', label="Linear Regression")

            # SVM Regression Line
            plt.plot(years.flatten(), svm_predicted_values, 'b--', label="SVM Regression")

            # KNN Regression Line
            plt.plot(years.flatten(), knn_predicted_values, 'g-.', label="KNN Regression")

            # Predictions
            plt.scatter(prediction_year, predicted_lr, color='red', marker='o', label="LR Prediction")
            plt.scatter(prediction_year, predicted_svm, color='blue', marker='x', label="SVM Prediction")
            plt.scatter(prediction_year, predicted_knn, color='green', marker='s', label="KNN Prediction")

            plt.axvline(x=prediction_year, color='gray', linestyle='--', label=f"Prediction for {prediction_year}")
            plt.title(f'Predictions for {area}')
            plt.xlabel('Year')
            plt.ylabel('Frequency')
            plt.legend()

            # Save Plot as Image
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            img_base64 = base64.b64encode(img.getvalue()).decode()
            plt.close()

            results.append({
                'area': area,
                'plot': img_base64,
                'predicted_lr': round(predicted_lr, 2),
                'predicted_svm': round(float(predicted_svm), 2),
                'predicted_knn': round(predicted_knn, 2)
            })

    return render_template('index.html', results=results, year=prediction_year)

if __name__ == '__main__':
    app.run(debug=True)
