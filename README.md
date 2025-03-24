Patent Trend Prediction

Overview

This project is a Flask-based web application that predicts patent trends using various machine learning models. Users can select a patent area, enter a target year, and choose a prediction model to estimate the number of patents published in that year.

Features

Supports multiple machine learning models:

Linear Regression

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Polynomial Regression

Dynamic visualization of historical and predicted patent trends

Long-term patent prediction up to 30 years ahead

Interactive UI for user input and prediction results

Installation & Setup

Prerequisites

Ensure you have the following installed:

Python 3.x

Flask

Pandas

Matplotlib

Seaborn

Scikit-learn

Installation Steps

Clone the repository:

git clone https://github.com/your-username/patent-trend-prediction.git
cd patent-trend-prediction

Install dependencies:

pip install -r requirements.txt

Run the Flask app:

python app.py

Open a web browser and go to:

http://127.0.0.1:5000/

Usage

Select the Patent Area from the dropdown.

Enter a Year for prediction.

Choose a Machine Learning Model.

Click on the Predict button.

The predicted number of patents and a visualization of the trend will be displayed.

File Structure

📂 patent-trend-prediction
│-- 📄 app.py  # Main Flask application
│-- 📄 requirements.txt  # Python dependencies
│-- 📂 templates/
│   ├── 📄 index.html  # Frontend UI
│-- 📂 static/
│   ├── 📄 styles.css  # CSS styling
│-- 📄 patent.xlsx  # Dataset file

License

This project is licensed under the MIT License.

Contact

For any queries, contact: ktadhithya2006@gmail.com

