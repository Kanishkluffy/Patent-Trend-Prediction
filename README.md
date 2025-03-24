# Patent Trend Prediction

## Overview
This project is a Flask-based web application that predicts patent trends using various machine learning models. Users can select a patent area, enter a target year, and choose a prediction model to estimate the number of patents published in that year.

## Features
- Supports multiple machine learning models:
  - Linear Regression
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Polynomial Regression
- Dynamic visualization of historical and predicted patent trends
- Long-term patent prediction up to 30 years ahead
- Interactive UI for user input and prediction results

## Technologies Used
- Python (Flask, Pandas, Matplotlib, Seaborn)
- Machine Learning Models (Linear Regression, SVM, KNN, Polynomial Regression)
- Frontend (HTML, Bootstrap)

## Installation & Setup
### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/patent-trend-prediction.git
   cd patent-trend-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Open a web browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage
1. Select the **Patent Area** from the dropdown.
2. Enter a **Year** for prediction.
3. Choose a **Machine Learning Model**.
4. Click on the **Predict** button.
5. The predicted number of patents and a visualization of the trend will be displayed.

## File Structure
```
📂 patent-trend-prediction
│-- 📄 app.py  # Main Flask application
│-- 📄 requirements.txt  # Python dependencies
│-- 📂 templates/
│   ├── 📄 index.html  # Frontend UI
│-- 📂 static/
│   ├── 📄 styles.css  # CSS styling
│-- 📄 patent.xlsx  # Dataset file
```



## Contact
For any queries, contact: ktadhithya2006@gmail.com

