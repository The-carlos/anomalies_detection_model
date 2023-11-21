# Anomalies Detection Model

## Overview

The Anomalies Detection Model is a neural network designed to predict the future Total Payment Volume (TPV) behavior of a group of merchants at two-hour intervals. The script leverages these predictions to estimate a minimum and maximum TPV for each time point. If the actual TPV falls outside this range, a visual alarm is triggered.

## Technologies Used

The project is implemented using a variety of technologies, including:

- **Pandas:** Data manipulation and analysis.
- **LSTM TensorFlow Neural Networks:** Building and training the predictive model.
- **SQL:** Managing and querying databases.
- **AWS Athena:** Serverless query service for data analysis.
- **Matplotlib:** Creating visualizations for analysis.

## Final Product

The culmination of this project is a Power BI dashboard that automatically refreshes every two hours. The dashboard provides a comprehensive view of the predicted and actual TPV, highlighting anomalies that require attention.

## Usage

To run the model and generate predictions, follow these steps:

1. **Data Preparation:**
   - Ensure your dataset is formatted correctly.
   - Preprocess the data using the provided preprocessing scripts.

2. **Model Training:**
   - Execute the training script to train the LSTM TensorFlow Neural Network.

3. **Prediction and Alarm:**
   - Run the prediction script to obtain future TPV predictions.
   - The script will raise a visual alarm if the actual TPV is outside the predicted range.

4. **Power BI Dashboard:**
   - Access the Power BI dashboard for a real-time visualization of TPV anomalies.

## Dependencies

Make sure you have the following dependencies installed:

```bash
pip install pandas tensorflow sqlAlchemy matplotlib
