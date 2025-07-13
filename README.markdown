# Stock Market Predictor

[Click for a try!](https://stock-prediction-by-lstm-ntqvmpntpsyeqvtoqy6uud.streamlit.app)

## Overview
The Stock Market Predictor is a Python application that uses Long Short-Term Memory (LSTM) neural networks to forecast stock prices. It retrieves historical stock data from Yahoo Finance, preprocesses it using StandardScaler and MinMaxScaler, trains two LSTM models, and visualizes predictions through an interactive Streamlit web interface. Users can input a stock symbol (e.g., TSLA) and select data intervals to view historical trends, moving averages, and future price forecasts.

## Features
- **Data Retrieval**: Fetches historical stock data from Yahoo Finance using `yfinance`.
- **Data Preprocessing**: Normalizes data with StandardScaler and MinMaxScaler for LSTM model training.
- **LSTM Models**: Implements two three-layer LSTM models for price prediction, one per scaler.
- **Visualization**: Displays:
  - Historical stock data (from January 1, 2015).
  - 100-day and 200-day moving averages.
  - Historical predictions comparing actual vs. predicted prices.
  - 7-day future price forecasts with confidence intervals.
- **Performance Metrics**: Shows Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) for model evaluation.
- **Interactive Interface**: Streamlit interface for inputting stock symbols and selecting data intervals (1d, 5d, 1wk, 1mo, 3mo).

## Files
- **App.py**: Main Streamlit application script for data retrieval, model predictions, and visualizations.
- **LSTM_Stock_Prediction.ipynb**: Jupyter Notebook detailing data preprocessing, model training, and evaluation.
- **models/LSTM_3_Layers.keras**: Pre-trained LSTM model using StandardScaler.
- **models/LSTM_3_Layers_MINMAX.keras**: Pre-trained LSTM model using MinMaxScaler.
- **requirements.txt**: Lists Python dependencies for deployment.

## Requirements
The following Python libraries are required:
- `yfinance==0.2.65`
- `pandas==2.3.1`
- `numpy==2.1.3`
- `streamlit==1.46.1`
- `scikit-learn==1.7.0`
- `matplotlib==3.10.3`
- `keras==3.10.0`
- `tensorflow==2.19.0`
- Additional dependencies (e.g., `altair`, `protobuf`) listed in `requirements.txt`.

Install dependencies using `pip`:
```bash
pip install -r requirements.txt
```

For Anaconda users, install core dependencies with:
```bash
conda install -c conda-forge yfinance=0.2.65 pandas=2.3.1 numpy=2.1.3 streamlit=1.46.1 scikit-learn=1.7.0 matplotlib=3.10.3 keras=3.10.0 tensorflow=2.19.0
pip install -r requirements.txt
```

## Setup and Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<6309-lcy>/stock-prediction-by-lstm.git
   cd stock-prediction-by-lstm
   ```

2. **Install Dependencies**:
   - Use the `requirements.txt` file to install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - For Anaconda, follow the commands above to install core packages and use `pip` for additional dependencies.


3. **Run the Application Locally**:
   ```bash
   streamlit run App.py
   ```
   This launches the Streamlit app in your default browser.

4. **Deploy to Streamlit Cloud**:
   - Push the repository to GitHub, ensuring `requirements.txt` and the `models` directory are included.
   - In Streamlit Cloud, create a new app, select your repository, and specify `App.py` as the main file.
   - Reboot the app to deploy.

## Usage
1. **Input Stock Symbol**: Enter a valid stock ticker (e.g., TSLA) in the Streamlit interface.
2. **Select Data Interval**: Choose from 1 day, 5 days, 1 week, 1 month, or 3 months.
3. **View Results**:
   - **Stock Data**: Historical data since January 1, 2015.
   - **Performance Metrics**: RMSE and MAE for both LSTM models on the test dataset.
   - **Visualizations**:
     - Closing Price: Plot of historical closing prices.
     - Moving Averages: Includes 100-day and 200-day moving averages.
     - Historical Predictions: Actual vs. predicted prices for the test set.
     - Future Predictions: 7-day forecast with confidence intervals.

## Model Details
- **Data Source**: Yahoo Finance, from January 1, 2015, to the current date.
- **Data Splitting**: 85% training, 15% testing.
- **Preprocessing**:
  - Uses closing price for predictions.
  - 60-day sliding window for LSTM input sequences.
  - Normalizes data with StandardScaler and MinMaxScaler.
- **LSTM Architecture**:
  - Two models (StandardScaler and MinMaxScaler) with three LSTM layers (24 units, 16 units, dense output).
  - Includes BatchNormalization and Dropout (0.2) to prevent overfitting.
- **Training** (in `LSTM_Stock_Prediction.ipynb`):
  - Adam optimizer, Mean Squared Error (MSE) loss.
  - Early stopping with 20-epoch patience.
- **Future Predictions**: 7-day forecast using the last 60 days of data.


## Limitations
- **Data Dependency**: Relies on Yahoo Finance, which may have access restrictions.
- **Prediction Accuracy**: Stock predictions are uncertain and not suitable for sole investment decisions.

## Future Improvements
- Training the model with more features like RSI, MA, High and Low

## Contact
For questions or contributions, open an issue on GitHub or contact the repository maintainer.
