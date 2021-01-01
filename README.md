This project contains 30 minutes CO, NO, PM10, and PM2.5 species forecast of Central London.

For forecasting CO, NO, PM10, and PM2.5 for the next 30 minutes run the script forecast_aqi.py . There are two models trained on 30 minutes data trend for the period 01-Jan-2019 to 25-Dec-2020.
1. Stacked LSTM model
2. CNN LSTM model
Both trained models are stored in 'models' directory with names stack_lstm.h5 and cnn_lstm.h5 respectively.
As per performance testing, CNN-LSTM is more accurate in the forecast that the Stacked LSTM model, and the default model for the forecast is cnn_lstm.h5
