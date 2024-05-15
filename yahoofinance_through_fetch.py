#!/usr/bin/env python
# coding: utf-8

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import pickle
import plotly.graph_objects as go

# Function to fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

# Function to preprocess data for LSTM
def preprocess_data(data, time_step):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data)-time_step-1):
        X.append(scaled_data[i:(i+time_step), 0])
        y.append(scaled_data[i + time_step:i + time_step + 2, 0])  # Include both opening and closing prices
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

# Function to create and train LSTM model
def train_lstm_model(X_train, y_train, epochs=20, batch_size=64):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(2))  # Predicting both opening and closing prices
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Function to make predictions for the next day
def make_predictions(model, data, scaler, time_step):
    last_time_steps = data[-time_step:].values  # Convert Series to NumPy array
    last_time_steps = last_time_steps.reshape(-1, 1)
    last_time_steps = scaler.transform(last_time_steps)
    last_time_steps = last_time_steps.reshape(1, -1, 1)
    prediction = model.predict(last_time_steps)
    prediction = scaler.inverse_transform(prediction)
    return prediction

# Function to make predictions for multiple days
def make_multiple_predictions(model, data, scaler, time_step, num_days):
    last_time_steps = data[-time_step:].values
    last_time_steps = last_time_steps.reshape(-1, 1)
    last_time_steps = scaler.transform(last_time_steps)
    predictions = []
    for _ in range(num_days):
        last_time_steps_reshaped = last_time_steps.reshape(1, -1, 1)
        prediction = model.predict(last_time_steps_reshaped)
        predictions.append(prediction)
        # Update last_time_steps for next prediction
        last_time_steps = np.append(last_time_steps[1:], prediction[-1])  # Replace the oldest value with the latest prediction
    predictions = np.array(predictions)
    # Reshape predictions to match the expected format for inverse transformation
    predictions = predictions.reshape(predictions.shape[0], predictions.shape[2])
    predictions = scaler.inverse_transform(predictions)
    return predictions


# Fetch historical stock data
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=365*10)).strftime('%Y-%m-%d')  # 10 years of data
stock_data = fetch_stock_data('GC=F', start_date, end_date)

# Print start and end dates
print("Start date:", start_date)
print("End date:", end_date)

# Preprocess data and create dataset for LSTM
time_step = 60
X, y, scaler = preprocess_data(stock_data['Close'], time_step)

# Split data into train and test sets
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size,:], X[train_size:len(X),:]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Train LSTM model
model = train_lstm_model(X_train, y_train)

# Make predictions for the previous days
train_predict = scaler.inverse_transform(model.predict(X_train))
test_predict = scaler.inverse_transform(model.predict(X_test))

# Print today's actual closing price
today_actual_closing_price = stock_data['Close'].iloc[-1]
print("\nToday's actual closing price:", today_actual_closing_price)

# Print today's predicted closing price
today_predicted_closing_price = test_predict[-1][1]  # Predicted closing price for today
print("Today's predicted closing price:", today_predicted_closing_price)

# Make predictions for the next day's opening and closing prices
next_day_prediction = make_predictions(model, stock_data['Close'], scaler, time_step)
print("Predicted opening and closing prices for next day:", next_day_prediction)

# Make predictions for the next five days
num_days_to_predict = 5
next_five_days_predictions = make_multiple_predictions(model, stock_data['Close'], scaler, time_step, num_days_to_predict)
print("\nPredicted opening and closing prices for the next five days:")
for i in range(num_days_to_predict):
    print("Day", i+1, ":", next_five_days_predictions[i])

# Serialize data into a dictionary
data_to_serialize = {
    'model': model,  # Serialize the trained LSTM model
    'scaler': scaler,  # Serialize the MinMaxScaler object
    'today_actual_closing_price': today_actual_closing_price,
    'today_predicted_closing_price': today_predicted_closing_price,
    'next_day_prediction': next_day_prediction,
    'next_five_days_predictions': next_five_days_predictions
}

# Define the filename for the pickle file
pickle_filename = 'stock_prediction_model.pkl'

# Serialize the data into a pickle file
with open(pickle_filename, 'wb') as f:
    pickle.dump(data_to_serialize, f)

print("\nSerialized data saved to", pickle_filename)

# Load the pickle file
with open(pickle_filename, 'rb') as file:
    loaded_data = pickle.load(file)

# Extract loaded objects
loaded_model = loaded_data['model']
loaded_scaler = loaded_data['scaler']
loaded_today_actual_closing_price = loaded_data['today_actual_closing_price']
loaded_today_predicted_closing_price = loaded_data['today_predicted_closing_price']
loaded_next_day_prediction = loaded_data['next_day_prediction']
loaded_next_five_days_predictions = loaded_data['next_five_days_predictions']

# Verify loaded data
print("\nLoaded model:", loaded_model)
print("Loaded scaler:", loaded_scaler)
print("Loaded today's actual closing price:", loaded_today_actual_closing_price)
print("Loaded today's predicted closing price:", loaded_today_predicted_closing_price)
print("Loaded next day prediction:", loaded_next_day_prediction)
print("Loaded next five days predictions:", loaded_next_five_days_predictions)

## Create candlestick trace for actual data
candlestick = go.Candlestick(x=stock_data.index,
                             open=stock_data['Open'],
                             high=stock_data['High'],
                             low=stock_data['Low'],
                             close=stock_data['Close'],
                             name='Actual Prices')

# Create trace for predicted closing prices
predicted_trace = go.Scatter(x=stock_data.index[-len(test_predict):],
                              y=test_predict[:, 1],
                              mode='lines',
                              name='Predicted Closing Prices',
                              line=dict(color='blue', width=2))  # Change color to blue and increase width for visibility

# Create trace for predicted opening price
predicted_opening_trace = go.Scatter(x=stock_data.index[-len(test_predict):],
                                      y=test_predict[:, 0],
                                      mode='lines',
                                      name='Predicted Opening Price',
                                      line=dict(color='green', width=2, dash='dash'))  # Use green color and dash style

# Create trace for predicted next five days opening and closing prices
next_five_days = pd.date_range(start=stock_data.index[-1] + timedelta(days=1), periods=num_days_to_predict, freq='D')
next_five_days_opening = next_five_days_predictions[:, 0]
next_five_days_closing = next_five_days_predictions[:, 1]

next_five_days_opening_trace = go.Scatter(x=next_five_days,
                                          y=next_five_days_opening,
                                          mode='lines',
                                          name='Predicted Next Five Days Opening Prices',
                                          line=dict(color='orange', width=2, dash='dot'))

next_five_days_closing_trace = go.Scatter(x=next_five_days,
                                          y=next_five_days_closing,
                                          mode='lines',
                                          name='Predicted Next Five Days Closing Prices',
                                          line=dict(color='red', width=2, dash='dot'))

# Create figure and add traces
fig = go.Figure(data=[candlestick, predicted_trace, predicted_opening_trace, next_five_days_opening_trace, next_five_days_closing_trace])

# Update layout
fig.update_layout(title='Actual vs Predicted Closing Prices',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  hovermode='x unified')  # Display hover info for all traces at the same x-coordinate

# Show the plot
fig.show()
