import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import tensorflow as tf 

# Downloading stock data
stock_symbol = 'AAPL'
stock_data = yf.download(stock_symbol, start='2000-01-01')

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# Creating training data
train_data = scaled_data[:-30]
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
history = model.fit(x_train, y_train, batch_size=1, epochs=7)

# Prepare=ing test data
test_data = scaled_data[-90:]  # Last 90 days for making predictions for 30 days ahead

x_test = []
for i in range(60,90 ): 
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Making predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Getting the actual prices
actual_prices = stock_data['Close'][-30:].values

# Calculating future dates
future_dates_month = [stock_data.index[-30] + timedelta(days=i) for i in range(30)]
future_dates_week = [stock_data.index[-30] + timedelta(days=i) for i in range(7)]
future_dates_fortnight = [stock_data.index[-30] + timedelta(days=i) for i in range(14)]

# Plotting predictions vs. actual prices for 1 week
plt.figure(figsize=(10, 6))
plt.plot(future_dates_week, predictions[:7], label='Predicted Prices (1 week)', linestyle='dotted', marker='o', color='blue')
plt.plot(future_dates_week, actual_prices[:7], label='Actual Prices', linestyle='solid', marker='x', color='green')
plt.title('Predicted vs. Actual Prices for 1 Week')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('predicted_vs_actual_prices_1_week.jpg')
plt.show()

# Plotting predictions vs. actual prices for 1 fortnight
plt.figure(figsize=(10, 6))
plt.plot(future_dates_fortnight, predictions[:14], label='Predicted Prices (1 fortnight)', linestyle='dashed', marker='o', color='blue')
plt.plot(future_dates_fortnight, actual_prices[:14], label='Actual Prices', linestyle='solid', marker='x', color='green')
plt.title('Predicted vs. Actual Prices for 1 Fortnight')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('predicted_vs_actual_prices_1_fortnight.jpg')
plt.show()

# Plotting predictions vs. actual prices for 1 month
plt.figure(figsize=(10, 6))
plt.plot(future_dates_month, predictions, label='Predicted Prices (1 month)', linestyle='solid', marker='o', color='blue')
plt.plot(future_dates_month, actual_prices, label='Actual Prices', linestyle='solid', marker='x', color='green')
plt.title('Predicted vs. Actual Prices for 1 Month')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('predicted_vs_actual_prices_1_month.jpg')
plt.show()

# Plotting loss function
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Loss.jpg')
plt.show()


