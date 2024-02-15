import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('CJD_VTI.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

close_prices = df[['Close']].values

scaler = MinMaxScaler(feature_range=(0,1))
close_prices_scaled = scaler.fit_transform(close_prices)

def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data)-sequence_length-1):
        x = data[i:(i+sequence_length)]
        y = data[i+sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 60 
X, y = create_sequences(close_prices_scaled, sequence_length)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Splitting the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Building the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(60, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

# Making predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Invert normalization

# You can now compare these predictions with the actual prices to evaluate the model

import matplotlib.pyplot as plt

# Assuming y_test is also scaled using the same scaler
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.plot(actual_prices, color='red', label='Actual Prices')
plt.plot(predictions, color='blue', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
