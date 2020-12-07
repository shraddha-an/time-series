# Stock Price Prediction with LSTM

# Importing libraries
import matplotlib.pyplot as plt, seaborn as sb
import pandas as pd, numpy as np

# Importing the data
df = pd.read_csv('msft.csv')

# Splitting into X & y
dataset_train, dataset_test = df.loc[:220, 'Open'], df.loc[221:, 'Open']

X = dataset_train.iloc[:,].values.reshape(-1, 1)

# Scaling - Normalization
from sklearn.preprocessing import MinMaxScaler
mn = MinMaxScaler()
X = mn.fit_transform(X)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 221):
    X_train.append(X[i-60:i, ])
    y_train.append(X[i, ])

X_train, y_train = np.array(X_train), np.array(y_train)

# Preparing the test set
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)

inputs1 = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs1 = inputs1.reshape(-1,1)
inputs1 = mn.transform(inputs1)

X_test = []
for i in range(60, 90):
    X_test.append(inputs1[i-60:i, ])

X_test = np.array(X_test, dtype = float)

# Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = 10, verbose = 1)


# Getting the predicted stock price
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = mn.inverse_transform(predicted_stock_price)

predicted_stock_price = pd.DataFrame(predicted_stock_price).reset_index(drop = True)
dataset_test = pd.DataFrame(dataset_test).reset_index(drop = True)

# Visualising the results
plt.plot(dataset_test, color = 'red', label = 'Real MSFT Stock Price')#.autoscale(axis = 'x',tight = True)
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted MSFT Stock Price')
plt.title('MSFT Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('MSFT Stock Price')
plt.legend()
plt.show()

# RMSE
accuracy = dict()
from sklearn.metrics import mean_squared_error as mse
from math import sqrt

print('\nRMSE: ', sqrt(mse(dataset_test, predicted_stock_price)))

accuracy['2_50'] = sqrt(mse(dataset_test, predicted_stock_price))
