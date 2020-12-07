# Different models for forecasting Microsoft Stock Prices

# Importing libraries
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sb

# Importing dataset
# Univariate data
df = pd.read_csv('msft.csv')[['Open']].rename(columns = {'Open': 'value'})

# Multivariate data
ds = pd.read_csv('msft.csv').drop(['Date'], axis = 1)

# Splitting ds into train & test subsets
n_obs = 500
ds_train, ds_test = ds[0: -n_obs], ds[-n_obs:]

df_train, df_test = df[0: -n_obs], df[-n_obs:]

# Regression Accuracy Metrics
accuracy = {}

# Evaluating Model Performance with metrics
def metrics(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE

    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)** 2)** 0.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr

    mins = np.amin(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax

    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse,
            'corr':corr, 'minmax':minmax})

# ================================ ARIMA Model ==============================
# ARIMA Model with (p, d, q) = (2, 1, 2)

# Plot of train & testing values
plt.figure(figsize = (12, 7))
plt.title('Microsoft Prices')
plt.xlabel('Dates')
plt.ylabel('Prices')

plt.plot(df_train['value'], '#303F9F', label = 'Training Data')
plt.plot(df_test['value'], '#FF6F00', label = 'Testing Data')

plt.xticks(np.arange(0, 7982, 1300))
plt.legend(loc = 'upper center', shadow = True, facecolor = '#F2F4F4')
plt.show()

# Choosing 'd' with ACF Plot
# Plotting an Autocorrelation Function plot to figure out the optimal
# order of differencing required to make the TS stationary
# d is chosen when the ACF plot stabilizes to 0. || d = 1
from statsmodels.graphics.tsaplots import plot_acf

fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax = axes[0, 1])

axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

#  Choosing 'p' with PACF Plot
# Partial Autocorrelation Plot to figure out order of the p term.
# PACF is plotted after differencing the TS once as d = 1. || p = 1
from statsmodels.graphics.tsaplots import plot_pacf

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff().dropna()); axes[0].set_title('1st Differencing')
axes[1].set(ylim = (0,5))
plot_pacf(df.value.diff().dropna(), ax=axes[1])

plt.show()

# Choosing 'q' with ACF Plot
# ACF plot of 1st differenced TS as d = 1.
# Number of lags above the significance limit (blue region) give the q. || q = 1

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df.value.diff().dropna(), ax=axes[1])

plt.show()

# 2,1,2 ARIMA Model
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(df_train.value, order = (2, 1, 2))
model_fit = model.fit(disp = 0)
print(model_fit.summary())

# Plotting residuals to ensure there is no pattern left behind
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title = "Residuals", ax = ax[0])
residuals.plot(kind = 'kde', title = 'Density', ax = ax[1])
plt.show()

# Out of Sample Forecasts.
# Returns Forecasts, Standard Error & 2D Array of Confidence Interval for forecast
fc, std_error, conf = model_fit.forecast(len(df_test), alpha = 0.05)  # 95% conf

# Make as pandas series
forecast = pd.Series(fc, index = df_test.index)
lower_series = pd.Series(conf[:, 0], index = df_test.index)
upper_series = pd.Series(conf[:, 1], index = df_test.index)

# Actual vs Forecasted Plot
plt.figure(figsize = (12,5), dpi = 100)

plt.plot(df_train, label = 'training')
plt.plot(df_test, label = 'actual')
plt.plot(forecast, label = 'forecast')

plt.fill_between(lower_series.index, lower_series, upper_series,
                 color = 'k', alpha = 0.15)
plt.title('Forecast vs Actual Values')
plt.legend(loc = 'upper center', fontsize = 12, shadow = True, facecolor = '#F2F4F4')
plt.show()

# Performance Metric
accuracy['ARIMA'] = forecast_accuracy(forecast, df_test.value)

# ~~~~~~~~~~~~~~~~~~~~~ Auto ARIMA Forecast ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Building Auto ARIMA model
import pmdarima as pm
model = pm.auto_arima(df_train.value, start_p = 1, start_q = 1,
                      test = 'adf',       # use adftest to find optimal 'd'
                      max_p = 3, max_q = 3, # maximum p and q
                      m = 1,              # frequency of series
                      d = None,           # let model determine 'd'
                      seasonal = False,   # No Seasonality
                      start_P = 0,
                      D = 0,
                      trace = True,
                      error_action = 'ignore',
                      suppress_warnings = True,
                      stepwise = True)

print(model.summary())

# Residual Error Plots
model.plot_diagnostics(figsize = (7, 5))
plt.show()

# Forecast
fc, confint = model.predict(n_periods = n_obs, return_conf_int = True)
index_of_fc = np.arange(len(df.value), len(df.value) + n_obs)

# make series for plotting purpose
fc_series = pd.Series(fc, index = index_of_fc)
lower_series = pd.Series(confint[:, 0], index = index_of_fc)
upper_series = pd.Series(confint[:, 1], index = index_of_fc)

# Actual vs Forecast Plot
plt.plot(df.value)
plt.plot(fc_series, color = 'darkgreen')
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color = 'k', alpha = 0.15)

plt.title("Final Forecast of MSFT Stock Prices")
plt.show()

# Performance
accuracy['Auto ARIMA'] = forecast_accuracy(fc, df_test.value)

# ========================== Normal regression ================================
# Linear Rgression Model
X_train, y_train = ds_train.iloc[:, 1:].values, ds_train.iloc[:,0].values
X_test, y_test = ds_test.iloc[:, 1:].values, ds_test.iloc[:,0].values

# Fitting model on training data
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# metrics
forecast_accuracy(y_pred, y_test)

# Actual vs Predicted Plot
sb.set_style('darkgrid')
plt.plot(y_pred, color = '#FF33CC', label = 'Forecasts')
plt.plot(y_test, color = '#0000CC', label = 'Actual Values')

plt.xlabel('Time')
plt.ylabel('Stock Prices ($)')
plt.legend(shadow = True, loc = 'upper center')
plt.title("Forecast of MSFT Stock Prices")
plt.show()

# Performance Metric
accuracy['Linear Regression'] = forecast_accuracy(y_pred, y_test)

# ===========================  VAR  =======================================
# Vector Autoregression Model

# Plot each TS to look at trends
sb.set_style('darkgrid')
plt.figure(figsize = (10, 6))

fig, axes = plt.subplots(nrows  = int(len(ds.columns)) // 2, ncols = 2, dpi = 120)
for i, ax in enumerate(axes.flatten()):
    data = ds[ds.columns[i]]
    ax.plot(data, color = '#0288D1', linewidth = 1)
    ax.set_title(ds.columns[i])
    plt.tick_params(labelsize = 6)

    ax.set_title(ds.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)

plt.suptitle('Time-Series Variables')
plt.tight_layout()

# Granger's Causality Test to check if TS are causing each other
from statsmodels.tsa.stattools import grangercausalitytests

maxlag = 15
test = 'ssr_chi2test'

def grangers_causation_matrix(ds, var, test = 'ssr_chi2test', verbose = False):
    """
    Apply Granger's Causality test to all variables in the dataset to check if each
    of the time-series are causing each other.

    Null-Hypothesis: Coefficients of past values in the VAR equation are 0, i.e,
    Each T.S does not cause the other.
    """
    df = pd.DataFrame(np.zeros(((len(var)), len(var))), columns = var, index = var)

    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(ds[[r, c]], maxlag = maxlag, verbose = True)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]

            if verbose:
                print(f'Y = {r}, X = {c}, P Values = {p_values}')
            df.loc[r, c] = np.min(p_values)

    return df

grangers_causation_matrix(ds, var = ds.columns, verbose = False)

# Augmented Dickey-Fuller Test (ADF Test) to check for stationarity
from statsmodels.tsa.stattools import adfuller

def adf_test(ds):
    dftest = adfuller(ds, autolag='AIC')
    adf = pd.Series(dftest[0:4], index = ['Test Statistic','p-value','# Lags','# Observations'])

    for key, value in dftest[4].items():
       adf['Critical Value (%s)'%key] = value
    print (adf)

    p = adf['p-value']
    if p <= 0.05:
        print("\nSeries is Stationary")
    else:
        print("\nSeries is Non-Stationary")


for i in ds_train.columns:
    print("Column: ",i)
    print('--------------------------------------')
    adf_test(ds_train[i])
    print('\n')
    
# Differencing all variables to get rid of Stationarity
ds_differenced = ds_train.diff().dropna()

# Running the ADF test once again on the 1st differenced series to test for stationarity
for i in ds_differenced.columns:
    print("Column: ",i)
    print('--------------------------------------')
    adf_test(ds_differenced[i])
    print('\n')


# Checking the right order
from statsmodels.tsa.api import VAR

model = VAR(df_diff)
for i in [1,2,3,4,5,6,7,8,9, 11, 12, 13, 14,15]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

x = model.select_order(maxlags = 20)
print(x.summary())

# Fitting the VAR model of order 1 to the training data
model_fitted = model.fit(8)
model_fitted.summary()

# Durbin Watson's Statistic to check for correlations in residual errors
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(ds.columns, out):
    print((col), ':', round(val, 2))

# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = df_diff.values[-lag_order:]

# Forecast
fc = model_fitted.forecast(y = forecast_input, steps = n_obs)
forecast_var = pd.DataFrame(fc, index = ds.index[-n_obs:], columns = ds.columns)

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)].cumsum()
    return df_fc

results = invert_transformation(ds_train, forecast_var, second_diff = False)

var_result = np.split(results, [6], axis = 1)[1]

# Actual vs Predicted Plot
fig, axes = plt.subplots(nrows = int(len(ds.columns)//2), ncols = 2, dpi = 150, figsize = (8,8))
for i, (col,ax) in enumerate(zip(ds.columns, axes.flatten())):

    results[col+'_forecast'].plot(legend = True, ax = ax).autoscale(axis = 'x',tight = True)
    ds_test[col][-n_obs:].plot(legend = True, ax = ax)
    ax.legend(loc = 'upper center', prop = {'size': 6})

    ax.set_title(col + ": Forecast vs Actuals")

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize = 6)

plt.tight_layout()

# Performance Metric
accuracy['VAR'] = forecast_accuracy(var_result['Open_forecast'], df_test.value)

# ================================== AR Model ==============================
from statsmodels.tsa.ar_model import AR

ar_model = AR(df_train.value)
arm_fitted = ar_model.fit()

arm_fitted.k_ar

ar_preds = pd.DataFrame(arm_fitted.predict(start = len(df_train), end = len(df_train) + len(df_test) - 1))

# Performance Metric
accuracy['AR'] = forecast_accuracy(ar_preds.iloc[:, 0], df_test.value)

# Actual vs Predicted Plot
ar_plot = pd.concat([ar_preds, df_test], axis = 1, ignore_index = True)
ar_plot.rename(columns = {0: 'Forecasted', 1: 'Actual'}, inplace = True)
ar_plot.plot()
