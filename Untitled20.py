#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd

# Download GBP/USD historical data from Yahoo Finance
ticker = "GBPUSD=X"
start_date = "1999-01-01"
end_date = "2024-07-06"

# Fetch data
df = yf.download(ticker, start=start_date, end=end_date)

# Display the first few rows of the dataset
print(df.head())



# In[2]:


pip install --upgrade bottleneck


# In[3]:


# Check for missing values
print(df.isnull().sum())

# Fill or drop missing values
df.dropna(inplace=True)

# Calculate daily returns
df['Returns'] = df['Adj Close'].pct_change()

# Display summary statistics
print(df.describe())


# In[4]:


# Simple Moving Averages
df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()

# Relative Strength Index (RSI)
def compute_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = compute_rsi(df['Adj Close'], 14)

# Display the first few rows with new features
print(df.head())


# In[5]:


# Define the target variable and feature set
X = df[['Adj Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI']].dropna()
y = df['Returns'].dropna().apply(lambda x: 1 if x > 0 else 0)

# Ensure the target and features are aligned
y = y[X.index]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Check the shape of the split data
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[6]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)


# In[7]:


# Add predictions to the test set
X_test['Predicted_Signal'] = y_pred
X_test['Returns'] = df['Returns'].loc[X_test.index]

# Calculate strategy returns
X_test['Strategy_Returns'] = X_test['Returns'] * X_test['Predicted_Signal']

# Calculate cumulative returns
X_test['Cumulative_Returns'] = (X_test['Strategy_Returns'] + 1).cumprod()

# Plot the cumulative returns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(X_test['Cumulative_Returns'], label='Strategy Returns')
plt.plot((df['Returns'].loc[X_test.index] + 1).cumprod(), label='Market Returns')
plt.legend()
plt.show()


# In[8]:


import numpy as np

# Total return
total_return = X_test['Cumulative_Returns'].iloc[-1]

# Sharpe ratio
sharpe_ratio = X_test['Strategy_Returns'].mean() / X_test['Strategy_Returns'].std() * np.sqrt(252)

# Max drawdown
rolling_max = X_test['Cumulative_Returns'].cummax()
drawdown = X_test['Cumulative_Returns'] / rolling_max - 1
max_drawdown = drawdown.min()

print("Total Return:", total_return)
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)


# In[9]:


import yfinance as yf
import pandas as pd

# Download GBP/USD historical data from Yahoo Finance
ticker = "GBPUSD=X"
start_date = "1999-01-01"
end_date = "2023-07-06"

# Fetch data
df = yf.download(ticker, start=start_date, end=end_date)

# Drop missing values
df.dropna(inplace=True)

# Calculate daily returns
df['Returns'] = df['Adj Close'].pct_change()

# Feature Engineering
df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()

# Relative Strength Index (RSI)
def compute_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = compute_rsi(df['Adj Close'], 14)

# Define the target variable and feature set
X = df[['Adj Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI']].dropna()
y = df['Returns'].dropna().apply(lambda x: 1 if x > 0 else 0)

# Ensure the target and features are aligned
y = y.loc[X.index]
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Check the shape of the split data
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
# Add predictions to the test set
X_test['Predicted_Signal'] = y_pred
X_test['Returns'] = df['Returns'].loc[X_test.index]

# Calculate strategy returns
X_test['Strategy_Returns'] = X_test['Returns'] * X_test['Predicted_Signal']

# Calculate cumulative returns
X_test['Cumulative_Strategy_Returns'] = (X_test['Strategy_Returns'] + 1).cumprod()
X_test['Cumulative_Market_Returns'] = (X_test['Returns'] + 1).cumprod()

# Plot the cumulative returns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(X_test.index, X_test['Cumulative_Strategy_Returns'], label='Strategy Returns')
plt.plot(X_test.index, X_test['Cumulative_Market_Returns'], label='Market Returns')
plt.legend()
plt.show()
import numpy as np

# Total return
total_return = X_test['Cumulative_Strategy_Returns'].iloc[-1] - 1

# Sharpe ratio
sharpe_ratio = X_test['Strategy_Returns'].mean() / X_test['Strategy_Returns'].std() * np.sqrt(252)

# Max drawdown
rolling_max = X_test['Cumulative_Strategy_Returns'].cummax()
drawdown = X_test['Cumulative_Strategy_Returns'] / rolling_max - 1
max_drawdown = drawdown.min()

print("Total Return:", total_return)
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)


# In[10]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Download USD/JPY historical data from Yahoo Finance
ticker = "USDJPY=X"
start_date = "1999-01-01"
end_date = "2024-07-06"

# Fetch data
df = yf.download(ticker, start=start_date, end=end_date)

# Drop missing values
df.dropna(inplace=True)

# Calculate daily returns
df['Returns'] = df['Adj Close'].pct_change()

# Feature Engineering
df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()

# Relative Strength Index (RSI) function
def compute_rsi(data, window):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Calculate RSI
df['RSI'] = compute_rsi(df['Adj Close'], 14)

# Define the target variable and feature set
X = df[['Adj Close', 'Volume', 'SMA_50', 'SMA_200', 'RSI']].dropna()
y = df['Returns'].dropna().apply(lambda x: 1 if x > 0 else 0)

# Ensure the target and features are aligned
y = y.loc[X.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Add predictions to the test set
X_test['Predicted_Signal'] = y_pred
X_test['Returns'] = df['Returns'].loc[X_test.index]

# Calculate strategy returns
X_test['Strategy_Returns'] = X_test['Returns'] * X_test['Predicted_Signal']

# Calculate cumulative returns
X_test['Cumulative_Strategy_Returns'] = (X_test['Strategy_Returns'] + 1).cumprod()
X_test['Cumulative_Market_Returns'] = (X_test['Returns'] + 1).cumprod()

# Plot the cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(X_test.index, X_test['Cumulative_Strategy_Returns'], label='Strategy Returns')
plt.plot(X_test.index, X_test['Cumulative_Market_Returns'], label='Market Returns')
plt.title('Cumulative Returns - USD/JPY')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Performance metrics
total_return = X_test['Cumulative_Strategy_Returns'].iloc[-1] - 1
sharpe_ratio = X_test['Strategy_Returns'].mean() / X_test['Strategy_Returns'].std() * np.sqrt(252)
rolling_max = X_test['Cumulative_Strategy_Returns'].cummax()
drawdown = X_test['Cumulative_Strategy_Returns'] / rolling_max - 1
max_drawdown = drawdown.min()

print("Total Return:", total_return)
print("Sharpe Ratio:", sharpe_ratio)
print("Max Drawdown:", max_drawdown)


# In[12]:


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Download USD/JPY historical data from Yahoo Finance
ticker = "USDJPY=X"
start_date = "1999-01-01"
end_date = "2024-06-30"

# Fetch data
df = yf.download(ticker, start=start_date, end=end_date)

# Drop missing values
df.dropna(inplace=True)

# Compute Ichimoku Cloud indicators
def compute_ichimoku(data, n9=26, n26=52):
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Tenkan-sen (Conversion Line)
    nine_period_high = high.rolling(window=n9).max()
    nine_period_low = low.rolling(window=n9).min()
    data['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    # Kijun-sen (Base Line)
    period_high = high.rolling(window=n26).max()
    period_low = low.rolling(window=n26).min()
    data['kijun_sen'] = (period_high + period_low) / 2

    # Senkou Span A (Leading Span A)
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(n26)

    # Senkou Span B (Leading Span B)
    period_high = high.rolling(window=n26 * 2).max()
    period_low = low.rolling(window=n26 * 2).min()
    data['senkou_span_b'] = ((period_high + period_low) / 2).shift(n26 * 2)

    return data

# Apply Ichimoku Cloud calculation
df = compute_ichimoku(df)

# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Drop rows with NaN values introduced by the calculations
df.dropna(inplace=True)

# Define the target variable and feature set
X = df[['Close', 'Volume', 'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b']]
y = df['Returns']

# Ensure the target and features are aligned
X = X.loc[y.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R^2 Score:", r2)
print("Mean Squared Error:", mse)

# Visualize predicted vs actual returns
plt.figure(figsize=(10, 6))
plt.plot(X_test.index, y_test, label='Actual Returns', color='blue')
plt.plot(X_test.index, y_pred, label='Predicted Returns', color='orange')
plt.title('Actual vs Predicted Returns - USD/JPY')
plt.xlabel('Date')
plt.ylabel('Daily Returns')
plt.legend()
plt.show()

# Correlation of volume with price drift
correlation = X_test['Volume'].corr(pd.Series(y_pred))
print("Correlation between Volume and Predicted Returns:", correlation)


# In[13]:


import yfinance as yf
import pandas as pd
import numpy as np

# Download USD/JPY historical data from Yahoo Finance
ticker = "USDJPY=X"
start_date = "2000-01-01"
end_date = "2024-07-06"

# Fetch data
df = yf.download(ticker, start=start_date, end=end_date)

# Drop missing values
df.dropna(inplace=True)

# Compute Ichimoku Cloud indicators
def compute_ichimoku(data, n9=26, n26=52):
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Tenkan-sen (Conversion Line)
    nine_period_high = high.rolling(window=n9).max()
    nine_period_low = low.rolling(window=n9).min()
    data['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    # Kijun-sen (Base Line)
    period_high = high.rolling(window=n26).max()
    period_low = low.rolling(window=n26).min()
    data['kijun_sen'] = (period_high + period_low) / 2

    # Senkou Span A (Leading Span A)
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(n26)

    # Senkou Span B (Leading Span B)
    period_high = high.rolling(window=n26 * 2).max()
    period_low = low.rolling(window=n26 * 2).min()
    data['senkou_span_b'] = ((period_high + period_low) / 2).shift(n26 * 2)

    return data

# Apply Ichimoku Cloud calculation
df = compute_ichimoku(df)

# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Drop rows with NaN values introduced by the calculations
df.dropna(inplace=True)

# Create a column for the prediction signal
df['Signal'] = 0  # Initialize with 0, to be updated based on model predictions

# Define a function to generate trading signals based on predictions
def generate_signals(data):
    # Assuming you have already trained and have a model for prediction
    # Replace the following with your actual prediction logic
    data['Signal'] = np.where(data['Predicted_Returns'] > 0, 1, -1)  # Example logic

# Function to simulate trading strategy
def simulate_strategy(data):
    capital = 100000  # Starting capital in USD
    position = 0  # Initially no position
    initial_price = None
    trades = []
    
    for index, row in data.iterrows():
        if row['Signal'] == 1 and position == 0:  # Buy signal
            position = capital / row['Close']
            capital = 0
            initial_price = row['Close']
            trades.append(('BUY', index, row['Close']))
        elif row['Signal'] == -1 and position > 0:  # Sell signal
            capital = position * row['Close']
            position = 0
            trades.append(('SELL', index, row['Close']))
    
    # Closing any open position at the end
    if position > 0:
        capital = position * data['Close'].iloc[-1]
        trades.append(('SELL', data.index[-1], data['Close'].iloc[-1]))
    
    # Calculate portfolio performance
    final_value = capital
    initial_value = 100000
    total_return = (final_value - initial_value) / initial_value * 100
    
    return total_return, trades

# Example prediction and strategy simulation (Replace with actual prediction logic)
df['Predicted_Returns'] = np.random.randn(len(df))  # Replace with your model predictions

# Generate signals based on predictions
generate_signals(df)

# Simulate trading strategy
total_return, trades = simulate_strategy(df)

# Print total return and sample trades
print("Total Return:", total_return)
print("Sample Trades:")
for trade in trades[:5]:  # Print first 5 trades
    print(trade)


# In[14]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download USD/JPY historical data from Yahoo Finance
ticker = "USDJPY=X"
start_date = "2000-01-01"
end_date = "2024-07-06"

# Fetch data
df = yf.download(ticker, start=start_date, end=end_date)

# Drop missing values
df.dropna(inplace=True)

# Compute Ichimoku Cloud indicators
def compute_ichimoku(data, n9=26, n26=52):
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Tenkan-sen (Conversion Line)
    nine_period_high = high.rolling(window=n9).max()
    nine_period_low = low.rolling(window=n9).min()
    data['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    # Kijun-sen (Base Line)
    period_high = high.rolling(window=n26).max()
    period_low = low.rolling(window=n26).min()
    data['kijun_sen'] = (period_high + period_low) / 2

    # Senkou Span A (Leading Span A)
    data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(n26)

    # Senkou Span B (Leading Span B)
    period_high = high.rolling(window=n26 * 2).max()
    period_low = low.rolling(window=n26 * 2).min()
    data['senkou_span_b'] = ((period_high + period_low) / 2).shift(n26 * 2)

    return data

# Apply Ichimoku Cloud calculation
df = compute_ichimoku(df)

# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Drop rows with NaN values introduced by the calculations
df.dropna(inplace=True)

# Define a function to generate trading signals based on Ichimoku Cloud
def generate_signals(data):
    data['Signal'] = 0
    data['Signal'] = np.where(data['Close'] > data['tenkan_sen'], 1, 0)  # Buy signal: Close > Tenkan-sen
    data['Signal'] = np.where(data['Close'] < data['tenkan_sen'], -1, data['Signal'])  # Sell signal: Close < Tenkan-sen

# Generate signals based on Ichimoku Cloud
generate_signals(df)

# Function to simulate scalping strategy
def simulate_scalping_strategy(data):
    capital = 100  # Starting capital in USD
    position = 0  # Initially no position
    trades = []

    for index, row in data.iterrows():
        if row['Signal'] == 1 and position == 0:  # Buy signal
            position = capital / row['Close']
            capital = 0
            trades.append(('BUY', index, row['Close']))
        elif row['Signal'] == -1 and position > 0:  # Sell signal
            capital = position * row['Close']
            position = 0
            trades.append(('SELL', index, row['Close']))

    # Closing any open position at the end
    if position > 0:
        capital = position * data['Close'].iloc[-1]
        trades.append(('SELL', data.index[-1], data['Close'].iloc[-1]))

    # Calculate portfolio performance
    final_value = capital
    initial_value = 100000
    total_return = (final_value - initial_value) / initial_value * 100

    return total_return, trades

# Simulate scalping strategy
total_return, trades = simulate_scalping_strategy(df)

# Print total return and sample trades
print("Total Return:", total_return)
print("Sample Trades:")
for trade in trades[:5]:  # Print first 5 trades
    print(trade)

# Plotting the strategy
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price', alpha=0.7)
plt.plot(df['tenkan_sen'], label='Tenkan-sen', linestyle='--')
plt.plot(df[df['Signal'] == 1].index, df['Close'][df['Signal'] == 1], '^', markersize=10, color='g', label='Buy Signal')
plt.plot(df[df['Signal'] == -1].index, df['Close'][df['Signal'] == -1], 'v', markersize=10, color='r', label='Sell Signal')
plt.title('USD/JPY Scalping Strategy with Ichimoku Cloud')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[15]:


# Function to perform backtesting
def backtest_strategy(data):
    data['Position'] = 0
    data['Position'] = np.where(data['Signal'] == 1, 1, 0)  # Long position
    data['Position'] = np.where(data['Signal'] == -1, -1, data['Position'])  # Short position

    data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']

    # Calculate cumulative returns
    data['Cumulative_Strategy_Returns'] = (data['Strategy_Returns'] + 1).cumprod()

    # Plot cumulative returns
    plt.figure(figsize=(14, 7))
    plt.plot(data['Cumulative_Strategy_Returns'], label='Strategy Returns', color='b')
    plt.plot(data['Close'] / data['Close'].iloc[0], label='USD/JPY', color='gray', linestyle='--')
    plt.title('Scalping Strategy Backtest')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

    # Print final cumulative return
    final_return = data['Cumulative_Strategy_Returns'].iloc[-1] - 1
    print("Final Cumulative Return:", final_return * 100, "%")

# Perform backtesting
backtest_strategy(df)


# In[16]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download USD/JPY historical data from Yahoo Finance
ticker = "USDJPY=X"
start_date = "2000-01-01"
end_date = "2023-07-06"

# Fetch data
df = yf.download(ticker, start=start_date, end=end_date)

# Drop missing values
df.dropna(inplace=True)

# Compute Ichimoku Cloud indicators
def compute_ichimoku(data, n9=9, n26=26):
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Tenkan-sen (Conversion Line)
    nine_period_high = high.rolling(window=n9).max()
    nine_period_low = low.rolling(window=n9).min()
    data['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    # Kijun-sen (Base Line)
    period_high = high.rolling(window=n26).max()
    period_low = low.rolling(window=n26).min()
    data['kijun_sen'] = (period_high + period_low) / 2

    return data

# Apply Ichimoku Cloud calculation
df = compute_ichimoku(df)

# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Drop rows with NaN values introduced by the calculations
df.dropna(inplace=True)

# Define a function to generate trading signals based on Ichimoku Cloud
def generate_signals(data):
    data['Signal'] = 0
    data['Signal'] = np.where((data['Close'] > data['tenkan_sen']) & (data['Close'] > data['kijun_sen']), 1, 0)  # Buy signal
    data['Signal'] = np.where(data['Close'] < data['tenkan_sen'], -1, data['Signal'])  # Sell signal

# Generate signals based on Ichimoku Cloud
generate_signals(df)

# Function to simulate scalping strategy
def simulate_scalping_strategy(data):
    capital = 100000  # Starting capital in USD
    position = 0  # Initially no position
    trades = []

    for index, row in data.iterrows():
        if row['Signal'] == 1 and position == 0:  # Buy signal
            position = capital / row['Close']
            capital = 0
            trades.append(('BUY', index, row['Close']))
        elif row['Signal'] == -1 and position > 0:  # Sell signal
            capital = position * row['Close']
            position = 0
            trades.append(('SELL', index, row['Close']))

    # Closing any open position at the end
    if position > 0:
        capital = position * data['Close'].iloc[-1]
        trades.append(('SELL', data.index[-1], data['Close'].iloc[-1]))

    # Calculate portfolio performance
    final_value = capital
    initial_value = 100000
    total_return = (final_value - initial_value) / initial_value * 100

    return total_return, trades

# Simulate scalping strategy
total_return, trades = simulate_scalping_strategy(df)

# Print total return and sample trades
print("Total Return:", total_return)
print("Sample Trades:")
for trade in trades[:5]:  # Print first 5 trades
    print(trade)

# Plotting the strategy
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price', alpha=0.7)
plt.plot(df['tenkan_sen'], label='Tenkan-sen', linestyle='--')
plt.plot(df['kijun_sen'], label='Kijun-sen', linestyle='--')
plt.plot(df[df['Signal'] == 1].index, df['Close'][df['Signal'] == 1], '^', markersize=10, color='g', label='Buy Signal')
plt.plot(df[df['Signal'] == -1].index, df['Close'][df['Signal'] == -1], 'v', markersize=10, color='r', label='Sell Signal')
plt.title('USD/JPY Scalping Strategy with Ichimoku Cloud (Refined)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[17]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download USD/JPY historical data from Yahoo Finance
ticker = "USDJPY=X"
start_date = "2000-01-01"
end_date = "2024-07-06"

# Fetch data
df = yf.download(ticker, start=start_date, end=end_date)

# Drop missing values
df.dropna(inplace=True)

# Compute Ichimoku Cloud indicators
def compute_ichimoku(data, n9=9, n26=26):
    high = data['High']
    low = data['Low']
    close = data['Close']

    # Tenkan-sen (Conversion Line)
    nine_period_high = high.rolling(window=n9).max()
    nine_period_low = low.rolling(window=n9).min()
    data['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    # Kijun-sen (Base Line)
    period_high = high.rolling(window=n26).max()
    period_low = low.rolling(window=n26).min()
    data['kijun_sen'] = (period_high + period_low) / 2

    return data

# Apply Ichimoku Cloud calculation
df = compute_ichimoku(df)

# Calculate daily returns
df['Returns'] = df['Close'].pct_change()

# Drop rows with NaN values introduced by the calculations
df.dropna(inplace=True)

# Assume hypothetical interest rates for USD and JPY
usd_interest_rate = 0.02  # Example USD interest rate (annualized)
jpy_interest_rate = -0.01  # Example JPY interest rate (annualized)

# Define a function to calculate interest rate differential
def calculate_interest_rate_differential(data):
    # Assume constant interest rates for this example
    data['Interest_Rate_Differential'] = usd_interest_rate - jpy_interest_rate

# Function to generate trading signals based on Ichimoku Cloud and interest rate differential
def generate_signals(data):
    data['Signal'] = 0
    data['Signal'] = np.where((data['Close'] > data['tenkan_sen']) & (data['Close'] > data['kijun_sen']) & (data['Interest_Rate_Differential'] > 0), 1, 0)  # Buy signal
    data['Signal'] = np.where((data['Close'] < data['tenkan_sen']) | (data['Interest_Rate_Differential'] <= 0), -1, data['Signal'])  # Sell signal

# Apply interest rate differential calculation
calculate_interest_rate_differential(df)

# Generate signals based on Ichimoku Cloud and interest rate differential
generate_signals(df)

# Function to simulate scalping strategy with interest rate differential
def simulate_scalping_strategy(data):
    capital = 100000  # Starting capital in USD
    position = 0  # Initially no position
    trades = []

    for index, row in data.iterrows():
        if row['Signal'] == 1 and position == 0:  # Buy signal
            position = capital / row['Close']
            capital = 0
            trades.append(('BUY', index, row['Close']))
        elif row['Signal'] == -1 and position > 0:  # Sell signal
            capital = position * row['Close']
            position = 0
            trades.append(('SELL', index, row['Close']))

    # Closing any open position at the end
    if position > 0:
        capital = position * data['Close'].iloc[-1]
        trades.append(('SELL', data.index[-1], data['Close'].iloc[-1]))

    # Calculate portfolio performance
    final_value = capital
    initial_value = 100000
    total_return = (final_value - initial_value) / initial_value * 100

    return total_return, trades

# Simulate scalping strategy with interest rate differential
total_return, trades = simulate_scalping_strategy(df)

# Print total return and sample trades
print("Total Return:", total_return)
print("Sample Trades:")
for trade in trades[:5]:  # Print first 5 trades
    print(trade)

# Plotting the strategy
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price', alpha=0.7)
plt.plot(df['tenkan_sen'], label='Tenkan-sen', linestyle='--')
plt.plot(df['kijun_sen'], label='Kijun-sen', linestyle='--')
plt.plot(df[df['Signal'] == 1].index, df['Close'][df['Signal'] == 1], '^', markersize=10, color='g', label='Buy Signal')
plt.plot(df[df['Signal'] == -1].index, df['Close'][df['Signal'] == -1], 'v', markersize=10, color='r', label='Sell Signal')
plt.title('USD/JPY Scalping Strategy with Ichimoku Cloud and Interest Rate Differential')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[1]:


echo "# GBP-USD-API-ALGORITHM-" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/worker8223/GBP-USD-API-ALGORITHM-.git
git push -u origin main


# In[ ]:




