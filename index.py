import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math

# Load the data
df = pd.read_csv("nabil.csv")

# Convert the Unix timestamp to a human-readable date
df['date'] = pd.to_datetime(df['t'], unit='s')

# Create a previous close feature
df['prev_close'] = df['c'].shift()
df.dropna(inplace=True)

# Prepare features and target
x = df[['o', 'prev_close']]
y = df['c']

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Generate data for only the next day
last_row = df.iloc[-1]
next_day = pd.date_range(start=last_row['date'], periods=2, freq='D')[1]

# Create a dataframe for the next day prediction
next_day_df = pd.DataFrame(index=[next_day])
next_day_df['o'] = last_row['o']
next_day_df['prev_close'] = last_row['c']
next_day_df['h'] = last_row['h']
next_day_df['l'] = last_row['l']
next_day_df['v'] = last_row['v']

# Predict the next day's closing price
next_day_prediction = regressor.predict(next_day_df[['o', 'prev_close']])[0]

# Print the prediction
print(f"Predicted closing price for {next_day.date()}: {next_day_prediction:.2f}")
