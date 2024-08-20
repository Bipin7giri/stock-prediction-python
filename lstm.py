import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Setting the style for matplotlib
plt.style.use('fivethirtyeight')

# Load the CSV data
df = pd.read_csv("data.csv")
df['date'] = pd.to_datetime(df['t'], unit='s')

# Set the 'date' column as the index
df.set_index('date', inplace=True)

# Create a previous close feature
df['prev_close'] = df['c'].shift()
df.dropna(inplace=True)

# Visualize the closing price history
# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['c'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price in NRS', fontsize=18)
# plt.show()


# creating dataframe with only 'C'

data = df.filter(['c'])
dataset = data.values
training_data_len = math.ceil(len(dataset)*.8)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)



#create the training data set
#create the scaled training price
train_data = scaled_data[0:training_data_len,:] 
#split the data into x_train, y_train dataset
x_train = []
y_train = []

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    # if i<=60:
    #     print(x_train)
    #     print(y_train)
    #     print()
        

# converting x_train and y_train to numpy array
x_train,y_train = np.array(x_train),np.array(y_train)

#reshape the data

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)


#Build LSTM model

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False,))
model.add(Dense(25))
model.add(Dense(1))


#compile model

model.compile(optimizer="adam",loss="mean_squared_error")


#Train the model
model.fit(x_train,y_train,batch_size=1,epochs=1)


#Create a testing dataset

#create a new array containing scaled values
test_data  = scaled_data[training_data_len-60:,:]

# create a datasets x_test and y_test
x_test = []
y_test = dataset[training_data_len:,:]


for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
    
    
#convert to numpy array

x_test = np.array(x_test)


#reshape the data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
print(x_test)

#get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


#get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions-y_test)**2)

#Plot the data

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#visualize data 
plt.figure(figsize=(16,8))
plt.title('Modal')
plt.xlabel('Date',fontsize=18)
plt.xlabel('Close Price NRP',fontsize=18)
plt.plot(train['c'])
plt.plot(valid[['c','Predictions']])
plt.legend(['Train','Val','Predictions'],loc="lower right")
plt.show()


#show the valid and predicted price
# valid

#Get the quote

adbl_quote = pd.read_csv("data.csv")
adbl_quote['date'] = pd.to_datetime(df['t'], unit='s')

new_df = adbl_quote.filter(['c'])
#get the last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#scale the data to be values between 0 and 1

last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
#append the past 60 days
X_test.append(last_60_days_scaled)
#Convert the X_data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#Get the predicted scaled price
pred_price = model.predict(X_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)   



