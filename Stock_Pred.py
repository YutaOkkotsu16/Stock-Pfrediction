# This program predicts stock prices by using machine learning models

# Install dependencies

import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Get the stock data
data = quandl.get("WIKI/AMZN")
# print(data.head())

# Get the Adjusted Close Price
df = data[['Adj. Close']]
# print(df.head())

# Create a variable to predict 'n' days out into the future
forecast_out = 30

# Create another column (the target) shifted 'n' units up
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

# Create the independent data set (x)
# Convert the dataframe to a numpy array
x = np.array(df.drop(['Prediction'], axis=1))
# Remove the 'n' rows
x = x[:-forecast_out]

# Create the dependent data set (y)
# Convert data frame to a numpy array
y = np.array(df['Prediction'])
# Get all of the y values except the last 'n' rows
y = y[:-forecast_out]


# Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create and train the Support Vector Regression (SVR) model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

# Testing model: Score returns the coefficient of determination R^2 of the prediction
svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

# Create and train a Linear Regression Model
lr = LinearRegression()
# Train the model
lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

# Set x_forecast equal to the last 30 rows of the original data frame
x_forecast = np.array(df.drop(['Prediction'], axis=1))[-forecast_out:]


# Print the linear regression predictions for the next 'n' days
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

# Print the svm predictions for the next 'n' days
svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)

import matplotlib.pyplot as plt

# Initialize lists to store confidence scores
lr_confidences = []
svm_confidences = []

# Run the models 100 times and store the confidence scores
for _ in range(100):
    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    # Train and score the Linear Regression model
    lr.fit(x_train, y_train)
    lr_confidences.append(lr.score(x_test, y_test))
    
    # Train and score the SVR model
    svr_rbf.fit(x_train, y_train)
    svm_confidences.append(svr_rbf.score(x_test, y_test))

# Plot the confidence scores
plt.figure(figsize=(10, 5))
plt.plot(lr_confidences, label='Linear Regression Confidence')
plt.plot(svm_confidences, label='SVR Confidence')
plt.xlabel('Iteration')
plt.ylabel('Confidence Score')
plt.title('Comparison of Linear Regression and SVR Model Confidence Scores over 100 Iterations')
plt.legend()
plt.show()
