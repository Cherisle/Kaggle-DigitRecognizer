#Machine Learning Code beginnings House Price Practice

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
####Read in Data####
melbourne_file_path = ('C:/Users/Nicholas Trigueros/Desktop/Python/DigitRecognizer/melb_data.csv')
#Create DataFrame
melbourne_data = pd.read_csv(melbourne_file_path)
#Drop all null values
melbourne_data.dropna(axis=0)
#set prediction target which is the price of the houses
y = melbourne_data.Price;
#features to include
melbourne_features = ['Rooms','Bathroom','Landsize','Lattitude','Longtitude'];
X = melbourne_data[melbourne_features];
#Print DataFrames for Inspection
print('DataFrame for Data Set melbourne_data')
print(X.describe())
#Shows First 5 Lines of Data
DF_1 = X.head()
####Model Definition###
#Define Model with random state
melbourne_model = DecisionTreeRegressor(random_state=1);
melbourne_model.fit(X,y);
####Predictions####
print('Predictions for following 5 houses')
print(X.head())
print('Predictions are')
print(melbourne_model.predict(DF_1))
#Actual 1st 5 Prices
print('Actual first 5 home prices')
print(y.head())
####Model Validation####
