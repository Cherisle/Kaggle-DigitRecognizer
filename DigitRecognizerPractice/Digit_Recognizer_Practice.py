#Machine Learning Code beginnings House Price Practice

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
####Read in Data####
melbourne_file_path = ('C:/Users/Nicholas Trigueros/Documents/GitHub/Kaggle-DigitRecognizer\DigitRecognizerPractice/melb_data.csv')
#Create DataFrame
melbourne_data = pd.read_csv(melbourne_file_path)
#Drop all null values
melbourne_data.dropna(axis=0)
#set prediction target which is the price of the houses
y = melbourne_data.Price
#features to include
melbourne_features = ['Rooms','Bathroom','Landsize','Lattitude','Longtitude']
X = melbourne_data[melbourne_features]
#Print DataFrames for Inspection
print('\nDataFrame for Data Set melbourne_data')
print(X.describe())
#Shows First 5 Lines of Data
DF_1 = X.head()


####Model Validation & Predictions####
#Generation of Validation Data using train_test_split

#Split the data to use some to train with and the other to validate with
#This will allow us to build a non biased model to work with
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)
#Define Model
melbourne_model = DecisionTreeRegressor()
#Fit Model
melbourne_model.fit(train_X,train_y)
#Predict using validation Data
val_predictions = melbourne_model.predict(val_X)
print(f'\nUsing train/test splitting, the mean absolute error is {mean_absolute_error(val_y, val_predictions)}\n')

#Function to reduce over and underfitting problems
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X,train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
#Initialize dictionary d for later use
d = {}
for max_leaf_nodes in [5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y)
    #To get specific best size in Code as output we will extract the min of this below
    #This will make a dict with the mae as the keys and leaf nodes as the values
    d[my_mae] = max_leaf_nodes
    print(f'Max leaf nodes: {max_leaf_nodes} \t\t Mean Absolute Error: {my_mae}')

#Here we will get the specific best tree size
best_tree_size = d[min(d)]
print('\nAccounting for over and underfitting errors...')
print(f'The best tree size will be {best_tree_size}')

####Final Model with ALL Data####

#Here we will create the final model_selection
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size,random_state=0)
#Instead of using the split data we now will be using all of it.
final_model.fit(X,y)
#Prediction time
final_predictions = final_model.predict(X)
print(f'\nFinal Predicted Prices are...{final_predictions}')

####Random Forest Method####
rf_model = RandomForestRegressor(random_state=1)
rf_model = rf_model.fit(train_X,train_y)

rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions,val_y)

print(f'\nValidation Mean Absolute error for Random Forest Model {rf_val_mae}')
#for submitting competition predictions
#output = pd.DataFrame({'Id': melbourne_data.Id,
#                       'Price':final_predictions })
#output.to_csv('submission.csv', index=False)
