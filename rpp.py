#rpp = real-estate price pridiction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


# step one see the data breifly bu using data in csv and using .describe(parameters) , .hist(parameters)(histrogram graphs)
#read.hist(bins=50,figsize=(20,15))
read = pd.read_csv('data/data.csv')
#print(read.info())

#Step 2 now split the data into train and test data using sklearn
train_set, test_set = train_test_split(read, test_size=0.2, random_state=42)
read = train_set.copy()

read = train_set.drop("Y house price of unit area", axis =1)

read_label =train_set["Y house price of unit area"].copy()

#step 3 for the data in the source having a important feature , it has to be divided equal for the test and train 
#so that model wil find the variations and learn it .like chas have only 0 and 1 it is note that all the 1 will not go to
#the test part as 0 one will like a unkwon to the model .anyway it is in the model.
# for this we do stratifiedShuffleSplit
'''
from sklearn.mode_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_spllts=1,test_size=0.2,random_sate=42)
for train_index, test_index in split.split(read,read['CHAS']):
          strat_train_set = housing.loc[train_index]
          strat_test_set = housing.loc[test_index]
          time span 1:56:50
'''

#step 4 Looking for correlations
#corr_matrix  = read.corr()
#print(corr_matrix['Y house price of unit area'].sort_values(ascending=False))

#scikit Learn design
'''
primarily,Three types of objects
1. Estimators - It estimates some parameter based on a dataset.eg .imputer
It has a fit method and transform method.
Fit method =Fits the dataset and calculates internal parameters

2. Transformers -transform method takes input and returns output based on the learning from fit(). It also has a convenience
function called fit_tranform() which fits and transforms.

3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions.
it also gives score() function which will evaluate the predictions.
# 
'''
# FEATURE Scaling
'''Primarily .two types of feature scaling mathods:
1. Min-max scaling (normalisation)
   (value-min)/(max-min)
   sklearn provides a class called MinMaxScaler for this

2.Standardization
   (value-mean)/std
   sklearn provides a class called StandardScaler for this
'''
#Creating a pipeline (series of steps)

my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])
read_num_tr = my_pipeline.fit_transform(read)

#SELECTING A DESIRED MODEL FOR DRAGON REAL ESTATES
#model = LinearRegression() #this model is giving very high mse
#model =DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(read_num_tr,read_label)# here we are giving fit(x n_samples or n_features(independent),y (target value dependent))
# after that we are taking samples of data x to predict y from the same traine data so that we can get y predicted
some_data=read.iloc[:5]
some_label = read_label.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
print(model.predict(prepared_data))
print(list(some_label))
# now check the rmse or mse value of the model so that we can check the model is good or not
#Evaluating the model
read_prediction=model.predict(read_num_tr)# we are predicting vale's y using x
mse=mean_squared_error(read_label,read_prediction)
rmse=np.sqrt(mse)
print(rmse)
print(mse)
#Therefore the mse and rmse are big enough to produce error we have to minimise it using the another model
'''Therefore the new decisiontreeRegressor model have overfit the data read the noise also 
,now doing Cross Validation
'''
# now checking which is best model for the above model
'''
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,read_num_tr,read_label,scoring="neg_mean_squared_error",)
rmse_scores=np.sqrt(-scores)

# creating function for the mean error comming out from the model
def print_scores(scores):
    print("Scores: ",scores)
    print("Mean: ",scores.mean())
    print("Standard daviation: ", scores.std())

#print_scores(rmse_scores)
'''
#Therefore analysing the all model error the lowest is the best Thus choosing the randomforestregressor

#Now testing the model using the test data
x_test = test_set.drop("Y house price of unit area", axis =1)
y_label = test_set["Y house price of unit area"].copy()

prepared_data = my_pipeline.transform(x_test)
final_prediction=model.predict(prepared_data)
print(final_prediction)
print("Actual Y to be come",list(y_label))
fmse = mean_squared_error(y_label,final_prediction)
frmse = np.sqrt(fmse)
print(fmse)
print(frmse)
