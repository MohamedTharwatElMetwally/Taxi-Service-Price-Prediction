import numpy as np
import pandas as pd
from pandas.core.reshape import concat
from sklearn import datasets, svm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from Pre_processing import *
from datetime import datetime


# Load taxi data
data_taxi_rides = pd.read_csv('taxi-rides-classification.csv')
data_test=pd.read_csv('taxi-test-samples.csv')
data_weather = pd.read_csv('weather.csv')

data_concat=pd.concat([data_taxi_rides,data_test,],axis=0)

###################################  Preprocessing  ################################### 

############ [1] Check Missing Values or Null Existence

# -------------taxi-rides--------------

# To know which columns have missing values
print('Check Missing Values Or Null Existence [taxi-rides-classification.csv]:')
print('(Rows,Columns):', data_concat.shape)
print(data_concat.isna().sum())
print('______________________________________\n')

# ---------------weather----------------

# To know which columns have missing values
print('Check Missing Values Or Null Existence [weather.csv]:')
print('(Rows,Columns):', data_weather.shape)
print(data_weather.isna().sum())
print('______________________________________\n')

# If Found, fill fill those missing values By 0.0
data_weather.fillna(value=0.0, inplace=True)
print('After dealing with them --fill those missing values By 0.0-- [weather.csv]:')
print('(Rows,Columns):', data_weather.shape)
print(data_weather.isna().sum())
print('______________________________________\n')


############ [2] Merge --taxi_rides & weather--


data_concat['time_stamp']=data_concat['time_stamp'].apply(lambda x: int(x/1000))
data_concat['time_stamp']=data_concat['time_stamp'].apply(lambda ts: datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d'))
data_weather['time_stamp']=data_weather['time_stamp'].apply(lambda ts: datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d'))

data_weather.drop_duplicates(subset=["time_stamp",'location'],inplace=True)

data_taxi_rides_weather= pd.merge(data_concat, data_weather , how= 'inner'
     , left_on= ["time_stamp","source"] , right_on=["time_stamp",'location']) 


print('Merge Details: \n')                      
print('Before Merge: \n')                            
print('Taxi_Rides: (Rows,Columns):',data_concat.shape)
print('Taxi_Rides:  Duplicated:',data_concat.duplicated().sum())
print('Weather: (Rows,Columns):',data_weather.shape)
print('Weather:  Duplicated:',data_weather.duplicated().sum())                                   
print('\nAfter Merge:\n')                            
print('(Rows,Columns):',data_taxi_rides_weather.shape)
print('Duplicated: ',data_taxi_rides_weather.duplicated().sum())
print('______________\n')



############ [3] Data Encoding

cols=['cab_type','destination','source','product_id','name','location','RideCategory']
data_taxi_rides_weather=Feature_Encoder(data_taxi_rides_weather,cols)

X=data_taxi_rides_weather.iloc[:,:] #Features
Y=data_taxi_rides_weather['RideCategory'] #Label
X.drop(['time_stamp','id','RideCategory'],axis=1,inplace=True)

print(Y.value_counts())

"""
###################################  Feature Selection  ###################################

#Get the correlation between the features
corr = data_taxi_rides_weather.corr()
print('Correlation:\n',abs(corr['RideCategory'].sort_values()))

X.drop(['cab_type', 'time_stamp', 'destination', 'source',
       'surge_multiplier', 'id', 'product_id', 'RideCategory', 'temp',
       'location', 'clouds', 'pressure', 'rain', 'humidity', 'wind'],axis=1,inplace=True)

print('the top features= ',X.columns)
print('______________\n')


"""
####################  Separate main dataset & test sample  #####################


test_X=X.iloc[-4:,:]
test_Y=Y.iloc[-4]
X=X.iloc[0:-4,:]
Y=Y.iloc[0:-4]


#######################  Classification & Data Splitting #######################

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,shuffle=True,random_state=10)

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=30),algorithm="SAMME",n_estimators=100)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
test_X = scaler.transform(test_X)

error = []

clf = tree.DecisionTreeClassifier(max_depth=30)
clf.fit(X_train,y_train)
y_prediction = clf.predict(test_X)
accuracy=np.mean(y_prediction == test_Y)*100
print ("The achieved accuracy using Decision Tree is " + str(accuracy))

bdt.fit(X_train,y_train)
y_prediction = bdt.predict(test_X)
accuracy=np.mean(y_prediction == test_Y)*100
print ("The achieved accuracy using Adaboost is " + str(accuracy))


print('_________________________________________________________________________')
print('_________________________________________________________________________')


"""
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,shuffle=True,random_state=10)

# Training the model
model_tree = DecisionTreeClassifier(criterion='entropy',max_depth=30,min_samples_leaf=30,min_samples_split=150,random_state=10)
model_tree.fit(x_train,y_train)

# Making predictions
y_pred_train_dtree=model_tree.predict(x_train)
y_pred_test_dtree=model_tree.predict(x_test)

# Calculating accuracies
t_acc=accuracy_score(y_train,y_pred_train_dtree)
v_acc=accuracy_score(y_test,y_pred_test_dtree)

print('trainning acc={}'.format(t_acc))
print('validation acc={}'.format(v_acc))
"""
