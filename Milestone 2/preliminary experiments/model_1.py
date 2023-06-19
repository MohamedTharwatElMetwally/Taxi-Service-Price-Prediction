import numpy as np
import pandas as pd
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
data_weather = pd.read_csv('weather.csv')

###################################  Preprocessing  ################################### 

############ [1] Check Missing Values or Null Existence

# -------------taxi-rides--------------

# To know which columns have missing values
print('Check Missing Values Or Null Existence [taxi-rides-classification.csv]:')
print('(Rows,Columns):', data_taxi_rides.shape)
print(data_taxi_rides.isna().sum())
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


data_taxi_rides['time_stamp']=data_taxi_rides['time_stamp'].apply(lambda x: int(x/1000))
data_taxi_rides['time_stamp']=data_taxi_rides['time_stamp'].apply(lambda ts: datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d'))
data_weather['time_stamp']=data_weather['time_stamp'].apply(lambda ts: datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d'))

data_weather.drop_duplicates(subset=["time_stamp",'location'],inplace=True)

data_taxi_rides_weather= pd.merge(data_taxi_rides, data_weather , how= 'inner' 
     , left_on= ["time_stamp","source"] , right_on=["time_stamp",'location']) 


print('Merge Details: \n')                      
print('Before Merge: \n')                            
print('Taxi_Rides: (Rows,Columns):',data_taxi_rides.shape)
print('Taxi_Rides:  Duplicated:',data_taxi_rides.duplicated().sum())  
print('Weather: (Rows,Columns):',data_weather.shape)
print('Weather:  Duplicated:',data_weather.duplicated().sum())                                   
print('\nAfter Merge:\n')                            
print('(Rows,Columns):',data_taxi_rides_weather.shape)
print('Duplicated: ',data_taxi_rides_weather.duplicated().sum())
print('______________\n')



############ [3] Data Encoding

#============== X
#-------------One-Hot-Encoding--------------

cols=['cab_type','destination','source','product_id','name','location']
data_taxi_rides_weather=pd.get_dummies(data=data_taxi_rides_weather,columns=cols)

cols=('RideCategory',)
data_taxi_rides_weather=Feature_Encoder(data_taxi_rides_weather,cols)

X=data_taxi_rides_weather.iloc[:,:] #Features
Y=data_taxi_rides_weather['RideCategory'] #Label

print(Y.value_counts())


###################################  Feature Selection  ###################################

#Get the correlation between the features
corr = data_taxi_rides_weather.corr()
print('Correlation:\n',abs(corr['RideCategory']))

"""
#Top Correlation training features with RideCategory
top_feature = corr.index[abs(corr['RideCategory'])>=0.2]
print(top_feature)
print('the top features= ',X.columns)
"""

X.drop(['product_id_8cf7e821-f0d3-49c6-8eba-e679c0ebcf6a',
       'product_id_997acbb5-e102-41e1-b155-9df7de0a73f2',
       'product_id_lyft_line', 'product_id_lyft_luxsuv','RideCategory','time_stamp', 'surge_multiplier', 'id',
       'temp', 'clouds', 'pressure', 'rain', 'humidity', 'wind',
       'cab_type_Lyft', 'cab_type_Uber', 'destination_Back Bay',
       'destination_Beacon Hill', 'destination_Boston University',
       'destination_Fenway', 'destination_Financial District',
       'destination_Haymarket Square', 'destination_North End',
       'destination_North Station', 'destination_Northeastern University',
       'destination_South Station', 'destination_Theatre District',
       'destination_West End', 'source_Back Bay', 'source_Beacon Hill',
       'source_Boston University', 'source_Fenway',
       'source_Financial District', 'source_Haymarket Square',
       'source_North End', 'source_North Station',
       'source_Northeastern University', 'source_South Station',
       'source_Theatre District', 'source_West End',
       'product_id_55c66225-fbe7-4fd5-9072-eab1ece5e23e',
       'product_id_6c84fd89-3f11-4782-9b50-97c468b19529',
       'product_id_6d318bcc-22a3-4af6-bddd-b409bfce1546',
       'product_id_6f72dfc5-27f1-42e8-84db-ccc7a75f6969',
       'product_id_9a0e7b09-b92b-4c41-9779-2ad22b4d779d', 'product_id_lyft',
       'product_id_lyft_lux',
       'product_id_lyft_plus', 'product_id_lyft_premier',
       'location_Back Bay', 'location_Beacon Hill',
       'location_Boston University', 'location_Fenway',
       'location_Financial District', 'location_Haymarket Square',
       'location_North End', 'location_North Station',
       'location_Northeastern University', 'location_South Station',
       'location_Theatre District', 'location_West End'],axis=1,inplace=True)

print('the top features= ',X.columns)
print('______________\n')


#######################  Classification & Data Splitting #######################

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,shuffle=True,random_state=10)

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=30),algorithm="SAMME",n_estimators=100)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

error = []

clf = tree.DecisionTreeClassifier(max_depth=30)
clf.fit(X_train,y_train)
y_prediction = clf.predict(X_test)
accuracy=np.mean(y_prediction == y_test)*100
print ("The achieved accuracy using Decision Tree is " + str(accuracy))

bdt.fit(X_train,y_train)
y_prediction = bdt.predict(X_test)
accuracy=np.mean(y_prediction == y_test)*100
print ("The achieved accuracy using Adaboost is " + str(accuracy))


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