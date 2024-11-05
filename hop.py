import pandas as pd
import json
import requests
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model   import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle


df1 = pd.read_csv('19901999.csv')
df2 = pd.read_csv("2000Feb2012.csv")
df3 = pd.read_csv("Mar2012toDec2014.csv")
df4 = pd.read_csv("Jan2015toDec2016.csv")
df5 = pd.read_csv("Jan2017onwards.csv")
house = pd.concat([df1 , df2 , df3, df4, df5],axis=0)
#print(df5.isnull().sum())

house = house.dropna()

house['month'] = pd.to_datetime(house['month'])
house['year'] = house['month'].dt.year
house['month_of_year'] = house['month'].dt.month
house['lease_commence_year'] = pd.to_datetime(house['lease_commence_date'],format = '%Y').dt.year

data = house['remaining_lease']
house_new = pd.DataFrame(data)


lease_info = house['remaining_lease'].str.extract(r'(\d+) years (\d+) months')
lease_info.columns = ['years', 'months']


house['remaining_lease_years'] = pd.to_numeric(lease_info['years'])
house['remaining_lease_months'] = pd.to_numeric(lease_info['months'])

house_new_data = house.copy()

house_new_data['remaining_lease_years'].fillna(house['remaining_lease_years'].mean(),inplace = True)
house_new_data['remaining_lease_months'].fillna(house['remaining_lease_months'].mean(),inplace = True)

house_new_data.drop(columns=['month','lease_commence_date','remaining_lease'],inplace = True)

encoder = LabelEncoder()
house_new_data['town'] = encoder.fit_transform(house_new_data['town'])
house_new_data['flat_type'] = encoder.fit_transform(house_new_data['flat_type'])
house_new_data['storey_range'] = encoder.fit_transform(house_new_data['storey_range'])
house_new_data['flat_model'] = encoder.fit_transform(house_new_data['flat_model'])


#def plot(house_new_data,column):
  #  plt.figure(figsize =(15,6))
  #  plt.subplot(1,3,1)
  #  sns.boxplot(data = house_new_data ,x = column)
  #  plt.title(f'box plot for {column}')
    

  #  plt.subplot(1,3,2)
  #  sns.histplot(data = house_new_data ,x = column,kde = True ,bins = 40)
  #  plt.title(f'distribution  plot for {column}')
 #   plt.show()


#for i in ['floor_area_sqm','resale_price','lease_commence_year','remaining_lease_years']:
#    plot(house_new_data ,i)

house_new_data['floor_area_sqm'] = np.log(house_new_data['floor_area_sqm'])
house_new_data['resale_price'] = np.log(house_new_data['resale_price'])

def outlier(house_new_data ,column):
    IQR = house_new_data[column].quantile(0.75)-house_new_data[column].quantile(0.25)
    upper_value = house_new_data[column].quantile(0.75)+1.5*IQR
    lower_value = house_new_data[column].quantile(0.25)-1.5*IQR
    
    house_new_data[column] =     house_new_data[column].clip(upper_value,lower_value)

outlier(house_new_data, 'floor_area_sqm')
outlier(house_new_data, 'resale_price')
house_new_data1 = house_new_data.copy()

house_new_data1.drop(columns=['block','street_name'],inplace = True)

x = house_new_data1[['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model',
        'year', 'month_of_year', 'lease_commence_year',
       'remaining_lease_years', 'remaining_lease_months']]
y = house_new_data1[['resale_price']]

encoder = StandardScaler()

encoder.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)
x_train.shape,x_test.shape

RFR = RandomForestRegressor()

RFR= RandomForestRegressor(n_estimators= 50 ,random_state = 0)

RFR.fit(x_train,y_train)

y_pred_train = RFR.predict(x_train)
y_pred_test = RFR.predict(x_test)

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test,y_pred_test)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)

param = {'max_depth'        : [20],
              'min_samples_split': [ 5, ],
              'min_samples_leaf' : [ 2, ],
              'max_features'     : ['log2']}
grid_searchcv = GridSearchCV(RandomForestRegressor(),param_grid = param,  cv = 5)
grid_searchcv.fit(x_train, y_train)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)
x_train.shape,x_test.shape

RFR = RandomForestRegressor()

Hyper_model= RandomForestRegressor(max_depth= 20 ,max_features='log2' ,min_samples_leaf=2, min_samples_split=5)

Hyper_model.fit(x_train,y_train)

y_pred_train = Hyper_model.predict(x_train)
y_pred_test = Hyper_model.predict(x_test)

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test,y_pred_test)

r2_train,r2_test


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)
x_train.shape,x_test.shape

RFR = RandomForestRegressor()

Hyper_model= RandomForestRegressor(max_depth= 20 ,max_features='log2' ,min_samples_leaf=2, min_samples_split=5)

Hyper_model.fit(x_train,y_train)

y_pred_train = Hyper_model.predict(x_train)
y_pred_test = Hyper_model.predict(x_test)  

# manually passed the user input and predict the selling price

user_data = np.array([[0,1,3,3.785069,5,2017,1,1979,61.000000,4.000000]])
y_prediction = Hyper_model.predict(user_data)

user_data = np.array([[4,3,2,4.785069,4,2023,3,1989,69.000000,4.000000]])

y_prediction = Hyper_model.predict(user_data)
y_prediction[0]

np.exp(y_prediction[0])

with open("house_price_model.pkl", 'wb') as f:
    pickle.dump(Hyper_model, f)
     

with open("house_price_model.pkl", 'rb') as f:
    model = pickle.load(f)

#testing

user_data = np.array([[4, 3, 2, 4.785069, 4, 2023, 3, 1989, 69.000000, 4.000000]])
prediction = model.predict(user_data)
predicted_price = prediction[0]
#print(predicted_price)
np.exp(predicted_price)






