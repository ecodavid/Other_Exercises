#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
mdata = pd.read_csv('melb_data.csv')
mdata.head()


# In[2]:


mdata.isna().sum()


# In[3]:


mdata.shape


# In[4]:


# dropna drops missing values or N.A
mdata = mdata.dropna(axis = 0)


# In[5]:


len(mdata)


# In[6]:


# Selecting the predicted Target 'y' which is the column Price
y = mdata.Price


# In[7]:


# Select the number of features by providing a list of column names inside brackets []
m_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']


# In[8]:


m_features


# In[9]:


# Use the previous list to select a subset of the original dataframe
x = mdata[m_features]

# it can be done in one single line: 
# x = mdata[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]


# In[10]:


x.head()


# In[11]:


import seaborn as sns
# visualize the relationship between the features and the response using scatterplots
sns.pairplot(mdata, x_vars=['Rooms','Bathroom','Landsize','Lattitude','Longtitude'], y_vars='Price', size=7, aspect=0.7, 
             kind='reg')


# In[12]:


# Using scikit-learn to create models following these steps:
# 1. Define what type of model will it be
# 2. Fit by capturing patterns from provided data
# 3. Predict an outcome
# 4. Evaluate. Determine how acurate the model predictions are


# In[13]:


# 1. Define what type of model will it be
from sklearn.tree import DecisionTreeRegressor
m_model = DecisionTreeRegressor(random_state = 1)


# In[14]:


# 2. Fit by capturing patterns from provided data
m_model.fit(x,y)


# In[15]:


# 3. Pediction of values
print("Making predictions for the following 5 houses:")
print (x.head())
print("The predictions are")
print(m_model.predict(x.head()))


# In[16]:


m_model.predict([[5,3,175.4,-38.3,165.98]])


# In[17]:


# Splitting x and y into training and testing sets


# In[18]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)


# In[19]:


# default split is 75% for training and 25% for testing
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[20]:


# 1. Define the model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()


# In[21]:


# 2. Fit the model
linreg.fit(x_train, y_train)


# In[22]:


# Interpreting model coefficients


# In[23]:


# print the intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)


# In[24]:


# pair the feature names with the coefficients
list(zip(m_features, linreg.coef_))


# In[25]:


# y = -143324115 + 255399 x Rooms + 236820 x Bathroom + 21.99 x Landsize - 1417616 x Latitude + 618423 x Longtitude 


# In[26]:


# 3. Prediction on validation or testing data
y_pred = linreg.predict(x_test)


# In[27]:


# 4. Evaluate. Determine how acurate the model predictions are
# Metrics
# MAE is the easiest to understand, because it's the average error. Mean Absolute Error
# MSE is more popular than MAE, because MSE "punishes" larger errors. Mean Squared Error
# RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.

# Computing the RMSE (Root Mean Squared Error) for our property Price prediction

from sklearn import metrics
from sklearn.metrics import mean_absolute_error
import numpy as np

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[28]:


linreg.predict([[5,3,175.4,-38.3,165.98]])


# In[29]:


# Removing features for better predicitons

# create a Python list of feature names
m_features = ['Rooms', 'Bathroom', 'Landsize']

# use the list to select a subset of the original DataFrame
x = mdata[m_features]

# select a Series from the DataFrame
y = mdata.Price

# split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# fit the model to the training data (learn the coefficients)
linreg.fit(x_train, y_train)

# make predictions on the testing set
y_pred = linreg.predict(x_test)

# compute the RMSE of our predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[30]:


# Adding features seeking better predictions

# create a Python list of feature names
m_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']

# use the list to select a subset of the original DataFrame
x = mdata[m_features]

# select a Series from the DataFrame
y = mdata.Price

# split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

# fit the model to the training data (learn the coefficients)
linreg.fit(x_train, y_train)

# make predictions on the testing set
y_pred = linreg.predict(x_test)

# compute the RMSE of our predictions
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




