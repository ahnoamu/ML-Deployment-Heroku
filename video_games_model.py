#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#== import video games data
vg_data = pd.read_csv('/home/arnold/Desktop/Data_Glacier/Week4/FlaskDeployment/vgsales.csv')


# In[7]:


#Drop any duplicate rows
vg_data.drop_duplicates(inplace=True)
#Set the rank as the index for dataset
vg_data.set_index('Rank', inplace=True)


# In[9]:


#drop year column
vg_data.drop("Year", axis=1, inplace=True)
#fill in missing values in publisher with its mode value
vg_data["Publisher"] = vg_data["Publisher"].fillna(vg_data["Publisher"].mode()[0]).reset_index(drop=True)


#== Generate regression models
from sklearn import preprocessing
from sklearn import linear_model
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import scipy.stats as ss
import pickle


# In[15]:


#Transform features (inputs) into numpy arrays
X = np.array(vg_data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']])
#transform labels (output) into numpy array
y = np.array(vg_data['Global_Sales'])


# In[16]:


#Split data into train & test set
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size = 0.3, random_state = 100)


# In[17]:


#build linear regression model
lin_mod = linear_model.LinearRegression(fit_intercept = False)
#Fitting model with trainig data
lin_mod.fit(X_train, y_train)



# In[18]:


#Test model
y_score = lin_mod.predict(X_test) 

# print scores
print(np.around(y_score[:10],2)) #y_predicted
print(y_test[:10])               #y_original



# Saving model to disk
pickle.dump(lin_mod, open('reg_model.pkl', 'wb'))


# Loading model to compare the results
model = pickle.load(open('reg_model.pkl','rb'))
print(model.predict([[22, 2200, 65, 73]]))

