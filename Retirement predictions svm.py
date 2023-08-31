#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization


# In[2]:


bank_df = pd.read_csv('Bank_Customer_retirement.csv')


# In[3]:


bank_df.keys()


# In[4]:


bank_df.shape


# In[5]:


bank_df.head()


# In[6]:


bank_df.tail()


# In[7]:


sns.pairplot(bank_df, hue = 'Retire', vars = ['Age', '401K Savings'] )


# In[8]:


sns.countplot(bank_df['Retire'], label = "Retirement") 


# In[9]:


bank_df = bank_df.drop(['Customer ID'],axis=1)


# In[10]:


# Let's drop the target label coloumns
X = bank_df.drop(['Retire'],axis=1)


# In[11]:


X


# In[12]:


y = bank_df['Retire']
y


# In[13]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)


# In[14]:


X_train.shape


# In[15]:


X_test.shape


# In[16]:


y_train.shape


# In[17]:


y_test.shape


# In[18]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)


# In[19]:


y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)


# In[20]:


sns.heatmap(cm, annot=True)


# In[21]:


print(classification_report(y_test, y_predict))


# In[22]:


min_train = X_train.min()
min_train


# In[23]:


range_train = (X_train - min_train).max()
range_train


# In[24]:


X_train_scaled = (X_train - min_train)/range_train


# In[25]:


X_train_scaled


# In[26]:


y_train


# In[28]:


sns.scatterplot(x = X_train['Age'], y = X_train['401K Savings'], hue = y_train)


# In[29]:


sns.scatterplot(x = X_train_scaled['Age'], y = X_train_scaled['401K Savings'], hue = y_train)


# In[30]:


min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test


# In[31]:


from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


# In[32]:


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm,annot=True,fmt="d")


# In[33]:


print(classification_report(y_test,y_predict))


# In[34]:


param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 


# In[35]:


from sklearn.model_selection import GridSearchCV


# In[36]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)


# In[37]:


grid.fit(X_train_scaled,y_train)


# In[38]:


grid.best_params_


# In[39]:


grid.best_estimator_


# In[40]:


grid_predictions = grid.predict(X_test_scaled)


# In[41]:


cm = confusion_matrix(y_test, grid_predictions)


# In[42]:


sns.heatmap(cm, annot=True)


# In[43]:


print(classification_report(y_test,grid_predictions))

