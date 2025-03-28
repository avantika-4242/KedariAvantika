#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[18]:


from sklearn.datasets import load_iris


# In[19]:


iris = load_iris()


# In[20]:


df = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[21]:


df['Species'] = iris.target


# In[22]:


print(df.head())


# In[23]:


print("\nChecking for Null values:")
print(df.isnull().sum()) 


# In[24]:


X = df.drop('Species', axis=1)  
Y = df['Species']            


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[27]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[28]:


print("\nPreprocessing Complete!")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")


# In[30]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[31]:


nb_model = GaussianNB()
nb_model.fit(X_train_scaled, Y_train)
Y_pred = nb_model.predict(X_test_scaled)


# In[32]:


accuracy = accuracy_score(Y_test, Y_pred)
print(f"\nAccuracy of the Naive Bayes model: {accuracy:.4f}")


# In[33]:


print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))


# In[34]:


print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))


# In[37]:


y_train_pred = nb_model.predict(X_train_scaled)
y_test_pred = nb_model.predict(X_test_scaled)
print("\nPredictions for the Training Set (y_train_pred):")
print(y_train_pred)

print("\nPredictions for the Test Set (y_test_pred):")
print(y_test_pred)


# In[ ]:




