#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[19]:


df=pd.read_csv("C:/Users/Welcome/Downloads/Student performance2.csv")


# In[20]:


df


# In[21]:


df.isnull()


# In[24]:


print(df.columns)


# In[28]:


series=pd.isnull(df['Math_Score'])


# In[41]:


df[series]


# In[39]:


missing_values = ["Na", "na"]
df = pd.read_csv("C:/Users/Welcome/Downloads/Student performance2.csv",na_values=missing_values)


# In[40]:


df


# In[42]:


import pandas as pd
import numpy as np


# In[43]:


df=pd.read_csv("C:/Users/Welcome/Downloads/Student performance2.csv")


# In[44]:


df


# In[46]:


ndf=df
ndf.fillna(0)


# In[47]:


m_v=df['Math_Score'].mean()
df['Math_Score'].fillna(value=m_v, inplace=True)


# In[48]:


df


# In[49]:


ndf.replace(to_replace = np.nan, value = -99)


# In[ ]:




