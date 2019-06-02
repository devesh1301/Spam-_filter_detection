#!/usr/bin/env python
# coding: utf-8

# In[11]:


# impoert all these libraries 
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[2]:


# load the dataset
data=pd.read_csv('spambase.data').as_matrix()


# In[7]:


data.shape


# In[9]:


# selecting  starting 48 columns 
X=data[:,:48]


# In[10]:


# output(spam=1,not spam=0)
y=data[:,-1]


# In[18]:


X_shuffle,y_shuffle=shuffle(X,y)


# In[34]:


#X_scaled=StandardScaler().fit_transform(X_shuffle)


# In[38]:


# dividing of  data into train and testing
X_train,X_test,Y_train,Y_test=train_test_split(X_shuffle,y_shuffle,test_size=0.2)


# In[39]:


#model selection from sklearn
model=MultinomialNB()


# In[40]:


model.fit(X_train,Y_train)


# In[41]:


print("Prediction score for NB",model.score(X_test,Y_test))


# In[42]:


from sklearn.ensemble import AdaBoostClassifier


# In[43]:


model=AdaBoostClassifier()


# In[44]:


model.fit(X_train,Y_train)


# In[45]:


print("Prediction score for ADa:",model.score(X_test,Y_test))


# In[ ]:





# In[ ]:




