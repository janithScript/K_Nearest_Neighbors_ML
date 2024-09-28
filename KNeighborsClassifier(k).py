#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv("iris.csv")
data.tail()


# In[4]:


data.shape


# In[5]:


data['Species'].value_counts()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.head()


# In[9]:


x = data.iloc[: , 1:5]
x.head()


# In[10]:


y = data.iloc[: , -1]
y.head()


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[12]:


x = scaler.fit_transform(x)
x[0:5]


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


# In[14]:


x_train.shape


# In[15]:


x_test.shape


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 1)
model.fit(x_train, y_train)


# In[25]:


#pred = model.predict(x_test)
#pred[0:5]


# In[21]:


y_test[0:5]


# In[23]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, pred)
accuracy


# In[24]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm


# In[26]:


result = pd.DataFrame(data= [y_test.values,pred] , index = ['y_test', 'pred'])
result.transpose()


# In[49]:


#correct_sum = []
#for i in range (1,20):
#    model = KNeighborsClassifier(n_neighbors = i)
#    model.fit(x_train,y_train)
#    pred = model.predict(x_test)
#    correct = np.sum(pred == y_test)
#    correct_sum.append(correct)


# In[44]:


correct_sum


# In[45]:


result = pd.DataFrame(data= correct_sum)
result.index = result.index+1
result.T


# In[48]:


#model = KNeighborsClassifier(n_neighbors = 14)
#model.fit(x_train, y_train)
#pred = model.predict(x_test)


# In[47]:


accuracy = accuracy_score(y_test, pred)
accuracy


# In[ ]:




