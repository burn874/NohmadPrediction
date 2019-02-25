#!/usr/bin/env python
# coding: utf-8

#test classifier

# In[ ]:


import numpy as np


# In[20]:


def relu(x):
    return np.maximum(0, x)
def matmul(x, y):
    return np.matmul(x, y)
def argmax(x):
    return np.argmax(x, axis = 1)


# In[21]:


m1 = np.load("m1.npy")
b1 = np.load("b1.npy")
m2 = np.load("m2.npy")
b2 = np.load("b2.npy")
X = np.load("Xtrain.npy")


# In[22]:


h = argmax(matmul(relu(matmul(X, m1) + b1), m2) + b2)


# In[23]:


print(h.shape)


# In[24]:


print(set(h))


# In[ ]:




