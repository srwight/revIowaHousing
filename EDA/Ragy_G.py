#!/usr/bin/env python
# coding: utf-8

# In[30]:


"""" Ragy GadElkareem		Feature Set number:
ExterQual	 Evaluates the quality of the material on the exterior 			
ExterCond	 Evaluates the present condition of the material on the exterior			
Foundation	 Type of foundation
BsmtQual Evaluates the height of the basement Importing packages """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv("house_price.csv")
data.columns


# In[31]:


# save all categorical columns in list
categorical_columns = [col for col in data.columns.values if data[col].dtype == 'object']

# dataframe with categorical features
data_cat = data[categorical_columns]


# In[32]:


# Using describe function in numeric dataframe 
data_cat.describe()


# In[33]:


data_cat.columns


# In[34]:


my_features = data[['BsmtQual','ExterQual','ExterCond','Foundation']]
my_features.head()


# In[35]:


my_features.isnull().sum()


# In[36]:


baseQual_null = my_features.BsmtQual.isnull().sum()/data.BsmtQual.count()*100
baseQual_null


# In[37]:


my_features.BsmtQual.value_counts()


# In[38]:


my_features.ExterCond.value_counts()


# In[39]:


my_features.ExterQual.value_counts()


# In[40]:


my_features.Foundation.value_counts()


# In[41]:


my_features.isna().sum()


# In[42]:


my_features.head()


# In[43]:


# GET THIS
##my_dummy_features = pd.concat([pd.get_dummies(my_features[['ExterQual','ExterCond','Foundation','BsmtQual']]), my_features[['ExterQual','ExterCond','Foundation','BsmtQual']]])
my_dummy_features = pd.get_dummies(my_features)

# Still to do: Get Logarithms if you need them


# In[44]:


my_dummy_features.head()


# In[45]:


sns.barplot(my_features.ExterQual,target)


# In[ ]:





# In[46]:


sns.barplot(my_features.ExterCond,target)


# In[47]:


# As it appears there is a corlation between sales price and the exterior Condition from highest to lowest(EX->TA->GD->FA->PO)


# In[48]:


sns.barplot(data.BsmtQual,target)


# In[49]:


### As it appears there is a corlation between sales price and the basement quality from highest to lowest(EX->GD->TA->FA)


# In[50]:


sns.barplot(data.Foundation,target)


# In[51]:


#As it appears there is a corlation between sales price and the foundation type from highest to lowest
#(pCon->wood->stone->CBlock->BrKTil->Slab)

