"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def feature_extract() :
    #getting my feautres into a seprate data frame
    my_features = data[['BsmtQual','ExterQual','ExterCond','Foundation']]
    #creating a list of my features 
    listoffeatures=['BsmtQual','ExterQual','ExterCond']
    #checking if there is any missing value and fill it in with 0
    my_features = my_features.fillna('NA')
    #creating a list of the the ratings 
    myvars = ['NA','Po', 'Fa', 'TA', 'Gd', 'Ex']
    # Then we make a list of values from 0 to 5 that we want to apply to our ratings
    myvals = list(range(6))
    #creating a dictionary to map values to ratings 
    mymap = dict(zip(myvars, myvals))
    #creating a loop in my features list to check what rate and pass a value in a new list
    for column in listoffeatures:
        newcol = my_features[column].apply(lambda x: mymap[x])
        #drop the old list 
        my_features.drop(column, axis=1, inplace=True)
        #assign the new list to my old list 
        my_features[column]=newcol
    #getting my dummy features     
    my_dummy_features = pd.get_dummies(my_features)
    #concate all my features (the dummy and my_features) to one data frame 
    my_features = pd.concat([my_features,my_dummy_features],axis = 1)
    #return the data frame 
    return my_features """

"""" Ragy GadElkareem		Features :
ExterQual	 Evaluates the quality of the material on the exterior 			
ExterCond	 Evaluates the present condition of the material on the exterior			
Foundation	 Type of foundation
BsmtQual Evaluates the height of the basement Importing packages """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("house_price.csv")
data.columns


# In[32]:
# save all categorical columns in list
categorical_columns = [col for col in data.columns.values if data[col].dtype == 'object']

# dataframe with categorical features
data_cat = data[categorical_columns]


# Printing 5 head observation in categorical dataframe
data_cat.head()


# In[33]:

# checking my columns 
data_cat.columns


# In[34]:

# export my features into a dataframe 
my_features = data[['BsmtQual','ExterQual','ExterCond','Foundation']]
my_features.head()


# In[35]:

#checking the nulity of my features 
my_features.isnull().sum()


# In[36]:


#since we hsve only nulity in basequal, we will check how many they are and the percentage of it 
baseQual_null = my_features.BsmtQual.isnull().sum()/data.BsmtQual.count()*100
baseQual_null


# In[37]:

#checking the different values of my features 
my_features.BsmtQual.value_counts()


# In[38]:


my_features.ExterCond.value_counts()


# In[39]:


my_features.ExterQual.value_counts()


# In[40]:


my_features.Foundation.value_counts()




# In[42]:


my_features.head()



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

