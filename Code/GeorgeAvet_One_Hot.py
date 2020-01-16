

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("train.csv")
data.head()

from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define Condition 1 
data_Condition1 = data['Condition1']


# # integer encode
label_encoder1 = LabelEncoder()
integer_encoded1 = label_encoder.fit_transform(data_Condition1)

# binary encode
onehot_encoder1 = OneHotEncoder(sparse=False)
integer_encoded1 = integer_encoded.reshape(len(integer_encoded1), 1)
onehot_encoded_Condition1 = onehot_encoder.fit_transform(integer_encoded1)
# print(onehot_encoded_Condition1)



# define Condition 2 
data_Condition2 = data['Condition2']


# integer encode
label_encoder2 = LabelEncoder()
integer_encoded2 = label_encoder.fit_transform(data_Condition2)

# binary encode
onehot_encoder2 = OneHotEncoder(sparse=False)
integer_encoded2 = integer_encoded.reshape(len(integer_encoded2), 1)
onehot_encoded_Condition2 = onehot_encoder.fit_transform(integer_encoded2)
# print(onehot_encoded_Condition2)





