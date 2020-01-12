import pandas as pd
from joblib import load
from clean_data import clean

df_test = pd.read_csv('test.csv')
df_Id = df_test.Id

#Drop the ID column
df_test.drop('Id',1,inplace=True)

#clean up the data
df_num, df_obj = clean(df_test)

#impute missing values in numeric data to our stored means
sr_means = load('num_means.joblib')
df_num.swapaxes('index','columns', copy=False).fillna(sr_means, axis=0, inplace=True)

#one-hot our object columns using our stored one-hot data
enc1H = load('onehot.joblib')
df_dummies = pd.DataFrame(enc1H.transform(df_obj))

df_treated = pd.concat(
    [
        df_num,
        df_dummies
    ],
    axis=1
)

#apply PCA
reducer = load('pca.joblib')
df_rotated = reducer.transform(df_treated)

#Make Predictions
model_ols = load('models_OLS.joblib')
predictions_ols = pd.Series(model_ols.predict(df_rotated)).rename('OLS')

model_ridge = load('models_ridge.joblib')
predictions_ridge = pd.Series(model_ridge.predict(df_rotated)).rename('Ridge')

model_SGD = load('models_SGD.joblib')
predictions_SGD = pd.Series(model_SGD.predict(df_rotated)).rename('SGD')

results = pd.concat(
    [
        df_Id,
        predictions_ols,
        predictions_ridge,
        predictions_SGD
    ],
    axis=1
)

print(results)