import pandas as pd
from joblib import load
from clean_data import clean
import numpy as np

df_test = pd.read_csv('test.csv')
df_Id = df_test.Id

#Drop the ID column
df_test.drop('Id',1,inplace=True)

#clean up the data
df_num, df_obj = clean(df_test)

#apply log1p to columns we logged in the training
skewcols = load('skewcols.joblib')
df_num[skewcols] = np.log1p(df_num[skewcols])

#impute missing values in numeric data to our stored means
sr_means = load('num_means.joblib')
df_num.swapaxes('index','columns', copy=False).fillna(sr_means, axis=0, inplace=True)

#one-hot our object columns using our stored one-hot data
enc1H = load('onehot.joblib')
df_dummies = pd.DataFrame(enc1H.transform(df_obj))

df = pd.concat(
    [
        df_num,
        df_dummies
    ],
    axis=1
)

#apply PCA
reducer = load('pca.joblib')
df_rotated = reducer.transform(df)

#Make Predictions
#OLS No PCA
model_ols = load('models_OLS.joblib')
predictions_ols = pd.Series(model_ols.predict(df)).rename('OLS')

#OLS PCA
model_olsp = load('modelspca_OLS.joblib')
predictions_olsp = pd.Series(model_olsp.predict(df_rotated)).rename('OLS PCA')

#Ridge No PCA
model_ridge = load('models_ridge.joblib')
predictions_ridge = pd.Series(model_ridge.predict(df)).rename('Ridge')

#Ridge PCA
model_ridgep = load('modelspca_ridge.joblib')
predictions_ridgep = pd.Series(model_ridgep.predict(df_rotated)).rename('Ridge PCA')

#SGD No PCA
model_SGD = load('models_SGD.joblib')
predictions_SGD = pd.Series(model_SGD.predict(df)).rename('SGD')

#SGD PCA
model_SGDp = load('modelspca_SGD.joblib')
predictions_SGDp = pd.Series(model_SGDp.predict(df_rotated)).rename('SGD PCA')

results = pd.concat(
    [
        df_Id,
        predictions_ols.map('${:,.2f}'.format),
        predictions_olsp.map('${:,.2f}'.format),
        predictions_ridge.map('${:,.2f}'.format),
        predictions_ridgep.map('${:,.2f}'.format),
        predictions_SGD.map('${:,.2f}'.format),
        predictions_SGDp.map('${:,.2f}'.format)
    ],
    axis=1
)

results.to_csv('predictions.csv')
print(results)