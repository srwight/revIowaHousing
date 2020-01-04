import pandas as pd

def feature_extract(df:pd.DataFrame) -> pd.DataFrame:
      """Extracts 'Fireplaces', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd' where 
      KitchenAbvGr is excluded due to its negative correlation to sale price being inversely related to proper sales price predictions"""
    features = df[['Fireplaces', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
    return features
