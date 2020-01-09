#@author: winegardner
""" """
import pandas as pd
def feature_extract(data:pd.DataFrame):
   """Converting Na to 0 for FireplaceQu,GarageType,GarageFinish"""
   data['FireplaceQu'].fillna(0, inplace=True)
   data['GarageType'].fillna(0, inplace=True)
   data['GarageFinish'].fillna(0, inplace=True)
   """Categorical Features converted to ordinals."""
   # features = data[['Functional', 'FireplaceQu','GarageType','GarageFinish']]
   data.Functional.replace({'Typ':0, 'Min1':1,'Min2':2,'Mod':3,'Maj1':4,'Maj2':5, 'Sev':6, 'Svg':7}, inplace=True)
   data.Functional = data.Functional.apply(int)
   # data.loc[data['Functional']=='Typ','Functional'] = 0
   # data.loc[data['Functional']=='Min1','Functional'] = 1
   # data.loc[data['Functional']=='Min2','Functional'] = 2
   # data.loc[data['Functional']=='Mod','Functional'] = 3
   # data.loc[data['Functional']=='Maj1','Functional'] = 4
   # data.loc[data['Functional']=='Maj2','Functional'] = 5
   # data.loc[data['Functional']=='Sev','Functional'] = 6
   # data.loc[data['Functional']=='Svg','Functional'] = 7
 
   data.FireplaceQu.replace({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 'Sev':6}, inplace=True)
   data.FireplaceQu = data.FireplaceQu.apply(int)
   # data.loc[data['FireplaceQu']=='Po','FireplaceQu'] = 1
   # data.loc[data['FireplaceQu']=='Fa','FireplaceQu'] = 2
   # data.loc[data['FireplaceQu']=='TA','FireplaceQu'] = 3
   # data.loc[data['FireplaceQu']=='Gd','FireplaceQu'] = 4
   # data.loc[data['FireplaceQu']=='Ex','FireplaceQu'] = 5
   # data.loc[data['FireplaceQu']=='Sev','FireplaceQu'] = 6
   
   data.GarageType.replace({'CarPort':1, '2Types':2, 'Basment':3, 'Detchd':4, 'BuiltIn':5, 'Attchd':6}, inplace=True)
   data.GarageType = data.GarageType.apply(int)
   # data.loc[data['GarageType']=='CarPort','GarageType'] = 1
   # data.loc[data['GarageType']=='2Types','GarageType'] = 2
   # data.loc[data['GarageType']=='Basment','GarageType'] = 3
   # data.loc[data['GarageType']=='Detchd','GarageType'] = 4
   # data.loc[data['GarageType']=='Builtin','GarageType'] = 5
   # data.loc[data['GarageType']=='Attchd','GarageType'] = 6
   
   data.GarageFinish.replace({'Unf':1, 'RFn':2, 'Fin':3}, inplace = True)
   # data.loc[data['GarageFinish']=='Unf','GarageFinish'] = 1
   # data.loc[data['GarageFinish']=='RFn','GarageFinish'] = 2
   # data.loc[data['GarageFinish']=='Fin','GarageFinish'] = 3
   return data[['Functional', 'FireplaceQu','GarageType','GarageFinish']]

def main():
   df = pd.read_csv('train.csv')
   newdf=feature_extract(df)
   print(newdf.dtypes)

if __name__ == '__main__':
   main()
   

    
   



