### THIS IS WHERE WE CLEAN UP ORDINAL AND UNIQUE FEATURES
import pandas as pd
import 

def ordinals(df) -> pd.DataFrame:

    ### Dictionaries and fill values
    ## Put your feature’s information here in the below-outlined format.
    ## Leave a space after each feature.
        
    ## BsmtExposure
    BsmtExposure_dict = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No':1, 'NA':0}
    BsmtExposure_fillval = 0

    ## Functional
    Functionaldict = {
        "Typ": 0,
        "Min1": 1,
        "Min2": 2,
        "Mod": 3,
        "Maj1": 4,
        "Maj2": 5,
        "Sev": 6,
        "Svg": 7
    }
    Functional_fill = 0

    # FireplaceQu
    FireplaceQudict = {
        "Po": 1,
        "Fa": 2,
        "Ta": 3,
        "Gd": 4,
        "EX": 5
    }
    FireplaceQu_fill = 0

    # GarageFinish
    GarageFinishdict = {
        "Unf": 1,
        "RFn": 2,
        "Fin": 3
    }
    GarageFinish_fill = 0
    
    # LandSlope
    landSlope_dict = {'Sev':3, 'Mod':2, 'Gtl':1}
    landSlope_fill = 1

    # The Rest
    generic_dict = {'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'Na':0}
    generic_list = ['BsmtCond','BsmtQual','FireplaceQu','GarageQual','GarageCond','PoolQC','ExterQual','KitchenQual','HeatingQC','ExterCond']
    generic_filler = 0

    ### Standard Ordinal Replacement Calls
    df_out = pd.concat(
        [
            ordinalRepl(df.BsmtExposure, BsmtExposure_dict, BsmtExposure_fillval),
            ordinalRepl(df.Functional, Functionaldict, Functional_fill),
            ordinalRepl(df.FireplaceQu, FireplaceQudict, FireplaceQu_fill),
            ordinalRepl(df.GarageFinish, GarageFinishdict, GarageFinish_fill),
            ordinalRepl(df.LandSlope, landSlope_dict, landSlope_fill)
            #insert your function call above this comment
        ],
        axis = 1  
    )

    df_generic = pd.concat([ordinalRepl(df[x], generic_dict, generic_filler) for x in generic_list], axis = 1)
    df_out = pd.concat([df_out, df_generic], axis = 1)
        
    ### Bulk Ordinal Replacement Call
    # This is where the call for all of the ones that have the same 
    # dictionary will go - (including BsmtExposure?)

    ### Unique Variable Function Calls
    # Put your function in your own file, but call it here.
    def fence_uniq(df_in):
        df_out = pd.DataFrame()
        df_out['Fence_Wood'] = df_in.replace(["MnWw", "GdWo", "NA", "MnPrv", "GdPrv"], [1, 2, 0, 0, 0]).fillna(0)
        df_out['Fence_Private'] = df_in.replace(["MnWw", "GdWo", "NA", "MnPrv", "GdPrv"], [0, 0, 0, 1, 2]).fillna(0)
        return df_out
	

if __name__ == '__main__':
	df = pd.read_csv('train.csv')
	print(ordinals(df).head())


