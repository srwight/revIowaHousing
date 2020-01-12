import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

def basic_ordinal(sr_in: pd.Series, dictmap={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'Na':0}, filler:int = 0) -> pd.Series:
    '''

    Parameters:
    sr_in: Pandas Series includes string ratings to be
    mapped to numerical values based on the dictmap dictionary.

    dictmap: Dictionary whose keys are string markers from the dataset, and whose values are
    the values we wish to replace them with. The default values are:

        Ex - 5
        Gd - 4
        TA - 3
        Fa - 2
        Po - 1
        Na - 0

    filler: Value to fill nan's with.

    Return:
    Pandas Series of int64 type
    '''
    sr_in.fillna(filler, inplace=True)
    return sr_in.replace(dictmap).apply(int)

def BsmntFins(df_in:pd.DataFrame) -> pd.DataFrame:
    '''

    Parameters:
    type1 and type2: Semi-ordinal series including:
        Ordinal Values Representing Living Space:
            GLQ -   3
            ALQ -   2
            BLQ -   1
            else -  0
        
        Nominal Values:
            Rec
            LwQ
            Unf
            Na
    
    Returns:
        DataFrame with the following columns:
            BsmtFin_Living (0, 1, 2, or 3)
            BsmtFin_Rec (0 or 1)
            BsmtFin_LwQ (0 or 1)
            BsmtFin_Unf (0 or 1)
    '''

    BsmtLiving_Dict = {'GLQ':3,
                       'ALQ':2,
                       'BLQ':1,
                       'Rec':np.NaN,
                       'LwQ':np.NaN,
                       'Unf':np.NaN,
                       'Na':np.NaN}

    Rec_Dict = {'GLQ':np.NaN,
                'ALQ':np.NaN,
                'BLQ':np.NaN,
                'LwQ':np.NaN,
                'Unf':np.NaN,
                'Na':np.NaN,
                'Rec':1}
    
    LwQ_Dict = {'GLQ':np.NaN,
                'ALQ':np.NaN,
                'BLQ':np.NaN,
                'Rec':np.NaN,
                'Unf':np.NaN,
                'Na':np.NaN,
                'LwQ':1}

    Unf_Dict = {'GLQ':np.NaN,
                'ALQ':np.NaN,
                'BLQ':np.NaN,
                'Rec':np.NaN,
                'LwQ':np.NaN,
                'Na':np.NaN,
                'Unf':1}

    ret = pd.DataFrame()
    ret['BsmtFin_Living'] = df_in.BsmtFinType1.replace(BsmtLiving_Dict).fillna(df_in.BsmtFinType2.replace(BsmtLiving_Dict)).fillna(0).apply(int)
    ret['BsmtFin_Rec'] = df_in.BsmtFinType1.replace(Rec_Dict).fillna(df_in.BsmtFinType2.replace(Rec_Dict)).fillna(0).apply(int)
    ret['BsmtFin_LwQ'] = df_in.BsmtFinType1.replace(LwQ_Dict).fillna(df_in.BsmtFinType2.replace(LwQ_Dict)).fillna(0).apply(int)
    ret['BsmtFin_Unf'] = df_in.BsmtFinType1.replace(Unf_Dict).fillna(df_in.BsmtFinType2.replace(Unf_Dict)).fillna(0).apply(int)

    return ret

def fence(sr_in:pd.Series) -> pd.DataFrame:
    '''

    Parameters:
    sr_in - pandas Series
        Accepts row with values:
            GdPrv - 2
            MnPrv - 1

            GdWo - 2
            MnWo - 1

            NA - 0
    
    Returns:
        DataFrame with the following columns:
            Fence_Privacy (0, 1, 2)
            Fence_Wood (0, 1, 2)
    '''

    PrvDict = {'GdPrv':2, 
               'MnPrv':1,
               'GdWo':0,
               'MnWw':0,
               'NA':0}
    WoodDict = {'GdPrv':0,
                'MnPrv':0,
                'GdWo':2,
                'MnWw':1,
                'NA':0}
    
    Fence_Privacy = sr_in.replace(PrvDict).fillna(0)
    Fence_Wood = sr_in.replace(WoodDict).fillna(0)

    ret = pd.concat([Fence_Privacy,Fence_Wood], axis=1)
    ret.columns = ['Fence_Privacy','Fence_Wood']
    return ret

def conditions(df_in:pd.DataFrame) -> pd.DataFrame:
    '''
    Accepts Condition1 and Condition2, then combines them into a
    2-hot binary array.

    arguments:
    df_in - this DataFrame will hold the "Condition1 and Condition2"
    features.

    Return:
    pd.DataFrame including the 2-hot variables
    '''
    df_out = pd.DataFrame()

    condlist = [
        'Artery',
        'Feedr',
        'Norm',
        'RRNn',
        'RRAn',
        'PosN',
        'PosA',
        'RRNe',
        'RRAe'
    ]

    for x in condlist:
        colname = 'Cond_' + x
        df_out[colname] = ((df_in.Condition1 == x)|(df_in.Condition2 == x)).replace({False:0, True:1})

    return df_out

def clean(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Arguments:
    df - Original dataframe

    Returns:
    Tuple of Dataframes:
    (df_num, df_obj)
    df_num: DataFrame with ordinal features treated and integerized.
    df_obj: Categorical data ready to be treated with OneHotEncoder
    '''
    
    generic_ordinal_features = ['BsmtCond',
                                'BsmtQual',
                                'ExterCond',
                                'ExterQual',
                                'FireplaceQu',
                                'GarageCond',
                                'GarageQual',
                                'HeatingQC',
                                'KitchenQual',
                                'PoolQC']
    df_out = df[generic_ordinal_features].apply(basic_ordinal)
    df.drop(generic_ordinal_features, 1,inplace=True)

    Basement_Finish_Type_Features = ['BsmtFinType1','BsmtFinType2']
    df_out = pd.concat([df_out, BsmntFins(df[Basement_Finish_Type_Features])],axis=1)
    df.drop(Basement_Finish_Type_Features,1,inplace=True)

    Fence_Features = 'Fence'
    df_out = pd.concat([df_out, fence(df[Fence_Features])],axis=1)
    df.drop(Fence_Features,1,inplace=True)

    Functionality_Features = 'Functional'
    Functionality_dict = {'Typ':7,
                         'Min1':6,
                         'Min2':5,
                         'Mod':4,
                         'Maj1':3,
                         'Maj2':2,
                         'Sev':1,
                         'Sal':0}
    Functionality_filler = 7
    df_out = pd.concat([
            df_out,
            basic_ordinal(df[Functionality_Features], 
                          Functionality_dict, 
                          Functionality_filler)
            ],
        axis=1
    )
    df.drop(Functionality_Features,1,inplace=True)
    
    BsmtExposure_dict = {'Gd':4,
                         'Av':3,
                         'Mn':2,
                         'No':1}
    BsmtExposure_filler = 0

    df_out = pd.concat(
        [
            df_out,
            basic_ordinal(
                df.BsmtExposure,
                BsmtExposure_dict,
                BsmtExposure_filler
                )
        ],
        axis=1
    )
    df.drop('BsmtExposure', 1, inplace=True)

    GarageFinish_dict = {'Fin':3,
                         'RFn':2,
                         'Unf':1,
                         'NA':0}
    GarageFinish_filler = 0

    df_out = pd.concat(
        [
            df_out,
            basic_ordinal(
                df.GarageFinish,
                GarageFinish_dict,
                GarageFinish_filler
            )
        ],
        axis=1
    )
    df.drop('GarageFinish',1,inplace=True)

    LandSlope_dict = {'Gtl':1,
                      'Mod':2,
                      'Sev':3}
    LandSlope_filler = 1

    df_out = pd.concat(
        [
            df_out,
            basic_ordinal(
                df.LandSlope,
                LandSlope_dict,
                LandSlope_filler
            )
        ],
        axis=1
    )
    df.drop('LandSlope',1,inplace=True)

    LotShape_dict = {'Reg':0,
                     'IR1':1,
                     'IR2':2,
                     'IR3':3}
    LotShape_filler = 0

    df_out = pd.concat(
        [
            df_out,
            basic_ordinal(
                df.LotShape, 
                LotShape_dict, 
                LotShape_filler
            )
        ], 
        axis=1
    )
    df.drop('LotShape',1,inplace=True)
    
    PavedDrive_dict = {'Y':3,
                       'P':2,
                       'N':1,
                       'Na':0}
    PavedDrive_filler = 0

    df_out = pd.concat(
        [
            df_out,
            basic_ordinal(
                df.PavedDrive,
                PavedDrive_dict,
                PavedDrive_filler
            )
        ], 
        axis=1
    )
    df.drop('PavedDrive',1,inplace=True)

    df_out = pd.concat(
        [
            df_out,
            conditions(df[['Condition1','Condition2']])
        ],
        axis=1
    )
    df.drop(['Condition1','Condition2'], 1, inplace=True)

    overall_feats = ['OverallQual','OverallCond']
    df_out = pd.concat(
        [
            df_out,
            df[overall_feats]
        ],
        axis=1
    )
    df.drop(overall_feats,1,inplace=True)

    df_out['MoSold'] = df.MoSold.apply(str)
    df.drop('MoSold', 1,inplace=True)
    
    objects = [col for col in df.columns.values if df[col].dtype == 'object']
    df_out = pd.concat(
        [
            df_out,
            df[objects].fillna('None')
        ],
        axis = 1
    )
    
    df_obj = df_out.loc[:,(df_out.dtypes == 'object')]
    df_num = df_out.loc[:,~(df_out.dtypes == 'object')]

    df_num = df_num.apply(lambda x: x.apply(np.log1p))

    return (df_num, df_obj)



if __name__ == '__main__':
    df = clean(pd.read_csv('train.csv'))
    print(df.isna().sum())
