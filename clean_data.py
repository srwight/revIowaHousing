import pandas as pd
import numpy as np

def basic_ordinal(
    sr_in: pd.Series, 
    dictmap={'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'Na':0}, 
    filler:int = 0
) -> pd.Series:
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
    # Define Replacement Dictionaries
    BsmtLiving_Dict = {
        'GLQ':3,
        'ALQ':2,
        'BLQ':1,
        'Rec':np.NaN,
        'LwQ':np.NaN,
        'Unf':np.NaN,
        'Na':np.NaN
    }

    Rec_Dict = {
        'GLQ':np.NaN,
        'ALQ':np.NaN,
        'BLQ':np.NaN,
        'LwQ':np.NaN,
        'Unf':np.NaN,
        'Na':np.NaN,
        'Rec':1
    }
    
    LwQ_Dict = {
        'GLQ':np.NaN,
        'ALQ':np.NaN,
        'BLQ':np.NaN,
        'Rec':np.NaN,
        'Unf':np.NaN,
        'Na':np.NaN,
        'LwQ':1
    }

    Unf_Dict = {
        'GLQ':np.NaN,
        'ALQ':np.NaN,
        'BLQ':np.NaN,
        'Rec':np.NaN,
        'LwQ':np.NaN,
        'Na':np.NaN,
        'Unf':1
    }

    
    # Replace the incoming data based on the above dictionary. 
    # Leave NaN's - they're important
    BsmtFinT1Score = df_in.BsmtFinType1.replace(BsmtLiving_Dict)
    BsmtFinT2Score = df_in.BsmtFinType2.replace(BsmtLiving_Dict)

    BsmtRecT1Score = df_in.BsmtFinType1.replace(Rec_Dict)
    BsmtRecT2Score = df_in.BsmtFinType2.replace(Rec_Dict)

    BsmtLwQT1Score = df_in.BsmtFinType1.replace(LwQ_Dict)
    BsmtLwQT2Score = df_in.BsmtFinType2.replace(LwQ_Dict)

    BsmtUnfT1Score = df_in.BsmtFinType1.replace(Unf_Dict)
    BsmtUnfT2Score = df_in.BsmtFinType2.replace(Unf_Dict)

    # Now shuffle them together by replacing the NaN's in T1 with the values in T2.
    # We can also safely fill the remaining NaN's with zeroes, cast to integer, and
    # rename the series.
      
    BsmtFin = BsmtFinT1Score.fillna(BsmtFinT2Score).fillna(0).apply(int).rename('BsmtFin_Living')
    BsmtRec = BsmtRecT1Score.fillna(BsmtRecT2Score).fillna(0).apply(int).rename('BsmtFin_Rec')
    BsmtLwQ = BsmtLwQT1Score.fillna(BsmtLwQT2Score).fillna(0).apply(int).rename('BsmtFin_LwQ')
    BsmtUnf = BsmtUnfT1Score.fillna(BsmtUnfT2Score).fillna(0).apply(int).rename('BsmtFin_Unf')

    #Concatenate the resultant columns into a single dataFrame for return
    ret = pd.concat(
        [
            BsmtFin,
            BsmtRec,
            BsmtLwQ,
            BsmtUnf
        ], 
        axis=1
    
    )
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
    
    #These dictionaries will be used to split the data into two columns.
    PrvDict = {
        'GdPrv':2, 
        'MnPrv':1,
        'GdWo':0,
        'MnWw':0,
        'NA':0
    }
    WoodDict = {
        'GdPrv':0,
        'MnPrv':0,
        'GdWo':2,
        'MnWw':1,
        'NA':0
    }
    
    # If the column has GdPrv or MnPrv, it will receive a gradient value 
    # for that entry. If it has any other value, it will receive a 0.
    Fence_Privacy = sr_in.replace(PrvDict).fillna(0).rename('Fence_Privacy')

    # If the column has GdWo or GdPrv, it will receive a gradient value 
    # for that entry. If it has any other value, it will receive a 0.
    Fence_Wood = sr_in.replace(WoodDict).fillna(0).rename('Fence_Wood')

    ret = pd.concat(
        [
            Fence_Privacy,
            Fence_Wood
        ], 
        axis=1
    )
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

    # There's no method for 2-hot Encoding, so this is what I did.
    # Start with an empty DataFrame
    df_out = pd.DataFrame()

    # Loop through the conditions strings. Look for each string in
    # Condition1 and Condition2. Use an 'or' test to set the output 
    # column to True if it appears in either of the original columns. 
    # Then replace the Trues and Falses with 1's and 0's respectively.
    for x in condlist:
        colname = 'Cond_' + x
        df_out[colname] = ((df_in.Condition1 == x)|(df_in.Condition2 == x))
        df_out[colname] = df_out[colname].replace({False:0, True:1})

    return df_out

def clean(df:pd.DataFrame) -> pd.DataFrame:
    '''
    This function specifically treats the *ordinal* features of the dataset.
    Its only other function is to split the dataset into sets of nominal and
    numerical data.

    It is assumed that treatment of numerical data will occur in the main thread.
    This is so that saving and retrieving data can occur in the module calling
    this one, rather than in this module.

    Arguments:
    df - Original dataframe

    Returns:
    Tuple of Dataframes:
    (df_num, df_obj)
    df_num: DataFrame with:
        All numerical variables unchanged
        All ordinal features treated and integerized.
    df_obj: Categorical data ready to be treated with OneHotEncoder
    '''
    ## BASIC ORDINALITY
    #region
    #These features all use the same Ordinality keywords
    generic_ordinal_features = [
        'BsmtCond',
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

    # The next several features simply needs a basic ordinality applied using
    # the values indicated.
    Functionality_Features = 'Functional'
    Functionality_dict = {
        'Typ':7,
        'Min1':6,
        'Min2':5,
        'Mod':4,
        'Maj1':3,
        'Maj2':2,
        'Sev':1,
        'Sal':0
    }
    Functionality_filler = 7
    df_out = pd.concat(
        [
            df_out,
            basic_ordinal(df[Functionality_Features], 
                          Functionality_dict, 
                          Functionality_filler)
        ],
        axis=1
    )
    df.drop(Functionality_Features,1,inplace=True)
    
    BsmtExposure_dict = {
        'Gd':4,
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

    GarageFinish_dict = {
        'Fin':3,
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

    LandSlope_dict = {
        'Gtl':1,
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

    LotShape_dict = {
        'Reg':0,
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
    
    PavedDrive_dict = {
        'Y':3,
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
    #endregion
    ##Two-Hot Encoding
    #region
    #These features need to be passed into their specific 2-hot routine
    Basement_Finish_Type_Features = ['BsmtFinType1','BsmtFinType2']
    df_out = pd.concat(
        [
            df_out, 
            BsmntFins(df[Basement_Finish_Type_Features])
        ],
        axis=1
    )
    df.drop(Basement_Finish_Type_Features,1,inplace=True)

    # These also need to be 2-hot encoded. In the future, perhaps a single
    # 2-hot encoding methot could be generated.
    df_out = pd.concat(
        [
            df_out,
            conditions(df[['Condition1','Condition2']])
        ],
        axis=1
    )
    df.drop(['Condition1','Condition2'], 1, inplace=True)

    #endregion
    
    # 'Fence' needs to be split into two gradient columns.
    Fence_Features = 'Fence'
    df_out = pd.concat(
        [
            df_out, 
            fence(df[Fence_Features])
        ],
        axis=1
    )
    df.drop(Fence_Features,1,inplace=True)

    # MoSold needs to be handled as categorical.
    df_out['MoSold'] = df.MoSold.apply(str)
    df.drop('MoSold', 1,inplace=True)
    
    # For all categorical values, it is now safe to replace empty 
    # cells with "None"
    objects = df.loc[:,(df.dtypes == 'object')].columns
    df_out = pd.concat(
        [
            df_out,
            df[objects].fillna('None')
        ],
        axis = 1
    )
    
    df_obj = df_out.loc[:,(df_out.dtypes == 'object')]
    df_num = df_out.loc[:,~(df_out.dtypes == 'object')]

    return (df_num, df_obj)
