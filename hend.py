    '''
    @author: dhend
    '''

import pandas as pd
def feature_extract(df):
    '''
    This addresses the following variables: exterior1st, exterior2nd, roofmatl, lotarea, lotfrontage, masvnrarea, BsmtFinSF1, BsmtFinSF2, and masvnrtype.
    Exterior2nd, BsmtFinSf1, and BsmtFinSf2 are dropped. The continuous variables have skew applied.
    '''

    #Exterior2nd was nearly identical to Exterior1st, so I dropped it.
    #df.drop(['Exterior2nd'], axis=1, inplace = True)

    #Imputing missing numerical data with the mean and categorical data with the mode.
    df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)
    df['MasVnrArea'].fillna(df['MasVnrArea'].mode(), inplace=True)

    #Determing interquartile range 
    Q1 = df['LotFrontage'].quantile(.25)
    Q3 = df['LotFrontage'].quantile(.75)
    IQR = Q3 - Q1

    #I clipped outliers to be no more than Q3 + 1 1/2 the interquartile range.
    df['LotFrontage'].clip(0,(Q3 + 1.5 * IQR))

    #The same with MasVnrArea.
    Q1 = df['MasVnrArea'].quantile(.25)
    Q3 = df['MasVnrArea'].quantile(.75)
    IQR = Q3 - Q1

    df['MasVnrArea'].clip(0,(Q3 + 1.5 * IQR))

    #Applying log+1 to each numerical category.
    df['LotFrontage'] = np.log(df['LotFrontage'] + 1)
    df['MasVnrArea'] = np.log(df['MasVnrArea'] + 1)
    df['LotArea'] = np.log(df['LotArea'] + 1)

    #Due to the presence of 'TotalBsmtSF' (which is a sum of the below two variables) and sparseness of data, I drop these two.
    #df.drop(['BsmtFinSF1'], axis = 1, inplace=True)
    #df.drop(['BsmtFinSF2'], axis = 1, inplace=True)

    features = df[['Exterior1st', 'RoofMatl', 'MasVnrType', 'MasVnrArea', 'LotFrontage', 'LotArea']]

    #Dummy variables are given for categorical variables.
    ex1st = pd.get_dummies(df['Exterior1st'])
    roof = pd.get_dummies(df['RoofMatl'])
    mast = pd.get_dummies(df['MasVnrType'])

    features = pd.concat([features, ex1st, roof, mast], axis = 1)

    return features


def main():
    df=pd.read_csv('train.csv')
    print(feature_extract(df))
if __name__ == '__main__':
    main()
