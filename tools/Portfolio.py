import pandas as pd
import numpy as np
import math as m

def mu(df:pd.DataFrame,weights:np.array=None)->pd.Series:
    """
    Calculates the portfolio return series for a dataframe of single asset returns with individual {weights}.
    If weights are not specified equal weights are applied.
    """
    n_assets = df.shape[1]

    #Default to equal weights if not specified
    if not weights:
        weights = np.array([1/n]*n_assets)
    #Some sanity checks
    assert m.isclose(np.sum(weights),1)
    assert len(weights) == n_assets

    return df.apply(lambda row: weights.T.dot(row),axis=1)

def std(df:pd.DataFrame,period:int=250,weights:np.array=None)->pd.Series:
    """
    Calculate the daily portfolio standard deviation for day t based on the last {period} returns
    using a variance-covariance matrix.
    If no weights are specified assets are assumed to be equally weighted.
    """
    n_assets = df.shape[1]

    #Default to equal weights if not specified
    if not weights:
        weights = np.array([1/n_assets]*n_assets)

    #Some sanity checks
    assert m.isclose(np.sum(weights),1)
    assert len(weights) == n_assets

    
    def portfolio_std(table):
        """
        Callback for rolling window calculation
        """
        if table.shape != (period,n_assets):
            return np.nan
        else:
            cov = np.cov(table,rowvar=False)
            std = np.sqrt(weights.T.dot(cov).dot(weights))

        return std

    #Rolling portfolio sigma calculation
    sigma_pf = df.rolling(period,method="table",closed="left").apply(portfolio_std,raw=True,engine="numba")
    s = sigma_pf.iloc[:,0]
    s.name=None
    return s