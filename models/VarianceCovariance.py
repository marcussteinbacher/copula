import pandas as pd
import numpy as np
import math as m
from scipy.stats import norm


def expected_portfolio_mu(df:pd.DataFrame,period:int=250,weights:np.array=None)->pd.Series:
    """
    Calculate the expected portfolio returns of an {weights} weighted portfolio 
    based on the last {period:int} returns.
    If no weights are specified assets are assumed to be equally weighted.
    """
    n_assets = df.shape[1]
    if not weights:
        weights = np.array([1/n_assets for _ in range(n_assets)])
    
    assert m.isclose(np.sum(weights), 1)
    assert len(weights) == n_assets

    #Apply weights columnwise
    mu_pf_daily = df.apply(lambda row: weights.T.dot(row),axis=1)

    #Expected portfolio return from mean of previous {period} observations
    return mu_pf_daily.rolling(period,closed="left").mean()

def expected_quantile(mu:pd.Series,sigma:pd.Series,q:float=0.01,shift:int=1):
    """
    Calculates the expected t+{shift} {q}-% quantile from day t portfolio return {mu} and portfolio
    standard deviation {std}
    """
    return pd.Series(norm.ppf(q,loc=mu,scale=sigma),index=mu.index).shift(shift)