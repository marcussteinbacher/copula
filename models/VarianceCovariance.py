import pandas as pd
import numpy as np
import math as m
from scipy.stats import norm
from scipy.integrate import quad

def mu_pf_expected(window:np.array,weights=None):
    '''
    Calculate the expected portfolio return as the mean of the len(window) daily portfolio returns.
    '''
    n = window.shape[1]

    assert len(window) == 250
    
    if not weights:
        weights = np.array([1/n]*n)
        
    return np.mean(np.sum(window*weights,axis=1))

def sigma_pf_expected(window:np.array,weights=None):
    '''
    Caluclate the expected portfolio standard deviation based on the variance-covariance matrix 
    of the daily returns in {window}.
    '''
    n = window.shape[1]

    assert len(window) == 250
    
    if not weights:
        weights = np.array([1/n]*n)
        
    cov = np.cov(window,rowvar=False)
    
    return np.sqrt(weights.T.dot(cov).dot(weights))

def quantile(window,weights=None,alpha=0.01): #quantiles
    '''
    Returns the {alpha} quantile of portfolio returns in {window}.
    '''
    mu = mu_pf_expected(window,weights=weights)
    sigma = sigma_pf_expected(window,weights=weights)
    
    return norm.ppf(alpha,loc=mu,scale=sigma)

def expected_shortfall(window,weights=None,alpha=0.01):
    '''
    Returns the expected tail return below the {alpha} level quantile integrated from -inf to
    1%-quantile.
    '''
    mu = mu_pf_expected(window,weights=weights)
    sigma = sigma_pf_expected(window,weights=weights)
    var = mu + norm.ppf(alpha)*sigma

    def integrand(x):
        return x*(1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x-mu)**2/(2*sigma**2)))

    I =  1/alpha * quad(integrand,-np.inf,var)[0]
    ES = mu - sigma*norm.pdf(norm.ppf(alpha))/(alpha)

    return I
