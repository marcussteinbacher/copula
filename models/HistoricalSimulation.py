import numpy as np

def quantile(window:np.array,weights=None,alpha=0.01):
    '''
    Returns the empirical {alpha}-quantile of portfolio returns calculated from asset returns in {window}
    '''
    n = window.shape[1]
    
    if not weights:
        weights = np.array([1/n]*n)
    
    mu_pfs = np.sum(weights*window,axis=1)
    
    return np.quantile(mu_pfs,alpha)

def expected_shortfall(window:np.array,weights=None,alpha=0.01):
    '''
    Returns the arithmetic mean of portfolio returns below the {alpha}-level.
    '''
    n = window.shape[1]
    if not weights:
        weights = np.array([1/n]*n)
    
    mu_pfs = np.sum(weights*window,axis=1)
    var = np.quantile(mu_pfs,alpha)
    
    return np.mean(mu_pfs,where=mu_pfs<var)