#add parent directory to path to enable python to find the module tools on the same level
if __name__ == "__main__":
    import os 
    import sys 
    from pathlib import Path 

    file_parent_path = Path(__file__).parent.parent
    cwd_parent_path = Path(os.getcwd()).parent
    sys.path.append(str(file_parent_path))

import numpy as np
from scipy.stats import ecdf
import rpy2.robjects as r
from rpy2.robjects.packages import importr

from tools.Transformations import antithetic_variates, ppf_transform

from tools.Converter import CONVERTER


r_copula = importr("copula")

r_str_vector = r.vectors.StrVector
r_float_vector = r.vectors.FloatVector

r_coef = r.r["coef"]


class CopulaSimulation:
    COPULAKWARGS = []
    FITKWARGS = ["method"]
    
    def __init__(self,copula,**kwargs):        
        self.copula = copula

        self._copula_kwargs = {key:val for key,val in kwargs.items() if key in self.COPULAKWARGS}
        self._fit_kwargs = {key:val for key,val in kwargs.items() if key in self.FITKWARGS}

        self._fitted = False
    
    @property
    def margins(self):
        return self._margins
    @margins.setter
    def margins(self,value:list[str]):
        self.__py_margins = value
        self._margins = r_str_vector(value)
    @margins.getter 
    def margins(self):
        return self.__py_margins
    
    @property 
    def paramMargins(self):
        return self._paramMargins 
    @paramMargins.setter
    def paramMargins(self,value):
        self._paramMargins = value 
    @paramMargins.getter 
    def paramMargins(self):
        return self._paramMargins

    #def fit(self,data:np.array,pseudo=True):
    def fit(self,data:np.array,pseudo=True,**fit_kwargs):
        """
        When estimating the parameters of a t-Copula without a fixed degree-of-freedom {df} the
        last parameter in the {_params} array represents the estimated value for df: E.g. params[:-1]
        are the `rhos` and params[-1] is the estimated degree-of-freedom.
        """
        assert len(data.shape) == 2, "Too much axes in data; must be 2-dimensional!"
        self.dim = data.shape[1]

        self._data = data

        self._fit_kwargs = {**self._fit_kwargs,**fit_kwargs} #update fit_kwargs if explicitely submitted in method call
        #print(fit_kwargs)

        with CONVERTER():
            if pseudo:
                obs = r_copula.pobs(data)
            else:
                obs = data

            copula = self.copula(dim=self.dim,**self._copula_kwargs)
            fit = r_copula.fitCopula(copula,obs,**self._fit_kwargs)

        self._params = r_coef(fit)
        self._fitted = True

    def predict_empirical(self,n:int=1e6,anti=True):
        assert self._fitted, "No data fitted to the copula yet! Call fit(data,...) first."
        
        rnd = self.predict(n=n,anti=anti)
        simulation = np.quantile(self._data,rnd) 

        return simulation
    
    def predict_parametric(self,n:int=1e6,anti=True,**margin_kwargs):
        """
        Draw {n} samples from the fitted copula re-transformed to the marginals' distributions.
        Uses scipy.stats.t.fit to estimate the parameters of the original margins' distributions
        in the {data} the copula was fit on.
        Each column in {data} is treated as t-distributed with paramteres mu, sd, and df={df}.
        Info: We do not let df vary when estimating the distribution parameters for robustness 
        (see. Venables & Ripley, 2002).
        """   
        #If params are not yet calculated by previously running estimate_params() on the instance
        assert self._fitted, "No data fitted to the copula yet! Call fit(data,...) first."
        
        simulations = self.predict(n=n,anti=anti)

        
        #transformed = ppf_transform(simulations,self._data,distribution="t",f0=3)
        transformed = ppf_transform(simulations,self._data,**margin_kwargs)
        
        return transformed


class normalCopulaSimulation(CopulaSimulation):
    COPULAKWARGS = ["dispstr"]

    def __init__(self,**kwargs):
        self.copula = r_copula.normalCopula
        super().__init__(self.copula,**kwargs)
    
    def predict(self,n=1e6,anti=True):
        """
        Draw {n} random vectors of size dim from the fitted copula.
        Uses the function rCopula from the R-package copula.
        Use antithetic variates with {anti}: default = True.
        Info: Samples can be re-transformed to the margins' distribution the inverse cdf with
        'predict_paramteric'.
        """
        
        assert self._fitted, "No data fitted to the copula yet! Call fit(data,...) first."

        #with CONVERTER():
        #copula = self.copula(self._params,dim=self.dim,**self._copula_kwargs)
        with CONVERTER():
            copula = self.copula(self._params,dim=self.dim,**self._copula_kwargs)
            __sims = r_copula.rCopula(n,copula)

        if anti:
            #simulations = np.append(__sims,1-__sims,axis=0) #Antithetics 1-U for Uniform
            simulations = antithetic_variates(__sims,method="1-U")
        else:
            simulations = __sims 
        
        return simulations
    
    


class tCopulaSimulation(CopulaSimulation):
    COPULAKWARGS = ["dispstr","df","df_fixed"]
    #MARGIN_KWARGS = ["distribution","f0"]

    def __init__(self,weights=None,**kwargs):
        self.copula = r_copula.tCopula
        super().__init__(self.copula,weights=weights,**kwargs)

        if "df" in kwargs.keys() and "df_fixed" in kwargs.keys():
            self.__eval_df = False 
        else:
            self.__eval_df = True

        #self._margin_kwargs = {"distribution":"t","f0":3} #default case: t margins with fixed degree of freedom
        #self._margin_kwargs = {k,v for k,v in kwargs.items() if k in self.MARGIN_KWARGS} #here

    def predict(self,n:int=1e6,anti=True):
        """
        Draw {n} random vectors of size dim from the fitted copula.
        Uses the function rCopula from the R-package copula.
        Use antithetic variates with {anti}: default = True.
        Info: Samples can be re-transformed to the margins' distribution the inverse cdf with
        'predict_paramteric'.
        """
        #If params are not yet calculated by previously running estimate_params() on the instance
        assert self._fitted, "No data fitted to the copula yet! Call fit(data,...) first."

        with CONVERTER():
            if self.__eval_df:
                rho = self._params[:-1]
                df = self._params[-1]
                self._copula_kwargs["df"] = df
            else:
                rho = self._params
        
            copula = self.copula(rho,dim=self.dim,**self._copula_kwargs)
            __sims = r_copula.rCopula(n,copula)
        
        if anti:
            #simulations = np.append(__sims,1-__sims,axis=0) #1-U for univariate (~pobs)
            simulations = antithetic_variates(__sims,method="1-U")
        else:
            simulations = __sims

        return simulations



def normQuantile(window:np.array,param,n:int=1e6,weights=None,shortfall=False,**kwargs)->tuple:
    """
    Returns the 1%-quantile of portfolio returns or the expected shortfall if {shortfall} of a {n}-sized sample drawn from a normal 
    copula that is explicitely paramterized with {param}.
    {param} defines the matrix parameter values (rho's): must be np.array or R FloatVector. 
    Samples are re-scaled to their original marginals' normal distributions in {window}.
    If no {weights} are given we use equal weights.
    """
    #catch-case for windows with nans
    COPULA_KEYS = ["dispstr"] #kwargs for R copula.tCopula()
    MARGIN_KEYS = ["distribution","f0"] #kwargs for Transformations.ppf_transform & scipy.stats.t.fit(), e.g. f0=3 excludes the degree of freedom from margin parameter estimation and sets it to 3

    copula_kwargs = {k:v for k,v in kwargs.items() if k in COPULA_KEYS}
    margin_kwargs = {k:v for k,v in kwargs.items() if k in MARGIN_KEYS}

    if np.any(np.isnan(window)):
        return np.nan
        
    dim = window.shape[1]

    if not weights:
        weights = np.array([1/dim]*dim)
    
    with CONVERTER():
        copula_obj = r_copula.normalCopula(param,dim=dim,**copula_kwargs)
        sample = r_copula.rCopula(n,copula_obj)

    #adding antithetic variates
    simulations = antithetic_variates(sample,method="1-U") #None, 1-U, -Z
    
    #Retransformation to marginals distributions
    simulated_returns = ppf_transform(simulations,window,**margin_kwargs)

    #calculating portfolio returns
    mu = weights*simulated_returns
    mu_pf = mu.sum(axis=1)

    #calculating the 1% quantile
    q = np.quantile(mu_pf,0.01,axis=0)

    #expected shortfall
    es = np.mean(mu_pf,where=mu_pf<q)

    return es if shortfall else q


#def tQuantile(window,param,n=1e6,weights=None,**copula_kwargs):
def tQuantile(window,param,n=1e6,weights=None,shortfall=False,**kwargs):
    """
    Returns the 1%-quantile or the expected shortfall if {shortfall} of portfolio returns of a {n}-sized sample drawn from a t-copula 
    that is explicitely paramterized with {param}. 
    Samples are re-scaled to their original marginals' t-distributions in {window}.
    If no {weights} are given we use equal weights.
    """
    COPULA_KEYS = ["dispstr","df"] #kwargs for R copula.tCopula()
    MARGIN_KEYS = ["distribution","f0"] #kwargs for Transformations.ppf_transform & scipy.stats.t.fit(), e.g. f0=3 excludes the degree of freedom from margin parameter estimation and sets it to 3

    copula_kwargs = {k:v for k,v in kwargs.items() if k in COPULA_KEYS}
    margin_kwargs = {k:v for k,v in kwargs.items() if k in MARGIN_KEYS}

    #catch-case for windows with nans
    if np.any(np.isnan(window)):
        return np.nan
        
    dim = window.shape[1]

    if not weights:
        weights = np.array([1/dim]*dim)
    
    with CONVERTER():
        #Random Vector Generation (uniform)
        copula_obj = r_copula.tCopula(param,dim=dim,**copula_kwargs)
        sample = r_copula.rCopula(n,copula_obj)
    
    #adding antithetic variates
    simulations = antithetic_variates(sample,method="1-U") #Univariate
    
    #Retransform to marginal distributions
    simulated_returns = ppf_transform(simulations,window,**margin_kwargs) #f0=3 doesn't let df vary; see Venables & Ripley (2002)

    #calculating portfolio returns
    mu = weights*simulated_returns
    mu_pf = mu.sum(axis=1)

    #Calculating th 1%-quantile of all simulated portfolio returns
    q = np.quantile(mu_pf,0.01,axis=0)
    
    #Expetced Shortfall
    es = np.mean(mu_pf,where=mu_pf<q)

    return es if shortfall else q


if __name__ == "__main__":
    WRITE = False
    
    import pandas as pd
    from tqdm import tqdm
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from numpy.lib.stride_tricks import sliding_window_view
    import time

    mu_pf_real = pd.read_pickle(str(file_parent_path)+"/data/mu_pf_real.pkl")
    #copula_params = pd.read_pickle("./data/df_normal_copula_mle.pkl").loc[:,"P"].to_numpy()
    copula_params = pd.read_pickle(str(file_parent_path)+"/data/df_normal_copula_mle_params.pkl").to_numpy()
    df_tr_adj = pd.read_pickle(str(file_parent_path)+"/data/df_tr_adj.pkl")

    #Creating window views to iterate over
    window_shape = (250,len(df_tr_adj.columns))
    windows = sliding_window_view(df_tr_adj.to_numpy(),window_shape).squeeze(1)

    start_norm = time.perf_counter()
    #Parallelized Calculation
    with ProcessPoolExecutor() as p:
        var_norm = [q for q in tqdm(p.map(partial(normQuantile,dispstr="un",n=1e3),windows,copula_params),total=len(windows))]
    end_norm = time.perf_counter()

    #align with realized portfolio returns
    pre = np.array([np.nan]*249)

    var_norm_mle = pd.Series(np.append(pre,var_norm),index=mu_pf_real.index)
    var_norm_mle = var_norm_mle.shift(1)

    print(f"Normal copula finished in {end_norm-start_norm} sec.")

    start_t = time.perf_counter()
    with ProcessPoolExecutor() as p:
        var_t =[q for q in tqdm(p.map(partial(tQuantile,dispstr="un",df=3,n=1e3),windows,copula_params),total=len(windows))]
    end_t = time.perf_counter()

    var_t_mle = pd.Series(np.append(pre,var_t),index=mu_pf_real.index)
    var_t_mle = var_t_mle.shift(1)

    print(f"T copula finished in {end_t-start_t} sec.")

    #Write files and metrics to disk
    if WRITE:
        var_norm_mle.to_pickle(str(file_parent_path)+"/data/var_norm_mle.pkl")
        var_t_mle.to_pickle("../data/var_t_mle.pkl")

        with open(str(file_parent_path)+"/data/simulation_metrics.txt","w") as f:
            f.write(f"Normal Copula: {end_norm-start_norm}s \nT Copula: {end_t-start_t}")
