import numpy as np
from scipy.stats import norm, t, genpareto
import rpy2.robjects as r
from rpy2.robjects.packages import importr

r_copula = importr("copula")

r_coef = r.r["coef"]

r_float_vector = r.vectors.FloatVector
r_str_vector = r.vectors.StrVector
r_list_vector = r.vectors.ListVector


def antithetic_variates(sample,method="1-u"):
    if not method:
        return sample
    elif method.lower() == "-z":
        res = np.append(sample,-sample,axis=0)
    elif method.lower() == "1-u":
        res = np.append(sample,1-sample,axis=0)
    else:
        raise NotImplementedError(f"Method {method} not implemented!")
    return res


def ppf_transform(sample:np.array,data:np.array,distribution:str,**fit_kwargs):
    """
    Transforms a {sample} drawn with rCopula from a copula fitted with pseudo observations based on {data}
    back to its original distribution.
    """
    if distribution == "norm":
        mu, sd = data.mean(axis=0), data.std(axis=0)
        trans = norm.ppf(sample,loc=mu,scale=sd)
    
    elif distribution == "t":
        df, mu, sd = [], [], []
        for i in range(data.shape[1]):
            params = t.fit(data[:,i],**fit_kwargs)
            df.append(params[0])
            mu.append(params[1])
            sd.append(params[2])
        trans = t.ppf(sample,df,loc=mu,scale=sd)

    elif distribution == "pareto":
        shape, mu, sd = [], [], []
        for i in range(data.shape[1]):
            params = genpareto.fit(data[:,i],**fit_kwargs)
            shape.append(params[0])
            mu.append(params[1])
            sd.append(params[2])
        trans = genpareto.ppf(sample,shape,loc=mu,scale=sd)

    elif distribution == "emp":
        #trans = np.quantile(data,sample)
        trans = np.array([np.quantile(col,q) for col,q in zip(data.T,sample.T)]).T


    else:
        raise NotImplementedError(f"Not implemented for a {distribution} distribution!")
    return trans


