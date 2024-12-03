# -*- coding: utf-8 -*-

from scipy import stats
import numpy as np
from scipy import optimize
from scipy.stats import chi2
import math as m
import pandas as pd
from typing import List, Union, Dict
from rpy2.robjects.packages import importr
from tools.Converter import CONVERTER
import warnings

warnings.filterwarnings("ignore")

r_rugarch = importr("rugarch")

DESCRIPTION = """The duration of time between VaR violations (no-hits) should ideally be independent and not cluster. Under the null hypothesis of a correctly specified risk model, the no-hit duration should have no memory. Since the only continuous distribution which is memory free is the exponential, the test can conducted on any distribution which embeds the exponential as a restricted case, and a likelihood ratio test then conducted to see whether the restriction holds. Following Christoffersen and Pelletier (2004), the Weibull distribution is used with parameter ‘b=1’ representing the case of the exponential."""

def residuals(mu_pf,var,es):
    '''
    Returns the residuals of the realized portfolio returns and the expected tail returns on hit days.
    '''
    hits = simple_hits(mu_pf,var)
    hit_days = hits.loc[hits==True].index
    return es.loc[hit_days] - mu_pf.loc[hit_days]


def zero_mean_test(
    data: pd.DataFrame, true_mu: float = 0, conf_level: float = 0.95
) -> Dict:
    """ Perfom a t-Test if mean of distribution:
         - null hypothesis (H0) = zero
         - alternative hypothesis (H1) != zero
         
        Parameters:
            data (dataframe):   pnl (distribution of profit and loss) or return
            true_mu (float):    expected mean of distribuition
            conf_level (float): test confidence level
        Returns:
            answer (dict):      statistics and decision of the test
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be dataframe.")

    significance = 1 - conf_level

    mean = np.mean(data)
    std = np.std(data, ddof=1)

    t = (mean - true_mu) / (std / np.sqrt(len(data)))
    """p<0.05, 2-tail"""
    t_padrao = stats.t.ppf(1 - round(significance / 2, 4), len(data) - 1)
    pvalue = stats.ttest_1samp(data, popmean=true_mu, alternative="two-sided")[-1]
    H0 = "Mean of distribution = 0"
    if pvalue > significance:  # ou t < np.abs(t_padrao):
        decision = "Fail to rejected H0."
    else:
        decision = "Reject H0."

    answer = {
        "null hypothesis": H0,
        "decision": decision,
        "t-test statistc": t,
        "t-tabuladed": t_padrao,
        "p-value": pvalue,
    }

    return answer


def duration_test(
    violations: Union[List[int], np.ndarray, pd.Series, pd.DataFrame],
    conf_level: float = 0.95,
) -> Dict:

    """Perform the Christoffersen and Pelletier Test (2004) called Duration Test.
        The main objective is to know if the VaR model responds quickly to market movements
         in order to do not form volatility clusters.
        Duration is time betwenn violations of VaR.
        This test verifies if violations has no memory i.e. should be independent.

        Parameters:
            violations (series): series of violations of VaR
            conf_level (float):  test confidence level
        Returns:
            answer (dict):       statistics and decision of the test
    """
    typeok = False
    if isinstance(violations, pd.core.series.Series) or isinstance(
        violations, pd.core.frame.DataFrame
    ):
        violations = violations.values.flatten()
        typeok = True
    elif isinstance(violations, np.ndarray):
        violations = violations.flatten()
        typeok = True
    elif isinstance(violations, list):
        typeok = True
    if not typeok:
        raise ValueError("Input must be list, array, series or dataframe.")

    N = int(sum(violations))
    first_hit = violations[0]
    last_hit = violations[-1]

    duration = [i + 1 for i, x in enumerate(violations) if x == 1]

    D = np.diff(duration)

    TN = len(violations)
    C = np.zeros(len(D))

    if not duration or (D.shape[0] == 0 and len(duration) == 0):
        duration = [0]
        D = [0]
        N = 1

    if first_hit == 0:
        C = np.append(1, C)
        D = np.append(duration[0], D)  # days until first violation

    if last_hit == 0:
        C = np.append(C, 1)
        D = np.append(D, TN - duration[-1])

    else:
        N = len(D)

    def likDurationW(x, D, C, N):
        b = x
        a = ((N - C[0] - C[N - 1]) / (sum(D ** b))) ** (1 / b)
        lik = (
            C[0] * np.log(pweibull(D[0], a, b, survival=True))
            + (1 - C[0]) * dweibull(D[0], a, b, log=True)
            + sum(dweibull(D[1 : (N - 1)], a, b, log=True))
            + C[N - 1] * np.log(pweibull(D[N - 1], a, b, survival=True))
            + (1 - C[N - 1]) * dweibull(D[N - 1], a, b, log=True)
        )

        if np.isnan(lik) or np.isinf(lik):
            lik = 1e10
        else:
            lik = -lik
        return lik

    # When b=1 we get the exponential
    def dweibull(D, a, b, log=False):
        # density of Weibull
        pdf = b * np.log(a) + np.log(b) + (b - 1) * np.log(D) - (a * D) ** b
        if not log:
            pdf = np.exp(pdf)
        return pdf

    def pweibull(D, a, b, survival=False):
        # distribution of Weibull
        cdf = 1 - np.exp(-((a * D) ** b))
        if survival:
            cdf = 1 - cdf
        return cdf

    optimizedBetas = optimize.minimize(
        likDurationW, x0=[2], args=(D, C, N), method="L-BFGS-B", bounds=[(0.001, 10)]
    )

    print(optimizedBetas.message)

    b = optimizedBetas.x
    uLL = -likDurationW(b, D, C, N)
    rLL = -likDurationW(np.array([1]), D, C, N)
    LR = 2 * (uLL - rLL)
    LRp = 1 - chi2.cdf(LR, 1)

    H0 = "Duration Between Exceedances have no memory (Weibull b=1 = Exponential)"
    # i.e. whether we fail to reject the alternative in the LR test that b=1 (hence correct model)
    if LRp < (1 - conf_level):
        decision = "Reject H0"
    else:
        decision = "Fail to Reject H0"

    answer = {
        "dur.b": b, #weibull exponential
        "dur.uLL": uLL, #unrestricted log-likelihood value
        "dur.rLL": rLL, # restricted log-likelihood value
        "dur.LRstat": LR, # likelihood ratio
        "dur.LRp": LRp, # likelihood ratio test statistic
        "dur.H0": H0,
        "dur.Decision": decision,
    }

    return answer


def failure_rate(violations: Union[List[int], pd.Series, pd.DataFrame]) -> Dict:
    if isinstance(violations, pd.core.series.Series) or isinstance(
        violations, pd.core.frame.DataFrame
    ):
        N = violations.sum()
    elif isinstance(violations, List) or isinstance(violations, np.ndarray):
        N = sum(violations)
    else:
        raise ValueError("Input must be list, array, series or dataframe.")
    TN = len(violations)

    answer = {"failure rate": N / TN}
    print(f"Failure rate of {round((N/TN)*100,2)}%")
    return answer


def simple_hits(actual:pd.Series,var:pd.Series)->pd.Series:
    """
    Returns a boolean Series of hits for {a<b} on overlapping dates that are not nan.
    """
    index = actual.dropna().index.intersection(var.dropna().index)

    return actual.loc[index]<var.loc[index]


def kupiec_test(
    var_level:float,
    violations: pd.Series,
    conf_level: float = 0.95,
) -> pd.Series:

    """Perform Kupiec Test (1995).
       The main goal is to verify if the number of violations, i.e. proportion of failures, is consistent with the
       violations predicted by the model.
       
        Parameters:
            violations (series):    series of violations of VaR
            var_level (float):      VaR  level
            conf_level (float):     test confidence level
        Returns:
            answer (dict):          statistics and decision of the test
    """

    alpha = var_level
    n = len(violations)
    I_alpha = sum(violations)
    alpha_hat = I_alpha/n

    LR = 2*m.log(((1-alpha_hat)/(1-alpha))**(n-I_alpha)*(alpha_hat/alpha)**I_alpha)

    critical_chi_square = chi2.ppf(conf_level, 1)  # one degree of freedom

    LRp = 1 - chi2.cdf(LR, 1)

    if LR > critical_chi_square:
        decision = "Reject H0"
    else:
        decision = "Fail to Reject H0"

    return pd.Series({
        "uc.H0": "Correct Exceedances",
        "uc.LRstat": LR,
        "uc.critical": critical_chi_square,
        "uc.LRp": LRp,
        "uc.Decision": decision,
    })

def VaRTest(alpha,actual:pd.Series,var:pd.Series,conf_level:float=0.95)-> pd.Series:
    """
    The test implements both the unconditional (Kupiec) and conditional (Christoffersen) 
    coverage tests for the correct number of exceedances.
    {q}: Quantile used for VaR
    {actual}: Series of realized values (observations)
    {var}: Series of alpha-% quantiles of the returns
    {conf_level}: Confidence level at wich the Null hypothsis is evaluated
    See. R package rucgarch.VaRTest.

    Christoffersen, P. (1998), Evaluating Interval Forecasts, International Economic Review, 39, 841–862.
    Christoffersen, P., Hahn,J. and Inoue, A. (2001), Testing and Comparing Value-at-Risk Measures, Journal of Empirical Finance, 8, 325–342.
    """
    index = actual.dropna().index.intersection(var.dropna().index)

    _a = actual.loc[index].to_numpy()
    _b = var.loc[index].to_numpy()

    assert len(_a) == len(_b)

    with CONVERTER():
        test = r_rugarch.VaRTest(alpha,_a,_b,conf_level=conf_level)
    
    test_values = [val[0] for _,val in test.items()]

    s = pd.Series(test_values,index=test.keys())
    s.loc["n"] = len(_a)
    return s

def christofferson_test(
    violations: pd.Series,
    conf_level: float = 0.95,
) -> pd.Series:

    df = pd.DataFrame(columns=["T","T+1"])
    df.loc[:,"T"] = violations.values[1:]
    df.loc[:,"T+1"] = violations.values[:-1]

    def nij(i:bool,j:bool):
        """"
        Returns the number of consecutive days on which j follows i
        """
        return len(df[(df["T"]==i) & (df["T+1"] == j)])

    n = len(violations)
    n00 = nij(False,False)
    n01 = nij(False,True)
    n10 = nij(True,False)
    n11 = nij(True,True)

    pi01 = n01/(n00+n01)
    pi11 = n11/(n10+n11)
    pi = (n01+n11)/(n00+n01+n10+n11)

    LR = -2*m.log(
        ((1-pi)**(n00+n10) * pi**(n01+n11) )
        /
        ((1-pi01)**n00 * pi01**n01 * (1-pi11)**n10 * pi11**n11)
    )

    critical_chi_square = chi2.ppf(conf_level, 1)  # one degree of freedom

    LRp = 1.0 - chi2.cdf(LR, 1)

    if LR > critical_chi_square:
        decision = "Reject H0"
    else:
        decision = "Fail to Reject H0"

    return pd.Series({
        "ind.H0": "Independent",
        "ind.LRstat": LR,
        "ind.critical": critical_chi_square,
        "ind.LRp": LRp,
        "ind.Decision": decision,
    })




