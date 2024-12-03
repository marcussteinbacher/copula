import pandas as pd
from arch.univariate import ConstantMean, GARCH, EWMAVariance


def garch_vol(window):
    """
    Pandas rolling callback function: Predicts the GARCH volatility on a cetrain day based on 
    the last len(window) observations.
    If the solver can't converge it uses EWMA(0.94).
    Returns sqrt(var) = std 
    """
    
    assert len(window) == 250

    am = ConstantMean(window,rescale=False)
    am.volatility = GARCH(p=1,q=1)
    res = am.fit(disp=False,show_warning=False)

    #If Convergence not succesful then apply EWMA estimation
    if not res._optim_output.success:
        start, end = window.index[0].strftime("%Y-%m-%d"),window.index[-1].strftime("%Y-%m-%d")
        print(f"No Convergence for GARCH in {start} - {end}. Using EWMA!")
            
        am.volatility = EWMAVariance(0.94)
        res = am.fit(disp=False,show_warning=False)

    return np.sqrt(res.forecast(horizon=1).variance.values[0][0])