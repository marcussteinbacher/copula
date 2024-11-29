import pandas as pd
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
pd.set_option('future.no_silent_downcasting', True)

"""
df_info: Dataframe containing additional information such as ISIN, Name etc. for each company
df_tr: Dataframe containing the total return index of each company by its ISIN
df_sr: Dataframe containing simple returns
df_tr_log: Daraframe containing the log returns
"""

dir = str(Path(__file__).resolve().parent)

with pd.ExcelFile(dir+"/ATX Constituents.xlsx") as xls:
    df_info = pd.read_excel(xls,"ATX50",usecols=[1,2,3,5],header=1,index_col=1,parse_dates=True,date_format="%y-%m-%d")
    df_tr = pd.read_excel(xls,"Totel Retun Indices - 21 Unt",header=5,index_col=0,parse_dates=True,date_format="%y-%m-%d")
    
df_tr.drop("CURRENCY",inplace=True)
df_tr.index = pd.DatetimeIndex(df_tr.index)
df_tr.index.name = "DAY"
df_tr = df_tr.astype(float)

#Renaming column titles with MNEMONIC for easier identification
df_tr.columns = df_tr.columns.map(lambda isin: df_info.loc[isin,"MNEMONIC"])

#Calculating simple and log retuns
df_sr = df_tr.pct_change()
df_tr_log = np.log(df_tr/df_tr.shift(1,fill_value=np.nan))

#Transformation: Use BAWAG for ZUMTOBEL as soon as possible
df_tr_log_red = df_tr_log.copy()
df_tr_log_red.loc[:,"O:BWGPZUS"] = df_tr_log_red.loc[:,"O:BWGP"].fillna(df_tr_log_red.loc[:,"O:ZUS"])
df_tr_log_red.drop(["O:BWGP","O:ZUS"],axis=1,inplace=True)

#Windowed Data
_df_r = pd.read_pickle("data/df_tr_log_red.pkl")
_df_sigma = pd.read_pickle("data/df_sigma_pred.pkl")
_r_windows = sliding_window_view(_df_r,(250,20)).squeeze(1)
_sigma_windows = sliding_window_view(_df_sigma,(250,20)).squeeze(1)
adj_factor_windows = np.array([window[-1]/window for window in _sigma_windows])
adj_return_windows = _r_windows*adj_factor_windows


if __name__ == "__main__":
    print(dir)
    print(df_info.head())
    print(df_tr.head())
    print(df_tr_log.head())