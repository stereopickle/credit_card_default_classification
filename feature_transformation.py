#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 08:55:24 2020

@author: stereopickles
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import math 
import pickle

def feat_transform(df):
    df.columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    pay_hist_names =[c for c in df if c.startswith('PAY')][0:6]
    bill_amt_names = [c for c in df if c.startswith('BILL_AMT')]
    pay_amt_names = [c for c in df if c.startswith('PAY_AMT')]
    
    payment_hist = pd.melt(df.reset_index(), id_vars = 'index', 
                           value_vars=pay_hist_names).groupby(['index', 'value']).count().unstack()
    payment_hist = payment_hist.fillna(0)
    payment_hist.columns = ['pay_hist_n2', 'pay_hist_n1', 'pay_hist_0', 
                            'pay_hist_1', 'pay_hist_2', 'pay_hist_3', 
                            'pay_hist_4', 'pay_hist_5', 'pay_hist_6', 
                            'pay_hist_7', 'pay_hist_8']
    
    df = df.join(payment_hist)
    cols_to_drop = pay_hist_names[2:]
    
    for i in range(6, 1, -1):
        df[f"BAL_{i}"] = df[f"BILL_AMT{i}"] - df[f"PAY_AMT{i-1}"]
    # average balance
    
    df['AVG_BAL'] = df[[c for c in df if c.startswith('BAL_')]].mean(axis = 1)
    
    cols_to_drop = cols_to_drop + bill_amt_names + ['PAY_AMT5', 'PAY_AMT4', 'PAY_AMT3', 'PAY_AMT2', 'PAY_AMT1'] + [c for c in df if c.startswith('BAL_')]
    
    for i in range(6, 2, -1):
        df[f"BAL_change_{i}"] = df[f"BAL_{i}"] - df[f"BAL_{i-1}"]
    
    df["cum_bal_change"] = df[[c for c in df if c.startswith('BAL_change')]].sum(axis = 1)
    
    cols_to_drop = cols_to_drop + [c for c in df if c.startswith("BAL_change")]
    
    df['final_balance'] = df['BILL_AMT1']
    
    df['final_payment'] = df['PAY_AMT1']
    
    for i in range(1, 7):
        df[f"DEF_{i}"] = np.where(((df[f"BILL_AMT{i}"] >= 35) & (df[f"PAY_AMT{i}"] < 35)) | 
                                  ((df[f"BILL_AMT{i}"] < 35) & (df[f"PAY_AMT{i}"] < df[f"BILL_AMT{i}"])), 1, 0)

    df['N_underpayment'] = df[[c for c in df if c.startswith('DEF')]].sum(axis = 1)
    df['avg_underpayment'] = df[[c for c in df if c.startswith('DEF')]].mean(axis = 1)

    cols_to_drop = cols_to_drop + [c for c in df if c.startswith('DEF')][2:]
    
    for i in range(1, 7):
        df[f"over_lim_{i}"] = np.where(df[f"BILL_AMT{i}"] > df[f"LIMIT_BAL"], 1, 0)    

    df["n_over_lim"] = df[[c for c in df if c.startswith('over_lim')]].sum(axis = 1)

    cols_to_drop = cols_to_drop + [c for c in df if c.startswith('over_lim')]
    
    for i in range(1, 7):
        df[f"percent_use_{i}"] = df[f"BILL_AMT{i}"] / df[f"LIMIT_BAL"]
    
    df['avg_percent_use'] = df[[c for c in df if c.startswith('percent_use')]].mean(axis = 1)
    
    cols_to_drop = cols_to_drop + [c for c in df if c.startswith('percent_use')]
    
    
    for i in range(1, 7):
        df[f"percent_paid_{i}"] = np.where(df[f"BILL_AMT{i}"] > 0, df[f"PAY_AMT{i}"] / df[f"BILL_AMT{i}"], 1)
        
    df['payment_patter_change'] = df['percent_paid_6'] - df['percent_paid_1']
    df['bill_change'] = df['BILL_AMT6'] - df['BILL_AMT1']        
            
    cols_to_drop = cols_to_drop + [c for c in df if c.startswith('percent_paid')]
    
    df['payment_average_p'] = df[[c for c in df if c.startswith('percent_paid')]].mean(axis = 1)
    df['max_bill'] = df[bill_amt_names].max(axis = 1)     
    df['ln_limit_bal'] = np.log(df['LIMIT_BAL'])
    df['low_limit_bal'] = np.where(df['ln_limit_bal'] < 10.5, 1, 0)
    
    cols_to_drop = cols_to_drop + ['LIMIT_BAL']
    
    df_clean = df.drop(cols_to_drop, axis = 1)
    
    categories = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2']
    df_ohe = pd.get_dummies(df_clean, columns = categories)
    
    
    scaler = StandardScaler()
    scaler.fit(df_ohe)
    df_std = pd.DataFrame(scaler.transform(df_ohe), columns= df_ohe.columns)
        
    interactions = pd.read_pickle('interaction.pkl')    
    df_std_intex = df_std.copy()
    
    tmp = interactions.sort_values(by = ['f1'], ascending = False)[0:10][['pair']]
    for i in range(len(tmp)):
        pair = tmp.iloc[0][0]
        df_std_intex[f'{pair[0]}X{pair[1]}'] = df_std_intex[pair[0]] * df_std_intex[pair[1]]
            
    return df_std_intex