import pandas as pd 
import math
import numpy as np 

def check_for_nan(df):
    '''
    check for any NaN value in the column of imported csv
    df : dataframe to be check
    RETURN list of column header with NaN value
    '''
    nan_columns = []
    for col in df.columns :
        columns_list = df[col].values
        if any([isinstance(ii, basestring) for ii in columns_list]):
            if any(pd.isnull(df[col])):
                nan_columns.append(col)
        else:
            if any(np.isnan(columns_list)):
                nan_columns.append(col)
    return nan_columns

def svm_get_C_Gamma(model):
    '''
    Get C and Gamma value based on classifier or regressor
    Increment 4
    mode : "R" for regressor    2**-20 to 2**20
           "C" for classifier   2**-10 to 2**10
    RETURN list of C and Gamma value for SVM
    '''
    if model == "R" :
        k = [ -10+ii*4 for ii in range(6)]
    elif model == "C" :
        k = [ -20+ii*4 for ii in range(11)]
        print k
    else:
        print "Error in get C, insert R or C!"
        exit(1)
    _C = [2**k[i] for i in range(len(k))]
    _G = [2**k[i] for i in range(len(k))]
    return _C,_G    

def svr_feature_scaler(X):
    '''
    Return scaled feature for regression
    X : unscaled nparray feature
    RETURN scaled feature
    '''
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    rescaled = scaler.fit_transform(X)
    #np.savetxt("./Titanic/rescaled.csv",rescaled,delimiter=",")
    return rescaled

def svm_gen_grid_list(config):
    '''
    Get List of C and Gamma based on configuration of GridSearchWindow
    config : dict of C or Gamma with start,stop,step as key
    RETURN a list of C or Gamma based on start,stop and step
    '''
    cfg_list = [config['start']]
    while cfg_list[-1] < config['stop']:
        cfg_list.append(cfg_list[-1]+config['step'])
    return cfg_list

def svm_update_grid(config,best):
    '''
    update configuration of C and Gamma
    Based on best parameter
    config : dict of C or Gamma with start,stop,step as key
    best :  best C or best Gamma as value
    RETURN updated config of C and Gamma
    '''
    #best = math.log(best,2)
    update = {'start': best - config['step'] ,'stop': best + config['step'], 'step':0.5*config['step']}
    if abs(config['stop']-best) > abs(config['start']-best):
            if abs(config['start']-best) < 1.5*config['step']:
                update['start'] -= 0.5 * config['step']
    elif abs(config['stop']-best) < abs(config['start']-best):
            if abs(config['start']-best) < 1.5*config['step']:
                update['stop'] += 0.5 * config['step']
    return update

def containWord(s,w):
    return (' '+w+' ') in (' '+s+' ')


