import pandas as pd 
import numpy as np
import sys
import math
from sklearn.model_selection import KFold,GridSearchCV,train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.base import TransformerMixin
sys.path.append("./")
from utilitiesData import svm_gen_grid_list,svm_update_grid,svm_get_C_Gamma, svr_feature_scaler,containWord

feature_to_use = ['Fare','Sex','SibSp','Parch','Survived','prefix']
target_to_pd = ['Age']


def split_based_age(df):
    _bol_list = np.isnan(df['Age']).tolist()
    _id_wo_age = [i for i , e in enumerate(_bol_list) if e == True]
    _id_age = list(set(range(len(df))) - set(_id_wo_age))
    _df_wo_age = df.iloc[_id_wo_age,:].reset_index(drop=True)
    _df_age = df.iloc[_id_age,:].reset_index(drop=True)
    return _df_age,_df_wo_age

def map_name_prefix(df):
    name_list = df['Name'].tolist()
    replace_list = []
    for ii in range(len(name_list)):
        if containWord(name_list[ii],"Mr."):
            replace_list.append(1)
        elif containWord(name_list[ii],"Mrs."):
            replace_list.append(2)
        elif containWord(name_list[ii],"Miss."):        
            replace_list.append(3)
        elif containWord(name_list[ii],"Master."):
            replace_list.append(4)
        else:
            replace_list.append(0)
    df = df.drop(columns="Name")
    df['prefix'] = replace_list
    return df

def do_lasso(X,y):
    from sklearn.linear_model import Lasso
    regreLasso = Lasso()
    regreLasso.fit(X,y)
    return regreLasso.coef_

def get_feature_target(df):
    _fv_df = df[feature_to_use].values
    _tg_df = df[target_to_pd].values
    return _fv_df,_tg_df

def test_prmtr_afterCV(X,y,CVResult):
    _best_prmt = [CVResult.loc[0,["param_C"]].values[0],CVResult.loc[0,["param_gamma"]].values[0]]
    _worst_prmt = [CVResult.loc[len(CVResult)-1,["param_C"]].values[0],CVResult.loc[len(CVResult)-1,["param_gamma"]].values[0]]
    _fv_train, _fv_test, _tg_train, _tg_test = train_test_split(X,y_tr,test_size=0.25)
    _list_pred = []
    _score = []
    _prmt_stg = [_best_prmt,_worst_prmt]
    for ii in range(len(_prmt_stg)):
        _reg = SVR(kernel='rbf',gamma=_prmt_stg[ii][1],C=_prmt_stg[ii][0])
        _reg.fit(_fv_train,_tg_train)
        _pred = _reg.predict(_fv_test)
        _score.append(_reg.score)
        _list_pred.append(_pred)    
    _compare = pd.DataFrame({'Predict_B':_list_pred[0],'Predict_W':_list_pred[1],'Target':_tg_test})
    _compare.to_csv("./Titanic/Check_result_GridCV.csv",index=None,header=True)
    print "Best parameter MSE is {}.".format(mean_squared_error(_tg_test,_list_pred[0]))
    print "Worst parameter MSE is {}.".format(mean_squared_error(_tg_test,_list_pred[1]))

def __cv__train(c,g,X,y):
    print "================================================================"
    print "================================================================"
    print "=====================Start SVM CV 1========with K-FOLD=========="
    print "================================================================"
    print "================================================================"
    grid_level = [1,2,3,4,5,6,7,8,9]
    for ii in grid_level :
        print "C grid of {} levels is {}.".format(ii,svm_gen_grid_list(c))
        print "Gamma grid of {} levels is {}.".format(ii,svm_gen_grid_list(g))
        C_value = [ 2.**i for i in svm_gen_grid_list(c)]
        Gamma_value = [ 2.**i for i in svm_gen_grid_list(g)]
        parameters = {'C':C_value,'gamma':Gamma_value}
        svr = SVR(kernel='rbf')
        reg = GridSearchCV(svr,parameters, cv=4, return_train_score=True)
        reg.fit(X,y)
        result_df = pd.DataFrame.from_dict(reg.cv_results_)
        print "Best Parameter after {} level is {}.".format(ii,reg.best_params_)
        print "Scores for Parameter combination above is {}.".format(reg.best_score_)
        result_df.to_csv("./Titanic/CV_result_level_{}.csv".format(ii),index=None,header=True)
        c = svm_update_grid(c,math.log(reg.best_params_['C'],2))
        g = svm_update_grid(g,math.log(reg.best_params_['gamma'],2))
        print "************************************************************************"
    best_cg = {'c':reg.best_params_['C'],'g':reg.best_params_['gamma']}
    return best_cg

def _cal_error(y,py):
    _diff = math.sqrt(sum(pow(y1-y2,2) for y1, y2 in zip(y,py)))
    return _diff

def __cv__train2 (c,g,X,y) :
    print "================================================================"
    print "================================================================"
    print "===================++=Start SVM CV 2============================"
    print "================================================================"
    print "================================================================"
    best_cg = None
    grid_level = [1,2,3,4,5,6,7,8,9]
    resultGrid = []
    for ii in grid_level:
        print "C grid of {} levels is {}.".format(ii,svm_gen_grid_list(c))
        print "Gamma grid of {} levels is {}.".format(ii,svm_gen_grid_list(g))
        _clist = svm_gen_grid_list(c)
        _glist = svm_gen_grid_list(g)

        for _c in _clist:
            _cval = 2**_c
            for _g in _glist:
                _gval = 2**_g
                _reg = SVR(kernel='rbf',C=_cval, gamma=_gval)
                _reg.fit(X,y)
                _pred = _reg.predict(X)
                _error = _cal_error(y, _pred)
                _score = _reg.score(X,y)
                resultGrid.append([ii,_c,_g,_error,_score])
                if best_cg == None:
                    best_cg = {'c': _c, 'g': _g, 'err':_error}
                elif _error < best_cg['err']:
                    best_cg = {'c': _c, 'g': _g, 'err':_error}
        c = svm_update_grid(c,best_cg['c'])
        g = svm_update_grid(g,best_cg['g'])
        print "Best Parameter after {} level is {}.".format(ii,best_cg)
        print "************************************************************************"
        df = pd.DataFrame(resultGrid)
        df.columns = ['Level','C value','Gamma Value','Error','R_square']
        df.to_csv('./Titanic/Result_CV2.csv',index=None,header=True)
    return best_cg

data_csv = pd.read_csv("./Titanic/train.csv", header=0)
data_csv['Sex'] = data_csv['Sex'].map({'male':0, 'female':1}).copy()
data_csv = map_name_prefix(data_csv)

data_df, predict_df = split_based_age(data_csv)
X,y = get_feature_target(data_df)
y_tr = np.ravel(y)
X_scaled = svr_feature_scaler(X)

c = {'start':-10.,'stop':10.,'step':4}
g = {'start':-10.,'stop':10,'step':4}

best_cg1 = __cv__train(c,g,X_scaled,y_tr)
best_cg2 = __cv__train2(c,g,X_scaled,y_tr)

print "First method best config : {}.".format(best_cg1)
print "Second method best config : {}.".format(best_cg2)
'''
result_df = pd.read_csv("./Titanic/CV_result.csv",header=0)
#print result_df
test_prmtr_afterCV(X,y_tr,result_df)
'''