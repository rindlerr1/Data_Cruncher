#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:55:52 2019

@author: Home
"""


import pandas as pd
from bokeh.models.widgets import Select, Button, Paragraph, RadioButtonGroup, TextInput, Panel, Tabs, PreText, Slider , RadioGroup
from bokeh.layouts import row, column
from bokeh.plotting import curdoc

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn import metrics

from joblib import dump, load

main_path = '/users/home/desktop/projects/data_cruncher/app/Data/'

df = pd.read_csv(main_path+'app_data.csv')


target_file = pd.read_csv(main_path+'app_data_target_actual.csv')


past_models = pd.read_csv(main_path+'/model_data/model_performance.csv')

past_model_num = len(past_models)
next_model = past_model_num+1


#Section for injecting into db
def update_variables():
    running_table1.text = ""
    running_table2.text = ""
    
    if target_file['Orig_Clean'][0] == 0:
        running_table1.text = 'Please complete file metadata module'
        running_table2.text = 'No model tuning running'
    
    model = model_select.value
    param_tuning = select_tuning.labels[select_tuning.active]
  
    
    
    if target_file['Orig_Clean'][0] == 1:
        df = pd.read_csv(main_path+'app_data_clean.csv')
        df = df.drop(['Date'], axis=1)
        
        running_table1.text = 'Model Beginning'
        cat_table = pd.read_csv(main_path+'cats_defined.csv')
        cols = [x for x in cat_table['Categorical_Variables']]

        
        dep_table = pd.read_csv(main_path+'dep_defined.csv')
        dep_var = dep_table['Dep_Variable'][0]  

        for item in cols:
                df[item] = df[item].astype("category").cat.codes +1
                train, test, y_train, y_test = train_test_split(df.drop([dep_var], axis=1), df[dep_var],
                                                            random_state=10, test_size=0.25)
              

        
        def auc_cat(m, train, test): 
                return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
                        metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))
        
        def auc_xg(m, train, test): 
            return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
                            metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))                
        
        def auc_gbm(m, train, test): 
            return (metrics.roc_auc_score(y_train,m.predict(train)),
                            metrics.roc_auc_score(y_test,m.predict(test)))   
    
        if model == 'CatBoost':   
            
            if param_tuning == 'Manual':
                clf = cb.CatBoostClassifier(eval_metric='AUC',depth=cbdepth.value, iterations= int(cbiterations.labels[cbiterations.active]), 
                                            l2_leaf_reg= cbl2_leaf_reg.value, learning_rate= cblearning_slider.value) 
                clf.fit(train,y_train)           
                #score = auc_cat(clf, train, test)
                dump(clf, '/users/home/desktop/catboost_'+str(next_model)+'.joblib') 
        
            #elif param_tuning == 'Grid Search':
            
            #elif param_tuning == 'Random Search':
            
            
        elif model == 'XGBoost':
            
            if param_tuning == 'Manual':
                        
                clf = xgb.XGBClassifier(max_depth=xgmax_depth.value, min_child_weight=xgmin_child_weight.value,  n_estimators=int(xgn_estimators.labels[xgn_estimators.active]),
                                        n_jobs= xgn_jobs.value , verbose=1,learning_rate=xglearning_rate.value)
                clf.fit(train,y_train)           
                #score = auc_xg(clf, train, test)
                dump(clf, '/users/home/desktop/xgboost_'+str(next_model)+'.joblib') 
        
            #elif param_tuning == 'Grid Search':
            
            #elif param_tuning == 'Random Search':
            
            
        elif model == 'LightGBM':
            
            if param_tuning == 'Manual':
                clf = cb.CatBoostClassifier(eval_metric='AUC',depth=cbdepth.value, iterations= int(cbiterations.labels[cbiterations.active]), l2_leaf_reg= cbl2_leaf_reg.value, learning_rate= cblearning_slider.value) 
                clf.fit(train,y_train)           
                #score = auc_cat(clf, train, test)
                dump(clf, '/users/home/desktop/lightgbm_'+str(next_model)+'.joblib') 
        
            #elif param_tuning == 'Grid Search':
            
            #elif param_tuning == 'Random Search':
        
        
            
            
            
                
           
        if model == 'XGBoost':
            params = str({'Depth':xgmax_depth.value, 
                           'Min Child Weight':xgmin_child_weight.value,  
                           'Num Estimators':int(xgn_estimators.labels[xgn_estimators.active]),
                           'Num Jobs':xgn_jobs.value, 
                           'Learning Rate':xglearning_rate.value} )
        
        #elif model == 'LightGBM':
            #params = str(str(lblearning_rate.value) +'-'+ str(lbmax_depth.value) +'-'+ str(xgreg_lambda.value) +'-'+ str(lbmin_child_weight.value) +'-'+ str(lbn_estimators.value))
        
        
        elif model == 'CatBoost':
            params = str({'Depth' : cbdepth.value, 
                          'Iterations' : int(cbiterations.labels[cbiterations.active]), 
                          'L2 Leaf Reg' : cbl2_leaf_reg.value, 
                          'Learning Rate' : cblearning_slider.value})
                    
    
            
            
            
            
            
        data_dict = {0:{'Model':str(model),
                        'Model #':str(next_model),
                        'Tuning_Type':str(param_tuning),
                        'Parameters':str(params),
                        'Score':str()}}
        model_data = pd.DataFrame.from_dict(data_dict).transpose()
        
        model_data = model_data.append(past_models)
        
        
        model_data.to_csv(main_path+'/model_data/model_performance.csv',index=False)
        
        running_table2.text = 'Model Complete'
        


#Model Selection widgets
model_select = Select(title="Model Selection:", value="XGBoost", options=["XGBoost", "LightGBM", "CatBoost"])



#Selection for parameter tuning technique
select_tuning = RadioButtonGroup(labels=["Manual","Grid Search", "Random Search"], active=0, width=300)
p6 = Paragraph(text='Select parameter tuning technique')
p7 = Paragraph(text="If manual selected please tune parameters below", width=300, height=1)
tuning_select = column(p6, select_tuning)


"""

cat_features_index = [0,1,2,3,4,5,6]

def auc3(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
                            metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))

# params = {'depth': [4, 7, 10],
#           'learning_rate' : [0.03, 0.1, 0.15],
#          'l2_leaf_reg': [1,4,9],
#          'iterations': [300]}
# cb = cb.CatBoostClassifier()
# cb_model = GridSearchCV(cb, params, scoring="roc_auc", cv = 3)
# cb_model.fit(train, y_train)

#With Categorical features
clf = cb.CatBoostClassifier(eval_metric="AUC", depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
clf.fit(train,y_train)
auc3(clf, train, test)

 """
 
 

#selection parameter inputs
#xgboost
xgmax_depth = Slider(start=0, end=50, value=5, step=5, title="Max Depth")
xgmin_child_weight = Slider(start= 0, end = 10, value = 1, step = 1, title = 'Min Child Weight')
xgn_estimators = RadioGroup(labels=["50", "100", "200", "500"], active=0)
xgn_jobs = Slider(start = -1, end =1, value = -1, step = 1, title= 'Number Jobs')
xglearning_rate = Slider(start=0, end=1, value=.2, step=.1, title="Learning Rate")


#lightgbm
lblearning_rate = TextInput(value="default", title="Learning Rate", width= 150)
lbmax_depth = TextInput(value="default", title="Max Depth", width= 150)
lbreg_lambda = TextInput(value="default", title="Lambda", width= 150)
lbmin_child_weight = TextInput(value="default", title="Min Child Weight", width= 150)
lbn_estimators = TextInput(value="default", title="N-Estimators", width= 150)


#catboost
cblearning_slider = Slider(start=0, end=1, value=.2, step=.1, title="Learning Rate")
cbdepth = Slider(start=0, end=50, value=5, step=5, title="Max Depth")
cbiterations = RadioGroup(labels=["50", "100", "200", "500"], active=0)
cbl2_leaf_reg = Slider(start=0, end=20, value=5, step=1, title="L2 Leaf Reg")





#lock model parameters
lock_params = Button(label="Lock Parameters/Run Parameter Tuning Technique", button_type="primary")
lock_params.on_click(update_variables)

#organize paramter details 
xg_col = column(xglearning_rate, xgmax_depth, xgmin_child_weight, xgn_estimators, xgn_jobs, width= 200)
lb_col = column(lblearning_rate, lbmax_depth, lbreg_lambda, lbmin_child_weight, lbn_estimators, width= 200)
cb_col = column(cblearning_slider, cbdepth, cbiterations, cbl2_leaf_reg, width=200)

tab1 = Panel(child = xg_col, title = 'XGBoost')
tab2 = Panel(child=lb_col, title = 'LightGBM')
tab3 = Panel(child=cb_col, title='CatBoost')

#param_row = row(xg_col, lb_col, cb_col)
tabs = Tabs(tabs=[ tab1, tab2 , tab3], width=900, height=300)

running_table1 = PreText(text='', width=900)
running_table2 = PreText(text='', width=900)

top_row = row(column(running_table1, running_table2), height = 125)

col_1 = column(model_select, p7)
col_2 = column(tuning_select)
col_3 = column(lock_params)

main_cols = row(col_1, col_2, col_3)

bottom_row = (tabs)

dash= column(top_row, main_cols, tabs)


curdoc().add_root(dash)
curdoc().title = "Please"


"""
import mysql.connector
#connect to db and pull in dataset
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Izzie722", database='icom_demo')


cursor = mydb.cursor()
query = ("SELECT * FROM file_metadata")
cursor.execute(query)
db_data=cursor.fetchall()

PATH = db_data[len(db_data)-1][1]
#PATH = '/users/home/desktop/data_cruncher/Data/titanic/train.csv'

"""