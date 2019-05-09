#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:55:52 2019

@author: Home
"""


import pandas as pd
import numpy as np

from bokeh.models.widgets import Select, Button, Paragraph, RadioButtonGroup, TextInput, Panel, Tabs, PreText, Slider , RadioGroup, RangeSlider, PreText
from bokeh.layouts import row, column
from bokeh.plotting import curdoc

import catboost as cb
#import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


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
        
        df[dep_var] = (df[dep_var]>10)*1

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
                params = str({'Depth' : max_depth.value, 
                          'Iterations' : int(iteration_num1.labels[iteration_num1.active]), 
                          'L2 Leaf Reg' : min_leaf.value, 
                          'Learning Rate' : learning_rate.value})
                    
    
            
                clf = cb.CatBoostClassifier(eval_metric='AUC',depth=max_depth.value, iterations= int(iteration_num1.labels[iteration_num1.active]), 
                                            l2_leaf_reg= min_leaf.value, learning_rate= learning_rate.value) 
                clf.fit(train,y_train)           
                score = auc_cat(clf, train, test)
                dump(clf, main_path+'/model_data/catboost_'+str(next_model)+'.joblib') 
        
            elif param_tuning == 'Grid Search':
                depths = [x for x in max_depth_r.value] 
                rates = [ x for x in learning_rate_r.value]
                l2_leaves = [x for x in min_leaf_r.value]
                its = [int(iteration_num2.labels[iteration_num2.active])]
                
                
                params = {'depth': depths,
                       'learning_rate' : rates,
                       'l2_leaf_reg': l2_leaves,
                       'iterations': its}
                model_ = cb.CatBoostClassifier()
                cb_model = GridSearchCV(model_, params, scoring="roc_auc", cv = 3)
                
                cb_model.fit(train, y_train)
                params = cb_model.best_params_
                score = auc_cat(cb_model, train, test)
            
            elif param_tuning == 'Random Search':
                depths = [x for x in max_depth_s.value]
                depths.append(max_depth_s.step) 
                depths = np.arange(depths[0], depths[1], depths[2])
                
                rates = [ x for x in learning_rate_s.value]
                rates.append(learning_rate_s.step)
                rates= np.arange(rates[0], rates[1], rates[2])
                if 0 in rates:
                    rates = np.delete(rates, 0)
                
                min_child = [x for x in min_leaf_s.value]
                min_child.append(min_leaf_s.step)
                min_child = np.arange(min_child[0], min_child[1], min_child[2])
                            
                its = [int(iteration_num3.labels[iteration_num3.active])]
            
                param_dist = {'depth': depths,
                               'learning_rate' : rates,
                               'l2_leaf_reg': min_child}                
#                               'iterations': its 
                
#for cat boost iterations is number of trees                
                n_iters = int(len(depths)*len(rates)*len(min_child)/6)
                model_ = model_ = cb.CatBoostClassifier()
                randomized_mse = RandomizedSearchCV(estimator=model_, param_distributions=param_dist, n_iter=n_iters, scoring='neg_mean_squared_error', verbose=1)
                randomized_mse.fit(train, y_train)
                params = randomized_mse.best_params_
                score = randomized_mse.best_score_
            
                
                
            
        elif model == 'XGBoost':
            
            if param_tuning == 'Manual':
                params = str({'Depth':max_depth.value, 
                           'Min Child Weight':min_leaf.value,  
                           'Num Estimators':int(iteration_num1.labels[iteration_num1.active]),
                           'Learning Rate':learning_rate.value} )
        
 #n_estimators number of trees                       
                clf = xgb.XGBClassifier(max_depth=max_depth.value, min_child_weight=min_leaf.value,  n_estimators=int(iteration_num1.labels[iteration_num1.active]),
                                        n_jobs= -1 , verbose=1,learning_rate=learning_rate.value)
                clf.fit(train,y_train)           
                score = auc_xg(clf, train, test)
                dump(clf, main_path+'/model_data/xgboost_'+str(next_model)+'.joblib') 
        
            elif param_tuning == 'Grid Search':
                depths = [x for x in max_depth_r.value] 
                rates = [ x for x in learning_rate_r.value]
                min_child = [x for x in min_leaf_r.value]
                its = [int(iteration_num2.labels[iteration_num2.active])]

                

                model_ = xgb.XGBClassifier()
                param_dist = {"max_depth": depths,
                              "min_child_weight" : min_child,
                              "learning_rate": rates}
#                              "n_estimators": its                
                
                
                grid_search = GridSearchCV(model_, param_grid=param_dist, cv = 3, 
                                   verbose=10, n_jobs=-1)
                grid_search.fit(train, y_train)
                params = grid_search.best_params_
                score = auc_cat(grid_search, train, test)
            

            elif param_tuning == 'Random Search':
                depths = [x for x in max_depth_s.value]
                depths.append(max_depth_s.step) 
                depths = np.arange(depths[0], depths[1], depths[2])
                
                rates = [ x for x in learning_rate_s.value]
                rates.append(learning_rate_s.step)
                rates= np.arange(rates[0], rates[1], rates[2])
                if 0 in rates:
                    rates = np.delete(rates, 0)
                    
                min_child = [x for x in min_leaf_s.value]
                min_child.append(min_leaf_s.step)
                min_child = np.arange(min_child[0], min_child[1], min_child[2])
                            
                its = [int(iteration_num3.labels[iteration_num3.active])]
            
                param_dist = {"max_depth": depths,
                              "min_child_weight" : min_child,
                              "n_estimators": its,
                              "learning_rate": rates}
                
                
                n_iters = int(len(depths)*len(rates)*len(min_child)/6)
                
                model_ = xgb.XGBRegressor()
                randomized_mse = RandomizedSearchCV(estimator=model_, param_distributions=param_dist, n_iter=n_iters, scoring='neg_mean_squared_error', cv=4, verbose=1)
                randomized_mse.fit(train, y_train)
                params = randomized_mse.best_params_
                score = randomized_mse.best_score_
            
        #elif model == 'LightGBM':
            
            #if param_tuning == 'Manual':
                #params = str(str(lblearning_rate.value) +'-'+ str(lbmax_depth.value) +'-'+ str(xgreg_lambda.value) +'-'+ str(lbmin_child_weight.value) +'-'+ str(lbn_estimators.value))
          
                #clf = cb.CatBoostClassifier(eval_metric='AUC',depth=cbdepth.value, iterations= int(cbiterations.labels[cbiterations.active]), l2_leaf_reg= cbl2_leaf_reg.value, learning_rate= cblearning_slider.value) 
                #clf.fit(train,y_train)           
                #score = auc_gbm(clf, train, test)
                #dump(clf, main_path+'/model_data/lightgbm_'+str(next_model)+'.joblib') 
        
            #elif param_tuning == 'Grid Search':
            
            #elif param_tuning == 'Random Search':
        
    
            
        data_dict = {0:{'Model':str(model),
                        'Model #':str(next_model),
                        'Tuning_Type':str(param_tuning),
                        'Parameters':str(params),
                        'Score':str(score)}}
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
#man_title = PreText(text="Manual Tuning")
#grid_title = PreText(text= "Grid Search")
 
 #manual
max_depth =  Slider(start=0, end=16, value=5, step=1, title="Max Depth")
learning_rate = Slider(start=0, end=1, value=.2, step=.1, title="Learning Rate") 
min_leaf = Slider(start= 0, end = 10, value = 1, step = 1, title = 'Min Leaf Weight')


#grid
max_depth_r = RangeSlider(start=0, end=16, value=(5, 9), step=1, title="Max Depth")
min_leaf_r = RangeSlider(start= 0, end = 10, value = (1, 3), step = 1, title = 'Min Child Weight')
learning_rate_r = RangeSlider(start=0, end=1, value=(.2, .6), step=.1, title="Learning Rate")

#random
max_depth_s = RangeSlider(start=0, end=16, value=(0, 16), step=1, title="Max Depth")
min_leaf_s = RangeSlider(start= 0, end = 10, value = (0, 10), step = 1, title = 'Min Child Weight')
learning_rate_s = RangeSlider(start=0, end=1, value=(0, 1), step=.1, title="Learning Rate")




#both
iteration_num1 = RadioGroup( labels=["50", "100", "200", "500"], active=0)
iteration_num2 = RadioGroup( labels=["50", "100", "200", "500"], active=0)
iteration_num3 = RadioGroup( labels=["50", "100", "200", "500"], active=0)



#lock model parameters
lock_params = Button(label="Lock Parameters/Run Parameter Tuning Technique", button_type="primary")
lock_params.on_click(update_variables)


col_man = column(learning_rate, max_depth, iteration_num1, min_leaf, width=200)
col_grid = column(learning_rate_r, max_depth_r, iteration_num2, min_leaf_r, width=200)
col_random = column(learning_rate_s, max_depth_s, min_leaf_s, width=200)


tab1 = Panel(child =col_man, title = 'Manual Tuning')
tab2 = Panel(child =col_grid, title = 'Grid Search Tuning')
tab3 = Panel(child=col_random, title = 'Random Search')

tabs = Tabs(tabs=[tab1, tab2, tab3], width=450, height=300)



running_table1 = PreText(text='', width=900)
running_table2 = PreText(text='', width=900)

top_row = row(column(running_table1, running_table2), height = 125)

col_1 = column(model_select, p7)
col_2 = column(tuning_select)
col_3 = column(lock_params)

main_cols = row(col_1, col_2, col_3)



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