#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:55:52 2019

@author: Home


df[item] = df[item].astype("category").cat.codes +1
train, test, y_train, y_test = train_test_split(df.drop([dep_var], axis=1), df[dep_var],
                                                            random_state=10, test_size=0.25)

for i in df.columns:
    df[i] = df[i].astype(float)

train = df[df.index <=103]   
test = df[df.index >103]    

y_train = train['y'] 
train = train.drop(['y'],axis=1)   

y_test = test['y']
test  = test.drop(['y'], axis=1)   
"""

import pandas as pd
import numpy as np
import os

from bokeh.models.widgets import Select, Button, Paragraph, RadioButtonGroup, Panel, Tabs,  Slider , RadioGroup, RangeSlider, PreText, TextInput
from bokeh.layouts import row, column
from bokeh.plotting import curdoc

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals.joblib import parallel_backend

from joblib import dump

main_path = os.getcwd()
main_path = main_path+'/Data/'

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
              
    
        if model == 'CatBoost':   
            
            cat_feature_index = []
            for i in range(0, len(train.columns)):
                if train.columns[i] in cols:
                    cat_feature_index.append(i)
                                                            
            
            if param_tuning == 'Manual':
                
                cat_depth = 0
                if max_depth.value >16:
                    cat_depth = 16
                else:
                    cat_depth = max_depth.value
                
                params = str({'Depth' : cat_depth,
                          'Learning Rate' : learning_rate.value,
                          'N_Estimators': n_estimators.value,
                          'l2_leaf_reg': reg_lambda.value,
                          "RSM": rsm.value})

                catboost = cb.CatBoostRegressor(eval_metric='RMSE',
                                                one_hot_max_size=31,
                                                depth=cat_depth, 
                                                n_estimators = n_estimators.value,
                                                learning_rate= learning_rate.value,
                                                l2_leaf_reg = reg_lambda.value,
                                                rsm = rsm.value,
                                                random_state = int(random_state.value)) 
                
                
                catboost.fit(train,y_train, cat_features = cat_feature_index, silent=True) 
                
                y_pred = catboost.predict(test)
                mse = metrics.mean_squared_error(y_test, y_pred)
                
                
                score = (metrics.r2_score(y_train, catboost.predict(train)),metrics.r2_score(y_test, catboost.predict(test)))
                                
                
                dump(catboost, main_path+'/model_data/CatBoost_'+str(next_model)+'.joblib') 
                
        
            elif param_tuning == 'Grid Search':
                depths = [x for x in max_depth_r.value] 
                
                cat_depth = 0
                if depths[1] > 16:
                    depths[1] = 16
                
                rates = [ x for x in learning_rate_r.value]
                trees = [x for x in n_estimators_r.value]
                lambdas = [x for x in reg_lambda_r.value]
                rsms = [x for x in rsm_r.value]
                
                params = {'depth': depths,
                       'learning_rate' : rates,
                       'n_estimators':trees,
                       'l2_leaf_reg': lambdas,
                       'rsm': rsms}
                
                catboost = cb.CatBoostRegressor()
                
                
                
                grid_search = GridSearchCV(catboost, 
                                        params, 
                                        scoring="neg_mean_squared_error", 
                                        cv = 3)
                                
                grid_search.fit(train, y_train, cat_features = cat_feature_index, silent = True)
                
                                
                params = grid_search.best_params_
                mse = grid_search.best_score_               
                
                best_catboost = cb.CatBoostRegressor(eval_metric='RMSE',
                                                     one_hot_max_size=31,
                                                     depth=params['depth'], 
                                                     n_estimators = params['n_estimators'],
                                                     learning_rate= params['learning_rate'],
                                                     l2_leaf_reg = params['l2_leaf_reg'],
                                                     rsm = params['rsm'],
                                                     random_state = int(random_state.value)) 
                
                best_catboost.fit(train, y_train)                                
                
                score = (metrics.r2_score(y_train, best_catboost.predict(train)),metrics.r2_score(y_test, best_catboost.predict(test)))
                                                
                dump(grid_search, main_path+'/model_data/CatBoost_'+str(next_model)+'.joblib') 

            
            elif param_tuning == 'Random Search':
                running_table2.text = "Error cannot run Random Search and CatBoost"
                """
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
                            
                its = [round(int(iteration_num3.labels[iteration_num3.active]) / 3)]
            
                param_dist = {'depth': depths,
                               'learning_rate' : rates,
                               'l2_leaf_reg': min_child,               
                               'iterations': [20] }
                
#for cat boost iterations is number of trees  
#iterations here is mutiplicative against n_iters
#basically youre doing 20 tress * its[0] * num folds
                
                
                
                
                
                
                model_ = cb.CatBoostClassifier()
                
                
                randomized_mse = RandomizedSearchCV(estimator=model_, 
                                                    param_distributions=param_dist, 
                                                    n_iter= its[0], 
                                                    scoring='roc_auc')
                
                randomized_mse.fit(train, y_train)
                
                params = randomized_mse.best_params_
                
                score = auc_cat(randomized_mse, train, test)
                dump(randomized_mse, main_path+'/model_data/CatBoost_'+str(next_model)+'.joblib') 
                """         
                
            
        elif model == 'XGBoost':
            
            if param_tuning == 'Manual':
                
                params = str({'Depth':max_depth.value, 
                           'Min Child Weight':min_child_weight.value,  
                           'Num Estimators':n_estimators.value,
                           'Learning Rate':learning_rate.value,
                           'Reg Lambda': reg_lambda.value,
                           "colsample_bytree": colsample_bytree.value,
                           "Random State":int(random_state.value)} )
         
                xgboost = xgb.XGBRegressor(max_depth=max_depth.value, 
                                           min_child_weight= min_child_weight.value,  
                                           n_estimators=n_estimators.value,
                                           reg_lambda = reg_lambda.value,
                                           colsample_bytree= colsample_bytree.value,
                                           random_state= random_state.value,
                                           n_jobs= -1 , 
                                           verbose=1,
                                           learning_rate=learning_rate.value)

                
                xgboost.fit(train,y_train) 
                
                y_pred = xgboost.predict(test)
                mse = metrics.mean_squared_error(y_test, y_pred)
                
                score = (xgboost.score(train, y_train),xgboost.score(test, y_test) )
                
                #out = pd.DataFrame(pd.Series(xgboost.predict(test)).rename('Predicted'))
                #out.to_csv('/Users/Home/out_pred.csv',index=False)
                #y_test.to_csv('/Users/Home/tested.csv',index=False)
                #test.to_csv('/Users/Home/Desktop/input_tested.csv', index=False)                
                                           
                dump(xgboost, main_path+'/model_data/XGBoost_'+str(next_model)+'.joblib') 
      
           
            
            elif param_tuning == 'Grid Search':
                depths = [x for x in max_depth_r.value] 
                rates = [ x for x in learning_rate_r.value]
                min_child = [x for x in min_child_weight_r.value]
                trees = [x for x in n_estimators_r.value]
                lambdas = [x for x in reg_lambda_r.value]
                colsamples = [x for x in colsample_bytree_r.value]
                

                xgboost = xgb.XGBRegressor()
                
                param_dist = {"max_depth": depths,
                              "min_child_weight" : min_child,
                              "learning_rate": rates,
                              "n_estimators": trees,
                              'reg_lambda': lambdas,
                              "colsample_bytree":colsamples,
                              "Random State": [int(random_state_r.value)]}               
                                
                grid_search = GridSearchCV(xgboost, 
                                           param_grid=param_dist, 
                                           cv = 3, 
                                           verbose=10, 
                                           n_jobs=-1,
                                           scoring='neg_mean_squared_error')                
                
                with parallel_backend('threading'):
                    grid_search.fit(train, y_train)
                    
                params = grid_search.best_params_
                mse = grid_search.best_score_  
                
                best_xgboost = xgb.XGBRegressor(max_depth=params['max_depth'],                                           
                                                min_child_weight= params['min_child_weight'],  
                                                n_estimators=params['n_estimators'],
                                                reg_lambda = params['reg_lambda'],
                                                colsample_bytree= params['colsample_bytree'],
                                                random_state= params['Random State'],
                                                n_jobs= -1 , 
                                                verbose=1,
                                                learning_rate=params['learning_rate'])
                

                best_xgboost.fit(train, y_train)
                
                score = (best_xgboost.score(train, y_train), best_xgboost.score(test, y_test))
                                                
                dump(best_xgboost, main_path+'/model_data/XGBoost_'+str(next_model)+'.joblib') 
 
                
            elif param_tuning == 'Random Search':
                depths = [x for x in max_depth_s.value]
                depths.append(max_depth_s.step) 
                depths = np.arange(depths[0], depths[1], depths[2])
                
                rates = [ x for x in learning_rate_s.value]
                rates.append(learning_rate_s.step)
                rates= np.arange(rates[0], rates[1], rates[2])
                if 0 in rates:
                    rates = np.delete(rates, 0)
                    
                min_child = [x for x in min_child_weight_s.value]
                min_child.append(min_child_weight_s.step)
                min_child = np.arange(min_child[0], min_child[1], min_child[2])
                            
                trees = [x for x in n_estimators_s.value]
                trees.append(n_estimators_s.step)
                trees = np.arange(trees[0], trees[1], trees[2])
                
                
                lambdas = [x for x in reg_lambda_s.value]
                lambdas.append(reg_lambda_s.step)
                lambdas = np.arange(lambdas[0], lambdas[1], lambdas[2])
                
                col_samples = [x for x in colsample_bytree_s.value]
                col_samples.append(colsample_bytree_s.step)
                col_samples = np.arange(col_samples[0], col_samples[1], col_samples[2])
            
                param_dist = {"max_depth": depths,
                              "min_child_weight" : min_child,
                              "n_estimators": trees,
                              "learning_rate": rates,
                              'reg_lambda': lambdas,
                              "colsample_bytree":col_samples,
                              "Random State":[int(random_state_s.value)],
                              'num_boost_round':[100]}
                
                          
                xgboost = xgb.XGBRegressor()
                
                randomized_mse = RandomizedSearchCV(estimator=xgboost, 
                                                    param_distributions=param_dist, 
                                                    n_iter=int(iteration_num3.labels[iteration_num3.active]), 
                                                    scoring='neg_mean_squared_error', 
                                                    cv=4, 
                                                    verbose=1)
                
                
                
                
                with parallel_backend('threading'):
                    randomized_mse.fit(train, y_train)
                
                params = randomized_mse.best_params_
                
                
                best_xgboost = xgb.XGBRegressor(max_depth=params['max_depth'],                                           
                                                min_child_weight= params['min_child_weight'],  
                                                n_estimators=params['n_estimators'],
                                                reg_lambda = params['reg_lambda'],
                                                colsample_bytree= params['colsample_bytree'],
                                                random_state= params['Random State'],
                                                n_jobs= -1 , 
                                                verbose=1,
                                                learning_rate=params['learning_rate'])
                

                best_xgboost.fit(train, y_train)
                
                score = (best_xgboost.score(train, y_train), best_xgboost.score(test, y_test))
                mse = randomized_mse.best_score_
                #out = pd.Series(best_xgboost.predict(train)) 
                
                
                dump(best_xgboost, main_path+'/model_data/XGBoost_'+str(next_model)+'.joblib') 
                                
#start here             
            
        elif model == 'LightGBM':
            
            if param_tuning == 'Manual':                   
                params = {"max_depth": max_depth.value,
                          "learning_rate" : learning_rate.value, 
                          'Num Estimators':n_estimators.value,
                          "min_data_child_weight": min_child_weight.value,
                          'Reg Lambda': reg_lambda.value,
                          "colsample_bytree":colsample_bytree.value,
                          "Random State":[int(random_state.value)]}

                lightgbm = lgb.LGBMRegressor(boosting_type = "gbdt",
                                             max_depth = max_depth.value,
                                             learning_rate = learning_rate.value, 
                                             n_estimators=n_estimators.value,
                                             min_child_weight= min_child_weight.value,
                                             reg_lambda = reg_lambda.value,
                                             colsample_bytree = colsample_bytree.value,
                                             random_state = random_state.value,
                                             silent=False)                
                
                lightgbm.fit(train, y_train)
                
                y_pred = lightgbm.predict(test)
                mse = metrics.mean_squared_error(y_test, y_pred)
                
                score = (lightgbm.score(train, y_train), lightgbm.score(test, y_test))

                params =str(params)                
                            
                dump(lightgbm, main_path+'/model_data/LightGBM_'+str(next_model)+'.joblib') 




            
            elif param_tuning == 'Grid Search':
                depths = [x for x in max_depth_r.value] 
                rates = [ x for x in learning_rate_r.value]
                min_child = [x for x in min_child_weight_r.value]
                trees = [x for x in n_estimators_r.value]
                lambdas = [x for x in reg_lambda_r.value]
                colsamples = [x for x in colsample_bytree_r.value]
            
                param_dist = {"max_depth": depths,
                              "learning_rate" : rates, 
                              "n_estimators": trees,
                              "min_child_weight": min_child,
                              'reg_lambda': lambdas,
                              'colsample_bytree':colsamples,
                              "Random State":[int(random_state_r.value)]}
          
                
                
                lightgbm = lgb.LGBMRegressor(silent=False)
                
                
                grid_search = GridSearchCV(lightgbm, n_jobs=-1, 
                                           param_grid=param_dist, 
                                           scoring="neg_mean_squared_error", 
                                           verbose=5)
                
                
                with parallel_backend('threading'):
                    grid_search.fit(train,y_train)
                
                params = grid_search.best_params_
                mse = grid_search.best_score_  
                
                best_lightgbm = lgb.LGBMRegressor(max_depth=params['max_depth'],                                           
                                                min_child_weight= params['min_child_weight'],  
                                                n_estimators=params['n_estimators'],
                                                reg_lambda = params['reg_lambda'],
                                                colsample_bytree= params['colsample_bytree'],
                                                random_state= params['Random State'],
                                                n_jobs= -1 , 
                                                verbose=1,
                                                learning_rate=params['learning_rate'])
                

                best_lightgbm.fit(train, y_train)
                
                score = (best_lightgbm.score(train, y_train), best_lightgbm.score(test, y_test))
                                
                
                
                dump(best_lightgbm, main_path+'/model_data/LightGBM_'+str(next_model)+'.joblib') 


            
            elif param_tuning == 'Random Search':
                depths = [x for x in max_depth_s.value]
                depths.append(max_depth_s.step) 
                depths = np.arange(depths[0], depths[1], depths[2])
                
                rates = [ x for x in learning_rate_s.value]
                rates.append(learning_rate_s.step)
                rates= np.arange(rates[0], rates[1], rates[2])
                if 0 in rates:
                    rates = np.delete(rates, 0)
                    
                min_child = [x for x in min_child_weight_s.value]
                min_child.append(min_child_weight_s.step)
                min_child = np.arange(min_child[0], min_child[1], min_child[2])
                            
                trees = [x for x in n_estimators_s.value]
                trees.append(n_estimators_s.step)
                trees = np.arange(trees[0], trees[1], trees[2])
                
                lambdas = lambdas = [x for x in reg_lambda_s.value]
                lambdas.append(reg_lambda_s.step)
                lambdas = np.arange(lambdas[0], lambdas[1], lambdas[2])
                
                col_samples = [x for x in colsample_bytree_s.value]
                col_samples.append(colsample_bytree_s.step)
                col_samples = np.arange(col_samples[0], col_samples[1], col_samples[2])

                
                param_dist = {"max_depth": depths,
                              "learning_rate" : rates, 
                              "n_estimators": trees,
                              "min_child_weight": min_child,
                              'reg_lambda': lambdas,
                              "colsample_bytree": col_samples,
                              "Random State":[int(random_state_s.value)],
                              'num_boost_round':[100]}
                
                lightgbm = lgb.LGBMRegressor(silent=False)
                
                
                randomized_mse = RandomizedSearchCV(estimator=lightgbm, 
                                param_distributions=param_dist, 
                                n_iter= int(iteration_num3.labels[iteration_num3.active]), 
                                scoring='neg_mean_squared_error', 
                                cv=4, 
                                verbose=10,
                                n_jobs=-1)
                
                with parallel_backend('threading'):
                    randomized_mse.fit(train, y_train)
                
                params = randomized_mse.best_params_
                mse = randomized_mse.best_score_
                
                best_lightgbm = lgb.LGBMRegressor(max_depth=params['max_depth'],                                           
                                                min_child_weight= params['min_child_weight'],  
                                                n_estimators=params['n_estimators'],
                                                reg_lambda = params['reg_lambda'],
                                                colsample_bytree= params['colsample_bytree'],
                                                random_state= params['Random State'],
                                                n_jobs= -1 , 
                                                verbose=1,
                                                learning_rate=params['learning_rate'])
                

                best_lightgbm.fit(train, y_train)
                
                score = (best_lightgbm.score(train, y_train), best_lightgbm.score(test, y_test))
                   

                dump(best_lightgbm, main_path+'/model_data/LightGBM_'+str(next_model)+'.joblib') 
    
        
        
        
        
        data_dict = {0:{'Model':str(model),
                        'Model #':str(next_model),
                        'Tuning_Type':str(param_tuning),
                        'Parameters':str(params),
                        'Score':str(score),
                        'MSE': str(mse)}}
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
#update max depth to be 1000
 
 #manual
max_depth =  Slider(start=0, end=16, value=5, step=1, title="Max Depth")
learning_rate = Slider(start=.1, end=1, value=.2, step=.1, title="Learning Rate") 
min_child_weight = Slider(start= 0, end = 10, value = 1, step = 1, title = 'Min Child Weight *Note this does not apply to CatBoost')
n_estimators = Slider(start= 0, end = 500, value = 50, step = 25, title = 'Number of Trees')
reg_lambda = Slider(start=.1, end=1, value=.2, step=.1, title="Reg Lambda aka L2 Lambda")
colsample_bytree = Slider(start=.1, end=1, value=.2, step=.1, title="Col-Sample by Tree")
random_state = TextInput(value="42", title="Random State *Note this does not apply to CatBoost")
rsm = Slider(start=.1, end=1, value=.2, step=.1, title="RSM *Note this ONLY applies to CatBoost") 

#grid
max_depth_r = RangeSlider(start=0, end=16, value=(5, 9), step=1, title="Max Depth")
min_child_weight_r = RangeSlider(start= 0, end = 10, value = (1, 3), step = 1, title = 'Min Child Weight *Note this does not apply to CatBoost')
learning_rate_r = RangeSlider(start=.1, end=1, value=(.2, .6), step=.1, title="Learning Rate")
n_estimators_r = RangeSlider(start= 0, end = 500, value = (50,200), step = 25, title = 'Number of Trees')
reg_lambda_r = RangeSlider(start=0, end=1, value=(.2, .6), step=.1, title="Reg Lambda aka L2 Lambda")
colsample_bytree_r = RangeSlider(start=0, end=1, value=(.2, .6), step=.1, title="Col-Sample by Tree *Note this does not apply to CatBoost")
random_state_r = TextInput(value="42", title="Random State *Note this does not apply to CatBoost")
rsm_r = RangeSlider(start=.1, end=1, value=(.2, .6), step=.1, title="RSM *Note this ONLY applies to CatBoost") 

#random
max_depth_s = RangeSlider(start=0, end=16, value=(0, 16), step=1, title="Max Depth")
min_child_weight_s = RangeSlider(start= 0, end = 10, value = (0, 10), step = 1, title = 'Min Child Weight *Note this does not apply to CatBoost')
learning_rate_s = RangeSlider(start=.1, end=1, value=(.1, 1), step=.1, title="Learning Rate")
n_estimators_s = RangeSlider(start= 0, end = 500, value = (0,500), step = 25, title = 'Number of Trees')
reg_lambda_s = RangeSlider(start=.1, end=1, value=(.1, 1), step=.1, title="Reg Lambda aka L2 Lambda")
colsample_bytree_s = RangeSlider(start=0, end=1, value=(.1, 1), step=.1, title="Col-Sample by Tree *Note this does not apply to CatBoost")
random_state_s = TextInput(value="42", title="Random State *Note this does not apply to CatBoost")

#both
iteration_num1 = RadioGroup( labels=["50", "100", "200", "500"], active=0)
iteration_num2 = RadioGroup( labels=["50", "100", "200", "500"], active=0)
iteration_num3 = RadioGroup( labels=["50", "100", "200", "500"], active=0)



#lock model parameters
lock_params = Button(label="Lock Parameters/Run Parameter Tuning Technique", button_type="primary")
lock_params.on_click(update_variables)


col_man = column(learning_rate, 
                 max_depth, 
                 iteration_num1, 
                 min_child_weight,
                 n_estimators,
                 reg_lambda,
                 colsample_bytree,
                 random_state,
                 rsm,
                 width=200)


col_grid = column(learning_rate_r, 
                  max_depth_r, 
                  iteration_num2, 
                  min_child_weight_r,
                  n_estimators_r,
                  reg_lambda_r,
                  colsample_bytree_r,
                  random_state_r,
                  rsm_r,
                  width=200)


col_random = column(learning_rate_s, 
                    max_depth_s, 
                    iteration_num3, 
                    min_child_weight_s,
                    n_estimators_s,
                    reg_lambda_s,
                    colsample_bytree_s,
                    random_state_s,
                    width=200)


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