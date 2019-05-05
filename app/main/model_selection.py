#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:55:52 2019

@author: Home
"""


import pandas as pd
from bokeh.models.widgets import CheckboxGroup, Select, Button, Paragraph, RadioButtonGroup, TextInput, Panel, Tabs, PreText
from bokeh.layouts import row, column
from bokeh.plotting import curdoc

PATH = '/users/home/desktop/data_cruncher/app/Data/app_data.csv'

df = pd.read_csv(PATH)

#Section for injecting into db
def update_variables():
    model = model_select.value
    if model == 'XGBoost':
        params = str(str(xglearning_rate.value) +'-'+ str(xgmax_depth.value) +'-'+ str(xgreg_lambda.value) +'-'+ str(xgmin_child_weight.value) +'-'+ str(xgn_estimators.value))
    elif model == 'LightGBM':
        params = str(str(lblearning_rate.value) +'-'+ str(lbmax_depth.value) +'-'+ str(xgreg_lambda.value) +'-'+ str(lbmin_child_weight.value) +'-'+ str(lbn_estimators.value))
    elif model == 'CatBoost':
        params = str(str(cblearning_rate.value) +'-'+ str(cbdepth.value) +'-'+ str(cbone_hot_max_size.value) +'-'+ str(cbiterations.value) +'-'+ str(cbl2_leaf_reg.value))
    
    param_tuning = select_tuning.labels[select_tuning.active] 
    
    ind_variables = [independent_group.labels[i] for i in independent_group.active]
    dep_variables = dependent_group.value
    data_dict = {0:{'Model':model,
                    'Tuning_Type':param_tuning,
                    'Parameters':params,
                    'Ind_Vars': ind_variables,
                    'Dep_Vars': dep_variables}}
    df = pd.DataFrame.from_dict(data_dict).transpose()
    
    running_table.text = 'Parameters & Variables Being Processed'

    df.to_csv('/users/home/desktop/data_cruncher/app/Data/parameter_data.csv',index=False)
    
    running_table.text = 'Parameters & Variables Ingested'
        
    
#create app to select variables and model type
variables = [x for x in df.columns]

#variable selection widgets
p1 = Paragraph(text="Select Independent Variables",width=200)
independent_group = CheckboxGroup(labels=variables, active=[0, 1], width=200)
p2 = Paragraph(text="Select Dependent Variable",width=200)
dependent_group = Select(title = 'Select Target Variable', value=variables[0], options = variables, width=200)


#organize all of the selection widgets
ind_criteria = column(p1, independent_group)
dep_criteria = column(p2, dependent_group)



#Model Selection widgets
model_select = Select(title="Model Selection:", value="XGBoost", options=["XGBoost", "LightGBM", "CatBoost"])



#Selection for parameter tuning technique
select_tuning = RadioButtonGroup(labels=["Manual","Grid Search", "Random Search"], active=0, width=450)
p6 = Paragraph(text='Select parameter tuning technique')
p7 = Paragraph(text="If manual selected please tune parameters below", width=450, height=1)
tuning_select = column(p6, select_tuning, p7)


 

#selection parameter inputs
#xgboost
xglearning_rate = TextInput(value="default", title="Learning Rate", width= 150)
xgmax_depth = TextInput(value="default", title="Max Depth", width= 150)
xgreg_lambda = TextInput(value="default", title="Lambda", width= 150)
xgmin_child_weight = TextInput(value="default", title="Min Child Weight", width= 150)
xgn_estimators = TextInput(value="default", title="N-Estimators", width= 150)

#lightgbm
lblearning_rate = TextInput(value="default", title="Learning Rate", width= 150)
lbmax_depth = TextInput(value="default", title="Max Depth", width= 150)
lbreg_lambda = TextInput(value="default", title="Lambda", width= 150)
lbmin_child_weight = TextInput(value="default", title="Min Child Weight", width= 150)
lbn_estimators = TextInput(value="default", title="N-Estimators", width= 150)

#catboost
cblearning_rate = TextInput(value="default", title="Learning Rate", width= 150)
cbdepth = TextInput(value="default", title="Depth", width= 150)
cbone_hot_max_size = TextInput(value="default", title="One Hot Max Size", width= 150)
cbiterations = TextInput(value="default", title="Iterations", width= 150)
cbl2_leaf_reg = TextInput(value="default", title="L2 Leaf Reg", width= 150)



#lock model parameters
lock_params = Button(label="Lock Parameters/Run Parameter Tuning Technique", button_type="primary")
lock_params.on_click(update_variables)

#organize paramter details 
xg_col = column(xglearning_rate, xgmax_depth, xgreg_lambda,  xgmin_child_weight, xgn_estimators, width= 200)
lb_col = column(lblearning_rate, lbmax_depth, lbreg_lambda, lbmin_child_weight, lbn_estimators, width= 200)
cb_col = column(cblearning_rate, cbdepth, cbone_hot_max_size, cbiterations, cbl2_leaf_reg, width=200)

tab1 = Panel(child = xg_col, title = 'XGBoost')
tab2 = Panel(child=lb_col, title = 'LightGBM')
tab3 = Panel(child=cb_col, title='CatBoost')

#param_row = row(xg_col, lb_col, cb_col)
tabs = Tabs(tabs=[ tab1, tab2 , tab3], width=400, height=300)

running_table = PreText(text='', width=300)

   

col_1 = column(model_select,tuning_select, tabs)
col_2 = column(dep_criteria, ind_criteria, width=250, height =600)
col_3 = column(lock_params, running_table)




dash= row(col_1, col_2, col_3)


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