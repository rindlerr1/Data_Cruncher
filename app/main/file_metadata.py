#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:27:30 2019

@author: Home
"""

import pandas as pd
import numpy as np

from bokeh.plotting import curdoc

from bokeh.models.widgets import Paragraph, Select, CheckboxGroup, Button
from bokeh.layouts import row, column


main_path = '/users/home/desktop/projects/data_cruncher/app/Data/'

df = pd.read_csv(main_path+'app_data.csv')

target_file = pd.read_csv(main_path+'app_data_target_actual.csv')

def all_columns(df):
    categorical_vars = []
    numeric_vars = []
    for i in range(0, len(df.columns)):
        if (type(df[df.columns[i]][0]) == np.int64 or type(df[df.columns[i]][0]) == np.float64):
            numeric_vars.append(df.columns[i])
        elif type(df[df.columns[i]][0]) == str:
            categorical_vars.append(df.columns[i])
    return categorical_vars, numeric_vars

if target_file['Orig_Clean'][0] == 0:
    categorical_vars, numeric_vars = all_columns(df)
elif target_file['Orig_Clean'][0] == 1:
    cat_table = pd.read_csv(main_path+'cats_defined.csv')
    num_table = pd.read_csv(main_path+'nums_defined.csv')
    categorical_vars = [x for x in cat_table['Categorical_Variables']]
    numeric_vars = [x for x in num_table['Numeric_Variables']]


def define_variables():
    df = pd.read_csv(main_path+'app_data.csv')

    cat_vars = [categorical_group.labels[i] for i in categorical_group.active]
    num_vars = [numeric_group.labels[i] for i in numeric_group.active]
      
    cat_table = pd.DataFrame({'Categorical_Variables':cat_vars})
    num_table = pd.DataFrame({'Numeric_Variables': num_vars})

    cat_table.to_csv(main_path+'cats_defined.csv', index=False)
    num_table.to_csv(main_path+'nums_defined.csv', index=False)
    
    dep_variable = dependent_group.value
    dep_table = pd.DataFrame({'Dep_Variable': [dep_variable]})
    dep_table.to_csv(main_path+'dep_defined.csv', index=False)
    
    target_file['Orig_Clean'][0] = 1
    target_file.to_csv(main_path+'app_data_target_actual.csv', index=False)
    

    
    date_variable = date_group.value
    
    list_of_vars = [date_variable]
    list_of_vars.append(dep_variable)
    list_of_vars.extend(num_vars)
    list_of_vars.extend(cat_vars)
    
    df = df[list_of_vars]
    df.to_csv(main_path+'app_data_clean.csv',index=False)

    


#create app to select variables and model type
variables = [x for x in df.columns]

#variable selection widgets
p1 = Paragraph(text="Select Independent Categorical Variables",width=200)
categorical_group = CheckboxGroup(labels=variables, active=[0, 1], width=200)
p2 = Paragraph(text="Select Independent Numeric Variable",width=200)
numeric_group = CheckboxGroup(labels=variables, active=[0, 1], width=200)

p3 = Paragraph(text="Select All Potential Dependent Variables",width=200)
dependent_group = Select(title = 'Select Dependent Variable', value=variables[0], options = variables, width=200)

p4 = Paragraph(text="Select Date Variable",width=200)
date_group = Select(title = 'Select Date Variable', value=variables[0], options = variables, width=200)

#lock model parameters
lock_vars = Button(label="Define Variable Types", button_type="primary")
lock_vars.on_click(define_variables)

cat_sel = column(p1, categorical_group)
num_sel = column(p2, numeric_group)
dep_sel = column(p3, dependent_group)
date_sel = column(p4, date_group)
selection_dash = row(cat_sel, num_sel,dep_sel, date_sel, lock_vars)

curdoc().add_root(selection_dash)
curdoc().title = "File Metadata"  