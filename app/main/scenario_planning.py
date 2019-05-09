#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:24:55 2019

@author: Home
"""

import mysql.connector
import pandas as pd
import os

from bokeh.layouts import row, column
from bokeh.plotting import curdoc

#connect to db and pull in dataset
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Izzie722", database='icom_demo')


cursor = mydb.cursor()
query = ("SELECT * FROM model_variables")
cursor.execute(query)
db_data=cursor.fetchall()

indep_vars = db_data[len(db_data)-1][1].split('-')
dep_vars = db_data[len(db_data)-1][2].split('-')
model_sel = db_data[len(db_data)-1][3]
params_sel = db_data[len(db_data)-1][4].split('-')

if model_sel == 'XGBoost':
    xg_dict = {"learning_rate":params_sel[0],"max_depth":params_sel[1],"reg_lambda":params_sel[2],"min_child_weight":params_sel[3],"n_estimators":params_sel[4]}
elif model_sel == "LightGBM":      
    lb_dict = {"learning_rate":params_sel[0],"max_depth":params_sel[1],"reg_lambda":params_sel[2],"min_child_weight":params_sel[3],"n_estimators":params_sel[4]}
elif model_sel =="CatBoost":
    cb_dict = {"learning_rate":params_sel[0],"depth":params_sel[1],"one_hot_max_siz":params_sel[2],"iterations":params_sel[3],"l2_leaf_reg":params_sel[4]}







