#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 21:24:55 2019

@author: Home
"""

import pandas as pd
import time
import os


from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn, Paragraph, RadioGroup, Tabs, Panel
from bokeh.layouts import row, column

from tornado import gen
from threading import Thread
from functools import partial

from joblib import dump, load

main_path = os.getcwd()
main_path = main_path+'/Data/model_data/'

df = pd.read_csv(main_path+'model_performance.csv')



# this must only be modified from a Bokeh session callback
source = ColumnDataSource(data=df.to_dict(orient='list'))

# This is important! Save curdoc() to make sure all threads
# see the same document.
doc = curdoc()

@gen.coroutine
def update(df):
    source.stream(df.to_dict(orient='list'))

def blocking_task():
    while True:
        # do some blocking computation
        time.sleep(30)
        
        df = pd.read_csv(main_path+'model_performance.csv')

        # but update the document from callback
        doc.add_next_tick_callback(partial(update, df))
        










