#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 08:52:37 2019

@author: Home
"""

main_path = '/users/home/desktop/projects/data_cruncher/app/Data/'

class PATHS(object):
    
    def __init__(self):
        
        self.ingested_data = main_path+'app_data.csv'
        
        self.cleaned_data = main_path+'app_data_clean.csv'
        
        self.target_holder = main_path+'app_data_target_empty.csv'
        
        self.target_informer = main_path+'app_data_target_actual.csv'
        
        self.variables = main_path+'variables_defined.csv'
        
        self.parameter_data = main_path+'parameter_data.csv'
        
        
    