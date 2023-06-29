#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 17:15:52 2022

@author: chenchen
"""

import pickle
    
def Save(saved_variables, name):

    #sample_list = [1, 2, 3]
    file_name = f"{name}.pkl"
    
    open_file = open(file_name, "wb")
    pickle.dump(saved_variables, open_file)
    open_file.close()

    return


def Load(name):
    
    file_name = f"{name}.pkl"
    open_file = open(file_name, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    return loaded_list