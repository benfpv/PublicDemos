#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:52:29 2020

@author: benedictpark
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import random
import math
from scipy.stats import beta
from collections import namedtuple

########################################################
## Descriptors
########################################################

num_variables = 7;

class Variable_Obj():
    def __init__(self, name, beta, est_payout_mean, est_payout_sd):
        self.name = name;
        self.beta = beta;
        self.est_payout_mean = est_payout_mean;
        self.est_payout_sd = est_payout_sd;
        
bandit_betas = [];

#plotting
y_name = ['Theta(x)'];
y_minmax = [0.0, 1.0];
x_name = ['x'];
x_minmax = [-2, 2];

