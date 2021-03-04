# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:59:17 2020

@author: benpark
"""
# MAB_Analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import random
import math
from scipy.stats import beta
from collections import namedtuple
import os
import shutil
import csv

################################################################################################################
## Introduction
################################################################################################################
# Drag and drop relevant variant_hist and decisionmaker csv files!
this_os = 1; #1 Mac, 2 Windows
unique_datas = ['variant_hist','decision_maker']; # Name each unique data in each folder.
datas_size = [50001, 15, '<U19']; # Row, Column, # of variants, dtype.
null_number = 999999 # WARNING! THIS NUMBER MUST NOT BE USED IN DATA, and must be positive! E.g., 9999 where data range 0-100.

# Create class according to unique_datas. Index placement must be identical. Must also change "IMPORT CSV - PREPARE DATA OBJ" accordingly.
class Data_Class():
    def __init__(self, variant_hist, decision_maker):
        self.variant_hist = variant_hist;
        self.decision_maker = decision_maker;

########################################################
## Prepare to import csv
########################################################
print('------ BEGINNING ANALYSIS PIPELINE ------');
print('Getting folder names: ');
# Get path and folders
cwd = os.getcwd();
folders = os.listdir(cwd);

# Dash
if this_os == 1:
    os_path_separator = '/';
elif this_os == 2:
    os_path_separator = '\\';

# Get folder names
folders_ts = [f for f in folders if f[1:2] == 'TS'];
folder_names = [0];
counter = -1;
for f in range(len(folders)):
    if folders[f][0:2] == 'TS':
        counter += 1;
        if counter == 0:
            folder_names[counter] = folders[f];
        else:
            folder_names.append(folders[f]);
print(folder_names);

################################################################################################################
## Folderwise
################################################################################################################
# Keep track max y values
max_ylim = 0;
min_ylim = 0;

# Import files from folders as folder names
for folder in range(len(folder_names)):
    folder_name = folder_names[folder];
    # Navigate to folder
    this_path = cwd + os_path_separator + folder_name;
    os.chdir(this_path);
    files = os.listdir(this_path);
    print('--- Working with folder '+str(folder+1)+'/'+str(len(folder_names))+': '+folder_name+' ---');
    
    # Find details of the condition associated with this folder
    this_decision_method = folder_name[0:2];
    this_iterations = folder_name[5:8];
    this_variants = folder_name[10:12];
    this_probability = folder_name[14:16];
    
    # find csv_files only
    files_csv = [0];
    counter = -1;
    for file in range(len(files)):
        if files[file][-3:] == 'csv':
            counter += 1;
            if counter == 0:
                files_csv[counter] = files[file];
            else:
                files_csv.append(files[file]);
    
    # Figure out which each unique file from folder_with_max to import
    counter_each = [0] * len(unique_datas);
    files_to_import = [0] * len(files_csv);
    for unique in range(len(unique_datas)):
        unique_counter = 0;
        total_counter = -1;
        for this_csv in range(len(files_csv)):
            total_counter += 1;
            if files_csv[this_csv][:len(unique_datas[unique])] == unique_datas[unique]:
                counter_each[unique] += 1;
                files_to_import[total_counter] = unique + 1;
                unique_counter += 1;
    
    ########################################################
    ## Import csv
    ########################################################
    # Prepare data object - MUST CHANGE according to CLASS/Introduction!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    init_variant_hist = np.array([[null_number] * datas_size[1]] * datas_size[0] * counter_each[0], dtype = datas_size[2]);
    init_decision_maker = np.array([[null_number] * datas_size[1]] * datas_size[0], dtype = datas_size[2]);
    data = Data_Class(init_variant_hist, init_decision_maker);
    
    print('Importing data...');
    import_counter = 1;
    # Import Variant_Hist
    for this_csv in range(len(files_csv)):
        if files_to_import[this_csv] == 1:
            import_row = (datas_size[0] * import_counter) - datas_size[0];
            data.variant_hist[import_row:import_row+datas_size[0]] = np.genfromtxt(this_path + os_path_separator + files_csv[this_csv], delimiter=",", dtype='<U19');
            import_counter += 1;
    
    # Import Decision_Maker
    import_counter = 1;
    for this_csv in range(len(files_csv)):
        if files_to_import[this_csv] == 2:
            import_row = (datas_size[0] * import_counter) - datas_size[0];
            data.decision_maker[import_row:import_row+datas_size[0]] = np.genfromtxt(this_path + os_path_separator + files_csv[this_csv], delimiter=",", dtype='<U19');
            import_counter += 1;
            
    # Check for non-relevant numbers in data
    print('Checking for null values...');
    data_clean = 1;
    # Variant_Hist
    for row in range(len(data.variant_hist)):
        if str.isdigit(data.variant_hist[row][0]) == True:
            if len(np.unique(data.variant_hist[row])) == 1 and np.unique(data.variant_hist[row]) == str(null_number):
                print('WARNING! Non-relevant row: '+str(row)+'. Please check data sizes');
                for unique in range(len(unique_datas)):
                    print(str(unique_datas[unique])+': ('+str(datas_size[0] * counter_each[unique])+','+str(datas_size[1])+')');
                data_clean = 0;
    # Decision_Maker
    for row in range(len(data.decision_maker)):
        if str.isdigit(data.decision_maker[row][0]) == True:
            if len(np.unique(data.decision_maker[row])) == 1 and np.unique(data.decision_maker[row]) == str(null_number):
                print('WARNING! Non-relevant row: '+str(row)+'. Please check data sizes');
                for unique in range(len(unique_datas)):
                    print(str(unique_datas[unique])+': ('+str(datas_size[0] * counter_each[unique])+','+str(datas_size[1])+')');
                data_clean = 0;

    ########################################################
    ## Data Analysis
    ########################################################
    num_blocks = int(data.decision_maker[-1,np.where(data.decision_maker[0] == 'block')[0][0]]);
    num_trials_per_block = int(data.decision_maker[-1,np.where(data.decision_maker[0] == 'trial')[0][0]]);
    
    print('Acquiring mean/sd blockwise Regret...');
    # Index and acquire mean and sd blockwise and trialwise regret for decision maker.
    total_regrets = np.zeros((num_blocks));
    for block in range(num_blocks): # blockwise
        total_regrets[block] = int(data.decision_maker[np.where(data.decision_maker[:,np.where(data.decision_maker[0] == 'block')[0][0]] == str(block+1))[0],np.where(data.decision_maker[0] == 'regret')[0][0]][-1]);
    mean_blockwise_regret = np.mean(total_regrets);
    sd_blockwise_regret = np.std(total_regrets);
    
    print('Acquiring mean/sd trialwise Regret...');
    # Acquire mean and sd trialwise regret for decision maker.
    all_regrets = np.zeros((num_blocks,num_trials_per_block));
    for block in range(num_blocks):
        if np.mod(block+1,10) == 0:
            print(str(block+1)+'/'+str(num_blocks));
        for trial in range(num_trials_per_block):
            all_regrets[block][trial] = int(data.decision_maker[np.where(data.decision_maker[:,np.where(data.decision_maker[0] == 'block')[0][0]] == str(block+1))[0],np.where(data.decision_maker[0] == 'regret')[0][0]][trial]);
    mean_trialwise_regret = np.zeros((num_trials_per_block));
    sd_trialwise_regret = np.zeros((num_trials_per_block));
    for trial in range(num_trials_per_block):
        mean_trialwise_regret[trial] = np.mean(all_regrets[:,trial]);
        sd_trialwise_regret[trial] = np.std(all_regrets[:,trial]);
    
    # Maximum y value keep track, for setting limits later.
    max_ylim_here = np.max(mean_trialwise_regret)+np.max(sd_trialwise_regret);
    min_ylim_here = np.min(mean_trialwise_regret)-np.max(sd_trialwise_regret);
    if max_ylim_here > max_ylim:
        max_ylim = max_ylim_here;
    if min_ylim_here < min_ylim:
        min_ylim = min_ylim_here;
    
    print('Plotting mean/sd trialwise Regret...');
    fig = plt.plot();
    plt.plot(mean_trialwise_regret, 'k-');
    plt.plot(mean_trialwise_regret+sd_trialwise_regret, 'k:');
    plt.plot(mean_trialwise_regret-sd_trialwise_regret, 'k:');
    plt.title('Mean Cumulative Regret over Trials'+'\n Decider: '+this_decision_method+' | Iterations: '+this_iterations+' | # Variants: '+this_variants+' | Probability: '+this_probability);
    plt.xlabel('Trial');
    plt.ylabel('Mean Cumulative Regret');
    plt.xlim(0-50, num_trials_per_block+50);
    plt.ylim(-235, 565);
    plt.savefig(folder_name+'_ylim_Mean_Cumulative_Regret_over_Trials.png');
    plt.show();
    
################################################################################################################
## Finished!
################################################################################################################
print('------ FINISHED ANALYSIS PIPELINE ------');
    
    