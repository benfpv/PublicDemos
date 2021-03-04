#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:31:20 2020

@author: root
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import os #, sys
import shutil
import psychopy
from psychopy import visual, core, event, gui, logging
import random
import math
import time
import csv

#%%
###########################
#### B. Quick Settings ####
###########################
debug = 1;
# open a white full screen window
screen_x=800;
screen_y=800;
screen_boundary=100;
framerate=60;
# background color
background_color=[0,0,0];
# durations in ms
iti_dur_min = 250;
iti_dur_max = 500;
makechoice_dur_min = 100;
makechoice_dur_max = 1000;
ri_dur_min = 100;
ri_dur_max = 200;
payout_dur_min = 400;
payout_dur_max = 500;

win = psychopy.visual.Window(size=[screen_x,screen_y],fullscr=False, allowGUI=True, color=background_color, units='height');

#%%
########################################################
## Descriptors
########################################################
original_num_variants = 2; #overridden by human experiment conditions
num_variants = original_num_variants;

num_blocks = 1; #overridden by human experiment conditions
num_trials_per_block = 2;

# decision method
decision_method = 2; #-1 none, 0 random, 1 e-greedy, 2 thompson sampling bernoulli

# [1] random sampling settings

# [2] epsilon greedy (e-greedy) sampling settings
decision_training = .20; #proportion out of 1 trials to randomly sample before entering epsilon mode.
decision_epsilon = .1; #probability out of 1 for random sampling.

# [3] thompson sampling (beta bernoulli) sampling settings
num_ts_iterations = 1; #number of times to thompson-sample each variant from the beta distribution. The higher this value, the quicker exploitation > exploration. If <1, then a variant is randomly sampled at 1-(input_value) probability.

# plot details
plot_every_n = num_trials_per_block; #plot the beta bernoulli only once every n trials. 0 = no plot.

# colours
variant_original_colours = [[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,0,1]]; # colours for the first 7 variants.

variant_colours = [[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,0,1]];
if (num_variants > 6): # if more than 7 variants
    for extra_variants in range(num_variants - 7):
        variant_colours.append([random.randint(0,100)/100, random.randint(0,100)/100, random.randint(0,100)/100, 1]);
#random.shuffle(variant_colours); # shuffle colours for more fun :)
plot_colours = variant_colours;

########################################################
## Human Experiment Conditions (Stim)
########################################################
#blockwise conditions
num_iterations = 2;
original_set_sizes = [2, 3, 5, 6];
num_blocks = len(original_set_sizes) * num_iterations;

set_sizes = [2, 3, 5, 6];
########################################################
## Priors
########################################################
variant_priors = [[1,1]] * num_variants; #priors in terms of #successes (alpha) and #failures (beta). [[alpha0, beta0],[alpha1, beta1], etc.].
variant_original_true_payouts = [.6, .5, .5, .5, .5, .5]; #true payout values... If empty, random.

# variant_true_payouts = [.3,.4,.5,.6,.7]; #true payout values... If empty, random.
#variant_true_payouts = []; #true payout values... If empty, random.
# variant_priors[0] = [6,4]; #example of priors.

variant_true_payouts = [.6, .5, .5, .5, .5, .5]; # for initial setup

########################################################
## Initialize Pre-Experiment
########################################################
# Create variants and objects
class Variant_Obj():
    def __init__(self, name, this_pulled, this_rank, true_payout, prior_alpha, prior_beta, this_counter, this_payout, total_payout, est_payout_mean, est_payout_sd, total_alpha, total_beta):
        self.name = name;
        self.this_pulled = this_pulled;
        self.this_rank = this_rank;
        self.true_payout = true_payout;
        self.prior_alpha = prior_alpha;
        self.prior_beta = prior_beta;
        self.this_counter = this_counter;
        self.this_payout = this_payout;
        self.total_payout = total_payout;
        self.est_payout_mean = est_payout_mean;
        self.est_payout_sd = est_payout_sd;
        self.total_alpha = total_alpha;
        self.total_beta = total_beta;

class Variant_History():
    def __init__(self, block, trial, name, this_pulled, this_rank, true_payout, prior_alpha, prior_beta, this_counter, this_payout, total_payout, est_payout_mean, est_payout_sd, total_alpha, total_beta):
        self.block = block;
        self.trial = trial;
        self.name = name;
        self.this_pulled = this_pulled;
        self.this_rank = this_rank;
        self.true_payout = true_payout;
        self.prior_alpha = prior_alpha;
        self.prior_beta = prior_beta;
        self.this_counter = this_counter;
        self.this_payout = this_payout;
        self.total_payout = total_payout;
        self.est_payout_mean = est_payout_mean;
        self.est_payout_sd = est_payout_sd;
        self.total_alpha = total_alpha;
        self.total_beta = total_beta;

class Decision_Maker():
    def __init__(self, block, trial, successes, failures, training, epsilon, action, ts_random, ideal_payout, regret):
        self.block = block;
        self.trial = trial;
        self.successes = successes;
        self.failures = failures;
        self.training = training;
        self.epsilon = epsilon;
        self.action = action;
        self.ts_random = ts_random;
        self.ideal_payout = ideal_payout;
        self.regret = regret;
        
class Human():
    def __init__(self, block, trial, successes, failures, action, ideal_payout, regret, set_size):
        self.block = block;
        self.trial = trial;
        self.successes = successes;
        self.failures = failures;
        self.action = action;
        self.ideal_payout = ideal_payout;
        self.regret = regret;
        self.set_size = set_size;

class Masterdata_Variant_History():
    def __init__(self, block, trial, name, this_pulled, this_rank, true_payout, prior_alpha, prior_beta, this_counter, this_payout, total_payout, est_payout_mean, est_payout_sd, total_alpha, total_beta):
        self.block = block;
        self.trial = trial;
        self.name = name;
        self.this_pulled = this_pulled;
        self.this_rank = this_rank;
        self.true_payout = true_payout;
        self.prior_alpha = prior_alpha;
        self.prior_beta = prior_beta;
        self.this_counter = this_counter;
        self.this_payout = this_payout;
        self.total_payout = total_payout;
        self.est_payout_mean = est_payout_mean;
        self.est_payout_sd = est_payout_sd;
        self.total_alpha = total_alpha;
        self.total_beta = total_beta;

class Masterdata_Decision_Maker():
    def __init__(self, block, trial, successes, failures, training, epsilon, action, ts_random, ideal_payout, regret):
        self.block = block;
        self.trial = trial;
        self.successes = successes;
        self.failures = failures;
        self.training = training;
        self.epsilon = epsilon;
        self.action = action;
        self.ts_random = ts_random;
        self.ideal_payout = ideal_payout;
        self.regret = regret;
        
class Masterdata_Human():
    def __init__(self, block, trial, t_block_start, t_trial_start, t_iti_dur, t_makechoice_dur, rt, t_ri_dur, t_payout_dur, successes, failures, action, ideal_payout, regret, set_size):
        self.block = block;
        self.trial = trial;
        self.t_block_start = t_block_start;
        self.t_trial_start = t_trial_start;
        self.t_iti_dur = t_iti_dur;
        self.t_makechoice_dur = t_makechoice_dur;
        self.rt = rt;
        self.t_ri_dur = t_ri_dur;
        self.t_payout_dur = t_payout_dur;
        self.successes = successes;
        self.failures = failures;
        self.action = action;
        self.ideal_payout = ideal_payout;
        self.regret = regret;
        self.set_size = set_size;

class Plot_Beta():
    def __init__(self, this_beta_x, this_beta_y):
        self.this_beta_x = this_beta_x;
        self.this_beta_y = this_beta_y;
    
class Plot_Human_Beta():
    def __init__(self, this_human_beta_x, this_human_beta_y):
        self.this_human_beta_x = this_human_beta_x;
        self.this_human_beta_y = this_human_beta_y;

# Assign payouts to variants
variant = [0] * num_variants;
human_variant = variant;
for this_variant in range(num_variants): # for each variant
    init_name = this_variant;
    init_this_pulled = 0;
    init_this_rank = 0;
    if not variant_true_payouts:
        init_true_payout = random.randint(0,100)/100;
    else:
        init_true_payout = variant_true_payouts[this_variant];
    init_prior_alpha = variant_priors[this_variant][0];
    init_prior_beta = variant_priors[this_variant][1];
    init_this_counter = 0;
    init_this_payout = 0;
    init_total_payout = 0;
    init_est_payout_mean = 0;
    init_est_payout_sd = 0;
    init_total_alpha = variant_priors[this_variant][0];
    init_total_beta = variant_priors[this_variant][1];
    variant[this_variant] = Variant_Obj(init_name, init_this_pulled, init_this_rank, init_true_payout, init_prior_alpha, init_prior_beta, init_this_counter, init_this_payout, init_total_payout, init_est_payout_mean, init_est_payout_sd, init_total_alpha, init_total_beta);
    human_variant[this_variant] = variant[this_variant];

# Assign payouts to human variants
human_variant = [0] * num_variants;
for this_variant in range(num_variants): # for each variant
    init_name = this_variant;
    init_this_pulled = 0;
    init_this_rank = 0;
    if not variant_true_payouts:
        init_true_payout = random.randint(0,100)/100;
    else:
        init_true_payout = variant_true_payouts[this_variant];
    init_prior_alpha = variant_priors[this_variant][0];
    init_prior_beta = variant_priors[this_variant][1];
    init_this_counter = 0;
    init_this_payout = 0;
    init_total_payout = 0;
    init_est_payout_mean = 0;
    init_est_payout_sd = 0;
    init_total_alpha = variant_priors[this_variant][0];
    init_total_beta = variant_priors[this_variant][1];
    human_variant[this_variant] = Variant_Obj(init_name, init_this_pulled, init_this_rank, init_true_payout, init_prior_alpha, init_prior_beta, init_this_counter, init_this_payout, init_total_payout, init_est_payout_mean, init_est_payout_sd, init_total_alpha, init_total_beta);

# Assign variant hist
variant_hist = [0] * num_variants;
for this_variant in range(num_variants): # for each variant
    init_block_hist = [0] * num_trials_per_block;
    init_trial_hist = [0] * num_trials_per_block;
    init_name_hist = [this_variant] * num_trials_per_block;
    init_this_pulled_hist = [init_this_pulled] * num_trials_per_block;
    init_this_rank_hist = [init_this_rank] * num_trials_per_block;
    if not variant_true_payouts:
        init_true_payout_hist = [random.randint(0,100)/100] * num_trials_per_block;
    else:
        init_true_payout_hist = [variant_true_payouts[this_variant]] * num_trials_per_block;
    init_prior_alpha_hist = [variant_priors[this_variant][0]] * num_trials_per_block;
    init_prior_beta_hist = [variant_priors[this_variant][1]] * num_trials_per_block;
    init_this_counter_hist = [init_this_counter] * num_trials_per_block;
    init_this_payout_hist = [init_this_payout] * num_trials_per_block;
    init_total_payout_hist = [init_total_payout] * num_trials_per_block;
    init_est_payout_mean_hist = [init_est_payout_mean] * num_trials_per_block;
    init_est_payout_sd_hist = [init_est_payout_sd] * num_trials_per_block;
    init_total_alpha_hist = [variant_priors[this_variant][0]] * num_trials_per_block;
    init_total_beta_hist = [variant_priors[this_variant][1]] * num_trials_per_block;
    variant_hist[this_variant] = Variant_History(init_block_hist, init_trial_hist, init_name_hist, init_this_pulled_hist, init_this_rank_hist, init_true_payout_hist, init_prior_alpha_hist, init_prior_beta_hist, init_this_counter_hist, init_this_payout_hist, init_total_payout_hist, init_est_payout_mean_hist, init_est_payout_sd_hist, init_total_alpha_hist, init_total_beta_hist);

# Assign human variant hist
human_variant_hist = [0] * num_variants;
for this_variant in range(num_variants): # for each variant
    init_block_hist = [0] * num_trials_per_block;
    init_trial_hist = [0] * num_trials_per_block;
    init_name_hist = [this_variant] * num_trials_per_block;
    init_this_pulled_hist = [init_this_pulled] * num_trials_per_block;
    init_this_rank_hist = [init_this_rank] * num_trials_per_block;
    if not variant_true_payouts:
        init_true_payout_hist = [random.randint(0,100)/100] * num_trials_per_block;
    else:
        init_true_payout_hist = [variant_true_payouts[this_variant]] * num_trials_per_block;
    init_prior_alpha_hist = [variant_priors[this_variant][0]] * num_trials_per_block;
    init_prior_beta_hist = [variant_priors[this_variant][1]] * num_trials_per_block;
    init_this_counter_hist = [init_this_counter] * num_trials_per_block;
    init_this_payout_hist = [init_this_payout] * num_trials_per_block;
    init_total_payout_hist = [init_total_payout] * num_trials_per_block;
    init_est_payout_mean_hist = [init_est_payout_mean] * num_trials_per_block;
    init_est_payout_sd_hist = [init_est_payout_sd] * num_trials_per_block;
    init_total_alpha_hist = [variant_priors[this_variant][0]] * num_trials_per_block;
    init_total_beta_hist = [variant_priors[this_variant][1]] * num_trials_per_block;
    human_variant_hist[this_variant] = Variant_History(init_block_hist, init_trial_hist, init_name_hist, init_this_pulled_hist, init_this_rank_hist, init_true_payout_hist, init_prior_alpha_hist, init_prior_beta_hist, init_this_counter_hist, init_this_payout_hist, init_total_payout_hist, init_est_payout_mean_hist, init_est_payout_sd_hist, init_total_alpha_hist, init_total_beta_hist);

# Assign masterdata variant hist
masterdata_variant_hist = [0] * num_variants;
for this_variant in range(num_variants): # for each variant
    init_block = [0] * num_trials_per_block * num_blocks;
    init_trial = [0] * num_trials_per_block * num_blocks;
    init_name = [this_variant] * num_trials_per_block * num_blocks;
    init_this_pulled = [init_this_pulled] * num_trials_per_block * num_blocks;
    init_this_rank = [init_this_rank] * num_trials_per_block * num_blocks;
    if not variant_true_payouts:
        init_true_payout = [random.randint(0,100)/100] * num_trials_per_block * num_blocks;
    else:
        init_true_payout = [variant_true_payouts[this_variant]] * num_trials_per_block * num_blocks;
    init_prior_alpha = [variant_priors[this_variant][0]] * num_trials_per_block * num_blocks;
    init_prior_beta = [variant_priors[this_variant][1]] * num_trials_per_block * num_blocks;
    init_this_counter = [init_this_counter] * num_trials_per_block * num_blocks;
    init_this_payout = [init_this_payout] * num_trials_per_block * num_blocks;
    init_total_payout = [init_total_payout] * num_trials_per_block * num_blocks;
    init_est_payout_mean = [init_est_payout_mean] * num_trials_per_block * num_blocks;
    init_est_payout_sd = [init_est_payout_sd] * num_trials_per_block * num_blocks;
    init_total_alpha = [variant_priors[this_variant][0]] * num_trials_per_block * num_blocks;
    init_total_beta = [variant_priors[this_variant][1]] * num_trials_per_block * num_blocks;
    masterdata_variant_hist[this_variant] = Masterdata_Variant_History(init_block, init_trial, init_name, init_this_pulled, init_this_rank, init_true_payout, init_prior_alpha, init_prior_beta, init_this_counter, init_this_payout, init_total_payout, init_est_payout_mean, init_est_payout_sd, init_total_alpha, init_total_beta);

# Assign masterdata human variant hist
masterdata_human_variant_hist = masterdata_variant_hist;
for this_variant in range(num_variants): # for each variant
    init_block = [0] * num_trials_per_block * num_blocks;
    init_trial = [0] * num_trials_per_block * num_blocks;
    init_name = [this_variant] * num_trials_per_block * num_blocks;
    init_this_pulled = [init_this_pulled] * num_trials_per_block * num_blocks;
    init_this_rank = [init_this_rank] * num_trials_per_block * num_blocks;
    if not variant_true_payouts:
        init_true_payout = [random.randint(0,100)/100] * num_trials_per_block * num_blocks;
    else:
        init_true_payout = [variant_true_payouts[this_variant]] * num_trials_per_block * num_blocks;
    init_prior_alpha = [variant_priors[this_variant][0]] * num_trials_per_block * num_blocks;
    init_prior_beta = [variant_priors[this_variant][1]] * num_trials_per_block * num_blocks;
    init_this_counter = [init_this_counter] * num_trials_per_block * num_blocks;
    init_this_payout = [init_this_payout] * num_trials_per_block * num_blocks;
    init_total_payout = [init_total_payout] * num_trials_per_block * num_blocks;
    init_est_payout_mean = [init_est_payout_mean] * num_trials_per_block * num_blocks;
    init_est_payout_sd = [init_est_payout_sd] * num_trials_per_block * num_blocks;
    init_total_alpha = [variant_priors[this_variant][0]] * num_trials_per_block * num_blocks;
    init_total_beta = [variant_priors[this_variant][1]] * num_trials_per_block * num_blocks;
    masterdata_human_variant_hist[this_variant] = Masterdata_Variant_History(init_block, init_trial, init_name, init_this_pulled, init_this_rank, init_true_payout, init_prior_alpha, init_prior_beta, init_this_counter, init_this_payout, init_total_payout, init_est_payout_mean, init_est_payout_sd, init_total_alpha, init_total_beta);

# Assign permanents
variant_sorted = sorted(variant, key=lambda x: x.true_payout, reverse=True);
human_variant_sorted = variant_sorted;

# Assign transients
variant_est_sorted = sorted(variant, key=lambda x: x.est_payout_mean, reverse=True);
variant_est_sorted_ranks = [0] * num_variants;
human_variant_est_sorted = sorted(human_variant, key=lambda x: x.est_payout_mean, reverse=True);
human_variant_est_sorted_ranks = variant_est_sorted_ranks;

if num_ts_iterations < 1:
    thompson_samples = [[0] * 1] * num_variants;
else:
    thompson_samples = [[0] * num_ts_iterations] * num_variants;
thompson_samples_mean = [0] * num_variants;

for this_variant in range(num_variants):
    variant_est_sorted_ranks[this_variant] = this_variant;
    human_variant_est_sorted_ranks[this_variant] = this_variant;
    
    if num_ts_iterations < 1:
        thompson_samples[this_variant][0] = 0;
    else:
        for this_iteration in range(num_ts_iterations):
            thompson_samples[this_variant][this_iteration] = 0;

# Assign decision maker
init_block = [0] * num_trials_per_block;
init_trial = [0] * num_trials_per_block;
init_successes = [0] * num_trials_per_block;
init_failures = [0] * num_trials_per_block;
init_training = [decision_training] * num_trials_per_block;
init_epsilon = [decision_epsilon] * num_trials_per_block;
init_action = [0] * num_trials_per_block;
init_ts_random = [0] * num_trials_per_block;
init_ideal_payout = [0] * num_trials_per_block;
init_regret = [0] * num_trials_per_block;
decision_maker = Decision_Maker(init_block, init_trial, init_successes, init_failures, init_training, init_epsilon, init_action, init_ts_random, init_ideal_payout, init_regret);

# Assign human
init_block = [0] * num_trials_per_block;
init_trial = [0] * num_trials_per_block;
init_successes = [0] * num_trials_per_block;
init_failures = [0] * num_trials_per_block;
init_action = [0] * num_trials_per_block;
init_ideal_payout = [0] * num_trials_per_block;
init_regret = [0] * num_trials_per_block;
init_set_size = [0] * num_trials_per_block;
human = Human(init_block, init_trial, init_successes, init_failures, init_action, init_ideal_payout, init_regret, init_set_size);

# Assign masterdata decision maker
init_block = [0] * num_trials_per_block * num_blocks;
init_trial = [0] * num_trials_per_block * num_blocks;
init_successes = [0] * num_trials_per_block * num_blocks;
init_failures = [0] * num_trials_per_block * num_blocks;
init_training = [decision_training] * num_trials_per_block * num_blocks;
init_epsilon = [decision_epsilon] * num_trials_per_block * num_blocks;
init_action = [0] * num_trials_per_block * num_blocks;
init_ts_random = [0] * num_trials_per_block * num_blocks;
init_ideal_payout = [0] * num_trials_per_block * num_blocks;
init_regret = [0] * num_trials_per_block * num_blocks;
masterdata_decision_maker = Masterdata_Decision_Maker(init_block, init_trial, init_successes, init_failures, init_training, init_epsilon, init_action, init_ts_random, init_ideal_payout, init_regret);

# Assign masterdata human
init_block = [0] * num_trials_per_block * num_blocks;
init_trial = [0] * num_trials_per_block * num_blocks;
init_t_block_start = [0] * num_trials_per_block * num_blocks;
init_t_trial_start = [0] * num_trials_per_block * num_blocks;
init_t_iti_dur = [0] * num_trials_per_block * num_blocks;
init_t_makechoice_dur = [0] * num_trials_per_block * num_blocks;
init_rt = [0] * num_trials_per_block * num_blocks;
init_t_ri_dur = [0] * num_trials_per_block * num_blocks;
init_t_payout_dur = [0] * num_trials_per_block * num_blocks;
init_successes = [0] * num_trials_per_block * num_blocks;
init_failures = [0] * num_trials_per_block * num_blocks;
init_action = [0] * num_trials_per_block * num_blocks;
init_ideal_payout = [0] * num_trials_per_block * num_blocks;
init_regret = [0] * num_trials_per_block * num_blocks;
init_set_size = [0] * num_trials_per_block * num_blocks;
masterdata_human = Masterdata_Human(init_block, init_trial, init_t_block_start, init_t_trial_start, init_t_iti_dur, init_t_makechoice_dur, init_rt, init_t_ri_dur, init_t_payout_dur, init_successes, init_failures, init_action, init_ideal_payout, init_regret, init_set_size);

# Plotter
# Beta bernoulli plot
init_this_beta_x = [[0]*101] * num_trials_per_block;
init_this_beta_y = [[0]*101] * num_trials_per_block;
plot_beta = Plot_Beta(init_this_beta_x, init_this_beta_y);
plot_every_n_counter = 0;

init_this_human_beta_x = [[0]*101] * num_trials_per_block;
init_this_human_beta_y = [[0]*101] * num_trials_per_block;
plot_human_beta = Plot_Human_Beta(init_this_human_beta_x, init_this_human_beta_y);

#%%
########################################################
## Experiment Init
########################################################
# Stimuli (Stims)
screencenter = [0,0];
if debug == 1:
    fig = plt.figure();

# Stims
stim_size = .3/(num_variants); #default stim size
stim_radius = stim_size/2;
current_stim_pos = [0] * num_variants;
for this_variant in range(num_variants):
    current_stim_pos[this_variant] = variant[this_variant].name;
payout_text = visual.TextStim(win=win,
                        name='payout_text',text='9',pos=(1,.05),
                        color='black',height=.05);
human_payout_text = visual.TextStim(win=win,
                        name='human_payout_text',text='9',pos=(1,-.05),
                        color='black',height=.05);

# Create map for object placement
map_angle = 360/(num_variants*2);
map_numpos = 360/map_angle;
map_radius = (stim_size*map_numpos)*0.6; #hypotenuse
if int(map_numpos) == map_numpos:
    map_pos = [0] * int(map_numpos);
    map_numpos = int(map_numpos);
else:
    print('error, num_variants is not divisible into whole number');
    exit();

for this_map_numpos in range(map_numpos):
    current_map_angle = map_angle * this_map_numpos;
    map_pos[this_map_numpos] = [np.sin(np.deg2rad(map_angle * this_map_numpos)) * map_radius, np.cos(np.deg2rad(map_angle * this_map_numpos)) * map_radius];

# Create stimuli
class Masterdata_Stim():
    def __init__(self, name, block, trial, map_angle, map_numpos, map_radius, map_pos, pos, colour):
        self.name = name;
        self.block = block;
        self.trial = trial;
        self.map_angle = map_angle;
        self.map_numpos = map_numpos;
        self.map_radius = map_radius;
        self.map_pos = map_pos;
        self.pos = pos;
        self.colour = colour;

masterdata_stim = [0] * num_variants;
for this_variant in range(num_variants):
    init_name = [str(this_variant)] * num_trials_per_block * num_blocks;
    init_block = [0] * num_trials_per_block * num_blocks;
    init_trial = [0] * num_trials_per_block * num_blocks;
    init_map_angle = [map_angle] * num_trials_per_block * num_blocks;
    init_map_numpos = [map_numpos] * num_trials_per_block * num_blocks;
    init_map_radius = [map_radius] * num_trials_per_block * num_blocks;
    init_map_pos = [map_pos] * num_trials_per_block * num_blocks;
    init_pos = [0] * num_trials_per_block * num_blocks;
    init_colour = [[0,0,0,0]] * num_trials_per_block * num_blocks;
    masterdata_stim[this_variant] = Masterdata_Stim(init_name, init_block, init_trial, init_map_angle, init_map_numpos, init_map_radius, init_map_pos, init_pos, init_colour);

# - If testing set size on regret, required to evenly randomize set size across trials or blocks.
# blockwise conditions
for iteration in range(num_iterations-1):
    for set_size in range(len(set_sizes)):
        set_sizes.append(set_sizes[set_size]);

random.shuffle(set_sizes);
#blockwise conditions randperm
counter = -1;
for iteration in range(num_iterations):
    for set_size in range(len(original_set_sizes)):
        for trial in range(num_trials_per_block):
            counter = counter+1;
            masterdata_human.set_size[counter] = set_sizes[set_size];

#trialwise conditions randperm

#init draw
fixation=psychopy.visual.Circle(win=win,pos=(0,0),color='black',radius=.002,edges=12);
stim = [0] * num_variants;
for this_variant in range(num_variants):
    stim[this_variant]=psychopy.visual.Circle(win=win,pos=[0,0],color=variant_colours[this_variant][0:3], radius=stim_radius, edges=45);

# Response
mouse=event.Mouse(visible=True,win=win);

# Start timer
exp_timer = core.Clock();

########################################################
## Block
########################################################
overall_trial_counter = -1;
for block in range(num_blocks):
    this_block = block+1;
    
    #blockwise condition setsize
    num_variants = masterdata_human.set_size[num_trials_per_block * this_block - num_trials_per_block + 1];
    
    #colours
    random.shuffle(variant_colours); # colours for the first 7 variants.
    plot_colours = variant_colours;
    for this_variant in range(num_variants):
        for trial in range(num_trials_per_block):
            masterdata_stim[this_variant].colour[overall_trial_counter+1 + trial] = variant_colours[this_variant];
    for this_variant in range(num_variants):
        stim[this_variant].color=variant_colours[this_variant][0:3];

    # block start screen
    blocktext=psychopy.visual.TextStim(win=win,
                        name='text',text='block '+str(block),pos=(1,.05),
                        color='black',height=.05);
    readytext=psychopy.visual.TextStim(win=win,
                        name='text',text='ready?',pos=(1,-.05),
                        color='black',height=.05);
    fixation.draw();
    blocktext.draw();
    readytext.draw();
    
    # then flip your window
    win.flip();
    
    # ready?
    time.sleep(1);
    mouse_response = 0;
    mouse.setPos([0,0]);
    if debug == 0:
        while mouse_response == 0:
            mouse_pos = mouse.getPos();
            mouse_press = mouse.getPressed();
            if mouse_press[0] == 1:
                mouse_response = 1;
                break;
    
    ########################################################
    ## Reset Priors
    ########################################################
    variant_priors = variant_priors; #priors in terms of #successes (alpha) and #failures (beta). [[alpha0, beta0],[alpha1, beta1], etc.].
    
    variant_true_payouts = [0] * num_variants;
    for this_variant in range(num_variants):
        variant_true_payouts[this_variant] = variant_original_true_payouts[this_variant];
    
    # variant_true_payouts = [.3,.4,.5,.6,.7]; #true payout values... If empty, random.
    #variant_true_payouts = []; #true payout values... If empty, random.
    # variant_priors[0] = [6,4]; #example of priors.
    
    # variant_true_payouts = random.shuffle(variant_true_payouts); #true payout values... If empty, random.
    
    ########################################################
    ## Initialize Per Block
    ########################################################
    # Assign payouts to variants
    variant = [0] * num_variants;
    for this_variant in range(num_variants): # for each variant
        init_name = this_variant;
        init_this_pulled = 0;
        init_this_rank = 0;
        if not variant_true_payouts:
            init_true_payout = random.randint(0,100)/100;
        else:
            init_true_payout = variant_true_payouts[this_variant];
        init_prior_alpha = variant_priors[this_variant][0];
        init_prior_beta = variant_priors[this_variant][1];
        init_this_counter = 0;
        init_this_payout = 0;
        init_total_payout = 0;
        init_est_payout_mean = 0;
        init_est_payout_sd = 0;
        init_total_alpha = variant_priors[this_variant][0];
        init_total_beta = variant_priors[this_variant][1];
        variant[this_variant] = Variant_Obj(init_name, init_this_pulled, init_this_rank, init_true_payout, init_prior_alpha, init_prior_beta, init_this_counter, init_this_payout, init_total_payout, init_est_payout_mean, init_est_payout_sd, init_total_alpha, init_total_beta);
        
    # Assign payouts to human variants
    human_variant = [0] * num_variants;
    for this_variant in range(num_variants): # for each variant
        init_name = this_variant;
        init_this_pulled = 0;
        init_this_rank = 0;
        if not variant_true_payouts:
            init_true_payout = random.randint(0,100)/100;
        else:
            init_true_payout = variant_true_payouts[this_variant];
        init_prior_alpha = variant_priors[this_variant][0];
        init_prior_beta = variant_priors[this_variant][1];
        init_this_counter = 0;
        init_this_payout = 0;
        init_total_payout = 0;
        init_est_payout_mean = 0;
        init_est_payout_sd = 0;
        init_total_alpha = variant_priors[this_variant][0];
        init_total_beta = variant_priors[this_variant][1];
        human_variant[this_variant] = Variant_Obj(init_name, init_this_pulled, init_this_rank, init_true_payout, init_prior_alpha, init_prior_beta, init_this_counter, init_this_payout, init_total_payout, init_est_payout_mean, init_est_payout_sd, init_total_alpha, init_total_beta);
    
    # Assign variant hist
    variant_hist = [0] * num_variants;
    for this_variant in range(num_variants): # for each variant
        init_block_hist = [0] * num_trials_per_block;
        init_trial_hist = [0] * num_trials_per_block;
        init_name_hist = [this_variant] * num_trials_per_block;
        init_this_pulled_hist = [init_this_pulled] * num_trials_per_block;
        init_this_rank_hist = [init_this_rank] * num_trials_per_block;
        if not variant_true_payouts:
            init_true_payout_hist = [random.randint(0,100)/100] * num_trials_per_block;
        else:
            init_true_payout_hist = [variant_true_payouts[this_variant]] * num_trials_per_block;
        init_prior_alpha_hist = [variant_priors[this_variant][0]] * num_trials_per_block;
        init_prior_beta_hist = [variant_priors[this_variant][1]] * num_trials_per_block;
        init_this_counter_hist = [init_this_counter] * num_trials_per_block;
        init_this_payout_hist = [init_this_payout] * num_trials_per_block;
        init_total_payout_hist = [init_total_payout] * num_trials_per_block;
        init_est_payout_mean_hist = [init_est_payout_mean] * num_trials_per_block;
        init_est_payout_sd_hist = [init_est_payout_sd] * num_trials_per_block;
        init_total_alpha_hist = [variant_priors[this_variant][0]] * num_trials_per_block;
        init_total_beta_hist = [variant_priors[this_variant][1]] * num_trials_per_block;
        variant_hist[this_variant] = Variant_History(init_block_hist, init_trial_hist, init_name_hist, init_this_pulled_hist, init_this_rank_hist, init_true_payout_hist, init_prior_alpha_hist, init_prior_beta_hist, init_this_counter_hist, init_this_payout_hist, init_total_payout_hist, init_est_payout_mean_hist, init_est_payout_sd_hist, init_total_alpha_hist, init_total_beta_hist);
        
    # Assign human variant hist
    human_variant_hist = [0] * num_variants;
    for this_variant in range(num_variants): # for each variant
        init_block_hist = [0] * num_trials_per_block;
        init_trial_hist = [0] * num_trials_per_block;
        init_name_hist = [this_variant] * num_trials_per_block;
        init_this_pulled_hist = [init_this_pulled] * num_trials_per_block;
        init_this_rank_hist = [init_this_rank] * num_trials_per_block;
        if not variant_true_payouts:
            init_true_payout_hist = [random.randint(0,100)/100] * num_trials_per_block;
        else:
            init_true_payout_hist = [variant_true_payouts[this_variant]] * num_trials_per_block;
        init_prior_alpha_hist = [variant_priors[this_variant][0]] * num_trials_per_block;
        init_prior_beta_hist = [variant_priors[this_variant][1]] * num_trials_per_block;
        init_this_counter_hist = [init_this_counter] * num_trials_per_block;
        init_this_payout_hist = [init_this_payout] * num_trials_per_block;
        init_total_payout_hist = [init_total_payout] * num_trials_per_block;
        init_est_payout_mean_hist = [init_est_payout_mean] * num_trials_per_block;
        init_est_payout_sd_hist = [init_est_payout_sd] * num_trials_per_block;
        init_total_alpha_hist = [variant_priors[this_variant][0]] * num_trials_per_block;
        init_total_beta_hist = [variant_priors[this_variant][1]] * num_trials_per_block;
        human_variant_hist[this_variant] = Variant_History(init_block_hist, init_trial_hist, init_name_hist, init_this_pulled_hist, init_this_rank_hist, init_true_payout_hist, init_prior_alpha_hist, init_prior_beta_hist, init_this_counter_hist, init_this_payout_hist, init_total_payout_hist, init_est_payout_mean_hist, init_est_payout_sd_hist, init_total_alpha_hist, init_total_beta_hist);
        
    # Assign decision maker
    init_block = [0] * num_trials_per_block;
    init_trial = [0] * num_trials_per_block;
    init_successes = [0] * num_trials_per_block;
    init_failures = [0] * num_trials_per_block;
    init_training = [decision_training] * num_trials_per_block;
    init_epsilon = [decision_epsilon] * num_trials_per_block;
    init_action = [0] * num_trials_per_block;
    init_ts_random = [0] * num_trials_per_block;
    init_ideal_payout = [0] * num_trials_per_block;
    init_regret = [0] * num_trials_per_block;
    decision_maker = Decision_Maker(init_block, init_trial, init_successes, init_failures, init_training, init_epsilon, init_action, init_ts_random, init_ideal_payout, init_regret);

    # Assign human
    init_block = [0] * num_trials_per_block;
    init_trial = [0] * num_trials_per_block;
    init_successes = [0] * num_trials_per_block;
    init_failures = [0] * num_trials_per_block;
    init_action = [0] * num_trials_per_block;
    init_ideal_payout = [0] * num_trials_per_block;
    init_regret = [0] * num_trials_per_block;
    init_set_size = [0] * num_trials_per_block;
    human = Human(init_block, init_trial, init_successes, init_failures, init_action, init_ideal_payout, init_regret, init_set_size);

    # Assign permanents
    variant_sorted = sorted(variant, key=lambda x: x.true_payout, reverse=True);
    human_variant_sorted = variant_sorted;

    # Assign transients
    variant_est_sorted = sorted(variant, key=lambda x: x.est_payout_mean, reverse=True);
    human_variant_est_sorted = sorted(human_variant, key=lambda x: x.est_payout_mean, reverse=True);
    variant_est_sorted_ranks = [0] * num_variants;
    human_variant_est_sorted_ranks = variant_est_sorted_ranks;
    
    if num_ts_iterations < 1:
        thompson_samples = [[0] * 1] * num_variants;
    else:
        thompson_samples = [[0] * num_ts_iterations] * num_variants;
    thompson_samples_mean = [0] * num_variants;
    
    for this_variant in range(num_variants):
        variant_est_sorted_ranks[this_variant] = this_variant;
        human_variant_est_sorted_ranks[this_variant] = this_variant;
        
        if num_ts_iterations < 1:
            thompson_samples[this_variant][0] = 0;
        else:
            for this_iteration in range(num_ts_iterations):
                thompson_samples[this_variant][this_iteration] = 0;
    
    # Plotter
    # Beta bernoulli plot
    init_this_beta_x = [[0]*101] * num_trials_per_block;
    init_this_beta_y = [[0]*101] * num_trials_per_block;
    plot_beta = Plot_Beta(init_this_beta_x, init_this_beta_y);
    plot_every_n_counter = 0;
    
    init_this_human_beta_x = [[0]*101] * num_trials_per_block;
    init_this_human_beta_y = [[0]*101] * num_trials_per_block;
    plot_human_beta = Plot_Human_Beta(init_this_human_beta_x, init_this_human_beta_y);
    
    # Spacebar to start!
    t_block_start = exp_timer.getTime();
    
    ########################################################
    ## Trial
    ########################################################
    for trial in range(num_trials_per_block): # for each trial
        overall_trial_counter = overall_trial_counter+1;
        this_trial = trial+1;
        print('block:'+str(this_block)+' trial:'+str(this_trial))
        
        masterdata_human.block[overall_trial_counter] = this_block;
        masterdata_human.trial[overall_trial_counter] = this_trial;
        for this_variant in range(num_variants):
            masterdata_stim[this_variant].block[overall_trial_counter] = this_block;
            masterdata_stim[this_variant].trial[overall_trial_counter] = this_trial;
        masterdata_human.t_block_start[overall_trial_counter] = t_block_start;
        masterdata_human.t_trial_start[overall_trial_counter] = exp_timer.getTime();
        
        #%% iti
        print('start iti')
        iti_start_timer = exp_timer.getTime();
        # human trial timing
        current_iti_dur=random.randint(iti_dur_min,iti_dur_max)/1000;
        current_makechoice_dur=random.randint(makechoice_dur_min,makechoice_dur_max)/1000;
        current_ri_dur=random.randint(ri_dur_min,ri_dur_max)/1000;
        current_payout_dur=random.randint(payout_dur_min,payout_dur_max)/1000;
        
        current_map_pos = map_pos;
        random.shuffle(current_map_pos);
        for this_variant in range(num_variants):
            masterdata_stim[this_variant].map_pos[overall_trial_counter] = current_map_pos;
            masterdata_stim[this_variant].pos[overall_trial_counter] = masterdata_stim[this_variant].map_pos[overall_trial_counter][this_variant];
        
        # housekeeping
        variant_est_sorted = sorted(variant, key=lambda x: x.est_payout_mean, reverse=True);
        for this_variant in range(num_variants):
            variant_est_sorted_ranks[this_variant] = int(variant_est_sorted[this_variant].name);
            if trial == 0:
                variant[this_variant].total_alpha = variant_hist[this_variant].total_alpha[0];
                variant[this_variant].total_beta = variant_hist[this_variant].total_beta[0];
            else:
                variant[this_variant].total_alpha = variant_hist[this_variant].total_alpha[trial-1];
                variant[this_variant].total_beta = variant_hist[this_variant].total_beta[trial-1];
        # housekeeping
        human_variant_est_sorted = sorted(human_variant, key=lambda x: x.est_payout_mean, reverse=True);
        for this_variant in range(num_variants):
            human_variant_est_sorted_ranks[this_variant] = int(human_variant_est_sorted[this_variant].name);
            if trial == 0:
                human_variant[this_variant].total_alpha = human_variant_hist[this_variant].total_alpha[0];
                human_variant[this_variant].total_beta = human_variant_hist[this_variant].total_beta[0];
            else:
                human_variant[this_variant].total_alpha = human_variant_hist[this_variant].total_alpha[trial-1];
                human_variant[this_variant].total_beta = human_variant_hist[this_variant].total_beta[trial-1];
        
        # plot every n
        plot_every_n_counter += 1;
        if plot_every_n_counter == plot_every_n:
            plot_every_n_counter = 0;
            
        # present iti
        fixation.draw();
        # then flip your window
        win.flip()
        
        ########################################################
        ## Decision_Maker (which arm do we pull?)
        ########################################################
        decision_maker.block[trial] = this_block;
        decision_maker.trial[trial] = this_trial;
        # correct for identical est_means
        identical_counter = 0;
        # variant ranks
        for this_variant in range(num_variants):
            variant[this_variant].this_rank = variant_est_sorted[this_variant].name;
            variant_hist[this_variant].this_rank[trial] = variant[this_variant].this_rank;
            #print('rank:'+str(this_variant)+' var#'+str(variant_est_sorted[this_variant].name)+' est_mean:'+str(round(variant_est_sorted[this_variant].est_payout_mean,2)));
        
        # sampling method
        if decision_method == 0: #random sampling
            decision_maker.action[trial] = random.randint(0,num_variants-1);
        elif decision_method == 1: #e-greedy sampling
            if this_trial <= round(num_trials_per_block * decision_maker.training[trial],0): # if training trial then sample randomly
                decision_maker.action[trial] = random.randint(0,num_variants-1);
            else: #if training is over then sample by epsilon
                this_roll = random.randint(0,100)/100;
                if this_roll <= decision_maker.epsilon[trial]:
                    decision_maker.action[trial] = random.randint(0,num_variants-1);
                    while decision_maker.action[trial] == variant_est_sorted_ranks[0]:
                        decision_maker.action[trial] = random.randint(0,num_variants-1);
                else:
                    decision_maker.action[trial] = variant_est_sorted_ranks[0];
        elif decision_method == 2: #thompson sampling:
            if num_ts_iterations < 1:
                #epsilon
                this_roll = random.randint(0,100)/100;
                if this_roll <= num_ts_iterations:
                    decision_maker.ts_random[trial] = 1;
                    decision_maker.action[trial] = random.randint(0,num_variants-1);
                else:
                    #thompson sampling
                    for this_variant in range(num_variants):
                        thompson_samples[this_variant][0] = np.random.beta(variant[this_variant].total_alpha, variant[this_variant].total_beta, size=None);
                        thompson_samples_mean[this_variant] = np.mean(thompson_samples[this_variant]);
                    decision_maker.action[trial] = thompson_samples_mean.index(max(thompson_samples_mean));
            else:
                #thompson sampling
                for this_variant in range(num_variants):
                    for this_iteration in range(num_ts_iterations):
                        thompson_samples[this_variant][this_iteration] = np.random.beta(variant[this_variant].total_alpha, variant[this_variant].total_beta, size=None);
                    thompson_samples_mean[this_variant] = np.mean(thompson_samples[this_variant]);
                decision_maker.action[trial] = thompson_samples_mean.index(max(thompson_samples_mean));
        #print('decision_action:'+str(decision_maker.action));
        
        #%% makechoice
        ########################################################
        ## Human Makechoice (which arm do we pull?)
        ########################################################
        human.block[trial] = this_block;
        human.trial[trial] = this_trial;
        # correct for identical est_means
        identical_counter = 0;
        # variant ranks
        for this_variant in range(num_variants):
            human_variant[this_variant].this_rank = human_variant_est_sorted[this_variant].name;
            human_variant_hist[this_variant].this_rank[trial] = human_variant[this_variant].this_rank;
            
        # record iti timer
        iti_end_timer = exp_timer.getTime();
        iti_timer = iti_end_timer-iti_start_timer;
        if iti_timer > current_iti_dur:
            current_iti_dur = iti_end_timer-iti_start_timer;
        else:
            time.sleep(current_iti_dur - iti_timer);
        
        # present makechoice
        fixation.draw();
        for this_variant in range(num_variants):
            stim[this_variant].pos = masterdata_stim[this_variant].pos[overall_trial_counter];
            stim[this_variant].draw();
        # then flip your window
        win.flip()
        
        # start makechoice/rt timer
        rt_start_timer = exp_timer.getTime();
        makechoice_start_timer = exp_timer.getTime();
        
        # human response
        #core.wait(.1);
        mouse_response = 0;
        mouse.setPos([0,0]);
        if debug == 0:
            while mouse_response == 0:
                mouse_pos = mouse.getPos();
                mouse_press = mouse.getPressed();
                if mouse_press[0] == 1:
                    for this_variant in range(num_variants):
                        if (mouse_pos[0] >= stim[this_variant].pos[0]-stim_radius) and (mouse_pos[0] <= stim[this_variant].pos[0]+stim_radius): #mouse x-value vs. stim x-value
                            if (mouse_pos[1] >= stim[this_variant].pos[1]-stim_radius) and (mouse_pos[1] <= stim[this_variant].pos[1]+stim_radius): #mouse y-value vs. stim y-value
                                human.action[trial] = this_variant;
                                mouse_response = 1;
                                break;
        else:
            human.action[trial] = random.randint(0,num_variants-1);
        
        # record makechoice timer
        rt_end_timer = exp_timer.getTime();
        rt_timer = rt_end_timer - rt_start_timer;
        masterdata_human.rt[overall_trial_counter] = rt_timer;
        
        #%% makechoice process
        ########################################################
        ## Variants
        ########################################################
        for this_variant in range(num_variants): # for each variant
            variant_hist[this_variant].block[trial] = this_block;
            variant_hist[this_variant].trial[trial] = this_trial;
            # within-trial decision-reward parameters
            if this_variant == decision_maker.action[trial]: #if this is the chose variable
                variant[this_variant].this_pulled = 1;
                variant[this_variant].this_counter = variant[this_variant].this_counter + 1;
                this_roll = random.randint(0,100)/100; # roll random # from 0 to 100
                if this_roll <= variant[this_variant].true_payout: # if roll <= prior payout
                    variant[this_variant].this_payout = 1;
                    variant[this_variant].total_payout = variant[this_variant].total_payout + 1;
                    decision_maker.successes[trial] += 1;
                else: # if roll > prior payout
                    variant[this_variant].this_payout = 0;
                    variant[this_variant].total_payout = variant[this_variant].total_payout;
                    decision_maker.failures[trial] += 1;
            else:
                variant[this_variant].this_pulled = 0;
                variant[this_variant].this_counter = variant[this_variant].this_counter;
                variant[this_variant].this_payout = 0;
                variant[this_variant].total_payout = variant[this_variant].total_payout;
            
            # save trial payout data
            variant_hist[this_variant].this_pulled[trial] = variant[this_variant].this_pulled;
            variant_hist[this_variant].this_counter[trial] = variant[this_variant].this_counter;
            variant_hist[this_variant].this_payout[trial] = variant[this_variant].this_payout;
            variant_hist[this_variant].total_payout[trial] = variant[this_variant].total_payout;
            
            # alpha beta
            variant[this_variant].total_alpha = variant[this_variant].prior_alpha + variant[this_variant].total_payout; # 1 + total successes
            variant[this_variant].total_beta = variant[this_variant].prior_beta + variant[this_variant].this_counter - variant[this_variant].total_payout; # 1 + total failures
            
            # save alpha beta data
            variant_hist[this_variant].total_alpha[trial] = variant[this_variant].total_alpha;
            variant_hist[this_variant].total_beta[trial] = variant[this_variant].total_beta;
            
            # cross-trial statistics
            if variant[this_variant].this_counter == 0:
                variant[this_variant].est_payout_mean = 0;
            else:
                variant[this_variant].est_payout_mean = variant[this_variant].total_payout / variant[this_variant].this_counter;
            #variant[this_variant].est_payout_sd = np.sqrt(np.sum((variant_hist[this_variant].est_payout_mean[0:this_trial] - np.mean(variant_hist[this_variant].est_payout_mean[0:this_trial])) ** 2) / this_trial);
            
            if debug == 1:
                print('PC '+str(variant[this_variant].this_pulled)+' V'+str(this_variant)+' e_r:'+str(variant_est_sorted[this_variant].name)+' tru:'+str(variant[this_variant].true_payout)+' pay:'+str(variant[this_variant].total_payout)+'/'+str(variant[this_variant].this_counter)+' e_m:'+str(round(variant[this_variant].est_payout_mean,2))+' e_sd:'+str(round(variant[this_variant].est_payout_sd,2))+' a:'+str(variant[this_variant].total_alpha)+' b:'+str(variant[this_variant].total_beta)+' ts_m:'+str(round(thompson_samples_mean[this_variant],2)));
            
            # save trial statistics data
            variant_hist[this_variant].est_payout_mean[trial] = variant[this_variant].est_payout_mean;
            variant_hist[this_variant].est_payout_sd[trial] = variant[this_variant].est_payout_sd;
            
            if debug == 1:
                # plot trial statistics
                for x in range(len(plot_beta.this_beta_x[trial])):
                    xi = x / 100;
                    plot_beta.this_beta_x[trial][x] = xi;
                    # normal distribution
                    # plot_y[x] = (1 / variant[this_variant].est_payout_sd * np.sqrt(2 * np.pi)) ** (-1/2 * ((xi - variant[this_variant].est_payout_mean) / variant[this_variant].est_payout_sd) ** 2);
                    # beta-bernoulli distribution
                    plot_beta.this_beta_y[trial][x] = ((1 / ((math.factorial(variant[this_variant].total_alpha - 1) * math.factorial(variant[this_variant].total_beta - 1)) / (math.factorial(variant[this_variant].total_alpha + variant[this_variant].total_beta - 1)))) * ((xi ** (variant[this_variant].total_alpha - 1)) * ((1 - xi) ** (variant[this_variant].total_beta - 1))));
                
                # plot_colours[this_variant][3] = 0.09 + ((1 - 0.09)/num_trials_per_block * this_trial); # make older graphs more transparent
                plot_colours_fade = [plot_colours[this_variant][0], plot_colours[this_variant][1], plot_colours[this_variant][2], round(0.02 + ((1 - 0.09)/num_trials_per_block * this_trial * .8),2)];
                if plot_every_n_counter == 0:
                    plt.plot(plot_beta.this_beta_x[trial], plot_beta.this_beta_y[trial], color = [plot_colours[this_variant][0], plot_colours[this_variant][1], plot_colours[this_variant][2]]);
            
        if debug == 1:
            # bernoulli plot
            if plot_every_n_counter == 0:
                plt.title('PC trial '+str(this_trial)+' / '+str(num_trials_per_block));
                plt.xlabel('p');
                plt.ylabel('beta');
                plt.show();
            
        ########################################################
        ## Human_Variants
        ########################################################
        for this_variant in range(num_variants): # for each variant
            human_variant_hist[this_variant].block[trial] = this_block;
            human_variant_hist[this_variant].trial[trial] = this_trial;
            # within-trial decision-reward parameters
            if this_variant == human.action[trial]: #if this is the chose variable
                human_variant[this_variant].this_pulled = 1;
                human_variant[this_variant].this_counter = human_variant[this_variant].this_counter + 1;
                this_roll = random.randint(0,100)/100; # roll random # from 0 to 100
                if this_roll <= human_variant[this_variant].true_payout: # if roll <= prior payout
                    human_variant[this_variant].this_payout = 1;
                    human_variant[this_variant].total_payout = human_variant[this_variant].total_payout + 1;
                    human.successes[trial] += 1;
                else: # if roll > prior payout
                    human_variant[this_variant].this_payout = 0;
                    human_variant[this_variant].total_payout = human_variant[this_variant].total_payout;
                    human.failures[trial] += 1;
            else:
                human_variant[this_variant].this_pulled = 0;
                human_variant[this_variant].this_counter = human_variant[this_variant].this_counter;
                human_variant[this_variant].this_payout = 0;
                human_variant[this_variant].total_payout = human_variant[this_variant].total_payout;
            
            # save trial payout data
            human_variant_hist[this_variant].this_pulled[trial] = human_variant[this_variant].this_pulled;
            human_variant_hist[this_variant].this_counter[trial] = human_variant[this_variant].this_counter;
            human_variant_hist[this_variant].this_payout[trial] = human_variant[this_variant].this_payout;
            human_variant_hist[this_variant].total_payout[trial] = human_variant[this_variant].total_payout;
            
            # alpha beta
            human_variant[this_variant].total_alpha = human_variant[this_variant].prior_alpha + human_variant[this_variant].total_payout; # 1 + total successes
            human_variant[this_variant].total_beta = human_variant[this_variant].prior_beta + human_variant[this_variant].this_counter - human_variant[this_variant].total_payout; # 1 + total failures
            
            # save alpha beta data
            human_variant_hist[this_variant].total_alpha[trial] = human_variant[this_variant].total_alpha;
            human_variant_hist[this_variant].total_beta[trial] = human_variant[this_variant].total_beta;
            
            # cross-trial statistics
            if human_variant[this_variant].this_counter == 0:
                human_variant[this_variant].est_payout_mean = 0;
            else:
                human_variant[this_variant].est_payout_mean = human_variant[this_variant].total_payout / human_variant[this_variant].this_counter;
            #variant[this_variant].est_payout_sd = np.sqrt(np.sum((variant_hist[this_variant].est_payout_mean[0:this_trial] - np.mean(variant_hist[this_variant].est_payout_mean[0:this_trial])) ** 2) / this_trial);
            
            if debug == 1:
                print('HU '+str(human_variant[this_variant].this_pulled)+' V'+str(this_variant)+' e_r:'+str(human_variant_est_sorted[this_variant].name)+' tru:'+str(human_variant[this_variant].true_payout)+' pay:'+str(human_variant[this_variant].total_payout)+'/'+str(human_variant[this_variant].this_counter)+' e_m:'+str(round(human_variant[this_variant].est_payout_mean,2))+' e_sd:'+str(round(human_variant[this_variant].est_payout_sd,2))+' a:'+str(human_variant[this_variant].total_alpha)+' b:'+str(human_variant[this_variant].total_beta)+' ts_m:'+str(round(thompson_samples_mean[this_variant],2)));
            
            # save trial statistics data
            human_variant_hist[this_variant].est_payout_mean[trial] = human_variant[this_variant].est_payout_mean;
            human_variant_hist[this_variant].est_payout_sd[trial] = human_variant[this_variant].est_payout_sd;
            
            if debug == 1:
                # plot trial statistics
                for x in range(len(plot_human_beta.this_human_beta_x[trial])):
                    xi = x / 100;
                    plot_human_beta.this_human_beta_x[trial][x] = xi;
                    # normal distribution
                    # plot_y[x] = (1 / variant[this_variant].est_payout_sd * np.sqrt(2 * np.pi)) ** (-1/2 * ((xi - variant[this_variant].est_payout_mean) / variant[this_variant].est_payout_sd) ** 2);
                    # beta-bernoulli distribution
                    plot_human_beta.this_human_beta_y[trial][x] = ((1 / ((math.factorial(human_variant[this_variant].total_alpha - 1) * math.factorial(human_variant[this_variant].total_beta - 1)) / (math.factorial(human_variant[this_variant].total_alpha + human_variant[this_variant].total_beta - 1)))) * ((xi ** (human_variant[this_variant].total_alpha - 1)) * ((1 - xi) ** (human_variant[this_variant].total_beta - 1))));
                
                # plot_colours[this_variant][3] = 0.09 + ((1 - 0.09)/num_trials_per_block * this_trial); # make older graphs more transparent
                plot_colours_fade = [plot_colours[this_variant][0], plot_colours[this_variant][1], plot_colours[this_variant][2], round(0.02 + ((1 - 0.09)/num_trials_per_block * this_trial * .8),2)];
                if plot_every_n_counter == 0:
                    plt.plot(plot_human_beta.this_human_beta_x[trial], plot_human_beta.this_human_beta_y[trial], color = [plot_colours[this_variant][0], plot_colours[this_variant][1], plot_colours[this_variant][2]]);
        
        if debug == 1:
            # bernoulli plot
            if plot_every_n_counter == 0:
                plt.title('HU trial '+str(this_trial)+' / '+str(num_trials_per_block));
                plt.xlabel('p');
                plt.ylabel('beta');
                plt.show();
            
        # record makechoice timer
        makechoice_end_timer = exp_timer.getTime();
        makechoice_timer = makechoice_end_timer-makechoice_start_timer;
        if makechoice_timer > current_makechoice_dur:
            current_makechoice_dur = makechoice_end_timer-makechoice_start_timer;
        else:
            time.sleep(current_makechoice_dur - makechoice_timer);
        masterdata_human.t_makechoice_dur[overall_trial_counter] = current_makechoice_dur;
        
        #%% ri
        ########################################################
        ## Human Retention Interval
        ########################################################
        # blank the screen
        fixation.draw();
        win.flip();
        
        time.sleep(current_ri_dur);
        
        #%% payout
        # present payout
        if human_variant[human.action[trial]].this_payout == 1:
            human_payout_text.text = '1';
        else:
            human_payout_text.text = '0';
        if variant[decision_maker.action[trial]].this_payout == 1:
            payout_text.text = '1';
        else:
            payout_text.text = '0';
        
        fixation.draw();
        human_payout_text.draw();
        win.flip();
        
        payout_start_timer = exp_timer.getTime();
        
        ########################################################
        ## End of trial
        ########################################################
        # regret analysis
        if (decision_maker.action[trial] == int(variant_sorted[0].name)):
            if (variant[int(variant_sorted[0].name)].this_payout > 0):
               decision_maker.ideal_payout[trial] = 1;
            else:
               decision_maker.ideal_payout[trial] = 0;
        else:
            for this_variant in range(num_variants):
                if decision_maker.ideal_payout[trial] == 0:
                    this_roll = random.randint(0,100)/100;
                    if this_roll <= variant[int(variant_sorted[this_variant].name)].true_payout:
                        decision_maker.ideal_payout[trial] = 1;
                    else:
                        decision_maker.ideal_payout[trial] = 0;
        decision_maker.regret[trial] = sum(decision_maker.ideal_payout) - sum(decision_maker.successes);
        
        # human regret analysis
        if (human.action[trial] == int(human_variant_sorted[0].name)):
            if (human_variant[int(human_variant_sorted[0].name)].this_payout > 0):
               human.ideal_payout[trial] = 1;
            else:
               human.ideal_payout[trial] = 0;
        else:
            for this_variant in range(num_variants):
                if human.ideal_payout[trial] == 0:
                    this_roll = random.randint(0,100)/100;
                    if this_roll <= variant[int(human_variant_sorted[this_variant].name)].true_payout:
                        human.ideal_payout[trial] = 1;
                    else:
                        human.ideal_payout[trial] = 0;
        human.regret[trial] = sum(human.ideal_payout) - sum(human.successes);
        
        if debug == 1:
            # print
            print('PC total success rate:'+str(sum(decision_maker.successes))+'/'+str(sum(decision_maker.successes)+sum(decision_maker.failures)));
            print('PC ideal success rate: '+str(sum(decision_maker.ideal_payout))+'/'+str(sum(decision_maker.successes)+sum(decision_maker.failures)));
            print('PC regret: '+str(decision_maker.regret[trial]));
            # print
            print('HU total success rate:'+str(sum(human.successes))+'/'+str(sum(human.successes)+sum(human.failures)));
            print('HU ideal success rate: '+str(sum(human.ideal_payout))+'/'+str(sum(human.successes)+sum(human.failures)));
            print('HU regret: '+str(human.regret[trial]));
        
        # amend trial data to masterdata
        for this_variant in range(num_variants):
            masterdata_variant_hist[this_variant].block[overall_trial_counter] = round(variant_hist[this_variant].block[trial],3);
            masterdata_variant_hist[this_variant].trial[overall_trial_counter] = round(variant_hist[this_variant].trial[trial],3);
            masterdata_variant_hist[this_variant].name[overall_trial_counter] = round(variant_hist[this_variant].name[trial],3);
            masterdata_variant_hist[this_variant].this_pulled[overall_trial_counter] = round(variant_hist[this_variant].this_pulled[trial],3);
            masterdata_variant_hist[this_variant].this_rank[overall_trial_counter] = round(variant_hist[this_variant].this_rank[trial],3);
            masterdata_variant_hist[this_variant].this_payout[overall_trial_counter] = round(variant_hist[this_variant].this_payout[trial],3);
            masterdata_variant_hist[this_variant].true_payout[overall_trial_counter] = round(variant_hist[this_variant].true_payout[trial],3);
            masterdata_variant_hist[this_variant].prior_alpha[overall_trial_counter] = round(variant_hist[this_variant].prior_alpha[trial],3);
            masterdata_variant_hist[this_variant].prior_beta[overall_trial_counter] = round(variant_hist[this_variant].prior_beta[trial],3);
            masterdata_variant_hist[this_variant].this_counter[overall_trial_counter] = round(variant_hist[this_variant].this_counter[trial],3);
            masterdata_variant_hist[this_variant].this_payout[overall_trial_counter] = round(variant_hist[this_variant].this_payout[trial],3);
            masterdata_variant_hist[this_variant].total_payout[overall_trial_counter] = round(variant_hist[this_variant].total_payout[trial],3);
            masterdata_variant_hist[this_variant].est_payout_mean[overall_trial_counter] = round(variant_hist[this_variant].est_payout_mean[trial],3);
            masterdata_variant_hist[this_variant].est_payout_sd[overall_trial_counter] = round(variant_hist[this_variant].est_payout_sd[trial],3);
            masterdata_variant_hist[this_variant].total_alpha[overall_trial_counter] = round(variant_hist[this_variant].total_alpha[trial],3);
            masterdata_variant_hist[this_variant].total_beta[overall_trial_counter] = round(variant_hist[this_variant].total_beta[trial],3);
        
        masterdata_decision_maker.block[overall_trial_counter] = round(decision_maker.block[trial],3);
        masterdata_decision_maker.trial[overall_trial_counter] = round(decision_maker.trial[trial],3);
        masterdata_decision_maker.successes[overall_trial_counter] = round(decision_maker.successes[trial],3);
        masterdata_decision_maker.failures[overall_trial_counter] = round(decision_maker.failures[trial],3);
        masterdata_decision_maker.training[overall_trial_counter] = round(decision_maker.training[trial],3);
        masterdata_decision_maker.epsilon[overall_trial_counter] = round(decision_maker.epsilon[trial],3);
        masterdata_decision_maker.action[overall_trial_counter] = round(decision_maker.action[trial],3);
        masterdata_decision_maker.ts_random[overall_trial_counter] = round(decision_maker.ts_random[trial],3);
        masterdata_decision_maker.ideal_payout[overall_trial_counter] = round(decision_maker.ideal_payout[trial],3);
        masterdata_decision_maker.regret[overall_trial_counter] = round(decision_maker.regret[trial],3);
        # amend trial data to masterdata
        for this_variant in range(num_variants):
            masterdata_human_variant_hist[this_variant].block[overall_trial_counter] = round(human_variant_hist[this_variant].block[trial],3);
            masterdata_human_variant_hist[this_variant].trial[overall_trial_counter] = round(human_variant_hist[this_variant].trial[trial],3);
            masterdata_human_variant_hist[this_variant].name[overall_trial_counter] = round(human_variant_hist[this_variant].name[trial],3);
            masterdata_human_variant_hist[this_variant].this_pulled[overall_trial_counter] = round(human_variant_hist[this_variant].this_pulled[trial],3);
            masterdata_human_variant_hist[this_variant].this_rank[overall_trial_counter] = round(human_variant_hist[this_variant].this_rank[trial],3);
            masterdata_human_variant_hist[this_variant].this_payout[overall_trial_counter] = round(human_variant_hist[this_variant].this_payout[trial],3);
            masterdata_human_variant_hist[this_variant].true_payout[overall_trial_counter] = round(human_variant_hist[this_variant].true_payout[trial],3);
            masterdata_human_variant_hist[this_variant].prior_alpha[overall_trial_counter] = round(human_variant_hist[this_variant].prior_alpha[trial],3);
            masterdata_human_variant_hist[this_variant].prior_beta[overall_trial_counter] = round(human_variant_hist[this_variant].prior_beta[trial],3);
            masterdata_human_variant_hist[this_variant].this_counter[overall_trial_counter] = round(human_variant_hist[this_variant].this_counter[trial],3);
            masterdata_human_variant_hist[this_variant].this_payout[overall_trial_counter] = round(human_variant_hist[this_variant].this_payout[trial],3);
            masterdata_human_variant_hist[this_variant].total_payout[overall_trial_counter] = round(human_variant_hist[this_variant].total_payout[trial],3);
            masterdata_human_variant_hist[this_variant].est_payout_mean[overall_trial_counter] = round(human_variant_hist[this_variant].est_payout_mean[trial],3);
            masterdata_human_variant_hist[this_variant].est_payout_sd[overall_trial_counter] = round(human_variant_hist[this_variant].est_payout_sd[trial],3);
            masterdata_human_variant_hist[this_variant].total_alpha[overall_trial_counter] = round(human_variant_hist[this_variant].total_alpha[trial],3);
            masterdata_human_variant_hist[this_variant].total_beta[overall_trial_counter] = round(human_variant_hist[this_variant].total_beta[trial],3);
        
        masterdata_human.block[overall_trial_counter] = round(human.block[trial],3);
        masterdata_human.trial[overall_trial_counter] = round(human.trial[trial],3);
        masterdata_human.successes[overall_trial_counter] = round(human.successes[trial],3);
        masterdata_human.failures[overall_trial_counter] = round(human.failures[trial],3);
        masterdata_human.action[overall_trial_counter] = round(human.action[trial],3);
        masterdata_human.ideal_payout[overall_trial_counter] = round(human.ideal_payout[trial],3);
        masterdata_human.regret[overall_trial_counter] = round(human.regret[trial],3);
        
        # record timer
        payout_end_timer = exp_timer.getTime();
        payout_timer = payout_end_timer-payout_start_timer;
        if payout_timer > current_payout_dur:
            current_payout_dur = payout_end_timer-payout_start_timer;
        else:
            time.sleep(current_payout_dur - payout_timer);

        masterdata_human.t_iti_dur[overall_trial_counter] = current_iti_dur;
        masterdata_human.t_makechoice_dur[overall_trial_counter] = current_makechoice_dur;
        masterdata_human.t_ri_dur[overall_trial_counter] = current_ri_dur;
        masterdata_human.t_payout_dur[overall_trial_counter] = current_payout_dur;
        
        if debug == 1:
            print('iti_dur: '+str(current_iti_dur));
            print('makechoice_dur: '+str(rt_timer));
            print('payout_dur: '+str(current_payout_dur));
            
            # segmentation line in console
            print('########################################################################');
        
    ########################################################
    ## End of block
    ########################################################
    if debug == 1:
        # plot cumulative regret
        plt.title('cumulative PC regret over '+str(num_trials_per_block)+' trials');
        plt.xlabel('trial');
        plt.ylabel('cumulative regret');
        plt.plot(decision_maker.regret);
        plt.show();
        # plot cumulative regret
        plt.title('cumulative HU regret over '+str(num_trials_per_block)+' trials');
        plt.xlabel('trial');
        plt.ylabel('cumulative regret');
        plt.plot(human.regret);
        plt.show();
    
    # save bernoulli plot
    # save cumulative regret plot
    
########################################################
## End of experiment
########################################################
# plot cumulative regret
plt.title('cumulative PC regret over '+str(num_trials_per_block*num_blocks)+' trials');
plt.xlabel('trial');
plt.ylabel('cumulative regret');
plt.plot(masterdata_decision_maker.regret);
plt.show();
# plot cumulative regret
plt.title('cumulative HU regret over '+str(num_trials_per_block*num_blocks)+' trials');
plt.xlabel('trial');
plt.ylabel('cumulative regret');
plt.plot(masterdata_human.regret);
plt.show();

# save data!
#masterdata_decision_maker
members = [attr for attr in dir(masterdata_decision_maker) if not callable(getattr(masterdata_decision_maker, attr)) and not attr.startswith("__")];
values = [getattr(masterdata_decision_maker, member) for member in members];
expert_data = 0;
with open("decision_maker.csv",'w') as resultFile:
    wr = csv.writer(resultFile);
    wr.writerow(members);
    export_data = zip(values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8],values[9]);
    for row in export_data:
        wr.writerow(row);

#masterdata_human
members = [attr for attr in dir(masterdata_human) if not callable(getattr(masterdata_human, attr)) and not attr.startswith("__")];
values = [getattr(masterdata_human, member) for member in members];
expert_data = 0;
with open("human.csv",'w') as resultFile:
    wr = csv.writer(resultFile);
    wr.writerow(members);
    export_data = zip(values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8],values[9],values[10],values[11],values[12],values[13],values[14]);
    for row in export_data:
        wr.writerow(row);

#masterdata_human_variant_hist
for this_variant in range(num_variants):
    this_export_name = "human_variant_hist_"+str(this_variant)+".csv";
    members = [attr for attr in dir(masterdata_human_variant_hist[this_variant]) if not callable(getattr(masterdata_human_variant_hist[this_variant], attr)) and not attr.startswith("__")];
    values = [getattr(masterdata_human_variant_hist[this_variant], member) for member in members];
    expert_data = 0;
    with open(this_export_name,'w') as resultFile:
        wr = csv.writer(resultFile);
        wr.writerow(members);
        export_data = zip(values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8],values[9],values[10],values[11],values[12],values[13],values[14]);
        for row in export_data:
            wr.writerow(row);

#masterdata_stim
for this_variant in range(num_variants):
    this_export_name = "stim_"+str(this_variant)+".csv";
    members = [attr for attr in dir(masterdata_stim[this_variant]) if not callable(getattr(masterdata_stim[this_variant], attr)) and not attr.startswith("__")];
    values = [getattr(masterdata_stim[this_variant], member) for member in members];
    expert_data = 0;
    with open(this_export_name,'w') as resultFile:
        wr = csv.writer(resultFile);
        wr.writerow(members);
        export_data = zip(values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8]);
        for row in export_data:
            wr.writerow(row);

#masterdata_variant_hist
for this_variant in range(num_variants):
    this_export_name = "variant_hist_"+str(this_variant)+".csv";
    members = [attr for attr in dir(masterdata_variant_hist[this_variant]) if not callable(getattr(masterdata_variant_hist[this_variant], attr)) and not attr.startswith("__")];
    values = [getattr(masterdata_variant_hist[this_variant], member) for member in members];
    expert_data = 0;
    with open(this_export_name,'w') as resultFile:
        wr = csv.writer(resultFile);
        wr.writerow(members);
        export_data = zip(values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8],values[9],values[10],values[11],values[12],values[13],values[14]);
        for row in export_data:
            wr.writerow(row);

#%% Close properly
core.wait(1);
win.close();
        