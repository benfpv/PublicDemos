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
import os
import shutil
import csv

########################################################
## Introduction
########################################################
# Welcome!
# 1 session = program is run once.
# 1 block = group of trials.
# 1 trial = a single permutation of the MAB and its environment/variants.

# todo:
    # variant_true_payout_sd make variants variable every trial.

########################################################
## Descriptors
########################################################
print_details = 1;

num_blocks = 1;
num_trials_per_block = 20; #>2200 or so, risk lacking computing precision... floating point division by 0, etc.

save_data_to_harddrive = 0; # 1 save, 0 no save.
########################################################
## Variants
########################################################
num_variants = 3;
variants_dynamic = 1; # 0 is static reward probability per block. 1 is dynamic per trial, linear sd amount.
num_contexts = 1;

########################################################
## Decision Method
########################################################
# decision method
decision_method = 3; #0 random, 1 e-greedy, 2 thompson sampling bernoulli, 3 parallel dynamic

# [0] random sampling settings

# [1] epsilon greedy (e-greedy) sampling settings
decision_training = .1; #proportion out of 1 trials to randomly sample before entering epsilon mode.
decision_epsilon = .1; #probability out of 1 for random sampling.

# [2] thompson sampling (beta bernoulli) settings
num_ts_iterations = 1; #number of times to thompson-sample each variant from the beta distribution. The higher this value, the quicker exploitation > exploration. If <1, then thompson sampling is done only on 1-(input_value)% of trials.

# [3] parallel dynamic thompson sampling settings
num_actions_per_trial = 2;

# plot details
plot_every_n = 1; #plot the beta bernoulli only once every n trials.
plot_regret = 0; #plot cumulative regret after every block? 1 yes 0 no.

variant_colours = [[0,0,0,1],[1,0,0,1],[0,1,0,1],[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,0,1]]; # colours for the first 7 variants.
if (num_variants > 7): # if more than 7 variants
    for extra_variants in range(num_variants - 7):
        variant_colours.append([random.randint(0,100)/100, random.randint(0,100)/100, random.randint(0,100)/100, 1]);
#random.shuffle(variant_colours); # shuffle colours for more fun :)
plot_colours = variant_colours;

########################################################
## Priors
########################################################
variant_priors = [[1,1]] * num_variants; #priors in terms of #successes (alpha) and #failures (beta). [[alpha0, beta0],[alpha1, beta1], etc.].
variant_true_payouts = [.5, .5, .5]; #true payout values... If empty, random.
variant_true_payout_sd = [.2, .05, .01]; #amount of variability per trial, for each variant... If empty, random per session.
# variant_priors[0] = [6,4]; #example of priors.

################################################################################################################
## Initialize Pre-Experiment
################################################################################################################
# Create variants and objects
class Variant_Obj():
    def __init__(self, name, this_pulled, this_rank, true_payout, true_payout_sd, this_true_payout, prior_alpha, prior_beta, this_counter, this_payout, total_payout, est_payout_mean, est_payout_sd, total_alpha, total_beta):
        self.name = name;
        self.this_pulled = this_pulled;
        self.this_rank = this_rank;
        self.true_payout = true_payout;
        self.true_payout_sd = true_payout_sd;
        self.this_true_payout = this_true_payout;
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
    def __init__(self, block, trial, name, this_pulled, this_rank, true_payout, true_payout_sd, this_true_payout, prior_alpha, prior_beta, this_counter, this_payout, total_payout, est_payout_mean, est_payout_sd, total_alpha, total_beta):
        self.block = block;
        self.trial = trial;
        self.name = name;
        self.this_pulled = this_pulled;
        self.this_rank = this_rank;
        self.true_payout = true_payout;
        self.true_payout_sd = true_payout_sd;
        self.this_true_payout = this_true_payout;
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

class Masterdata_Variant_History():
    def __init__(self, block, trial, name, this_pulled, this_rank, true_payout, true_payout_sd, this_true_payout, prior_alpha, prior_beta, this_counter, this_payout, total_payout, est_payout_mean, est_payout_sd, total_alpha, total_beta):
        self.block = block;
        self.trial = trial;
        self.name = name;
        self.this_pulled = this_pulled;
        self.this_rank = this_rank;
        self.true_payout = true_payout;
        self.true_payout_sd = true_payout_sd;
        self.this_true_payout = this_true_payout;
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
    def __init__(self, decision_method, num_variants, num_blocks, num_trials_per_block, num_ts_iterations, block, trial, successes, failures, training, epsilon, action, ts_random, ideal_payout, regret):
        self.decision_method = decision_method;
        self.num_variants = num_variants;
        self.num_blocks = num_blocks;
        self.num_trials_per_block = num_trials_per_block;
        self.num_ts_iterations = num_ts_iterations;
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

class Plot_Beta():
    def __init__(self, this_beta_x, this_beta_y):
        self.this_beta_x = this_beta_x;
        self.this_beta_y = this_beta_y;

# Assign payouts to variants
variant = [0] * num_variants;
for this_variant in range(num_variants): # for each variant
    init_name = str(this_variant);
    init_this_pulled = 0;
    init_this_rank = 0;
    if not variant_true_payouts:
        init_true_payout = random.randint(0,100)/100;
    else:
        init_true_payout = variant_true_payouts[this_variant];
    if not variant_true_payout_sd:
        init_true_payout_sd = (random.randint(0,100) - 50) /100 * .25;
    else:
        init_true_payout_sd = variant_true_payout_sd[this_variant];
    init_this_true_payout = init_true_payout;
    init_prior_alpha = variant_priors[this_variant][0];
    init_prior_beta = variant_priors[this_variant][1];
    init_this_counter = 0;
    init_this_payout = 0;
    init_total_payout = 0;
    init_est_payout_mean = 0;
    init_est_payout_sd = 0;
    init_total_alpha = variant_priors[this_variant][0];
    init_total_beta = variant_priors[this_variant][1];
    variant[this_variant] = Variant_Obj(init_name, init_this_pulled, init_this_rank, init_true_payout, init_true_payout_sd, init_this_true_payout, init_prior_alpha, init_prior_beta, init_this_counter, init_this_payout, init_total_payout, init_est_payout_mean, init_est_payout_sd, init_total_alpha, init_total_beta);

# Assign variant hist
variant_hist = [0] * num_variants;
for this_variant in range(num_variants): # for each variant
    init_block_hist = [0] * num_trials_per_block;
    init_trial_hist = [0] * num_trials_per_block;
    init_name_hist = [init_name] * num_trials_per_block;
    init_this_pulled_hist = [init_this_pulled] * num_trials_per_block;
    init_this_rank_hist = [init_this_rank] * num_trials_per_block;
    init_true_payout_hist = [init_true_payout] * num_trials_per_block;
    init_true_payout_sd_hist = [init_true_payout_sd] * num_trials_per_block;
    init_this_true_payout_hist = [init_this_true_payout] * num_trials_per_block;
    init_prior_alpha_hist = [init_prior_alpha] * num_trials_per_block;
    init_prior_beta_hist = [init_prior_beta] * num_trials_per_block;
    init_this_counter_hist = [init_this_counter] * num_trials_per_block;
    init_this_payout_hist = [init_this_payout] * num_trials_per_block;
    init_total_payout_hist = [init_total_payout] * num_trials_per_block;
    init_est_payout_mean_hist = [init_est_payout_mean] * num_trials_per_block;
    init_est_payout_sd_hist = [init_est_payout_sd] * num_trials_per_block;
    init_total_alpha_hist = [init_total_alpha] * num_trials_per_block;
    init_total_beta_hist = [init_total_beta] * num_trials_per_block;
    variant_hist[this_variant] = Variant_History(init_block_hist, init_trial_hist, init_name_hist, init_this_pulled_hist, init_this_rank_hist, init_true_payout_hist, init_true_payout_sd_hist, init_this_true_payout, init_prior_alpha_hist, init_prior_beta_hist, init_this_counter_hist, init_this_payout_hist, init_total_payout_hist, init_est_payout_mean_hist, init_est_payout_sd_hist, init_total_alpha_hist, init_total_beta_hist);

# Assign masterdata variant hist
masterdata_variant_hist = [0] * num_variants;
for this_variant in range(num_variants): # for each variant
    init_block = [0] * num_trials_per_block * num_blocks;
    init_trial = [0] * num_trials_per_block * num_blocks;
    init_name = [init_name] * num_trials_per_block * num_blocks;
    init_this_pulled = [init_this_pulled] * num_trials_per_block * num_blocks;
    init_this_rank = [init_this_rank] * num_trials_per_block * num_blocks;
    init_true_payout = [init_true_payout] * num_trials_per_block * num_blocks;
    init_true_payout_sd = [init_true_payout_sd] * num_trials_per_block * num_blocks;
    init_this_true_payout = [init_this_true_payout] * num_trials_per_block * num_blocks;
    init_prior_alpha = [init_prior_alpha] * num_trials_per_block * num_blocks;
    init_prior_beta = [init_prior_beta] * num_trials_per_block * num_blocks;
    init_this_counter = [init_this_counter] * num_trials_per_block * num_blocks;
    init_this_payout = [init_this_payout] * num_trials_per_block * num_blocks;
    init_total_payout = [init_total_payout] * num_trials_per_block * num_blocks;
    init_est_payout_mean = [init_est_payout_mean] * num_trials_per_block * num_blocks;
    init_est_payout_sd = [init_est_payout_sd] * num_trials_per_block * num_blocks;
    init_total_alpha = [init_total_alpha] * num_trials_per_block * num_blocks;
    init_total_beta = [init_total_beta] * num_trials_per_block * num_blocks;
    masterdata_variant_hist[this_variant] = Masterdata_Variant_History(init_block, init_trial, init_name, init_this_pulled, init_this_rank, init_true_payout, init_true_payout_sd, init_this_true_payout, init_prior_alpha, init_prior_beta, init_this_counter, init_this_payout, init_total_payout, init_est_payout_mean, init_est_payout_sd, init_total_alpha, init_total_beta);

# Assign permanents
variant_sorted = sorted(variant, key=lambda x: x.true_payout, reverse=True);

# Assign transients
variant_est_sorted = sorted(variant, key=lambda x: x.est_payout_mean, reverse=True);
variant_est_sorted_ranks = [0] * num_variants;
if num_ts_iterations < 1:
    thompson_samples = [[0] * 1] * num_variants;
else:
    thompson_samples = [[0] * num_ts_iterations] * num_variants;
thompson_samples_mean = [0] * num_variants;
sorted_thompson_samples_mean = [0] * num_variants;
for this_variant in range(num_variants):
    variant_est_sorted_ranks[this_variant] = this_variant;
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
if decision_method == 3:
    for action in range(num_actions_per_trial):
       init_action = [[0] * num_actions_per_trial] * num_trials_per_block;
init_ts_random = [0] * num_trials_per_block;
init_ideal_payout = [0] * num_trials_per_block;
init_regret = [0] * num_trials_per_block;
decision_maker = Decision_Maker(init_block, init_trial, init_successes, init_failures, init_training, init_epsilon, init_action, init_ts_random, init_ideal_payout, init_regret);

# Assign masterdata decision maker
init_decision_method = [decision_method] * num_trials_per_block * num_blocks;
init_num_variants = [num_variants] * num_trials_per_block * num_blocks;
init_num_blocks = [num_blocks] * num_trials_per_block * num_blocks;
init_num_trials_per_block = [num_trials_per_block] * num_trials_per_block * num_blocks;
init_num_ts_iterations = [num_ts_iterations] * num_trials_per_block * num_blocks;
init_block = [0] * num_trials_per_block * num_blocks;
init_trial = [0] * num_trials_per_block * num_blocks;
init_successes = [0] * num_trials_per_block * num_blocks;
init_failures = [0] * num_trials_per_block * num_blocks;
init_training = [decision_training] * num_trials_per_block * num_blocks;
init_epsilon = [decision_epsilon] * num_trials_per_block * num_blocks;
if decision_method == 3:
    for action in range(num_actions_per_trial):
       init_action = [[0] * num_actions_per_trial] * num_trials_per_block;
init_ts_random = [0] * num_trials_per_block * num_blocks;
init_ideal_payout = [0] * num_trials_per_block * num_blocks;
init_regret = [0] * num_trials_per_block * num_blocks;
masterdata_decision_maker = Masterdata_Decision_Maker(init_decision_method, init_num_variants, init_num_blocks, init_num_trials_per_block, init_num_ts_iterations, init_block, init_trial, init_successes, init_failures, init_training, init_epsilon, init_action, init_ts_random, init_ideal_payout, init_regret);

# Plotter
# Beta bernoulli plot
init_this_beta_x = [[0]*101] * num_trials_per_block;
init_this_beta_y = [[0]*101] * num_trials_per_block;
plot_beta = Plot_Beta(init_this_beta_x, init_this_beta_y);
plot_every_n_counter = 0;

########################################################
## Experiment
########################################################
plt.ion()
plt.style.use('ggplot')
fig, ax = plt.subplots()

########################################################
## Block
########################################################
overall_trial_counter = -1;
for block in range(num_blocks): # for each block

    this_block = block+1;
    print(this_block);
    
    # Assign payouts to variants
    variant = [0] * num_variants;
    for this_variant in range(num_variants): # for each variant
        init_name = str(this_variant);
        init_this_pulled = 0;
        init_this_rank = 0;
        if not variant_true_payouts:
            init_true_payout = random.randint(0,100)/100;
        else:
            init_true_payout = variant_true_payouts[this_variant];
        if not variant_true_payout_sd:
            init_true_payout_sd = (random.randint(0,100) - 50) /100 * .25;
        else:
            init_true_payout_sd = variant_true_payout_sd[this_variant];
        init_this_true_payout = init_true_payout;
        init_prior_alpha = variant_priors[this_variant][0];
        init_prior_beta = variant_priors[this_variant][1];
        init_this_counter = 0;
        init_this_payout = 0;
        init_total_payout = 0;
        init_est_payout_mean = 0;
        init_est_payout_sd = 0;
        init_total_alpha = variant_priors[this_variant][0];
        init_total_beta = variant_priors[this_variant][1];
        variant[this_variant] = Variant_Obj(init_name, init_this_pulled, init_this_rank, init_true_payout, init_true_payout_sd, init_this_true_payout, init_prior_alpha, init_prior_beta, init_this_counter, init_this_payout, init_total_payout, init_est_payout_mean, init_est_payout_sd, init_total_alpha, init_total_beta);
    
    # Assign variant hist
    variant_hist = [0] * num_variants;
    for this_variant in range(num_variants): # for each variant
        init_block_hist = [0] * num_trials_per_block;
        init_trial_hist = [0] * num_trials_per_block;
        init_name_hist = [str(this_variant)] * num_trials_per_block;
        init_this_pulled_hist = [0] * num_trials_per_block;
        init_this_rank_hist = [0] * num_trials_per_block;
        if not variant_true_payouts:
            init_true_payout_hist = [variant[this_variant].true_payout] * num_trials_per_block;
        else:
            init_true_payout_hist = [variant_true_payouts[this_variant]] * num_trials_per_block;
        if not variant_true_payout_sd:
            init_true_payout_sd_hist = [variant[this_variant].true_payout_sd] * num_trials_per_block;
        else:
            init_true_payout_sd_hist = [variant_true_payout_sd[this_variant]] * num_trials_per_block;
        init_this_true_payout = [0] * num_trials_per_block;
        init_prior_alpha_hist = [variant_priors[this_variant][0]] * num_trials_per_block;
        init_prior_beta_hist = [variant_priors[this_variant][1]] * num_trials_per_block;
        init_this_counter_hist = [0] * num_trials_per_block;
        init_this_payout_hist = [0] * num_trials_per_block;
        init_total_payout_hist = [0] * num_trials_per_block;
        init_est_payout_mean_hist = [0] * num_trials_per_block;
        init_est_payout_sd_hist = [0] * num_trials_per_block;
        init_total_alpha_hist = [variant_priors[this_variant][0]] * num_trials_per_block;
        init_total_beta_hist = [variant_priors[this_variant][1]] * num_trials_per_block;
        variant_hist[this_variant] = Variant_History(init_block_hist, init_trial_hist, init_name_hist, init_this_pulled_hist, init_this_rank_hist, init_true_payout_hist, init_true_payout_sd_hist, init_this_true_payout, init_prior_alpha_hist, init_prior_beta_hist, init_this_counter_hist, init_this_payout_hist, init_total_payout_hist, init_est_payout_mean_hist, init_est_payout_sd_hist, init_total_alpha_hist, init_total_beta_hist);
        
    # Assign decision maker
    init_block = [0] * num_trials_per_block;
    init_trial = [0] * num_trials_per_block;
    init_successes = [0] * num_trials_per_block;
    init_failures = [0] * num_trials_per_block;
    init_training = [decision_training] * num_trials_per_block;
    init_epsilon = [decision_epsilon] * num_trials_per_block;
    if decision_method == 3:
        for action in range(num_actions_per_trial):
            init_action = [[0] * num_actions_per_trial] * num_trials_per_block;
    init_ts_random = [0] * num_trials_per_block;
    init_ideal_payout = [0] * num_trials_per_block;
    init_regret = [0] * num_trials_per_block;
    decision_maker = Decision_Maker(init_block, init_trial, init_successes, init_failures, init_training, init_epsilon, init_action, init_ts_random, init_ideal_payout, init_regret);

    # Assign permanents
    variant_sorted = sorted(variant, key=lambda x: x.true_payout, reverse=True);

    # Assign transients
    variant_est_sorted = sorted(variant, key=lambda x: x.est_payout_mean, reverse=True);
    variant_est_sorted_ranks = [0] * num_variants;
    if num_ts_iterations < 1:
        thompson_samples = [[0] * 1] * num_variants;
    else:
        thompson_samples = [[0] * num_ts_iterations] * num_variants;
    thompson_samples_mean = [0] * num_variants;
    sorted_thompson_samples_mean = [0] * num_variants;
    for this_variant in range(num_variants):
        variant_est_sorted_ranks[this_variant] = this_variant;
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
    
    ########################################################
    ## Trial
    ########################################################
    for trial in range(num_trials_per_block): # for each trial
        overall_trial_counter = overall_trial_counter+1;
        this_trial = trial+1;
        if print_details == 1:
            print('block:'+str(this_block)+' trial:'+str(this_trial))
        
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
                
        # plot every n
        plot_every_n_counter += 1;
        if plot_every_n_counter == plot_every_n:
            plot_every_n_counter = 0;
            
        # dynamic variants change
        for this_variant in range(num_variants):
            if variant[this_variant].true_payout_sd < 0:
                variant[this_variant].this_true_payout = round(variant[this_variant].true_payout - random.randint(0,abs(int(variant[this_variant].true_payout_sd * 10000)))/10000,2);
            else:
                variant[this_variant].this_true_payout = round(variant[this_variant].true_payout + random.randint(0,int(variant[this_variant].true_payout_sd * 10000))/10000,2);
        
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
            for this_parallel_action in range(num_actions_per_trial):
                while decision_maker.action[trial][this_parallel_action] == decision_maker.action[trial][this_parallel_action-1]:
                    decision_maker.action[trial][this_parallel_action] = random.randint(0,num_variants-1);
        elif decision_method == 1: #e-greedy sampling
            if this_trial <= round(num_trials_per_block * decision_maker.training[trial],0): # if training trial then sample randomly
                for this_parallel_action in range(num_actions_per_trial):
                    while decision_maker.action[trial][this_parallel_action] == decision_maker.action[trial][this_parallel_action-1]:
                        decision_maker.action[trial][this_parallel_action] = random.randint(0,num_variants-1);
            else: #if training is over then sample by epsilon
                this_roll = random.randint(0,100)/100;
                if this_roll <= decision_maker.epsilon[trial]:
                    for this_parallel_action in range(num_actions_per_trial):
                        if this_parallel_action > 0:
                            while decision_maker.action[trial][this_parallel_action] == decision_maker.action[trial][this_parallel_action-1]:
                                decision_maker.action[trial][this_parallel_action] = random.randint(0,num_variants-1);
                        while decision_maker.action[trial][this_parallel_action] == variant_est_sorted_ranks[this_parallel_action]:
                            if this_parallel_action > 0:
                                while decision_maker.action[trial][this_parallel_action] == decision_maker.action[trial][this_parallel_action-1]:
                                    decision_maker.action[trial][this_parallel_action] = random.randint(0,num_variants-1);
                else:
                    for this_parallel_action in range(num_actions_per_trial):
                        decision_maker.action[trial][this_parallel_action] = variant_est_sorted_ranks[this_parallel_action];
        elif decision_method == 2: #thompson sampling
            if num_ts_iterations < 1:
                #epsilon
                this_roll = random.randint(0,100)/100;
                #random sampling
                if this_roll >= num_ts_iterations:
                    decision_maker.ts_random[trial] = 1;
                    decision_maker.action[trial] = random.randint(0,num_variants-1);
                else:
                    #thompson sampling
                    for this_variant in range(num_variants):
                        thompson_samples[this_variant][0] = np.random.beta(variant[this_variant].total_alpha, variant[this_variant].total_beta, size=None);
                        thompson_samples_mean[this_variant] = np.mean(thompson_samples[this_variant]);
                    for this_parallel_action in range(num_actions_per_trial):
                        sorted_thompson_samples_mean = sorted(thompson_samples_mean)[::-1];
                        decision_maker.action[trial][this_parallel_action] = thompson_samples_mean.index(sorted_thompson_samples_mean[this_parallel_action]);
            else:
                #thompson sampling
                for this_variant in range(num_variants):
                    for this_iteration in range(num_ts_iterations):
                        thompson_samples[this_variant][this_iteration] = np.random.beta(variant[this_variant].total_alpha, variant[this_variant].total_beta, size=None);
                    thompson_samples_mean[this_variant] = np.mean(thompson_samples[this_variant]);
                for this_parallel_action in range(num_actions_per_trial):
                    sorted_thompson_samples_mean = sorted(thompson_samples_mean)[::-1];
                    decision_maker.action[trial][this_parallel_action] = thompson_samples_mean.index(sorted_thompson_samples_mean[this_parallel_action]);
        elif decision_method == 3: #parallel dynamic
            if num_ts_iterations < 1:
                #epsilon
                this_roll = random.randint(0,100)/100;
                #random sampling
                if this_roll >= num_ts_iterations:
                    decision_maker.ts_random[trial] = 1;
                    for this_parallel_action in range(num_actions_per_trial):
                        decision_maker.action[trial][this_parallel_action] = random.randint(0,num_variants-1);
                else:
                    #thompson sampling
                    for this_variant in range(num_variants):
                        thompson_samples[this_variant][0] = np.random.beta(variant[this_variant].total_alpha, variant[this_variant].total_beta, size=None);
                        thompson_samples_mean[this_variant] = np.mean(thompson_samples[this_variant]);
                    for this_parallel_action in range(num_actions_per_trial):
                        sorted_thompson_samples_mean = sorted(thompson_samples_mean)[::-1];
                        decision_maker.action[trial][this_parallel_action] = thompson_samples_mean.index(sorted_thompson_samples_mean[this_parallel_action]);
            else:
                #thompson sampling
                for this_variant in range(num_variants):
                    for this_iteration in range(num_ts_iterations):
                        thompson_samples[this_variant][this_iteration] = np.random.beta(variant[this_variant].total_alpha, variant[this_variant].total_beta, size=None);
                    thompson_samples_mean[this_variant] = np.mean(thompson_samples[this_variant]);
                for this_parallel_action in range(num_actions_per_trial):
                    sorted_thompson_samples_mean = sorted(thompson_samples_mean)[::-1];
                    decision_maker.action[trial][this_parallel_action] = thompson_samples_mean.index(sorted_thompson_samples_mean[this_parallel_action]);
                
        ########################################################
        ## Variants
        ########################################################
        # clear previous plot
        if plot_every_n_counter == 0:
            plt.cla();
        
        # determine payouts
        for this_variant in range(num_variants): # for each variant
            variant_hist[this_variant].block[trial] = this_block;
            variant_hist[this_variant].trial[trial] = this_trial;
            # within-trial decision-reward parameters
            if this_variant in decision_maker.action[trial][0:num_actions_per_trial]: #if this is the chose variable
                variant[this_variant].this_pulled = 1;
                variant[this_variant].this_counter = variant[this_variant].this_counter + 1;
                this_roll = random.randint(0,100)/100; # roll random # from 0 to 100
                if this_roll <= variant[this_variant].true_payout: # if roll <= prior payout
                    variant[this_variant].this_payout = 1;
                    variant[this_variant].total_payout = variant[this_variant].total_payout + 1;
                    decision_maker.successes[trial] += 1;
                else: # if roll > prior payout
                    variant[this_variant].this_payout = variant[this_variant].this_payout;
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
            
            if print_details == 1:
                print(str(variant[this_variant].this_pulled)+' V'+str(this_variant)+' e_r:'+str(variant_est_sorted[this_variant].name)+' tru:'+str(variant[this_variant].this_true_payout)+' pay:'+str(variant[this_variant].total_payout)+'/'+str(variant[this_variant].this_counter)+' e_m:'+str(round(variant[this_variant].est_payout_mean,2))+' e_sd:'+str(round(variant[this_variant].est_payout_sd,2))+' a:'+str(variant[this_variant].total_alpha)+' b:'+str(variant[this_variant].total_beta)+' ts_m:'+str(round(thompson_samples_mean[this_variant],2)));
            
            # save trial statistics data
            variant_hist[this_variant].est_payout_mean[trial] = variant[this_variant].est_payout_mean;
            variant_hist[this_variant].est_payout_sd[trial] = variant[this_variant].est_payout_sd;
            
            # plot trial statistics
            for x in range(len(plot_beta.this_beta_x[trial])):
                xi = x / 100;
                plot_beta.this_beta_x[trial][x] = xi;
                # normal distribution
                # plot_beta.this_beta_y[trial][x] = (1 / variant[this_variant].est_payout_sd * np.sqrt(2 * np.pi)) ** (-1/2 * ((xi - variant[this_variant].est_payout_mean) / variant[this_variant].est_payout_sd) ** 2);
                # beta-bernoulli distribution
                plot_beta.this_beta_y[trial][x] = ((1.000 / ((math.factorial(variant[this_variant].total_alpha - 1.000) * math.factorial(variant[this_variant].total_beta - 1.000)) / (math.factorial(variant[this_variant].total_alpha + variant[this_variant].total_beta - 1.000)))) * ((math.pow(xi, (variant[this_variant].total_alpha - 1.000))) * (math.pow((1.000 - xi), (variant[this_variant].total_beta - 1.000)))));
            
            # colours start full opacity but fade with age
            plot_colours[this_variant][3] = 1.00;
            if plot_every_n_counter == 0:
                plt.plot(plot_beta.this_beta_x[trial], plot_beta.this_beta_y[trial], color = [plot_colours[this_variant][0], plot_colours[this_variant][1], plot_colours[this_variant][2], plot_colours[this_variant][3]]);
            
            # plot true_payout
            plt.axvline(x = variant[this_variant].true_payout, ymin=0, ymax=1, color = [plot_colours[this_variant][0], plot_colours[this_variant][1], plot_colours[this_variant][2]], ls = '--');
            # plot true_payout_sd
            plt.axvline(x = variant[this_variant].true_payout+variant[this_variant].true_payout_sd, ymin=0, ymax=1, color = [plot_colours[this_variant][0], plot_colours[this_variant][1], plot_colours[this_variant][2]], ls = ':');
            plt.axvline(x = variant[this_variant].true_payout-variant[this_variant].true_payout_sd, ymin=0, ymax=1, color = [plot_colours[this_variant][0], plot_colours[this_variant][1], plot_colours[this_variant][2]], ls = ':');
        
        ########################################################
        ## End of trial
        ########################################################
        # regret analysis
        for this_parallel_action in range(num_actions_per_trial):
            if (decision_maker.action[trial][this_parallel_action] == int(variant_sorted[this_parallel_action].name)):
                if (variant[int(variant_sorted[this_parallel_action].name)].this_payout > 0):
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
        
        # print
        if print_details == 1:
            print('total success rate:'+str(sum(decision_maker.successes))+'/'+str(sum(decision_maker.successes)+sum(decision_maker.failures)));
            print('ideal success rate: '+str(sum(decision_maker.ideal_payout))+'/'+str(sum(decision_maker.successes)+sum(decision_maker.failures)));
            print('regret: '+str(decision_maker.regret[trial]));
        
        # bernoulli plot
        if plot_every_n_counter == 0:
            plt.title('trial '+str(this_trial)+' / '+str(num_trials_per_block));
            plt.xlabel('p');
            plt.ylabel('beta');
            plt.show();
            plt.pause(0.01);
            
        # amend trial data to masterdata
        for this_variant in range(num_variants):
            masterdata_variant_hist[this_variant].block[overall_trial_counter] = variant_hist[this_variant].block[trial];
            masterdata_variant_hist[this_variant].trial[overall_trial_counter] = variant_hist[this_variant].trial[trial];
            masterdata_variant_hist[this_variant].name[overall_trial_counter] = variant_hist[this_variant].name[trial];
            masterdata_variant_hist[this_variant].this_pulled[overall_trial_counter] = variant_hist[this_variant].this_pulled[trial];
            masterdata_variant_hist[this_variant].this_rank[overall_trial_counter] = variant_hist[this_variant].this_rank[trial];
            masterdata_variant_hist[this_variant].this_payout[overall_trial_counter] = variant_hist[this_variant].this_payout[trial];
            masterdata_variant_hist[this_variant].true_payout[overall_trial_counter] = variant_hist[this_variant].true_payout[trial];
            masterdata_variant_hist[this_variant].prior_alpha[overall_trial_counter] = variant_hist[this_variant].prior_alpha[trial];
            masterdata_variant_hist[this_variant].prior_beta[overall_trial_counter] = variant_hist[this_variant].prior_beta[trial];
            masterdata_variant_hist[this_variant].this_counter[overall_trial_counter] = variant_hist[this_variant].this_counter[trial];
            masterdata_variant_hist[this_variant].this_payout[overall_trial_counter] = variant_hist[this_variant].this_payout[trial];
            masterdata_variant_hist[this_variant].total_payout[overall_trial_counter] = variant_hist[this_variant].total_payout[trial];
            masterdata_variant_hist[this_variant].est_payout_mean[overall_trial_counter] = variant_hist[this_variant].est_payout_mean[trial];
            masterdata_variant_hist[this_variant].est_payout_sd[overall_trial_counter] = variant_hist[this_variant].est_payout_sd[trial];
            masterdata_variant_hist[this_variant].total_alpha[overall_trial_counter] = variant_hist[this_variant].total_alpha[trial];
            masterdata_variant_hist[this_variant].total_beta[overall_trial_counter] = variant_hist[this_variant].total_beta[trial];
        
        masterdata_decision_maker.block[overall_trial_counter] = decision_maker.block[trial];
        masterdata_decision_maker.trial[overall_trial_counter] = decision_maker.trial[trial];
        masterdata_decision_maker.successes[overall_trial_counter] = decision_maker.successes[trial];
        masterdata_decision_maker.failures[overall_trial_counter] = decision_maker.failures[trial];
        masterdata_decision_maker.training[overall_trial_counter] = decision_maker.training[trial];
        masterdata_decision_maker.epsilon[overall_trial_counter] = decision_maker.epsilon[trial];
        masterdata_decision_maker.action[overall_trial_counter] = decision_maker.action[trial];
        masterdata_decision_maker.ts_random[overall_trial_counter] = decision_maker.ts_random[trial];
        masterdata_decision_maker.ideal_payout[overall_trial_counter] = decision_maker.ideal_payout[trial];
        masterdata_decision_maker.regret[overall_trial_counter] = decision_maker.regret[trial];
        
        # segmentation line in console
        if print_details == 1:
            print('########################################################################');
        
    ########################################################
    ## End of block
    ########################################################
    # plot cumulative regret
    if plot_regret == 1:
        plt.title('cumulative regret over '+str(num_trials_per_block)+' trials');
        plt.xlabel('trial');
        plt.ylabel('cumulative regret');
        plt.plot(decision_maker.regret);
        plt.show();
    # save bernoulli plot
    # save cumulative regret plot

########################################################
## End of experiment
########################################################
# plot cumulative regret
if plot_regret == 1:
    plt.title('cumulative regret over '+str(num_trials_per_block*num_blocks)+' trials');
    plt.xlabel('trial');
    plt.ylabel('cumulative regret');
    plt.plot(masterdata_decision_maker.regret);
    plt.show();

if save_data_to_harddrive == 1:
    # save datafile
    #masterdata_decision_maker
    members = [attr for attr in dir(masterdata_decision_maker) if not callable(getattr(masterdata_decision_maker, attr)) and not attr.startswith("__")];
    values = [getattr(masterdata_decision_maker, member) for member in members];
    expert_data = 0;
    with open("decision_maker.csv",'w') as resultFile:
        wr = csv.writer(resultFile);
        wr.writerow(members);
        export_data = zip(values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8],values[9],values[10],values[11],values[12],values[13],values[14]);
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
