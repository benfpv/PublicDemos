#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:22:42 2021

@author: root
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import scipy.signal 
import math
from scipy.stats import beta
from collections import namedtuple
import os
import shutil
import csv
import time

#import tensorflow as tf
#from tensorflow import keras

########################################################
## Warning! This is warning related.
########################################################
import warnings
warnings.filterwarnings('ignore') # ComplexWarning: Casting complex values to real discards the imaginary part. To do with Frequency Analysis.

########################################################
## Introduction
########################################################
# Welcome!
# Predict futures based on serial input.
# TODO/WARNINGS:
# **Warning - Double check nyquist.
# **Refactor for Performance?

########################################################
## Quick Options
########################################################
variant_names = ['TESTA']; #name - e.g., BTC
variant_rawCurrency = ['USD']; #native or raw currency type of data input - e.g., USD
#1440m = 1day, 10080m = 7.5days, 43200m = 30days.
variant_timeScales = [10, 60, 360]; #time scale in mins - e.g., 5 min per tick. Each value must be divisible by the minimum value without remainder.
variant_tickWindows = [32]; #tick window in the unit of timeScale - e.g., 10 ticks of timeScale units.

variant_tickWindowsPredict = [32]; #length of tickWindow to predict - e.g., predict next 1 tick.

FIRTickWindowsMin = 7;
FIRTickWindowsMax = 9; #variant_tickWindows[0] * .25;
FIRTickWindowsStep = 2;

########################################################
## Debug Options
########################################################
debugTimers = 0; #0 off 1 on.

################################################################################################################
## Initialize
################################################################################################################
print('Initializing');

########################################################
## Classes
########################################################
class VariantData(): #variant data
    def __init__(self, name, rawCurrency, timeScales, tickWindows, fs, nyq, dt, debugoNoiseValues, rawValues, linRegressValues, linRegressStats, linRegressSD, movingAverageValues, movingSDValues, staticPeaks, staticSupports, staticPeaksCollapsed, staticSupportsCollapsed, staticPeaksCollapsedInterp, staticSupportsCollapsedInterp, staticPeakIndeces, staticSupportIndeces, linRegressPeaksStats, linRegressPeaks, linRegressPeaksSD, linRegressSupportsStats, linRegressSupports, linRegressSupportsSD, linRegressMovingPeaks, linRegressMovingSupports, movingPeaks, movingPeaksSD, movingSupports, movingSupportsSD, fHat, fPSD, fDist, fFilterLevel, fPSDFilteredHat, fInverseValues, FIRFilteredValues, fHilb, fAmpEnv, fInstPhase, fInstFreq):
        self.name = name; #name/identifier - e.g., BTC
        self.rawCurrency = rawCurrency; #native currency type - e.g., USD
        self.timeScales = timeScales; #time scale - e.g., 5 min per tick
        self.tickWindows = tickWindows; #time window in the unit of timeScale - e.g., 10 ticks of timeScale units.
        self.fs = fs;
        self.nyq = nyq;
        self.dt = dt;
        self.debugoNoiseValues = debugoNoiseValues;
        self.rawValues = rawValues; #value in native currency type - e.g., 1 BTC
        self.linRegressValues = linRegressValues;
        self.linRegressStats = linRegressStats;
        self.linRegressSD = linRegressSD;
        self.movingAverageValues = movingAverageValues;
        self.movingSDValues = movingSDValues;
        self.staticPeaks = staticPeaks;
        self.staticSupports = staticSupports;
        self.staticPeaksCollapsed = staticPeaksCollapsed;
        self.staticSupportsCollapsed = staticSupportsCollapsed;
        self.staticPeaksCollapsedInterp = staticPeaksCollapsedInterp;
        self.staticSupportsCollapsedInterp = staticSupportsCollapsedInterp;
        self.staticPeakIndeces = staticPeakIndeces;
        self.staticSupportIndeces = staticSupportIndeces;
        self.linRegressPeaksStats = linRegressPeaksStats;
        self.linRegressPeaks = linRegressPeaks;
        self.linRegressPeaksSD = linRegressPeaksSD;
        self.linRegressSupportsStats = linRegressSupportsStats;
        self.linRegressSupports = linRegressSupports;
        self.linRegressSupportsSD = linRegressSupportsSD;
        self.linRegressMovingPeaks = linRegressMovingPeaks;
        self.linRegressMovingSupports = linRegressMovingSupports;
        self.movingPeaks = movingPeaks;
        self.movingPeaksSD = movingPeaksSD;
        self.movingSupports = movingSupports;
        self.movingSupportsSD = movingSupportsSD;
        self.fHat = fHat;
        self.fPSD = fPSD;
        self.fDist = fDist;
        self.fFilterLevel = fFilterLevel;
        self.fPSDFilteredHat = fPSDFilteredHat;
        self.fInverseValues = fInverseValues;
        self.FIRFilteredValues = FIRFilteredValues;
        self.fHilb = fHilb;
        self.fAmpEnv = fAmpEnv;
        self.fInstPhase = fInstPhase;
        self.fInstFreq = fInstFreq;
        
class VariantPredict(): #variant predictions
    def __init__(self, name, rawCurrency, timeScales, tickWindows, rawStaticMean, rawStaticSD, linRegressValues, linRegressStats, linRegressSD, movingAverage2DStats, movingAverage2D, movingAverage3DStats, movingAverage3D, movingAverage4DStats, movingAverage4D, movingAverage5DStats, movingAverage5D, movingSDValues, staticPeaksCollIntMean, staticPeaksCollIntSD, staticSupportsCollIntMean, staticSupportsCollIntSD, linRegressPeaksStats, linRegressPeaks, linRegressPeaksSD, linRegressSupportsStats, linRegressSupports, linRegressSupportsSD, movingPeaks2DStats, movingPeaks2D, movingSupports2DStats, movingSupports2D, movingPeaks3DStats, movingPeaks3D, movingSupports3DStats, movingSupports3D, movingPeaks4DStats, movingPeaks4D, movingSupports4DStats, movingSupports4D, movingPeaks5DStats, movingPeaks5D, movingSupports5DStats, movingSupports5D, fInverseValuesStats ,fInverseValues):
        self.name = name; #name/identifier - e.g., BTC
        self.rawCurrency = rawCurrency; #native currency type - e.g., USD
        self.timeScales = timeScales; #time scale - e.g., 5 min per tick
        self.tickWindows = tickWindows; #time window in the unit of timeScale - e.g., 10 ticks of timeScale units.
        self.rawStaticMean = rawStaticMean;
        self.rawStaticSD = rawStaticSD;
        self.linRegressValues =  linRegressValues;
        self.linRegressStats = linRegressStats;
        self.linRegressSD = linRegressSD;
        self.movingAverage2DStats = movingAverage2DStats;
        self.movingAverage2D = movingAverage2D;
        self.movingAverage3DStats = movingAverage3DStats;
        self.movingAverage3D = movingAverage3D;
        self.movingAverage4DStats = movingAverage4DStats;
        self.movingAverage4D = movingAverage4D;
        self.movingAverage5DStats = movingAverage5DStats;
        self.movingAverage5D = movingAverage5D;
        self.staticPeaksCollIntMean = staticPeaksCollIntMean;
        self.staticPeaksCollIntSD = staticPeaksCollIntSD;
        self.staticSupportsCollIntMean = staticSupportsCollIntMean;
        self.staticSupportsCollIntSD = staticSupportsCollIntSD;
        self.linRegressPeaksStats = linRegressPeaksStats;
        self.linRegressPeaks = linRegressPeaks;
        self.linRegressPeaksSD = linRegressPeaksSD;
        self.linRegressSupportsStats = linRegressSupportsStats;
        self.linRegressSupports = linRegressSupports;
        self.linRegressSupportsSD = linRegressSupportsSD;
        self.movingPeaks2DStats = movingPeaks2DStats;
        self.movingPeaks2D = movingPeaks2D;
        self.movingSupports2DStats = movingSupports2DStats;
        self.movingSupports2D = movingSupports2D;
        self.movingPeaks3DStats = movingPeaks3DStats;
        self.movingPeaks3D = movingPeaks3D;
        self.movingSupports3DStats = movingSupports3DStats;
        self.movingSupports3D = movingSupports3D;
        self.movingPeaks4DStats = movingPeaks4DStats;
        self.movingPeaks4D = movingPeaks4D;
        self.movingSupports4DStats = movingSupports4DStats;
        self.movingSupports4D = movingSupports4D;
        self.movingPeaks5DStats = movingPeaks5DStats;
        self.movingPeaks5D = movingPeaks5D;
        self.movingSupports5DStats = movingSupports5DStats;
        self.movingSupports5D = movingSupports5D;
        self.fInverseValuesStats = fInverseValuesStats;
        self.fInverseValues = fInverseValues;

FIRTickWindows = np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep, dtype='int');

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w;

def objective2D(x,a,b,c):
    return a * x**2 + b * x + c;

def objective3D(x,a,b,c,d):
    return a * x**3 + b * x**2 + c * x + d;

def objective4D(x,a,b,c,d,e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e;

def objective5D(x,a,b,c,d,e,f):
    return a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f;

def objectiveSin(x,a,b,c,d):
    return a * np.sin(b * x + c) + d;

# def NN_U1IS1E500P1(xs, ys):
#     # Take in X and Y into NN(units=1, input_shape=[1]) over 100 epochs to predict next tick of x.
#     model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]);
#     model.compile(optimizer='sgd', loss='mean_squared_logarithmic_error');
#     #xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float);
#     #ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float);
#     model.fit(xs, ys, epochs=500);
#     xPatternLastDiff = (xs[-1] - xs[-2]);
#     xPatternGuessMultiplier = xPatternLastDiff / (xs[-2] - xs[-3]);
#     xPatternGuess = xs[-1] + xPatternLastDiff * xPatternGuessMultiplier; #Works up to 1d exponential x movement.
#     predicted = model.predict([xPatternGuess]);
#     print('NN_U1IS1E500P1 - Prediction:' + str(predicted));
#     return predicted;

########################################################
## Establish VariantData Structures
########################################################
variantData = [0] * len(variant_names);
for var in range(len(variant_names)):
    init_rawValues = [];
    init_name = variant_names[var];
    init_rawCurrency = variant_rawCurrency[var];
    init_timeScales = variant_timeScales;
    init_tickWindows = variant_tickWindows;
    init_fs = np.empty((len(variant_timeScales)));
    init_fs[:] = np.NaN;
    init_nyq = np.empty((len(variant_timeScales)));
    init_nyq[:] = np.NaN;
    init_dt = np.empty((len(variant_timeScales)));
    init_dt[:] = np.NaN;
    init_debugoNoiseValues = np.empty((len(variant_timeScales), variant_tickWindows[0]));
    init_debugoNoiseValues[:] = np.NaN;
    init_rawValues = np.empty((len(variant_timeScales), variant_tickWindows[0]));
    init_rawValues[:] = np.NaN;
    init_linRegressValues = np.empty((len(variant_timeScales), variant_tickWindows[0]));
    init_linRegressValues[:] = np.NaN;
    init_linRegressStats = np.empty((len(variant_timeScales), len(sp.stats.linregress([0],[0]))));
    init_linRegressStats[:] = np.NaN;      
    init_linRegressSD = np.empty((len(variant_timeScales), variant_tickWindows[0]));
    init_linRegressSD[:] = np.NaN;
    init_movingAverageValues = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_movingAverageValues[:] = np.NaN;
    init_movingSDValues = np.empty((len(variant_timeScales),len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_movingSDValues[:] = np.NaN;
    init_staticPeaks = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_staticPeaks[:] = np.NaN;
    init_staticSupports = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_staticSupports[:] = np.NaN;
    init_staticPeaksCollapsed = np.empty((len(variant_timeScales), variant_tickWindows[0]));
    init_staticPeaksCollapsed[:] = np.NaN;
    init_staticSupportsCollapsed = np.empty((len(variant_timeScales), variant_tickWindows[0]));
    init_staticSupportsCollapsed[:] = np.NaN;
    init_staticPeaksCollapsedInterp = np.empty((len(variant_timeScales), variant_tickWindows[0]));
    init_staticPeaksCollapsedInterp[:] = np.NaN;
    init_staticSupportsCollapsedInterp = np.empty((len(variant_timeScales), variant_tickWindows[0]));
    init_staticSupportsCollapsedInterp[:] = np.NaN;
    init_staticPeakIndeces = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]), dtype = 'int');
    init_staticPeakIndeces[:] = np.NaN;
    init_staticSupportIndeces = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]), dtype = 'int');
    init_staticSupportIndeces[:] = np.NaN;
    init_linRegressPeaksStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), len(sp.stats.linregress([0],[0]))));
    init_linRegressPeaksStats[:] = np.NaN;
    init_linRegressPeaks = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_linRegressPeaks[:] = np.NaN;
    init_linRegressPeaksSD = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_linRegressPeaksSD[:] = np.NaN;
    init_linRegressSupportsStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), len(sp.stats.linregress([0],[0]))));
    init_linRegressSupportsStats[:] = np.NaN;
    init_linRegressSupports = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_linRegressSupports[:] = np.NaN;
    init_linRegressSupportsSD = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_linRegressSupportsSD[:] = np.NaN;
    init_linRegressMovingPeaks = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_linRegressMovingPeaks[:] = np.NaN;
    init_linRegressMovingSupports = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_linRegressMovingSupports[:] = np.NaN;
    init_movingPeaks = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_movingPeaks[:] = np.NaN;
    init_movingPeaksSD = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_movingPeaksSD[:] = np.NaN;
    init_movingSupports = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_movingSupports[:] = np.NaN;
    init_movingSupportsSD = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]));
    init_movingSupportsSD[:] = np.NaN;
    init_fHat = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fHat[:] = np.NaN;
    init_fPSD = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fPSD[:] = np.NaN;
    init_fDist = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fDist[:] = np.NaN;
    init_fFilterLevel = np.empty((len(variant_timeScales)));
    init_fFilterLevel[:] = np.NaN;
    init_fPSDFilteredHat = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fPSDFilteredHat[:] = np.NaN;
    init_fInverseValues = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fInverseValues[:] = np.NaN;
    init_FIRFilteredValues = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindows[0]), dtype = 'complex_');
    init_FIRFilteredValues[:] = np.NaN;
    init_fHilb = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fHilb[:] = np.NaN;
    init_fAmpEnv = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fAmpEnv[:] = np.NaN;
    init_fInstPhase = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fInstPhase[:] = np.NaN;
    init_fInstFreq = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fInstFreq[:] = np.NaN;
    variantData[var] = VariantData(init_name, init_rawCurrency, init_timeScales, init_tickWindows, init_fs, init_nyq, init_dt, init_debugoNoiseValues, init_rawValues, init_linRegressValues, init_linRegressStats, init_linRegressSD, init_movingAverageValues, init_movingSDValues, init_staticPeaks, init_staticSupports, init_staticPeaksCollapsed, init_staticSupportsCollapsed, init_staticPeaksCollapsedInterp, init_staticSupportsCollapsedInterp, init_staticPeakIndeces, init_staticSupportIndeces, init_linRegressPeaksStats, init_linRegressPeaks, init_linRegressPeaksSD, init_linRegressSupportsStats, init_linRegressSupports, init_linRegressSupportsSD, init_linRegressMovingPeaks, init_linRegressMovingSupports, init_movingPeaks, init_movingPeaksSD, init_movingSupports, init_movingSupportsSD, init_fHat, init_fPSD, init_fDist, init_fFilterLevel, init_fPSDFilteredHat, init_fInverseValues, init_FIRFilteredValues, init_fHilb, init_fAmpEnv, init_fInstPhase, init_fInstFreq);

########################################################
## Establish VariantPredict Structures
########################################################
variantPredict = [0] * len(variant_names);
for var in range(len(variant_names)):
    init_name = variant_names[var];
    init_rawCurrency = variant_rawCurrency[var];
    init_timeScales = variant_timeScales;
    init_tickWindows = variant_tickWindows;
    init_rawStaticMean = np.empty((len(variant_timeScales), variant_tickWindowsPredict[0]));
    init_rawStaticMean[:] = np.NaN;
    init_rawStaticSD = np.empty((len(variant_timeScales), variant_tickWindowsPredict[0]));
    init_rawStaticSD[:] = np.NaN;
    init_linRegressValues = np.empty((len(variant_timeScales), variant_tickWindowsPredict[0]));
    init_linRegressValues[:] = np.NaN;
    init_linRegressStats = np.empty((len(variant_timeScales), len(sp.stats.linregress([0],[0]))));
    init_linRegressStats[:] = np.NaN;
    init_linRegressSD = np.empty((len(variant_timeScales), variant_tickWindowsPredict[0]));
    init_linRegressSD[:] = np.NaN;
    init_movingAverage2DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 3));
    init_movingAverage2DStats[:] = np.NaN;
    init_movingAverage2D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingAverage2D[:] = np.NaN;
    init_movingAverage3DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 4));
    init_movingAverage3DStats[:] = np.NaN;
    init_movingAverage3D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingAverage3D[:] = np.NaN;
    init_movingAverage4DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 5));
    init_movingAverage4DStats[:] = np.NaN;
    init_movingAverage4D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingAverage4D[:] = np.NaN;
    init_movingAverage5DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 6));
    init_movingAverage5DStats[:] = np.NaN;
    init_movingAverage5D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingAverage5D[:] = np.NaN;
    init_movingSDValues = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingSDValues[:] = np.NaN;
    init_staticPeaksCollIntMean = np.empty((len(variant_timeScales), variant_tickWindowsPredict[0]));
    init_staticPeaksCollIntMean[:] = np.NaN;
    init_staticPeaksCollIntSD = np.empty((len(variant_timeScales), variant_tickWindowsPredict[0]));
    init_staticPeaksCollIntSD[:] = np.NaN;
    init_staticSupportsCollIntMean = np.empty((len(variant_timeScales), variant_tickWindowsPredict[0]));
    init_staticSupportsCollIntMean[:] = np.NaN;
    init_staticSupportsCollIntSD = np.empty((len(variant_timeScales), variant_tickWindowsPredict[0]));
    init_staticSupportsCollIntSD[:] = np.NaN;
    init_linRegressPeaksStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_linRegressPeaksStats[:] = np.NaN;
    init_linRegressPeaks = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_linRegressPeaks[:] = np.NaN;
    init_linRegressPeaksSD = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_linRegressPeaksSD[:] = np.NaN;
    init_linRegressSupportsStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_linRegressSupportsStats[:] = np.NaN;
    init_linRegressSupports = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_linRegressSupports[:] = np.NaN;
    init_linRegressSupportsSD = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_linRegressSupportsSD[:] = np.NaN;
    init_movingPeaks2DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 3));
    init_movingPeaks2DStats[:] = np.NaN;
    init_movingPeaks2D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingPeaks2D[:] = np.NaN;
    init_movingSupports2DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 3));
    init_movingSupports2DStats[:] = np.NaN;
    init_movingSupports2D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingSupports2D[:] = np.NaN;
    init_movingPeaks3DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 4));
    init_movingPeaks3DStats[:] = np.NaN;
    init_movingPeaks3D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingPeaks3D[:] = np.NaN;
    init_movingSupports3DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 4));
    init_movingSupports3DStats[:] = np.NaN;
    init_movingSupports3D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingSupports3D[:] = np.NaN;
    init_movingPeaks4DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 5));
    init_movingPeaks4DStats[:] = np.NaN;
    init_movingPeaks4D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingPeaks4D[:] = np.NaN;
    init_movingSupports4DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 5));
    init_movingSupports4DStats[:] = np.NaN;
    init_movingSupports4D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingSupports4D[:] = np.NaN;
    init_movingPeaks5DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 6));
    init_movingPeaks5DStats[:] = np.NaN;
    init_movingPeaks5D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingPeaks5D[:] = np.NaN;
    init_movingSupports5DStats = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), 6));
    init_movingSupports5DStats[:] = np.NaN;
    init_movingSupports5D = np.empty((len(variant_timeScales), len(np.arange(FIRTickWindowsMin, FIRTickWindowsMax, FIRTickWindowsStep)), variant_tickWindowsPredict[0]));
    init_movingSupports5D[:] = np.NaN;
    init_fInverseValuesStats = np.empty((len(variant_timeScales), 4));
    init_fInverseValuesStats[:] = np.NaN;
    init_fInverseValues = np.empty((len(variant_timeScales), variant_tickWindowsPredict[0]));
    init_fInverseValues[:] = np.NaN;
    variantPredict[var] = VariantPredict(init_name, init_rawCurrency, init_timeScales, init_tickWindows, init_rawStaticMean, init_rawStaticSD, init_linRegressValues, init_linRegressStats, init_linRegressSD, init_movingAverage2DStats, init_movingAverage2D, init_movingAverage3DStats, init_movingAverage3D, init_movingAverage4DStats, init_movingAverage4D, init_movingAverage5DStats, init_movingAverage5D, init_movingSDValues, init_staticPeaksCollIntMean, init_staticPeaksCollIntSD, init_staticSupportsCollIntMean, init_staticSupportsCollIntSD, init_linRegressPeaksStats, init_linRegressPeaks, init_linRegressPeaksSD, init_linRegressSupportsStats, init_linRegressSupports, init_linRegressSupportsSD, init_movingPeaks2DStats, init_movingPeaks2D, init_movingSupports2DStats, init_movingSupports2D, init_movingPeaks3DStats, init_movingPeaks3D, init_movingSupports3DStats, init_movingSupports3D, init_movingPeaks4DStats, init_movingPeaks4D, init_movingSupports4DStats, init_movingSupports4D, init_movingPeaks5DStats, init_movingPeaks5D, init_movingSupports5DStats, init_movingSupports5D, init_fInverseValuesStats, init_fInverseValues);

########################################################
## Generate/Import Variant's Initial rawData
########################################################
for var in range(len(variant_names)):
    print('=============== GENERATE/IMPORT INITIAL VARIANT DATA ===============');
    # Sort timeScales
    variant_timeScales = np.sort(variant_timeScales);
    # Generate Random VariantData of Max Length at Min (Best) Resolution
    t0 = time.time();
    print('Generating randomValues');
    totalTicks = (variant_tickWindows[0] * int(max(variant_timeScales) / min(variant_timeScales)))
    randomValues = np.empty(totalTicks);
    randomValues[:] = np.NaN;
    for randValue in range(len(randomValues)):
        randomValues[randValue] = random.randint(2500,7500) * .01; #25-75
    t1 = time.time();
    if debugTimers == 1:
        print(str(t1-t0));
    
    # Generate Oscillating Noise
    t0 = time.time();
    print('Adding Oscillating Noise to the randomValues');
    oNoiseValues = np.empty(variant_tickWindows[0] * int(max(variant_timeScales) / min(variant_timeScales)));
    oNoiseValues[:] = np.NaN;
    oNoiseNumMultipliers = 20;
    oNoiseMultipliers = np.empty(oNoiseNumMultipliers);
    oNoiseMultipliers[:] = np.NaN;
    oNoiseAmpMultipliers = np.empty(oNoiseNumMultipliers);
    oNoiseAmpMultipliers[:] = np.NaN;
    for oNoiseLooper in range(0,oNoiseNumMultipliers):
        print(str(oNoiseLooper+1) + '/' + str(oNoiseNumMultipliers));
        oNoiseMultipliers[oNoiseLooper] = min(variant_timeScales) * 2 * np.pi * (.1 ** random.randint(1,60));
        oNoiseAmpMultipliers[oNoiseLooper] = np.std(randomValues) * random.randint(3,12) * .1;
        for oNoiseValue in range(len(oNoiseValues)):
            if oNoiseLooper == 0:
                oNoiseValues[oNoiseValue] = np.sin(oNoiseMultipliers[oNoiseLooper] * oNoiseValue) * oNoiseAmpMultipliers[oNoiseLooper];
            else:
                oNoiseValues[oNoiseValue] += np.sin(oNoiseMultipliers[oNoiseLooper] * oNoiseValue) * oNoiseAmpMultipliers[oNoiseLooper];
    t1 = time.time();
    if debugTimers == 1:
        print(str(t1-t0));
    
    # Add Oscillating Noise to randomValues
    randomValues += oNoiseValues;
    
    # Map randomValues by timeScale
    t0 = time.time();
    print('Mapping randomValues by Timescale');
    for timeScale in range(len(variant_timeScales)):
        for value in range(variant_tickWindows[0]):
            tempIndex = int(len(randomValues) - (len(randomValues) / (max(variant_timeScales) / variant_timeScales[timeScale]))) + (value * int(variant_timeScales[timeScale] / min(variant_timeScales)));
            if int(variant_timeScales[timeScale] / min(variant_timeScales)) == 1:
                variantData[var].rawValues[timeScale][value] = randomValues[tempIndex];
            elif int(variant_timeScales[timeScale] / min(variant_timeScales)) > 1:
                variantData[var].rawValues[timeScale][value] = np.mean(randomValues[tempIndex : tempIndex + int(variant_timeScales[timeScale] / min(variant_timeScales))]);           
    t1 = time.time();
    if debugTimers == 1:
        print(str(t1-t0));
    
    # Map Oscillating Noise by timeScale (for debugO plotting)
    t0 = time.time();
    print('Mapping Oscillating Noise by Timescale');
    for timeScale in range(len(variant_timeScales)):
        for value in range(variant_tickWindows[0]):
            tempIndex = int(len(randomValues) - (len(randomValues) / (max(variant_timeScales) / variant_timeScales[timeScale]))) + (value * int(variant_timeScales[timeScale] / min(variant_timeScales)));
            if int(variant_timeScales[timeScale] / min(variant_timeScales)) == 1:
                variantData[var].debugoNoiseValues[timeScale][value] = oNoiseValues[tempIndex];
            elif int(variant_timeScales[timeScale] / min(variant_timeScales)) > 1:
                variantData[var].debugoNoiseValues[timeScale][value] = np.mean(oNoiseValues[tempIndex : tempIndex + int(variant_timeScales[timeScale] / min(variant_timeScales))]);
    t1 = time.time();
    if debugTimers == 1:
        print(str(t1-t0));
    
################################################################################################################
## Update/Re-Import Variant's rawData
################################################################################################################
t0Master = time.time();
t1Master = time.time();
t1MasterFin = 360;

while t1Master-t0Master < t1MasterFin:
    print('');
    print('=============== UPDATE/RE-IMPORT VARIANT DATA ===============');
    t1Master = time.time();
    print('Total Time Elapsed: ' + str(round(t1Master-t0Master,2)) + 's' + ' / ' + str(t1MasterFin) + 's');
    t0Loop = time.time();
    
    for var in range(len(variant_names)):
        # Update randomValues
        print('Update randomValues');
        t0 = time.time();
        randomValue = randomValues[-1]; #25-75
        if not randomValue:
            print('ERROR: NO UPDATE VALUES');
        else:
            # Add linear rising/falling trend
            randomValue += (random.randint(500,2500) * .01); #5-10
            randomValue -= (random.randint(500,2500) * .01); #5-10
            # Add exponential rising/falling trend
            # if random.randint(0,10) < 3:
            #     randomValue += (random.randint(500,2500) * .01) * (random.randint(500,2500) * .01) * .5; #5-10
            #     randomValue -= (random.randint(500,2500) * .01) * (random.randint(500,2500) * .01) * .5; #5-10
            # Append new value to existing values
            randomValues = np.append(randomValues, randomValue);
            randomValues = np.delete(randomValues, 0);
            
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        # Update Oscillating Noise
        print('Update Oscillating Noise');
        t0 = time.time();
        oNoiseValues = np.append(oNoiseValues,0);
        oNoiseValues = np.delete(oNoiseValues, 0);
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        # Map randomValues by timeScale
        t0 = time.time();
        print('Mapping randomValues by Timescale');
        for timeScale in range(len(variant_timeScales)):
            for value in range(variant_tickWindows[0]):
                tempIndex = int(len(randomValues) - (len(randomValues) / (max(variant_timeScales) / variant_timeScales[timeScale]))) + (value * int(variant_timeScales[timeScale] / min(variant_timeScales)));
                if int(variant_timeScales[timeScale] / min(variant_timeScales)) == 1:
                    variantData[var].rawValues[timeScale][value] = randomValues[tempIndex];
                elif int(variant_timeScales[timeScale] / min(variant_timeScales)) > 1:
                    variantData[var].rawValues[timeScale][value] = np.mean(randomValues[tempIndex : tempIndex + int(variant_timeScales[timeScale] / min(variant_timeScales))]);
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        # Map Oscillating Noise by timeScale (for debugO plotting)
        t0 = time.time();
        print('Mapping Oscillating Noise by Timescale');
        for timeScale in range(len(variant_timeScales)):
            for value in range(variant_tickWindows[0]):
                tempIndex = int(len(randomValues) - (len(randomValues) / (max(variant_timeScales) / variant_timeScales[timeScale]))) + (value * int(variant_timeScales[timeScale] / min(variant_timeScales)));
                if int(variant_timeScales[timeScale] / min(variant_timeScales)) == 1:
                    variantData[var].debugoNoiseValues[timeScale][value] = oNoiseValues[tempIndex];
                elif int(variant_timeScales[timeScale] / min(variant_timeScales)) > 1:
                    variantData[var].debugoNoiseValues[timeScale][value] = np.mean(oNoiseValues[tempIndex : tempIndex + int(variant_timeScales[timeScale] / min(variant_timeScales))]);
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
    ################################################################################################################
    ## Within-Variant Analyses
    ################################################################################################################
    print('=== START WITHIN-VARIANT ANALYSES ===');
    for var in range(len(variant_names)):
        ########################################################
        ## Prepare Output Structures
        ########################################################
        # Plot Structures
        numSubPlots = 10;
        fig, axs = plt.subplots(numSubPlots,len(variant_timeScales), dpi=150);
        #plt.style.use('ggplot');
        
        ########################################################
        ## Output Raw
        ########################################################
        print('Output Raw');
        t0 = time.time();
        # Plot Raw
        for timeScale in range(len(variant_timeScales)):
            axs[0,timeScale].plot(variantData[var].rawValues[timeScale], color='dimgrey');
            axs[0,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='RAW');
            axs[0,timeScale].set_ylim([np.nanmin(variantData[var].rawValues) - np.nanstd([np.nanmin(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) *.5, np.nanmax(variantData[var].rawValues) + np.nanstd([np.min(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) * .5]);

            axs[1,timeScale].plot(variantData[var].rawValues[timeScale], color='dimgrey', alpha=0.3);
            axs[1,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='r');
            axs[1,timeScale].set_ylim([np.nanmin(variantData[var].rawValues) - np.nanstd([np.nanmin(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) *.5, np.nanmax(variantData[var].rawValues) + np.nanstd([np.min(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) * .5]);
    
            axs[2,timeScale].plot(variantData[var].rawValues[timeScale], color='dimgrey', alpha=0.3);
            axs[2,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='BB');
            axs[2,timeScale].set_ylim([np.nanmin(variantData[var].rawValues) - np.nanstd([np.nanmin(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) *.5, np.nanmax(variantData[var].rawValues) + np.nanstd([np.min(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) * .5]);
    
            axs[3,timeScale].plot(variantData[var].rawValues[timeScale], color='dimgrey', alpha=0.3);
            axs[3,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='P&S');
            axs[3,timeScale].set_ylim([np.nanmin(variantData[var].rawValues) - np.nanstd([np.nanmin(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) *.5, np.nanmax(variantData[var].rawValues) + np.nanstd([np.min(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) * .5]);
    
            axs[4,timeScale].plot(variantData[var].rawValues[timeScale], color='dimgrey', alpha=0.3);
            axs[4,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='rP&S');
            axs[4,timeScale].set_ylim([np.nanmin(variantData[var].rawValues) - np.nanstd([np.nanmin(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) *.5, np.nanmax(variantData[var].rawValues) + np.nanstd([np.min(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) * .5]);
    
            axs[5,timeScale].plot(variantData[var].rawValues[timeScale], color='dimgrey', alpha=0.3);
            axs[5,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='BBP&S');
            axs[5,timeScale].set_ylim([np.nanmin(variantData[var].rawValues) - np.nanstd([np.nanmin(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) *.5, np.nanmax(variantData[var].rawValues) + np.nanstd([np.min(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) * .5]);
    
            axs[8,timeScale].plot(variantData[var].rawValues[timeScale], color='dimgrey', alpha=0.3);
            axs[8,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='fftInv');
            axs[8,timeScale].set_ylim([np.nanmin(variantData[var].rawValues) - np.nanstd([np.nanmin(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) *.5, np.nanmax(variantData[var].rawValues) + np.nanstd([np.min(variantData[var].rawValues), np.nanmax(variantData[var].rawValues)]) * .5]);
            
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
            
        ########################################################
        ## Predict Raw
        ########################################################
        print('Predict Raw');
        t0 = time.time();
        # Get rawStaticMean & rawStaticSD
        for timeScale in range(len(variant_timeScales)):
            variantPredict[var].rawStaticMean[timeScale][:] = np.nanmean(variantData[var].rawValues[timeScale]);
            variantPredict[var].rawStaticSD[timeScale][:] = np.nanstd(variantData[var].rawValues[timeScale]);
        
        # Plot rawStaticMean and rawStaticSD
        for timeScale in range(len(variant_timeScales)):
            axs[0,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].rawStaticMean[timeScale], '-', linewidth = 1, color='violet', alpha=0.8);
            axs[0,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].rawStaticMean[timeScale] + variantPredict[var].rawStaticSD[timeScale], '-', linewidth = 1, color='violet', alpha=0.4);
            axs[0,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].rawStaticMean[timeScale] - variantPredict[var].rawStaticSD[timeScale], '-', linewidth = 1, color='violet', alpha=0.4);
    
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Output Linear Regression Line (r)
        ########################################################
        print('Output Linear Regression Line (r)');
        t0 = time.time();
        # r
        for timeScale in range(len(variant_timeScales)):
            variantData[var].linRegressStats[timeScale] = sp.stats.linregress(np.arange(0,variant_tickWindows[0]),variantData[var].rawValues[timeScale]);
            for value in range(0,variant_tickWindows[0]):
                variantData[var].linRegressValues[timeScale][value] = variantData[var].linRegressStats[timeScale][0] * value + variantData[var].linRegressStats[timeScale][1];
        
        # r SD
        for timeScale in range(len(variant_timeScales)):
            variantData[var].linRegressSD[timeScale] = np.sqrt(sum((variantData[var].rawValues[timeScale] - variantData[var].linRegressValues[timeScale]) ** 2) / (len(variantData[var].rawValues[timeScale]) - 1));
        
        # Plot r +/- SD
        for timeScale in range(len(variant_timeScales)):
            axs[1,timeScale].plot(variantData[var].linRegressValues[timeScale], color='teal', alpha=0.6);
            axs[1,timeScale].plot(variantData[var].linRegressValues[timeScale] + variantData[var].linRegressSD[timeScale], color='teal', alpha=0.4);
            axs[1,timeScale].plot(variantData[var].linRegressValues[timeScale] - variantData[var].linRegressSD[timeScale], color='teal', alpha=0.4);
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Predict Linear Regression Line (r)
        ########################################################
        print('Predict Linear Regression Line (r)');
        t0 = time.time();
        # Predict r
        for timeScale in range(len(variant_timeScales)):
            for value in range(0,variant_tickWindowsPredict[0]):
                variantPredict[var].linRegressValues[timeScale][value] = variantData[var].linRegressValues[timeScale][-1] + variantData[var].linRegressStats[timeScale][0] * value;
        
        # Predict r SD
        for timeScale in range(len(variant_timeScales)):
            variantPredict[var].linRegressSD[timeScale] = variantData[var].linRegressSD[timeScale][-variant_tickWindowsPredict[0]:];
        
        # Plot predicted r +/- SD
        for timeScale in range(len(variant_timeScales)):
            axs[1,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].linRegressValues[timeScale], '-', linewidth = 1, color='violet', alpha=0.8);
            axs[1,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].linRegressValues[timeScale] + variantPredict[var].linRegressSD[timeScale], '-', linewidth = 1, color='violet', alpha=0.4);
            axs[1,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].linRegressValues[timeScale] - variantPredict[var].linRegressSD[timeScale], '-', linewidth = 1, color='violet', alpha=0.4);
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Output Moving Average +/- SD (Bollinger Bands)
        ########################################################
        t0 = time.time();
        print('Output Moving Average +/- SD (Bollinger Bands)');
        # Simple Moving Average and SD of rawValues ***CHECK TICKWINDOWINDEX SD CORRECT?***
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantData[var].movingAverageValues[timeScale][tickWindowIndex][tickWindowSize-1:] = moving_average(variantData[var].rawValues[timeScale], tickWindowSize);
                for value in range(0,variant_tickWindows[0] - tickWindowSize + 1):
                    variantData[var].movingSDValues[timeScale][tickWindowIndex][tickWindowSize + value - 1] = np.std(variantData[var].rawValues[timeScale][value:tickWindowSize + value]);
        
        # Plot SMA +/- SD for rawValues
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[2,timeScale].plot(variantData[var].movingAverageValues[timeScale][tickWindowIndex], color='teal', alpha=0.6); #plot simpleMovingAverage
                axs[2,timeScale].plot(variantData[var].movingAverageValues[timeScale][tickWindowIndex] + variantData[var].movingSDValues[timeScale][tickWindowIndex], '-', linewidth=1, color='lightseagreen', alpha=0.4); #plot simpleMovingAverage + SD
                axs[2,timeScale].plot(variantData[var].movingAverageValues[timeScale][tickWindowIndex] - variantData[var].movingSDValues[timeScale][tickWindowIndex], '-', linewidth=1, color='lightseagreen', alpha=0.4); #plot simpleMovingAverage - SD
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Predict Moving Average +/- SD (Bollinger Bands)
        ########################################################
        t0 = time.time();
        print('Predict Moving Average +/- SD (Bollinger Bands)');
        
        # 2nd Degree Polynomial CURVE_FIT (2D)
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingAverage2DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective2D, np.arange(variant_tickWindows[0] - tickWindowSize + 1), variantData[var].movingAverageValues[timeScale][tickWindowIndex][tickWindowSize-1:]);
                a = variantPredict[var].movingAverage2DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingAverage2DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingAverage2DStats[timeScale][tickWindowIndex][2];
                originalFitWindowSize = variant_tickWindows[0] - tickWindowSize + 1;
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingAverage2D[timeScale][tickWindowIndex][value] = a * value**2 + b * value + (a * originalFitWindowSize**2 + b * originalFitWindowSize + c);
        
        # 3rd Degree Polynomial CURVE_FIT (3D)
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingAverage3DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective3D, np.arange(variant_tickWindows[0] - tickWindowSize + 1), variantData[var].movingAverageValues[timeScale][tickWindowIndex][tickWindowSize-1:]);
                a = variantPredict[var].movingAverage3DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingAverage3DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingAverage3DStats[timeScale][tickWindowIndex][2];
                d = variantPredict[var].movingAverage3DStats[timeScale][tickWindowIndex][3];
                originalFitWindowSize = variant_tickWindows[0] - tickWindowSize + 1;
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingAverage3D[timeScale][tickWindowIndex][value] = a * value**3 + b * value**2 + c * value + (a * originalFitWindowSize**3 + b * originalFitWindowSize**2 + c * originalFitWindowSize + d);
        
        # 4th Degree Polynomial CURVE_FIT (4D)
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingAverage4DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective4D, np.arange(variant_tickWindows[0] - tickWindowSize + 1), variantData[var].movingAverageValues[timeScale][tickWindowIndex][tickWindowSize-1:]);
                a = variantPredict[var].movingAverage4DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingAverage4DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingAverage4DStats[timeScale][tickWindowIndex][2];
                d = variantPredict[var].movingAverage4DStats[timeScale][tickWindowIndex][3];
                e = variantPredict[var].movingAverage4DStats[timeScale][tickWindowIndex][4];
                originalFitWindowSize = variant_tickWindows[0] - tickWindowSize + 1;
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingAverage4D[timeScale][tickWindowIndex][value] = a * value**4 + b * value**3 + c * value**2 + d * value + (a * originalFitWindowSize**4 + b * originalFitWindowSize**3 + c * originalFitWindowSize**2 + d * originalFitWindowSize + e);
        
        # 5th Degree Polynomial CURVE_FIT (5D)
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingAverage5DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective5D, np.arange(variant_tickWindows[0] - tickWindowSize + 1), variantData[var].movingAverageValues[timeScale][tickWindowIndex][tickWindowSize-1:]);
                a = variantPredict[var].movingAverage5DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingAverage5DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingAverage5DStats[timeScale][tickWindowIndex][2];
                d = variantPredict[var].movingAverage5DStats[timeScale][tickWindowIndex][3];
                e = variantPredict[var].movingAverage5DStats[timeScale][tickWindowIndex][4];
                f = variantPredict[var].movingAverage5DStats[timeScale][tickWindowIndex][5];
                originalFitWindowSize = variant_tickWindows[0] - tickWindowSize + 1;
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingAverage5D[timeScale][tickWindowIndex][value] = a * value**5 + b * value**4 + c * value**3 + d * value**2 + e * value + (a * originalFitWindowSize**5 + b * originalFitWindowSize**4 + c * originalFitWindowSize**3 + d * originalFitWindowSize**2 + e * originalFitWindowSize + f);
        
        # Plot 2nd Degree Polynomial CURVE_FIT (2D)
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[2,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingAverage2D[timeScale][tickWindowIndex], linewidth=1, color='darkviolet', alpha=0.6); #plot simpleMovingAverage
        
        # Plot 3rd Degree Polynomial CURVE_FIT (3D)
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[2,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingAverage3D[timeScale][tickWindowIndex], linewidth=1, color='thistle', alpha=0.2); #plot simpleMovingAverage
        
        # Plot 4th Degree Polynomial CURVE_FIT (4D)
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[2,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingAverage4D[timeScale][tickWindowIndex], linewidth=1, color='thistle', alpha=0.2); #plot simpleMovingAverage
        
        # Plot 5th Degree Polynomial CURVE_FIT (5D)
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[2,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingAverage5D[timeScale][tickWindowIndex], linewidth=1, color='violet', alpha=0.6); #plot simpleMovingAverage
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Output Peaks and Supports
        ########################################################
        t0 = time.time();
        print('Output Peaks and Supports (values in movingAverage windows that are outside of 95% CI)');
        # Find Static Peaks and Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                for value in range(0,variant_tickWindows[0] - tickWindowIndex):
                    if variantData[var].rawValues[timeScale][value] > variantData[var].movingAverageValues[timeScale][tickWindowIndex][value] + variantData[var].movingSDValues[timeScale][tickWindowIndex][value]:
                        variantData[var].staticPeaks[timeScale][tickWindowIndex][value] = variantData[var].rawValues[timeScale][value]; # peaks
                    else:
                        variantData[var].staticPeaks[timeScale][tickWindowIndex][value] = np.nan;
                    if variantData[var].rawValues[timeScale][value] < variantData[var].movingAverageValues[timeScale][tickWindowIndex][value] - variantData[var].movingSDValues[timeScale][tickWindowIndex][value]:
                        variantData[var].staticSupports[timeScale][tickWindowIndex][value] = variantData[var].rawValues[timeScale][value]; # supports
                    else:
                        variantData[var].staticSupports[timeScale][tickWindowIndex][value] = np.nan;
        
        # Collapse across all timeWindowIndex
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                for value in range(variant_tickWindows[0]):
                    if ~np.isnan(variantData[var].staticPeaks[timeScale][tickWindowIndex][value]):
                        variantData[var].staticPeaksCollapsed[timeScale][value] = variantData[var].staticPeaks[timeScale][tickWindowIndex][value];
                    if ~np.isnan(variantData[var].staticSupports[timeScale][tickWindowIndex][value]):
                        variantData[var].staticSupportsCollapsed[timeScale][value] = variantData[var].staticSupports[timeScale][tickWindowIndex][value];
        
        # Interpolate Collapsed Peaks and Supports
        for timeScale in range(len(variant_timeScales)):
            variantData[var].staticPeaksCollapsedInterp[timeScale] = pd.Series(variantData[var].staticPeaksCollapsed[timeScale]).interpolate(limit_direction = 'forward', kind= 'linear'); #‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’
            variantData[var].staticSupportsCollapsedInterp[timeScale] = pd.Series(variantData[var].staticSupportsCollapsed[timeScale]).interpolate(limit_direction = 'forward', kind= 'linear');
        
        # Plot Static Peaks and Supports
        for timeScale in range(len(variant_timeScales)):  
            axs[3,timeScale].plot(variantData[var].staticPeaksCollapsed[timeScale], '.', linewidth=1, markersize=1, color='skyblue', alpha=0.8); #plot simpleMovingAverage + SD
            axs[3,timeScale].plot(variantData[var].staticSupportsCollapsed[timeScale], '.', linewidth=1, markersize=1, color='steelblue', alpha=0.8); #plot simpleMovingAverage - SD
            axs[3,timeScale].plot(variantData[var].staticPeaksCollapsedInterp[timeScale], '-', linewidth=1, markersize=1, color='skyblue', alpha=0.3); #plot simpleMovingAverage + SD
            axs[3,timeScale].plot(variantData[var].staticSupportsCollapsedInterp[timeScale], '-', linewidth=1, markersize=1, color='steelblue', alpha=0.3); #plot simpleMovingAverage - SD
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));

        ########################################################
        ## Predict Mean +/- SD for Peaks and Supports
        ########################################################
        t0 = time.time();
        print('Predict Mean +/- SD for Peaks and Supports');
        
        # Interpolated Collapsed Peaks and Supports Mean +/- SD
        for timeScale in range(len(variant_timeScales)):
             variantPredict[var].staticPeaksCollIntMean[timeScale] = np.nanmean(variantData[var].staticPeaksCollapsedInterp[timeScale]);
             variantPredict[var].staticPeaksCollIntSD[timeScale] = np.nanstd(variantData[var].staticPeaksCollapsedInterp[timeScale]);
             variantPredict[var].staticSupportsCollIntMean[timeScale] = np.nanmean(variantData[var].staticSupportsCollapsedInterp[timeScale]);
             variantPredict[var].staticSupportsCollIntSD[timeScale] = np.nanstd(variantData[var].staticSupportsCollapsedInterp[timeScale]);
        
        # Plot Interpolated Collapsed Peaks and Supports Mean +/- SD
        for timeScale in range(len(variant_timeScales)):  
            axs[3,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].staticPeaksCollIntMean[timeScale], '-', linewidth=1, markersize=1, color='violet', alpha=0.8); #plot simpleMovingAverage + SD
            axs[3,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].staticSupportsCollIntMean[timeScale], '-', linewidth=1, markersize=1, color='darkviolet', alpha=0.8); #plot simpleMovingAverage - SD
            axs[3,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].staticPeaksCollIntMean[timeScale] + variantPredict[var].staticPeaksCollIntSD[timeScale], '-', linewidth=1, markersize=1, color='violet', alpha=0.3); #plot simpleMovingAverage + SD
            axs[3,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].staticPeaksCollIntMean[timeScale] - variantPredict[var].staticPeaksCollIntSD[timeScale], '-', linewidth=1, markersize=1, color='violet', alpha=0.3); #plot simpleMovingAverage - SD
            axs[3,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].staticSupportsCollIntMean[timeScale] + variantPredict[var].staticSupportsCollIntSD[timeScale], '-', linewidth=1, markersize=1, color='darkviolet', alpha=0.3); #plot simpleMovingAverage + SD
            axs[3,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].staticSupportsCollIntMean[timeScale] - variantPredict[var].staticSupportsCollIntSD[timeScale], '-', linewidth=1, markersize=1, color='darkviolet', alpha=0.3); #plot simpleMovingAverage - SD
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Output r for Peaks and Supports
        ########################################################
        t0 = time.time();
        print('Output r for Peaks and Supports');
        # r and SD for Peaks and Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                # Get indeces of peaks/supports
                prePeakIndeces = [i for i, x in enumerate(~np.isnan(variantData[var].staticPeaks[timeScale][tickWindowIndex])) if x];
                preSupportIndeces = [i for i, x in enumerate(~np.isnan(variantData[var].staticSupports[timeScale][tickWindowIndex])) if x];
                if len(prePeakIndeces) > 1:
                    for value in range(len(prePeakIndeces)):
                        variantData[var].staticPeakIndeces[timeScale][tickWindowIndex][value] = prePeakIndeces[value];
                if len(preSupportIndeces) > 1:
                    for value in range(len(preSupportIndeces)):
                        variantData[var].staticSupportIndeces[timeScale][tickWindowIndex][value] = preSupportIndeces[value];
                realPeakIndeces = variantData[var].staticPeakIndeces[timeScale][tickWindowIndex][~np.isnan(variantData[var].staticPeakIndeces[timeScale][tickWindowIndex])];
                realSupportIndeces = variantData[var].staticSupportIndeces[timeScale][tickWindowIndex][~np.isnan(variantData[var].staticSupportIndeces[timeScale][tickWindowIndex])];
                # r and SD assignment, interpolate in-betweens manually
                if len(realPeakIndeces) > 1:
                    lenPeakIndeces = realPeakIndeces[len(realPeakIndeces)-1]-realPeakIndeces[0];
                    variantData[var].linRegressPeaksStats[timeScale][tickWindowIndex] = sp.stats.linregress(np.arange(0,len(variantData[var].staticPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].staticPeaks[timeScale][tickWindowIndex])])), variantData[var].staticPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].staticPeaks[timeScale][tickWindowIndex])]);
                    variantData[var].linRegressPeaksStats[timeScale][tickWindowIndex][0] = variantData[var].linRegressPeaksStats[timeScale][tickWindowIndex][0] * (len(realPeakIndeces) / lenPeakIndeces);
                    for t in range(lenPeakIndeces + (variant_tickWindows[0] - lenPeakIndeces - realPeakIndeces[0])):
                        variantData[var].linRegressPeaks[timeScale][tickWindowIndex][realPeakIndeces[0] + t] = variantData[var].linRegressPeaksStats[timeScale][tickWindowIndex][1] + (variantData[var].linRegressPeaksStats[timeScale][tickWindowIndex][0] * t);
                    variantData[var].linRegressPeaksSD[timeScale][tickWindowIndex] = np.sqrt(np.nansum((variantData[var].staticPeaks[timeScale][tickWindowIndex] - variantData[var].linRegressPeaks[timeScale][tickWindowIndex]) ** 2) / (len(realPeakIndeces) - 1));
                # Get linRegress for peaks/supports w/o NaNs, then Correct r slope for NaNs
                if len(realSupportIndeces) > 1:
                    lenSupportIndeces = realSupportIndeces[len(realSupportIndeces)-1]-realSupportIndeces[0];
                    variantData[var].linRegressSupportsStats[timeScale][tickWindowIndex] = sp.stats.linregress(np.arange(0,len(variantData[var].staticSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].staticSupports[timeScale][tickWindowIndex])])), variantData[var].staticSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].staticSupports[timeScale][tickWindowIndex])]);
                    variantData[var].linRegressSupportsStats[timeScale][tickWindowIndex][0] = variantData[var].linRegressSupportsStats[timeScale][tickWindowIndex][0] * (len(realSupportIndeces) / lenSupportIndeces);
                    for t in range(lenSupportIndeces + (variant_tickWindows[0] - lenSupportIndeces - realSupportIndeces[0])):
                        variantData[var].linRegressSupports[timeScale][tickWindowIndex][realSupportIndeces[0] + t] =  variantData[var].linRegressSupportsStats[timeScale][tickWindowIndex][1] + (variantData[var].linRegressSupportsStats[timeScale][tickWindowIndex][0] * t);
                    variantData[var].linRegressSupportsSD[timeScale][tickWindowIndex] = np.sqrt(np.nansum((variantData[var].staticSupports[timeScale][tickWindowIndex] - variantData[var].linRegressSupports[timeScale][tickWindowIndex]) ** 2) / (len(realSupportIndeces) - 1));

        # Plot r +/- SD for peaks and Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                axs[4,timeScale].plot(variantData[var].linRegressPeaks[timeScale][tickWindowIndex], color='skyblue', alpha=0.8);
                axs[4,timeScale].plot(variantData[var].linRegressPeaks[timeScale][tickWindowIndex] + variantData[var].linRegressPeaksSD[timeScale][tickWindowIndex], color='skyblue', alpha=0.4);
                axs[4,timeScale].plot(variantData[var].linRegressPeaks[timeScale][tickWindowIndex] - variantData[var].linRegressPeaksSD[timeScale][tickWindowIndex], color='skyblue', alpha=0.4);
                axs[4,timeScale].plot(variantData[var].linRegressSupports[timeScale][tickWindowIndex], color='steelblue', alpha=0.8);
                axs[4,timeScale].plot(variantData[var].linRegressSupports[timeScale][tickWindowIndex] + variantData[var].linRegressSupportsSD[timeScale][tickWindowIndex], color='steelblue', alpha=0.4);
                axs[4,timeScale].plot(variantData[var].linRegressSupports[timeScale][tickWindowIndex] - variantData[var].linRegressSupportsSD[timeScale][tickWindowIndex], color='steelblue', alpha=0.4);
                
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Predict r for Peaks and Supports
        ########################################################
        print('Predict r for Peaks and Supports');
        t0 = time.time();
        # Predict r +/- SD for Peaks and Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].linRegressPeaks[timeScale][tickWindowIndex][value] = variantData[var].linRegressPeaks[timeScale][tickWindowIndex][-1] + variantData[var].linRegressPeaksStats[timeScale][tickWindowIndex][0] * value;
                    variantPredict[var].linRegressSupports[timeScale][tickWindowIndex][value] = variantData[var].linRegressSupports[timeScale][tickWindowIndex][-1] + variantData[var].linRegressSupportsStats[timeScale][tickWindowIndex][0] * value;
                variantPredict[var].linRegressPeaksSD[timeScale][tickWindowIndex] = variantData[var].linRegressPeaksSD[timeScale][tickWindowIndex][-variant_tickWindowsPredict[0]:];
                variantPredict[var].linRegressSupportsSD[timeScale][tickWindowIndex] = variantData[var].linRegressSupportsSD[timeScale][tickWindowIndex][-variant_tickWindowsPredict[0]:];
        
        # Plot predicted r +/- SD for Peaks and Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                axs[4,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].linRegressPeaks[timeScale][tickWindowIndex], '-', linewidth = 1, color='violet', alpha=0.8);
                axs[4,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].linRegressPeaks[timeScale][tickWindowIndex] + variantPredict[var].linRegressPeaksSD[timeScale][tickWindowIndex], '-', linewidth = 1, color='violet', alpha=0.4);
                axs[4,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].linRegressPeaks[timeScale][tickWindowIndex] - variantPredict[var].linRegressPeaksSD[timeScale][tickWindowIndex], '-', linewidth = 1, color='violet', alpha=0.4);
                axs[4,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].linRegressSupports[timeScale][tickWindowIndex], '-', linewidth = 1, color='darkviolet', alpha=0.8);
                axs[4,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].linRegressSupports[timeScale][tickWindowIndex] + variantPredict[var].linRegressSupportsSD[timeScale][tickWindowIndex], '-', linewidth = 1, color='darkviolet', alpha=0.4);
                axs[4,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].linRegressSupports[timeScale][tickWindowIndex] - variantPredict[var].linRegressSupportsSD[timeScale][tickWindowIndex], '-', linewidth = 1, color='darkviolet', alpha=0.4);
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Output Moving Average +/- SD (Bollinger Bands) for Collapsed Peaks and Supports
        ########################################################
        t0 = time.time();
        print('Output Moving Average +/- SD (Bollinger Bands) for Collapsed Peaks and Supports');
        # Moving Average +/- SD (Bollinger Bands) for Collapsed Peaks and Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantData[var].movingPeaks[timeScale][tickWindowIndex][tickWindowSize-1:] = moving_average(variantData[var].staticPeaksCollapsedInterp[timeScale], tickWindowSize);
                variantData[var].movingSupports[timeScale][tickWindowIndex][tickWindowSize-1:] = moving_average(variantData[var].staticSupportsCollapsedInterp[timeScale], tickWindowSize);
                for value in range(0,variant_tickWindows[0] - tickWindowSize + 1):
                    variantData[var].movingPeaksSD[timeScale][tickWindowIndex][tickWindowSize + value - 1] = np.nanstd(variantData[var].rawValues[timeScale][value:tickWindowSize + value]);
                    variantData[var].movingSupportsSD[timeScale][tickWindowIndex][tickWindowSize + value - 1] = np.nanstd(variantData[var].rawValues[timeScale][value:tickWindowSize + value]);
        
        # Plot SMA +/- SD for Collapsed Peaks and Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                axs[5,timeScale].plot(variantData[var].movingPeaks[timeScale][tickWindowIndex], color='skyblue', alpha=0.8);
                axs[5,timeScale].plot(variantData[var].movingSupports[timeScale][tickWindowIndex], color='steelblue', alpha=0.8);
                axs[5,timeScale].plot(variantData[var].movingPeaks[timeScale][tickWindowIndex] + variantData[var].movingPeaksSD[timeScale][tickWindowIndex], '-', linewidth=1, color='skyblue', alpha=0.4); #plot peak + SD
                axs[5,timeScale].plot(variantData[var].movingPeaks[timeScale][tickWindowIndex] - variantData[var].movingPeaksSD[timeScale][tickWindowIndex], '-', linewidth=1, color='skyblue', alpha=0.4); #plot peak - SD
                axs[5,timeScale].plot(variantData[var].movingSupports[timeScale][tickWindowIndex] + variantData[var].movingSupportsSD[timeScale][tickWindowIndex], '-', linewidth=1, color='steelblue', alpha=0.4); #plot supp + SD
                axs[5,timeScale].plot(variantData[var].movingSupports[timeScale][tickWindowIndex] - variantData[var].movingSupportsSD[timeScale][tickWindowIndex], '-', linewidth=1, color='steelblue', alpha=0.4); #plot supp - SD
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Predict Moving Average +/- SD (Bollinger Bands) for Collapsed Peaks and Supports
        ########################################################
        t0 = time.time();
        print('Predict Moving Average +/- SD (Bollinger Bands) for Collapsed Peaks and Supports');
        # 2nd Degree Polynomial CURVE_FIT (2D) for Peaks
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingPeaks2DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective2D, np.arange(len(variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])])), variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])]);
                a = variantPredict[var].movingPeaks2DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingPeaks2DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingPeaks2DStats[timeScale][tickWindowIndex][2];
                originalFitWindowSize = len(variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])]);
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingPeaks2D[timeScale][tickWindowIndex][value] = a * value**2 + b * value + (a * originalFitWindowSize**2 + b * originalFitWindowSize + c);

        # 3rd Degree Polynomial CURVE_FIT (3D) for Peaks
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingPeaks3DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective3D, np.arange(len(variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])])), variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])]);
                a = variantPredict[var].movingPeaks3DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingPeaks3DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingPeaks3DStats[timeScale][tickWindowIndex][2];
                d = variantPredict[var].movingPeaks3DStats[timeScale][tickWindowIndex][3];
                originalFitWindowSize = len(variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])]);
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingPeaks3D[timeScale][tickWindowIndex][value] = a * value**3 + b * value**2 + c * value + (a * originalFitWindowSize**3 + b * originalFitWindowSize**2 + c * originalFitWindowSize + d);
        
        # 4th Degree Polynomial CURVE_FIT (4D) for Peaks
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingPeaks4DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective4D, np.arange(len(variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])])), variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])]);
                a = variantPredict[var].movingPeaks4DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingPeaks4DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingPeaks4DStats[timeScale][tickWindowIndex][2];
                d = variantPredict[var].movingPeaks4DStats[timeScale][tickWindowIndex][3];
                e = variantPredict[var].movingPeaks4DStats[timeScale][tickWindowIndex][4];
                originalFitWindowSize = len(variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])]);
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingPeaks4D[timeScale][tickWindowIndex][value] = a * value**4 + b * value**3 + c * value**2 + d * value + (a * originalFitWindowSize**4 + b * originalFitWindowSize**3 + c * originalFitWindowSize**2 + d * originalFitWindowSize + e);
        
        # 5th Degree Polynomial CURVE_FIT (5D) for Peaks
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingPeaks5DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective5D, np.arange(len(variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])])), variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])]);
                a = variantPredict[var].movingPeaks5DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingPeaks5DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingPeaks5DStats[timeScale][tickWindowIndex][2];
                d = variantPredict[var].movingPeaks5DStats[timeScale][tickWindowIndex][3];
                e = variantPredict[var].movingPeaks5DStats[timeScale][tickWindowIndex][4];
                f = variantPredict[var].movingPeaks5DStats[timeScale][tickWindowIndex][5];
                originalFitWindowSize = len(variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])]);
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingPeaks5D[timeScale][tickWindowIndex][value] = a * value**5 + b * value**4 + c * value**3 + d * value**2 + e * value + (a * originalFitWindowSize**5 + b * originalFitWindowSize**4 + c * originalFitWindowSize**3 + d * originalFitWindowSize**2 + e * originalFitWindowSize + f);

        # 2nd Degree Polynomial CURVE_FIT (2D) for Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingSupports2DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective2D, np.arange(len(variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])])), variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])]);
                a = variantPredict[var].movingSupports2DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingSupports2DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingSupports2DStats[timeScale][tickWindowIndex][2];
                originalFitWindowSize = len(variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])]);
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingSupports2D[timeScale][tickWindowIndex][value] = a * value**2 + b * value + (a * originalFitWindowSize**2 + b * originalFitWindowSize + c);

        # 3rd Degree Polynomial CURVE_FIT (3D) for Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingSupports3DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective3D, np.arange(len(variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])])), variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])]);
                a = variantPredict[var].movingSupports3DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingSupports3DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingSupports3DStats[timeScale][tickWindowIndex][2];
                d = variantPredict[var].movingSupports3DStats[timeScale][tickWindowIndex][3];
                originalFitWindowSize = len(variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])]);
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingSupports3D[timeScale][tickWindowIndex][value] = a * value**3 + b * value**2 + c * value + (a * originalFitWindowSize**3 + b * originalFitWindowSize**2 + c * originalFitWindowSize + d);
        
        # 4th Degree Polynomial CURVE_FIT (4D) for Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingSupports4DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective4D, np.arange(len(variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])])), variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])]);
                a = variantPredict[var].movingSupports4DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingSupports4DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingSupports4DStats[timeScale][tickWindowIndex][2];
                d = variantPredict[var].movingSupports4DStats[timeScale][tickWindowIndex][3];
                e = variantPredict[var].movingSupports4DStats[timeScale][tickWindowIndex][4];
                originalFitWindowSize = len(variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])]);
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingSupports4D[timeScale][tickWindowIndex][value] = a * value**4 + b * value**3 + c * value**2 + d * value + (a * originalFitWindowSize**4 + b * originalFitWindowSize**3 + c * originalFitWindowSize**2 + d * originalFitWindowSize + e);
        
        # 5th Degree Polynomial CURVE_FIT (5D) for Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                variantPredict[var].movingSupports5DStats[timeScale][tickWindowIndex], _ = sp.optimize.curve_fit(objective5D, np.arange(len(variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])])), variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])]);
                a = variantPredict[var].movingSupports5DStats[timeScale][tickWindowIndex][0];
                b = variantPredict[var].movingSupports5DStats[timeScale][tickWindowIndex][1];
                c = variantPredict[var].movingSupports5DStats[timeScale][tickWindowIndex][2];
                d = variantPredict[var].movingSupports5DStats[timeScale][tickWindowIndex][3];
                e = variantPredict[var].movingSupports5DStats[timeScale][tickWindowIndex][4];
                f = variantPredict[var].movingSupports5DStats[timeScale][tickWindowIndex][5];
                originalFitWindowSize = len(variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])]);
                for value in range(0,variant_tickWindowsPredict[0]):
                    variantPredict[var].movingSupports5D[timeScale][tickWindowIndex][value] = a * value**5 + b * value**4 + c * value**3 + d * value**2 + e * value + (a * originalFitWindowSize**5 + b * originalFitWindowSize**4 + c * originalFitWindowSize**3 + d * originalFitWindowSize**2 + e * originalFitWindowSize + f);

        # Plot 2nd Degree Polynomial CURVE_FIT (2D) for Peaks
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[5,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingPeaks2D[timeScale][tickWindowIndex], linewidth=1, color='violet', alpha=0.6); #plot simpleMovingAverage
        
        # Plot 3rd Degree Polynomial CURVE_FIT (3D) for Peaks
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[5,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingPeaks3D[timeScale][tickWindowIndex], linewidth=1, color='thistle', alpha=0.2); #plot simpleMovingAverage
        
        # Plot 4th Degree Polynomial CURVE_FIT (4D) for Peaks
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[5,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingPeaks4D[timeScale][tickWindowIndex], linewidth=1, color='thistle', alpha=0.2); #plot simpleMovingAverage
        
        # Plot 5th Degree Polynomial CURVE_FIT (5D) for Peaks
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[5,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingPeaks5D[timeScale][tickWindowIndex], linewidth=1, color='violet', alpha=0.6); #plot simpleMovingAverage
        
        # Plot 2nd Degree Polynomial CURVE_FIT (2D) for Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[5,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingSupports2D[timeScale][tickWindowIndex], linewidth=1, color='darkviolet', alpha=0.6); #plot simpleMovingAverage
        
        # Plot 3rd Degree Polynomial CURVE_FIT (3D) for Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[5,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingSupports3D[timeScale][tickWindowIndex], linewidth=1, color='thistle', alpha=0.2); #plot simpleMovingAverage
        
        # Plot 4th Degree Polynomial CURVE_FIT (4D) for Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[5,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingSupports4D[timeScale][tickWindowIndex], linewidth=1, color='thistle', alpha=0.2); #plot simpleMovingAverage
        
        # Plot 5th Degree Polynomial CURVE_FIT (5D) for Supports
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[5,timeScale].plot(np.arange(variant_tickWindows[0]+variant_tickWindowsPredict[0])[-variant_tickWindowsPredict[0]:], variantPredict[var].movingSupports5D[timeScale][tickWindowIndex], linewidth=1, color='darkviolet', alpha=0.6); #plot simpleMovingAverage
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Output Fast Fourier Transform (FFT), Power Spectrum Density (fPSD), Frequency Distribution (fDist)
        ########################################################
        t0 = time.time();
        print('Output Fast Fourier Transform (FFT), Power Spectrum Density (fPSD), Frequency Distribution (fDist)');
        # Acquire frequency of sample (fs), nyquist frequency (nyq), and time per 1 sample in sec (dt)
        for timeScale in range(len(variant_timeScales)):
            variantData[var].fs[timeScale] = 1/(variant_timeScales[timeScale] * 60);
            variantData[var].nyq[timeScale] = variantData[var].fs[timeScale] * 0.5;
            variantData[var].dt[timeScale] = variant_timeScales[timeScale] * 60;
            
        # Fast Fourier Transform
        for timeScale in range(len(variant_timeScales)):
            variantData[var].fHat[timeScale] = np.fft.fft(variantData[var].rawValues[timeScale], variant_tickWindows[0]);
        # Power Spectrum Density
        for timeScale in range(len(variant_timeScales)):
            variantData[var].fPSD[timeScale] = variantData[var].fHat[timeScale] * np.conj(variantData[var].fHat[timeScale]) / variant_tickWindows[0];
        # Frequency Distribution
        for timeScale in range(len(variant_timeScales)):
            dt = variantData[var].dt[timeScale]; #time per 1 sample in sec
            variantData[var].fDist[timeScale] = (1/(dt * variant_tickWindows[0])) * np.arange(variant_tickWindows[0]);
        
        # Set noise level of fPSD as Mean + STD of PSD(except 1st value).
        for timeScale in range(len(variant_timeScales)):
            variantData[var].fFilterLevel[timeScale] = np.mean(variantData[var].fPSD[timeScale][1:]) + np.std(variantData[var].fPSD[timeScale][1:]);
        
        # Plot fftL fDist vs. fPSD (except first value)
        for timeScale in range(len(variant_timeScales)):
            fs = variantData[var].fs[timeScale]; #frequency of sample (in Hz)
            nyq = variantData[var].nyq[timeScale]; #nyquist
            axs[6,timeScale].plot(variantData[var].fDist[timeScale][1:], variantData[var].fPSD[timeScale][1:], color='teal', alpha=0.6);
            axs[6,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='fftL');
            axs[6,timeScale].set_ylim([np.min(variantData[var].fPSD[1:]) - np.std([np.min(variantData[var].fPSD[1:]), (np.max(variantData[var].fPSD[1:]) - np.max(variantData[var].fPSD[1:]) * .95)]) * .5, (np.max(variantData[var].fPSD[1:]) - np.max(variantData[var].fPSD[1:]) * .95) + np.std([np.min(variantData[var].fPSD[1:]), (np.max(variantData[var].fPSD[1:]) - np.max(variantData[var].fPSD[1:]) * .95)]) * .5]);
            # Plot nyquist line
            axs[6,timeScale].axvline(x=nyq, color='darkviolet', alpha=0.6);
            # Plot Hat/Inverse Filtering cutoff lines
            axs[6,timeScale].axhline(y=variantData[var].fFilterLevel[timeScale], color='violet', alpha=0.6);
            
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Output PSD Filter and Inverse FFT (IFFT)
        ########################################################
        t0 = time.time();
        print('Output PSD Filter and Inverse FFT');
        # Filter out noise with PSD Filter then Inverse FFT the Filtered Values
        for timeScale in range(len(variant_timeScales)):
            variantData[var].fPSDFilteredHat[timeScale] = (variantData[var].fPSD[timeScale] > variantData[var].fFilterLevel[timeScale]) * variantData[var].fHat[timeScale];
            variantData[var].fInverseValues[timeScale] = np.fft.ifft(variantData[var].fPSDFilteredHat[timeScale]);
            
        # Do fInverseValues separately for each frequency of interest (this independence important for creating predictive sin functions);
        # plt.plot(variantData[var].fDist[timeScale][1:], variantData[var].fPSD[timeScale][1:], color='teal', alpha=0.6);
        # variantData[var].fDist[timeScale][1:]
        #for timeScale in range(len(variant_timeScales)):
            #variantData[var].fPSDFilteredHat[timeScale] = variantData[var].fDist[timeScale][variantData[var].fPSD[timeScale] > variantData[var].fFilterLevel[timeScale]];
            
        # Plot Debug Oscillation Noise Only
        for timeScale in range(len(variant_timeScales)):
            axs[7,timeScale].plot(variantData[var].debugoNoiseValues[timeScale], color='deepskyblue', alpha=0.8);
            axs[7,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='debugO');
            axs[7,timeScale].set_ylim([np.min(variantData[var].debugoNoiseValues) - np.std([np.min(variantData[var].debugoNoiseValues), np.max(variantData[var].debugoNoiseValues)]) *.5, np.max(variantData[var].debugoNoiseValues) + np.std([np.min(variantData[var].debugoNoiseValues), np.max(variantData[var].debugoNoiseValues)]) * .5]);
        
        # Plot Inverse Values
        for timeScale in range(len(variant_timeScales)):
            axs[8,timeScale].plot(variantData[var].fInverseValues[timeScale], color='teal', alpha=0.8);
            axs[8,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='ifft');
            axs[8,timeScale].set_ylim([np.min(variantData[var].rawValues) - np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) *.5, np.max(variantData[var].rawValues) + np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) * .5]);
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        ########################################################
        ## Predict Inverse FFT (IFFT)
        ########################################################
        t0 = time.time();
        print('Predict Inverse FFT');
        # For each frequency of interest, curve_fit a sinusoidal function for it
        
        # Try out multiple phases for each sin function such that their conjunction matches original ifft OR rawData*
        
        # OR create ifft of each individual frequency of interest, curve-fit them, then add together to predict
        
        # for timeScale in range(len(variant_timeScales)):
        #     variantPredict[var].fInverseValuesStats[timeScale], _ = sp.optimize.curve_fit(objectiveSin, np.arange(len(variantData[var].fInverseValues[timeScale])), variantData[var].fInverseValues[timeScale]);
        #     a = variantPredict[var].fInverseValuesStats[timeScale][0];
        #     b = variantPredict[var].fInverseValuesStats[timeScale][1];
        #     c = variantPredict[var].fInverseValuesStats[timeScale][2];
        #     d = variantPredict[var].fInverseValuesStats[timeScale][3];
        #     originalFitWindowSize = variant_tickWindows[0];
        #     for value in range(0,variant_tickWindowsPredict[0]):
        #         variantPredict[var].fInverseValues[timeScale][tickWindowIndex][value] = a * np.sin(b * value + c) + (a * np.sin(b * originalFitWindowSize + c) + d); # a * np.sin(b * x + c) + d

        # Plot Predicted Inverse FFT
        #for timeScale in range(len(variant_timeScales)):
            
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
    
        ########################################################
        ## Finite & Infinite Impulse Response Filters (FIR & IIR)
        ########################################################
        t0 = time.time();
        print('Output FIR Filter');
        # Obtain Desired Frequencies of Interest (latter is obtained from FFT) and Create Their Boundaries. Then FIR filter these frequencies from rawValues.
        for timeScale in range(len(variant_timeScales)):
            fs = 1/(variant_timeScales[timeScale] * 60); #frequency of sample (in Hz)
            nyq = fs * 0.5; #nyquist
            fOfInterest = [];
            fOfInterest = variantData[var].fPSDFilteredHat[timeScale];
            for freq in range(len(fOfInterest)):
                if freq == 0:
                    fOfInterest[freq] = 0;
                elif (freq != 0 and fOfInterest[freq] > 0) or (freq != 0 and fOfInterest[freq] < 0):
                    fOfInterest[freq] = 1;
            for tickWindowIndex in range(len(FIRTickWindows)):
                tickWindowSize = FIRTickWindows[tickWindowIndex];
                FIRFilterPoints = [];
                # FIR filter only the frequencies below 1 nyquist
                FIRFilterPoints = sp.signal.firls(tickWindowSize, variantData[var].fDist[timeScale][(variantData[var].fDist[timeScale]/nyq) < 1], fOfInterest[(variantData[var].fDist[timeScale]/nyq) < 1], fs=fs);
                variantData[var].FIRFilteredValues[timeScale][tickWindowIndex] = sp.signal.lfilter(FIRFilterPoints, 1.0, variantData[var].rawValues[timeScale]);
    
        # Plot FIR Filtered rawValues
        for timeScale in range(len(variant_timeScales)):
            for tickWindowIndex in range(len(FIRTickWindows)):
                axs[9,timeScale].plot(variantData[var].FIRFilteredValues[timeScale][tickWindowIndex], color='teal', alpha=0.6); #plot simpleMovingAverage
                axs[9,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='FIR');
                axs[9,timeScale].set_ylim([np.nanmin(variantData[var].FIRFilteredValues) - np.std([np.nanmin(variantData[var].FIRFilteredValues), np.nanmax(variantData[var].FIRFilteredValues)]) *.5, np.nanmax(variantData[var].FIRFilteredValues) + np.std([np.nanmin(variantData[var].FIRFilteredValues), np.nanmax(variantData[var].FIRFilteredValues)]) * .5]);
    
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
    
        ########################################################
        ## Output Hilbert Transform (HILB)
        ########################################################
        t0 = time.time();
        print('Output Hilbert Transform (HILB)');
        # Hilbert Transform
        for timeScale in range(len(variant_timeScales)):
            variantData[var].fHilb[timeScale] = sp.signal.hilbert(variantData[var].rawValues[timeScale]);
        # Amplitude Envelope
        for timeScale in range(len(variant_timeScales)):
            variantData[var].fAmpEnv[timeScale] = np.abs(variantData[var].fHilb[timeScale]);
        # Instantaneous Phase
        for timeScale in range(len(variant_timeScales)):
            variantData[var].fInstPhase[timeScale] = np.unwrap(np.angle(variantData[var].fHilb[timeScale]));
        # Instantaneous Frequency
        for timeScale in range(len(variant_timeScales)):
            fs = 1/(variant_timeScales[timeScale] * 60); #frequency of sample (in Hz)
            variantData[var].fInstFreq[timeScale][1:] = np.diff(variantData[var].fInstPhase[timeScale]) / (2.0 * np.pi) * fs;
    
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
            
        ########################################################
        ## Variant Plot
        ########################################################
        t0 = time.time();
        print('Plotting Outputs...');
        for ax in axs.flat:
            ax.label_outer(); #hide x labels and tick labels for top plots and y ticks for right plots.
            ax.grid(False);
            plt.rcParams.update({'font.size': 6});
            plt.setp(ax.spines.values(), linewidth=.2);
        fig.suptitle(variantData[var].rawCurrency + ' per ' + variantData[var].name);
        # for oNoiseLooper in range(0,oNoiseNumMultipliers):
        #     plt.figtext(0.5, - .05 - (.05 * oNoiseLooper), ['Oscillation = ' + str("%.3g" % oNoiseMultipliers[oNoiseLooper]) + 'Hz or ' + str("%.3g" % (oNoiseMultipliers[oNoiseLooper] * (60 * 60))) + 'cph or ' + str("%.3g" % (oNoiseMultipliers[oNoiseLooper] * (60 * 60 * 24))) + 'cpd or ' + str("%.3g" % (oNoiseMultipliers[oNoiseLooper] * (60 * 60 * 24 * 365))) + 'cpy.'], ha="center", fontsize=7, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
        plt.show();
        
        t1 = time.time();
        if debugTimers == 1:
            print(str(t1-t0));
        
        t1Loop = time.time();
        print('Loop Time: ' + str(round(t1Loop-t0Loop,2)) + 's');
        print('===== FIN WITHIN-VARIANT ANALYSES =====');
print('=============== FIN ===============');

