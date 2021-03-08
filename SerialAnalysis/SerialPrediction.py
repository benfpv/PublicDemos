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

########################################################
## Introduction
########################################################
# Welcome!
# Predict futures based on serial input.
# Warning - Due to nature of digitally represented serial data (tick graphs), oscillations may not be captured naturally (recording fast-oscillating objects via camera results in different percieved Hz).
print('=============== START ===============');

########################################################
## Quick Options
########################################################
variant_names = ['TEST', 'TESTA']; #name - e.g., BTC
variant_rawCurrency = ['USD', 'USD']; #native or raw currency type of data input - e.g., USD
#1440m = 1day, 10080m = 7.5days, 43200m = 30days.
variant_timeScales = [10, 60, 720, 1440]; #time scale in mins - e.g., 5 min per tick. Each value must be divisible by the minimum value without remainder.
variant_tickWindows = [100]; #tick window in the unit of timeScale - e.g., 10 ticks of timeScale units.

########################################################
## Classes
########################################################
class VariantData(): #variant data
    def __init__(self, name, rawCurrency, timeScales, tickWindows, debugoNoiseValues, rawValues, linRegressValues, movingAverageValues, movingSDValues, movingPeaks, movingSupports, linRegressPeaks, linRegressSupports, fHat, fPSD, fDist, fPSDFilteredHat, fInverseValues, FIRFilteredValues, fHilb, fAmpEnv, fInstPhase, fInstFreq):
        self.name = name; #name/identifier - e.g., BTC
        self.rawCurrency = rawCurrency; #native currency type - e.g., USD
        self.timeScales = timeScales; #time scale - e.g., 5 min per tick
        self.tickWindows = tickWindows; #time window in the unit of timeScale - e.g., 10 ticks of timeScale units.
        self.debugoNoiseValues = debugoNoiseValues;
        self.rawValues = rawValues; #value in native currency type - e.g., 1 BTC
        self.linRegressValues = linRegressValues;
        self.movingAverageValues = movingAverageValues;
        self.movingSDValues = movingSDValues;
        self.movingPeaks = movingPeaks;
        self.movingSupports = movingSupports;
        self.linRegressPeaks = linRegressPeaks;
        self.linRegressSupports = linRegressSupports;
        self.fHat = fHat;
        self.fPSD = fPSD;
        self.fDist = fDist;
        self.fPSDFilteredHat = fPSDFilteredHat;
        self.fInverseValues = fInverseValues;
        self.FIRFilteredValues = FIRFilteredValues;
        self.fHilb = fHilb;
        self.fAmpEnv = fAmpEnv;
        self.fInstPhase = fInstPhase;
        self.fInstFreq = fInstFreq;

########################################################
## Establish VariantData Structures
########################################################
variantData = [0] * len(variant_names);
for var in range(len(variant_names)):
    init_name = variant_names[var];
    init_rawCurrency = variant_rawCurrency[var];
    init_timeScales = variant_timeScales;
    init_timeWindows = variant_tickWindows;
    init_rawValues = [];
    init_name = variant_names[var];
    init_rawCurrency = variant_rawCurrency[var];
    init_timeScales = variant_timeScales;
    init_tickWindows = variant_tickWindows;
    init_debugoNoiseValues = np.empty((len(variant_timeScales), variant_tickWindows[0]));
    init_debugoNoiseValues[:] = np.NaN;
    init_rawValues = np.empty((len(variant_timeScales), variant_tickWindows[0]));
    init_rawValues[:] = np.NaN;
    init_linRegressValues = np.empty((len(variant_timeScales), variant_tickWindows[0]));
    init_linRegressValues[:] = np.NaN;
    init_movingAverageValues = np.empty((len(variant_timeScales), len(np.arange(2, variant_tickWindows[0] * .25, 2)), variant_tickWindows[0]));
    init_movingAverageValues[:] = np.NaN;
    init_movingSDValues = np.empty((len(variant_timeScales), len(np.arange(2, variant_tickWindows[0] * .25, 2)), variant_tickWindows[0]));
    init_movingSDValues[:] = np.NaN;
    init_movingPeaks = np.empty((len(variant_timeScales), len(np.arange(2, variant_tickWindows[0] * .25, 2)), variant_tickWindows[0]));
    init_movingPeaks[:] = np.NaN;
    init_movingSupports = np.empty((len(variant_timeScales), len(np.arange(2, variant_tickWindows[0] * .25, 2)), variant_tickWindows[0]));
    init_movingSupports[:] = np.NaN;
    init_linRegressPeaks = np.empty((len(variant_timeScales), len(np.arange(2, variant_tickWindows[0] * .25, 2)), variant_tickWindows[0]));
    init_linRegressPeaks[:] = np.NaN;
    init_linRegressSupports = np.empty((len(variant_timeScales), len(np.arange(2, variant_tickWindows[0] * .25, 2)), variant_tickWindows[0]));
    init_linRegressSupports[:] = np.NaN;
    init_fHat = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fHat[:] = np.NaN;
    init_fPSD = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fPSD[:] = np.NaN;
    init_fDist = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fDist[:] = np.NaN;
    init_fPSDFilteredHat = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fPSDFilteredHat[:] = np.NaN;
    init_fInverseValues = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fInverseValues[:] = np.NaN;
    init_FIRFilteredValues = np.empty((len(variant_timeScales), len(np.arange(2, variant_tickWindows[0] * .25, 2)), variant_tickWindows[0]), dtype = 'complex_');
    init_FIRFilteredValues[:] = np.NaN;
    init_fHilb = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fHilb[:] = np.NaN;
    init_fAmpEnv = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fAmpEnv[:] = np.NaN;
    init_fInstPhase = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fInstPhase[:] = np.NaN;
    init_fInstFreq = np.empty((len(variant_timeScales), variant_tickWindows[0]), dtype = 'complex_');
    init_fInstFreq[:] = np.NaN;
    variantData[var] = VariantData(init_name, init_rawCurrency, init_timeScales, init_timeWindows, init_debugoNoiseValues, init_rawValues, init_linRegressValues, init_movingAverageValues, init_movingSDValues, init_movingPeaks, init_movingSupports, init_linRegressPeaks, init_linRegressSupports, init_fHat, init_fPSD, init_fDist, init_fPSDFilteredHat, init_fInverseValues, init_FIRFilteredValues, init_fHilb, init_fAmpEnv, init_fInstPhase, init_fInstFreq);

    ########################################################
    ## Generate/Import Variant's rawData
    ########################################################
    # Sort timeScales
    variant_timeScales = np.sort(variant_timeScales);
    # Generate Random VariantData of Max Length at Min (Best) Resolution
    print('Generating Random VariantData');
    t0 = time.time();
    randomValues = np.empty((variant_tickWindows[0] * int(max(variant_timeScales) / min(variant_timeScales))));
    randomValues[:] = np.NaN;
    for randValue in range(len(randomValues)):
        randomValues[randValue] = random.randint(0,10000) * .01;
    # Map RandomVariantData by timeScale
    for timeScale in range(len(variant_timeScales)):
        for value in range(variant_tickWindows[0]):
            tempIndex = int(len(randomValues) - (len(randomValues) / (max(variant_timeScales) / variant_timeScales[timeScale]))) + (value * int(variant_timeScales[timeScale] / min(variant_timeScales)));
            if int(variant_timeScales[timeScale] / min(variant_timeScales)) == 1:
                variantData[var].rawValues[timeScale][value] = randomValues[tempIndex];
            elif int(variant_timeScales[timeScale] / min(variant_timeScales)) > 1:
                variantData[var].rawValues[timeScale][value] = np.mean(randomValues[tempIndex : tempIndex + int(variant_timeScales[timeScale] / min(variant_timeScales))]);
    t1 = time.time();
    print(str(t1-t0));

    # Add Oscillating Noise (Random between .01Hz (10 sec) to .0000001Hz (10,000,000 sec or ~116 days)), of amplitude 1 SD of, and to, the randomValues.
    print('Adding Oscillating Noise to the randomValues');
    t0 = time.time();
    oNoiseValues = np.empty(variant_tickWindows[0] * int(max(variant_timeScales) / min(variant_timeScales)));
    oNoiseValues[:] = np.NaN;
    #oNoiseNumMultipliers = random.randint(1,6);
    oNoiseNumMultipliers = random.randint(6,12);
    oNoiseMultipliers = np.empty(oNoiseNumMultipliers);
    oNoiseMultipliers[:] = np.NaN;
    oNoiseAmpMultipliers = np.empty(oNoiseNumMultipliers);
    oNoiseAmpMultipliers[:] = np.NaN;
    for oNoiseLooper in range(0,oNoiseNumMultipliers):
        print(str(oNoiseLooper+1) + '/' + str(oNoiseNumMultipliers));
        #oNoiseMultipliers[oNoiseLooper] = min(variant_timeScales) * 2 * np.pi * (.01 ** random.randint(1,6));
        oNoiseMultipliers[oNoiseLooper] = min(variant_timeScales) * 2 * np.pi * (.1 ** random.randint(1,60));
        oNoiseAmpMultipliers[oNoiseLooper] = np.std(randomValues) * random.randint(1,20) * .1;
        for oNoiseValue in range(len(oNoiseValues)):
            if oNoiseLooper == 0:
                oNoiseValues[oNoiseValue] = np.sin(oNoiseMultipliers[oNoiseLooper] * oNoiseValue) * oNoiseAmpMultipliers[oNoiseLooper];
            else:
                oNoiseValues[oNoiseValue] += np.sin(oNoiseMultipliers[oNoiseLooper] * oNoiseValue) * oNoiseAmpMultipliers[oNoiseLooper];
    # Map RandomVariantData by timeScale
    for timeScale in range(len(variant_timeScales)):
        for value in range(variant_tickWindows[0]):
            tempIndex = int(len(oNoiseValues) - (len(oNoiseValues) / (max(variant_timeScales) / variant_timeScales[timeScale]))) + (value * int(variant_timeScales[timeScale] / min(variant_timeScales)));
            if int(variant_timeScales[timeScale] / min(variant_timeScales)) == 1:
                variantData[var].debugoNoiseValues[timeScale][value] = oNoiseValues[tempIndex];
                variantData[var].rawValues[timeScale][value] += oNoiseValues[tempIndex];
            elif int(variant_timeScales[timeScale] / min(variant_timeScales)) > 1:
                variantData[var].debugoNoiseValues[timeScale][value] = np.mean(oNoiseValues[tempIndex : tempIndex + int(variant_timeScales[timeScale] / min(variant_timeScales))]);
                variantData[var].rawValues[timeScale][value] += np.mean(oNoiseValues[tempIndex : tempIndex + int(variant_timeScales[timeScale] / min(variant_timeScales))]);
    t1 = time.time();
    print(str(t1-t0));

    ########################################################
    ## Prepare Output Structures
    ########################################################
    # Plot Structures
    numSubPlots = 9;
    fig, axs = plt.subplots(numSubPlots,len(variant_timeScales), dpi=300);
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
        axs[0,timeScale].set_ylim([np.min(variantData[var].rawValues) - np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) *.5, np.max(variantData[var].rawValues) + np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) * .5]);

        axs[1,timeScale].plot(variantData[var].rawValues[timeScale], color='dimgrey', alpha=0.3);
        axs[1,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='r');
        axs[1,timeScale].set_ylim([np.min(variantData[var].rawValues) - np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) *.5, np.max(variantData[var].rawValues) + np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) * .5]);
        
        axs[2,timeScale].plot(variantData[var].rawValues[timeScale], color='dimgrey', alpha=0.3);
        axs[2,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='BB');
        axs[2,timeScale].set_ylim([np.min(variantData[var].rawValues) - np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) *.5, np.max(variantData[var].rawValues) + np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) * .5]);

        axs[3,timeScale].plot(variantData[var].rawValues[timeScale], color='dimgrey', alpha=0.3);
        axs[3,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='Extr');
        axs[3,timeScale].set_ylim([np.min(variantData[var].rawValues) - np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) *.5, np.max(variantData[var].rawValues) + np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) * .5]);

        axs[7,timeScale].plot(variantData[var].rawValues[timeScale], color='dimgrey', alpha=0.3);
        axs[7,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='fftInv');
        axs[7,timeScale].set_ylim([np.min(variantData[var].rawValues) - np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) *.5, np.max(variantData[var].rawValues) + np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) * .5]);

    t1 = time.time();
    print(str(t1-t0));

    ########################################################
    ## Output Linear Regression Line (r)
    ########################################################
    print('Output Linear Regression Line (r)');
    t0 = time.time();
    # Linear Regression
    for timeScale in range(len(variant_timeScales)):
        linRegress = np.NaN;
        linRegress = sp.stats.linregress(np.arange(0,variant_tickWindows[0]),variantData[var].rawValues[timeScale]);
        for value in range(0,variant_tickWindows[0]):
            variantData[var].linRegressValues[timeScale][value] = linRegress.slope * value + linRegress.intercept;

    # Plot r
    for timeScale in range(len(variant_timeScales)):
        axs[1,timeScale].plot(variantData[var].linRegressValues[timeScale], color='teal', alpha=0.6);
        
    t1 = time.time();
    print(str(t1-t0));

    ########################################################
    ## Output Moving Average +/- SD (Bollinger Bands)
    ########################################################
    t0 = time.time();
    print('Output Moving Average +/- SD (Bollinger Bands)');
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w;

    FIRTickWindows = np.arange(3, variant_tickWindows[0] * .25, 2, dtype='int');
    # Simple Moving Average of Native Currency
    for timeScale in range(len(variant_timeScales)):
        for tickWindowIndex in range(len(FIRTickWindows)):
            tickWindowSize = FIRTickWindows[tickWindowIndex];
            variantData[var].movingAverageValues[timeScale][tickWindowIndex][tickWindowSize-1:] = moving_average(variantData[var].rawValues[timeScale], tickWindowSize);
            for value in range(0,variant_tickWindows[0] - tickWindowIndex + 1):
                variantData[var].movingSDValues[timeScale][tickWindowIndex][tickWindowIndex + value - 1] = np.std(variantData[var].rawValues[timeScale][value:tickWindowIndex + value]);

    # Plot SMA +/- SD
    for timeScale in range(len(variant_timeScales)):
        for tickWindowIndex in range(len(FIRTickWindows)):
            axs[2,timeScale].plot(variantData[var].movingAverageValues[timeScale][tickWindowIndex], color='teal', alpha=0.1); #plot simpleMovingAverage
            axs[2,timeScale].plot(variantData[var].movingAverageValues[timeScale][tickWindowIndex] + variantData[var].movingSDValues[timeScale][tickWindowIndex], '-', linewidth=1, color='teal', alpha=0.05); #plot simpleMovingAverage + SD
            axs[2,timeScale].plot(variantData[var].movingAverageValues[timeScale][tickWindowIndex] - variantData[var].movingSDValues[timeScale][tickWindowIndex], '-', linewidth=1, color='teal', alpha=0.05); #plot simpleMovingAverage - SD

    t1 = time.time();
    print(str(t1-t0));

    ########################################################
    ## Output Peaks and Supports
    ########################################################
    t0 = time.time();
    print('Output Peaks and Supports (values in movingAverage windows that are outside of 95% CI)');
    # Find Pseudo Peaks and Supports
    for timeScale in range(len(variant_timeScales)):
        for tickWindowIndex in range(len(FIRTickWindows)):
            tickWindowSize = FIRTickWindows[tickWindowIndex];
            for value in range(0,variant_tickWindows[0] - tickWindowIndex):
                if variantData[var].rawValues[timeScale][value] > variantData[var].movingAverageValues[timeScale][tickWindowIndex][value] + variantData[var].movingSDValues[timeScale][tickWindowIndex][value]:
                    variantData[var].movingPeaks[timeScale][tickWindowIndex][value] = variantData[var].rawValues[timeScale][value]; # peaks
                if variantData[var].rawValues[timeScale][value] < variantData[var].movingAverageValues[timeScale][tickWindowIndex][value] - variantData[var].movingSDValues[timeScale][tickWindowIndex][value]:
                    variantData[var].movingSupports[timeScale][tickWindowIndex][value] = variantData[var].rawValues[timeScale][value]; # supports
    
    # Plot Pseudo Peaks and Supports
    for timeScale in range(len(variant_timeScales)):
        for tickWindowIndex in range(len(FIRTickWindows)):
            axs[3,timeScale].plot(variantData[var].movingPeaks[timeScale][tickWindowIndex], '.', linewidth=1, markersize=1, color='skyblue', alpha=0.8); #plot simpleMovingAverage + SD
            axs[3,timeScale].plot(variantData[var].movingSupports[timeScale][tickWindowIndex], '.', linewidth=1, markersize=1, color='steelblue', alpha=0.8); #plot simpleMovingAverage - SD
    
    t1 = time.time();
    print(str(t1-t0));
    
    ########################################################
    ## Output r for Peaks and Supports **************FIX*****************??
    ########################################################
    # r for Peaks and Supports
    for timeScale in range(len(variant_timeScales)):
        for tickWindowIndex in range(len(FIRTickWindows)):
            tickWindowSize = FIRTickWindows[tickWindowIndex];
            linRegressPeaks = np.NaN;
            linRegressPeaksLength = len(variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])])
            if len(np.unique(np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex]))) > 1:
                linRegressPeaks = sp.stats.linregress(np.arange(0,len(variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])])), variantData[var].movingPeaks[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingPeaks[timeScale][tickWindowIndex])]);
            linRegressSupports = np.NaN;
            linRegressSupportsLength = len(variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])]);
            if len(np.unique(np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex]))) > 1:
                linRegressSupports = sp.stats.linregress(np.arange(0,len(variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])])), variantData[var].movingSupports[timeScale][tickWindowIndex][~np.isnan(variantData[var].movingSupports[timeScale][tickWindowIndex])]);

    # Plot r for peaks and Supports
    for timeScale in range(len(variant_timeScales)):
        for tickWindowIndex in range(len(FIRTickWindows)):
            tickWindowSize = FIRTickWindows[tickWindowIndex];
            axs[3,timeScale].plot(variantData[var].linRegressPeaks[timeScale][tickWindowIndex], color='skyblue', alpha=0.2);
            axs[3,timeScale].plot(variantData[var].linRegressSupports[timeScale][tickWindowIndex], color='steelblue', alpha=0.2);
            
    ########################################################
    ## Output Fast Fourier Transform (FFT), Power Spectrum Density (fPSD), Frequency Distribution (fDist)
    ########################################################
    t0 = time.time();
    print('Output Fast Fourier Transform (FFT), Power Spectrum Density (fPSD), Frequency Distribution (fDist)');
    # Fast Fourier Transform
    for timeScale in range(len(variant_timeScales)):
        variantData[var].fHat[timeScale] = np.fft.fft(variantData[var].rawValues[timeScale], variant_tickWindows[0]);
    # Power Spectrum Density
    for timeScale in range(len(variant_timeScales)):
        variantData[var].fPSD[timeScale] = variantData[var].fHat[timeScale] * np.conj(variantData[var].fHat[timeScale]) / variant_tickWindows[0];
    # Frequency Distribution
    for timeScale in range(len(variant_timeScales)):
        dt = variant_timeScales[timeScale] * 60; #time per 1 sample in sec
        variantData[var].fDist[timeScale] = (1/(dt * variant_tickWindows[0])) * np.arange(variant_tickWindows[0]);
    
    # Plot FHat (Except first value)
    for timeScale in range(len(variant_timeScales)):
        fs = 1/(variant_timeScales[timeScale] * 60); #frequency of sample (in Hz)
        nyq = fs * 0.5; #nyquist
        axs[4,timeScale].plot(variantData[var].fHat[timeScale], color='teal', alpha=0.8);
        axs[4,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='fftHat');
        axs[4,timeScale].set_ylim([np.min(variantData[var].fHat) - np.std([np.min(variantData[var].fHat), np.max(variantData[var].fHat)]) * .5, np.max(variantData[var].fHat) + np.std([np.min(variantData[var].fHat), np.max(variantData[var].fHat)]) * .5]);
        
    # Plot fftL fDist vs. fPSD (except first value)
    for timeScale in range(len(variant_timeScales)):
        fs = 1/(variant_timeScales[timeScale] * 60); #frequency of sample (in Hz)
        nyq = fs * 0.5; #nyquist
        axs[5,timeScale].plot(variantData[var].fDist[timeScale][1:], variantData[var].fPSD[timeScale][1:], color='teal', alpha=0.8);
        axs[5,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='fftL');
        axs[5,timeScale].set_ylim([np.min(variantData[var].fPSD) - np.std([np.min(variantData[var].fPSD), (np.max(variantData[var].fPSD) - np.max(variantData[var].fPSD) * .95)]) * .5, (np.max(variantData[var].fPSD) - np.max(variantData[var].fPSD) * .95) + np.std([np.min(variantData[var].fPSD), (np.max(variantData[var].fPSD) - np.max(variantData[var].fPSD) * .95)]) * .5]);
        # Plot nyquist line
        axs[5,timeScale].axvline(x=nyq, color='violet', alpha=0.8);
        # Plot Hat/Inverse Filtering cutoff lines
        axs[5,timeScale].axhline(y=np.mean(variantData[var].fPSD[timeScale]) * .6, color='violet', alpha=0.8);
        
    t1 = time.time();
    print(str(t1-t0));

    ########################################################
    ## Output PSD Filter and Inverse FFT (IFFT)
    ########################################################
    t0 = time.time();
    print('Output PSD Filter and Inverse FFT');
    # Filter out noise with PSD Filter then Inverse FFT the Filtered Values
    for timeScale in range(len(variant_timeScales)):
        variantData[var].fPSDFilteredHat[timeScale] = (variantData[var].fPSD[timeScale] > np.mean(variantData[var].fPSD[timeScale]) * .6) * variantData[var].fHat[timeScale];
        variantData[var].fInverseValues[timeScale] = np.fft.ifft(variantData[var].fPSDFilteredHat[timeScale]);

    # Plot Debug Oscillation Noise Only
    for timeScale in range(len(variant_timeScales)):
        axs[6,timeScale].plot(variantData[var].debugoNoiseValues[timeScale], color='deepskyblue', alpha=0.8);
        axs[6,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='debugO');
        axs[6,timeScale].set_ylim([np.min(variantData[var].debugoNoiseValues) - np.std([np.min(variantData[var].debugoNoiseValues), np.max(variantData[var].debugoNoiseValues)]) *.5, np.max(variantData[var].debugoNoiseValues) + np.std([np.min(variantData[var].debugoNoiseValues), np.max(variantData[var].debugoNoiseValues)]) * .5]);

    # Plot Inverse Values
    for timeScale in range(len(variant_timeScales)):
        axs[7,timeScale].plot(variantData[var].fInverseValues[timeScale], color='teal', alpha=0.8);
        axs[7,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='ifft');
        axs[7,timeScale].set_ylim([np.min(variantData[var].rawValues) - np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) *.5, np.max(variantData[var].rawValues) + np.std([np.min(variantData[var].rawValues), np.max(variantData[var].rawValues)]) * .5]);

    t1 = time.time();
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
            axs[8,timeScale].plot(variantData[var].FIRFilteredValues[timeScale][tickWindowIndex], color='teal', alpha=0.1); #plot simpleMovingAverage
            axs[8,timeScale].set(xlabel=str(variant_timeScales[timeScale]) + ' min tick', ylabel='FIR');
            axs[8,timeScale].set_ylim([np.nanmin(variantData[var].FIRFilteredValues) - np.std([np.nanmin(variantData[var].FIRFilteredValues), np.nanmax(variantData[var].FIRFilteredValues)]) *.5, np.nanmax(variantData[var].FIRFilteredValues) + np.std([np.nanmin(variantData[var].FIRFilteredValues), np.nanmax(variantData[var].FIRFilteredValues)]) * .5]);

    t1 = time.time();
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
    print(str(t1-t0));

    ########################################################
    ## Variant Plot
    ########################################################
    for ax in axs.flat:
        ax.label_outer(); #hide x labels and tick labels for top plots and y ticks for right plots.
        ax.grid(False);
        plt.rcParams.update({'font.size': 7});
        plt.setp(ax.spines.values(), linewidth=.5);
    fig.suptitle(variantData[var].rawCurrency + ' per ' + variantData[var].name);
    # for oNoiseLooper in range(0,oNoiseNumMultipliers):
    #     plt.figtext(0.5, - .05 - (.05 * oNoiseLooper), ['Oscillation = ' + str("%.3g" % oNoiseMultipliers[oNoiseLooper]) + 'Hz or ' + str("%.3g" % (oNoiseMultipliers[oNoiseLooper] * (60 * 60))) + 'cph or ' + str("%.3g" % (oNoiseMultipliers[oNoiseLooper] * (60 * 60 * 24))) + 'cpd or ' + str("%.3g" % (oNoiseMultipliers[oNoiseLooper] * (60 * 60 * 24 * 365))) + 'cpy.'], ha="center", fontsize=7, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.show();

########################################################
## Output Phase-Locking Value (PLV)
########################################################
t0 = time.time();
print('Output Phase-Locking Value (PLV)');



t1 = time.time();
print(str(t1-t0));





print('================ FIN ================');
