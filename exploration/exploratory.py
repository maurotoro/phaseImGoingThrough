# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:50:16 2016

Explorations on Claudia Fernstein PhD thesis,
Load the behavioral data and create a plot that displays all the behavioral 
data over a trial breathing frequency.

@author: soyunkope
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import linregress

def DZBP(signal, time, low=0.1, high=30, order=3, sr=1893.9393939393942):
    """
    Detrend, signal - linearRegression
    Z-Score, (x - mean(signal))/std(signal)
    BandPass, filtfilt(butterworth[a,b, hig-low, order])
    """
    y = signal
    x = time
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    yR = (slope*x+intercept)
    sample = y - yR
    z = (sample-sample.mean())/sample.std()
    nyq = 0.5 * sr
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype="bandpass", analog=False)
    res = filtfilt(b, a, z)
    return res


def LPZ(signal, cutoff=40, order=3, sr=1893.9393939393942):
    """
    Low-Pass & Z-score
    """
    nyq = 0.5 * sr
    b, a = butter(order, cutoff/nyq, btype="lowpass", analog=False)
    xF = filtfilt(b, a, signal)
    z = (xF-xF.mean())/xF.std()
    return z

def loadZC(rat, day):
    """
    Function to load the data from Claudia 2006:
    
    DATASET:
    -------
        {"f4": ["05_31_06b", "06_01_06b", "06_02_06b", "06_03_06b"],
        "f5": ["05_31_06b", "06_03_06b", "06_05_06b", "06_06_06b"],
        "p9": ["11_18_04", "11_20_04", "11_23_04", "11_26_04", 
        "11_29_04", "12_01_04", "12_03_04", "12_07_04",
        "11_19_04", "11_22_04", "11_25_04", "11_28_04", 
        "11_30_04", "12_02_04", "12_06_04"]}

    params:
    ------
        rat: string
            Subject to look at.
        
        day: int
            What session to look at

    returns:
    -------
        rdata: struct 
            filled of the analysis from Claudia, to be used or...
    """
    rats = {"f4": ["05_31_06b", "06_01_06b", "06_02_06b", "06_03_06b"],
            "f5": ["05_31_06b", "06_03_06b", "06_05_06b", "06_06_06b"],
            "p9": ["11_18_04", "11_20_04", "11_23_04", "11_26_04", 
                   "11_29_04", "12_01_04", "12_03_04", "12_07_04",
                   "11_19_04", "11_22_04", "11_25_04", "11_28_04", 
                   "11_30_04", "12_02_04", "12_06_04"]}
    fpa = "/Users/soyunkope/Documents/INDP2015/2016_S02/R02_ZachClau/"
    fpb = "phaseImGoingThrough/data/"
    fna = rat
    fnb = rats[rat][day]
    fname = fpa+fpb+fna+"/"+fnb+"/"+"sniff_raw.mat"
    mdata = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    rdata = mdata["sniff"]
    return rdata


