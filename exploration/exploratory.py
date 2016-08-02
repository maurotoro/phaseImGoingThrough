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


def HLB(signal, sr=1893.9393939393942):
    '''
    Do the Hilbert transform of the data, and returns the analytic signal,
    envelope, instantaneous phase, intantaneous frequency and analytic phase
    of the signal.
    
    Params:
    ------
        signal: array
            Signal to be analysed
    Returns:
    -------
     ana_sig: array 
         analytic signal
     envelope: array
         instantaneous amplitude
     in_phase: array
         instantaneous phase
     ins_freq: array
         instantaneous frequency
     ana_phase: array
         analytic phase
    '''
    ana_sig = hilbert(signal)
    envelope = np.abs(ana_sig)
    ins_phase = np.unwrap(np.angle(ana_sig))
    insF = np.diff(ins_phase) * (sr/(2.0*np.pi))
    ins_freq = np.hstack((0, insF))
    ana_phase = np.arctan2(np.imag(ana_sig), np.real(ana_sig))
    return ana_sig, envelope, ins_phase, ins_freq, ana_phase


def loadZCB(rat, day):
    """
    Function to load the behavioral data from Claudia 2006. Use RDC to be
    called from rat_day_cells.py to get all the rats and their respective
    sessions.
    
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
        rat : string
            Subject to look at.
        
        day : string
            What session to look at

    returns:
    -------
        rdata : record 
            Collection of the analysis from Claudia, to be used or...
    """
    fpa = "/Users/soyunkope/Documents/INDP2015/2016_S02/R02_ZachClau/"
    fpb = "phaseImGoingThrough/data/"
    fname = fpa+fpb+rat+"/"+day+"/"+"sniff_raw.mat"
    mdata = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    rdata = mdata["sniff"]
    return rdata



# Load RDC from rat_day_cells.py !!!
rat = "f4"
ses = "05_31_06b"
rd = loadZCB(rat, ses)
cells = RDC[rat][ses]
t0 = rd.data.t0
sr = rd.data.SampFreq
dt = 1/sr
L = len(rd.data.breath)
dur = L/sr
x_time = np.linspace(t0, dur+t0, num=L)
breath = LPZ(rd.data.breath, cutoff=100, sr=sr)
ana_sig, envelope, ins_phase, ins_freq, ana_phase = HLB(breath, sr=sr)
valTrials = np.array(np.nonzero(~np.isnan(sum((rd.events.WaterPokeIn,
                                               rd.events.OdorValveOn),
                                              axis=0)))[0])
trials_TS = rd.events.TrialStart[valTrials]
odorON_TS = trials_TS+rd.events.OdorValveOn[valTrials]
pokeIn_TS = trials_TS+rd.events.OdorPokeIn[valTrials]
pokeOut_TS = trials_TS+rd.events.OdorPokeOut[valTrials]
waterIn_TS = trials_TS+rd.events.WaterPokeIn[valTrials]
reward_TS = trials_TS+rd.events.WaterValveOn[valTrials]
odorValveID = rd.events.OdorValveID[valTrials]
waterValveID = rd.events.WaterPokeID[valTrials]
odors = np.unique(odorValveID)


for odors in odors:
    pass



"""
# LOOKOUT! Seems that DZBP works better, use low=1 and high=30
fig = plt.figure( figsize=(25,10))
ax = fig.add_subplot(111)
ax.plot(x_time, breath, 'k', lw=.5)
ax.plot(trials_TS[valTrials], np.zeros_like(valTrials), 'ob', ms=9)
ax.plot(pokeIn_TS[valTrials], np.zeros_like(valTrials), 'dg', ms=9)
ax.plot(pokeOut_TS[valTrials], np.zeros_like(valTrials), 'dr', ms=9)
ax.plot(odorON_TS[valTrials], np.zeros_like(valTrials), '*m', ms=13)
ax.plot(waterIn_TS[valTrials], np.zeros_like(valTrials), 'sg', ms=9)
ax.plot(reward_TS[valTrials], np.zeros_like(valTrials), '^g', ms=9)
plt.tight_layout()
"""