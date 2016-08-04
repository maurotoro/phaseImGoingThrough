# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:50:16 2016

Explorations on Claudia Fernstein PhD thesis,
Load the behavioral data and create a plot that displays all the behavioral 
data over a trial breathing frequency.


Distributed AS IS, it's your fault now.
If no licence on the containing folder, asume GPLv3+CRAPL
@author: Mauro Toro
@email: mauricio.toro@neuro.fchampalimaud.org
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from pyhht import emd
import scipy.io as sio
from scipy.signal import hilbert, butter, filtfilt, gaussian
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
    envelope, instantaneous phase, instantaneous frequency and analytic phase
    of the signal.
    TODO: also use the angular freq? (Hurtado Rubchinsky & Sigvardt 2004)
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
     in_freq: array
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


def firstInhDet(ana_phase, events):
    """
    Detect first inhalation after some events.
    Goes trough ana_phase at events and depending on the phase at each event 
    look for the next inhalation, if already on one, go to the next one, if on
    exhalation, go to the next inhalation
    
    Returns
    -------
    array with the index ofinhalation start index for each event
    """
    finh = np.zeros_like(events)
    for i in range(len(events)):
        if ana_phase[events[i]] < -(np.pi/5)*4:
            finh[i] = events[i]
        else:
            finh[i] = np.nonzero(ana_phase[events[i]:]<-(np.pi/5)*4)[0].min() \
                      + events[i]
    return finh


def HHT(signal, n_imfs=2, t=None, maxiter=200):
    HH = emd.EmpiricalModeDecomposition(signal, t=t,n_imfs=n_imfs,
                                        maxiter=maxiter, nbsym=10)
    IMF = HH.decompose()
    return IMF
    

def figurify(signal, IMF, fig, tit):
    cols = 'rgbm'
    axa = fig.add_subplot(2, 1, 1)
    axb = fig.add_subplot(2, 1, 2, sharex=axa)
    [axa.plot(IMF[i], cols[i]) for i in range(len(IMF))]
    axb.plot(signal)
    axa.axis('tight')
    axa.set_title(tit, fontsize=15)
    return fig

def normalize(signal, method="ZC"):
    if method == "ZC":
        res = (signal-np.mean(signal))/np.std(signal)
    elif method == "FS":
        mn = np.min(signal)
        mx = np.max(signal)
        res = (signal-mn)/(mx-mn)
    return res

def gaussFil(signal, sr="1893.9393939393942", freq=50):
    """
    Implements a lowpass Gaussian Filter over the signal with
    cutoff frequency freq. Works...
    Creates a gaussian window of the size of the cutoff frequency 
    and convolves the signal to it.
    """
    M = sr/freq
    std = M/2
    ker = gaussian(M, std)
    res = np.convolve(signal, ker, mode='same')
    return res

# Load RDC from rat_day_cells.py !!!
rat = "f5"
ses = "06_03_06b"

rd = loadZCB(rat, ses)
cells = RDC[rat][ses]
t0 = rd.data.t0
sr = rd.data.SampFreq
dt = 1/sr
L = len(rd.data.breath)
dur = L/sr
x_time = np.linspace(t0, dur+t0, num=L)
breath = DZBP(rd.data.breath, x_time, low=.5, high=40, sr=sr)

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

pokeIn_IX = np.array([np.nonzero(x_time > pokeIn_TS[i])[0].min()
                      for i in range(len(valTrials))])
odorOn_IX = np.array([np.nonzero(x_time > odorON_TS[i])[0].min()
                      for i in range(len(valTrials))])
pokeOut_IX = np.array([np.nonzero(x_time > pokeOut_TS[i])[0].min()
                       for i in range(len(valTrials))])
marks = [pokeIn_IX-int(sr), pokeOut_IX+int(sr)]


sampB = np.array([breath[marks[0][i]:marks[1][i]] for i in range(valTrials.size)])
sampT = np.array([x_time[marks[0][i]:marks[1][i]] for i in range(valTrials.size)])
# ana_sig, envelope, ins_phase, ins_freq, ana_phase = HLB(breath, sr=sr)
ana_phase = np.array([HLB(sampB[i], sr=sr) for i in range(valTrials.size)])[:,4]

"""
# Looking to use IMF, fuck me, and my dreams....
for rat in RDC.keys():
    for ses in RDC[rat].keys():
        rd = loadZCB(rat, ses)
        cells = RDC[rat][ses]
        t0 = rd.data.t0
        sr = rd.data.SampFreq
        dt = 1/sr
        L = len(rd.data.breath)
        dur = L/sr
        x_time = np.linspace(t0, dur+t0, num=L)
        breath = gaussFil(rd.data.breath, sr=sr, freq=50)
        breath = normalize(breath)
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
        # timestamps into indexes
        pokeIn_IX = np.array([np.nonzero(x_time > pokeIn_TS[i])[0].min()
                              for i in range(len(valTrials))])
        odorOn_IX = np.array([np.nonzero(x_time > odorON_TS[i])[0].min()
                              for i in range(len(valTrials))])
        pokeOut_IX = np.array([np.nonzero(x_time > pokeOut_TS[i])[0].min()
                               for i in range(len(valTrials))])
        marks = [pokeIn_IX-int(sr), pokeOut_IX+int(sr)]
        # Sample Time and breathing to make sense out of this
        sampB = np.array([breath[marks[0][i]:marks[1][i]]
                          for i in range(len(pokeIn_IX))])
        sampT = np.array([x_time[marks[0][i]:marks[1][i]]
                          for i in range(len(pokeIn_IX))])
        IMF = [HHT(sampB[i], t=sampT[i], n_imfs=3) for i in range(len(sampB))]
        tit = rat+'-'+ses
        pp = PdfPages(tit+'.pdf')
        for i in range(len(pokeIn_IX)):
            tit = tit = rat+'-'+ses+'-'+str(i)
            print(tit)
            fig = plt.figure(tit, figsize=(8.27, 11.69), dpi=100)
            f = figurify(sampB[i], IMF[i], fig, tit)
            f.savefig(pp, format='pdf')
            plt.close()
        pp.close()
"""