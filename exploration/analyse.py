# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:37:16 2016

Start the serious stuff...
Only take ana_phase arround behavioral event.
    After DZBP!
Make the PSHT's, using time and phase

Distributed AS IS, it's your fault now.
If no licence on the containing folder, asume GPLv3+CRAPL
@author: Mauro Toro
@email: mauricio.toro@neuro.fchampalimaud.org
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
from pyhht import emd
import scipy.io as sio
from scipy.signal import hilbert, butter, filtfilt, gaussian
from scipy.stats import linregress


def DBPN(signal, low=0.1, high=30, order=3, sr=1893.9393939393942, norm='ZS'):
    """
    Detrend, signal - linearRegression
    BandPass, filtfilt(butterworth[a,b, hig-low, order])
    Normalize: either z-score o feature-scaling
    """
    y = signal
    x = np.arange(0,len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    yR = (slope*x+intercept)
    sample = y - yR
    nyq = 0.5 * sr
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype="bandpass", analog=False)
    sig = filtfilt(b, a, sample)
    res = normalize(sig, method=norm)
    return res


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
    signal = np.asarray(signal)
    ana_sig = hilbert(signal)
    envelope = np.abs(ana_sig)
    ins_phase = np.unwrap(np.angle(ana_sig))
    insF = np.diff(ins_phase) * (sr/(2.0*np.pi))
    ins_freq = np.hstack((0, insF))
    ana_phase = np.arctan2(np.imag(ana_sig), np.real(ana_sig))
    return ana_sig, envelope, ins_phase, ins_freq, ana_phase


def loadCellTS(rat, ses, neu):
    """
    Loads the spike trains from a particular cell from a session of a rat
    Use RDC on rat_day_cells to get the rat, ses and neu strings!
    Parameters
    ----------
    rat : string
        which rat to look at
        
    ses : string
        which sesion to look at
        
    neu : string
        which neuron to look at
    
    Returns
    -------
    """
    fpa = "/Users/soyunkope/Documents/INDP2015/2016_S02/R02_ZachClau/"
    fpb = "phaseImGoingThrough/data/"
    fname = fpa+fpb+rat+'/'+ses+'/'+neu
    nn = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    cell = nn['TS']/10000
    return cell


def normalize(signal, method="ZS"):
    signal = np.asarray(signal)
    res=[]
    if method == "ZS":
        res = (signal-np.mean(signal))/np.std(signal)
    elif method == "FS":
        mn = np.min(signal)
        mx = np.max(signal)
        res = (signal)/(mx-mn)
    return res


def rastify(tXo, cell, marks, x_time):
    for trial, x in zip(tXo, range(len(tXo))):
        rast = [np.nonzero((cell > marks[1][0][trial[i]]) & \
                (cell < marks[1][1][trial[i]])) for i in range(len(trial))]
        ts = [cell[np.array(rast[i])] - x_time[marks[0][trial][i]]
              for i in range(len(trial))]
    return ts


def rastrix(ts, dT=.5, res=.002):
    """
    Look for spikes arround each odor presentation.
    Gives and array of odors size, each odor has as many rows as presentations
    and as many columns as spikes on that presentation
    """
    raster = []
    a = np.linspace(-dT, dT, num=1/res)
    for trial in ts:
        rast = np.zeros((len(trial), len(a)))
        rNX = [[np.nonzero(a >= i)[0].min() for i in j[0]] for j in ts]
        for row in range(len(rast)):
            rast[row][rNX[row]] = 1
        raster.append(rast)
    return raster


def loadData(rat, ses, file='data_11.AUG.16.h5'):
    fpa = '/Users/soyunkope/Documents/scriptkidd/git/phaseImGoingThrough/'
    fpd = 'data/'
    dfName = fpa+fpd+file
    dfile = h5py.File(dfName, mode='a')
    sets = [key for key in sorted(dfile[rat][ses].keys())]
    dd = [key for key in sorted(dfile[rat][ses][sets[0]].keys())]
    ts = [key for key in sorted(dfile[rat][ses][sets[1]].keys())]
    data = {i: np.array(dfile[rat][ses]['data'].get(i)) for i in dd}  
    events = {i: np.array(dfile[rat][ses]['events_timestamps'].get(i))
              for i in ts}
    data['x_time'] = np.linspace(data['time_start'],
                                 data['duration']+data['time_start'],
                                 num=len(data['respiration']))
    dataset = {'data': data, 'events_ts': events}
    dfile.close()
    return dataset


def inhDetection(breath, events, x_time, sr=1893.9393939393942,
                 low=1, high=30, order=3, norm='ZS', frac=5, ratio=1):
    """
    Detects first inhalation after timestamps. Gives the detrended
    bandpassed and normalized breathing sec/ratio arround the event,
    the analytic phase of the breathing for further analysis, and
    the indices of used for everything.
    """
    ev0 = int(sr/ratio)
    ev_ndx = np.array([(x_time >= i).nonzero()[0].min() for i in events])
    pe_ndx = np.array([ev_ndx-ev0, ev_ndx+ev0])
    sniff = np.array([breath[i:j] for i, j in zip(pe_ndx[0], pe_ndx[1])])
    resps = np.array([DBPN(sn, low=low, high=high, order=order, sr=sr, norm=norm)
                      for sn in sniff])
    ana_sig = hilbert(resps)
    ana_phase = np.arctan2(ana_sig.imag, ana_sig.real) 
    finh = np.zeros_like(ev_ndx)
    thresh = -(np.pi/frac)*(frac-1)
    for i in range(len(ev_ndx)):
        print(i)
        if ana_phase[i][ev0] < thresh:
            finh[i] = ev0
        else:
            finh[i] = (ana_phase[i][ev0:] < thresh).nonzero()[0].min()+ev0
    return resps, ana_phase, finh, [ev_ndx, pe_ndx]


file='data_10.AUG.16.h5'
plt.ioff()
expe = {}
for rat in sorted(RDC.keys()):
    expe[rat]={}
    for ses in sorted(RDC[rat]):
        expe[rat][ses]={}
        dataset = loadData(rat, ses, file=file)
        etsN = ['poke_in', 'odor_on', 'poke_out', 'water_in', '1SBPi'] 
        x_time = dataset['data']['x_time']
        breath = dataset['data']['respiration']
        sr = dataset['data']['samp_rate']
        fTit = rat+' '+ses
        #fig = plt.figure(fTit, figsize=(20, 7))
        val = len(etsN)
        axs = [fig.add_subplot(1, val, i+1) for i in range(val)]
        cols = 'rgbmc'
        time = int(sr/3)
        for x in range(val):
            if x == val-1:
                events = dataset['events_ts']['poke_in']-1
            else:
                events = dataset['events_ts'][etsN[x]]
            resps, ana_phase, finh, marks = inhDetection(breath, events, x_time,
                                                          sr=sr)
            expe[rat][ses][etsN[x]] = {'sniffs': resps, 'phase': ana_phase,
                                        'first_inhalation': finh,
                                        'marks': marks}
            print(rat, ses, etsN[x], ': done')
            


