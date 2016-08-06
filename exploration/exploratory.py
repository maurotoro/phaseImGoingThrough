# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:50:16 2016

Explorations on Claudia Feierstein PhD thesis,
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

    Params:
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


def firstInhDet(ana_phase, events):
    """
    Detect first inhalation after some events.
    Goes trough ana_phase at events and depending on the phase at each event
    look for the next inhalation, if already on one, go to the next one if
    more than 1/5 of the inhalation has , if on exhalation look for the next
    inhalation
    Parameters
    ----------
    ana_phase : array
        analytic phase of the signal

    events : array
        event index to look for.
    Returns
    -------
    array with the index ofinhalation start index for each event
    """
    finh = np.zeros_like(events)
    for i in range(len(events)):
        if ana_phase[events[i]] < -(np.pi/5)*4:
            finh[i] = events[i]
        else:
            finh[i] = np.nonzero(ana_phase[events[i]:] < \
                                 -(np.pi/5)*4)[0].min() + events[i]
    return finh


def HHT(signal, n_imfs=2, t=None, maxiter=200):
    """
    Use and abuse the EMD from jaidevd, still not so useful, may need
    more tricks to work
    """
    HH = emd.EmpiricalModeDecomposition(signal, t=t, n_imfs=n_imfs,
                                        maxiter=maxiter, nbsym=10)
    IMF = HH.decompose()
    return IMF


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


def figurify(signal, IMF, fig, tit):
    """
    Simple plots, the EMD on top 3 max + res; the signal on bottom
    """
    cols = 'rgbm'
    axa = fig.add_subplot(2, 1, 1)
    axb = fig.add_subplot(2, 1, 2, sharex=axa)
    [axa.plot(IMF[i], cols[i]) for i in range(len(IMF))]
    axb.plot(signal)
    axa.axis('tight')
    axa.set_title(tit, fontsize=15)
    return fig


def rastFigs(tXo, x_time, cell, marks, fig, tit):
    """
    Hardcore simple rasters for the dataset.
    Take the trial divided by odors, tXo, look at the marks from timestams
    marks[1]
    """
    for trial, x in zip(tXo, range(len(tXo))):
        rastify = [np.nonzero((cell > marks[1][0][trial[i]]) & \
                   (cell < marks[1][1][trial[i]])) for i in range(len(trial))]
        ts = [cell[np.array(rastify[i])] - x_time[marks[0][trial][i]]
              for i in range(len(trial))]
        ax = fig.add_subplot(len(tXo)+1, 1, x+1)
        ylab = 'Odor-'+str(x+1)
        for i in range(len(ts)):
            ax.plot(ts[i], np.zeros_like(ts[i])+i, '.'+cols[x])
            ax.set_ylabel(ylab)
        ax.axis((-.504, .504, -.4, (len(ts)+.4)))
        ax.plot(np.zeros(2), [-.4, (len(ts)+.4)], 'k', lw=1.5)
    fig.suptitle(tit, fontsize=20)
    return fig


def rastify(tXo, cell, marks, x_time):
    for trial, x in zip(tXo, range(len(tXo))):
        rast = [np.nonzero((cell > marks[1][0][trial[i]]) & \
                (cell < marks[1][1][trial[i]])) for i in range(len(trial))]
        ts = [cell[np.array(rast[i])] - x_time[marks[0][trial][i]]
              for i in range(len(trial))]
    return ts

# Load RDC from rat_day_cells.py !!!

# rat = "f4"
# ses = "05_31_06b"
ppOO = PdfPages('rat_ses_cell-t0_OdorOn.pdf')
ppPI = PdfPages('rat_ses_cell-t0_PokeIn.pdf')
for rat in sorted(RDC.keys()):
    if rat == "f4":
        continue
    for ses in sorted(RDC[rat].keys()):
        if rat == "f5":        
            if ses == '05_31_06b':
                continue
            if ses == '06_03_06b':
                continue
            if ses == '06_05_06b':
                continue
            if ses == '06_06_06b':
                continue
            if ses == '06_07_06b':
                continue
            if ses == '06_08_06b':
                continue
            if ses == '06_09_06b':
                continue
            if ses == '06_11_06b':
                continue
        rd = loadZCB(rat, ses)
        t0 = rd.data.t0
        sr = rd.data.SampFreq
        dt = 1/sr
        L = len(rd.data.breath)
        dur = L/sr
        x_time = np.linspace(t0, dur+t0, num=L)
        breath = gaussFil(rd.data.breath, sr=sr, freq=50)
        breath = normalize(breath)
        valTrials = np.array(np.nonzero(~np.isnan(sum((rd.events.OdorPokeIn,
                                                       rd.events.OdorValveOn,
                                                       rd.events.WaterPokeIn,),
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
        trialXodor = [np.nonzero(odorValveID == odor)[0] for odor in odors]
        
        pokeIn_IX = np.array([np.nonzero(x_time > pokeIn_TS[i])[0].min()
                              for i in range(len(valTrials))])
        odorOn_IX = np.array([np.nonzero(x_time > odorON_TS[i])[0].min()
                              for i in range(len(valTrials))])
        pokeOut_IX = np.array([np.nonzero(x_time > pokeOut_TS[i])[0].min()
                               for i in range(len(valTrials))])
        ana_sig, envelope, ins_phase, ins_freq, ana_phase = HLB(breath, sr=sr)
        fiPI = firstInhDet(ana_phase, pokeIn_IX)
        fiOO = firstInhDet(ana_phase, odorOn_IX)
        dT = .5
        marksPI = [x_time[fiPI]-dT, x_time[fiPI]+dT]
        marksOO = [x_time[fiOO]-dT, x_time[fiOO]+dT]
        tL = [[fiOO, marksOO], [fiPI, marksPI]]
        cols = 'rgbcmy'
        for neu in RDC[rat][ses]:
            cell = loadCellTS(rat, ses, neu)
            titF = rat+'_'+ses+'_'+neu
            print(titF)
            for marks,pp in zip(tL, [ppOO, ppPI]):
                fig = plt.figure(titF, figsize=(8.27, 11.69), dpi=100)
                #ts = rastify(trialXodor, cell, tL, x_time)
                f = rastFigs(trialXodor, x_time, cell, marks, fig, titF)
                f.savefig(pp, format='pdf')
                plt.close()
ppOO.close()
ppPI.close()

## OJO pages arround session F5_ 06_11_06b - 06_13_06b, may have problems
     
"""
# Looking to use IMF, fuck me, and my dreams....
for rat in RDC.keys():
    for ses in RDC[rat].keys():
        rd = loadZCB(rat, ses)
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
            tit = rat+'-'+ses+'-'+str(i)
            print(tit)
            fig = plt.figure(tit, figsize=(8.27, 11.69), dpi=100)
            f = figurify(sampB[i], IMF[i], fig, tit)
            f.savefig(pp, format='pdf')
            plt.close()
        pp.close()
"""
