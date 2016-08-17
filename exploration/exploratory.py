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
import h5py
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
    signal = np.asarray(signal)
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


def markify(events_TS, x_time, ana_phase, frac=5, dT=.5):
    """
        Creates marks for first inhalation given a event timestamp. Gives the
        first inhalations indices, but also the timestamps for +-dT seconds
        from that inhalation to get the spike trains.
        Asumes that if the event ocurred in less than pi/frac of the
        inhalation, that is the first inhalation.
        TODO: Try to speed up procces by starting each time closer to event..
    """
    events = np.array([(x_time >= ev).nonzero[0].min() for ev in events_TS])
    finh = np.zeros_like(events)
    for i in range(len(events)):
        if ana_phase[events[i]] < -(np.pi/frac)*(frac-1):
            finh[i] = events[i]
        else:
            finh[i] = np.nonzero(ana_phase[events[i]:] < \
                                 -(np.pi/frac)*(frac-1))[0].min() + events[i]
    marks = np.array([x_time[finh]-dT, x_time[finh]+dT])
    return events, [finh, marks]


def HHT(signal, n_imfs=2, t=None, maxiter=200):
    """
    Use and abuse the EMD from jaidevd, still not so useful, may need
    more tricks to work
    """
    HH = emd.EmpiricalModeDecomposition(signal, t=t, n_imfs=n_imfs,
                                        maxiter=maxiter, nbsym=10)
    IMF = HH.decompose()
    return IMF


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


def gaussFil(signal, sr=1893.9393939393942, freq=50):
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


def DBPN(signal, low=0.1, high=30, order=3, sr=1893.9393939393942, norm='ZS'):
    """
    Detrend, signal - linearRegression
    BandPass, filtfilt(butterworth[a,b, hig-low, order])
    Normalize: either z-score o feature-scaling
    """
    y = signal
    x = np.arange(len(y))
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


def loadData(rat, ses, file='data_10.AUG.16.h5'):
    fpa = '/Users/soyunkope/Documents/scriptkidd/git/phaseImGoingThrough/'
    fpd = 'data/'
    dfName = fpa+fpd+file
    dfile = h5py.File(dfName, mode='r')
    dd = ['time_start', 'samp_rate', 'cutoff_gaussF', 'duration',
          'respiration']
    ts = ['trials_start', 'poke_in', 'odor_id', 'odor_on', 'poke_out',
          'water_in', 'water_id', 'reward']
    data = {i: np.array(dfile[rat][ses]['data'].get(i)) for i in dd}
    events = {i: np.array(dfile[rat][ses]['events_timestamps'].get(i))
              for i in ts}
    data['x_time'] = np.linspace(data['time_start'],
                                 data['duration']+data['time_start'],
                                 num=len(data['respiration']))
    dataset = {'data': data, 'events_ts': events}
    return dataset


def inhDetection(breath, events, x_time, sr=1893.9393939393942,
                 low=1, high=30, order=3, nmeth='ZS', frac=5, ratio=1):
    """
    Detects first inhalation after timestamps. Gives the detrended
    bandpassed and normalized breathing sec/ratio arround the event,
    the analytic phase of the breathing for further analysis, and
    the indices used for everything in breath scale.
    If the odor was presented during the bregining of the inhalation,
    consider that as the first inhalation. 
    
    Parameters
    ----------
    
    breath : array, one dimension
        Breathing frequency to analyze

    events : array, one dimension
        Timestamps of the events of interet

    x_time : array, one dimension
        Sesion time dimension
    
    sr : float
        Sampling rate of the breathing frequency
    
    low : float
        Lower bound for the bandpass filter

    high : float
        Higher bound for the bandpass filter

    order : int
        Order of the bandpass filter

    nmeth : string
        Type of normalization to use after the bandpass

    frac : float
        (frac-1)*pi/frac defines the begining of the inhalation

    ratio : float
        How much indices around the event to analyze,
        (ratio 1) = (1 sec)
    
    Returns:
    -------
    
    
    """
    ev0 = int(sr/ratio)
    dt = 1/sr
    t0 = x_time[0]
    ev_ndx = np.array(((events-t0)/dt).astype(int))
    pe_ndx = np.array([ev_ndx-ev0, ev_ndx+ev0])
    sniff = np.array([breath[i:j] for i, j in zip(pe_ndx[0], pe_ndx[1])])
    resps = np.array([DBPN(sn, low=low, high=high, order=order, sr=sr,
                           norm=nmeth) for sn in sniff])
    ana_sig = hilbert(resps)
    ana_phase = np.arctan2(ana_sig.imag, ana_sig.real)
    finh = np.zeros_like(ev_ndx)
    thresh = -(np.pi/frac)*(frac-1)
    for i in range(len(ev_ndx)):
        if ana_phase[i][ev0] < thresh:
            finh[i] = (ana_phase[i][:ev0] < thresh).nonzero()[0].max()
        else:
            finh[i] = (ana_phase[i][ev0:] < thresh).nonzero()[0].min()+ev0
    return resps, ana_phase, finh, [ev_ndx, pe_ndx]


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
        ax = fig.add_subplot(len(tXo)+1, 4, x+1)
        ylab = 'Odor-'+str(x+1)
        for i in range(len(ts)):
            ax.plot(ts[i], np.zeros_like(ts[i])+i, '|'+cols[x])
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


def figInhwaves(marks, breath, tit):
    val = len(marks)
    fig = plt.figure(tit, figsize=(20, 7))
    ax = [fig.add_subplot(1, val, i+1) for i in range(val)]
    cols = 'rgbmc'
    for x in range(val):
        [ax[x].plot(breath[i-600:i+600], cols[x], alpha=.3) for i in marks[x][0]]
    fig.suptitle(tit, fontsize=15)
    fig.tight_layout(rect=(0,0,1,.97))
    return fig

# Load RDC from rat_day_cells.py !!!
pp = PdfPages(pdfName)
file='data_11.AUG.16.h5'
plt.ioff()
for rat in sorted(RDC.keys()):
    for ses in sorted(RDC[rat]):
        dataset = loadData(rat, ses, file=file)
        etsN = ['poke_in', 'odor_on', 'poke_out', 'water_in', '1.8SBPi'] 
        x_time = dataset['data']['x_time']
        breath = dataset['data']['respiration']
        sr = dataset['data']['samp_rate']
        fTit = rat+' '+ses
        fig = plt.figure(fTit, figsize=(20, 7))
        val = len(etsN)
        axs = [fig.add_subplot(1, val, i+1) for i in range(val)]
        cols = 'rgbmc'
        time = int(sr/3)
        for x in range(val):
            if x == val-1:
                events = dataset['events_ts']['poke_in']
                events = events-1.8
            else:
                events = dataset['events_ts'][etsN[x]]
            resps, ana_phase, finh, marks = inhDetection(breath,
                                                         events, x_time,
                                                         sr=sr, ratio=.5)
            [axs[x].plot(r[i-time:i+time], cols[x], alpha=.25)
             for r, i in zip(resps, finh)]
            axs[x].set_title(etsN[x])
            axs[x].axis('off')
        fig.suptitle(fTit, fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(pp, format='pdf')
        print(fTit, ': done')
        plt.close()
pp.close()

def main_InhalationLocked(pdfName='rat_ses-INH_1.8SBPi.pdf'):
    '''
    Creates the PDF of the respiration locked to first inhalations for
    some behavioral events.
    '''
    pp = PdfPages(pdfName)
    file='data_11.AUG.16.h5'
    plt.ioff()
    for rat in sorted(RDC.keys()):
        for ses in sorted(RDC[rat]):
            dataset = loadData(rat, ses, file=file)
            etsN = ['poke_in', 'odor_on', 'poke_out', 'water_in', '1.8SBPi'] 
            x_time = dataset['data']['x_time']
            breath = dataset['data']['respiration']
            sr = dataset['data']['samp_rate']
            fTit = rat+' '+ses
            fig = plt.figure(fTit, figsize=(20, 7))
            val = len(etsN)
            axs = [fig.add_subplot(1, val, i+1) for i in range(val)]
            cols = 'rgbmc'
            time = int(sr/3)
            for x in range(val):
                if x == val-1:
                    events = dataset['events_ts']['poke_in']
                    events = events-1.8
                else:
                    events = dataset['events_ts'][etsN[x]]
                resps, ana_phase, finh, marks = inhDetection(breath,
                                                             events, x_time,
                                                             sr=sr, ratio=.5)
                [axs[x].plot(r[i-time:i+time], cols[x], alpha=.25)
                 for r, i in zip(resps, finh)]
                axs[x].set_title(etsN[x])
                axs[x].axis('off')
            fig.suptitle(fTit, fontsize=15)
            fig.tight_layout(rect=(0, 0, 1, .97))
            fig.savefig(pp, format='pdf')
            print(fTit, ': done')
            plt.close()
    pp.close()
"""

# rat = "f4"
# ses = "05_31_06b"
# neu = RDC[rat][ses][0]
# To create the hdf5 file, beautiful hdf5...
file = h5py.File('../data/data_11.AUG.16.h5', 'a')
for rat in sorted(RDC.keys()):
    rdata = file.create_group(rat)
    for ses in sorted(RDC[rat].keys()):
        sdata = rdata.create_group(ses)
        rd = loadZCB(rat, ses)
        t0 = rd.data.t0
        sr = rd.data.SampFreq
        dt = 1/sr
        L = len(rd.data.breath)
        dur = L/sr
        x_time = np.linspace(t0, dur+t0, num=L)
        freq = 30
        breath = gaussFil(rd.data.breath, sr=sr, freq=freq)
        breath = normalize(breath)
        valTrials = np.array(np.nonzero(~np.isnan(sum((rd.events.OdorPokeIn,
                                                       rd.events.OdorValveOn,
                                                       rd.events.WaterPokeIn,),
                                                      axis=0)))[0])
        trials_TS = rd.events.TrialStart[valTrials]
        pokeIn_TS = trials_TS+rd.events.OdorPokeIn[valTrials]
        odorON_TS = trials_TS+rd.events.OdorValveOn[valTrials]
        odorValveID = rd.events.OdorValveID[valTrials]
        pokeOut_TS = trials_TS+rd.events.OdorPokeOut[valTrials]
        waterIn_TS = trials_TS+rd.events.WaterPokeIn[valTrials]
        waterValveID = rd.events.WaterPokeID[valTrials]
        reward_TS = trials_TS+rd.events.WaterValveOn[valTrials]
        waterOut_TS = trials_TS+rd.events.WaterPokeOut[valTrials]
        odors = np.unique(odorValveID)
        trialXodor = [np.nonzero(odorValveID == odor)[0] for odor in odors]
        # Create data Structure:
        rsdata = {'time_start': t0, 'samp_rate': sr, 'duration': dur,
                  'respiration': breath, "cutoff_gaussF": 30,
                  'odor_id': odorValveID, 'water_id': waterValveID,}
        events_TS = {'trials_start': trials_TS, 'poke_in': pokeIn_TS,
                     'odor_on': odorON_TS, 'poke_out': pokeOut_TS,
                     'water_in': waterIn_TS, 'reward': reward_TS}
        rsd = sdata.create_group('data')
        for key in rsdata.keys():
            rsd.create_dataset(key, data=rsdata[key])
        rse = sdata.create_group('events_timestamps')
        for key in events_TS.keys():
            rse.create_dataset(key, data=events_TS[key])
        print(rat, ses)
file.close()
        
        
        ana_sig, envelope, ins_phase, ins_freq, ana_phase = HLB(breath, sr=sr)
        dT = .5 # sec arround first inhalation for the events
        frac = 6 # fraction of pi to asume first inhalation
        tBPi = 3 # Time before poke to use
        marksPI = markify(pokeIn_TS, x_time, ana_phase, frac=frac, dT=dT)
        marksOO = markify(odorON_TS, x_time, ana_phase, frac=frac, dT=dT)
        marksPO = markify(pokeOut_TS, x_time, ana_phase, frac=frac, dT=dT)
        marksWI = markify(waterIn_TS, x_time, ana_phase, frac=frac, dT=dT)
        # TODO marksRND
        marksBP = markify(pokeIn_TS-tBPi, x_time, ana_phase, frac=frac, dT=dT)
        marks = {'poke_in': marksPI, 'odor_on': marksOO, 'poke_out': marksPO,
                 'water_in': marksWI}
        tL = [marksPI[1], marksOO[1], marksPO[1], marksWI[1],  marksBP[1]]
 """       



"""
TODO organizedly look for neurons that look interesting and save them apart
     Convolve the spike trains with subsampled breathing+EMD, that might look 
     better and more interesting....also, point process into oscilation to
     look for synchronicity.
     Calculate the vector strenght and spread of the locked neurons,
         compare to other inhalations outside the odor poke.
             Using high SNR breaths, to be sure.
"""
     
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
"""
# First plots 
ppOO = PdfPages('rat_ses_cell-t0_OdorOn.pdf')
ppPI = PdfPages('rat_ses_cell-t0_PokeIn.pdf')
for rat in sorted(RDC.keys()):
    for ses in sorted(RDC[rat].keys()):
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
"""

"""
# Creates raster by behavioral responses.
ppA = PdfPages('rat_ses_cell-Pi_Oo_3sPi.pdf')
data = {}
for rat in sorted(RDC.keys()):
    data[rat] = {}
    for ses in sorted(RDC[rat].keys()):
        data[rat][ses] = {}
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
        pokeIn_TS = trials_TS+rd.events.OdorPokeIn[valTrials]
        odorON_TS = trials_TS+rd.events.OdorValveOn[valTrials]
        odorValveID = rd.events.OdorValveID[valTrials]
        pokeOut_TS = trials_TS+rd.events.OdorPokeOut[valTrials]
        waterIn_TS = trials_TS+rd.events.WaterPokeIn[valTrials]
        waterValveID = rd.events.WaterPokeID[valTrials]
        reward_TS = trials_TS+rd.events.WaterValveOn[valTrials]
        waterOut_TS = trials_TS+rd.events.WaterPokeOut[valTrials]
        odors = np.unique(odorValveID)
        trialXodor = [np.nonzero(odorValveID == odor)[0] for odor in odors]
        ana_sig, envelope, ins_phase, ins_freq, ana_phase = HLB(breath, sr=sr)
        dT = .5 # sec arround first inhalation for the events
        frac = 5 # fraction of pi to asume first inhalation
        tBPi = 3 # Time before poke to use
        marksPI = markify(pokeIn_TS, x_time, ana_phase, frac=5, dT=dT)
        marksOO = markify(odorON_TS, x_time, ana_phase, frac=5, dT=dT)
        marksPO = markify(pokeOut_TS, x_time, ana_phase, frac=5, dT=dT)
        marksWI = markify(waterIn_TS, x_time, ana_phase, frac=5, dT=dT)
        
        marksBP = markify(pokeIn_TS-tBPi, x_time, ana_phase, frac=5, dT=dT)
        tL = [marksPI[1], marksOO[1], marksPO[1], marksBP[1]] # marksWI[1],  
        # Create data Structure:
        data[rat][ses]['data'] = {'time_start': t0, 'samp_rate': sr,
                                  'respiration': breath}
        events_TS = {'trials_start': trials_TS, 'poke_in': pokeIn_TS,
                     'odor_on': odorON_TS, 'odor_id': odorValveID,
                     'poke_out': pokeOut_TS, 'water_in': waterIn_TS,
                     'water_id': waterValveID, 'reward': reward_TS}
        marks = {'poke_in': marksPI, 'odor_on': marksOO, 'poke_out': marksPO,
                 'water_in': marksWI}
        data[rat][ses]['events_timestamps'] = events_TS
        data[rat][ses]['events_first_inh'] = marks
        cols = 'rgbcmy'
        for neu in RDC[rat][ses]:
            cell = loadCellTS(rat, ses, neu)
            titF = rat+' '+ses+' '+neu
            print(titF)
            fig = plt.figure(titF, figsize=(15.69, 8.27), dpi=100)
            tits = ['Poke in', 'Odor on', 'Poke Out',  '3SBPi']
            for marks,y in zip(tL, range(len(tL))):
                ts = rastify(trialXodor, cell, marks, x_time)
                for x in range(len(ts)):
                    ax = fig.add_subplot(len(ts), len(tL), 1+(x*len(tL))+y)
                    ylab = 'Odor-'+str(x+1)
                    for n in range(len(ts[x])):
                        ax.plot(ts[x][n], np.zeros_like(ts[x][n])+n,
                                '.'+cols[x], ms=2)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if x+1 == len(ts):
                        ax.set_xticks([-.5, -.25, 0, .25, .5])
                    if x == 0:
                        ax.set_title(tits[y])
                    if y == 0:
                        ax.set_yticks(np.linspace(0,len(ts[x]), num=5, dtype='int'))
                    if y == len(tL)-1:
                        ax.set_ylabel(ylab)
                        ax.yaxis.set_label_position('right')
                    ax.axis((-.504, .504, -.4, (len(ts[x])+.4)))
                    ax.plot(np.zeros(2), [-.4, (len(ts[x])+.4)], 'k', lw=1.5)
            fig.suptitle(titF, fontsize=14)
            fig.savefig(ppA, format='pdf')
            plt.close()
ppA.close()
"""

"""
Create pdf with the inhalations 
pp = PdfPages('rat_ses-INH.pdf')
for rat in sorted(RDC.keys()):
    for ses in sorted(RDC[rat].keys()):
        rdata = data.create_group(ses)
        rd = loadZCB(rat, ses)
        t0 = rd.data.t0
        sr = rd.data.SampFreq
        dt = 1/sr
        L = len(rd.data.breath)
        dur = L/sr
        x_time = np.linspace(t0, dur+t0, num=L)
        
        breath = gaussFil(rd.data.breath, sr=sr, freq=30)
        breath = normalize(breath)
        valTrials = np.array(np.nonzero(~np.isnan(sum((rd.events.OdorPokeIn,
                                                       rd.events.OdorValveOn,
                                                       rd.events.WaterPokeIn,),
                                                      axis=0)))[0])
        trials_TS = rd.events.TrialStart[valTrials]
        pokeIn_TS = trials_TS+rd.events.OdorPokeIn[valTrials]
        odorON_TS = trials_TS+rd.events.OdorValveOn[valTrials]
        odorValveID = rd.events.OdorValveID[valTrials]
        pokeOut_TS = trials_TS+rd.events.OdorPokeOut[valTrials]
        waterIn_TS = trials_TS+rd.events.WaterPokeIn[valTrials]
        waterValveID = rd.events.WaterPokeID[valTrials]
        reward_TS = trials_TS+rd.events.WaterValveOn[valTrials]
        waterOut_TS = trials_TS+rd.events.WaterPokeOut[valTrials]
        odors = np.unique(odorValveID)
        trialXodor = [np.nonzero(odorValveID == odor)[0] for odor in odors]
        ana_sig, envelope, ins_phase, ins_freq, ana_phase = HLB(breath, sr=sr)
        dT = .5 # sec arround first inhalation for the events
        frac = 6 # fraction of pi to asume first inhalation
        tBPi = 3 # Time before poke to use
        marksPI = markify(pokeIn_TS, x_time, ana_phase, frac=frac, dT=dT)
        marksOO = markify(odorON_TS, x_time, ana_phase, frac=frac, dT=dT)
        marksPO = markify(pokeOut_TS, x_time, ana_phase, frac=frac, dT=dT)
        marksWI = markify(waterIn_TS, x_time, ana_phase, frac=frac, dT=dT)
        # TODO marksRND
        marksBP = markify(pokeIn_TS-tBPi, x_time, ana_phase, frac=frac, dT=dT)
        tL = [marksPI[1], marksOO[1], marksPO[1], marksWI[1],  marksBP[1]]
        tit = rat+' '+ses
        val = len(tL)
        fig = figInhwaves(tL, breath, tit)
        fig.savefig(pp, format='pdf')
        print(tit, ': done')
        plt.close()
"""