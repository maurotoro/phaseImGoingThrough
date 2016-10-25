# -*- coding: utf-8 -*-
"""
Created on Tue 25 Oct 15:24:03 2016

Looking for ways of only keeping the good cycles or quantify the quality
of the segmentation

Distributed AS IS, it's your fault now.
If no licence on the containing folder, asume GPLv3+CRAPL

@author: Mauro Toro
@email: mauricio.toro@neuro.fchampalimaud.org
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from peakdetect import peakdetect
import datetime as dt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import scipy.io as sio
from scipy.signal import hilbert, butter, filtfilt, gaussian
from scipy.stats import linregress


def DBPN(signal, low=0.1, high=30, order=3, sr=1893.9393939393942, norm='ZS'):
    """
    Detrend, signal - linearRegression
    BandPass, filtfilt(butterworth[a,b, hig-low, order])
    Normalize: either z-score o feature-scaling

    Parameters
    ----------

    signal : 1d array
        Signal to work with

    low : float
        Lower bound for the bandpass filter

    high : float
        Higher bound for the bandpass filter

    order : integer
        Order for the Butterworth filter

    sr : float
        Sampling rate of the signal

    norm: string
        Normalization method to use, either 'ZS', 'FS' or 'DF', for
        z-score, feature-scaling or density-function respectively.
        More ingo on normalization()

    Returns
    -------

    res : 1d array
        Detrended, Bandpassed and normalized signal
    """
    y = signal
    x = np.arange(0, len(y))
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

    cell : array
        time stamps of action potentials of the given neuron
    """
    fpa = "/Users/soyunkope/Documents/INDP2015/2016_S02/R02_ZachClau/"
    fpb = "phaseImGoingThrough/data/"
    fname = fpa+fpb+rat+'/'+ses+'/'+neu
    nn = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    cell = nn['TS']/10000
    return cell


def normalize(signal, method="ZS"):
    """
    Normalize a signal by different methods

    Parameters
    ----------

    signal : ndarray
        Signal to normalize

    method : {'ZS', 'FS', 'DF'}
        Type of normalization to use, z-score, feature-scaling,
        or density-function.

    Returns
    -------

    res : ndarray
        Normalized signal
    """
    signal = np.asarray(signal)
    res = []
    if method == "ZS":
        res = (signal-np.mean(signal))/np.std(signal)
    elif method == "FS":
        mn = np.min(signal)
        mx = np.max(signal)
        res = (signal)/(mx-mn)
    elif method == "DF":
        res = signal/sum(signal)
    return res


def gaussFil(signal, sr=1893.9393939393942, freq=50):
    """
    Implements a lowpass Gaussian Filter over the signal with
    cutoff frequency freq. Works...
    Creates a gaussian window of the size of the cutoff frequency
    and convolves the signal to it.

    Parameters
    ----------

    signal : 1d array
        Signal to filter

    sr : float
        Sampling rate of the siganl

    freq : float
        Frequency cutoff for the filter

    Returns
    -------

    res : 1d array
        Filtered signal
    """
    M = sr/freq
    std = M/2
    ker = gaussian(M, std)
    res = np.convolve(signal, ker, mode='same')
    return res


def loadData(rat, ses, file='data_11.AUG.16.h5'):
    """
    Load data saved in HDF5 files from analysis.

    Parameters
    ----------

    rat : ratName, string
        Codename of the rat to load, in this case: {'f4', 'f5', 'p9'}

    ses : sesionDate, string
        Date of the session, look at rat_day_session for an idea

    file : filename, string
        HDF5 file to read from

    Returns
    -------

    dataset : dict
        Dictionary with ton of useful info about the task
    """
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


def inhDetXPeaks(breath, events, x_time, sr=1893.9393939393942,
                 low=1, high=30, order=3, nmeth='ZS', frac=2, ratio=.75):
    """
    Detects first inhalation after timestamps. Gives the detrended
    bandpassed and normalized breathing sec/ratio arround the event,
    looks for local minimas and maximas to detect inhalations and exhalations
    If the odor was presented during the begining of the inhalation,
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
        how much of the inhalation to consider 1/frac

    ratio : float
        How much indices around the event to analyze, in secs

    Returns:
    -------

    resps : ndarray
        breathing activity detrended, bandpassed and normalized,
        one row for each trial

    ana_phase : ndarray
        Phase evolution of the analytic signal, calculated as the arctan2 of
        the imaginary and real parts of the analytic signal

    finh : ndarray
        Index of the first inhalation after some behavioral event

    cycles : ndarray x 3
        Previous, actual and posterior cycles with respect to the first
        inhalation after behavioral event, each cycle is inhalation,
        exhalation and next inhalation begin.

    marks : ndarray
        The behavioral event indices and the start and stop indices
        of the respirations.
    """
    ev0 = round(sr*ratio)
    t0 = x_time[0]
    ev_ndx = np.array(((events-t0)*sr).astype(int))
    pe_ndx = np.array([ev_ndx-ev0, ev_ndx+ev0])
    sniff = np.array([breath[i:j] for i, j in zip(pe_ndx[0], pe_ndx[1])])
    resps = np.array([DBPN(sn, low=low, high=high, order=order, sr=sr,
                           norm=nmeth) for sn in sniff])
    finh = np.zeros_like(ev_ndx)
    cycles = np.zeros((len(ev_ndx), 3, 3))
    for i in range(len(ev_ndx)):
        mxr, mnr = peakdetect(resps[i], lookahead=round(sr/high))
        mxi = mxr[:, 0]
        mni = mnr[:, 0]
        if mxi.size != mni.size:
            if mxi.size > mni.size:
                mxi = mxi[:-1]
            elif mni.size > mxi.size:
                mni = mni[:-1]
        thresh = abs((mxi-mni).mean()/frac)
        dist_evin = mni-ev0
        virt_finh = (dist_evin < 0).nonzero()[0].max()
        if abs(dist_evin[virt_finh]) < thresh:
            fip = virt_finh
        else:
            fip = virt_finh+1
        if mxi[fip] > mni[fip]:
            cycles[i] = np.array([[mni[fip-1], mxi[fip-1], mni[fip]],
                                  [mni[fip], mxi[fip], mni[fip+1]],
                                  [mni[fip+1], mxi[fip+1], mni[fip+2]]])
            finh[i] = mni[fip]
        else:
            cycles[i] = np.array([[mni[fip-1], mxi[fip], mni[fip]],
                                  [mni[fip], mxi[fip+1], mni[fip+1]],
                                  [mni[fip+1], mxi[fip+2], mni[fip+2]]])
            finh[i] = mni[fip]
    marks = [ev_ndx, pe_ndx]
    return resps, finh, cycles, marks


def psCycleTBins(x_time, cycles, marks, nbins=10):
    cycles_in = np.array([(cycles[i]+marks[1][0, i]).astype('int')
                         for i in range(len(marks[0]))])
    prevC, currC, nextC = [cycles_in[:, i, :] for i in range(3)]
    cycleB_TS, cycleN_TS, cycleA_TS = [[np.array(
        [np.linspace(x_time[cycle[i, 0]], x_time[cycle[i, 1]], num=nbins+1),
         np.linspace(x_time[cycle[i, 1]], x_time[cycle[i, 2]], num=nbins+1)])
                 for i in range(len(marks[0]))]
        for cycle in [prevC, currC, nextC]]
    return cycleB_TS, cycleN_TS, cycleA_TS



def PSTH_phase(cycle_TS, cell, ret='xTrial'):
    ntrials, ie, nbins = np.shape(cycle_TS)
    inh = np.array([[sum((cell > cycle_TS[j][0][i]) &\
                         (cell < cycle_TS[j][0][i+1]))
                     for i in range(nbins-1)]
                    for j in range(ntrials)])
    exh = np.array([[sum((cell > cycle_TS[j][1][i]) &\
                         (cell < cycle_TS[j][1][i+1]))
                     for i in range(nbins-1)]
                    for j in range(ntrials)])
    if ret == 'xSes':
        psth = np.hstack((sum(inh, axis=0), sum(exh, axis=0)))
    elif ret == 'xTrial':
        psth = np.hstack((inh, exh))
    return psth


def circDatify(psth):
    bins = np.shape(psth)[-1]
    rad = np.arange(0, 2*np.pi, (2*np.pi)/bins)
    d = rad[1]
    w = sum(psth, axis=0)
    spkRad = np.hstack([np.ones(sum(psth[:, i]))*d*i
                        for i in range(bins)])
    return spkRad, rad, w, d


def plot_cycles(resps, cycles, finh, ana_phase, axA, axB, axC):
    exh = [[(ana_phase[i][cycles[i, j, 0]:cycles[i, j, 1]] > 0).nonzero()[0][0]
            for j in range(3)]
           for i in range(len(finh))]
    exh = np.array(exh)
    cycleB = np.array([cycles[:, 0, 0],
                       cycles[:, 0, 0]+exh[:, 0],
                       cycles[:, 0, 1]])
    cycleN = np.array([cycles[:, 1, 0],
                       cycles[:, 1, 0]+exh[:, 1],
                       cycles[:, 1, 1]])
    cycleA = np.array([cycles[:, 2, 0],
                       cycles[:, 2, 0]+exh[:, 2],
                       cycles[:, 2, 1]])
    for ax, cycle in zip([axA, axB, axC], [cycleB, cycleN, cycleA]):
        for x in range(len(resps)):
            ax.plot(np.arange(cycle[0, x]-cycle[1, x], 0),
                    resps[x][cycle[0, x]:cycle[1, x]], 'b')
            ax.plot(np.arange(0, -cycle[1, x]+cycle[2, x]),
                    resps[x][cycle[1, x]:cycle[2, x]], 'g')
        ax.axis('off')


def plot_psth_All(psth, ax):
    L = len(psth)
    num = int(L/3)
    x = np.linspace(0, L, num=1000)
    y = np.sin(2*np.pi*x*(1/num)+3*(np.pi/2))
    y = -.5+(y+2)/2
    psthN = psth/max(psth)
    ax.plot(x, y, 'r--', lw=3)
    ax.plot(x, y, 'r:', lw=3)
    ax.bar(range(L), psthN)
    [ax.plot([num*(1+i), num*(1+i)], [0, 1], 'k--')
     for i in range(3)]
    ax.set_yticks([])
    ax.set_xticks([num/2, num+num/2, num/2+num*2])
    ax.set_xticklabels(['Cycle\nBefore',
                        'Event', 'Cycle\nAfter'])
    [ax.spines[i].set_visible(False)
     for i in ['top', 'bottom', 'left', 'right']]
    ax.set_title('Inhalation | Exhalation', fontsize=12)
    ax.set_xlim(0, num*3)
    return ax


def plot_psthXodor(psthXodor, odors, ax):
    oID = np.unique(odors)
    L = len(psthXodor[0])
    cols = 'rgbcmy'
    for x, oi in enumerate(oID):
        psth = psthXodor[x]/max(psthXodor[x])
        ax.bar(range(L), psth, bottom=x+.2*x, color=cols[x])
    num = int(L/3)
    x = np.linspace(0, L, num=1000)
    y = np.sin(2*np.pi*x*(1/num)+3*(np. pi/2))
    y = -.5+(y+2)/2
    y = y*(len(oID)+1)
    ax.plot(x, y, 'r--', lw=3)
    ax.plot(x, y, 'r:', lw=3)
    [ax.plot([num*(1+i), num*(1+i)], [0, len(oID)+1], 'k--') for i in range(3)]
    ax.set_yticks([.5+x+.2*x for x in range(len(oID))])
    ax.set_yticklabels(['Odor '+str(int(i)) for i in oID], rotation=90)
    ax.set_xticks([])
    ax.set_xlim(0, num*3)
    return ax


def rastifyXneu(x_time, finh, marks, cell, odors, ax, tit):
    pstime = ([x_time[finh+marks[1][0, :]]-.5,
               x_time[finh+marks[1][0, :]]+.5])
    cols = 'rgbcmy'
    oid = np.unique(odors)
    rastify = [((cell > pstime[0][i]) & (cell < pstime[1][i])).nonzero()[0]
               for i in range(len(finh))]
    ts = [cell[rastify[i]]-x_time[finh[i]+marks[1][0][i]]
          for i in range(len(rastify))]
    for y in range(len(finh)):
        ax.plot(ts[y], np.zeros_like(ts[y])+y,
                cols[(odors[0] == oid).nonzero()[0]]+'.', ms=3)
    ax.axis((-.504, .504, -.4, (len(ts)+.4)))
    ax.plot(np.zeros(2), [-.4, (len(ts)+.4)], 'k', lw=.5)
    ax.set_xticks(np.linspace(-.5, .5, num=5))
    ax.set_yticks(np.linspace(0, len(ts), num=5).astype(int))
    ax.set_title(tit)
    return ax


def rastifyXneuNOD(x_time, finh, marks, cell, ax, tit):
    pstime = ([x_time[finh+marks[0, :]]-.5,
               x_time[finh+marks[0, :]]+.5])
    rastify = [((cell > pstime[0][i]) & (cell < pstime[1][i])).nonzero()[0]
               for i in range(len(finh))]
    ts = [cell[rastify[i]]-x_time[finh[i]+marks[0][i]]
          for i in range(len(rastify))]
    for y in range(len(finh)):
        ax.plot(ts[y], np.zeros_like(ts[y])+y, 'b.', ms=3)
    ax.axis((-.504, .504, -.4, (len(ts)+.4)))
    ax.plot(np.zeros(2), [-.4, (len(ts)+.4)], 'k', lw=.5)
    ax.set_xticks(np.linspace(-.5, .5, num=5))
    ax.set_yticks(np.linspace(0, len(ts), num=5).astype(int))
    ax.set_title(tit)
    return ax


def rastifyXneu_NINO(x_time, finh, marks, cell, ax, tit):
    pstime = ([x_time[marks]-.5, x_time[marks]+.5])
    rastify = [((cell > pstime[0][i]) & (cell < pstime[1][i])).nonzero()[0]
               for i in range(len(finh))]
    ts = [cell[rastify[i]]-x_time[marks[i]]
          for i in range(len(rastify))]
    for y in range(len(finh)):
        ax.plot(ts[y], np.zeros_like(ts[y])+y, 'b.', ms=3)
    ax.axis((-.504, .504, -.4, (len(ts)+.4)))
    ax.plot(np.zeros(2), [-.4, (len(ts)+.4)], 'k', lw=.5)
    ax.set_xticks(np.linspace(-.5, .5, num=5))
    ax.set_yticks(np.linspace(0, len(ts), num=5).astype(int))
    ax.set_title(tit)
    return ax


"""
TODO: Make inhDet to only make one hilbert over breath to not fuck the filter
      Compare resulting order with nowish....should be, at least, less
      complicated. Also, meassure the fucking amount of noise in the cycles

"""


def psthDatify(dict, RDC, file='data_11.AUG.16.h5', low=1.5,
               high=30, frac=2, etsN=('poke_in', 'odor_on'),
               ratio=.45, nbins=6, order=3):
    """
    Kind of a main(), creates data dictionaries that could be usefull
    eventually...
    """
    date = dt.date.today().strftime('%Y_%m_%d')
    data = {}
    data['VALS'] = {'file': file, 'low': low, 'high': high, 'frac': frac,
                    'ratio': ratio, 'nbins': nbins,
                    'etsN': etsN, 'date': date, 'order': order}
    for rat in sorted(RDC.keys()):
        if rat == 'VALS':
            continue
        data[rat] = {}
        for ses in sorted(RDC[rat].keys()):
            data[rat][ses] = {}
            dataset = loadData(rat, ses, file=file)
            x_time = dataset['data']['x_time']
            breath = dataset['data']['respiration']
            sr = dataset['data']['samp_rate']
            odors = dataset['data']['odor_id']
            val = len(etsN)
            for x in range(val):
                events = dataset['events_ts'][etsN[x]]
                resps, finh, cycles, marks = \
                    inhDetXPeaks(breath, events, x_time, low=low, frac=frac,
                                 sr=sr, ratio=ratio, order=order)
                cycleB_TS, cycleN_TS, cycleA_TS =\
                    psCycleTBins(x_time, cycles, marks, nbins=nbins)
                data[rat][ses][etsN[x]] = {}
                data[rat][ses][etsN[x]]['inh'] = {'finhs': finh,
                                                  'marks': marks,
                                                  'cycles': cycles}
                data[rat][ses][etsN[x]]['cycles'] = {'A_Before': cycleB_TS,
                                                     'B_Actual': cycleN_TS,
                                                     'C_After': cycleA_TS}
                print(rat, ses, etsN[x], ': done')
                data[rat][ses][etsN[x]]['neurons'] = {}
            for neu in RDC[rat][ses]:
                cell = loadCellTS(rat, ses, neu)
                for x in range(val):
                    finh = data[rat][ses][etsN[x]]['inh']['finhs']
                    marks = data[rat][ses][etsN[x]]['inh']['marks']
                    cycles = [data[rat][ses][etsN[x]]['cycles'][k]
                              for k in
                              sorted(data[rat][ses][etsN[x]]['cycles'].keys())]
                    psth = np.hstack([PSTH_phase(cycle_TS, cell, ret='xTrial')
                                      for cycle_TS in cycles])
                    data[rat][ses][etsN[x]]['neurons'].update({neu: psth})
    return data


def cherry_plotsT(RDC_d, file='data_11.AUG.16.h5', low=1, high=30, frac=6,
                  ratio=.25, nbins=6, order=3,
                  etsN=['poke_in', 'odor_on', 'poke_out', '1.5'],):
    date = dt.date.today().strftime('%Y_%m_%d')
    plt.ioff()
    ppN = PdfPages('Cherry_plots-RastPSTH-low1.5-frac6_'+date+'.pdf')
    ppR = PdfPages('Cherry_plots-SniffWaves-low1.5-frac6_'+date+'.pdf')
    datas = {}
    SBPi = float(etsN[-1])
    for rat in sorted(RDC_d.keys()):
        if rat == 'VALS':
            continue
        datas[rat] = {}
        for ses in sorted(RDC_d[rat]):
            datas[rat][ses] = {}
            dataset = loadData(rat, ses, file=file)
            x_time = dataset['data']['x_time']
            breath = dataset['data']['respiration']
            sr = dataset['data']['samp_rate']
            odors = dataset['data']['odor_id']
            val = len(etsN)
            trialXodor = [(odors == odid).nonzero()[0]
                          for odid in np.unique(odors)]
            cols = 'rgbmcy'
            time = int(sr/3)
            for x in range(val):
                if x == val-1:
                    events = dataset['events_ts']['poke_in']
                    events = events-SBPi
                else:
                    events = dataset['events_ts'][etsN[x]]
                resps, finh, cycles, marks = \
                    inhDetXPeaks(breath, events, x_time, low=low, frac=frac,
                                 sr=sr, ratio=ratio, order=order)
                cycleB_TS, cycleN_TS, cycleA_TS =\
                    psCycleTBins(x_time, cycles, marks, nbins=nbins)
                titF = rat+' '+ses
                figR = plt.figure(titF, figsize=(25, 19))
                axPSR = figR.add_subplot(val, 4, 1+(x*4))
                axEv = [figR.add_subplot(val, 4, i+2)
                        for i in np.arange(4*x, (4*x+val)-1)]
                [axPSR.plot(r[i-time:i+time], cols[x], alpha=.25)
                 for r, i in zip(resps, finh)]
                axPSR.set_title(etsN[x])
                axPSR.axis('off')
                plot_cycles(resps, cycles, finh, ana_phase,
                            axEv[0], axEv[1], axEv[2])
                if x == 0:
                    for ax, tit in zip([axEv[0], axEv[1], axEv[2]],
                                       ['Before', 'Event', 'After']):
                        ax.set_title(tit)
                datas[rat][ses][etsN[x]] = {}
                datas[rat][ses][etsN[x]]['inh'] = {'finhs': finh,
                                                   'marks': marks,
                                                   'cycles': cycles}
                datas[rat][ses][etsN[x]]['cycles'] = {'A_Before': cycleB_TS,
                                                      'B_Actual': cycleN_TS,
                                                      'C_After': cycleA_TS}
                print(rat, ses, etsN[x], ': done')
            figR.suptitle(titF, fontsize=18)
            figR.savefig(ppR, format='pdf')
            for neu in RDC_d[rat][ses]:
                datas[rat][ses][neu] = {}
                cell = loadCellTS(rat, ses, neu)
                titF = rat+' '+ses+' '+neu
                figN = plt.figure(titF, figsize=(25, 19))
                ax_RAST = [figN.add_subplot(2, val, i+1) for i in range(val)]
                ax_PSTH = [figN.add_subplot(6, val, i+12) for i in range(val)]
                ax_PSTHxO = [figN.add_subplot(3, val, i+9) for i in range(val)]
                for x in range(val):
                    tit = etsN[x]
                    finh = datas[rat][ses][etsN[x]]['inh']['finhs']
                    marks = datas[rat][ses][etsN[x]]['inh']['marks']
                    cycles = [datas[rat][ses][etsN[x]]['cycles'][k]
                              for k in
                              sorted(datas[rat][ses][etsN[x]]['cycles'].keys())]
                    psth = np.hstack([PSTH_phase(cycle_TS, cell, ret='xSes')
                                      for cycle_TS in cycles])
                    psthB = np.hstack([PSTH_phase(cycle_TS, cell, ret='xTrial')
                                      for cycle_TS in cycles])
                    datas[rat][ses][neu][etsN[x]] = psthB
                    psthXodor = [np.hstack(
                                 [PSTH_phase(cycles_TS[trialXodor[oid]], cell,
                                             ret='xSes')
                                  for cycles_TS in cycles])
                                 for oid in range(len(np.unique(odors)))]
                    axa = rastifyXneu(x_time, finh, marks, cell,
                                      odors, ax_RAST[x], tit)
                    axb = plot_psth_All(psth, ax_PSTH[x])
                    axc = plot_psthXodor(psthXodor, odors, ax_PSTHxO[x])

                figN.suptitle(titF, fontsize=18)
                figN.savefig(ppN, format='pdf')
                plt.close()
    return datas, ppN, ppR


def valTrialsPDF(dataD, RDC_d, file='data_11.AUG.16.h5'):
    date = dt.date.today().strftime('%Y_%m_%d')
    pPi = date+'-ValTrials-PokeIn.pdf'
    pOo = date+'-ValTrials-OdorOn.pdf'
    for rat in sorted(data.keys()):
        if rat == 'VALS':
            continue
        for ses in sorted(dataD[rat].keys()):
            dataset = loadData(rat, ses, file=file)
            x_time = dataset['data']['x_time']
            piInhs = dataD[rat][ses]['poke_in']['inh']
            ooInhs = dataD[rat][ses]['odor_on']['inh']
            valTrials = np.union1d(piInhs['validCycles'],
                                   ooInhs['validCycles'])
