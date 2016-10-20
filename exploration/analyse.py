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
    elif method == "DF":
        res = signal/sum(signal)
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
                 low=1, high=30, order=3, nmeth='ZS', frac=5, ratio=3):
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
        (ratio 1) = (1 sec); (ratio 2) = (.5 sec)
    
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
    inhs = [None]*len(ev_ndx)
    thresh = -(np.pi/frac)*(frac-1)
    for i in range(len(ev_ndx)):
        inhs[i] = np.array((np.hstack((0, np.diff(ana_phase[i])))\
                            < -np.pi).nonzero()[0])
        if ana_phase[i][ev0] < thresh:
            vfin = (ana_phase[i][:ev0] < thresh).nonzero()[0].max()
            finh[i] = inhs[i][(inhs[i]-vfin <= 0).nonzero()[0].max()]
        else:
            vfin = (ana_phase[i][ev0:] < thresh).nonzero()[0].min()+ev0
            finh[i] = inhs[i][(inhs[i]-vfin >= 0).nonzero()[0].min()]
    finhATinh = np.hstack([(inhs[i] == finh[i]).nonzero()[0]
                           for  i in range(len(finh))])
    cycles = [[[inhs[i][finhATinh[i]-1], finh[i]],
               [finh[i], inhs[i][finhATinh[i]+1]],
               [inhs[i][finhATinh[i]+1], inhs[i][finhATinh[i]+2]]]
              for i in range(len(finh))]
    cycles = np.array(cycles)
    marks = [ev_ndx, pe_ndx]
    return resps, ana_phase, finh, cycles, marks


def periStimCycleTBins(x_time, ana_phase, finh, cycles, marks, nbins=12):
    exh = [[(ana_phase[i][cycles[i, j, 0]:cycles[i, j, 1]]>0).nonzero()[0][0]
            for j in range(3)]
           for i in range(len(finh))]
    exh = np.array(exh)
    num = nbins+1
    cycleB = np.array([cycles[:, 0, 0],
                       cycles[:, 0, 0]+exh[:, 0],
                       cycles[:, 0, 1]])+marks[1][0]
    cycleN = np.array([cycles[:, 1, 0],
                       cycles[:, 1, 0]+exh[:, 1],
                       cycles[:, 1, 1]])+marks[1][0]
    cycleA = np.array([cycles[:, 2, 0],
                       cycles[:, 2, 0]+exh[:, 2],
                       cycles[:, 2, 1]])+marks[1][0]
    # Make the TS for the bins of inhalation and exhalation in the diff cycles
    cycleB_TS = np.array([[np.linspace(x_time[cycleB][:, i][0],
                                       x_time[cycleB][:, i][1], num=num),
                           np.linspace(x_time[cycleB][:, i][1],
                                       x_time[cycleB][:, i][2], num=num)]
                          for i in range(len(finh))])
    cycleN_TS = np.array([[np.linspace(x_time[cycleN][:, i][0],
                                       x_time[cycleN][:, i][1], num=num),
                           np.linspace(x_time[cycleN][:, i][1],
                                       x_time[cycleN][:, i][2], num=num)]
                          for i in range(len(finh))])
    cycleA_TS = np.array([[np.linspace(x_time[cycleA][:, i][0],
                                       x_time[cycleA][:, i][1], num=num),
                           np.linspace(x_time[cycleA][:, i][1],
                                       x_time[cycleA][:, i][2], num=num)]
                          for i in range(len(finh))])
    return cycleB_TS, cycleN_TS, cycleA_TS


def plot_cycles(resps, cycles, finh, ana_phase, axA, axB, axC, alf=.01):
    exh = [[(ana_phase[i][cycles[i, j, 0]:cycles[i, j, 1]]>0).nonzero()[0][0]
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
            ax.plot(np.arange(cycle[0,x]-cycle[1,x],0),
                 resps[x][cycle[0,x]:cycle[1,x]], 'b', alpha=alf)
            ax.plot(np.arange(0,-cycle[1,x]+cycle[2,x]),
                 resps[x][cycle[1,x]:cycle[2,x]], 'g', alpha=alf)
        ax.axis('off')


def PSTH_phase(cycle_TS, cell, nmeth='FS'):
    ntrials, ie, nbins = np.shape(cycle_TS) 
    inh = np.array([[sum((cell > cycle_TS[j][0][i]) &\
                         (cell < cycle_TS[j][0][i+1]))
                     for i in range(nbins-1)]
                    for j in range(ntrials)])
    exh = np.array([[sum((cell > cycle_TS[j][1][i]) &\
                         (cell < cycle_TS[j][1][i+1]))
                     for i in range(nbins-1)]
                    for j in range(ntrials)])
    psth = np.hstack((sum(inh, axis=0),sum(exh, axis=0)))
    return psth

def CherryPicked(file='data_11.AUG.16.h5'):
    date = datetime.date.today().strftime('%Y_%m_%d')
    plt.ioff()
    ppN = PdfPages('Cherry_plots-RastPSTH-low1.5-frac6-'+date+'.pdf')
    ppR = PdfPages('Cherry_plots-SniffWaves-low1.5-frac6-'+date+'.pdf')
    datas = {}
    for rat in sorted(RDC_cherryPick.keys()):
        datas[rat] = {}
        for ses in sorted(RDC_cherryPick[rat]):
            datas[rat][ses] = {}
            dataset = loadData(rat, ses, file=file)
            etsN = ['poke_in', 'odor_on','1.5SBPi']
            x_time = dataset['data']['x_time']
            breath = dataset['data']['respiration']
            sr = dataset['data']['samp_rate']
            odors = dataset['data']['odor_id']
            val = len(etsN)
            trialXodor = [(odors == odid).nonzero()[0]
                          for odid in np.unique(odors)]
            cols = 'rgbmc'
            time = int(sr/3)
            for x in range(val):
                if x == val-1:
                    events = dataset['events_ts']['poke_in']
                    events = events-1.5
                else:
                    events = dataset['events_ts'][etsN[x]]
                resps, ana_phase, finh, cycles, marks = \
                    inhDetection(breath, events, x_time, low=1.5, frac=10,
                                 sr=sr, ratio=.45)
                cycleB_TS, cycleN_TS, cycleA_TS =\
                    periStimCycleTBins(x_time, ana_phase, finh,\
                        cycles, marks, nbins=6)
                titF = rat+' '+ses
                figR = plt.figure(titF, figsize=(25, 19))
                axPSR = figR.add_subplot(val,4, 1+(x*4))
                axEv =  [figR.add_subplot(val,4, i+2)
                            for i in np.arange(4*x,4*x+val)]
                [axPSR.plot(r[i-time:i+time], cols[x], alpha=.1)
                 for r, i in zip(resps, finh)]
                axPSR.set_title(etsN[x])
                axPSR.axis('off') 
                plot_cycles(resps, cycles, finh, ana_phase,
                            axEv[0], axEv[1], axEv[2])
                if x == 0:
                    for ax,tit in zip([axEv[0], axEv[1], axEv[2]],
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
            for neu in RDC_cherryPick[rat][ses]:
                cell = loadCellTS(rat, ses, neu)
                titF = rat+' '+ses+' '+neu
                figN = plt.figure(titF, figsize=(25, 19))
                ax_RAST = [figN.add_subplot(2,3,i+1) for i in range(val)]
                ax_PSTH = [figN.add_subplot(6,3,i+10) for i in range(val)]
                ax_PSTHxO = [figN.add_subplot(3,3,i+7) for i in range(val)]
                for x in range(val):
                    tit = etsN[x]
                    finh = datas[rat][ses][etsN[x]]['inh']['finhs']
                    marks = datas[rat][ses][etsN[x]]['inh']['marks']
                    cycles = [datas[rat][ses][etsN[x]]['cycles'][k]
                              for k in\
                              sorted(datas[rat][ses][etsN[x]]['cycles'].keys())]
                    psth = np.hstack([PSTH_phase(cycle_TS, cell)
                                      for cycle_TS in cycles])
                    psthXodor = [np.hstack(\
                                 [PSTH_phase(cycles_TS[trialXodor[oid]],cell)
                                  for cycles_TS in cycles])
                                 for oid in range(len(unique(odors)))]
                    axa = rastifyXneu(x_time, finh, marks, cell,
                                      odors, ax_RAST[x], tit)
                    axb = plot_psth_All(psth, ax_PSTH[x])
                    axc = plot_psthXodor(psthXodor, odors, ax_PSTHxO[x])
                                      
                figN.suptitle(titF, fontsize=18)
                figN.savefig(ppN, format='pdf')
                plt.close()
    return datas, ppN, ppR

def PDF_RastWaves(file='data_11.AUG.16.h5'):
    date = datetime.date.today().strftime('%Y_%m_%d')
    plt.ioff()
    ppN = PdfPages('plots-RastPSTH-low1-frac5-ratio4-order4-'+date+'.pdf')
    ppR = PdfPages('plots-SniffWaves-low1-frac5-ratio4-order4-'+date+'.pdf')
    datas = {}
    for rat in sorted(RDC.keys()):
        datas[rat] = {}
        for ses in sorted(RDC[rat]):
            datas[rat][ses] = {}
            dataset = loadData(rat, ses, file=file)
            etsN = ['poke_in', 'odor_on', 'poke_out', '1.5SBPi']
            x_time = dataset['data']['x_time']
            breath = dataset['data']['respiration']
            sr = dataset['data']['samp_rate']
            odors = dataset['data']['odor_id']
            val = len(etsN)
            trialXodor = [(odors == odid).nonzero()[0]
                          for odid in np.unique(odors)]
            cols = 'rgbmc'
            time = int(sr/3)
            for x in range(val):
                if x == val-1:
                    events = dataset['events_ts']['poke_in']
                    events = events-1.5
                else:
                    events = dataset['events_ts'][etsN[x]]
                resps, ana_phase, finh, cycles, marks = \
                    inhDetection(breath, events, x_time, low=1, frac=5,
                                 sr=sr, ratio=.4, order=4)
                cycleB_TS, cycleN_TS, cycleA_TS =\
                    periStimCycleTBins(x_time, ana_phase, finh,\
                        cycles, marks, nbins=6)
                titF = rat+' '+ses
                figR = plt.figure(titF, figsize=(25, 19))
                axPSR = figR.add_subplot(4,4, 1+(x*4))
                axEv =  [figR.add_subplot(4,4, i+2)
                            for i in np.arange(4*x,4*x+3)]
                [axPSR.plot(r[i-time:i+time], cols[x], alpha=.01)
                 for r, i in zip(resps, finh)]
                axPSR.set_title(etsN[x])
                axPSR.axis('off') 
                plot_cycles(resps, cycles, finh, ana_phase,
                            axEv[0], axEv[1], axEv[2])
                if x == 0:
                    for ax,tit in zip([axEv[0], axEv[1], axEv[2]],
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
            #figR.tight_layout(rect=[0,0,1,.95])
            figR.suptitle(titF, fontsize=18)
            figR.savefig(ppR, format='pdf')
            for neu in RDC[rat][ses]:
                cell = loadCellTS(rat, ses, neu)
                titF = rat+' '+ses+' '+neu
                figN = plt.figure(titF, figsize=(25, 19))
                ax_RAST = [figN.add_subplot(2,4,i+1) for i in range(val)]
                ax_PSTH = [figN.add_subplot(6,4,i+13) for i in range(val)]
                ax_PSTHxO = [figN.add_subplot(3,4,i+9) for i in range(val)]
                for x in range(val):
                    tit = etsN[x]
                    finh = datas[rat][ses][etsN[x]]['inh']['finhs']
                    marks = datas[rat][ses][etsN[x]]['inh']['marks']
                    cycles = [datas[rat][ses][etsN[x]]['cycles'][k]
                              for k in\
                              sorted(datas[rat][ses][etsN[x]]['cycles'].keys())]
                    psth = np.hstack([PSTH_phase(cycle_TS, cell)
                                      for cycle_TS in cycles])
                    psthXodor = [np.hstack(\
                                 [PSTH_phase(cycles_TS[trialXodor[oid]],cell)
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


def plot_psth_All(psth, ax):
    L = len(psth)
    num = int(L/3)
    x = np.linspace(0,L, num=1000)
    y = np.sin(2*np.pi*x*(1/num)+11)
    y = -.5+(y+2)/2
    psthN = psth/max(psth)
    ax.plot(x, y, 'r--', lw=3)
    ax.plot(x, y, 'r:', lw=3)
    ax.bar(range(L), psthN)
    [ax.plot([num*(1+i),num*(1+i)], [0,1], 'k--') for i in range(3)]
    ax.set_yticks([])
    ax.set_xticks([num/2, num+num/2, num/2+num*2])
    ax.set_xticklabels(['Cycle\nBefore',\
                        'Event', 'Cycle\nAfter'])
    [ax.spines[i].set_visible(False)
     for i in ['top', 'bottom', 'left', 'right']]
    ax.set_title('Inhalation | Exhalation', fontsize=12)
    ax.set_xlim(0,num*3)
    return ax


def plot_psthXodor(psthXodor, odors, ax):
    oID = np.unique(odors)
    L = len(psthXodor[0])
    cols = 'rgbcmy'
    for x, oi in enumerate(oID):
        psth = psthXodor[x]/max(psthXodor[x])
        ax.bar(range(L), psth, bottom=x+.2*x, color=cols[x])
    num = int(L/3)
    x = np.linspace(0,L, num=1000)
    y = np.sin(2*pi*x*(1/num)+11)
    y = -.5+(y+2)/2
    y = y*(len(oID)+1)
    ax.plot(x, y, 'r--', lw=3)
    ax.plot(x, y, 'r:', lw=3)
    [ax.plot([num*(1+i),num*(1+i)], [0,len(oID)+1], 'k--') for i in range(3)]
    ax.set_yticks([.5+x+.2*x for x  in range(len(oID))])
    ax.set_yticklabels(['Odor '+str(int(i)) for i in oID], rotation=90)
    ax.set_xticks([])
    ax.set_xlim(0,num*3)
    return ax


def rastifyXneu(x_time, finh, marks, cell, odors, ax, tit):
    pstime = ([x_time[finh+marks[1][0,:]]-.5,
                        x_time[finh+marks[1][0,:]]+.5])
    cols = '0rg0bcmy'
    rastify = [((cell > pstime[0][i]) & (cell < pstime[1][i])).nonzero()[0]
               for i in range(len(finh))]
    ts = [cell[rastify[i]]-x_time[finh[i]+marks[1][0][i]]
          for i in range(len(rastify))]
    for y in range(len(finh)):
        ax.plot(ts[y], np.zeros_like(ts[y])+y, cols[int(odors[y])]+'.', ms=3)
    ax.axis((-.504, .504, -.4, (len(ts)+.4)))
    ax.plot(np.zeros(2), [-.4, (len(ts)+.4)], 'k', lw=.5)
    ax.set_xticks(np.linspace(-.5,.5, num=5))
    ax.set_yticks(np.linspace(0,len(ts), num=5).astype(int))
    ax.set_title(tit)
    return ax

def rastifyXneu_NI(x_time, finh, marks, cell, odors, ax, tit):
    pstime = ([x_time[marks[0]]-.5,
                        x_time[marks[0]]+.5])
    cols = '0rg0bcmy'
    rastify = [((cell > pstime[0][i]) & (cell < pstime[1][i])).nonzero()[0]
               for i in range(len(finh))]
    ts = [cell[rastify[i]]-x_time[marks[0][i]]
          for i in range(len(rastify))]
    for y in range(len(finh)):
        ax.plot(ts[y], np.zeros_like(ts[y])+y, cols[int(odors[y])]+'.', ms=3)
    ax.axis((-.504, .504, -.4, (len(ts)+.4)))
    ax.plot(np.zeros(2), [-.4, (len(ts)+.4)], 'k', lw=.5)
    ax.set_xticks(np.linspace(-.5,.5, num=5))
    ax.set_yticks(np.linspace(0,len(ts), num=5).astype(int))
    ax.set_title(tit)
    return ax

def main_rast_NINH(pdfName='Rasters-rat_ses_neuXevent_NoInh.pdf'):
    file='data_11.AUG.16.h5'
    plt.ioff()
    pp = PdfPages(pdfName)
    for rat in sorted(RDC.keys()):
        #expe[rat] = {}
        for ses in sorted(RDC[rat]):
            #expe[rat][ses] = {}
            dataset = loadData(rat, ses, file=file)
            etsN = ['poke_in', 'odor_on', 'poke_out', '1.8SBPi']
            x_time = dataset['data']['x_time']
            t0 = x_time[0]
            breath = dataset['data']['respiration']
            sr = dataset['data']['samp_rate']
            odors = dataset['data']['odor_id']
            val = len(etsN)
            datas = {}
            for x in range(val):
                if x == val-1:
                    events = dataset['events_ts']['poke_in']
                    events = events-1.8
                else:
                    events = dataset['events_ts'][etsN[x]]
                resps, ana_phase, finh, inhs, marks = \
                    inhDetection(breath, events, x_time, sr=sr, ratio=.5)
                datas[etsN[x]] = {'finhs': finh, 'marks': marks,
                                  'inhs': inhs}
                print(rat, ses, etsN[x], ': done')
            for x in range(len(RDC[rat][ses])):
                neu = RDC[rat][ses][x]
                cell = loadCellTS(rat, ses, neu)
                titF = rat+' '+ses+' '+neu+'b'
                fig = plt.figure(titF, figsize=(15.69, 8.27))
                for y in range(val):
                    ax = fig.add_subplot(1,4,y+1)
                    tit = etsN[y]
                    finh = datas[tit]['finhs']
                    marks = datas[tit]['marks']
                    axb = rastifyXneu_NI(x_time, finh, marks, cell, odors, ax, tit)
                    if (y > 0):
                        axb.set_yticklabels([])
                fig.suptitle(titF, fontsize=18)
                fig.savefig(pp, format='pdf')
                plt.close()
    pp.close()


def main_rast_INH(pdfName='Rasters-rat_ses_neuXevent_Inh.pdf'):
    file='data_11.AUG.16.h5'
    plt.ioff()
    pp = PdfPages(pdfName)
    for rat in sorted(RDC.keys()):
        #expe[rat] = {}
        for ses in sorted(RDC[rat]):
            #expe[rat][ses] = {}
            dataset = loadData(rat, ses, file=file)
            etsN = ['poke_in', 'odor_on', 'poke_out', '1.8SBPi']
            x_time = dataset['data']['x_time']
            t0 = x_time[0]
            breath = dataset['data']['respiration']
            sr = dataset['data']['samp_rate']
            odors = dataset['data']['odor_id']
            val = len(etsN)
            datas = {}
            for x in range(val):
                if x == val-1:
                    events = dataset['events_ts']['poke_in']
                    events = events-1.8
                else:
                    events = dataset['events_ts'][etsN[x]]
                resps, ana_phase, finh, inhs, marks = \
                    inhDetection(breath, events, x_time, sr=sr, ratio=.5)
                datas[etsN[x]] = {'finhs': finh, 'marks': marks,
                                  'inhs': inhs}
                print(rat, ses, etsN[x], ': done')
            for x in range(len(RDC[rat][ses])):
                neu = RDC[rat][ses][x]
                cell = loadCellTS(rat, ses, neu)
                titF = rat+' '+ses+' '+neu+'b'
                fig = plt.figure(titF, figsize=(15.69, 8.27))
                for y in range(val):
                    ax = fig.add_subplot(1,4,y+1)
                    tit = etsN[y]
                    finh = datas[tit]['finhs']
                    marks = datas[tit]['marks']
                    axb = rastifyXneu(x_time, finh, marks, cell, odors, ax, tit)
                    if (y > 0):
                        axb.set_yticklabels([])
                fig.suptitle(titF, fontsize=18)
                fig.savefig(pp, format='pdf')
                plt.close()
    pp.close()                        