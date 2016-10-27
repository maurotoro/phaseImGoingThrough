# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:08:03 2016


Distributed AS IS, it's your fault now.
If no licence on the containing folder, asume GPLv3+CRAPL
@author: Mauro Toro
@email: mauricio.toro@neuro.fchampalimaud.org
"""
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
import h5py
import scipy.io as sio
import pycircstat as pcirc
import pickle as pk


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


def circDatify(psth):
    bins = np.shape(psth)[-1]
    rad = np.arange(0, 2*np.pi, (2*np.pi)/bins)
    d = rad[1]
    w = np.sum(psth, axis=0)
    return rad, w, d


def circvVals(rad, w, d):
    vStr = pcirc.vector_strength(rad, w=w, d=d/2, ci=None)
    vDir = pcirc.mean(rad, w=w, d=d/2, ci=None)
    astd = pcirc.astd(rad, w=w, d=d/2, ci=None)
    skew = pcirc.skewness(rad, w=w, ci=None)
    kurt = pcirc.kurtosis(rad, w=w, ci=None)
    return vStr, vDir, astd, skew, kurt


def plot_polarPSTH(rad, w, d, col, ax, alpha=.5):
    nw = w/max(w)
    ax.fill(rad, nw, alpha=alpha, color=col)
    return ax


def plot_VSD(vStr, vDir, astd, col, ax):
    ax.stem(np.hstack((0, vDir)), np.hstack((0, vStr)), col+'s-', mfc=col)
    return ax


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
    [ax.plot([num*(1+i), num*(1+i)], [0, 1], 'k--') for i in range(3)]
    ax.set_yticks([])
    ax.set_xticks([num/2, num+num/2, num/2+num*2])
    ax.set_xticklabels(['Cycle\nBefore', 'Event', 'Cycle\nAfter'])
    [ax.spines[i].set_visible(False)
     for i in ['top', 'bottom', 'left', 'right']]
    ax.set_title('Inhalation | Exhalation', fontsize=12)
    ax.set_xlim(0, num*3)
    return ax


def rastifyXneu_NOD(x_time, finh, marksI, cell, ax, tit):
    """
    Raster plots locked to first inhalation after behavioral event
    """
    pstime = ([x_time[finh+marksI[0, :]]-.5, x_time[finh+marksI[0, :]]+.5])
    rastify = [((cell > pstime[0][i]) & (cell < pstime[1][i])).nonzero()[0]
               for i in range(len(finh))]
    ts = [cell[rastify[i]]-x_time[finh[i]+marksI[0][i]]
          for i in range(len(rastify))]
    for y in range(len(finh)):
        ax.plot(ts[y], np.zeros_like(ts[y])+y, 'b.', ms=3)
    ax.axis((-.504, .504, -.4, (len(ts)+.4)))
    ax.plot(np.zeros(2), [-.4, (len(ts)+.4)], 'k', lw=.5)
    ax.set_xticks(np.linspace(-.5, .5, num=5))
    ax.set_yticks(np.linspace(0, len(ts), num=5).astype('int'))
    ax.set_title(tit)
    return ax


def rastifyXneu_NINO(x_time, finh, marksO, cell, ax, tit):
    """
    Raster plots locked to behavioral event
    """
    pstime = ([x_time[marksO]-.5, x_time[marksO]+.5])
    rastify = [((cell > pstime[0][i]) & (cell < pstime[1][i])).nonzero()[0]
               for i in range(len(finh))]
    ts = [cell[rastify[i]]-x_time[marksO[i]]
          for i in range(len(rastify))]
    for y in range(len(finh)):
        ax.plot(ts[y], np.zeros_like(ts[y])+y, 'b.', ms=3)
    ax.axis((-.504, .504, -.4, (len(ts)+.4)))
    ax.plot(np.zeros(2), [-.4, (len(ts)+.4)], 'k', lw=.5)
    ax.set_xticks(np.linspace(-.5, .5, num=5))
    ax.set_yticks(np.linspace(0, len(ts), num=5).astype('int'))
    ax.set_title(tit)
    return ax


def plot_popPSTH(RDC_d, psth, ax, PER=.1, inter='none', indx='none',
                 event='poke_in', mark='current', sortMeth=[1]):
    popP = getPSTH(RDC_d, psth, event, mark, 'psth')[:, 1:]
    if indx == 'none':
        popS = getPSTH(RDC_d, psth, event, mark, 'sumary')
        ndx = np.array(sorted(popS, key=lambda x: [x[i] for i in sortMeth])[:])
        ndx = ndx[:, 0].astype('int')
    else:
        ndx = indx
    pP = np.nan_to_num(np.vstack([i/i.max() for i in popP]))
    y, x = np.shape(popP)
    xtick = np.linspace(0, x, x-1)[:-1]
    xlabs = np.linspace(0, 360, x-1).astype('int')[:-1]
    if PER == 1:
        ax.imshow(pP[ndx], aspect='auto', interpolation=inter)
    else:
        TP = int(np.ceil(y*PER))
        ax.imshow(pP[ndx[-TP:]], aspect='auto', interpolation=inter)
    ax.set_xticks(xtick)
    ax.set_xticklabels(xlabs)
    ax.set_ylabel('Neuron ID')
    ax.set_xlabel('Angle')
    return ax


def popPSTH(RDC_d, minN=50,
            hfile='data_11.AUG.16.h5',
            dfile='PSTH_All-07SEP16-10_Bins.data'):
    """
    Creates a population psth for each cell, organized by animal and ses
    Also, saves some sumary statistics on it, [vStr, vDir, astd, skew, kurt]
    """
    dataD = pk.load(open(dfile, 'rb'))
    psths = {}
    for rat in sorted(dataD.keys()):
        if rat == 'VALS':
            continue
        psths[rat] = {}
        for ses in sorted(dataD[rat].keys()):
            psths[rat][ses] = {}
            piInhs = dataD[rat][ses]['poke_in']['inh']
            ooInhs = dataD[rat][ses]['odor_on']['inh']
            valTrials = np.union1d(piInhs['validCycles'],
                                   ooInhs['validCycles'])
            dataset = loadData(rat, ses)
            psths[rat][ses]['odor_ids'] = dataset['data']['odor_id'][valTrials]
            for neu in RDC_d[rat][ses]:
                psths[rat][ses][neu] = {}
                for event in ['poke_in', 'odor_on']:
                    psths[rat][ses][neu][event] = {}
                    psth = dataD[rat][ses][event]['neurons'][neu][valTrials]
                    psths[rat][ses][neu][event]['psth'] = psth
                    L = int(np.shape(psth)[-1]/3)
                    psths[rat][ses][neu][event]['bc'] = {}
                    for n in range(3):
                        rad, w, d = circDatify(psth[:, n*L:L*(n+1)])
                        if sum(w) <= (minN):
                            vStr, vDir, astd, skew, kurt = np.zeros(5)
                        else:
                            vStr, vDir, astd, skew, kurt = circvVals(rad, w, d)
                        psths[rat][ses][neu][event]['bc'][str(n)] =\
                            {'psth': w,
                             'sumary': [vStr, vDir, astd, skew, kurt]}
                    rad, w, d = circDatify(psth[:, L:L*2])
                    if sum(w) <= (minN*2):
                        vStr, vDir, astd, skew, kurt = np.zeros(5)
                    else:
                        vStr, vDir, astd, skew, kurt = circvVals(rad, w, d)
                    psths[rat][ses][neu][event]['current'] = {'psth': w,
                                                            'sumary': [vStr,
                                                                       vDir,
                                                                       astd,
                                                                       skew,
                                                                       kurt]}
                    rad, w, d = circDatify(np.vstack((psth[:, L:L*2],
                                                      psth[:, L*2:L*3])))
                    if sum(w) <= (minN*2):
                        vStr, vDir, astd, skew, kurt = np.zeros(5)
                    else:
                        vStr, vDir, astd, skew, kurt = circvVals(rad, w, d)
                    psths[rat][ses][neu][event]['after'] = {'psth': w,
                                                            'sumary': [vStr,
                                                                       vDir,
                                                                       astd,
                                                                       skew,
                                                                       kurt]}
                    rad, w, d = circDatify(np.vstack((psth[:, :L],
                                                      psth[:, L:L*2],
                                                      psth[:, L*2:L*3])))
                    if sum(w) <= (minN*3):
                        vStr, vDir, astd, skew, kurt = np.zeros(5)
                    else:
                        vStr, vDir, astd, skew, kurt = circvVals(rad, w, d)
                    psths[rat][ses][neu][event]['all'] = {'psth': w,
                                                          'sumary': [vStr,
                                                                     vDir,
                                                                     astd,
                                                                     skew,
                                                                     kurt]}
    psths['VALS'] = {'angles': rad, 'binDist': d}
    return psths


def getPSTH(RDCD, psthB, event, mark, val):
    """
    event = ['poke_in', 'odor_on']
    mark = ['current', 'after', 'all', 'by_cycle']
    val = ['psth', 'sumary']
    """
    pop = []
    indx = 0
    for rat in sorted(RDCD.keys()):
        for ses in sorted(RDCD[rat].keys()):
            for neu in RDCD[rat][ses]:
                if mark == 'by_cycle':
                    # print('not implemented yet...\n8O()')
                    # pass
                    data = np.hstack(
                                     [psthB[rat][ses][neu][event]['bc']\
                                      [str(n)][val] for n in range(3)])
                else:
                    data = psthB[rat][ses][neu][event][mark][val]
                newNeu = np.hstack((int(indx), data))
                pop.append(newNeu)
                indx += 1
    pop = np.array(pop)
    L = np.shape(pop)[-1]
    popN = pop[np.isnan(pop).nonzero()[0], 0].astype('int')
    pop[popN, 1:] = np.zeros(L-1)
    return pop


def psth_by_odor(RDC_d, psth):
    psthXodor = {}
    for rat in sorted(RDC_d.keys()):
        if rat == 'VALS':
            continue
        psthXodor[rat] = {}
        for ses in sorted(RDC_d[rat].keys()):
            odor_ids = psth[rat][ses]['odor_ids']
            psthXodor[rat][ses] = {}
            for neu in RDC_d[rat][ses]:
                psthXodor[rat][ses][neu] = {}
                for od in np.unique(odor_ids):
                    indx = (odor_ids == od).nonzero()[0]
                    psthXodor[rat][ses][neu][str(int(od))] =\
                        psth[rat][ses][neu]['odor_on']['psth'][indx, 10:]

    return psthXodor

def valTrialsPDF(dataD, RDC_d, hfile='data_11.AUG.16.h5'):
    date = dt.date.today().strftime('%Y_%m_%d')
    pPi = PdfPages(date+'-ValTrials-10_Bins-PokeIn.pdf')
    pOo = PdfPages(date+'-ValTrials-10_Bins-OdorOn.pdf')
    plt.ioff()
    for rat in sorted(dataD.keys()):
        if rat == 'VALS':
            continue
        print(rat)
        for ses in sorted(dataD[rat].keys()):
            dataset = loadData(rat, ses, file=hfile)
            x_time = dataset['data']['x_time']
            piInhs = dataD[rat][ses]['poke_in']['inh']
            ooInhs = dataD[rat][ses]['odor_on']['inh']
            valTrials = np.union1d(piInhs['validCycles'],
                                   ooInhs['validCycles'])
            print(ses)
            for event, pp in zip(['poke_in', 'odor_on'], [pPi, pOo]):
                finh = dataD[rat][ses][event]['inh']['finhs'][valTrials]
                marksO = dataD[rat][ses][event]['inh']['marks'][0][valTrials]
                marksI = dataD[rat][ses][event]['inh']['marks'][1][:, valTrials]
                for neu in RDC_d[rat][ses]:
                    cell = loadCellTS(rat, ses, neu)
                    psth = dataD[rat][ses][event]['neurons'][neu][valTrials]
                    L = int(np.shape(psth)[-1]/3)
                    colN = ['VStr', 'VDir', 'ASTD']
                    rowN = ['Before', 'During', 'After']
                    vals = []
                    fig = plt.figure(figsize=(15.69, 8.27))
                    axa = fig.add_subplot(221)
                    axb = fig.add_subplot(223)
                    axc = fig.add_subplot(222)
                    axd = fig.add_subplot(224, projection='polar')
                    titNO = 'Locked to Behavior'
                    titNI = 'Locked to Inhalation'
                    axa = rastifyXneu_NINO(x_time, finh, marksO, cell,
                                           axa, titNO)
                    axb = rastifyXneu_NOD(x_time, finh, marksI, cell,
                                          axb, titNI)
                    axc = plot_psth_All(sum(psth, axis=0), axc)
                    for n, col in enumerate('rgb'):
                        rad, w, d = circDatify(psth[:, n*L:L*(n+1)])
                        vStr, vDir, astd = circvVals(rad, w, d)
                        vals.append([round(vStr, 2),
                                     round(np.rad2deg(vDir), 2),
                                     round(astd, 2)])
                        axd = plot_polarPSTH(rad, w, d, col, axd, alpha=.25)
                        axd = plot_VSD(vStr, vDir, astd, col, axd)
                    axd.table(rowLabels=rowN, colLabels=colN, cellText=vals,
                              colLoc='left', rowColours='rgb', alpha=.25,
                              loc='bottom right', colWidths=[.21, .21, .21])
                    fig.suptitle(rat+' '+ses+' '+neu)
                    fig.savefig(pp, format='pdf')
                    plt.close()
                    print(rat+': '+ses+' '+event+' '+neu)
    pPi.close()
    pOo.close()

"""
ppath = '/Users/soyunkope/Documents/scriptkidd/git/phaseImGoingThrough/'
fpath = 'data/explorations/PSTH_All-07SEP16-10_Bins.data'
dfile = ppath+fpath
psth = popPSTH(RDC, minN=10, dfile=dfile)

events = ['poke_in', 'odor_on']
marks = ['current', 'after', 'all', 'by_cycle']

sortMeth=[6]
mark = marks[3]
ta = 'Population PSTH All Cycles around Event\n'
tb =  (Sorted by vector strength during odor on)'
tit = ta+tb
popS = getPSTH(RDC, psth, events[1], mark, 'sumary')
ndx = np.array(sorted(popS, key=lambda x: [x[i] for i in sortMeth])[:])
ndx = ndx[:, 0].astype('int')

fig = plt.figure(figsize=(22,12))
ax = [fig.add_subplot(2,2,i+1) for i in range(4)]

ax[0] = plot_popPSTH(RDC, psth, ax[0], event=events[0], mark=mark, PER=1,
                     sortMeth=sortMeth, inter='kaiser', indx=ndx)
ax[1] = plot_popPSTH(RDC, psth, ax[1], event=events[0], mark=mark, PER=.1,
                       sortMeth=sortMeth, inter='kaiser', indx=ndx)
ax[2] = plot_popPSTH(RDC, psth, ax[2], event=events[1], mark=mark, PER=1,
                    sortMeth=sortMeth, inter='kaiser', indx=ndx)
ax[3] = plot_popPSTH(RDC, psth, ax[3], event=events[1], mark=mark, PER=.1,
                      sortMeth=sortMeth, inter='kaiser', indx=ndx)

fig.suptitle(tit, fontsize=15)
tot = fig.text(.27, .91, 'ALL Neurons', fontsize=12)
sub = fig.text(.7, .91, 'Upper 10% Neurons', fontsize=12)
tpi = fig.text(.085, .73, 'Poke In', rotation=90, fontsize=15)
too = fig.text(.085, .3, 'Odor On', rotation=90, fontsize=15)

"""
