run explorations_by_peaks.py
run rat_day_cells.py

rats = sorted(RDC.keys())
rat = rats[0]
ses = sorted(RDC[rat].keys())[5]
dataset = loadData(rat, ses)
breath = dataset['data']['respiration']
sr = dataset['data']['samp_rate']
sampT = int(sr*60)
cutoff = dataset['data']['cutoff_gaussF']
poke_in = dataset['events_ts']['poke_in']
odor_on = dataset['events_ts']['odor_on']
poke_out = dataset['events_ts']['poke_out']
x_time = dataset['data']['x_time']
neu = RDC[rat][ses]
ev = 'poke_in'
events = dataset['events_ts'][ev]
low=0; high=30; frac=2;ratio=1.5; nbins=10; order=3
resps, finh, cycles, marks = \
                    inhDetXPeaks(breath, events, x_time, low=low, frac=frac,
                                 sr=sr, ratio=ratio, order=order)

rimf = [pyeemd.emd(resps[i], num_imfs=5) for i in range(len(resps))]
xlabs = ['original', 'IMF_1', 'IMF_2', 'IMF_3', 'IMF_4', 'IMF_5']
ppOO = PdfPages('imfs_libeemd.pdf-PokeIn.pdf')
plt.ioff()
for x in range(len(resps)):
    fig = plt.figure(figsize=(11.69, 8.27))
    tit=rat+'-'+ses+'-Trial-'+str(x)+'-Event-'+ev
    ax = [fig.add_subplot(6, 1, i+1) for i in range(6)]
    ax[0].plot(resps[x])
    [ax[i+1].plot(rimf[x][i]) for i in range(5)]
    [ax[i].set(xlim=(0, 5682), xticks=(), xlabel=xlabs[i], yticks=())
                for i in range(6)]
    fig.suptitle(tit, fontsize=18)
    plt.tight_layout()
    fig.savefig(ppOO, format='pdf')
    plt.close()
ppOO.close()
