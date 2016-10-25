
import numpy as np
import matplotlib.pyplot as plt
from anal_inhale import loadData, gaussFil
from rat_day_cells import RDC

rats = sorted(RDC.keys())
rat = rats[0]
ses = sorted(RDC[rat].keys())

dataset = loadData(rat, ses[0])
resp = dataset['data']['respiration']
sr = dataset['data']['samp_rate']
sampT = int(sr*60)
cutoff = dataset['data']['cutoff_gaussF']
poke_in = dataset['events_ts']['poke_in']
odor_on = dataset['events_ts']['odor_on']
poke_out = dataset['events_ts']['poke_out']

dresp = np.hstack((0, np.diff(resp)))
Udresp = (dresp >= 0).nonzero()[0]
discont = np.hstack((0, (np.diff(Udresp) > 1).nonzero()[0]))
pairs = np.array([discont[i:i+2] for i in range(len(discont)-1)])

# After taking the upper parts, take only the increases larger than
# sr/cutoff_gaussF, this may help...
# Then, use argmin and argmax of the increases as inh, exh begin
# Detected, motherfucker....
# Make this compatible with previous datatypes for god sake...

inLim = round(sr/cutoff).astype('int')


ddresp = gaussFil(np.hstack((0, np.diff(dresp))), sr=sr, freq=cutoff)
ctz = np.isclose(dresp, 0, rtol=abs(np.median(dresp)),
                 atol=np.median(dresp)+np.std(dresp)/8).nonzero()[0]
dctz = (np.diff(ctz) > sr/cutoff*.5).nonzero()[0]
mc = (ctz[dctz] < sampT).nonzero()[0].argmax()
x = np.arange(len(resp))
fig = plt.figure(figsize=(22, 6))
ax = fig.add_subplot(111)
ax.plot(x[:sampT], resp[:sampT], 'b-', lw=2)
ax.plot(ctz[dctz[:mc]], resp[ctz[dctz[:mc]]], 'sr',
        x[:sampT], dresp[:sampT], 'g',
        x[:sampT], ddresp[:sampT], 'm',
        x[:sampT], np.zeros(sampT), 'r')

# b = resp[ctz[dctz]]
# c = (abs(diff(b)) > .7).nonzero()[0]
# ax.plot(ctz[dctz[c[:300]]], resp[ctz[dctz[c[:300]]]], 'sg')

presp = dresp > 0
vresp = dresp < 0
