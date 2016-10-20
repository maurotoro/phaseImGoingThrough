
import numpy as np
import matplotlib.pyplot as plt
from Explorations/rat_day_cells import RDC
#

rats = sorted(RDC.keys())
rat = rats[0]
dataset = loadData(rat, ses[4])
ses = sorted(RDC[rat].keys())
dataset = loadData(rat, ses[4])
resp = dataset['data']['respiration']
sr = dataset['data']['samp_rate']
sampT = int(sr*60)
cutoff = 30
dresp = np.hstack((0, gaussFil(np.diff(resp), sr=sr, freq=cutoff)))
ddresp = np.hstack((0, gaussFil(np.diff(dresp), sr=sr, freq=cutoff)))
ctz = np.isclose(dresp, 0, rtol=median(dresp),
                 atol=median(dresp)+std(dresp)/8).nonzero()[0]
dctz = (np.diff(ctz) > sr/cutoff).nonzero()[0]
    
mc = (ctz[dctz] < sampT).nonzero()[0].argmax()
b = resp[ctz[dctz]]
c = (abs(diff(b)) >.7).nonzero()[0]
ax.plot(x[:sampT], resp[:sampT], 'b-', lw=2)
ax.plot(x[:sampT], dresp[:sampT], 'g',
        x[:sampT], ddresp[:sampT], 'm',
        x[:sampT], zeros(sampT), 'r',
        ctz[dctz[:mc]],
            resp[ctz[dctz[:mc]]], 'sr')
ax.plot(ctz[dctz[c[:300]]], resp[ctz[dctz[c[:300]]]],'sg')





presp = dresp > 0
vresp = dresp < 0
