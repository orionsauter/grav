import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read table of injected signals
def getInjections():
    f = open('injection_list.txt')
    inject = dict()
    skip = 1
    for line in f:
        # Skip header
        if (skip):
            skip = 0
            continue
        vals = line.split()
        inject[vals[0]] = map(float,vals[3:])
    f.close()
    return inject

# Read DAG file to find frequencies
def getOutputSegs(dag):
    f = open(dag)
    segments = dict()
    patt = re.compile('PID="(A\d+_\d+)".+F0="([0-9\.]+)"')
    for line in f:
        m = patt.search(line)
        if (not m):
            continue
        segments[float(m.group(2))] = m.group(1)
    f.close()
    return segments

# Traverse output directory
def getData(direc):
    data = list([])
    patt = re.compile('\d+-\d+\.\d+')
    for root, dirs, files in os.walk(direc):
        m = patt.search(root)
        if (not m):
            continue
        for name in files:
            if (name != 'powerflux.log'):
                continue
            f = open(os.path.join(root,name))
            for line in f:
                line.rstrip()
                data = data + list([line.split(' ')])
            f.close()
    return data

if __name__ == "__main__":
    dag = sys.argv[1]
    segments = getOutputSegs(dag)
    inject = getInjections()
    patt = re.compile('_all')
    dets = np.zeros_like(inject.keys(),dtype=float)
    rats = np.zeros_like(inject.keys(),dtype=float)
    i = -1
    
    # For each injection, gather output within frequency range
    for injName in inject.keys():
        i = i + 1
        injInfo = inject[injName]
        freqlo = float(injInfo[19])
        freqhi = float(injInfo[20])
        segList = list([])
        for freq in segments.keys():
            if (freq <= freqhi and freq >= freqlo):
                segList = segList + list([segments[freq]])
        maxSNR = 0
        for seg in segList:
            data = getData('./output/'+seg)
            for line in data:
                # Look for UL lines with nonzero ul
                if (len(line) > 20 and line[1] == 'ul' and float(line[14]) > 0):
                    ratio = injInfo[10]/float(line[14])
                    if (float(line[13]) < maxSNR):
                        continue
                    rats[i] = ratio
                    maxSNR = float(line[13])
                    # Detection if SNR > 7 and freq within 0.002
                    if (patt.search(line[5]) and float(line[13]) > 7 and
                        abs(float(line[7]) - injInfo[6]) < 0.002):
                        dets[i] = 1
                        
    # Evenly distribute h0/ul values into bins
    xbins = stats.mstats.mquantiles(rats,np.arange(11)*0.1)
    nbins = xbins.size - 1
    detmap = np.digitize(rats,xbins)
    dets = np.array(dets)
    percent = np.zeros(nbins)
    # Bin detections by h0/ul
    for i in range(nbins):
        subset = dets[detmap == i]
        if (subset.size > 0):
            percent[i] = np.mean(subset)*100.0
    plt.plot(xbins[0:nbins],percent,'o')
    plt.xlabel('h0/ul')
    plt.ylabel('Percent Detected')
    plt.savefig('detection.png',bbox_inches='tight')
