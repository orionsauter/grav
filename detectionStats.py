import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt

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
    dets = []
    rats = []
    for injName in inject.keys():
        injInfo = inject[injName]
        freqlo = float(injInfo[19])
        freqhi = float(injInfo[20])
        segList = list([])
        for freq in segments.keys():
            if (freq <= freqhi and freq >= freqlo):
                segList = segList + list([segments[freq]])
        for seg in segList:
            data = getData('./output/'+seg)
            for line in data:
                if (len(line) > 20 and line[1] == 'ul' and float(line[14]) > 0):
                    rats.append(injInfo[10]/float(line[14]))
                    if (patt.search(line[5]) and float(line[13]) > 7 and
                        abs(float(line[7]) - injInfo[6]) < 0.002):
                        dets.append(1)
                    else:
                        dets.append(0)
    norm, xbins = np.histogram(rats)
    nbins = xbins.size - 1
    detmap = np.digitize(rats,xbins)
    dets = np.array(dets)
    percent = np.zeros(nbins)
    for i in range(nbins): 
        percent[i] = np.sum(dets[detmap == i])*100.0/norm[i]
    plt.plot(xbins[0:nbins],percent,'o')
    plt.savefig('detection.png',bbox_inches='tight')
