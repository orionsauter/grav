import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import sys
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

epsilon = 23.439281 * np.pi/180
eclipticPole = [0, -np.sin(epsilon), np.cos(epsilon)]

def eclipticDist(ra1, dec1, ra2, dec2):
    x = np.cos(ra1) * np.cos(dec1) - np.cos(ra2) * np.cos(dec2)
    y = np.sin(ra1) * np.cos(dec1) - np.sin(ra2) * np.cos(dec2)
    z = np.sin(dec1) - np.sin(dec2)

    a = x * eclipticPole[0] + y * eclipticPole[1] + z * eclipticPole[2]

    return np.sqrt(np.power(x-a*eclipticPole[0],2) +
                   np.power(y-a*eclipticPole[1],2) +
                   np.power(z-a*eclipticPole[2],2))

def cosSphericalDist(ra1,dec1,ra2,dec2):
    return np.sin(dec1)*np.sin(dec2)+np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2)

def sphericalDist(ra1,dec1,ra2,dec2):
    a = cosSphericalDist(ra1,dec1,ra2,dec2)
    if (a >= 1.0):
        return 0.0
    elif (a <= -1.0):
        return np.pi
    else:
        return np.arccos(a)

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
                cols = line.split(' ')
                if (len(cols) > 20 and cols[1] == 'ul' and float(cols[14]) > 0):
                    data = data + list([cols])
            f.close()
    return data

def getDetections(segments,inject):
    patt = re.compile('_all')
    ninjects = len(inject.keys())
    dets = np.zeros(ninjects)
    rats = np.zeros(ninjects)
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
    return [dets,rats]

def makePlot(dets,rats):
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

def diffHistos(segments,inject):
    patt = re.compile('_all')
    dets = list([])
    deltaSph = list([])
    deltaEcl = list([])
    deltaSD = list([])
    for freq in segments.keys():
        print freq
        data = getData('./output/'+segments[freq])
        for injName in inject.keys():
            injInfo = inject[injName]
            if (abs(freq - injInfo[6]) > 2.0):
                continue
            for line in data:
                # Look for UL lines with nonzero ul
                if (len(line) > 20 and line[1] == 'ul' and float(line[14]) > 0):
                    deltaSph = deltaSph + list([sphericalDist(injInfo[0],injInfo[1],float(line[9]),float(line[10]))*float(line[7])])
                    deltaEcl = deltaEcl + list([eclipticDist(injInfo[0],injInfo[1],float(line[9]),float(line[10]))*float(line[7])])
                    deltaSD = deltaSD + list([(injInfo[7] - float(line[8]))])
                    # Detection if SNR > 7 and freq within 0.002
                    if (patt.search(line[5]) and float(line[13]) > 7 and
                        abs(float(line[7]) - injInfo[6]) < 0.002):
                        dets = dets + list([1])
                    else:
                        dets = dets + list([0])
    plt.clf()
    plt.hist([deltaSph[i] for i in range(len(dets)) if (dets[i] == 1)],bins=25)
    plt.title('Difference in Spherical Distance')
    plt.savefig('SphDetHist.png',bbox_inches='tight')
    plt.clf()
    plt.hist([deltaEcl[i] for i in range(len(dets)) if (dets[i] == 1)],bins=25)
    plt.title('Difference in Ecliptic Distance')
    plt.savefig('EclDetHist.png',bbox_inches='tight')
    plt.clf()
    plt.hist([deltaSD[i] for i in range(len(dets)) if (dets[i] == 1)],bins=25)
    plt.title('Difference in Spindown')
    plt.savefig('SDDetHist.png',bbox_inches='tight')
    plt.clf()

if __name__ == "__main__":
    dag = sys.argv[1]
    segments = getOutputSegs(dag)
    inject = getInjections()
    #detect = getDetections(segments,inject)
    #makePlot(detect[0],detect[1])
    diffHistos(segments,inject)
