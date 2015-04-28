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

def eclipticLong(ra1, dec1, ra2, dec2):
    x1 = np.cos(ra1)
    y1 = np.sin(ra1) * np.cos(epsilon) + np.tan(dec1) * np.sin(epsilon)
    l1 = np.arctan2(y1,x1)
    
    x2 = np.cos(ra2)
    y2 = np.sin(ra2) * np.cos(epsilon) + np.tan(dec2) * np.sin(epsilon)
    l2 = np.arctan2(y2,x2)

    return (l2 - l1)

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
    skip = 0
    i = 1
    for line in f:
        # Skip header
        if (skip):
            skip = 0
            continue
        vals = line.split()
        inject[i] = map(float,vals[3:])
        i = i + 1
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
                line = line.rstrip()
                cols = line.split(' ')
                if (len(cols) > 20 and cols[1] == 'snr' and float(cols[14]) > 0):
                    data = data + list([cols])
            f.close()
    return data

def getDetections(segments,inject):
    allpatt = re.compile('_all')
    segpatt = re.compile('A(\d+)_\d+')
    ninjects = len(inject.keys())
    dets = np.zeros(ninjects)
    rats = np.zeros(ninjects)
    i = -1
    
    # For each injection, gather output within frequency range
    for injName in inject.keys():
        i = i + 1
        injInfo = inject[injName]
        segList = list([])
        for freq in segments.keys():
            m = segpatt.search(segments[freq])
            if (m and float(m.group(1)) == injName):
                segList = segList + list([segments[freq]])
        maxSNR = 0
        for seg in segList:
            data = getData('./output/'+seg)
            for line in data:
                # Look for SNR lines with nonzero ul
                if (float(line[14]) > 0):
                    ratio = injInfo[10]/float(line[14])
                    if (float(line[13]) < maxSNR):
                        continue
                    rats[i] = ratio
                    maxSNR = float(line[13])
                    # Detection if SNR > 7 and freq within 0.002
                    if (allpatt.search(line[5]) and float(line[13]) > 7 and
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

def diffHistos(inject):
    allpatt = re.compile('_all')
    segpatt = re.compile('A(\d+)_\d+')
    dets = list([])
    deltaSph = list([])
    deltaEcl = list([])
    deltaSD = list([])
    deltaf = list([])
    injKeys = sorted(inject.keys())
    for injName in injKeys:
        seg = 'A'+str(injName)+'_0'
        data = getData('./output/'+seg)
        injInfo = inject[injName]
        maxSNR = 0
        sph = 0
        ecl = 0
        sd = 0
        df = 0
        det = 0
        for line in data:
            # Look for SNR lines with nonzero ul
            if (allpatt.search(line[5]) and float(line[13]) > maxSNR):
                maxSNR = float(line[13])
                sph = sphericalDist(injInfo[0],injInfo[1],float(line[9]),float(line[10]))*float(line[7])
                ecl = eclipticLong(injInfo[0],injInfo[1],float(line[9]),float(line[10]))*float(line[7])
                sd = -(injInfo[7] - float(line[8]))
                df = -(injInfo[6] - float(line[7]))
                # Detection if SNR > 7 and freq within 10 mHz
                if (float(line[13]) >= 7 and
                    abs(float(line[7]) - injInfo[6]) < 0.01):
                #if (float(line[13]) > 7):
                    det = 1
                else:
                    det = 0
        deltaSph = deltaSph + list([sph])
        deltaEcl = deltaEcl + list([ecl])
        deltaSD = deltaSD + list([sd])
        deltaf = deltaf + list([df])
        dets = dets + list([det])
    detSegs = [str(injKeys[i]) for i in range(len(injKeys)) if dets[i] == 1]
    f = open('detSegments.txt','w')
    f.writelines("\n".join(detSegs))
    f.close()
    
    plt.clf()
    n, bins, patches = plt.hist(deltaSph,bins=25)
    mu = np.mean(deltaSph)
    sigma = np.std(deltaSph)
    textstr = '$\mu=%e$\n$\sigma=%e$'%(mu, sigma)
    ax = plt.gca()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Difference in Spherical Distance (All)')
    plt.xlabel('frequency-scaled distance (rad*Hz)')
    plt.ylabel('counts')
    plt.savefig('SphAllHist.png',bbox_inches='tight')
    
    plt.clf()
    deltaSphDet = [deltaSph[i] for i in range(len(dets)) if (dets[i] == 1)]
    plt.hist(deltaSphDet,bins=bins)
    mu = np.mean(deltaSphDet)
    sigma = np.std(deltaSphDet)
    textstr = '$\mu=%e$\n$\sigma=%e$'%(mu, sigma)
    ax = plt.gca()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Difference in Spherical Distance (Detected)')
    plt.xlabel('frequency-scaled distance (rad*Hz)')
    plt.ylabel('counts')
    plt.savefig('SphDetHist.png',bbox_inches='tight')
    
    plt.clf()
    n, bins, patches = plt.hist(deltaEcl,bins=25)
    mu = np.mean(deltaEcl)
    sigma = np.std(deltaEcl)
    textstr = '$\mu=%e$\n$\sigma=%e$'%(mu, sigma)
    ax = plt.gca()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Difference in Ecliptic Longitude (All)')
    plt.xlabel('frequency-scaled distance (rad*Hz)')
    plt.ylabel('counts')
    plt.savefig('EclAllHist.png',bbox_inches='tight')
    
    plt.clf()
    deltaEclDet = [deltaEcl[i] for i in range(len(dets)) if (dets[i] == 1)]
    plt.hist(deltaEclDet,bins=bins)
    mu = np.mean(deltaEclDet)
    sigma = np.std(deltaEclDet)
    textstr = '$\mu=%e$\n$\sigma=%e$'%(mu, sigma)
    ax = plt.gca()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Difference in Ecliptic Longitude (Detected)')
    plt.xlabel('frequency-scaled distance (rad*Hz)')
    plt.ylabel('counts')
    plt.savefig('EclDetHist.png',bbox_inches='tight')
    
    plt.clf()
    n, bins, patches = plt.hist(deltaSD,bins=25)
    mu = np.mean(deltaSD)
    sigma = np.std(deltaSD)
    textstr = '$\mu=%e$\n$\sigma=%e$'%(mu, sigma)
    ax = plt.gca()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Difference in Spindown (All)')
    plt.xlabel('difference (Hz/s)')
    plt.ylabel('counts')
    plt.savefig('SDAllHist.png',bbox_inches='tight')
    
    plt.clf()
    deltaSDDet = [deltaSD[i] for i in range(len(dets)) if (dets[i] == 1)]
    plt.hist(deltaSDDet,bins=bins)
    mu = np.mean(deltaSDDet)
    sigma = np.std(deltaSDDet)
    textstr = '$\mu=%e$\n$\sigma=%e$'%(mu, sigma)
    ax = plt.gca()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Difference in Spindown (Detected)')
    plt.xlabel('difference (Hz/s)')
    plt.ylabel('counts')
    plt.savefig('SDDetHist.png',bbox_inches='tight')
    
    plt.clf()
    n, bins, patches = plt.hist(deltaf,bins=25)
    mu = np.mean(deltaf)
    sigma = np.std(deltaf)
    textstr = '$\mu=%e$\n$\sigma=%e$'%(mu, sigma)
    ax = plt.gca()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Difference in Frequency (All)')
    plt.xlabel('difference (Hz)')
    plt.ylabel('counts')
    plt.savefig('FreqAllHist.png',bbox_inches='tight')
    
    plt.clf()
    deltafDet = [deltaf[i] for i in range(len(dets)) if (dets[i] == 1)]
    plt.hist(deltafDet,bins=bins)
    mu = np.mean(deltafDet)
    sigma = np.std(deltafDet)
    textstr = '$\mu=%e$\n$\sigma=%e$'%(mu, sigma)
    ax = plt.gca()
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top')
    plt.title('Difference in Frequency (Detected)')
    plt.xlabel('difference (Hz)')
    plt.ylabel('counts')
    plt.savefig('FreqDetHist.png',bbox_inches='tight')

def diffScatter(inject):
    allpatt = re.compile('_all')
    snrs = list([])
    deltaf = list([])
    origf = list([])
    deltaSD = list([])
    deltaEcl = list([])
    deltaEclUnsc = list([])
    deltaRA = list([])
    for injName in inject.keys():
        seg = 'A'+str(injName)+'_0'
        data = getData('./output/'+seg)
        injInfo = inject[injName]
        maxSNR = 0
        if (abs(injInfo[1]) > np.pi/4):
            continue
        for line in data:
            # Look for SNR lines with nonzero ul
            if (allpatt.search(line[5]) and float(line[13]) > maxSNR):
                maxSNR = float(line[13])
                df = -(injInfo[6] - float(line[7]))
                dSD = -(injInfo[7] - float(line[8]))
                eclUnsc = eclipticLong(injInfo[0],injInfo[1],float(line[9]),float(line[10]))
                eclUnsc = ((eclUnsc + 3*np.pi) % (2*np.pi)) - np.pi
                ecl = eclUnsc*float(line[7])
                dRA = -(injInfo[0] - float(line[9]))
                dRA = ((dRA + 3*np.pi) % (2*np.pi)) - np.pi
                if (dRA > 1.0):
                    print str(injInfo[0])+' '+line[9]
                if (eclUnsc > 2.0):
                    print injName
        snrs = snrs + list([maxSNR])
        deltaf = deltaf + list([df])
        origf = origf + list([injInfo[6]])
        deltaSD = deltaSD + list([dSD])
        deltaEcl = deltaEcl + list([ecl])
        deltaEclUnsc = deltaEclUnsc + list([eclUnsc])
        deltaRA = deltaRA + list([dRA])
    plt.clf()
    x = deltaf
    y = deltaSD
    plt.plot(x,y,'.r',label='All')

    x = [deltaf[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    y = [deltaSD[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    plt.plot(x,y,'.b',label='SNR > 6')

    x = [deltaf[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    y = [deltaSD[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    plt.plot(x,y,'.g',label='SNR >= 7')

    plt.xlabel('frequency difference (Hz)')
    plt.ylabel('spindown difference (Hz/s)')
    plt.legend()
    plt.savefig('f-sdScatter.png',bbox_inches='tight')
    
    plt.clf()
    x = deltaf
    y = deltaEcl
    plt.loglog(x,np.abs(y),'.r',label='All')

    x = [deltaf[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    y = [deltaEcl[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    plt.loglog(x,np.abs(y),'.b',label='SNR > 6')

    x = [deltaf[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    y = [deltaEcl[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    plt.loglog(x,np.abs(y),'.g',label='SNR >= 7')

    plt.xlabel('log frequency difference (Hz)')
    plt.ylabel('log frequncy-scaled ecliptic difference (rad*Hz)')
    plt.legend()
    plt.savefig('f-EclScatterLog.png',bbox_inches='tight')
    
    plt.clf()
    x = origf
    y = deltaEcl
    plt.loglog(x,np.abs(y),'.r',label='All')

    x = [origf[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    y = [deltaEcl[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    plt.loglog(x,np.abs(y),'.b',label='SNR > 6')

    x = [origf[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    y = [deltaEcl[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    plt.loglog(x,np.abs(y),'.g',label='SNR >= 7')

    plt.xlabel('log injection frequency (Hz)')
    plt.ylabel('log frequncy-scaled ecliptic difference (rad*Hz)')
    plt.legend()
    plt.savefig('origf-EclScatterLog.png',bbox_inches='tight')
    
    plt.clf()
    x = origf
    y = deltaEclUnsc
    plt.loglog(x,np.abs(y),'.r',label='All')

    x = [origf[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    y = [deltaEclUnsc[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    plt.loglog(x,np.abs(y),'.b',label='SNR > 6')

    x = [origf[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    y = [deltaEclUnsc[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    plt.loglog(x,np.abs(y),'.g',label='SNR >= 7')

    plt.xlabel('log injection frequency (Hz)')
    plt.ylabel('log ecliptic difference (rad)')
    plt.legend()
    plt.savefig('origf-EclUnscScatterLog.png',bbox_inches='tight')

    plt.clf()
    x = deltaf
    y = deltaEcl
    plt.plot(x,y,'.r',label='All')

    x = [deltaf[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    y = [deltaEcl[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    plt.plot(x,y,'.b',label='SNR > 6')

    x = [deltaf[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    y = [deltaEcl[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    plt.plot(x,y,'.g',label='SNR >= 7')

    plt.xlabel('frequency difference (Hz)')
    plt.ylabel('frequncy-scaled ecliptic difference (rad*Hz)')
    plt.legend()
    plt.savefig('f-EclScatter.png',bbox_inches='tight')
    
    plt.clf()
    x = deltaf
    y = deltaRA
    plt.plot(x,y,'.r',label='All')

    x = [deltaf[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    y = [deltaRA[i] for i in range(len(snrs)) if (snrs[i] > 6)]
    plt.plot(x,y,'.b',label='SNR > 6')

    x = [deltaf[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    y = [deltaRA[i] for i in range(len(snrs)) if (snrs[i] >= 7)]
    plt.plot(x,y,'.g',label='SNR >= 7')

    plt.xlabel('frequency difference (Hz)')
    plt.ylabel('RA difference (rad)')
    plt.legend()
    plt.savefig('f-RAScatter.png',bbox_inches='tight')

if __name__ == "__main__":
    dag = sys.argv[1]
    segments = getOutputSegs(dag)
    inject = getInjections()
    #detect = getDetections(segments,inject)
    #makePlot(detect[0],detect[1])
    diffHistos(inject)
    diffScatter(inject)
