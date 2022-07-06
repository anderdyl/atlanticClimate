import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import xarray as xr
import matplotlib.pyplot as plt
from dipy.segment.clustering import QuickBundles
from mpl_toolkits.basemap import Basemap
import geopy.distance
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import Metric
import random


import os
from netCDF4 import Dataset
import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
import datetime as DT
from matplotlib import cm
import pickle
from scipy.io.matlab.mio5_params import mat_struct
import scipy.io as sio
from datetime import datetime
from datetime import timedelta
def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    :param datenum: Date in datenum format
    :return:        Datetime object corresponding to datenum.
    """
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    return datetime.fromordinal(int(datenum)) \
           + timedelta(days=int(days)) \
           + timedelta(hours=int(hours)) \
           + timedelta(minutes=int(minutes)) \
           + timedelta(seconds=round(seconds)) \
           - timedelta(days=366)


def ReadMatfile(p_mfile):
    'Parse .mat file to nested python dictionaries'

    def RecursiveMatExplorer(mstruct_data):
        # Recursive function to extrat mat_struct nested contents

        if isinstance(mstruct_data, mat_struct):
            # mstruct_data is a matlab structure object, go deeper
            d_rc = {}
            for fn in mstruct_data._fieldnames:
                d_rc[fn] = RecursiveMatExplorer(getattr(mstruct_data, fn))
            return d_rc

        else:
            # mstruct_data is a numpy.ndarray, return value
            return mstruct_data

    # base matlab data will be in a dict
    mdata = sio.loadmat(p_mfile, squeeze_me=True, struct_as_record=False)
    mdata_keys = [x for x in mdata.keys() if x not in
                  ['__header__','__version__','__globals__']]

    # use recursive function
    dout = {}
    for k in mdata_keys:
        dout[k] = RecursiveMatExplorer(mdata[k])
    return dout



def dt2cal(dt):
    """
    Convert array of datetime64 to a calendar array of year, month, day, hour,
    minute, seconds, microsecond with these quantites indexed on the last axis.

    Parameters
    ----------
    dt : datetime64 array (...)
        numpy.ndarray of datetimes of arbitrary shape

    Returns
    -------
    cal : uint32 array (..., 7)
        calendar array with last axis representing year, month, day, hour,
        minute, second, microsecond
    """

    # allocate output
    out = np.empty(dt.shape + (7,), dtype="u4")
    # decompose calendar floors
    Y, M, D, h, m, s = [dt.astype(f"M8[{x}]") for x in "YMDhms"]
    out[..., 0] = Y + 1970 # Gregorian Year
    out[..., 1] = (M - Y) + 1 # month
    out[..., 2] = (D - M) + 1 # dat
    out[..., 3] = (dt - D).astype("m8[h]") # hour
    out[..., 4] = (dt - h).astype("m8[m]") # minute
    out[..., 5] = (dt - m).astype("m8[s]") # second
    out[..., 6] = (dt - s).astype("m8[us]") # microsecond
    return out


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color



def datevec2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [DT.datetime(d[0], d[1], d[2], d[3], d[4]) for d in d_vec]

def dateDay2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [DT.datetime(d[0], d[1], d[2]) for d in d_vec]


class GPSDistance(Metric):
    """computer the average GPS distance between two streamlines"""
    def __init__(self):
        super(GPSDistance, self).__init__(feature=ResampleFeature(nb_points=20))

    def are_compatible(self, shape1, shape2):
        return len(shape1) == len(shape2)

    # def dist(self,v1,v2):
    #     x = [geopy.distance.vincenty([p[0][0],p[0][1]], [p[1][0],p[1][1]]).km for p in list(zip(v1,v2))]
    #     currD = np.mean(x)
    #     return currD
    def dist(self, v1, v2):
        x = [geopy.distance.distance([p[0][0], p[0][1]], [p[1][0], p[1][1]]).kilometers for p in list(zip(v1, v2))]
        currD = np.mean(x)
        return currD



streams = []
sequencecStreams = []
singularStreams = []
collisions = []

for pp in range(100):

    file = r"/media/dylananderson/Elements/Sims/simulation{}.pickle".format(pp)

    with open(file, "rb") as input_file:
        simsInput = pickle.load(input_file)
    simulationData = simsInput['simulationData']
    df = simsInput['df']
    time = simsInput['time']
    year = np.array([tt.year for tt in time])
    df['year'] = year
    month = np.array([tt.month for tt in time])
    df['month'] = month
    hs = simulationData[:,0]
    tp = simulationData[:,1]
    ss = simulationData[:,3]
    dmCombined = simulationData[:,2]

    waveNorm = dmCombined - 72
    neg = np.where((waveNorm > 180))
    waveNorm[neg[0]] = waveNorm[neg[0]]-360
    offpos = np.where((waveNorm>90))
    offneg = np.where((waveNorm<-90))
    waveNorm[offpos[0]] = waveNorm[offpos[0]]*0
    waveNorm[offneg[0]] = waveNorm[offneg[0]]*0



    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n


    dataTide = ReadMatfile('/home/dylananderson/projects/duckGeomorph/emulatorTide2122.mat')

    tide = dataTide['synTide'][0:len(hs)]


    hsCombined = hs
    hsSmooth = moving_average(hs,3)+0.05
    tpCombined = tp

    tC = np.array(time)


    Bf = 0.083
    L0 = 9.81 * np.square(tpCombined) / (2 * np.pi)
    H0 = hsCombined
    R2 = 1.1 * ((0.35 * Bf * np.power(np.multiply(H0, L0), 0.5)) + np.divide(
        np.power((H0 * L0 * (0.563 * (np.power(Bf, 2)) + 0.004)), (0.5)), 2))


    TWL = ss + R2 + tide
    wl = ss + tide
    hAbove = np.where((TWL > 3))

    # plt.figure()
    # plt.plot(time,TWL)
    # plt.plot(time,wl)


    avgHs = np.nanmean(hsCombined)
    hs98 = np.nanpercentile(hsCombined,98)
    hs95 = np.nanpercentile(hsCombined,95)
    hs90 = np.nanpercentile(hsCombined,90)
    hs85 = np.nanpercentile(hsCombined,85)

    H0 = moving_average(hs,3)+0.05
    hsCombined = H0

    stormHsInd = np.where((hsSmooth > 1.95))
    # stormHsInd = np.where((H0 > 1.95))

    # stormHsInd = np.where((hsSmooth > hs90))
    stormHsList = [list(group) for group in mit.consecutive_groups(stormHsInd[0])]


    hsStormList = []
    hsStormMaxList = []
    hsStormMinList = []
    tpStormList = []
    tpStormMaxList = []
    tpStormMinList = []
    dmStormList = []
    #ntrStormList = []
    timeStormList = []
    timeStartList = []
    timeEndList = []
    hourStormList = []
    indStormList = []
    indNTRStormList = []
    #bmuStormList = []
    wlStormList = []
    for xx in range(len(stormHsList)-2):

        i1 = stormHsList[xx][0]
        i2 = stormHsList[xx][-1]
        t1 = tC[i1]
        t2 = tC[i2]

        tempWave = np.where((tC < t2) & (tC > t1))
        if len(tempWave[0]) > 10:

            tempWave = np.where((tC < t2) & (tC > t1))
            indices = np.arange(i1,i2)
            hsStormList.append(hsCombined[tempWave])

            wlStormList.append(wl[tempWave])

            tpStormList.append(tpCombined[tempWave])
            dmStormList.append(waveNorm[tempWave])
            timeStormList.append(tC[tempWave])
            timeStartList.append(t1)
            timeEndList.append(t2)

            indStormList.append(indices)


    hsMaxStorm = []
    hsMaxTimeInStorm = []
    #bmuMaxTimeInStorm = []
    durationHoursStorm = []
    dmAvgStorm = []
    #ntrMaxStorm = []
    wlMaxStorm = []
    tpMaxStorm = []
    timeStorm = []
    startStorm = []
    endStorm = []
    for x in range(len(hsStormList)):
        tempHs = hsStormList[x]
        tempTp = tpStormList[x]
        timeStorm.append(timeStormList[x][0])
        #timeStorm.append(timeStormList[x][12])
        startStorm.append(timeStartList[x])
        endStorm.append(timeEndList[x])

        tempMaxInds = np.where((np.nanmax(tempHs) == tempHs))
        hsMaxStorm.append(np.nanmax(tempHs))
        dayOf = int(np.floor(tempMaxInds[0][-1]/24))
        hsMaxTimeInStorm.append(tempMaxInds[0][-1])
        tpMaxStorm.append(tempTp[tempMaxInds[0][-1]])
        #bmuMaxTimeInStorm.append(bmuStormList[x][dayOf])
        durationHoursStorm.append(len(tempHs))
        tempDm = dmStormList[x]
        dmAvgStorm.append(np.nanmean(tempDm))
        #tempNTR = ntrStormList[x]
        tempWL = wlStormList[x]
        if len(tempWL) > 0:
            # ntrMaxStorm.append(np.nanmax(tempNTR))
            wlMaxStorm.append(np.nanmax(tempWL))
        else:
            wlMaxStorm.append(np.nan)



    hsStormArray = np.array(hsMaxStorm)
    tpStormArray = np.array(tpMaxStorm)
    dmStormArray = np.array(dmAvgStorm)
    # ntrStormArray = np.array(ntrMaxStorm)
    wlStormArray = np.array(wlMaxStorm)
    durationStormArray = np.array(durationHoursStorm)
    timeStormArray = np.array(timeStorm)
    endStormArray = np.array(endStorm)
    startStormArray = np.array(startStorm)

    # badIndex = np.isnan(ntrStormArray)
    badIndex = np.isnan(wlStormArray)

    # filteredHs = hsStormArray[~np.isnan(ntrStormArray)]
    # filteredTp = tpStormArray[~np.isnan(ntrStormArray)]
    # filteredDm = dmStormArray[~np.isnan(ntrStormArray)]
    # filteredDur = durationStormArray[~np.isnan(ntrStormArray)]
    # filteredTime = timeStormArray[~np.isnan(ntrStormArray)]
    # #filteredNTR = ntrStormArray[~np.isnan(ntrStormArray)]
    # filteredStartTime = startStormArray[~np.isnan(ntrStormArray)]
    # filteredEndTime = endStormArray[~np.isnan(ntrStormArray)]

    filteredHs = hsStormArray[~np.isnan(wlStormArray)]
    filteredTp = tpStormArray[~np.isnan(wlStormArray)]
    filteredDm = dmStormArray[~np.isnan(wlStormArray)]
    filteredDur = durationStormArray[~np.isnan(wlStormArray)]
    filteredTime = timeStormArray[~np.isnan(wlStormArray)]
    #filteredNTR = ntrStormArray[~np.isnan(ntrStormArray)]
    filteredStartTime = startStormArray[~np.isnan(wlStormArray)]
    filteredEndTime = endStormArray[~np.isnan(wlStormArray)]

    mask = np.arange(0,len(hsStormList))
    keep = mask[~np.isnan(wlStormArray)]
    hsStormSubList = []
    [hsStormSubList.append(hsStormList[ii]) for ii in keep]
    tpStormSubList = []
    [tpStormSubList.append(tpStormList[ii]) for ii in keep]
    #ntrStormSubList = []
    #[ntrStormSubList.append(ntrStormList[ii]) for ii in keep]
    wlStormSubList = []
    [wlStormSubList.append(wlStormList[ii]) for ii in keep]


    # plt.figure()
    # plt.plot(filteredTime,filteredHs,'o')

    numDaysBetween = 4
    windowOfStorms = []
    counter = 0
    counter2 = 0
    counter3 = 0
    counter4 = 0
    counter5 = 0
    counter6 = 0
    counter7 = 0
    storms2 = []
    storms3 = []
    storms4 = []
    storms5 = []
    storms6 = []
    storms7 = []
    storms8 = []
    for ii in range(len(filteredTime)-2):
        indexNear = np.where((filteredStartTime > filteredEndTime[ii]) & (filteredStartTime < filteredEndTime[ii]+timedelta(days=30)))
        windowOfStorms.append(indexNear)
        timeDiff = filteredStartTime[ii+1]-filteredEndTime[ii]
        #print(timeDiff)
        if timeDiff.days < numDaysBetween:
            counter = counter + 1
            #print('3 days or less: {} times'.format(counter))
            if (ii + 2) > (len(filteredStartTime)-1):
                print('rut roh')
                days2 = 10
            else:
                timeDiff2 = filteredStartTime[ii+2]-filteredEndTime[ii+1]
                days2 = timeDiff2.days
            storms2.append([ii,ii+1])
            if days2 < numDaysBetween:
                counter2 = counter2 + 1
                if (ii + 3) > (len(filteredStartTime)-1):
                    print('rut roh')
                    days3 = 10
                else:
                #print('alright, found a three storm cluster: {} times'.format(counter2))
                    timeDiff3 = filteredStartTime[ii+3]-filteredEndTime[ii+2]
                    days3 = timeDiff3.days
                storms3.append([ii, ii + 1, ii + 2])
                if days3 < numDaysBetween:
                    counter3 = counter3 + 1
                    storms4.append([ii, ii + 1, ii + 2, ii + 3])
                    if (ii + 4) > (len(filteredStartTime)-1):
                        print('rut roh')
                        days4 = 10
                    else:
                        timeDiff4 = filteredStartTime[ii + 4] - filteredEndTime[ii + 3]
                        days4 = timeDiff4.days
                    #print('alright, found a four storm cluster: {} times'.format(counter3))
                    if days4 < numDaysBetween:
                        counter4 = counter4 + 1
                        storms5.append([ii, ii + 1, ii + 2, ii + 3, ii + 4])
                        print('alright, found a five storm cluster: {} times'.format(counter4))
                        if (ii + 5) > (len(filteredStartTime)-1):
                            print('rut roh')
                            days5 = 10
                        else:
                            timeDiff5 = filteredStartTime[ii + 5] - filteredEndTime[ii + 4]
                            days5 = timeDiff5.days
                        if days5 < numDaysBetween:
                            counter5 = counter5 + 1
                            storms6.append([ii, ii + 1, ii + 2, ii + 3, ii + 4, ii + 5])
                            print('oh wow, found a six storm cluster: {} times'.format(counter5))
                            if (ii + 6) > (len(filteredStartTime)-1):
                                print('rut roh')
                                days6 = 10
                            else:
                                timeDiff6 = filteredStartTime[ii + 6] - filteredEndTime[ii + 5]
                                days6 = timeDiff6.days
                            if days6 < numDaysBetween:
                                counter6 = counter6 + 1
                                storms7.append([ii, ii + 1, ii + 2, ii + 3, ii + 4, ii + 5, ii + 6])
                                print('even more of a wow, found a seven storm cluster: {} times'.format(counter6))
                                if (ii + 6) > (len(filteredStartTime)-1):
                                    print('rut roh')
                                    days7 = 10
                                else:
                                    timeDiff7 = filteredStartTime[ii + 7] - filteredEndTime[ii + 6]
                                    days7 = timeDiff7.days
                                if days7 < numDaysBetween:
                                    counter7 = counter7 + 1
                                    storms8.append([ii, ii + 1, ii + 2, ii + 3, ii + 4, ii + 5, ii + 6, ii + 7])
                                    print('even more of an oh my goodness, found a seven storm cluster: {} times'.format(counter7))
                                    if (ii + 7) > (len(filteredStartTime)-1):
                                        print('rut roh')
                                        days8 = 10
                                    else:
                                        timeDiff8 = filteredStartTime[ii + 8] - filteredEndTime[ii + 7]
                                        days8 = timeDiff8.days
    ### ok, so let's find all of the "clusters" already captured in the 5-storm clusters

    sevenAlreadyInEight = []
    for ii in range(len(storms7)):
        subii = storms7[ii]
        for qq in subii:
            if qq in np.array(storms8).flatten():
                # print(ii)
                sevenAlreadyInEight.append(ii)
    sevenAlreadyInEight = np.unique(sevenAlreadyInEight)
    print(sevenAlreadyInEight)

    sixAlreadyInSeven = []
    for ii in range(len(storms6)):
        subii = storms6[ii]
        for qq in subii:
            if qq in np.array(storms7).flatten():
                # print(ii)
                sixAlreadyInSeven.append(ii)
    sixAlreadyInSeven = np.unique(sixAlreadyInSeven)
    print(sixAlreadyInSeven)

    fiveAlreadyInSix = []
    for ii in range(len(storms5)):
        subii = storms5[ii]
        for qq in subii:
            if qq in np.array(storms6).flatten():
                #print(ii)
                fiveAlreadyInSix.append(ii)
    fiveAlreadyInSix = np.unique(fiveAlreadyInSix)
    print(fiveAlreadyInSix)

    ### ok, so let's find all of the "clusters" already captured in the 5-storm clusters
    fourAlreadyInFive = []
    for ii in range(len(storms4)):
        subii = storms4[ii]
        for qq in subii:
            if qq in np.array(storms5).flatten():
                #print(ii)
                fourAlreadyInFive.append(ii)
    fourAlreadyInFive = np.unique(fourAlreadyInFive)
    print(fourAlreadyInFive)

    ### and again for the three captured in the 4-storm clusters
    threeAlreadyInFour = []
    for ii in range(len(storms3)):
        subii = storms3[ii]
        for qq in subii:
            if qq in np.array(storms4).flatten():
                #print(ii)
                threeAlreadyInFour.append(ii)
    threeAlreadyInFour = np.unique(threeAlreadyInFour)

    ### and again for the two captured in the 3-storm clusters
    twoAlreadyInThree = []
    for ii in range(len(storms2)):
        subii = storms2[ii]
        for qq in subii:
            if qq in np.hstack((np.array(storms3).flatten(),np.array(storms4).flatten())):
                #print(ii)
                twoAlreadyInThree.append(ii)

    twoAlreadyInThree = np.unique(twoAlreadyInThree)


    ## ok now need to remove these indices
    stormSequences = storms8
    # get rid of the 7s first
    index7 = np.arange(0,len(storms7))
    mask7 = np.delete(index7,sevenAlreadyInEight)
    [stormSequences.append(storms7[ii]) for ii in mask7]
    # get rid of the 6s first
    index6 = np.arange(0,len(storms6))
    mask6 = np.delete(index6,sixAlreadyInSeven)
    [stormSequences.append(storms6[ii]) for ii in mask6]
    # get rid of the 5s first
    index5 = np.arange(0,len(storms5))
    mask5 = np.delete(index5,fiveAlreadyInSix)
    [stormSequences.append(storms5[ii]) for ii in mask5]
    # stormSequences = storms5
    # get rid of the 4s first
    index4 = np.arange(0,len(storms4))
    mask4 = np.delete(index4,fourAlreadyInFive)
    [stormSequences.append(storms4[ii]) for ii in mask4]
    # get rid of the 3s next
    index3 = np.arange(0,len(storms3))
    mask3 = np.delete(index3,threeAlreadyInFour)
    [stormSequences.append(storms3[ii]) for ii in mask3]
    # now the 2s in 3s
    index2 = np.arange(0,len(storms2))
    mask2 = np.delete(index2,twoAlreadyInThree)
    [stormSequences.append(storms2[ii]) for ii in mask2]

    def flatten_list(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    # We will need to find all of the singular events not in a cluster
    allClusteredStorms = np.sort(flatten_list(stormSequences))
    allStorms = np.arange(0,len(hsStormSubList))
    singularStorms = np.delete(allStorms,allClusteredStorms)


    print('working on simulation #{}'.format(pp))

    # lets find the start time and end time of each cluster
    # plt.figure()
    Bf = 0.1
    duneToe = 3
    hmax = 25
    # ax1 = plt.subplot2grid((8,8),(0,0),rowspan=8,colspan=4)
    # ax3 = plt.subplot2grid((8,8),(4,4),rowspan=4,colspan=1)
    # ax4 = plt.subplot2grid((8,8),(4,5),rowspan=4,colspan=1)
    # ax5 = plt.subplot2grid((8,8),(4,6),rowspan=4,colspan=1)
    # ax6 = plt.subplot2grid((8,8),(4,7),rowspan=4,colspan=1)

    for ii in range(len(stormSequences)):#range(5):#
        tempStormSequence = stormSequences[ii]
        sStorm = tempStormSequence[0]
        eStorm = tempStormSequence[-1]
        sTime = filteredStartTime[sStorm]
        eTime = filteredEndTime[eStorm]
        # NEED TO GET WAVES #
        sIndices = np.where((tC > sTime) & (tC < eTime))
        timeDelts = tC[sIndices]-tC[sIndices[0][0]]
        timeInt = [(pp.seconds/(60*60*24)+pp.days) for pp in timeDelts]
        wp = np.cumsum(1025*np.square(9.81)*np.square((hsCombined[sIndices]))*tpCombined[sIndices]/(64*np.pi))
        L0 = 9.81*np.square(tpCombined[sIndices])/(2*np.pi)
        H0 = hsCombined[sIndices]
        R2 = 1.1*((0.35*Bf*np.power(np.multiply(H0,L0),0.5)) + np.divide(np.power((H0*L0*(0.563*(np.power(Bf,2)) + 0.004)),(0.5)),2))
        # NEED TO GET WATER LEVELS #
        #wIndices = np.where((wlTime > sTime) & (wlTime < eTime))
        #timeDeltsW = wlTime[wIndices]-wlTime[wIndices[0][0]]
        #timeIntW = [(pp.seconds/(60*60*24)+pp.days) for pp in timeDeltsW]
        timeIntW = timeInt
        wlsub = wl[sIndices]

        if len(R2) > len(wlsub):
            TWL = wlsub + R2[0:len(wlsub)]
            wp = wp[0:len(wlsub)]
            timeSub = timeIntW
        else:
            TWL = wlsub[0:len(R2)] + R2
            timeSub = timeInt

        lat_lng_data = np.c_[timeSub, wp]
        streams.append(lat_lng_data)
        sequencecStreams.append(lat_lng_data)

        highWL = np.where((TWL > duneToe))
        zerosDune = np.zeros((np.size(TWL)))
        zerosDune[highWL] = np.ones((np.size(highWL),))
        collisionDune = np.cumsum(zerosDune)
        collisions.append(collisionDune)
        #
        # ax1.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
        #
        # if len(highWL[0]) > 0:
        #     print('stormSequences {} has {}'.format(ii,len(highWL[0])))
        # #plt.scatter(timeSub,wp,marker='.',c=cm.Blues(collisionDune),vmin=0,vmax=30,edgecolor='none')
        # # plt.plot(timeSub,np.cumsum(TWL),'b-')
        # # plt.plot(timeSub,(TWL),'r-')
        # # plt.plot(timeInt,wp,'b-')
        # monthTemp=sTime.month
        # if monthTemp > 2 and monthTemp < 6:
        #     #print('in spring time, but not plotting?')
        #
        #     ax6.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
        # elif monthTemp > 5 and monthTemp < 9:
        #     #print('summer storm... skipping')
        #     ax3.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
        #
        # elif monthTemp > 8 and monthTemp < 12:
        #     #print('in fall time, but not plotting?')
        #
        #     ax4.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
        # else:
        #     #print('in winter time, but not plotting?')
        #
        #     ax5.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')


    for ii in range(len(singularStorms)):
        tempStormSequence = singularStorms[ii]
        sStorm = tempStormSequence #[0]
        eStorm = tempStormSequence#[-1]
        sTime = filteredStartTime[sStorm]
        eTime = filteredEndTime[eStorm]
        # NEED TO GET WAVES #
        sIndices = np.where((tC > sTime) & (tC < eTime))
        timeDelts = tC[sIndices]-tC[sIndices[0][0]]
        timeInt = [(pp.seconds/(60*60*24)+pp.days) for pp in timeDelts]
        wp = np.cumsum(1025*np.square(9.81)*np.square((hsCombined[sIndices]))*tpCombined[sIndices]/(64*np.pi))
        L0 = 9.81*np.square(tpCombined[sIndices])/(2*np.pi)
        H0 = hsCombined[sIndices]
        R2 = 1.1*((0.35*Bf*np.power(np.multiply(H0,L0),0.5)) + np.divide(np.power((H0*L0*(0.563*(np.power(Bf,2)) + 0.004)),(0.5)),2))

        # NEED TO GET WATER LEVELS #
        #wIndices = np.where((wlTime > sTime) & (wlTime < eTime))
        #timeDeltsW = wlTime[wIndices]-wlTime[wIndices[0][0]]
        #timeIntW = [(pp.seconds/(60*60*24)+pp.days) for pp in timeDeltsW]
        timeIntW = timeInt
        wlsub = wl[sIndices]

        if len(R2) > len(wlsub):
            TWL = wlsub + R2[0:len(wlsub)]
            wp = wp[0:len(wlsub)]
            timeSub = timeIntW
        else:
            TWL = wlsub[0:len(R2)] + R2
            timeSub = timeInt

        lat_lng_data = np.c_[timeSub, wp]
        streams.append(lat_lng_data)
        singularStreams.append(lat_lng_data)

        highWL = np.where((TWL > duneToe))
        zerosDune = np.zeros((np.size(TWL)))
        zerosDune[highWL] = np.ones((np.size(highWL),))
        collisionDune = np.cumsum(zerosDune)
        collisions.append(collisionDune)
        # if len(highWL[0]) > 0:
        #     print('singleStorms {} has {}'.format(ii,len(highWL[0])))
        # #plt.scatter(timeSub,wp,marker='.',c=cm.YlOrRd(collisionDune),vmin=0,vmax=20,edgecolor='none')
        # p1 = ax1.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
        # # plt.plot(timeSub,(TWL),'r-')
        # # plt.plot(timeSub,np.cumsum(TWL),'r-')
        # #plt.plot(timeInt,wp,'r-')
        # # print(sTime.month)
        # monthTemp = sTime.month
        # if monthTemp > 2 and monthTemp < 6:
        #     ax6.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
        # elif monthTemp> 5 and monthTemp < 9:
        #     #print('summer storm... skipping')
        #     ax3.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
        #
        # elif monthTemp > 8 and monthTemp < 12:
        #     #print('in fall time, but not plotting?')
        #
        #     ax4.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
        # else:
        #     #print('in winter time, but not plotting?')
        #
        #     ax5.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
        #

    #
    #
    # ii = 18
    # tempStormSequence = stormSequences[ii]
    # sStorm = tempStormSequence[0]
    # eStorm = tempStormSequence[-1]
    # sTime = filteredStartTime[sStorm]
    # eTime = filteredEndTime[eStorm]
    # # NEED TO GET WAVES #
    # sIndices = np.where((tC > sTime) & (tC < eTime))
    # timeDelts = tC[sIndices]-tC[sIndices[0][0]]
    # timeInt = [(pp.seconds/(60*60*24)+pp.days) for pp in timeDelts]
    # wp = np.cumsum(1025*np.square(9.81)*np.square((hsCombined[sIndices]))*tpCombined[sIndices]/(64*np.pi))
    # L0 = 9.81*np.square(tpCombined[sIndices])/(2*np.pi)
    # H0 = hsCombined[sIndices]
    # R2 = 1.1*((0.35*Bf*np.power(np.multiply(H0,L0),(1/2))) + (np.power((H0*L0*(0.563*(np.power(Bf,2)) + 0.004)),(1/2))/2))
    # # NEED TO GET WATER LEVELS #
    # # wIndices = np.where((wlTime > sTime) & (wlTime < eTime))
    # # timeDeltsW = wlTime[wIndices]-wlTime[wIndices[0][0]]
    # # timeIntW = [(pp.seconds/(60*60*24)+pp.days) for pp in timeDeltsW]
    # timeIntW = timeInt
    # wlsub = wl[sIndices]
    #
    # if len(R2) > len(wlsub):
    #     TWL = wlsub + R2[0:len(wlsub)]
    #     wp = wp[0:len(wlsub)]
    #     wlnew = wlsub
    #     timeSub = timeIntW
    # else:
    #     TWL = wlsub[0:len(R2)] + R2
    #     wlnew = wlsub[0:len(R2)]
    #     timeSub = timeInt
    #
    # highWL = np.where((TWL > 4))
    # zerosDune = np.zeros((np.size(TWL)))
    # zerosDune[highWL] = np.ones((np.size(highWL),))
    # collisionDune = np.cumsum(zerosDune)
    #
    # ax2 = plt.subplot2grid((8,8),(0,4),rowspan=4,colspan=4)
    # ax2.plot(timeSub,H0)
    # # ax2.plot(timeSub,TWL,label='SWL+R2')
    # # ax2.plot(timeSub,wlnew,'--',label='SWL')
    # # ax2.plot(timeSub,duneToe*np.ones((np.size(timeSub))),'k--')
    #
    #
    # ax3.set_title('Summer')
    # ax3.set_xlabel('Days')
    # ax3.set_xlim([0,20])
    # ax3.set_ylim([0,1.4e7])
    # ax4.set_title('Fall')
    # ax4.set_xlabel('Days')
    # ax4.set_xlim([0,20])
    # ax4.set_ylim([0,1.4e7])
    # ax5.set_title('Winter')
    # ax5.set_xlabel('Days')
    # ax5.set_xlim([0,20])
    # ax5.set_ylim([0,1.4e7])
    # ax6.set_title('Spring')
    # ax6.set_xlabel('Days')
    # ax6.set_xlim([0,20])
    # ax6.set_ylim([0,1.4e7])
    # ax2.set_xlabel('Days during example cluster')
    # ax2.set_title('Example Storm Sequence')
    # ax1.set_xlabel('Days since beginning of event')
    # ax2.set_ylabel('Water Level (m)')
    # ax1.set_ylabel('Cumulative Wave Power')
    # ax1.set_xlim([0,20])
    # cb1 = plt.colorbar(p1,ax=ax1)
    # cb1.set_label('hours of dune collision')


# What attributes would be included...
#   - total length of time
#   - total WP
#   - total collision
#   - average WP
#   - steepest WP gradient
#   - time of steepest WP gradient

attributes = []
for ii in range(len(streams)):
    endDays = streams[ii][-1,0]
    endWP = streams[ii][-1,1]
    endDune = collisions[ii][-1]
    avgWP = np.mean(streams[ii][:,1])
    maxGrd = np.max(np.gradient(streams[ii][:,1]))
    maxTimeGrd = np.argmax(np.gradient(streams[ii][:,1]))
    attributes.append([endDays,endWP,endDune,avgWP,maxGrd,maxTimeGrd])


attribs = np.array(attributes)

meanAttribs = np.mean(attribs,axis=0)
stdAttribs = np.std(attribs,axis=0)

normed = np.divide(np.subtract(attribs,meanAttribs),stdAttribs)

from sklearn.cluster import KMeans
num_clusters = 20
kma = KMeans(n_clusters=num_clusters, n_init=2000).fit(normed)
# groupsize
_, group_size = np.unique(kma.labels_, return_counts=True)
# groups
d_groups = {}
k_groups = []
for k in range(num_clusters):
    d_groups['{0}'.format(k)] = np.where(kma.labels_ == k)
    k_groups.append(np.where(kma.labels_ == k))

from sklearn.metrics import pairwise_distances_argmin_min
closest, _ = pairwise_distances_argmin_min(kma.cluster_centers_, normed)

# # centroids
# centroids = np.dot(kma.cluster_centers_, EOFsub)
# # km, x and var_centers
# km = np.multiply(
#     centroids,
#     np.tile(SlpGrdStd, (num_clusters, 1))
# ) + np.tile(SlpGrdMean, (num_clusters, 1))

# from scipy.cluster.vq import vq
# # centroids: N-dimensional array with your centroids
# # points:    N-dimensional array with your data points
# closest, distances = vq(centroids, points)
# clusters = num_clusters
etcolors = cm.rainbow(np.linspace(0, 1,num_clusters))

fig = plt.figure()
p2 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
#sortInd = np.flipud(np.argsort(np.asarray([len(ii) for ii in clusters.clusters])))
for clustersIndex in range(num_clusters):
    p2.plot(streams[closest[clustersIndex]][:, 0], streams[closest[clustersIndex]][:, 1], marker=None, color=etcolors[clustersIndex],
            label=len(k_groups[clustersIndex][0]))
plt.legend()
    #cInd = sortInd[clustersIndex]
    #if len(clusters.clusters[cInd]) > 1:
    #    color = randomcolor()
    #    p2.plot(clusters.centroids[cInd][:,0], clusters.centroids[cInd][:,1], marker=None, color=etcolors[clustersIndex],label = len(clusters.clusters[cInd]))  # convert to map projection coordinate
plt.legend()






from dipy.segment.metric import CenterOfMassFeature
feature = CenterOfMassFeature()
from dipy.segment.metric import AveragePointwiseEuclideanMetric
# metric = GPSDistance()
# metric = AveragePointwiseEuclideanMetric(feature)
metric = AveragePointwiseEuclideanMetric()

# qb = QuickBundles(threshold=2750,metric=metric)
# qb = QuickBundles(threshold=1900,metric=metric)
qb = QuickBundles(threshold=450000,metric=metric)
from dipy.tracking.streamline import set_number_of_points
streamlines = set_number_of_points(streams, nb_points=20)

clusters = qb.cluster(streamlines)
print("Nb. clusters:",len(clusters))


numStorms = np.arange(0,len(streams))
# plt.style.use('dark_background')
# fig = plt.figure()
# p1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
# for i in numStorms:
#     p1.plot(streams[int(i)][:, 0], streams[int(i)][:, 1])







etcolors = cm.rainbow(np.linspace(0, 1,len(clusters)))

fig = plt.figure()
p2 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
sortInd = np.flipud(np.argsort(np.asarray([len(ii) for ii in clusters.clusters])))
for clustersIndex in range(len(clusters)):
    cInd = sortInd[clustersIndex]
    if len(clusters.clusters[cInd]) > 1:
        color = randomcolor()
        p2.plot(clusters.centroids[cInd][:,0], clusters.centroids[cInd][:,1], marker=None, color=etcolors[clustersIndex],label = len(clusters.clusters[cInd]))  # convert to map projection coordinate
plt.legend()




import pickle

outPickle = '100simulationsCusteredWP.pickle'
output = {}
output['clusters'] = clusters
output['streams'] = streams
output['collisions'] = collisions
output['streamlines'] = streamlines
output['kma'] = kma
output['num_clusters'] = num_clusters
output['d_groups'] = d_groups

with open(outPickle,'wb') as f:
    pickle.dump(output, f)



