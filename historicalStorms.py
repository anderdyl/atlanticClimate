import os
from netCDF4 import Dataset
import numpy as np
import more_itertools as mit
import matplotlib.pyplot as plt
import datetime as DT
from matplotlib import cm

waterLevelDir = '/media/dylananderson/Elements/frfWaterLevel'
files = os.listdir(waterLevelDir)
files.sort()
files_path = [os.path.join(os.path.abspath(waterLevelDir), x) for x in files]
wls = Dataset(files_path[0])

def getWaterLevel(file):
    wldata = Dataset(file)
    waterLevel = wldata.variables['waterLevel'][:]
    predictedWaterLevel = wldata.variables['predictedWaterLevel'][:]
    residualWaterLevel = wldata.variables['residualWaterLevel'][:]
    timeWl = wldata.variables['time'][:]
    output = dict()
    output['waterLevel'] = waterLevel
    output['predictedWaterLevel'] = predictedWaterLevel
    output['residualWaterLevel'] = residualWaterLevel
    output['time'] = timeWl
    return output


#### WATER LEVELS ARE IN NAVD88 ##### - 0.128
timeWaterLevelFRF = []
waterLevelFRF = []
predictedWaterLevelFRF = []
residualWaterLevelFRF = []
for i in files_path:
    waterLevels = getWaterLevel(i)
    waterLevelFRF = np.append(waterLevelFRF,waterLevels['waterLevel']+0.128)
    predictedWaterLevelFRF = np.append(predictedWaterLevelFRF,waterLevels['predictedWaterLevel']+0.128)
    residualWaterLevelFRF = np.append(residualWaterLevelFRF,waterLevels['residualWaterLevel'])
    timeWaterLevelFRF = np.append(timeWaterLevelFRF,waterLevels['time'].flatten())



badWaterLevel = np.where((residualWaterLevelFRF < -99))
tWaterLevelFRF = [DT.datetime.fromtimestamp(x) for x in timeWaterLevelFRF]
tWaterLevelFRF = np.asarray(tWaterLevelFRF)

goodWaterLevel = np.where((waterLevelFRF > -99))
wlFRF = waterLevelFRF[goodWaterLevel]
tFRF = tWaterLevelFRF[goodWaterLevel]

tBefore = np.where((tFRF > DT.datetime(1995,10,1)))

wl = np.hstack((wlFRF[0:tBefore[0][0]-1],wlFRF[tBefore[0][0]-1::10]))
wlTime = np.hstack((tFRF[0:tBefore[0][0]-1],tFRF[tBefore[0][0]-1::10]))

waterLevelFRF[badWaterLevel] = waterLevelFRF[badWaterLevel]*np.nan
predictedWaterLevelFRF[badWaterLevel] = predictedWaterLevelFRF[badWaterLevel]*np.nan
residualWaterLevelFRF[badWaterLevel] = residualWaterLevelFRF[badWaterLevel]*np.nan


wavedir = '/media/dylananderson/Elements/WIS_ST63218/'

# Need to sort the files to ensure correct temporal order...
files = os.listdir(wavedir)
files.sort()
files_path = [os.path.join(os.path.abspath(wavedir), x) for x in files]

wis = Dataset(files_path[0])

def getWIS(file):
    waves = Dataset(file)

    waveHs = waves.variables['waveHs'][:]
    waveTp = waves.variables['waveTp'][:]
    waveMeanDirection = waves.variables['waveMeanDirection'][:]

    waveTm = waves.variables['waveTm'][:]
    waveTm1 = waves.variables['waveTm1'][:]
    waveTm2 = waves.variables['waveTm2'][:]

    waveHsWindsea = waves.variables['waveHsWindsea'][:]
    waveTmWindsea = waves.variables['waveTmWindsea'][:]
    waveMeanDirectionWindsea = waves.variables['waveMeanDirectionWindsea'][:]
    waveSpreadWindsea = waves.variables['waveSpreadWindsea'][:]

    timeW = waves.variables['time'][:]

    waveTpSwell = waves.variables['waveTpSwell'][:]
    waveHsSwell = waves.variables['waveHsSwell'][:]
    waveMeanDirectionSwell = waves.variables['waveMeanDirectionSwell'][:]
    waveSpreadSwell = waves.variables['waveSpreadSwell'][:]


    output = dict()
    output['waveHs'] = waveHs
    output['waveTp'] = waveTp
    output['waveMeanDirection'] = waveMeanDirection

    output['waveTm'] = waveTm
    output['waveTm1'] = waveTm1
    output['waveTm2'] = waveTm2

    output['waveTpSwell'] = waveTpSwell
    output['waveHsSwell'] = waveHsSwell
    output['waveMeanDirectionSwell'] = waveMeanDirectionSwell
    output['waveSpreadSwell'] = waveSpreadSwell

    output['waveHsWindsea'] = waveHsWindsea
    output['waveTpWindsea'] = waveTmWindsea
    output['waveMeanDirectionWindsea'] = waveMeanDirectionWindsea
    output['waveSpreadWindsea'] = waveSpreadWindsea

    output['t'] = timeW

    return output

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



Hs = []
Tp = []
Dm = []
hsSwell = []
tpSwell = []
dmSwell = []
hsWindsea = []
tpWindsea = []
dmWindsea = []

timeWave = []
for i in files_path:
    waves = getWIS(i)
    Hs = np.append(Hs,waves['waveHs'])
    Tp = np.append(Tp,waves['waveTp'])
    Dm = np.append(Dm,waves['waveMeanDirection'])
    hsSwell = np.append(hsSwell,waves['waveHsSwell'])
    tpSwell = np.append(tpSwell,waves['waveTpSwell'])
    dmSwell = np.append(dmSwell,waves['waveMeanDirectionSwell'])
    hsWindsea = np.append(hsWindsea,waves['waveHsWindsea'])
    tpWindsea = np.append(tpWindsea,waves['waveTpWindsea'])
    dmWindsea = np.append(dmWindsea,waves['waveMeanDirectionWindsea'])
    #timeTemp = [datenum_to_datetime(x) for x in waves['t'].flatten()]
    timeWave = np.append(timeWave,waves['t'].flatten())

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n




hsCombined = Hs #np.append(Hs,hs26m)
hsSmooth = moving_average(Hs,3)+0.05
tpCombined = Tp #np.append(Tp,tp26m)
dmCombined = Dm #np.append(Dm,dm26m)
tWave = [DT.datetime.fromtimestamp(x) for x in timeWave]
# tWave = [datetime.fromtimestamp(x) for x in timeWave]
tC = np.array(tWave) #np.append(np.array(tWave),tWave26m)

# badDirs = np.where((dmCombined > 360))
# dmCombined[badDirs] = dmCombined[badDirs]*np.nan

waveNorm = dmCombined - 72
neg = np.where((waveNorm > 180))
waveNorm[neg[0]] = waveNorm[neg[0]]-360
offpos = np.where((waveNorm>90))
offneg = np.where((waveNorm<-90))
waveNorm[offpos[0]] = waveNorm[offpos[0]]*0
waveNorm[offneg[0]] = waveNorm[offneg[0]]*0



Bf = 0.083
L0 = 9.81 * np.square(tpCombined) / (2 * np.pi)
H0 = hsCombined
R2 = 1.1 * ((0.35 * Bf * np.power(np.multiply(H0, L0), 0.5)) + np.divide(
    np.power((H0 * L0 * (0.563 * (np.power(Bf, 2)) + 0.004)), (0.5)), 2))

overlapInd = np.where((tC > DT.datetime(1980,12,31,19,0,0)))
overlapInd2 = np.where((wlTime < DT.datetime(2019,12,31,20,0,0)))


interpWaveT = [(tt - tC[0]).total_seconds() / (3600 * 24) for tt in tC]
interpWaterT = [(tt - tC[0]).total_seconds() / (3600 * 24) for tt in wlTime]

#
#
print('interpolating')
interpWL = np.interp(interpWaveT, interpWaterT, wl)


TWL = interpWL + R2
hAbove = np.where((TWL > 4))


#plt.figure()
#plt.plot(tC,)



avgHs = np.nanmean(hsCombined)
hs98 = np.nanpercentile(hsCombined,98)
hs95 = np.nanpercentile(hsCombined,95)
hs90 = np.nanpercentile(hsCombined,90)
hs85 = np.nanpercentile(hsCombined,85)

stormHsInd = np.where((hsSmooth > hs90))
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

    # Don't need to care about the next storm in this workflow...
    #nexti1 = stormHsList[xx+1][0]
    #diff = nexti1 - i2
    # if tC[i1] > datetime.datetime(2019,1,1):
    #     numToBeat =
    # if diff < 24:
    #     i2 = stormHsList[xx+1][-1]
    #     t2 = tC[i2]
    #     nexti1 = stormHsList[xx+2][0]
    #     diff2 = nexti1-i2
    #     if diff2 < 24:
    #         i2 = stormHsList[xx + 2][-1]
    #         t2 = tC[i2]

    tempWave = np.where((tC < t2) & (tC > t1))
    if len(tempWave[0]) > 10:

        # # do we need waves to either side of the storm?
        # t1 = tC[i1-12]
        # t2 = tC[i2+12]

        # t3 = tC[i2+36]
        # t4 = tC[i1-18]

        tempWave = np.where((tC < t2) & (tC > t1))
        tempWaterLevel = np.where((wlTime < t2) & (wlTime > t1))
        #tempBMU = np.where((dwtTimes < t3) & (dwtTimes > t4))
        indices2 = tempWaterLevel[0]

        # indices = np.arange(i1-12,i2+12)
        indices = np.arange(i1,i2)


        #bmuStormList.append(dwtBMUS[tempBMU])
        hsStormList.append(hsCombined[tempWave])

        #ntrStormList.append(residualWaterLevelFRF[tempWaterLevel])

        wlStormList.append(wl[tempWaterLevel])

        # hsMaxList = np.append(hsMaxList,np.nanmax(hsCombined[tempWave]))
        # hsMinList = np.append(hsMinList,np.nanmin(hsCombined[tempWave]))
        tpStormList.append(tpCombined[tempWave])
        # tpMaxList = np.append(tpMaxList,np.nanmax(tpCombined[tempWave]))
        # tpMinList = np.append(tpMinList,np.nanmin(tpCombined[tempWave]))
        # dmStormList.append(dmCombined[tempWave])
        dmStormList.append(waveNorm[tempWave])

        timeStormList.append(tC[tempWave])
        timeStartList.append(t1)
        timeEndList.append(t2)

        indStormList.append(indices)
        indNTRStormList.append(indices2)


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


plt.figure()
plt.plot(filteredTime,filteredHs,'o')

numDaysBetween = 4
windowOfStorms = []
counter = 0
counter2 = 0
counter3 = 0
counter4 = 0
storms2 = []
storms3 = []
storms4 = []
storms5 = []
for ii in range(len(filteredTime)-2):
    indexNear = np.where((filteredStartTime > filteredEndTime[ii]) & (filteredStartTime < filteredEndTime[ii]+timedelta(days=30)))
    windowOfStorms.append(indexNear)
    timeDiff = filteredStartTime[ii+1]-filteredEndTime[ii]
    #print(timeDiff)
    if timeDiff.days < numDaysBetween:
        counter = counter + 1
        #print('3 days or less: {} times'.format(counter))
        timeDiff2 = filteredStartTime[ii+2]-filteredEndTime[ii+1]
        storms2.append([ii,ii+1])
        if timeDiff2.days < numDaysBetween:
            counter2 = counter2 + 1
            #print('alright, found a three storm cluster: {} times'.format(counter2))
            timeDiff3 = filteredStartTime[ii+3]-filteredEndTime[ii+2]
            storms3.append([ii, ii + 1, ii + 2])
            if timeDiff3.days < numDaysBetween:
                counter3 = counter3 + 1
                storms4.append([ii, ii + 1, ii + 2, ii + 3])
                timeDiff4 = filteredStartTime[ii + 4] - filteredEndTime[ii + 3]
                #print('alright, found a four storm cluster: {} times'.format(counter3))
                if timeDiff4.days < numDaysBetween:
                    counter4 = counter4 + 1
                    storms5.append([ii, ii + 1, ii + 2, ii + 3, ii + 4])
                    print('alright, found a five storm cluster: {} times'.format(counter4))



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
stormSequences = storms5
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

# lets find the start time and end time of each cluster
plt.figure()
Bf = 0.1
duneToe = 3
hmax = 25
ax1 = plt.subplot2grid((8,8),(0,0),rowspan=8,colspan=4)
ax3 = plt.subplot2grid((8,8),(4,4),rowspan=4,colspan=1)
ax4 = plt.subplot2grid((8,8),(4,5),rowspan=4,colspan=1)
ax5 = plt.subplot2grid((8,8),(4,6),rowspan=4,colspan=1)
ax6 = plt.subplot2grid((8,8),(4,7),rowspan=4,colspan=1)

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
    wIndices = np.where((wlTime > sTime) & (wlTime < eTime))
    timeDeltsW = wlTime[wIndices]-wlTime[wIndices[0][0]]
    timeIntW = [(pp.seconds/(60*60*24)+pp.days) for pp in timeDeltsW]
    wlsub = wl[wIndices]

    if len(R2) > len(wlsub):
        TWL = wlsub + R2[0:len(wlsub)]
        wp = wp[0:len(wlsub)]
        timeSub = timeIntW
    else:
        TWL = wlsub[0:len(R2)] + R2
        timeSub = timeInt

    highWL = np.where((TWL > duneToe))
    zerosDune = np.zeros((np.size(TWL)))
    zerosDune[highWL] = np.ones((np.size(highWL),))
    collisionDune = np.cumsum(zerosDune)
    ax1.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')

    if len(highWL[0]) > 0:
        print('stormSequences {} has {}'.format(ii,len(highWL[0])))
    #plt.scatter(timeSub,wp,marker='.',c=cm.Blues(collisionDune),vmin=0,vmax=30,edgecolor='none')
    # plt.plot(timeSub,np.cumsum(TWL),'b-')
    # plt.plot(timeSub,(TWL),'r-')
    # plt.plot(timeInt,wp,'b-')
    monthTemp=sTime.month
    if monthTemp > 2 and monthTemp < 6:
        #print('in spring time, but not plotting?')

        ax6.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
    elif monthTemp > 5 and monthTemp < 9:
        #print('summer storm... skipping')
        ax3.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')

    elif monthTemp > 8 and monthTemp < 12:
        #print('in fall time, but not plotting?')

        ax4.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
    else:
        #print('in winter time, but not plotting?')

        ax5.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')


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
    wIndices = np.where((wlTime > sTime) & (wlTime < eTime))
    timeDeltsW = wlTime[wIndices]-wlTime[wIndices[0][0]]
    timeIntW = [(pp.seconds/(60*60*24)+pp.days) for pp in timeDeltsW]
    wlsub = wl[wIndices]

    if len(R2) > len(wlsub):
        TWL = wlsub + R2[0:len(wlsub)]
        wp = wp[0:len(wlsub)]
        timeSub = timeIntW
    else:
        TWL = wlsub[0:len(R2)] + R2
        timeSub = timeInt

    highWL = np.where((TWL > duneToe))
    zerosDune = np.zeros((np.size(TWL)))
    zerosDune[highWL] = np.ones((np.size(highWL),))
    collisionDune = np.cumsum(zerosDune)
    if len(highWL[0]) > 0:
        print('singleStorms {} has {}'.format(ii,len(highWL[0])))
    #plt.scatter(timeSub,wp,marker='.',c=cm.YlOrRd(collisionDune),vmin=0,vmax=20,edgecolor='none')
    p1 = ax1.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
    # plt.plot(timeSub,(TWL),'r-')
    # plt.plot(timeSub,np.cumsum(TWL),'r-')
    #plt.plot(timeInt,wp,'r-')
    # print(sTime.month)
    monthTemp = sTime.month
    if monthTemp > 2 and monthTemp < 6:
        ax6.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
    elif monthTemp> 5 and monthTemp < 9:
        #print('summer storm... skipping')
        ax3.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')

    elif monthTemp > 8 and monthTemp < 12:
        #print('in fall time, but not plotting?')

        ax4.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')
    else:
        #print('in winter time, but not plotting?')

        ax5.scatter(timeSub,wp,marker='.',c=collisionDune,vmin=0,vmax=hmax,edgecolor='none')





ii = 32
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
R2 = 1.1*((0.35*Bf*np.power(np.multiply(H0,L0),(1/2))) + (np.power((H0*L0*(0.563*(np.power(Bf,2)) + 0.004)),(1/2))/2))
# NEED TO GET WATER LEVELS #
wIndices = np.where((wlTime > sTime) & (wlTime < eTime))
timeDeltsW = wlTime[wIndices]-wlTime[wIndices[0][0]]
timeIntW = [(pp.seconds/(60*60*24)+pp.days) for pp in timeDeltsW]
wlsub = wl[wIndices]

if len(R2) > len(wlsub):
    TWL = wlsub + R2[0:len(wlsub)]
    wp = wp[0:len(wlsub)]
    wlnew = wlsub
    timeSub = timeIntW
else:
    TWL = wlsub[0:len(R2)] + R2
    wlnew = wlsub[0:len(R2)]
    timeSub = timeInt

highWL = np.where((TWL > 4))
zerosDune = np.zeros((np.size(TWL)))
zerosDune[highWL] = np.ones((np.size(highWL),))
collisionDune = np.cumsum(zerosDune)

ax2 = plt.subplot2grid((8,8),(0,4),rowspan=4,colspan=4)

ax2.plot(timeSub,TWL,label='SWL+R2')
ax2.plot(timeSub,wlnew,'--',label='SWL')
ax2.plot(timeSub,duneToe*np.ones((np.size(timeSub))),'k--')

ax3.set_title('Summer')
ax3.set_xlabel('Days')
ax3.set_xlim([0,20])
ax3.set_ylim([0,1.4e7])
ax4.set_title('Fall')
ax4.set_xlabel('Days')
ax4.set_xlim([0,20])
ax4.set_ylim([0,1.4e7])
ax5.set_title('Winter')
ax5.set_xlabel('Days')
ax5.set_xlim([0,20])
ax5.set_ylim([0,1.4e7])
ax6.set_title('Spring')
ax6.set_xlabel('Days')
ax6.set_xlim([0,20])
ax6.set_ylim([0,1.4e7])
ax2.set_xlabel('Days during example cluster')
ax2.set_title('Example Storm Sequence')
ax1.set_xlabel('Days since beginning of event')
ax2.set_ylabel('Water Level (m)')
ax1.set_ylabel('Cumulative Wave Power')
ax1.set_xlim([0,20])
cb1 = plt.colorbar(p1,ax=ax1)
cb1.set_label('hours of dune collision')

