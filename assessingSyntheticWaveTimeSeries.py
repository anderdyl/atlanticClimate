import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

with open(r"realWaves.pickle", "rb") as input_file:
   wavesInput = pickle.load(input_file)
tWave = wavesInput['tWave'][5:]
tC = wavesInput['tC'][5:]
hsCombined = wavesInput['hsCombined'][5:]
tpCombined = wavesInput['tpCombined'][5:]
dmCombined = wavesInput['dmCombined'][5:]
waveNorm = wavesInput['waveNorm']
wlFRF = wavesInput['wlFRF']
tFRF = wavesInput['tFRF']
resFRF = wavesInput['resFRF']

data = np.array([hsCombined,tpCombined,dmCombined])
ogdf = pd.DataFrame(data=data.T, index=tC, columns=["hs", "tp", "dm"])
year = np.array([tt.year for tt in tC])
ogdf['year'] = year
month = np.array([tt.month for tt in tC])
ogdf['month'] = month


dailyMaxHs = ogdf.resample("d")['hs'].max()

c = 0
twoDayMax = []
while c < len(hsCombined):
    twoDayMax.append(np.nanmax(hsCombined[c:c+48]))
    c = c + 48
twoDayMaxHs = np.asarray(twoDayMax)

c = 0
threeDayMax = []
while c < len(hsCombined):
    threeDayMax.append(np.nanmax(hsCombined[c:c+72]))
    c = c + 72
threeDayMaxHs = np.asarray(threeDayMax)

c = 0
fourDayMax = []
while c < len(hsCombined):
    fourDayMax.append(np.nanmax(hsCombined[c:c+96]))
    c = c + 96
fourDayMaxHs = np.asarray(fourDayMax)


seasonalMean = ogdf.groupby('month').mean()
seasonalStd = ogdf.groupby('month').std()
yearlyMax = ogdf.groupby('year').max()
g2 = ogdf.groupby(pd.Grouper(freq="M")).mean()

simSeasonalMean = np.nan * np.ones((50,12))
simSeasonalStd = np.nan * np.ones((50,12))
simYearlyMax = np.nan * np.ones((50,101))

for hh in range(1):
   # file = r"/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle".format(hh)
   file = r"/media/dylananderson/Elements/Sims/simulation{}.pickle".format(hh)

   with open(file, "rb") as input_file:
      simsInput = pickle.load(input_file)
   simulationData = simsInput['simulationData']
   df = simsInput['df']
   time = simsInput['time']
   year = np.array([tt.year for tt in time])
   df['year'] = year
   month = np.array([tt.month for tt in time])
   df['month'] = month

   g1 = df.groupby(pd.Grouper(freq="M")).mean()
   simSeasonalMean[hh,:] = df.groupby('month').mean()['hs']
   simSeasonalStd[hh,:] = df.groupby('month').std()['hs']
   simYearlyMax[hh,:] = df.groupby('year').max()['hs']


dailyMaxHsSim = df.resample("d")['hs'].max()


c = 0
twoDayMaxSim = []
while c < len(simulationData):
    twoDayMaxSim.append(np.nanmax(simulationData[c:c+48,0]))
    c = c + 48
twoDayMaxHsSim = np.asarray(twoDayMaxSim)

c = 0
threeDayMaxSim = []
while c < len(simulationData):
    threeDayMaxSim.append(np.nanmax(simulationData[c:c+72,0]))
    c = c + 72
threeDayMaxHsSim = np.asarray(threeDayMaxSim)

c = 0
fourDayMaxSim = []
while c < len(simulationData):
    fourDayMaxSim.append(np.nanmax(simulationData[c:c+96,0]))
    c = c + 96
fourDayMaxHsSim = np.asarray(fourDayMaxSim)


import more_itertools as mit


tC = np.array(time)
hsSmooth = simulationData[:,0]
tpSmooth = simulationData[:,1]
dmSmooth = simulationData[:,2]

stormHsInd = np.where((hsSmooth > 3))
stormHsList = [list(group) for group in mit.consecutive_groups(stormHsInd[0])]


hsStormList = []
hsStormMaxList = []
hsStormMinList = []
tpStormList = []
tpStormMaxList = []
tpStormMinList = []
dmStormList = []
ntrStormList = []
timeStormList = []
hourStormList = []
indStormList = []
#indNTRStormList = []
#bmuStormList = []
for xx in range(len(stormHsList)-2):

    i1 = stormHsList[xx][0]
    i2 = stormHsList[xx][-1]
    t1 = tC[i1]
    t2 = tC[i2]
    nexti1 = stormHsList[xx+1][0]
    diff = nexti1 - i2
    # if tC[i1] > datetime.datetime(2019,1,1):
    #     numToBeat =
    if diff < 24:
        i2 = stormHsList[xx+1][-1]
        t2 = tC[i2]
        nexti1 = stormHsList[xx+2][0]
        diff2 = nexti1-i2
        if diff2 < 24:
            i2 = stormHsList[xx + 2][-1]
            t2 = tC[i2]

    tempWave = np.where((tC < t2) & (tC > t1))
    if len(tempWave[0]) > 12:
        # t1 = tC[i1-12]
        # t2 = tC[i2+12]
        t1 = tC[i1-1]
        t2 = tC[i2+1]
        #t3 = tC[i2+36]
        #t4 = tC[i1-18]

        tempWave = np.where((tC < t2) & (tC > t1))
        #tempWaterLevel = np.where((tWaterLevelFRF < t2) & (tWaterLevelFRF > t1))
        #tempBMU = np.where((dwtTimes < t3) & (dwtTimes > t4))
        #indices2 = tempWaterLevel[0]
        # indices = np.arange(i1-12,i2+12)
        indices = np.arange(i1,i2)


        #bmuStormList.append(dwtBMUS[tempBMU])
        hsStormList.append(hsSmooth[tempWave])
        #ntrStormList.append(residualWaterLevelFRF[tempWaterLevel])
        # hsMaxList = np.append(hsMaxList,np.nanmax(hsCombined[tempWave]))
        # hsMinList = np.append(hsMinList,np.nanmin(hsCombined[tempWave]))
        tpStormList.append(tpSmooth[tempWave])
        # tpMaxList = np.append(tpMaxList,np.nanmax(tpCombined[tempWave]))
        # tpMinList = np.append(tpMinList,np.nanmin(tpCombined[tempWave]))
        # dmStormList.append(dmCombined[tempWave])
        dmStormList.append(dmSmooth[tempWave])

        timeStormList.append(tC[tempWave])
        indStormList.append(indices)
        #indNTRStormList.append(indices2)



hsMaxStorm = []
hsMaxTimeInStorm = []
# bmuMaxTimeInStorm = []
durationHoursStorm = []
dmAvgStorm = []
# ntrMaxStorm = []
tpMaxStorm = []
timeStorm = []
for x in range(len(hsStormList)):
    tempHs = hsStormList[x]
    tempTp = tpStormList[x]
    timeStorm.append(timeStormList[x][12])
    tempMaxInds = np.where((np.nanmax(tempHs) == tempHs))
    hsMaxStorm.append(np.nanmax(tempHs))
    dayOf = int(np.floor(tempMaxInds[0][-1]/24))
    hsMaxTimeInStorm.append(tempMaxInds[0][-1])
    tpMaxStorm.append(tempTp[tempMaxInds[0][-1]])
    #bmuMaxTimeInStorm.append(bmuStormList[x][dayOf])
    durationHoursStorm.append(len(tempHs))
    tempDm = dmStormList[x]
    dmAvgStorm.append(np.nanmean(tempDm))
    # tempNTR = ntrStormList[x]
    # if len(tempNTR) > 0:
    #     ntrMaxStorm.append(np.nanmax(tempNTR))
    # else:
    #     ntrMaxStorm.append(np.nan)



plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
ax1.plot(tC,hsSmooth)
for hh in range(len(hsStormList)):
    ax1.plot(timeStormList[hh],hsStormList[hh],'r')




from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

base = importr('base')
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('POT') #installing POT package
from thresholdmodeling import thresh_modeling #importing package
import pandas as pd #importing pandas

#url = 'https://raw.githubusercontent.com/iagolemos1/thresholdmodeling/master/dataset/rain.csv' #saving url
#df =  pd.read_csv(url, error_bad_lines=False) #getting data
# data = df['hs'].values.ravel() #turning data into an array

#data =

# thresh_modeling.MRL(data, 0.05)
# thresh_modeling.Parameter_Stability_plot(data, 0.05)
# thresh_modeling.return_value(data, 30, 0.05, 365, 36500, 'mle')
data = np.asarray(hsMaxStorm)
# dataDecluster, data2 = thresh_modeling.decluster(data,3,24*2)
thresh_modeling.return_value(np.asarray(dailyMaxHs.values), 4, 0.05, 365, 36500, 'mle')
thresh_modeling.return_value(np.asarray(dailyMaxHsSim.values), 4, 0.05, 365, 36500, 'mle')

thresh_modeling.return_value(twoDayMaxHs, 2.5, 0.05, 365/2, 36500/2, 'mle')
thresh_modeling.return_value(twoDayMaxHsSim, 2.5, 0.05, 365/2, 36500/2, 'mle')


thresh_modeling.return_value(threeDayMaxHs, 2.5, 0.05, 365/3, 36500/3, 'mle')
thresh_modeling.return_value(threeDayMaxHsSim, 2.5, 0.05, 365/3, 36500/3, 'mle')


thresh_modeling.return_value(fourDayMaxHs, 3, 0.05, 365/4, 36500/4, 'mle')
thresh_modeling.return_value(fourDayMaxHsSim, 3, 0.05, 365/4, 36500/4, 'mle')

example = thresh_modeling.gpdfit(fourDayMaxHs, 3, 'mle')
plt.figure()
plt.plot(example[3],example[4],'-')

