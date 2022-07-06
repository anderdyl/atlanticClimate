import os
import numpy as np
import datetime
from netCDF4 import Dataset
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
import pickle
from scipy.io.matlab.mio5_params import mat_struct
from datetime import datetime, date, timedelta
import random
import itertools
import operator
import scipy.io as sio
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.stats import norm, genpareto, t
from scipy.special import ndtri  # norm inv
import matplotlib.dates as mdates
from scipy.stats import  genextreme, gumbel_l, spearmanr, norm, weibull_min
from scipy.spatial import distance
import pickle
import calendar
import xarray as xr
import pandas


def toTimestamp(d):
    return calendar.timegm(d.timetuple())


with open(r"simulations1000Chopped.pickle", "rb") as input_file:
   simsChoppedInput = pickle.load(input_file)
simBmuLengthChopped = simsChoppedInput['simBmuLengthChopped']
simBmuGroupsChopped = simsChoppedInput['simBmuGroupsChopped']
simBmuChopped = simsChoppedInput['simBmuChopped']

with open(r"gevCopulaSims125000.pickle", "rb") as input_file:
   gevCopulaSimsInput = pickle.load(input_file)
gevCopulaSims = gevCopulaSimsInput['gevCopulaSims']

with open(r"normalizedWaveHydrographs.pickle", "rb") as input_file:
   normalizedWaveHydrographs = pickle.load(input_file)
normalizedHydros = normalizedWaveHydrographs['normalizedHydros']
bmuDataMin = normalizedWaveHydrographs['bmuDataMin']
bmuDataMax = normalizedWaveHydrographs['bmuDataMax']
bmuDataStd = normalizedWaveHydrographs['bmuDataStd']
bmuDataNormalized = normalizedWaveHydrographs['bmuDataNormalized']

with open(r"hydrographCopulaData.pickle", "rb") as input_file:
   hydrographCopulaData = pickle.load(input_file)
copulaData = hydrographCopulaData['copulaData']


dt = datetime(2022, 6, 1, 0, 0, 0)
end = datetime(2122, 5, 31, 23, 0, 0)
step = timedelta(hours=1)
hourlyTime = []
while dt < end:
    hourlyTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step

deltaT = [(tt - hourlyTime[0]).total_seconds() / (3600*24) for tt in hourlyTime]
# # Create datetime objects for each time (a and b)
# dateTimeA = datetime.combine(datetime.date.today(), a)
# dateTimeB = datetime.combine(datetime.date.today(), b)
# # Get the difference between datetimes (as timedelta)
# dateTimeDifference = dateTimeA - dateTimeB
# # Divide difference in seconds by number of seconds in hour (3600)
# dateTimeDifferenceInHours = dateTimeDifference.total_seconds() / 3600


def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index], closest_index


simulationsHs = list()
simulationsTp = list()
simulationsDm = list()
simulationsSs = list()
simulationsTime = list()

for simNum in range(500):

    simHs = []
    simTp = []
    simDm = []
    simSs = []
    simTime = []
    print('filling in simulation #{}'.format(simNum))

    for i in range(len(simBmuChopped[simNum])):
        if np.remainder(i,1000) == 0:
            print('done with {} hydrographs'.format(i))
        tempBmu = int(simBmuChopped[simNum][i]-1)
        randStorm = random.randint(0, 9999)
        stormDetails = gevCopulaSims[tempBmu][randStorm]
        if stormDetails[0] > 10:
            print('oh boy, we''ve picked a {}m storm wave in BMU #{}'.format(stormDetails[0],tempBmu))
        durSim = simBmuLengthChopped[simNum][i]

        simDmNorm = (stormDetails[4] - np.asarray(bmuDataMin)[tempBmu,0]) / (np.asarray(bmuDataMax)[tempBmu,0]-np.asarray(bmuDataMin)[tempBmu,0])
        simSsNorm = (stormDetails[5] - np.asarray(bmuDataMin)[tempBmu,1]) / (np.asarray(bmuDataMax)[tempBmu,1]-np.asarray(bmuDataMin)[tempBmu,1])
        test, closeIndex = closest_node([simDmNorm,simSsNorm],np.asarray(bmuDataNormalized)[tempBmu])
        actualIndex = int(np.asarray(copulaData[tempBmu])[closeIndex,6])

        tempHs = ((normalizedHydros[tempBmu][actualIndex]['hsNorm']) * (stormDetails[0]-stormDetails[1]) + stormDetails[1]).filled()
        tempTp = ((normalizedHydros[tempBmu][actualIndex]['tpNorm']) * (stormDetails[2]-stormDetails[3]) + stormDetails[3]).filled()
        tempDm = ((normalizedHydros[tempBmu][actualIndex]['dmNorm']) + stormDetails[4])
        tempSs = ((normalizedHydros[tempBmu][actualIndex]['ssNorm']) + stormDetails[5])
        if len(normalizedHydros[tempBmu][actualIndex]['hsNorm']) < len(normalizedHydros[tempBmu][actualIndex]['timeNorm']):
            print('Time is shorter than Hs in bmu {}, index {}'.format(tempBmu,actualIndex))
        if stormDetails[1] < 0:
            print('woah, we''re less than 0 over here')
            asdfg
        if len(tempSs) < len(normalizedHydros[tempBmu][actualIndex]['timeNorm']):
            # print('Ss is shorter than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
            tempLength = len(normalizedHydros[tempBmu][actualIndex]['timeNorm'])
            tempSs = np.zeros((len(normalizedHydros[tempBmu][actualIndex]['timeNorm']),))
            tempSs[0:len((normalizedHydros[tempBmu][actualIndex]['ssNorm']) + stormDetails[5])] = ((normalizedHydros[tempBmu][actualIndex]['ssNorm']) + stormDetails[5])
        if len(tempSs) > len(normalizedHydros[tempBmu][actualIndex]['timeNorm']):
            # print('Now Ss is longer than Time in bmu {}, index {}'.format(tempBmu,actualIndex))
            # print('{} vs. {}'.format(len(tempSs),len(normalizedHydros[tempBmu][actualIndex]['timeNorm'])))
            tempSs = tempSs[0:-1]

        simHs.append(tempHs)
        simTp.append(tempTp)
        simDm.append(tempDm)
        simSs.append(tempSs)
        #simTime.append(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
        #dt = np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
        simTime.append(np.hstack((np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim), np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)[-1])))


    #
    # asdfg
    #
    #
    #
    # simulationsHs.append(np.hstack(simHs))
    # simulationsTp.append(np.hstack(simTp))
    # simulationsDm.append(np.hstack(simDm))
    # simulationsSs.append(np.hstack(simSs))
    cumulativeHours = np.cumsum(np.hstack(simTime))
    newDailyTime = [datetime(2022, 6, 1) + timedelta(days=ii) for ii in cumulativeHours]
    simDeltaT = [(tt - newDailyTime[0]).total_seconds() / (3600 * 24) for tt in newDailyTime]

    # simulationsTime.append(newDailyTime)
    # rng = newDailyTime
    #
    simData = np.array(np.vstack((np.hstack(simHs).T,np.hstack(simTp).T,np.hstack(simDm).T,np.hstack(simSs).T)))
    # simData = np.array((np.ma.asarray(np.hstack(simHs)),np.ma.asarray(np.hstack(simTp)),np.ma.asarray(np.hstack(simDm)),np.ma.asarray(np.hstack(simSs))))
    # simData = np.array([np.hstack(simHs).filled(),np.hstack(simTp).filled(),np.hstack(simDm).filled(),np.hstack(simSs)])

    ogdf = pandas.DataFrame(data=simData.T,index=newDailyTime,columns=["hs","tp","dm","ss"])

    print('interpolating')
    interpHs = np.interp(deltaT,simDeltaT,np.hstack(simHs))
    interpTp = np.interp(deltaT,simDeltaT,np.hstack(simTp))
    interpDm = np.interp(deltaT,simDeltaT,np.hstack(simDm))
    interpSs = np.interp(deltaT,simDeltaT,np.hstack(simSs))

    simDataInterp = np.array([interpHs,interpTp,interpDm,interpSs])

    df = pandas.DataFrame(data=simDataInterp.T,index=hourlyTime,columns=["hs","tp","dm","ss"])
    # resampled = df.resample('H')
    # interped = resampled.interpolate()
    # simulationData = interped.values
    # testTime = interped.index  # to_pydatetime()
    # testTime2 = testTime.to_pydatetime()

    # simsPickle = ('/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle'.format(simNum))
    simsPickle = ('/media/dylananderson/Elements/Sims/simulation{}.pickle'.format(simNum))

    outputSims= {}
    outputSims['simulationData'] = simDataInterp.T
    outputSims['df'] = df
    outputSims['simHs'] = np.hstack(simHs)
    outputSims['simTp'] = np.hstack(simTp)
    outputSims['simDm'] = np.hstack(simDm)
    outputSims['simSs'] = np.hstack(simSs)
    outputSims['time'] = hourlyTime

    with open(simsPickle, 'wb') as f:
        pickle.dump(outputSims, f)

    #
    # # ts = pandas.Series(np.hstack(simHs), index=newDailyTime)
    # # resampled = ts.resample('H')
    # # interp = resampled.interpolate()
    #
    # testTime = interped.index  # to_pydatetime()
    # testTime2 = testTime.to_pydatetime()
    # testData = interped.values



# simsPickle = '/home/dylananderson/projects/atlanticClimate/Sims/allSimulations.pickle'
# outputSims= {}
# outputSims['simulationsHs'] = simulationsHs
# outputSims['simulationsTime'] = simulationsTime
# outputSims['simulationsTp'] = simulationsTp
# outputSims['simulationsDm'] = simulationsDm
# outputSims['simulationsSs'] = simulationsSs
#
# with open(simsPickle, 'wb') as f:
#     pickle.dump(outputSims, f)


plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
# ax1.plot(newDailyTime,simulationsHs[-1])
# ax1.plot(newDailyTime,)
ax1.plot(hourlyTime,simDataInterp[0,:])
### TODO: Need to assess the statistics of these hypothetical scenarios... Yearly max Hs? Wave Energy?

### TODO: Which requires interpolating the time series to hourly values...

# for qq in len(simulationsTime):

