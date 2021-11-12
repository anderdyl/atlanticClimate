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
from scipy.stats import gumbel_l, genextreme
from scipy.spatial import distance








with open(r"waveHydrographs.pickle", "rb") as input_file:
   waveHydrographs = pickle.load(input_file)

hydros = waveHydrographs['hydros']


with open(r"hydrographCopulaData.pickle", "rb") as input_file:
   hydrographCopulaData = pickle.load(input_file)

copulaData = hydrographCopulaData['copulaData']

with open(r"historicalData.pickle", "rb") as input_file:
   historicalData = pickle.load(input_file)

grouped = historicalData['grouped']
groupLength = historicalData['groupLength']
bmuGroup = historicalData['bmuGroup']
timeGroup = historicalData['timeGroup']

with open(r"copulaSamplesTest.pickle", "rb") as input_file:
   copulaSamplesTest = pickle.load(input_file)

gevCopulaSims = copulaSamplesTest['gevCopulaSims']


with open(r"normalizedWaveHydrographs.pickle", "rb") as input_file:
   normalizedWaveHydrographs = pickle.load(input_file)

normalizedHydros = normalizedWaveHydrographs['normalizedHydros']
bmuDataMin = normalizedWaveHydrographs['bmuDataMin']
bmuDataMax = normalizedWaveHydrographs['bmuDataMax']
bmuDataStd = normalizedWaveHydrographs['bmuDataStd']
bmuDataNormalized = normalizedWaveHydrographs['bmuDataNormalized']




### TODO: use the historical to develop a new chronology

numRealizations = 50

simBmuChopped = []
simBmuLengthChopped = []
simBmuGroupsChopped = []
for pp in range(numRealizations):


    simGroupLength = []
    simGrouped = []
    simBmu = []
    for i in range(len(groupLength)):
        tempGrouped = grouped[i]
        tempBmu = int(bmuGroup[i])
        if groupLength[i] > 5:
            # random days between 3 and 5
            randLength = random.randint(1, 3) + 2
            remainingDays = groupLength[i] - randLength
            # add this to the record
            simGroupLength.append(int(randLength))
            simGrouped.append(grouped[i][0:randLength])
            simBmu.append(tempBmu)
            # remove those from the next step
            tempGrouped = np.delete(tempGrouped,np.arange(0,randLength))

            if remainingDays > 5:
                randLength2 = random.randint(1, 3) + 2
                remainingDays2 = remainingDays - randLength2

                simGroupLength.append(int(randLength2))
                simGrouped.append(tempGrouped[0:randLength2])
                simBmu.append(tempBmu)
                # remove those from the next step
                tempGrouped = np.delete(tempGrouped, np.arange(0, randLength2))

                if remainingDays2 > 5:
                    randLength3 = random.randint(1, 3) + 2
                    remainingDays3 = remainingDays2 - randLength3

                    simGroupLength.append(int(randLength3))
                    simGrouped.append(tempGrouped[0:randLength3])
                    simBmu.append(tempBmu)
                    # remove those from the next step
                    tempGrouped = np.delete(tempGrouped, np.arange(0, randLength3))

                    if remainingDays3 > 5:
                        randLength4 = random.randint(1, 3) + 2
                        remainingDays4 = remainingDays3 - randLength4

                        simGroupLength.append(int(randLength4))
                        simGrouped.append(tempGrouped[0:randLength4])
                        simBmu.append(tempBmu)
                        # remove those from the next step
                        tempGrouped = np.delete(tempGrouped, np.arange(0, randLength4))


                        if remainingDays4 > 5:
                            randLength5 = random.randint(1, 3) + 2
                            remainingDays5 = remainingDays4 - randLength5

                            simGroupLength.append(int(randLength5))
                            simGrouped.append(tempGrouped[0:randLength5])
                            simBmu.append(tempBmu)
                            # remove those from the next step
                            tempGrouped = np.delete(tempGrouped, np.arange(0, randLength5))

                            if remainingDays5 > 5:
                                randLength6 = random.randint(1, 3) + 2
                                remainingDays6 = remainingDays5 - randLength6

                                simGroupLength.append(int(randLength6))
                                simGrouped.append(tempGrouped[0:randLength6])
                                simBmu.append(tempBmu)
                                # remove those from the next step
                                tempGrouped = np.delete(tempGrouped, np.arange(0, randLength6))

                                if remainingDays6 > 5:
                                    randLength7 = random.randint(1, 3) + 2
                                    remainingDays7 = remainingDays6 - randLength7

                                    simGroupLength.append(int(randLength7))
                                    simGrouped.append(tempGrouped[0:randLength7])
                                    simBmu.append(tempBmu)
                                    # remove those from the next step
                                    tempGrouped = np.delete(tempGrouped, np.arange(0, randLength7))

                                    if remainingDays7 > 5:
                                        randLength8 = random.randint(1, 3) + 2
                                        remainingDays8 = remainingDays7 - randLength8
                                        #print('after 7 breaks still have: {} days left'.format(remainingDays8))
                                        simGroupLength.append(int(randLength8))
                                        simGrouped.append(tempGrouped[0:randLength8])
                                        simBmu.append(tempBmu)
                                        # remove those from the next step
                                        tempGrouped = np.delete(tempGrouped, np.arange(0, randLength8))
                                        if remainingDays8 > 5:
                                            randLength9 = random.randint(1, 3) + 2
                                            remainingDays9 = remainingDays8 - randLength9
                                            print('after 8 breaks still have: {} days left'.format(remainingDays9))
                                            simGroupLength.append(int(randLength9))
                                            simGrouped.append(tempGrouped[0:randLength9])
                                            simBmu.append(tempBmu)
                                            # remove those from the next step
                                            tempGrouped = np.delete(tempGrouped, np.arange(0, randLength8))
                                        else:
                                            simGroupLength.append(int(len(tempGrouped)))
                                            simGrouped.append(tempGrouped)
                                            simBmu.append(tempBmu)
                                    else:
                                        simGroupLength.append(int(len(tempGrouped)))
                                        simGrouped.append(tempGrouped)
                                        simBmu.append(tempBmu)
                                else:
                                    simGroupLength.append(int(len(tempGrouped)))
                                    simGrouped.append(tempGrouped)
                                    simBmu.append(tempBmu)
                            else:
                                simGroupLength.append(int(len(tempGrouped)))
                                simGrouped.append(tempGrouped)
                                simBmu.append(tempBmu)
                        else:
                            simGroupLength.append(int(len(tempGrouped)))
                            simGrouped.append(tempGrouped)
                            simBmu.append(tempBmu)
                    else:
                        simGroupLength.append(int(len(tempGrouped)))
                        simGrouped.append(tempGrouped)
                        simBmu.append(tempBmu)
                else:
                    simGroupLength.append(int(len(tempGrouped)))
                    simGrouped.append(tempGrouped)
                    simBmu.append(tempBmu)
            else:
                simGroupLength.append(int(len(tempGrouped)))
                simGrouped.append(tempGrouped)
                simBmu.append(tempBmu)
        else:
            simGroupLength.append(int(groupLength[i]))
            simGrouped.append(grouped[i])
            simBmu.append(tempBmu)

    simBmuLengthChopped.append(np.asarray(simGroupLength))
    simBmuGroupsChopped.append(simGrouped)
    simBmuChopped.append(np.asarray(simBmu))

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index], closest_index


simulationsHs = list()
simulationsTp = list()
simulationsDm = list()
simulationsSs = list()
simulationsTime = list()

for simNum in range(50):

    simHs = []
    simTp = []
    simDm = []
    simSs = []
    simTime = []

    for i in range(len(simBmuChopped[simNum])):
        tempBmu = int(simBmuChopped[simNum][i])
        randStorm = random.randint(0, 9999)
        stormDetails = gevCopulaSims[tempBmu][randStorm]
        durSim = simBmuLengthChopped[simNum][i]

        simDmNorm = (stormDetails[4] - np.asarray(bmuDataMin)[tempBmu,0]) / (np.asarray(bmuDataMax)[tempBmu,0]-np.asarray(bmuDataMin)[tempBmu,0])
        simSsNorm = (stormDetails[5] - np.asarray(bmuDataMin)[tempBmu,1]) / (np.asarray(bmuDataMax)[tempBmu,1]-np.asarray(bmuDataMin)[tempBmu,1])
        test, closeIndex = closest_node([simDmNorm,simSsNorm],np.asarray(bmuDataNormalized)[tempBmu])
        actualIndex = int(np.asarray(copulaData[tempBmu])[closeIndex,6])

        simHs.append((normalizedHydros[tempBmu][actualIndex]['hsNorm']) * (stormDetails[0]-stormDetails[1]) + stormDetails[1])
        simTp.append((normalizedHydros[tempBmu][actualIndex]['tpNorm']) * (stormDetails[2]-stormDetails[3]) + stormDetails[3])
        simDm.append((normalizedHydros[tempBmu][actualIndex]['tpNorm']) + stormDetails[4])
        simSs.append((normalizedHydros[tempBmu][actualIndex]['ssNorm']) + stormDetails[5])
        #simTime.append(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
        #dt = np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
        simTime.append(np.hstack((np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim), np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)[-1])))
        if len(normalizedHydros[tempBmu][actualIndex]['hsNorm']) < len(normalizedHydros[tempBmu][actualIndex]['timeNorm']):
            print('Time is shorter than Hs in bmu {}, index {}'.format(tempBmu,actualIndex))

    simulationsHs.append(np.hstack(simHs))
    simulationsTp.append(np.hstack(simTp))
    simulationsDm.append(np.hstack(simDm))
    simulationsSs.append(np.hstack(simSs))
    cumulativeHours = np.cumsum(np.hstack(simTime))
    newDailyTime = [datetime(1979, 2, 1) + timedelta(days=ii) for ii in cumulativeHours]
    simulationsTime.append(newDailyTime)




### TODO: interpolate to even hourly times across all simulations....

plt.figure()
ax1 = plt.subplot2grid((4,1),(0,0),rowspan=1,colspan=1)
# ax1.pcolor(np.asarray(simBmu)[1:1000])
ax1.plot(simulationsTime[4],simulationsHs[4])
ax2 = plt.subplot2grid((4,1),(1,0),rowspan=1,colspan=1)
ax2.plot(simulationsTime[0],simulationsHs[0])
ax3 = plt.subplot2grid((4,1),(2,0),rowspan=1,colspan=1)
ax3.plot(simulationsTime[1],simulationsHs[1])
ax4 = plt.subplot2grid((4,1),(3,0),rowspan=1,colspan=1)
ax4.plot(simulationsTime[2],simulationsHs[2])


for i in range(50):
    dataHs = simulationsHs[i]
    dataTime = simulationsTime[i]

    years = np.arange(1979,2020)
    for yr in years:
        indexYR = np.where((dataTime > datetime(yr,1,1,0,0,0)) and (dataTime < datetime(yr,12,31,23,0,0)))

