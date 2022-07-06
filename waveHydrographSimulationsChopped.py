import numpy as np
from datetime import datetime, date, timedelta
import random
import itertools
import operator
import pickle


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

with open(r"historicalData.pickle", "rb") as input_file:
   historicalData = pickle.load(input_file)
grouped = historicalData['grouped']
groupLength = historicalData['groupLength']
bmuGroup = historicalData['bmuGroup']
timeGroup = historicalData['timeGroup']

with open(r"gevCopulaSims.pickle", "rb") as input_file:
   gevCopulaSimsInput = pickle.load(input_file)
gevCopulaSims = gevCopulaSimsInput['gevCopulaSims']


with open(r"dwtFutureSimulations1000.pickle", "rb") as input_file:
   dwtFutureSimulations = pickle.load(input_file)
evbmus_sim = dwtFutureSimulations['evbmus_sim']
# sim_years = dwtFutureSimulations['sim_years']
dates_sim = dwtFutureSimulations['dates_sim']
# awtBMUsim = dwtFutureSimulations['awtBMUsim']
# awtPC1sim = dwtFutureSimulations['awtPC1sim']
# awtPC2sim = dwtFutureSimulations['awtPC2sim']
# awtPC3sim = dwtFutureSimulations['awtPC3sim']
# mjoRMM1Sim = dwtFutureSimulations['mjoRMM1Sim']
# mjoRMM2Sim = dwtFutureSimulations['mjoRMM2Sim']




dt = datetime(2021,6, 1)
end = datetime(2121, 6, 1)
step = timedelta(days=1)
midnightTime = []
while dt < end:
    midnightTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step

groupedList = list()
groupLengthList = list()
bmuGroupList = list()
#timeGroupList = list()
for hh in range(1000):
    print('breaking up hydrogrpahs for simulation {}'.format(hh))
    bmus = evbmus_sim[:,hh]
    tempBmusGroup = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(bmus), key=operator.itemgetter(1))]
    groupedList.append(tempBmusGroup)
    groupLengthList.append(np.asarray([len(i) for i in tempBmusGroup]))
    bmuGroupList.append(np.asarray([bmus[i[0]] for i in tempBmusGroup]))
    #timeGroupList.append([np.asarray(midnightTime)[i] for i in tempBmusGroup])





### TODO: use the historical to develop a new chronology

numRealizations = 1000
#
# simBmuChopped = []
# simBmuLengthChopped = []
# simBmuGroupsChopped = []
# for pp in range(numRealizations):
#
#     print('working on realization #{}'.format(pp))
#     bmuGroup = bmuGroupList[pp]
#     groupLength = groupLengthList[pp]
#     grouped = groupedList[pp]
#     simGroupLength = []
#     simGrouped = []
#     simBmu = []
#     for i in range(len(groupLength)):
#         if np.remainder(i,1000) == 0:
#             print('done with {} hydrographs'.format(i))
#         tempGrouped = grouped[i]
#         tempBmu = int(bmuGroup[i])
#         if groupLength[i] > 5:
#             # random days between 3 and 5
#             randLength = random.randint(1, 3) + 2
#             remainingDays = groupLength[i] - randLength
#             # add this to the record
#             simGroupLength.append(int(randLength))
#             simGrouped.append(grouped[i][0:randLength])
#             simBmu.append(tempBmu)
#             # remove those from the next step
#             tempGrouped = np.delete(tempGrouped,np.arange(0,randLength))
#
#             if remainingDays > 5:
#                 randLength2 = random.randint(1, 3) + 2
#                 remainingDays2 = remainingDays - randLength2
#
#                 simGroupLength.append(int(randLength2))
#                 simGrouped.append(tempGrouped[0:randLength2])
#                 simBmu.append(tempBmu)
#                 # remove those from the next step
#                 tempGrouped = np.delete(tempGrouped, np.arange(0, randLength2))
#
#                 if remainingDays2 > 5:
#                     randLength3 = random.randint(1, 3) + 2
#                     remainingDays3 = remainingDays2 - randLength3
#
#                     simGroupLength.append(int(randLength3))
#                     simGrouped.append(tempGrouped[0:randLength3])
#                     simBmu.append(tempBmu)
#                     # remove those from the next step
#                     tempGrouped = np.delete(tempGrouped, np.arange(0, randLength3))
#
#                     if remainingDays3 > 5:
#                         randLength4 = random.randint(1, 3) + 2
#                         remainingDays4 = remainingDays3 - randLength4
#
#                         simGroupLength.append(int(randLength4))
#                         simGrouped.append(tempGrouped[0:randLength4])
#                         simBmu.append(tempBmu)
#                         # remove those from the next step
#                         tempGrouped = np.delete(tempGrouped, np.arange(0, randLength4))
#
#
#                         if remainingDays4 > 5:
#                             randLength5 = random.randint(1, 3) + 2
#                             remainingDays5 = remainingDays4 - randLength5
#
#                             simGroupLength.append(int(randLength5))
#                             simGrouped.append(tempGrouped[0:randLength5])
#                             simBmu.append(tempBmu)
#                             # remove those from the next step
#                             tempGrouped = np.delete(tempGrouped, np.arange(0, randLength5))
#
#                             if remainingDays5 > 5:
#                                 randLength6 = random.randint(1, 3) + 2
#                                 remainingDays6 = remainingDays5 - randLength6
#
#                                 simGroupLength.append(int(randLength6))
#                                 simGrouped.append(tempGrouped[0:randLength6])
#                                 simBmu.append(tempBmu)
#                                 # remove those from the next step
#                                 tempGrouped = np.delete(tempGrouped, np.arange(0, randLength6))
#
#                                 if remainingDays6 > 5:
#                                     randLength7 = random.randint(1, 3) + 2
#                                     remainingDays7 = remainingDays6 - randLength7
#
#                                     simGroupLength.append(int(randLength7))
#                                     simGrouped.append(tempGrouped[0:randLength7])
#                                     simBmu.append(tempBmu)
#                                     # remove those from the next step
#                                     tempGrouped = np.delete(tempGrouped, np.arange(0, randLength7))
#
#                                     if remainingDays7 > 5:
#                                         randLength8 = random.randint(1, 3) + 2
#                                         remainingDays8 = remainingDays7 - randLength8
#                                         #print('after 7 breaks still have: {} days left'.format(remainingDays8))
#                                         simGroupLength.append(int(randLength8))
#                                         simGrouped.append(tempGrouped[0:randLength8])
#                                         simBmu.append(tempBmu)
#                                         # remove those from the next step
#                                         tempGrouped = np.delete(tempGrouped, np.arange(0, randLength8))
#                                         if remainingDays8 > 5:
#                                             randLength9 = random.randint(1, 3) + 2
#                                             remainingDays9 = remainingDays8 - randLength9
#                                             simGroupLength.append(int(randLength9))
#                                             simGrouped.append(tempGrouped[0:randLength9])
#                                             simBmu.append(tempBmu)
#                                             # remove those from the next step
#                                             tempGrouped = np.delete(tempGrouped, np.arange(0, randLength9))
#                                             if remainingDays9 > 5:
#                                                 randLength10 = random.randint(1, 3) + 2
#                                                 remainingDays10 = remainingDays9 - randLength10
#                                                 simGroupLength.append(int(randLength10))
#                                                 simGrouped.append(tempGrouped[0:randLength10])
#                                                 simBmu.append(tempBmu)
#                                                 # remove those from the next step
#                                                 tempGrouped = np.delete(tempGrouped, np.arange(0, randLength10))
#                                                 if remainingDays10 > 5:
#                                                     randLength11 = random.randint(1, 3) + 2
#                                                     remainingDays11 = remainingDays10 - randLength11
#                                                     simGroupLength.append(int(randLength11))
#                                                     simGrouped.append(tempGrouped[0:randLength11])
#                                                     simBmu.append(tempBmu)
#                                                     # remove those from the next step
#                                                     tempGrouped = np.delete(tempGrouped, np.arange(0, randLength11))
#                                                     if remainingDays11 > 5:
#                                                         randLength12 = random.randint(1, 3) + 2
#                                                         remainingDays12 = remainingDays11 - randLength12
#                                                         simGroupLength.append(int(randLength12))
#                                                         simGrouped.append(tempGrouped[0:randLength12])
#                                                         simBmu.append(tempBmu)
#                                                         # remove those from the next step
#                                                         tempGrouped = np.delete(tempGrouped, np.arange(0, randLength12))
#                                                         if remainingDays12 > 5:
#                                                             randLength13 = random.randint(1, 3) + 2
#                                                             remainingDays13 = remainingDays12 - randLength13
#                                                             simGroupLength.append(int(randLength13))
#                                                             simGrouped.append(tempGrouped[0:randLength13])
#                                                             simBmu.append(tempBmu)
#                                                             # remove those from the next step
#                                                             tempGrouped = np.delete(tempGrouped,
#                                                                                     np.arange(0, randLength13))
#                                                             if remainingDays13 > 5:
#                                                                 randLength14 = random.randint(1, 3) + 2
#                                                                 remainingDays14 = remainingDays13 - randLength14
#                                                                 print('after 13 breaks still have: {} days left'.format(
#                                                                     remainingDays14))
#                                                                 simGroupLength.append(int(randLength14))
#                                                                 simGrouped.append(tempGrouped[0:randLength14])
#                                                                 simBmu.append(tempBmu)
#                                                                 # remove those from the next step
#                                                                 tempGrouped = np.delete(tempGrouped,
#                                                                                         np.arange(0, randLength14))
#                                                             else:
#                                                                 simGroupLength.append(int(len(tempGrouped)))
#                                                                 simGrouped.append(tempGrouped)
#                                                                 simBmu.append(tempBmu)
#                                                         else:
#                                                             simGroupLength.append(int(len(tempGrouped)))
#                                                             simGrouped.append(tempGrouped)
#                                                             simBmu.append(tempBmu)
#                                                     else:
#                                                         simGroupLength.append(int(len(tempGrouped)))
#                                                         simGrouped.append(tempGrouped)
#                                                         simBmu.append(tempBmu)
#                                                 else:
#                                                     simGroupLength.append(int(len(tempGrouped)))
#                                                     simGrouped.append(tempGrouped)
#                                                     simBmu.append(tempBmu)
#                                             else:
#                                                 simGroupLength.append(int(len(tempGrouped)))
#                                                 simGrouped.append(tempGrouped)
#                                                 simBmu.append(tempBmu)
#                                         else:
#                                             simGroupLength.append(int(len(tempGrouped)))
#                                             simGrouped.append(tempGrouped)
#                                             simBmu.append(tempBmu)
#                                     else:
#                                         simGroupLength.append(int(len(tempGrouped)))
#                                         simGrouped.append(tempGrouped)
#                                         simBmu.append(tempBmu)
#                                 else:
#                                     simGroupLength.append(int(len(tempGrouped)))
#                                     simGrouped.append(tempGrouped)
#                                     simBmu.append(tempBmu)
#                             else:
#                                 simGroupLength.append(int(len(tempGrouped)))
#                                 simGrouped.append(tempGrouped)
#                                 simBmu.append(tempBmu)
#                         else:
#                             simGroupLength.append(int(len(tempGrouped)))
#                             simGrouped.append(tempGrouped)
#                             simBmu.append(tempBmu)
#                     else:
#                         simGroupLength.append(int(len(tempGrouped)))
#                         simGrouped.append(tempGrouped)
#                         simBmu.append(tempBmu)
#                 else:
#                     simGroupLength.append(int(len(tempGrouped)))
#                     simGrouped.append(tempGrouped)
#                     simBmu.append(tempBmu)
#             else:
#                 simGroupLength.append(int(len(tempGrouped)))
#                 simGrouped.append(tempGrouped)
#                 simBmu.append(tempBmu)
#         else:
#             simGroupLength.append(int(groupLength[i]))
#             simGrouped.append(grouped[i])
#             simBmu.append(tempBmu)
#     simBmuLengthChopped.append(np.asarray(simGroupLength))
#     simBmuGroupsChopped.append(simGrouped)
#     simBmuChopped.append(np.asarray(simBmu))
#




simBmuChopped = []
simBmuLengthChopped = []
simBmuGroupsChopped = []
for pp in range(numRealizations):

    print('working on realization #{}'.format(pp))
    bmuGroup = bmuGroupList[pp]
    groupLength = groupLengthList[pp]
    grouped = groupedList[pp]
    simGroupLength = []
    simGrouped = []
    simBmu = []
    for i in range(len(groupLength)):
        # if np.remainder(i,10000) == 0:
        #     print('done with {} hydrographs'.format(i))
        tempGrouped = grouped[i]
        tempBmu = int(bmuGroup[i])
        remainingDays = groupLength[i] - 5
        if groupLength[i] < 5:
            simGroupLength.append(int(groupLength[i]))
            simGrouped.append(grouped[i])
            simBmu.append(tempBmu)
        else:
            counter = 0
            while (len(grouped[i]) - counter) > 5:
                # print('we are in the loop with remainingDays = {}'.format(remainingDays))
                # random days between 3 and 5
                randLength = random.randint(1, 3) + 2
                # add this to the record
                simGroupLength.append(int(randLength))
                # simGrouped.append(tempGrouped[0:randLength])
                # print('should be adding {}'.format(grouped[i][counter:counter+randLength]))
                simGrouped.append(grouped[i][counter:counter+randLength])
                simBmu.append(tempBmu)
                # remove those from the next step
                # tempGrouped = np.delete(tempGrouped,np.arange(0,randLength))
                # do we continue forward
                remainingDays = remainingDays - randLength
                counter = counter + randLength

            if (len(grouped[i]) - counter) > 0:
                simGroupLength.append(int((len(grouped[i]) - counter)))
                # simGrouped.append(tempGrouped[0:])
                simGrouped.append(grouped[i][counter:])
                simBmu.append(tempBmu)
    simBmuLengthChopped.append(np.asarray(simGroupLength))
    simBmuGroupsChopped.append(simGrouped)
    simBmuChopped.append(np.asarray(simBmu))





simsChoppedPickle = 'simulations1000Chopped.pickle'
outputSimsChopped = {}
outputSimsChopped['simBmuLengthChopped'] = simBmuLengthChopped
outputSimsChopped['simBmuGroupsChopped'] = simBmuGroupsChopped
outputSimsChopped['simBmuChopped'] = simBmuChopped

with open(simsChoppedPickle,'wb') as f:
    pickle.dump(outputSimsChopped, f)



asdfg

#
# def closest_node(node, nodes):
#     closest_index = distance.cdist([node], nodes).argmin()
#     return nodes[closest_index], closest_index
#
#
#
# simulationsHs = list()
# simulationsTp = list()
# simulationsDm = list()
# simulationsSs = list()
# simulationsTime = list()
#
# for simNum in range(100):
#
#     simHs = []
#     simTp = []
#     simDm = []
#     simSs = []
#     simTime = []
#     print('filling in simulation #{}'.format(simNum))
#
#     for i in range(len(simBmuChopped[simNum])):
#         if np.remainder(i,1000) == 0:
#             print('done with {} hydrographs'.format(i))
#         tempBmu = int(simBmuChopped[simNum][i]-1)
#         randStorm = random.randint(0, 9999)
#         stormDetails = gevCopulaSims[tempBmu][randStorm]
#         durSim = simBmuLengthChopped[simNum][i]
#
#         simDmNorm = (stormDetails[4] - np.asarray(bmuDataMin)[tempBmu,0]) / (np.asarray(bmuDataMax)[tempBmu,0]-np.asarray(bmuDataMin)[tempBmu,0])
#         simSsNorm = (stormDetails[5] - np.asarray(bmuDataMin)[tempBmu,1]) / (np.asarray(bmuDataMax)[tempBmu,1]-np.asarray(bmuDataMin)[tempBmu,1])
#         test, closeIndex = closest_node([simDmNorm,simSsNorm],np.asarray(bmuDataNormalized)[tempBmu])
#         actualIndex = int(np.asarray(copulaData[tempBmu])[closeIndex,6])
#
#         simHs.append((normalizedHydros[tempBmu][actualIndex]['hsNorm']) * (stormDetails[0]-stormDetails[1]) + stormDetails[1])
#         simTp.append((normalizedHydros[tempBmu][actualIndex]['tpNorm']) * (stormDetails[2]-stormDetails[3]) + stormDetails[3])
#         simDm.append((normalizedHydros[tempBmu][actualIndex]['tpNorm']) + stormDetails[4])
#         simSs.append((normalizedHydros[tempBmu][actualIndex]['ssNorm']) + stormDetails[5])
#         #simTime.append(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
#         #dt = np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)
#         simTime.append(np.hstack((np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim), np.diff(normalizedHydros[tempBmu][actualIndex]['timeNorm']*durSim)[-1])))
#         if len(normalizedHydros[tempBmu][actualIndex]['hsNorm']) < len(normalizedHydros[tempBmu][actualIndex]['timeNorm']):
#             print('Time is shorter than Hs in bmu {}, index {}'.format(tempBmu,actualIndex))
#
#     simulationsHs.append(np.hstack(simHs))
#     simulationsTp.append(np.hstack(simTp))
#     simulationsDm.append(np.hstack(simDm))
#     simulationsSs.append(np.hstack(simSs))
#     cumulativeHours = np.cumsum(np.hstack(simTime))
#     newDailyTime = [datetime(2022, 6, 1) + timedelta(days=ii) for ii in cumulativeHours]
#     simulationsTime.append(newDailyTime)
#
#
# plt.figure()
# ax1 = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
# ax1.plot(simulationsTime[0],simulationsHs[0])
#
# ### TODO: Need to assess the statistics of these hypothetical scenarios... Yearly max Hs? Wave Energy?
#
# ### TODO: Which requires interpolating the time series to hourly values...
#
# # for qq in len(simulationsTime):
#
#


