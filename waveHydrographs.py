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

def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [date(d[0], d[1], d[2]) for d in d_vec]

def dateDay2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime(d[0], d[1], d[2]) for d in d_vec]



# with open(r"dwtsAll6TCTracksClusters.pickle", "rb") as input_file:
with open(r"dwts49Clusters.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

timeDWTs = historicalDWTs['SLPtime']
# outputDWTs['slpDates'] = slpDates
dwtBmus = historicalDWTs['bmus_corrected']

# with open(r"dwtsOfExtraTropicalDays.pickle", "rb") as input_file:
with open(r"dwtsOfExtraTropicalDays21Clusters.pickle", "rb") as input_file:
   historicalTWTs = pickle.load(input_file)
#timeTCs = historicalTWTs['tcDates']
twtBmus = historicalTWTs['bmus_corrected']
twtOrder = historicalTWTs['kma_order']
tcIndices = historicalTWTs['tcIndices']
TIMEtcs = historicalTWTs['TIMEtcs']


timeSLPs = dateDay2datetimeDate(timeDWTs)
timeTCs = dateDay2datetimeDate(TIMEtcs)

overlap = [x for x in timeSLPs if x in timeTCs]
ind_dict = dict((k,i) for i,k in enumerate(timeSLPs))
inter = set(timeSLPs).intersection(timeTCs)
indices = [ ind_dict[x] for x in inter ]
indices.sort()

mask = np.ones(len(timeDWTs), np.bool)
mask[indices] = 0
timeEWTs = timeDWTs[mask,:]

mask2 = np.zeros(len(timeDWTs), np.bool)
mask2[indices] = 1
timeTWTs = timeDWTs[mask2,:]

bmus = np.nan * np.ones((len(timeDWTs)))
bmus[mask] = dwtBmus+0
bmus[mask2] = twtBmus+49

bmus_dates = dateDay2datetimeDate(timeDWTs)
bmus_dates_months = np.array([d.month for d in bmus_dates])
bmus_dates_days = np.array([d.day for d in bmus_dates])

# bmus = bmus[120:]+1
# timeDWTs = timeDWTs[120:]
# bmus_dates = bmus_dates[120:]


#
# waterLevelDir = '/media/dylananderson/Elements/frfWaterLevel'
# files = os.listdir(waterLevelDir)
# files.sort()
# files_path = [os.path.join(os.path.abspath(waterLevelDir), x) for x in files]
# wls = Dataset(files_path[0])
#
# def getWaterLevel(file):
#     wldata = Dataset(file)
#     waterLevel = wldata.variables['waterLevel'][:]
#     predictedWaterLevel = wldata.variables['predictedWaterLevel'][:]
#     residualWaterLevel = wldata.variables['residualWaterLevel'][:]
#     timeWl = wldata.variables['time'][:]
#     output = dict()
#     output['waterLevel'] = waterLevel
#     output['predictedWaterLevel'] = predictedWaterLevel
#     output['residualWaterLevel'] = residualWaterLevel
#     output['time'] = timeWl
#     return output
#
# timeWaterLevelFRF = []
# waterLevelFRF = []
# predictedWaterLevelFRF = []
# residualWaterLevelFRF = []
# for i in files_path:
#     waterLevels = getWaterLevel(i)
#     waterLevelFRF = np.append(waterLevelFRF,waterLevels['waterLevel'])
#     predictedWaterLevelFRF = np.append(predictedWaterLevelFRF,waterLevels['predictedWaterLevel'])
#     residualWaterLevelFRF = np.append(residualWaterLevelFRF,waterLevels['residualWaterLevel'])
#     timeWaterLevelFRF = np.append(timeWaterLevelFRF,waterLevels['time'].flatten())
#
# tWaterLevelFRF = [datetime.fromtimestamp(x) for x in timeWaterLevelFRF]
# tWaterLevelFRF = np.asarray(tWaterLevelFRF)
# goodWaterLevel = np.where((residualWaterLevelFRF > -99))
# wlFRF = waterLevelFRF[goodWaterLevel[0]]
# predFRF = predictedWaterLevelFRF[goodWaterLevel[0]]
# resFRF = residualWaterLevelFRF[goodWaterLevel[0]]
# tFRF = tWaterLevelFRF[goodWaterLevel[0]]
# # badWaterLevel = np.where((residualWaterLevelFRF < -99))
# # tWaterLevelFRF = [datetime.fromtimestamp(x) for x in timeWaterLevelFRF]
# # tWaterLevelFRF = np.asarray(tWaterLevelFRF)
# # waterLevelFRF[badWaterLevel] = waterLevelFRF[badWaterLevel]*np.nan
# # predictedWaterLevelFRF[badWaterLevel] = predictedWaterLevelFRF[badWaterLevel]*np.nan
# # residualWaterLevelFRF[badWaterLevel] = residualWaterLevelFRF[badWaterLevel]*np.nan
# #


wls = ReadMatfile('/home/dylananderson/projects/atlanticClimate/frfTideGaugeSerafin2.mat')

tide = wls['dailyData']['tide']
wl = wls['dailyData']['wl']
seasonal = wls['dailyData']['seasonal']
msl = wls['dailyData']['msl']
mmsla = wls['dailyData']['mmsla']
dsla = wls['dailyData']['dsla']
ss = wls['dailyData']['ss']
timeHourly = wls['dailyData']['hourlyDateVec']
timeMonthly = wls['dailyData']['monthDateVec']
mmslaMonth = wls['dailyData']['mmsla_month']


def hourlyVec2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime(d[0], d[1], d[2], d[3]) for d in d_vec]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

timeMMSLA = hourlyVec2datetime(timeMonthly)

# timeMMSLAChopped = timeMMSLA[2:]
#
timeWL = hourlyVec2datetime(timeHourly)
#
# plt.figure()
# plt.plot(timeWL,climatologyDaily)
#
index = np.where((np.isnan(mmslaMonth)))
MMSLAcopy = mmslaMonth
MMSLAcopy[index] = 0 * np.ones(len(index[0]))
smooth = running_mean(MMSLAcopy,9)

tFRF = np.asarray(timeWL)
resFRF = ss
wlFRF = wl

plt.figure()
plt.plot(tFRF,resFRF)

# Lets find the closest to each hour?






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


hsCombined = Hs
tpCombined = Tp
dmCombined = Dm
hsSwellCombined = hsSwell
tpSwellCombined = tpSwell
dmSwellCombined = dmSwell
hsWindseaCombined = hsWindsea
tpWindseaCombined = tpWindsea
dmWindseaCombined = dmWindsea

badDirs = np.where((dmCombined > 360))
dmCombined[badDirs] = dmCombined[badDirs]*np.nan

badDirsSwell = np.where((dmSwellCombined > 360))
dmSwellCombined[badDirsSwell] = dmSwellCombined[badDirsSwell]*np.nan
badDirsWindsea = np.where((dmWindseaCombined > 360))
dmWindseaCombined[badDirsWindsea] = dmWindseaCombined[badDirsWindsea]*np.nan

waveNorm = dmCombined - 72
neg = np.where((waveNorm > 180))
waveNorm[neg[0]] = waveNorm[neg[0]]-360
# offpos = np.where((waveNorm>90))
# offneg = np.where((waveNorm<-90))
# waveNorm[offpos[0]] = waveNorm[offpos[0]]*0
# waveNorm[offneg[0]] = waveNorm[offneg[0]]*0

waveNormSwell = dmSwellCombined - 72
negSwell = np.where((waveNormSwell > 180))
waveNormSwell[negSwell[0]] = waveNormSwell[negSwell[0]]-360
# offposSwell = np.where((waveNormSwell>90))
# offnegSwell = np.where((waveNormSwell<-90))
# waveNormSwell[offposSwell[0]] = waveNormSwell[offposSwell[0]]*0
# waveNormSwell[offnegSwell[0]] = waveNormSwell[offnegSwell[0]]*0

waveNormWindsea = dmWindseaCombined - 72
negWindsea = np.where((waveNormWindsea > 180))
waveNormWindsea[negWindsea[0]] = waveNormWindsea[negWindsea[0]]-360
# offposWindsea = np.where((waveNormWindsea>90))
# offnegWindsea = np.where((waveNormWindsea<-90))
# waveNormWindsea[offposWindsea[0]] = waveNormWindsea[offposWindsea[0]]*0
# waveNormWindsea[offnegWindsea[0]] = waveNormWindsea[offnegWindsea[0]]*0

tWave = [datetime.fromtimestamp(x) for x in timeWave]
tC = np.array(tWave)


plt.plot(tWave,hsCombined)

realWavesPickle = 'realWaves.pickle'
outputReal = {}
outputReal['tWave'] = tWave
outputReal['tC'] = tC
outputReal['hsCombined'] = hsCombined
outputReal['tpCombined'] = tpCombined
outputReal['dmCombined'] = dmCombined
outputReal['waveNorm'] = waveNorm
outputReal['wlFRF'] = wlFRF
outputReal['tFRF'] = tFRF
outputReal['resFRF'] = resFRF

with open(realWavesPickle, 'wb') as f:
    pickle.dump(outputReal, f)


    asdfg

dt = datetime(1979, 2, 1)
end = datetime(2021, 6, 1)
step = timedelta(days=1)
midnightTime = []
while dt < end:
    midnightTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step



grouped = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(bmus), key=operator.itemgetter(1))]

groupLength = np.asarray([len(i) for i in grouped])
bmuGroup = np.asarray([bmus[i[0]] for i in grouped])
timeGroup = [np.asarray(midnightTime)[i] for i in grouped]

startTimes = [i[0] for i in timeGroup]
endTimes = [i[-1] for i in timeGroup]

hydros = list()
c = 0
for p in range(len(np.unique(bmuGroup))):
    index = np.where((bmuGroup==p))[0][:]
    subLength = groupLength[index]
    m = np.ceil(np.sqrt(len(index)))
    tempList = list()
    for i in range(len(index)):
        st = startTimes[index[i]]
        et = endTimes[index[i]] + timedelta(days=1)

        if et > datetime(2020,1,1):
            print('done up to 2020')
        else:
            waveInd = np.where((tC < et) & (tC >= st))
            waterInd = np.where((tFRF < et) & (tFRF >= st))
            if len(waveInd[0]) > 0:

                if subLength[i] > 5:
                    randLength = random.randint(1,3)+2
                    etNew = startTimes[index[i]] + timedelta(days=randLength)
                    waveInd = np.where((tC < etNew) & (tC >= st))
                    waterInd = np.where((tFRF < etNew) & (tFRF >= st))
                    deltaDays = et-etNew
                    #print('After first cut off = {}'.format(deltaDays.days))
                    c = c + 1
                    tempDict = dict()
                    tempDict['time'] = tC[waveInd[0]]
                    tempDict['numDays'] = subLength[i]
                    tempDict['hs'] = hsCombined[waveInd[0]]
                    tempDict['tp'] = tpCombined[waveInd[0]]
                    tempDict['dm'] = waveNorm[waveInd[0]]
                    tempDict['res'] = resFRF[waterInd[0]]
                    tempDict['wl'] = wlFRF[waterInd[0]]
                    tempDict['cop'] = np.asarray([np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                                  np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                                  np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                    tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                    tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                    tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                    tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                    tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                    tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                    tempList.append(tempDict)
                    if deltaDays.days > 5:
                        randLength = random.randint(1, 3) + 2
                        etNew2 = etNew + timedelta(days=randLength)
                        waveInd = np.where((tC < etNew2) & (tC >= etNew))
                        waterInd = np.where((tFRF < etNew2) & (tFRF >= etNew))
                        deltaDays2 = et - etNew2
                        #print('After second cut off = {}'.format(deltaDays2.days))
                        c = c + 1
                        tempDict = dict()
                        tempDict['time'] = tC[waveInd[0]]
                        tempDict['numDays'] = subLength[i]
                        tempDict['hs'] = hsCombined[waveInd[0]]
                        tempDict['tp'] = tpCombined[waveInd[0]]
                        tempDict['dm'] = waveNorm[waveInd[0]]
                        tempDict['res'] = resFRF[waterInd[0]]
                        tempDict['wl'] = wlFRF[waterInd[0]]
                        tempDict['cop'] = np.asarray([np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                                      np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                                      np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                        tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                        tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                        tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                        tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                        tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                        tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                        tempList.append(tempDict)
                        if deltaDays2.days > 5:
                            randLength = random.randint(1, 3) + 2
                            etNew3 = etNew2 + timedelta(days=randLength)
                            waveInd = np.where((tC < etNew3) & (tC >= etNew2))
                            waterInd = np.where((tFRF < etNew3) & (tFRF >= etNew2))
                            deltaDays3 = et - etNew3
                            #print('After third cut off = {}'.format(deltaDays3.days))
                            c = c + 1
                            tempDict = dict()
                            tempDict['time'] = tC[waveInd[0]]
                            tempDict['numDays'] = subLength[i]
                            tempDict['hs'] = hsCombined[waveInd[0]]
                            tempDict['tp'] = tpCombined[waveInd[0]]
                            tempDict['dm'] = waveNorm[waveInd[0]]
                            tempDict['res'] = resFRF[waterInd[0]]
                            tempDict['wl'] = wlFRF[waterInd[0]]
                            tempDict['cop'] = np.asarray(
                                [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                 np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                 np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                            tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                            tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                            tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                            tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                            tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                            tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                            tempList.append(tempDict)
                            if deltaDays3.days > 5:
                                randLength = random.randint(1, 3) + 2
                                etNew4 = etNew3 + timedelta(days=randLength)
                                waveInd = np.where((tC < etNew4) & (tC >= etNew3))
                                waterInd = np.where((tFRF < etNew4) & (tFRF >= etNew3))
                                deltaDays4 = et - etNew4
                                #print('After fourth cut off = {}'.format(deltaDays4.days))
                                c = c + 1
                                tempDict = dict()
                                tempDict['time'] = tC[waveInd[0]]
                                tempDict['numDays'] = subLength[i]
                                tempDict['hs'] = hsCombined[waveInd[0]]
                                tempDict['tp'] = tpCombined[waveInd[0]]
                                tempDict['dm'] = waveNorm[waveInd[0]]
                                tempDict['res'] = resFRF[waterInd[0]]
                                tempDict['wl'] = wlFRF[waterInd[0]]
                                tempDict['cop'] = np.asarray(
                                    [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                     np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                     np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                                tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                                tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                                tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                                tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                                tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                                tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                                tempList.append(tempDict)
                                if deltaDays4.days > 5:
                                    randLength = random.randint(1, 3) + 2
                                    etNew5 = etNew4 + timedelta(days=randLength)
                                    waveInd = np.where((tC < etNew5) & (tC >= etNew4))
                                    waterInd = np.where((tFRF < etNew5) & (tFRF >= etNew4))
                                    deltaDays5 = et - etNew5
                                    print('After fifth cut off = {}'.format(deltaDays5.days))
                                    c = c + 1
                                    tempDict = dict()
                                    tempDict['time'] = tC[waveInd[0]]
                                    tempDict['numDays'] = subLength[i]
                                    tempDict['hs'] = hsCombined[waveInd[0]]
                                    tempDict['tp'] = tpCombined[waveInd[0]]
                                    tempDict['dm'] = waveNorm[waveInd[0]]
                                    tempDict['res'] = resFRF[waterInd[0]]
                                    tempDict['wl'] = wlFRF[waterInd[0]]
                                    tempDict['cop'] = np.asarray(
                                        [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                         np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                         np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                                    tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                                    tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                                    tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                                    tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                                    tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                                    tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                                    tempList.append(tempDict)
                                    if deltaDays5.days > 5:
                                        randLength = random.randint(1, 3) + 2
                                        etNew6 = etNew5 + timedelta(days=randLength)
                                        waveInd = np.where((tC < etNew6) & (tC >= etNew5))
                                        waterInd = np.where((tFRF < etNew6) & (tFRF >= etNew5))
                                        deltaDays6 = et - etNew6
                                        print('After sixth cut off = {}'.format(deltaDays6.days))
                                        c = c + 1
                                        tempDict = dict()
                                        tempDict['time'] = tC[waveInd[0]]
                                        tempDict['numDays'] = subLength[i]
                                        tempDict['hs'] = hsCombined[waveInd[0]]
                                        tempDict['tp'] = tpCombined[waveInd[0]]
                                        tempDict['dm'] = waveNorm[waveInd[0]]
                                        tempDict['res'] = resFRF[waterInd[0]]
                                        tempDict['wl'] = wlFRF[waterInd[0]]
                                        tempDict['cop'] = np.asarray(
                                            [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                             np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                             np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                                        tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                                        tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                                        tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                                        tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                                        tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                                        tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                                        tempList.append(tempDict)
                                        if deltaDays6.days > 5:
                                            randLength = random.randint(1, 3) + 2
                                            etNew7 = etNew6 + timedelta(days=randLength)
                                            waveInd = np.where((tC < etNew7) & (tC >= etNew6))
                                            waterInd = np.where((tFRF < etNew7) & (tFRF >= etNew6))
                                            deltaDays7 = et - etNew7
                                            print('After seventh cut off = {}'.format(deltaDays7.days))
                                            c = c + 1
                                            tempDict = dict()
                                            tempDict['time'] = tC[waveInd[0]]
                                            tempDict['numDays'] = subLength[i]
                                            tempDict['hs'] = hsCombined[waveInd[0]]
                                            tempDict['tp'] = tpCombined[waveInd[0]]
                                            tempDict['dm'] = waveNorm[waveInd[0]]
                                            tempDict['res'] = resFRF[waterInd[0]]
                                            tempDict['wl'] = wlFRF[waterInd[0]]
                                            tempDict['cop'] = np.asarray(
                                                [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                                 np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                                 np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                                            tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                                            tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                                            tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                                            tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                                            tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                                            tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                                            tempList.append(tempDict)
                                            if deltaDays7.days > 5:
                                                print('jesus need another')
                                            else:
                                                waveInd = np.where((tC < et) & (tC >= etNew7))
                                                waterInd = np.where((tFRF < et) & (tFRF >= etNew7))
                                                c = c + 1
                                                tempDict = dict()
                                                tempDict['time'] = tC[waveInd[0]]
                                                tempDict['numDays'] = subLength[i]
                                                tempDict['hs'] = hsCombined[waveInd[0]]
                                                tempDict['tp'] = tpCombined[waveInd[0]]
                                                tempDict['dm'] = waveNorm[waveInd[0]]
                                                tempDict['res'] = resFRF[waterInd[0]]
                                                tempDict['wl'] = wlFRF[waterInd[0]]
                                                tempDict['cop'] = np.asarray(
                                                    [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                                     np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                                     np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                                                tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                                                tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                                                tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                                                tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                                                tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                                                tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                                                tempList.append(tempDict)



                                        else:
                                            waveInd = np.where((tC < et) & (tC >= etNew6))
                                            waterInd = np.where((tFRF < et) & (tFRF >= etNew6))
                                            c = c + 1
                                            tempDict = dict()
                                            tempDict['time'] = tC[waveInd[0]]
                                            tempDict['numDays'] = subLength[i]
                                            tempDict['hs'] = hsCombined[waveInd[0]]
                                            tempDict['tp'] = tpCombined[waveInd[0]]
                                            tempDict['dm'] = waveNorm[waveInd[0]]
                                            tempDict['res'] = resFRF[waterInd[0]]
                                            tempDict['wl'] = wlFRF[waterInd[0]]
                                            tempDict['cop'] = np.asarray(
                                                [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                                 np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                                 np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                                            tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                                            tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                                            tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                                            tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                                            tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                                            tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                                            tempList.append(tempDict)


                                    else:
                                        waveInd = np.where((tC < et) & (tC >= etNew5))
                                        waterInd = np.where((tFRF < et) & (tFRF >= etNew5))
                                        c = c + 1
                                        tempDict = dict()
                                        tempDict['time'] = tC[waveInd[0]]
                                        tempDict['numDays'] = subLength[i]
                                        tempDict['hs'] = hsCombined[waveInd[0]]
                                        tempDict['tp'] = tpCombined[waveInd[0]]
                                        tempDict['dm'] = waveNorm[waveInd[0]]
                                        tempDict['res'] = resFRF[waterInd[0]]
                                        tempDict['wl'] = wlFRF[waterInd[0]]
                                        tempDict['cop'] = np.asarray(
                                            [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                             np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                             np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                                        tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                                        tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                                        tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                                        tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                                        tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                                        tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                                        tempList.append(tempDict)

                                else:
                                    waveInd = np.where((tC < et) & (tC >= etNew4))
                                    waterInd = np.where((tFRF < et) & (tFRF >= etNew4))
                                    c = c + 1
                                    tempDict = dict()
                                    tempDict['time'] = tC[waveInd[0]]
                                    tempDict['numDays'] = subLength[i]
                                    tempDict['hs'] = hsCombined[waveInd[0]]
                                    tempDict['tp'] = tpCombined[waveInd[0]]
                                    tempDict['dm'] = waveNorm[waveInd[0]]
                                    tempDict['res'] = resFRF[waterInd[0]]
                                    tempDict['wl'] = wlFRF[waterInd[0]]
                                    tempDict['cop'] = np.asarray(
                                        [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                         np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                         np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                                    tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                                    tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                                    tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                                    tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                                    tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                                    tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                                    tempList.append(tempDict)


                            else:
                                waveInd = np.where((tC < et) & (tC >= etNew3))
                                waterInd = np.where((tFRF < et) & (tFRF >= etNew3))
                                c = c + 1
                                tempDict = dict()
                                tempDict['time'] = tC[waveInd[0]]
                                tempDict['numDays'] = subLength[i]
                                tempDict['hs'] = hsCombined[waveInd[0]]
                                tempDict['tp'] = tpCombined[waveInd[0]]
                                tempDict['dm'] = waveNorm[waveInd[0]]
                                tempDict['res'] = resFRF[waterInd[0]]
                                tempDict['wl'] = wlFRF[waterInd[0]]
                                tempDict['cop'] = np.asarray(
                                    [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                     np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                     np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                                tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                                tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                                tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                                tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                                tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                                tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                                tempList.append(tempDict)
                        else:
                            waveInd = np.where((tC < et) & (tC >= etNew2))
                            waterInd = np.where((tFRF < et) & (tFRF >= etNew2))
                            c = c + 1
                            tempDict = dict()
                            tempDict['time'] = tC[waveInd[0]]
                            tempDict['numDays'] = subLength[i]
                            tempDict['hs'] = hsCombined[waveInd[0]]
                            tempDict['tp'] = tpCombined[waveInd[0]]
                            tempDict['dm'] = waveNorm[waveInd[0]]
                            tempDict['res'] = resFRF[waterInd[0]]
                            tempDict['wl'] = wlFRF[waterInd[0]]
                            tempDict['cop'] = np.asarray(
                                [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                                 np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                                 np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                            tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                            tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                            tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                            tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                            tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                            tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                            tempList.append(tempDict)
                    else:
                        waveInd = np.where((tC < et) & (tC >= etNew))
                        waterInd = np.where((tFRF < et) & (tFRF >= etNew))
                        c = c + 1
                        tempDict = dict()
                        tempDict['time'] = tC[waveInd[0]]
                        tempDict['numDays'] = subLength[i]
                        tempDict['hs'] = hsCombined[waveInd[0]]
                        tempDict['tp'] = tpCombined[waveInd[0]]
                        tempDict['dm'] = waveNorm[waveInd[0]]
                        tempDict['res'] = resFRF[waterInd[0]]
                        tempDict['wl'] = wlFRF[waterInd[0]]
                        tempDict['cop'] = np.asarray(
                            [np.nanmin(hsCombined[waveInd[0]]), np.nanmax(hsCombined[waveInd[0]]),
                             np.nanmin(tpCombined[waveInd[0]]), np.nanmax(tpCombined[waveInd[0]]),
                             np.nanmean(waveNorm[waveInd[0]]), np.nanmean(resFRF[waveInd[0]])])
                        tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                        tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                        tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                        tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                        tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                        tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                        tempList.append(tempDict)
                else:
                    #waveInd = np.where((tC < et) & (tC >= etNew))
                    #waterInd = np.where((tFRF < et) & (tFRF >= etNew))
                    c = c + 1
                    tempDict = dict()
                    tempDict['time'] = tC[waveInd[0]]
                    tempDict['numDays'] = subLength[i]
                    tempDict['hs'] = hsCombined[waveInd[0]]
                    tempDict['tp'] = tpCombined[waveInd[0]]
                    tempDict['dm'] = waveNorm[waveInd[0]]
                    tempDict['res'] = resFRF[waterInd[0]]
                    tempDict['wl'] = wlFRF[waterInd[0]]
                    tempDict['cop'] = np.asarray([np.nanmin(hsCombined[waveInd[0]]),np.nanmax(hsCombined[waveInd[0]]),
                                                 np.nanmin(tpCombined[waveInd[0]]),np.nanmax(tpCombined[waveInd[0]]),
                                                 np.nanmean(waveNorm[waveInd[0]]),np.nanmean(resFRF[waveInd[0]])])
                    tempDict['hsMin'] = np.nanmin(hsCombined[waveInd[0]])
                    tempDict['hsMax'] = np.nanmax(hsCombined[waveInd[0]])
                    tempDict['tpMin'] = np.nanmin(tpCombined[waveInd[0]])
                    tempDict['tpMax'] = np.nanmax(tpCombined[waveInd[0]])
                    tempDict['dmMean'] = np.nanmean(waveNorm[waveInd[0]])
                    tempDict['ssMean'] = np.nanmean(resFRF[waveInd[0]])
                    tempList.append(tempDict)
    hydros.append(tempList)




myFmt = mdates.DateFormatter('%d')
#
# bmuPlot = 2
#
# m = np.ceil(np.sqrt(len(hydros[bmuPlot])))
# plt.figure()
# gs = gridspec.GridSpec(int(m), int(m), wspace=0.1, hspace=0.15)
# for i in range(len(hydros[bmuPlot])):
#     ax = plt.subplot(gs[i])
#     ax.plot(hydros[bmuPlot][i]['time'], hydros[bmuPlot][i]['tp'], 'k-')
#     ax.set_ylim([4, 12])
#     ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
#     ax.xaxis.set_major_formatter(myFmt)
#
#
# ### TODO: look at the distributions within each copula
# plt.figure()
# gs = gridspec.GridSpec(10, 7, wspace=0.1, hspace=0.15)
# bins = np.linspace(0,4,25)
# for bmuPlot in range(len(np.unique(bmuGroup))):
#     ax = plt.subplot(gs[bmuPlot])
#     tempHsMin = list()
#     tempHsMax = list()
#     for i in range(len(hydros[bmuPlot])):
#         tempHsMin.append(hydros[bmuPlot][i]['hsMin'])
#         tempHsMax.append(hydros[bmuPlot][i]['hsMax'])
#
#     ax.hist(np.asarray(tempHsMax),bins,alpha=0.5,label='hs max',density=True)
#     ax.hist(np.asarray(tempHsMin),bins,alpha=0.5,label='hs min',density=True)
#     ax.set_ylim([0,2])
# ax.legend(loc='upper right')
#
# plt.figure()
# gs = gridspec.GridSpec(10, 7, wspace=0.1, hspace=0.15)
# bins = np.linspace(3,14,40)
# for bmuPlot in range(len(np.unique(bmuGroup))):
#     ax = plt.subplot(gs[bmuPlot])
#     tempHsMin = list()
#     tempHsMax = list()
#     for i in range(len(hydros[bmuPlot])):
#         tempHsMin.append(hydros[bmuPlot][i]['tpMin'])
#         tempHsMax.append(hydros[bmuPlot][i]['tpMax'])
#
#     ax.hist(np.asarray(tempHsMax),bins,alpha=0.5,label='tp max',density=True)
#     ax.hist(np.asarray(tempHsMin),bins,alpha=0.5,label='tp min',density=True)
#     ax.set_ylim([0,0.6])
# ax.legend(loc='upper right')
#
#
# plt.figure()
# gs = gridspec.GridSpec(10, 7, wspace=0.1, hspace=0.15)
# bins = np.linspace(-0.4,0.8,20)
# for bmuPlot in range(len(np.unique(bmuGroup))):
#     ax = plt.subplot(gs[bmuPlot])
#     tempSS= list()
#     for i in range(len(hydros[bmuPlot])):
#         tempSS.append(hydros[bmuPlot][i]['ssMean'])
#
#     ax.hist(np.asarray(tempSS),bins,alpha=0.5,label='tp min',density=True)
#     ax.set_ylim([0,3.25])
# ax.legend(loc='upper right')



### TODO: make a copula fit for each of the 70 DWTs
copulaData = list()
for i in range(len(np.unique(bmuGroup))):
    tempHydros = hydros[i]
    dataCop = []
    for kk in range(len(tempHydros)):
        if np.isnan(tempHydros[kk]['ssMean']):
            print('skipping a Nan in storm surge')
        else:
            dataCop.append(list([tempHydros[kk]['hsMax'],tempHydros[kk]['hsMin'],tempHydros[kk]['tpMax'],
                       tempHydros[kk]['tpMin'],tempHydros[kk]['dmMean'],tempHydros[kk]['ssMean'],kk]))
    copulaData.append(dataCop)


# bmuData = list()
# for i in range(len(np.unique(bmuGroup))):
#     tempHydros = hydros[i]
#     dataCop = []
#     for kk in range(len(tempHydros)):
#         if np.isnan(tempHydros[kk]['ssMean']):
#             print('skipping a Nan in storm surge')
#         else:
#             dataCop.append(list([tempHydros[kk]['hsMax'],tempHydros[kk]['hsMin'],tempHydros[kk]['tpMax'],
#                        tempHydros[kk]['tpMin'],tempHydros[kk]['dmMean'],tempHydros[kk]['ssMean'],kk]))
#     bmuData.append(dataCop)



# bmuData = list()
# for i in range(len(np.unique(bmuGroup))):
#     tempHydros = hydros[i]
#     dataCop = []
#     for kk in range(len(tempHydros)):
#         if np.isnan(tempHydros[kk]['ssMean']):
#             print('skipping a Nan in storm surge')
#         else:
#             dataCop.append(list([tempHydros[kk]['hsMax'],tempHydros[kk]['hsMin'],tempHydros[kk]['tpMax'],
#                        tempHydros[kk]['tpMin'],tempHydros[kk]['dmMean'],tempHydros[kk]['ssMean'],kk]))
#     bmuData.append(dataCop)


bmuDataNormalized = list()
bmuDataMin = list()
bmuDataMax = list()
bmuDataStd = list()
for i in range(len(np.unique(bmuGroup))):
    maxDm = np.nanmax(np.asarray(copulaData[i])[:,4])
    minDm = np.nanmin(np.asarray(copulaData[i])[:,4])
    stdDm = np.nanstd(np.asarray(copulaData[i])[:,4])
    dmNorm = (np.asarray(copulaData[i])[:,4] - minDm) / (maxDm-minDm)
    maxSs = np.nanmax(np.asarray(copulaData[i])[:,5])
    minSs = np.nanmin(np.asarray(copulaData[i])[:,5])
    stdSs = np.nanstd(np.asarray(copulaData[i])[:,5])
    ssNorm = (np.asarray(copulaData[i])[:,5] - minSs) / (maxSs-minSs)
    bmuDataNormalized.append(np.vstack((dmNorm,ssNorm)).T)
    bmuDataMin.append([minDm,minSs])
    bmuDataMax.append([maxDm,maxSs])
    bmuDataStd.append([stdDm,stdSs])

#
# def closest_node(node, nodes):
#     closest_index = distance.cdist([node], nodes).argmin()
#     return nodes[closest_index], closest_index
#

### TODO: Need to normalize all of the hydrographs with respect to time AND magnitude


normalizedHydros = list()
for i in range(len(np.unique(bmuGroup))):
    tempHydros = hydros[i]
    tempList = list()
    for mm in range(len(tempHydros)):
        tempDict = dict()
        tempDict['hsNorm'] = (tempHydros[mm]['hs'] - tempHydros[mm]['hsMin']) / (tempHydros[mm]['hsMax']- tempHydros[mm]['hsMin'])
        tempDict['tpNorm'] = (tempHydros[mm]['tp'] - tempHydros[mm]['tpMin']) / (tempHydros[mm]['tpMax']- tempHydros[mm]['tpMin'])
        tempDict['timeNorm'] = np.arange(0,1,1/len(tempHydros[mm]['time']))[0:len(tempDict['hsNorm'])]
        tempDict['dmNorm'] = (tempHydros[mm]['dm']) - tempHydros[mm]['dmMean']
        tempDict['ssNorm'] = (tempHydros[mm]['res']) - tempHydros[mm]['ssMean']
        tempList.append(tempDict)
    normalizedHydros.append(tempList)




import pickle

normHydrosPickle = 'normalizedWaveHydrographs.pickle'
outputHydrosNorm = {}
outputHydrosNorm['normalizedHydros'] = normalizedHydros
outputHydrosNorm['bmuDataMin'] = bmuDataMin
outputHydrosNorm['bmuDataMax'] = bmuDataMax
outputHydrosNorm['bmuDataStd'] = bmuDataStd
outputHydrosNorm['bmuDataNormalized'] = bmuDataNormalized

with open(normHydrosPickle,'wb') as f:
    pickle.dump(outputHydrosNorm, f)


hydrosPickle = 'waveHydrographs.pickle'
outputHydros = {}
outputHydros['hydros'] = hydros
with open(hydrosPickle,'wb') as f:
    pickle.dump(outputHydros, f)

copPickle = 'hydrographCopulaData.pickle'
outputCopula = {}
outputCopula['copulaData'] = copulaData
with open(copPickle,'wb') as f:
    pickle.dump(outputCopula, f)


historicalPickle = 'historicalData.pickle'
outputHistorical = {}
outputHistorical['grouped'] = grouped
outputHistorical['groupLength'] = groupLength
outputHistorical['bmuGroup'] = bmuGroup
outputHistorical['timeGroup'] = timeGroup

with open(historicalPickle,'wb') as f:
    pickle.dump(outputHistorical, f)



asdfg

import xarray as xr


def CDF_Distribution(self, vn, vv, xds_GEV_Par, d_shape, i_wt):
    '''
    Switch function: GEV / Empirical / Weibull

    Check variable distribution and calculates CDF

    vn - var name
    vv - var value
    i_wt - Weather Type index
    xds_GEV_Par , d_shape: GEV data used in sigma correlation
    '''

    # get GEV / EMPIRICAL / WEIBULL variables list
    vars_GEV = self.vars_GEV
    vars_EMP = self.vars_EMP
    vars_WBL = self.vars_WBL

    # switch variable name
    if vn in vars_GEV:

        # gev CDF
        sha_g = d_shape[vn][i_wt]
        loc_g = xds_GEV_Par.sel(parameter='location')[vn].values[i_wt]
        sca_g = xds_GEV_Par.sel(parameter='scale')[vn].values[i_wt]
        norm_VV = genextreme.cdf(vv, -1 * sha_g, loc_g, sca_g)

    elif vn in vars_EMP:

        # empirical CDF
        ecdf = ECDF(vv)
        norm_VV = ecdf(vv)

    elif vn in vars_WBL:

        # Weibull CDF
        norm_VV = weibull_min.cdf(vv, *weibull_min.fit(vv))

    return norm_VV


def ICDF_Distribution(self, vn, vv, pb, xds_GEV_Par, i_wt):
    '''
    Switch function: GEV / Empirical / Weibull

    Check variable distribution and calculates ICDF

    vn - var name
    vv - var value
    pb - var simulation probs
    i_wt - Weather Type index
    xds_GEV_Par: GEV parameters
    '''

    # optional empirical var_wt override
    fv = '{0}_{1}'.format(vn, i_wt + 1)
    if fv in self.sim_icdf_empirical_override:
        ppf_VV = Empirical_ICDF(vv, pb)
        return ppf_VV

    # get GEV / EMPIRICAL / WEIBULL variables list
    vars_GEV = self.vars_GEV
    vars_EMP = self.vars_EMP
    vars_WBL = self.vars_WBL

    # switch variable name
    if vn in vars_GEV:

        # gev ICDF
        sha_g = xds_GEV_Par.sel(parameter='shape')[vn].values[i_wt]
        loc_g = xds_GEV_Par.sel(parameter='location')[vn].values[i_wt]
        sca_g = xds_GEV_Par.sel(parameter='scale')[vn].values[i_wt]
        ppf_VV = genextreme.ppf(pb, -1 * sha_g, loc_g, sca_g)

    elif vn in vars_EMP:

        # empirical ICDF
        ppf_VV = Empirical_ICDF(vv, pb)

    elif vn in vars_WBL:

        # Weibull ICDF
        ppf_VV = weibull_min.ppf(pb, *weibull_min.fit(vv))

    return ppf_VV


def Calc_GEVParams(self, xds_KMA_MS, xds_WVS_MS):
    '''
    Fits each WT (KMA.bmus) waves families data to a GEV distribtion
    Requires KMA and WVS families at storms max. TWL

    Returns xarray.Dataset with GEV shape, location and scale parameters
    '''

    vars_gev = self.vars_GEV
    bmus = xds_KMA_MS.bmus.values[:]
    cenEOFs = xds_KMA_MS.cenEOFs.values[:]
    n_clusters = len(xds_KMA_MS.n_clusters)

    xds_GEV_Par = xr.Dataset(
        coords = {
            'n_cluster' : np.arange(n_clusters)+1,
            'parameter' : ['shape', 'location', 'scale'],
        }
    )

    # Fit each wave family var to GEV distribution (using KMA bmus)
    for vn in vars_gev:
        gp_pars = FitGEV_KMA_Frechet(
            bmus, n_clusters, xds_WVS_MS[vn].values[:])

        xds_GEV_Par[vn] = (('n_cluster', 'parameter',), gp_pars)

    return xds_GEV_Par


def fitGEVparams(var):
    '''
    Returns stationary GEV/Gumbel_L params for KMA bmus and varible series

    bmus        - KMA bmus (time series of KMA centroids)
    n_clusters  - number of KMA clusters
    var         - time series of variable to fit to GEV/Gumbel_L

    returns np.array (n_clusters x parameters). parameters = (shape, loc, scale)
    for gumbel distributions shape value will be ~0 (0.0000000001)
    '''

    param_GEV = np.empty((3,))

    # get variable at cluster position
    var_c = var
    var_c = var_c[~np.isnan(var_c)]

    # fit to Gumbel_l and get negative loglikelihood
    loc_gl, scale_gl = gumbel_l.fit(-var_c)
    theta_gl = (0.0000000001, -1*loc_gl, scale_gl)
    nLogL_gl = genextreme.nnlf(theta_gl, var_c)

    # fit to GEV and get negative loglikelihood
    c = -0.1
    shape_gev, loc_gev, scale_gev = genextreme.fit(var_c, c)
    theta_gev = (shape_gev, loc_gev, scale_gev)
    nLogL_gev = genextreme.nnlf(theta_gev, var_c)

    # store negative shape
    theta_gev_fix = (-shape_gev, loc_gev, scale_gev)

    # apply significance test if Frechet
    if shape_gev < 0:

        # TODO: cant replicate ML exact solution
        if nLogL_gl - nLogL_gev >= 1.92:
            param_GEV[i,:] = list(theta_gev_fix)
        else:
            param_GEV[i,:] = list(theta_gl)
    else:
        param_GEV[i,:] = list(theta_gev_fix)

    return param_GEV

def Smooth_GEV_Shape(cenEOFs, param):
    '''
    Smooth GEV shape parameter (for each KMA cluster) by promediation
    with neighbour EOFs centroids

    cenEOFs  - (n_clusters, n_features) KMA centroids
    param    - GEV shape parameter for each KMA cluster

    returns smoothed GEV shape parameter as a np.array (n_clusters)
    '''

    # number of clusters
    n_cs = cenEOFs.shape[0]

    # calculate distances (optimized)
    cenEOFs_b = cenEOFs.reshape(cenEOFs.shape[0], 1, cenEOFs.shape[1])
    D = np.sqrt(np.einsum('ijk, ijk->ij', cenEOFs-cenEOFs_b, cenEOFs-cenEOFs_b))
    np.fill_diagonal(D, np.nan)

    # sort distances matrix to find neighbours
    sort_ord = np.empty((n_cs, n_cs), dtype=int)
    D_sorted = np.empty((n_cs, n_cs))
    for i in range(n_cs):
        order = np.argsort(D[i,:])
        sort_ord[i,:] = order
        D_sorted[i,:] = D[i, order]

    # calculate smoothed parameter
    denom = np.sum(1/D_sorted[:,:4], axis=1)
    param_c = 0.5 * np.sum(np.column_stack(
        [
            param[:],
            param[sort_ord[:,:4]] * (1/D_sorted[:,:4])/denom[:,None]
        ]
    ), axis=1)

    return param_c



def ksdensity_CDF(x):
    '''
    Kernel smoothing function estimate.
    Returns cumulative probability function at x.
    '''

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF at x position (kde.support = x)
    fint = interp1d(kde.support, kde.cdf)

    return fint(x)

def ksdensity_ICDF(x, p):
    '''
    Returns Inverse Kernel smoothing function at p points
    '''

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF to get support values 
    fint = interp1d(kde.cdf, kde.support)

    # ensure p inside kde.cdf
    p[p<np.min(kde.cdf)] = kde.cdf[0]
    p[p>np.max(kde.cdf)] = kde.cdf[-1]

    return fint(p)

def GeneralizedPareto_CDF(x):
    '''
    Generalized Pareto fit
    Returns cumulative probability function at x.
    '''

    # fit a generalized pareto and get params
    shape, _, scale = genpareto.fit(x)

    # get generalized pareto CDF
    cdf = genpareto.cdf(x, shape, scale=scale)

    return cdf

def GeneralizedPareto_ICDF(x, p):
    '''
    Generalized Pareto fit
    Returns inverse cumulative probability function at p points
    '''

    # fit a generalized pareto and get params
    shape, _, scale = genpareto.fit(x)

    # get percent points (inverse of CDF)
    icdf = genpareto.ppf(p, shape, scale=scale)

    return icdf

def Empirical_CDF(x):
    '''
    Returns empirical cumulative probability function at x.
    '''

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    return cdf

def Empirical_ICDF(x, p):
    '''
    Returns inverse empirical cumulative probability function at p points
    '''

    # TODO: build in functionality for a fill_value?

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    # interpolate KDE CDF to get support values 
    fint = interp1d(
        cdf, x,
        fill_value=(np.nanmin(x), np.nanmax(x)),
        #fill_value=(np.min(x), np.max(x)),
        bounds_error=False
    )
    return fint(p)


def copulafit(u, family='gaussian'):
    '''
    Fit copula to data.
    Returns correlation matrix and degrees of freedom for t student
    '''

    rhohat = None  # correlation matrix
    nuhat = None  # degrees of freedom (for t student)

    if family=='gaussian':
        u[u>=1.0] = 0.999999
        inv_n = ndtri(u)
        rhohat = np.corrcoef(inv_n.T)

    elif family=='t':
        raise ValueError("Not implemented")

        # TODO:
        x = np.linspace(np.min(u), np.max(u),100)
        inv_t = np.ndarray((len(x), u.shape[1]))

        for j in range(u.shape[1]):
            param = t.fit(u[:,j])
            t_pdf = t.pdf(x,loc=param[0],scale=param[1],df=param[2])
            inv_t[:,j] = t_pdf

        # TODO CORRELATION? NUHAT?
        rhohat = np.corrcoef(inv_n.T)
        nuhat = None

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return rhohat, nuhat

def copularnd(family, rhohat, n):
    '''
    Random vectors from a copula
    '''

    if family=='gaussian':
        mn = np.zeros(rhohat.shape[0])
        np_rmn = np.random.multivariate_normal(mn, rhohat, n)
        u = norm.cdf(np_rmn)

    elif family=='t':
        # TODO
        raise ValueError("Not implemented")

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return u

def CopulaSimulation(U_data, kernels, num_sim):
    '''
    Fill statistical space using copula simulation

    U_data: 2D nump.array, each variable in a column
    kernels: list of kernels for each column at U_data (KDE | GPareto)
    num_sim: number of simulations
    '''

    # kernel CDF dictionary
    d_kf = {
        'KDE' : (ksdensity_CDF, ksdensity_ICDF),
        'GPareto' : (GeneralizedPareto_CDF, GeneralizedPareto_ICDF),
        'ECDF' : (Empirical_CDF, Empirical_ICDF),
    }

    # check kernel input
    if any([k not in d_kf.keys() for k in kernels]):
        raise ValueError(
            'wrong kernel: {0}, use: {1}'.format(
                kernel, ' | '.join(d_kf.keys())
            )
        )

    # normalize: calculate data CDF using kernels
    U_cdf = np.zeros(U_data.shape) * np.nan
    ic = 0
    for d, k in zip(U_data.T, kernels):
        cdf, _ = d_kf[k]  # get kernel cdf
        U_cdf[:, ic] = cdf(d)
        ic += 1

    # fit data CDFs to a gaussian copula
    rhohat, _ = copulafit(U_cdf, 'gaussian')

    # simulate data to fill probabilistic space
    U_cop = copularnd('gaussian', rhohat, num_sim)

    # de-normalize: calculate data ICDF
    U_sim = np.zeros(U_cop.shape) * np.nan
    ic = 0
    for d, c, k in zip(U_data.T, U_cop.T, kernels):
        _, icdf = d_kf[k]  # get kernel icdf
        U_sim[:, ic] = icdf(d, c)
        ic += 1

    return U_sim


### TODO: copula simulation using GEV params


gevCopulaSims = list()
for i in range(len(np.unique(bmuGroup))):
    tempCopula = np.asarray(copulaData[i])
    kernels = ['KDE','KDE','KDE','KDE','KDE','KDE',]
    samples = CopulaSimulation(tempCopula[:,0:6],kernels,10000)
    gevCopulaSims.append(samples)


# from copula import pyCopula
#
# copulaSims = list()
# for i in range(len(np.unique(bmuGroup))):
#     tempCopula = copulaData[i]
#     cop = pyCopula.Copula(tempCopula)
#     samples = cop.gendata(10000)
#     copulaSims.append(samples)





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


# bmuData = list()
# for i in range(len(np.unique(bmuGroup))):
#     tempHydros = hydros[i]
#     dataCop = []
#     for kk in range(len(tempHydros)):
#         if np.isnan(tempHydros[kk]['ssMean']):
#             print('skipping a Nan in storm surge')
#         else:
#             dataCop.append(list([tempHydros[kk]['hsMax'],tempHydros[kk]['hsMin'],tempHydros[kk]['tpMax'],
#                        tempHydros[kk]['tpMin'],tempHydros[kk]['dmMean'],tempHydros[kk]['ssMean'],kk]))
#     bmuData.append(dataCop)


# bmuDataNormalized = list()
# bmuDataMin = list()
# bmuDataMax = list()
# bmuDataStd = list()
# for i in range(len(np.unique(bmuGroup))):
#     maxDm = np.nanmax(np.asarray(bmuData[i])[:,4])
#     minDm = np.nanmin(np.asarray(bmuData[i])[:,4])
#     stdDm = np.nanstd(np.asarray(bmuData[i])[:,4])
#     dmNorm = (np.asarray(bmuData[i])[:,4] - minDm) / (maxDm-minDm)
#     maxSs = np.nanmax(np.asarray(bmuData[i])[:,5])
#     minSs = np.nanmin(np.asarray(bmuData[i])[:,5])
#     stdSs = np.nanstd(np.asarray(bmuData[i])[:,5])
#     ssNorm = (np.asarray(bmuData[i])[:,5] - minSs) / (maxSs-minSs)
#     bmuDataNormalized.append(np.vstack((dmNorm,ssNorm)).T)
#     bmuDataMin.append([minDm,minSs])
#     bmuDataMax.append([maxDm,maxSs])
#     bmuDataStd.append([stdDm,stdSs])



def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index], closest_index


# ### TODO: Need to normalize all of the hydrographs with respect to time AND magnitude
#
# normalizedHydros = list()
# for i in range(len(np.unique(bmuGroup))):
#     tempHydros = hydros[i]
#     tempList = list()
#     for mm in range(len(tempHydros)):
#         tempDict = dict()
#         tempDict['hsNorm'] = (tempHydros[mm]['hs'] - tempHydros[mm]['hsMin']) / (tempHydros[mm]['hsMax']- tempHydros[mm]['hsMin'])
#         tempDict['tpNorm'] = (tempHydros[mm]['tp'] - tempHydros[mm]['tpMin']) / (tempHydros[mm]['tpMax']- tempHydros[mm]['tpMin'])
#         tempDict['timeNorm'] = np.arange(0,1,1/len(tempHydros[mm]['time']))[0:len(tempDict['hsNorm'])]
#         tempDict['dmNorm'] = (tempHydros[mm]['dm']) - tempHydros[mm]['dmMean']
#         tempDict['ssNorm'] = (tempHydros[mm]['res']) - tempHydros[mm]['ssMean']
#         tempList.append(tempDict)
#     normalizedHydros.append(tempList)
#








asdfg

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




### TODO: verify wave distributions in each DWT













# simHsTest = np.hstack(simHs)
#
# plt.figure()
# plt.plot(newDailyTime,simHsTest)


#
# plt.figure()
# plt.plot(np.asarray(bmuData[tempBmu])[:,4],np.asarray(bmuData[tempBmu])[:,5],'.',color='black')
# plt.plot(stormDetails[4],stormDetails[5],'o',color='red')
# plt.plot(np.asarray(bmuData[tempBmu])[closeIndex,4],np.asarray(bmuData[tempBmu])[closeIndex,5],'.',color='orange')
#




    #
    # etNew = startTimes[index[i]] + timedelta(days=randLength)
    # waveInd = np.where((tC < etNew) & (tC >= st))
    # waterInd = np.where((tFRF < etNew) & (tFRF >= st))
    # deltaDays = et - etNew
    # print('After first cut off = {}'.format(deltaDays.days))
# grouped = [[e[0] for e in d[1]] for d in itertools.groupby(enumerate(bmus), key=operator.itemgetter(1))]
#
# groupLength = np.asarray([len(i) for i in grouped])
# bmuGroup = np.asarray([bmus[i[0]] for i in grouped])
# timeGroup = [np.asarray(midnightTime)[i] for i in grouped]

# startTimes = [i[0] for i in timeGroup]
# endTimes = [i[-1] for i in timeGroup]
#
# hydros = list()
# c = 0
# for p in range(len(np.unique(bmuGroup))):
#     index = np.where((bmuGroup==p))[0][:]
#     subLength = groupLength[index]
#     m = np.ceil(np.sqrt(len(index)))
#     tempList = list()
#     for i in range(len(index)):
#         st = startTimes[index[i]]
#         et = endTimes[index[i]] + timedelta(days=1)
#
#         if et > datetime(2020,1,1):
#             print('done up to 2020')
#         else:
#             waveInd = np.where((tC < et) & (tC >= st))
#             waterInd = np.where((tFRF < et) & (tFRF >= st))






### TODO: load in a BMU simulation








