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




# with open(r"dwtsAll6TCTracksALLDATA.pickle", "rb") as input_file:
with open(r"dwts49ALLDATA.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

SlpGrdMeanET = historicalDWTs['SlpGrdMean']
SlpGrdStdET = historicalDWTs['SlpGrdStd']
sorted_centroidsET = historicalDWTs['sorted_centroids']
X_inET = historicalDWTs['X_in']
Y_inET = historicalDWTs['Y_in']
kma_orderET = historicalDWTs['kma_order']
SLPET = historicalDWTs['SLP']
group_sizeET = historicalDWTs['group_size']
timeDWTs = historicalDWTs['SLPtime']
dwtBmus = historicalDWTs['bmus_corrected']


# with open(r"dwtsOfExtraTropicalDays.pickle", "rb") as input_file:
with open(r"dwtsOfExtraTropicalDays21Clusters.pickle", "rb") as input_file:
   historicalTWTs = pickle.load(input_file)
SlpGrdMeanTC = historicalTWTs['SlpGrdMean']
SlpGrdStdTC = historicalTWTs['SlpGrdStd']
sorted_centroidsTC = historicalTWTs['sorted_centroids']
X_inTC = historicalTWTs['X_in']
Y_inTC = historicalTWTs['Y_in']
kma_orderTC = historicalTWTs['kma_order']
SLPTC = historicalTWTs['SLP']
group_sizeTC = historicalTWTs['group_size']
twtBmus = historicalTWTs['bmus_corrected']
twtOrder = historicalTWTs['kma_order']
tcIndices = historicalTWTs['tcIndices']
TIMEtcs = historicalTWTs['TIMEtcs']


repmatDesviacionET = np.tile(SlpGrdStdET, (49,1))
repmatMediaET = np.tile(SlpGrdMeanET, (49,1))
Km_ET = np.multiply(sorted_centroidsET,repmatDesviacionET) + repmatMediaET
[mK, nK] = np.shape(Km_ET)
Km_slpET = Km_ET[:,0:int(nK/2)]
Km_grdET = Km_ET[:,int(nK/2):]
X_BET = X_inET
Y_BET = Y_inET
#SLP_C = SLP
Km_slpET = Km_slpET[:,0:len(X_BET)]
Km_grdET = Km_grdET[:,0:len(X_BET)]


repmatDesviacionTC = np.tile(SlpGrdStdTC, (21,1))
repmatMediaTC = np.tile(SlpGrdMeanTC, (21,1))
Km_slpTC = np.multiply(sorted_centroidsTC,repmatDesviacionTC) + repmatMediaTC
XsTC = np.arange(np.min(X_inTC),np.max(X_inTC),2)
YsTC = np.arange(np.min(Y_inTC),np.max(Y_inTC),2)
lenXBTC = len(X_inTC)
[XRTC,YRTC] = np.meshgrid(XsTC,YsTC)



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

bmus_dateTimes = dateDay2datetime(timeDWTs)

bmus = bmus[120:]
timeDWTs = timeDWTs[120:]
bmus_dates = bmus_dates[120:]
bmus_dateTimes = bmus_dateTimes[120:]

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
offpos = np.where((waveNorm>90))
offneg = np.where((waveNorm<-90))
waveNorm[offpos[0]] = waveNorm[offpos[0]]*0
waveNorm[offneg[0]] = waveNorm[offneg[0]]*0

waveNormSwell = dmSwellCombined - 72
negSwell = np.where((waveNormSwell > 180))
waveNormSwell[negSwell[0]] = waveNormSwell[negSwell[0]]-360
offposSwell = np.where((waveNormSwell>90))
offnegSwell = np.where((waveNormSwell<-90))
waveNormSwell[offposSwell[0]] = waveNormSwell[offposSwell[0]]*0
waveNormSwell[offnegSwell[0]] = waveNormSwell[offnegSwell[0]]*0

waveNormWindsea = dmWindseaCombined - 72
negWindsea = np.where((waveNormWindsea > 180))
waveNormWindsea[negWindsea[0]] = waveNormWindsea[negWindsea[0]]-360
offposWindsea = np.where((waveNormWindsea>90))
offnegWindsea = np.where((waveNormWindsea<-90))
waveNormWindsea[offposWindsea[0]] = waveNormWindsea[offposWindsea[0]]*0
waveNormWindsea[offnegWindsea[0]] = waveNormWindsea[offnegWindsea[0]]*0

lwpC = 1025*np.square(hsCombined)*tpCombined*(9.81/(64*np.pi))*np.cos(waveNorm*(np.pi/180))*np.sin(waveNorm*(np.pi/180))
weC = np.square(hsCombined)*tpCombined

lwpSwell = 1025*np.square(hsSwellCombined)*tpSwellCombined*(9.81/(64*np.pi))*np.cos(waveNormSwell*(np.pi/180))*np.sin(waveNormSwell*(np.pi/180))
weSwell = np.square(hsSwellCombined)*tpSwellCombined

lwpWindsea = 1025*np.square(hsWindseaCombined)*tpWindseaCombined*(9.81/(64*np.pi))*np.cos(waveNormWindsea*(np.pi/180))*np.sin(waveNormWindsea*(np.pi/180))
weWindsea = np.square(hsWindseaCombined)*tpWindseaCombined
tWave = [datetime.fromtimestamp(x) for x in timeWave]

tC = np.array(tWave)




# DWT = ReadMatfile('/media/dylananderson/Elements1/NC_climate/Nags_Head_DWTS_25_w50minDates_plus5TCs.mat')
#
# PCA = ReadMatfile('/media/dylananderson/Elements1/NC_climate/Nags_Head_SLPS_1degree_memory.mat')

# mycols = ReadMatfile('/media/dylananderson/Elements1/shusin6_contents/codes/mycmap_col.mat')
# mycmap = mycols['mycmap_col']
# Need to get the dates for the bmus into the correct format (datetime)
def datevec2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime(d[0], d[1], d[2], d[3], d[4]) for d in d_vec]



dwtDates = bmus_dateTimes #np.array(datevec2datetime(DWT['DWT']['dates']))
dwtBmus = bmus #DWT['DWT']['bmus']
# removeTooSoon = np.where(dwtDates < tC[-45])
#
dwtTimes = dwtDates #[removeTooSoon]
dwtBMUS = dwtBmus+1 #[removeTooSoon]
# Next, we need to split the waves up into each DWT
numDWTs = 70
dwtHs = []
dwtMaxHs = []
dwtHsSwell = []
dwtHsWindsea = []
dwtTp = []
dwtTpSwell = []
dwtTpWindsea = []
dwtDm = []
dwtDmSwell = []
dwtDmWindsea = []
dwtLWP = []
dwtWE = []
dwtT = []
for xx in range(numDWTs):
    tempHs = []
    tempMaxHs = []
    tempHsSwell = []
    tempHsWindsea = []
    tempTpSwell = []
    tempTpWindsea = []
    tempDmSwell = []
    tempDmWindsea = []
    tempTp = []
    tempDm = []
    tempLWP = []
    tempWE = []
    tempTi = []

    wInd = np.where((dwtBMUS[0:-1] == (xx+1)))
    for tt in range(len(wInd[0])):
        tempT = np.where((tC < dwtTimes[wInd[0][tt]+1]) & (tC > dwtTimes[wInd[0][tt]]))
        if len(tempT[0]) > 0:
            tempMaxHs = np.append(tempMaxHs, np.nanmax(hsCombined[tempT]))

        tempHs = np.append(tempHs, hsCombined[tempT])
        tempHsSwell = np.append(tempHsSwell, hsSwellCombined[tempT])
        tempHsWindsea = np.append(tempHsWindsea, hsWindseaCombined[tempT])
        tempTpSwell = np.append(tempTpSwell, tpSwellCombined[tempT])
        tempTpWindsea = np.append(tempTpWindsea, tpWindseaCombined[tempT])
        tempDmSwell = np.append(tempDmSwell, waveNormSwell[tempT])
        tempDmWindsea = np.append(tempDmWindsea, waveNormWindsea[tempT])
        tempTp = np.append(tempTp, tpCombined[tempT])
        tempDm = np.append(tempDm, waveNorm[tempT])
        tempLWP = np.append(tempLWP, lwpC[tempT])
        tempWE = np.append(tempWE, weC[tempT])
        tempTi = np.append(tempTi, tC[tempT])

    dwtHs.append(tempHs)
    dwtMaxHs.append(tempMaxHs)
    dwtHsWindsea.append(tempHsWindsea)
    dwtHsSwell.append(tempHsSwell)
    dwtTpWindsea.append(tempTpWindsea)
    dwtTpSwell.append(tempTpSwell)
    dwtDmWindsea.append(tempDmWindsea)
    dwtDmSwell.append(tempDmSwell)
    dwtTp.append(tempTp)
    dwtDm.append(tempDm)
    dwtLWP.append(tempLWP)
    dwtWE.append(tempWE)
    dwtT.append(tempTi)



order = np.arange(0,70,1)
order = np.hstack((kma_orderET,kma_orderTC))
#order = DWT['DWT']['order']
meanDWTHs = np.zeros((np.shape(order)))
for xx in range(numDWTs):
    data = dwtHs[xx]
    meanDWTHs[xx] = np.nanmean(data)

# newOrder = np.argsort(meanDWTHs)
#
# orderedDWTs = np.zeros((np.shape(dwtBMUS)))
# for xx in range(numDWTs):
#     dateInd = np.where((dwtBMUS == xx))
#     orderedDWTs[dateInd] = newOrder[xx]


#plt.style.use('dark_background')


etcolors = cm.viridis(np.linspace(0, 1, 70-20))
tccolors = np.flipud(cm.autumn(np.linspace(0,1,21)))
dwtcolors = np.vstack((etcolors,tccolors[1:,:]))



dist_space = np.linspace(0, 4, 80)
fig = plt.figure(figsize=(10,10))
gs2 = gridspec.GridSpec(10, 7)

colorparam = np.zeros((numDWTs,))
counter = 0
plotIndx = 0
plotIndy = 0
for xx in range(numDWTs):
    #dwtInd = xx
    dwtInd = xx#order[xx]
    #dwtInd = newOrder[xx]

    #ax = plt.subplot2grid((6, 5), (plotIndx, plotIndy), rowspan=1, colspan=1)
    ax = plt.subplot(gs2[xx])

    # normalize = mcolors.Normalize(vmin=np.min(colorparams), vmax=np.max(colorparams))
    normalize = mcolors.Normalize(vmin=.25, vmax=2.5)

    ax.set_xlim([0, 3])
    ax.set_ylim([0, 2])
    data = dwtHs[dwtInd]
    if len(data) > 0:
        kde = gaussian_kde(data)
        colorparam[counter] = np.nanmean(data)
        colormap = cm.Reds
        color = colormap(normalize(colorparam[counter]))
        ax.plot(dist_space, kde(dist_space), linewidth=1, color=color)
        ax.spines['bottom'].set_color([0.5, 0.5, 0.5])
        ax.spines['top'].set_color([0.5, 0.5, 0.5])
        ax.spines['right'].set_color([0.5, 0.5, 0.5])
        ax.spines['left'].set_color([0.5, 0.5, 0.5])
        # ax.text(1.8, 1, np.round(colorparam*100)/100, fontweight='bold')

    else:
        ax.spines['bottom'].set_color([0.3, 0.3, 0.3])
        ax.spines['top'].set_color([0.3, 0.3, 0.3])
        ax.spines['right'].set_color([0.3, 0.3, 0.3])
        ax.spines['left'].set_color([0.3, 0.3, 0.3])

    if plotIndx < 9:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    counter = counter + 1
    if plotIndy < 6:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
    print(plotIndy, plotIndx)

plt.show()
s_map = cm.ScalarMappable(norm=normalize, cmap=colormap)
s_map.set_array(colorparam)
fig.subplots_adjust(right=0.86)
cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
cbar = fig.colorbar(s_map, cax=cbar_ax)
cbar.set_label('Mean Hs (m)')





