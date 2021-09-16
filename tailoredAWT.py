import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
   return [datetime.date(d[0], d[1], d[2]) for d in d_vec]

def dateDay2datetimeDatetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime.datetime(d[0], d[1], d[2]) for d in d_vec]



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



# loading in a North Atlantic continuous SLP record without any of the memory built into it
# SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/NorthAtlanticSLPs_June2021_bigger.mat')
SLPs = ReadMatfile('/media/dylananderson/Elements1/NC_climate/NorthAtlanticSLPs_June2021_ESTELA_area_smaller.mat')

X_in = SLPs['X_in']
Y_in = SLPs['Y_in']
SLP = SLPs['slp_mem']
SLPtime = SLPs['time']
sea = SLPs['sea_sq']
XRsq = SLPs['XRsq']
YRsq = SLPs['YRsq']
# Applying PC to the those such that every day has a value in the first couple PCs..
SlpGrd = SLP
SlpMean = np.mean(SLP,axis=0)
SlpStd = np.std(SLP,axis=0)
SlpNorm = (SLP[:,:] - SlpMean) / SlpStd
SlpNorm[np.isnan(SlpNorm)] = 0

# principal components analysis
ipca = PCA(n_components=min(SlpNorm.shape[0], SlpNorm.shape[1]))
PCs = ipca.fit_transform(SlpNorm)
EOFs = ipca.components_
variance = ipca.explained_variance_
nPercent = variance / np.sum(variance)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0
nterm = np.where(APEV <= 0.95 * 100)[0][-1]

# The first 4 explain 51.5% of the variance... good enough?

# Lets load in the DWT assignments and separate the PCs by their DWT number

with open(r"dwts49Clusters.pickle", "rb") as input_file:
# with open(r"dwtsAll6TCTracksClusters.pickle", "rb") as input_file:
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


etcolors = cm.viridis(np.linspace(0, 1, 70-20))
tccolors = np.flipud(cm.autumn(np.linspace(0,1,21)))
dwtcolors = np.vstack((etcolors,tccolors[1:,:]))


# Let's trim these things so they are the same length?
# Just need to remove January from SLPs...
# Or maybe take January-May away, and Feb-May away from the bmus?
# bmus_dates = dateDay2datetimeDatetime(timeDWTs)
# bmus_dates = bmus_dates[120:]
bmus_dates = timeDWTs[120:,:]
bmus = bmus[120:]
SLPtime = SLPtime[151:]
PCs = PCs[151:,:]
# SLPtime = SLPtime[120:]
# PCs = PCs[120:,:]

DailyPCs = PCs
DailyDatesMatrix = bmus_dates
DailySortedBmus = bmus
# DailyDatesMatrix = np.array([dt2cal(hh) for hh in bmus_dates])

s1 = np.full([len(np.unique(DailySortedBmus)), len(np.unique(DailyDatesMatrix[:,0])) - 1], np.nan)
# June/June #!!!!
for i in range(len(np.unique(DailyDatesMatrix[:,0])) - 1):
    s = np.where((DailyDatesMatrix[:,0] == np.unique(DailyDatesMatrix[:,0])[i]) & (DailyDatesMatrix[:,1] == 6))
    ss = np.where((DailyDatesMatrix[:,0] == np.unique(DailyDatesMatrix[:,0])[i] + 1) & (DailyDatesMatrix[:,1] == 5))
    for j in range(len(np.unique(DailySortedBmus))):
        s1[j, i] = len(np.where(DailySortedBmus[s[0][0]:ss[0][-1]] == j)[0])


PC1 = np.full([len(np.unique(DailySortedBmus)), len(np.unique(DailyDatesMatrix[:,0])) - 1], np.nan)
PC2 = np.full([len(np.unique(DailySortedBmus)), len(np.unique(DailyDatesMatrix[:,0])) - 1], np.nan)
PC3 = np.full([len(np.unique(DailySortedBmus)), len(np.unique(DailyDatesMatrix[:,0])) - 1], np.nan)
PC4 = np.full([len(np.unique(DailySortedBmus)), len(np.unique(DailyDatesMatrix[:,0])) - 1], np.nan)
PC5 = np.full([len(np.unique(DailySortedBmus)), len(np.unique(DailyDatesMatrix[:,0])) - 1], np.nan)
PC6 = np.full([len(np.unique(DailySortedBmus)), len(np.unique(DailyDatesMatrix[:,0])) - 1], np.nan)

# June/June #!!!!
for i in range(len(np.unique(DailyDatesMatrix[:,0])) - 1):
    s = np.where((DailyDatesMatrix[:,0] == np.unique(DailyDatesMatrix[:,0])[i]) & (DailyDatesMatrix[:,1] == 6))
    ss = np.where((DailyDatesMatrix[:,0] == np.unique(DailyDatesMatrix[:,0])[i] + 1) & (DailyDatesMatrix[:,1] == 5))
    for j in range(len(np.unique(DailySortedBmus))):
        yearlyBmus = DailySortedBmus[s[0][0]:ss[0][-1]]     # lets get the bmus for this year
        yearlyPCs = DailyPCs[s[0][0]:ss[0][-1],:]           # lets get the PCs for this year
        indBmus = np.where(yearlyBmus == j)[0]         # which of those DWTs do we care about in this iteration
        # PC1[indDWT,i] = np.nanmean(DailyPCs[0,indBmus])
        # PC2[indDWT,i] = np.nanmean(DailyPCs[1,indBmus])
        # PC3[indDWT,i] = np.nanmean(DailyPCs[2,indBmus])
        PC1[j,i] = np.nansum(yearlyPCs[indBmus,0])
        PC2[j,i] = np.nansum(yearlyPCs[indBmus,1])
        PC3[j,i] = np.nansum(yearlyPCs[indBmus,2])
        PC4[j,i] = np.nansum(yearlyPCs[indBmus,3])
        PC5[j,i] = np.nansum(yearlyPCs[indBmus,4])
        PC6[j,i] = np.nansum(yearlyPCs[indBmus,5])


npercent = nPercent
tempPC1 = np.nansum(PC1,axis=0)
tempPC2 = np.nansum(PC2,axis=0)
tempPC3 = np.nansum(PC3,axis=0)
tempPC4 = np.nansum(PC4,axis=0)
tempPC5 = np.nansum(PC5,axis=0)
tempPC6 = np.nansum(PC6,axis=0)

normPC1 = np.divide(tempPC1,np.nanmax(tempPC1))*npercent[0]
normPC2 = np.divide(tempPC2,np.nanmax(tempPC2))*npercent[1]
normPC3 = np.divide(tempPC3,np.nanmax(tempPC3))*npercent[2]
normPC4 = np.divide(tempPC4,np.nanmax(tempPC4))*npercent[3]
normPC5 = np.divide(tempPC5,np.nanmax(tempPC5))*npercent[4]
normPC6 = np.divide(tempPC6,np.nanmax(tempPC6))*npercent[5]

pcAggregates = np.full((len(normPC1),5),np.nan)
pcAggregates[:,0] = normPC1
pcAggregates[:,1] = normPC2
pcAggregates[:,2] = normPC3
pcAggregates[:,3] = normPC4
pcAggregates[:,4] = normPC5
# pcAggregates[:,5] = normPC6




n_clusters = 6

kmeans = KMeans(n_clusters, init='k-means++', random_state=100)  # 80

n_components = 5 # !!!!
data = pcAggregates#[:, 0:n_components]

#    data1=data/np.std(data,axis=0)

awt_bmus = kmeans.fit_predict(data)



fig = plt.figure(figsize=[14, 9])
gs2 = gridspec.GridSpec(n_components + 1, 1)
for nn in range(n_components):
    ax2 = fig.add_subplot(gs2[nn])
    ax2.plot(np.unique(DailyDatesMatrix[:,0])[:-1], pcAggregates[:, nn], 'k.-', linewidth=1.8, markersize=8)

    ax2.set_ylabel('PC-' + str(nn + 1), fontsize=13)
    ax2.grid('minor')
    ax2.set_xticklabels([])

ax2 = fig.add_subplot(gs2[nn + 1])
ax2.plot(np.unique(DailyDatesMatrix[:,0])[:-1], awt_bmus + 1, 'k.:', linewidth=1.8, markersize=10, color='grey')




dailyAWT = np.ones((len(DailySortedBmus),))
for i in range(len(awt_bmus)):
    s = np.where((DailyDatesMatrix[:,0] == np.unique(DailyDatesMatrix[:,0])[i]) & (DailyDatesMatrix[:,1] == 6))
    ss = np.where((DailyDatesMatrix[:,0] == np.unique(DailyDatesMatrix[:,0])[i] + 1) & (DailyDatesMatrix[:,1] == 5))
    dailyAWT[s[0][0]:ss[0][-1]+1] = awt_bmus[i]*dailyAWT[s[0][0]:ss[0][-1]+1]




awtSLPs = SLP[120:,:]/100 - np.mean(SLP[120:,:],axis=0)/100
fig = plt.figure(figsize=(10, 6))

gs1 = gridspec.GridSpec(2, 3)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
for i in range(len(np.unique(awt_bmus))):

    m, n = np.shape(XRsq)
    ind = np.where((dailyAWT == i))
    avgSLP = np.mean(awtSLPs[ind[0],:],axis=0)
    wt = np.nan * np.ones((m*n,))
    for j in range(len(avgSLP)):
        wt[sea[j]-1] = avgSLP[j]
    spatialField = wt.reshape(n,m)

    ax = plt.subplot(gs1[i])
    # ax = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
    clevels = np.arange(-9, 9, 1)

    m = Basemap(projection='merc', llcrnrlat=-5, urcrnrlat=55, llcrnrlon=255, urcrnrlon=360, lat_ts=10, resolution='c')
    m.fillcontinents(color=[0.5,0.5,0.5])
    cx, cy = m(XRsq, YRsq)
    m.drawcoastlines()
    # m.bluemarble()
    CS = m.contourf(cx, cy, spatialField.T, clevels, vmin=-9, vmax=9, cmap=cm.RdBu_r, shading='gouraud')
    tx, ty = m(320, -0)
    parallels = np.arange(0,360,10)
    # m.drawparallels(parallels,labels=[True,True,True,False],textcolor='white')
    m.drawparallels(parallels,labels=[False,False,False,False],textcolor='white')

    # ax.text(tx, ty, '{}'.format((group_size[num])))
    meridians = np.arange(0,360,20)
    # m.drawmeridians(meridians,labels=[True,True,True,True],textcolor='white')
    m.drawmeridians(meridians,labels=[False,False,False,False],textcolor='white')

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(CS, cax=cbar_ax)
    cbar.set_label('SLP (mbar)')
    #    cb = plt.colorbar(CS)
    cbar.set_clim(-9.0, 9.0)




