import numpy as np
import pickle
import datetime
from dateutil.relativedelta import relativedelta
import random
from matplotlib import gridspec
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.cluster import KMeans, MiniBatchKMeans
from mpl_toolkits.basemap import Basemap
import os
import scipy.io as sio
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.stats import norm, genpareto, t
from scipy.special import ndtri  # norm inv
import matplotlib.dates as mdates
from scipy.spatial import distance
from scipy.stats import  genextreme, gumbel_l, spearmanr, norm, weibull_min
from itertools import groupby
import matplotlib.cm as cm


def running_mean(x, N, mode_str='mean'):
    '''
    computes a running mean (also known as moving average)
    on the elements of the vector X. It uses a window of 2*M+1 datapoints

    As always with filtering, the values of Y can be inaccurate at the
    edges. RUNMEAN(..., MODESTR) determines how the edges are treated. MODESTR can be
    one of the following strings:
      'edge'    : X is padded with first and last values along dimension
                  DIM (default)
      'zeros'   : X is padded with zeros
      'ones'    : X is padded with ones
      'mean'    : X is padded with the mean along dimension DIM

    X should not contains NaNs, yielding an all NaN result.
    '''

    # if nan in data, return nan array
    if np.isnan(x).any():
        return np.full(x.shape, np.nan)

    nn = 2*N+1

    if mode_str == 'zeros':
        x = np.insert(x, 0, np.zeros(N))
        x = np.append(x, np.zeros(N))

    elif mode_str == 'ones':
        x = np.insert(x, 0, np.ones(N))
        x = np.append(x, np.ones(N))

    elif mode_str == 'edge':
        x = np.insert(x, 0, np.ones(N)*x[0])
        x = np.append(x, np.ones(N)*x[-1])

    elif mode_str == 'mean':
        x = np.insert(x, 0, np.ones(N)*np.mean(x))
        x = np.append(x, np.ones(N)*np.mean(x))


    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[nn:] - cumsum[:-nn]) / float(nn)


with open(r"sstWTsPCsAndAllData.pickle", "rb") as input_file:
   sstAtlantic = pickle.load(input_file)

PCs = sstAtlantic['PCs']
EOFs = sstAtlantic['EOFs']
nPercent = sstAtlantic['nPercent']
awt_bmus = sstAtlantic['awt_bmus']
n_components = sstAtlantic['n_components']
variance = sstAtlantic['variance']
ocean = sstAtlantic['ocean']
land = sstAtlantic['land']
realDataAnoms = sstAtlantic['realDataAnoms']
tempdata_runavg = sstAtlantic['tempdata_runavg']
collapsed = sstAtlantic['collapsed']
annual = sstAtlantic['annual']
annualTime = sstAtlantic['annualTime']
subset = sstAtlantic['subset']
#data = sstAtlantic['data']

PC1 = PCs[:,0]
PC2 = PCs[:,1]
PC3 = PCs[:,2]

npercent = nPercent

normPC1 = np.divide(PC1,np.nanmax(PC1))*npercent[0]
normPC2 = np.divide(PC2,np.nanmax(PC2))*npercent[1]
normPC3 = np.divide(PC3,np.nanmax(PC3))*npercent[2]

n_components = 3 # !!!!

pcAggregates = np.full((len(normPC1),n_components),np.nan)
pcAggregates[:,0] = normPC1
pcAggregates[:,1] = normPC2
pcAggregates[:,2] = normPC3


num_clusters = 6



### TODO: generate new clusters in this script

n_clusters = 6
kmeans = KMeans(n_clusters, init='k-means++', random_state=100)  # 80
n_components = 3 # !!!!
data = pcAggregates#PCs[:, 0:n_components]
data1=data/np.std(data,axis=0)
awt_bmus_og = kmeans.fit_predict(data1)
# awt_bmus2 = awt_bmus
awt_bmus2 = np.nan*np.ones((np.shape(awt_bmus_og)))
order = [0,4,5,3,2,1]
for hh in np.arange(0,6):
    indexOR = np.where(awt_bmus==order[hh])
    awt_bmus2[indexOR] = np.ones((len(indexOR[0],)))*hh
awt_bmus = awt_bmus2

### TODO: Plot the AWT's spatially
# plt.figure()
# p1 = plt.subplot2grid((1, 1), (0, 0))
# spatialField = np.reshape(annual[:,98],(36,33))#np.reshape(var_anom_mean.values,(33,36))
# Xs = subset.longitude.values
# Ys = subset.latitude.values
# [XR, YR] = np.meshgrid(Xs, Ys)
# m = Basemap(projection='merc', llcrnrlat=-40, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10, resolution='c')
# m.drawcoastlines()
# cx, cy = m(XR, YR)
# CS = m.contour(cx, cy, spatialField.T, np.arange(-2,2,.2), cmap=cm.RdBu_r, shading='gouraud')
# plt.colorbar(CS)


plt.figure()
gs2 = gridspec.GridSpec(2,3)
for hh in np.unique(awt_bmus):
    indexAWT = np.where(awt_bmus2 == hh)
    # rectField = np.nanmean(subset['SST'][:, :, indexAWT[0]], axis=2)
    #rectField = np.nanmean(tempdata_runavg[:, :, indexAWT[0]], axis=2)
    rectField = np.reshape(np.nanmean(annual[:,indexAWT[0]],axis=1),(36,33))
    ax = plt.subplot(gs2[int(hh)])
    Xs = subset.longitude.values
    Ys = subset.latitude.values
    [XR, YR] = np.meshgrid(Xs, Ys)
    m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10, resolution='c')
    m.drawcoastlines()
    cx, cy = m(XR, YR)
    CS = m.contour(cx, cy, rectField.T, np.arange(-0.8,0.8,.05), cmap=cm.RdBu_r, shading='gouraud')
    ax.set_title('AWT #{} = {} years'.format(int(hh),len(indexAWT[0])))

plt.colorbar(CS,ax=ax)

### TODO: Order the AWT's with known AMO
## NAO AS AN INDEX

with open('/home/dylananderson/projects/atlanticClimate/amo.txt', 'r') as fd:
    c = 0
    dataAMO = list()
    for line in fd:
        splitLine = line.split(',')
        #secondSplit = splitLine[-1].split('/')
        dataAMO.append([h for h in [float(r) for r in splitLine[1:]]])
amo = np.asarray(dataAMO).flatten()[0:-9]

dt = datetime.date(1870, 1, 1)
end = datetime.date(2021, 4, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(months=1)
amoTime = []
while dt < end:
    amoTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step

d1 = datetime.datetime(1979, 6, 1)
dt = datetime.datetime(1979, 6, 1)
end = datetime.datetime(2021, 6, 2)
# step = datetime.timedelta(months=1)
step = relativedelta(days=1)
dailyTime = []
while dt < end:
    dailyTime.append(dt)
    dt += step

DailyDatesMatrix = np.array([[r.year,r.month,r.day] for r in dailyTime])


signal = running_mean(amo,13)
pos_signal = signal.copy()
neg_signal = signal.copy()
pos_signal[pos_signal <= 0] = np.nan
neg_signal[neg_signal > 0] = np.nan


plt.figure()
ax1 = plt.subplot2grid((2,1),(0,0),rowspan=1,colspan=1)
ax1.plot(amoTime,amo,'k')
ax1.plot(amoTime,pos_signal,'r')
ax1.plot(amoTime,neg_signal,'b')
ax1.set_xlim([amoTime[0],amoTime[-1]])
ax2 = plt.subplot2grid((2,1),(1,0),rowspan=1,colspan=1)
ax2.plot(annualTime,awt_bmus2)
ax2.set_xlim([amoTime[0],amoTime[-1]])




### TODO: convert AWT's to daily values...


dailyAWT = np.ones((len(dailyTime),))
dailyPC1 = np.ones((len(dailyTime),))
dailyPC2 = np.ones((len(dailyTime),))
dailyPC3 = np.ones((len(dailyTime),))

anIndex = np.where(np.array(annualTime) >= datetime.datetime(1979,5,31))
subsetAnnualTime = np.array(annualTime)[anIndex]
#subsetAnnualTime = np.array(annualTime)
subsetAWT = awt_bmus2[anIndex]
# subsetPCs = pcAggregates[anIndex[0],:]#PCs[anIndex,:]
subsetPCs = PCs[anIndex[0],:]

for i in range(len(subsetAWT)):
    sSeason = np.where((DailyDatesMatrix[:, 0] == subsetAnnualTime[i].year) & (DailyDatesMatrix[:, 1] == subsetAnnualTime[i].month) & (DailyDatesMatrix[:, 2] == 1))
    # if i == 168:
    #     ssSeason = np.where((DailyDatesMatrix[:, 0] == subsetAnnualTime[i].year) & (DailyDatesMatrix[:, 1] == (subsetAnnualTime[i].month + 2)) & (DailyDatesMatrix[:, 2] == 31))
    # else:
    #     if subsetAnnualTime[i].month < 10:
    #         ssSeason = np.where((DailyDatesMatrix[:, 0] == subsetAnnualTime[i].year) & (DailyDatesMatrix[:, 1] == (subsetAnnualTime[i].month + 3)) & (DailyDatesMatrix[:, 2] == 1))
    #     else:
    #         ssSeason = np.where((DailyDatesMatrix[:, 0] == (subsetAnnualTime[i].year + 1)) & (DailyDatesMatrix[:, 1] == 3) & (DailyDatesMatrix[:, 2] == 1))
    ssSeason = np.where((DailyDatesMatrix[:, 0] == subsetAnnualTime[i].year+1) & (DailyDatesMatrix[:, 1] == subsetAnnualTime[i].month) & (DailyDatesMatrix[:, 2] == 1))


    dailyAWT[sSeason[0][0]:ssSeason[0][0]+1] = subsetAWT[i]*dailyAWT[sSeason[0][0]:ssSeason[0][0]+1]
    dailyPC1[sSeason[0][0]:ssSeason[0][0]+1] = subsetPCs[i,0]*np.ones(len(dailyAWT[sSeason[0][0]:ssSeason[0][0]+1]),)
    dailyPC2[sSeason[0][0]:ssSeason[0][0]+1] = subsetPCs[i,1]*np.ones(len(dailyAWT[sSeason[0][0]:ssSeason[0][0]+1]),)
    dailyPC3[sSeason[0][0]:ssSeason[0][0]+1] = subsetPCs[i,2]*np.ones(len(dailyAWT[sSeason[0][0]:ssSeason[0][0]+1]),)
    # dailyPC4[sSeason[0][0]:ssSeason[0][0]+1] = normPC4[i]*np.ones(len(dailyAWT[sSeason[0][0]:ssSeason[0][0]+1]),)


### TODO: make a markov chain of the AWT clusters

chain = {}
n_words = len(awt_bmus)
for i, key1 in enumerate(awt_bmus):
    if n_words > i + 2:
        key2 = awt_bmus[i + 1]
        word = awt_bmus[i + 2]
        if (key1, key2) not in chain:
            chain[(key1, key2)] = [word]
        else:
            chain[(key1, key2)].append(word)

print('Chain size: {0} distinct bmu pairs.'.format(len(chain)))

chain3 = {}
n_words = len(awt_bmus)
for i, key1 in enumerate(awt_bmus):
    if n_words > i + 3:
        key2 = awt_bmus[i + 1]
        key3 = awt_bmus[i + 2]
        word = awt_bmus[i + 3]
        if (key1, key2, key3) not in chain3:
            chain3[(key1, key2, key3)] = [word]
        else:
            chain3[(key1, key2, key3)].append(word)
print('Chain size: {0} distinct bmu pairs.'.format(len(chain3)))


# groups = [len(np.where(kk==awt_bmus)[0]) for kk in np.unique(awt_bmus)]
# print(groups)
# seasonalMonth = np.array([i.month for i in annualTime])
#
# monthNumber = [seasonalMonth[np.where(kk==awt_bmus)[0]] for kk in np.unique(awt_bmus)]
#
# mwtByMonth = [awt_bmus[np.where(kk==seasonalMonth[0:-1])[0]]for kk in np.unique(seasonalMonth)]
#
# seasonalMonthSim = np.array([i.month for i in dates_sim])


sim_num = 100
sim_years = 500
evbmus_sim = np.nan*np.ones((sim_num,(sim_years)))
key = (awt_bmus[-2], awt_bmus[-1])
for gg in range(sim_num):
    bmu_sim = [awt_bmus[-2], awt_bmus[-1]]
    c = 2
    while len(bmu_sim) < (sim_years):
        w = random.choice(chain[key])
        # temp = chain[key]
        # counter = 0
        # while w not in mwtByMonth[seasonalMonthSim[c]-1]:
        #     print('stuck trying to put {} in {}'.format(w,seasonalMonthSim[c]))
        #     counter = counter + 1
        #     if counter < 20:
        #         w = random.choice(chain[key])
        #     else:
        #         newW = random.choice(mwtByMonth[seasonalMonthSim[c] - 1])
        #         newWminus1 = random.choice(mwtByMonth[seasonalMonthSim[c] - 2])
        #         key = (newWminus1,newW)
        #         w = newW
        bmu_sim.append(w)
        key = (key[1], w)
        c = c + 1
    evbmus_sim[gg, :] = bmu_sim


#
# sim_num = 100
# sim_years = 500
# evbmus_sim = np.nan*np.ones((sim_num,(sim_years)))
# key = (awt_bmus[-3], awt_bmus[-2], awt_bmus[-1])
# for gg in range(sim_num):
#     bmu_sim = [awt_bmus[-3], awt_bmus[-2], awt_bmus[-1]]
#     c = 2
#     while len(bmu_sim) < (sim_years):
#         w = random.choice(chain3[key])
#         # temp = chain[key]
#         # counter = 0
#         # while w not in mwtByMonth[seasonalMonthSim[c]-1]:
#         #     print('stuck trying to put {} in {}'.format(w,seasonalMonthSim[c]))
#         #     counter = counter + 1
#         #     if counter < 20:
#         #         w = random.choice(chain[key])
#         #     else:
#         #         newW = random.choice(mwtByMonth[seasonalMonthSim[c] - 1])
#         #         newWminus1 = random.choice(mwtByMonth[seasonalMonthSim[c] - 2])
#         #         key = (newWminus1,newW)
#         #         w = newW
#         bmu_sim.append(w)
#         key = (key[1], key[2], w)
#         c = c + 1
#     evbmus_sim[gg, :] = bmu_sim
#



# sim_num = 100
bmus = awt_bmus#[1:]
evbmus_sim = evbmus_sim#evbmus_simALR.T

# Lets make a plot comparing probabilities in sim vs. historical
probH = np.nan*np.ones((num_clusters,))
probS = np.nan * np.ones((sim_num,num_clusters))
for h in np.unique(bmus):
    findH = np.where((bmus == h))[0][:]
    probH[int(h-1)] = len(findH)/len(bmus)

    for s in range(sim_num):
        findS = np.where((evbmus_sim[s,:] == h))[0][:]
        probS[s,int(h-1)] = len(findS)/len(evbmus_sim[s,:])


from alrPlotting import colors_mjo
from alrPlotting import colors_awt
etcolors = cm.jet(np.linspace(0, 1, 24))#70-20))
tccolors = np.flipud(cm.autumn(np.linspace(0,1,2)))#21)))
dwtcolors = np.vstack((etcolors,tccolors[1:,:]))
dwtcolors = colors_mjo()


plt.figure()
# plt.plot(probH,np.mean(probS,axis=0),'.')
# plt.plot([0,0.03],[0,0.03],'.--')
ax = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
tempPs = np.nan*np.ones((6,))
for i in range(6):
    temp = probS[:,i]
    temp2 = probH[i]
    box1 = ax.boxplot(temp,positions=[temp2],widths=.01,notch=True,patch_artist=True,showfliers=False)
    plt.setp(box1['boxes'],color=dwtcolors[i])
    plt.setp(box1['means'],color=dwtcolors[i])
    plt.setp(box1['fliers'],color=dwtcolors[i])
    plt.setp(box1['whiskers'],color=dwtcolors[i])
    plt.setp(box1['caps'],color=dwtcolors[i])
    plt.setp(box1['medians'],color=dwtcolors[i],linewidth=0)
    tempPs[i] = np.mean(temp)
    #box1['boxes'].set(facecolor=dwtcolors[i])
    #plt.set(box1['fliers'],markeredgecolor=dwtcolors[i])
ax.plot([0,0.3],[0,0.3],'k--', zorder=10)
plt.xlim([0,0.3])
plt.ylim([0,0.3])
plt.xticks([0,0.05,0.10,0.15,0.20,0.25,0.3], ['0','0.05','0.10','0.15','0.20','0.25','0.3'])
plt.xlabel('Historical Probability')
plt.ylabel('Simulated Probability')
plt.title('Validation of ALR SWT Simulations')



from itertools import groupby

a = list(bmus)
seq = list()
for i in np.arange(1,7):
    temp = [len(list(v)) for k,v in groupby(a) if k==i-1]
    seq.append(temp)

simseqPers = list()
for hhh in range(sim_num):
    b = list(evbmus_sim[hhh,:])
    seq_sim = list()
    for i in np.arange(1,7):
        temp2 = [len(list(v)) for k,v in groupby(b) if k==i-1]
        seq_sim.append(temp2)
    simseqPers.append(seq_sim)

persistReal = np.nan * np.ones((6,5))
for dwt in np.arange(1,7):
    sortDurs = np.sort(seq[dwt-1])
    realPercent = np.nan*np.ones((5,))
    for qq in np.arange(1,6):
        realInd = np.where((sortDurs <= qq))
        realPercent[qq-1] = len(realInd[0])/len(sortDurs)
    persistReal[dwt-1,:] = realPercent

persistSim = list()
for dwt in np.arange(1,7):
    persistDWT = np.nan * np.ones((sim_num, 5))
    for simInd in range(sim_num):

        sortDursSim = np.sort(simseqPers[simInd][dwt-1])
        simPercent = np.nan*np.ones((5,))
        for qq in np.arange(1,6):
            simIndex = np.where((sortDursSim <= qq))
            simPercent[qq-1] = len(simIndex[0])/len(sortDursSim)
        persistDWT[simInd,:] = simPercent
    persistSim.append(persistDWT)


x = [0.5,1.5,1.5,2.5,2.5,3.5,3.5,4.5,4.5,5.5]
plt.figure()
gs2 = gridspec.GridSpec(2, 3)
for xx in range(6):
    ax = plt.subplot(gs2[xx])
    ax.boxplot(persistSim[xx])
    y = [persistReal[xx, 0], persistReal[xx, 0], persistReal[xx, 1], persistReal[xx, 1], persistReal[xx, 2],
         persistReal[xx, 2], persistReal[xx, 3],persistReal[xx, 3],persistReal[xx, 4],persistReal[xx, 4],]
    ax.plot(x, y, color=dwtcolors[xx])
    ax.set_ylim([0.25, 1.05])



### TODO: create Gaussian copulas of the separate clusters
# bmus_dates_months = np.array([d.month for d in dates_sim])
# bmus_dates_days = np.array([d.day for d in dates_sim])



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


def Calc_SigmaCorrelation_AllOn_Chromosomes(self, xds_KMA_MS, xds_WVS_MS, xds_GEV_Par):
    'Calculate Sigma Pearson correlation for each WT, all on chrom combo'

    bmus = xds_KMA_MS.bmus.values[:]
    cenEOFs = xds_KMA_MS.cenEOFs.values[:]
    n_clusters = len(xds_KMA_MS.n_clusters)
    wvs_fams = self.fams
    vars_extra = self.extra_variables
    vars_GEV = self.vars_GEV

    # smooth GEV shape parameter
    d_shape = {}
    for vn in vars_GEV:
        sh_GEV = xds_GEV_Par.sel(parameter='shape')[vn].values[:]
        d_shape[vn] = Smooth_GEV_Shape(cenEOFs, sh_GEV)

    # Get sigma correlation for each KMA cluster
    d_sigma = {}  # nested dict [WT][crom]
    for iwt in range(n_clusters):
        c = iwt+1
        pos = np.where((bmus==c))[0]
        d_sigma[c] = {}

        # current cluster waves
        xds_K_wvs = xds_WVS_MS.isel(time=pos)

        # append data for spearman correlation
        to_corr = np.empty((0, len(xds_K_wvs.time)))

        # solve normal inverse GEV/EMP/WBL CDF for each waves family
        for fam_n in wvs_fams:

            # get wave family variables
            vn_Hs = '{0}_Hs'.format(fam_n)
            vn_Tp = '{0}_Tp'.format(fam_n)
            vn_Dir = '{0}_Dir'.format(fam_n)

            vv_Hs = xds_K_wvs[vn_Hs].values[:]
            vv_Tp = xds_K_wvs[vn_Tp].values[:]
            vv_Dir = xds_K_wvs[vn_Dir].values[:]

            # fix fams nan: Hs 0, Tp mean, dir mean
            p_nans = np.where(np.isnan(vv_Hs))[0]
            vv_Hs[p_nans] = 0
            vv_Tp[p_nans] = np.nanmean(vv_Tp)
            vv_Dir[p_nans] = np.nanmean(vv_Dir)

            # Hs
            norm_Hs = self.CDF_Distribution(vn_Hs, vv_Hs, xds_GEV_Par, d_shape, iwt)

            # Tp
            norm_Tp = self.CDF_Distribution(vn_Tp, vv_Tp, xds_GEV_Par, d_shape, iwt)

            # Dir
            norm_Dir = self.CDF_Distribution(vn_Dir, vv_Dir, xds_GEV_Par, d_shape, iwt)

            # normal inverse CDF
            u_cdf = np.column_stack([norm_Hs, norm_Tp, norm_Dir])
            u_cdf[u_cdf>=1.0] = 0.999999
            inv_n = ndtri(u_cdf)

            # concatenate data for correlation
            to_corr = np.concatenate((to_corr, inv_n.T), axis=0)

        # concatenate extra variables for correlation
        for vn in vars_extra:
            vv = xds_K_wvs[vn].values[:]

            norm_vn = self.CDF_Distribution(vn, vv, xds_GEV_Par, d_shape, iwt)
            norm_vn[norm_vn>=1.0] = 0.999999

            inv_n = ndtri(norm_vn)
            to_corr = np.concatenate((to_corr, inv_n[:, None].T), axis=0)

        # sigma: spearman correlation
        corr, pval = spearmanr(to_corr, axis=1)

        # store data at dict (keep cromosomes structure)
        d_sigma[c][0] = {
            'corr': corr, 'data': len(xds_K_wvs.time), 'wt_crom': 1
        }

    return d_sigma




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
            param_GEV = list(theta_gev_fix)
        else:
            param_GEV = list(theta_gl)
    else:
        param_GEV = list(theta_gev_fix)

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




def gev_CDF(x):
    '''
    :param x: observations
    :return: normalized cdf
    '''

    shape, loc, scale = fitGEVparams(x)
    # # gev CDF
    # sha_g = d_shape[vn][i_wt]
    # loc_g = xds_GEV_Par.sel(parameter='location')[vn].values[i_wt]
    # sca_g = xds_GEV_Par.sel(parameter='scale')[vn].values[i_wt]
    cdf = genextreme.cdf(x, -1 * shape, loc, scale)

    return cdf

def gev_ICDF(x,y):
    '''
    :param x: observations
    :param y: simulated probabilities
    :return: simulated values
    '''
    shape, loc, scale = fitGEVparams(x)
    ppf_VV = genextreme.ppf(y, -1 * shape, loc, scale)
    return ppf_VV


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
    kernels: list of kernels for each column at U_data (KDE | GPareto | Empirical | GEV)
    num_sim: number of simulations
    '''

    # kernel CDF dictionary
    d_kf = {
        'KDE' : (ksdensity_CDF, ksdensity_ICDF),
        'GPareto' : (GeneralizedPareto_CDF, GeneralizedPareto_ICDF),
        'ECDF' : (Empirical_CDF, Empirical_ICDF),
        'GEV': (gev_CDF, gev_ICDF),
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



copulaData = list()
for i in range(len(np.unique(bmus))):

    tempInd = np.where(((bmus)==i))
    dataCop = []
    for kk in range(len(tempInd[0])):
        # dataCop.append(list([PC1[tempInd[0][kk]],PC2[tempInd[0][kk]],PC3[tempInd[0][kk]]]))
        dataCop.append(list([PC1[tempInd[0][kk]],PC2[tempInd[0][kk]],PC3[tempInd[0][kk]]]))
    copulaData.append(dataCop)


gevCopulaSims = list()
for i in range(len(np.unique(bmus))):
    tempCopula = np.asarray(copulaData[i])
    kernels = ['KDE','KDE','KDE']
    samples = CopulaSimulation(tempCopula,kernels,100000)
    print('generating samples for AWT {}'.format(i))
    gevCopulaSims.append(samples)



### TODO: convert synthetic markovs to PC values


import random
### TODO: Fill in the Markov chain bmus with RMM vales
pc1Sims = list()
pc2Sims = list()
pc3Sims = list()
# pc4Sims = list()
for kk in range(sim_num):
    tempSimulation = evbmus_sim[kk,:]
    tempPC1 = np.nan*np.ones((np.shape(tempSimulation)))
    tempPC2 = np.nan*np.ones((np.shape(tempSimulation)))
    tempPC3 = np.nan*np.ones((np.shape(tempSimulation)))
    # tempPC4 = np.nan*np.ones((np.shape(tempSimulation)))

    groups = [list(j) for i, j in groupby(tempSimulation)]
    c = 0
    for gg in range(len(groups)):
        getInds = random.sample(range(1, 100000), len(groups[gg]))
        tempPC1s = gevCopulaSims[int(groups[gg][0])][getInds[0], 0]
        tempPC2s = gevCopulaSims[int(groups[gg][0])][getInds[0], 1]
        tempPC3s = gevCopulaSims[int(groups[gg][0])][getInds[0], 2]
        # tempPC4s = gevCopulaSims[int(groups[gg][0])][getInds[0], 3]
        tempPC1[c:c + len(groups[gg])] = tempPC1s
        tempPC2[c:c + len(groups[gg])] = tempPC2s
        tempPC3[c:c + len(groups[gg])] = tempPC3s
        # tempPC4[c:c + len(groups[gg])] = tempPC4s
        c = c + len(groups[gg])
    pc1Sims.append(tempPC1)
    pc2Sims.append(tempPC2)
    pc3Sims.append(tempPC3)
    # pc4Sims.append(tempPC4)


# sim_years = 100
# start simulation at PCs available data
d1 = datetime.datetime(2022,6,1)
d2 = datetime.datetime(d1.year+sim_years, d1.month, d1.day)
dates_sim2 = [d1 + datetime.timedelta(days=i) for i in range((d2-d1).days+1)]
# dates_sim = dates_sim[0:-1]

plt.figure()
plt.hist(PC1,alpha=0.5)
plt.hist(pc1Sims[0],alpha=0.5)

plt.figure()
ax1 = plt.subplot2grid((2,2),(0,0))
ax1.plot(PC1,PC2,'o')
ax1.plot(pc1Sims[0],pc2Sims[0],'.')

ax2 = plt.subplot2grid((2,2),(0,1))
ax2.plot(PC1,PC3,'o')
ax2.plot(pc1Sims[0],pc3Sims[0],'.')

ax3 = plt.subplot2grid((2,2),(1,0))
ax3.plot(PC2,PC3,'o')
ax3.plot(pc2Sims[0],pc3Sims[0],'.')


import pickle

awtPickle = 'awtPCs.pickle'
outputMWTs = {}
outputMWTs['PC1'] = PC1
outputMWTs['PC2'] = PC2
outputMWTs['PC3'] = PC3
outputMWTs['normPC1'] = normPC1
outputMWTs['normPC2'] = normPC2
outputMWTs['normPC3'] = normPC3
outputMWTs['awt_bmus'] = awt_bmus
outputMWTs['annualTime'] = annualTime
outputMWTs['dailyAWT'] = dailyAWT
outputMWTs['dailyDates'] = DailyDatesMatrix
outputMWTs['dailyTime'] = dailyTime
outputMWTs['dailyPC1'] = dailyPC1
outputMWTs['dailyPC2'] = dailyPC2
outputMWTs['dailyPC3'] = dailyPC3
# outputMWTs['dailyPC4'] = dailyPC4
outputMWTs['nPercent'] = nPercent


with open(awtPickle,'wb') as f:
    pickle.dump(outputMWTs, f)


# sim_years = 100
# start simulation at PCs available data
d1 = datetime.datetime(2022,6,1)#x2d(xds_cov_fit.time[0])
d2 = datetime.datetime(2022+int(sim_years),6,1)#datetime(d1.year+sim_years, d1.month, d1.day)
dt = datetime.date(2022, 6, 1)
end = datetime.date(2022+int(sim_years), 7, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(months=1)
dates_sim = []
while dt < end:
    dates_sim.append(dt)#.strftime('%Y-%m-%d'))
    dt += step



samplesPickle = 'awtSimulations.pickle'
outputSamples = {}
outputSamples['pc1Sims'] = pc1Sims
outputSamples['pc2Sims'] = pc2Sims
outputSamples['pc3Sims'] = pc3Sims
# outputSamples['pc4Sims'] = pc4Sims
outputSamples['evbmus_sim'] = evbmus_sim
outputSamples['dates_sim'] = dates_sim
with open(samplesPickle,'wb') as f:
    pickle.dump(outputSamples, f)
