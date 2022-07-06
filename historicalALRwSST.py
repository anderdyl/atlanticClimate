import xarray as xr
from scipy.io.matlab.mio5_params import mat_struct
import scipy.io as sio
from datetime import datetime, timedelta, date
import numpy as np
from time_operations import xds2datetime as x2d
from time_operations import xds_reindex_daily as xr_daily
from time_operations import xds_common_dates_daily as xcd_daily
import pickle
from dateutil.relativedelta import relativedelta
from alr import ALR_WRP


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
   return [date(d[0], d[1], d[2]) for d in d_vec]



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

bmus = bmus[120:]+1
timeDWTs = timeDWTs[120:]
bmus_dates = bmus_dates[120:]

xds_KMA_fit = xr.Dataset(
    {
        'bmus':(('time',), bmus),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in timeDWTs]}
)






# MJO historical: rmm1, rmm2 (first date 1979-01-01 in order to avoid nans)
dataMJO = ReadMatfile('/media/dylananderson/Elements/NC_climate/mjo_australia_2021.mat')

yearMonth = np.vstack((dataMJO['year'],dataMJO['month']))
Dates = np.vstack((yearMonth,dataMJO['day']))
Dates.T
xds_MJO_fit = xr.Dataset(
    {
        'rmm1': (('time',), dataMJO['rmm1']),
        'rmm2': (('time',), dataMJO['rmm2']),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in Dates.T]}
)
# reindex to daily data after 1979-01-01 (avoid NaN)
xds_MJO_fit = xr_daily(xds_MJO_fit, datetime(1979, 6, 1),datetime(2021,5,31))


##### AWT FROM ENSO SSTs
# with open(r"AWT1880to2020.pickle", "rb") as input_file:
#    historicalAWTs = pickle.load(input_file)
# awtClusters = historicalAWTs['clusters']
# awtPredictor = historicalAWTs['predictor']
#
# awtBmus = awtClusters.bmus.values
# pc1Annual = awtClusters.PCs[:,0]
# pc2Annual = awtClusters.PCs[:,1]
# pc3Annual = awtClusters.PCs[:,2]
#
#
# dt = datetime(1880, 6, 1)
# end = datetime(2021, 6, 1)
# #step = datetime.timedelta(months=1)
# step = relativedelta(years=1)
# sstTime = []
# while dt < end:
#     sstTime.append(dt)
#     dt += step
#
# years = np.arange(1979,2021)
# awtYears = np.arange(1880,2021)
#
# awtDailyBmus = np.nan * np.ones(np.shape(bmus))
# PC1 = np.nan * np.ones(np.shape(bmus))
# PC2 = np.nan * np.ones(np.shape(bmus))
# PC3 = np.nan * np.ones(np.shape(bmus))
#
# for hh in years:
#    indexDWT = np.where((np.asarray(bmus_dates) >= date(hh,6,1)) & (np.asarray(bmus_dates) <= date(hh+1,5,31)))
#    indexAWT = np.where((awtYears == hh))
#    awtDailyBmus[indexDWT] = awtBmus[indexAWT]*np.ones(len(indexDWT[0]))
#    PC1[indexDWT] = pc1Annual[indexAWT]*np.ones(len(indexDWT[0]))
#    PC2[indexDWT] = pc2Annual[indexAWT]*np.ones(len(indexDWT[0]))
#    PC3[indexDWT] = pc3Annual[indexAWT]*np.ones(len(indexDWT[0]))
#
# with open(r"mwtPCs3.pickle", "rb") as input_file:
#     historicalMWTs = pickle.load(input_file)
# # dailyPC1 = historicalMWTs['dailyPC1']
# # dailyPC2 = historicalMWTs['dailyPC2']
# # dailyPC3 = historicalMWTs['dailyPC3']
# # dailyPC4 = historicalMWTs['dailyPC4']
# # dailyDates = historicalMWTs['dailyDates']
# # awt_bmus = historicalMWTs['mwt_bmus']
# # seasonalTime = historicalMWTs['seasonalTime']
# # dailyMWT = historicalMWTs['dailyMWT']
# # with open(r"mwtPCs3.pickle", "rb") as input_file:
# #     historicalMWTs = pickle.load(input_file)
# dailyPC1 = historicalMWTs['dailyPC1'][92:]
# dailyPC2 = historicalMWTs['dailyPC2'][92:]
# dailyPC3 = historicalMWTs['dailyPC3'][92:]
# dailyPC4 = historicalMWTs['dailyPC4'][92:]
# dailyDates = historicalMWTs['dailyDates'][92:]
# awt_bmus = historicalMWTs['mwt_bmus'][1:]
# seasonalTime = historicalMWTs['seasonalTime'][1:]
# dailyMWT = historicalMWTs['dailyMWT'][92:]
#
# # AWT: PCs (Generated with copula simulation. Annual data, parse to daily)
# xds_PCs_fit = xr.Dataset(
#     {
#         'PC1': (('time',), dailyPC1),
#         'PC2': (('time',), dailyPC3),
#         'PC3': (('time',), dailyPC3),
#         'PC4': (('time',), dailyPC4),
#     },
#     coords = {'time': [datetime(r[0],r[1],r[2]) for r in dailyDates]}
# )
# # reindex annual data to daily data
# xds_PCs_fit = xr_daily(xds_PCs_fit, datetime(1979,6,1),datetime(2021,5,31))


with open(r"awtPCs.pickle", "rb") as input_file:
    historicalAWTs = pickle.load(input_file)

dailyPC1 = historicalAWTs['dailyPC1']
dailyPC2 = historicalAWTs['dailyPC2']
dailyPC3 = historicalAWTs['dailyPC3']
dailyDates = historicalAWTs['dailyDates']
awt_bmus = historicalAWTs['awt_bmus']
annualTime = historicalAWTs['annualTime']
dailyAWT = historicalAWTs['dailyAWT']

# AWT: PCs (Generated with copula simulation. Annual data, parse to daily)
xds_PCs_fit = xr.Dataset(
    {
        'PC1': (('time',), dailyPC1),
        'PC2': (('time',), dailyPC3),
        'PC3': (('time',), dailyPC3),
    },
    coords = {'time': [datetime(r[0],r[1],r[2]) for r in dailyDates]}
)
# reindex annual data to daily data
xds_PCs_fit = xr_daily(xds_PCs_fit, datetime(1979,6,1),datetime(2021,5,31))


### NAO AS AN INDEX
#
# with open('/home/dylananderson/projects/duckGeomorph/NAO2021.txt', 'r') as fd:
#     c = 0
#     dataNAO = list()
#     for line in fd:
#         splitLine = line.split(',')
#         secondSplit = splitLine[1].split('/')
#         dataNAO.append(float(secondSplit[0]))
# nao = np.asarray(dataNAO)
#
# dt = datetime.date(1950, 1, 1)
# end = datetime.date(2021, 6, 1)
# #step = datetime.timedelta(months=1)
# step = relativedelta(months=1)
# naoTime = []
# while dt < end:
#     naoTime.append(dt)#.strftime('%Y-%m-%d'))
#     dt += step
#
# naoTIME = naoTime[353:]
# data = nao[353:]
# naoShort = data
#
# bins = np.linspace(np.min(data)-.05, np.max(data)+.05, 7)
# digitized = np.digitize(data, bins)
# bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
#
#
# years = np.arange(1979,2022)
# months = np.arange(1,13)
# awtYears = np.arange(1880,2021)
#
#
# digitShort = digitized#[353:]
#
#
# naoTIME.append(datetime.date(2021,6,1))
# naoDailyBmus = np.nan * np.ones(np.shape(bmus))
# naoDaily = np.nan * np.ones(np.shape(bmus))
# for hh in range(len(naoTIME)-1):
#     #for mm in months:
#         # indexDWT = np.where((np.asarray(bmus_dates) >= datetime.date(hh,6,1)) & (np.asarray(bmus_dates) <= datetime.date(hh+1,6,1)))
#     indexDWT = np.where((np.asarray(bmus_dates) >= naoTIME[hh]) & (np.asarray(bmus_dates) <= naoTIME[hh+1]))
#     #indexAWT = np.where((awtYears == hh))
#     naoDaily[indexDWT] = naoShort[hh]*np.ones(len(indexDWT[0]))
#     naoDailyBmus[indexDWT] = digitShort[hh]*np.ones(len(indexDWT[0]))
#
#





# --------------------------------------
# Mount covariates matrix

# available data:
# model fit: xds_KMA_fit, xds_MJO_fit, xds_PCs_fit
# model sim: xds_MJO_sim, xds_PCs_sim

# covariates: FIT
d_covars_fit = xcd_daily([xds_MJO_fit, xds_PCs_fit, xds_KMA_fit])

# PCs covar
cov_PCs = xds_PCs_fit.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
cov_1 = cov_PCs.PC1.values.reshape(-1,1)
cov_2 = cov_PCs.PC2.values.reshape(-1,1)
cov_3 = cov_PCs.PC3.values.reshape(-1,1)

# MJO covars
cov_MJO = xds_MJO_fit.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
cov_4 = cov_MJO.rmm1.values.reshape(-1,1)
cov_5 = cov_MJO.rmm2.values.reshape(-1,1)

# join covars and norm.
cov_T = np.hstack((cov_1, cov_2, cov_3, cov_4, cov_5))


cov_T_mean = np.mean(cov_T,axis=0)
cov_T_std = np.std(cov_T,axis=0)
#cov_T_std = np.array(cov_T_std[0])
# multCovT = np.array([0.31804979/0.31804979, 0.16031134/0.31804979, 0.12182678/0.31804979, 0.09111769/0.31804979, 1, 1])
multCovT = np.array([0.4019148/0.4019148, 0.11355852/0.4019148, 0.10510168/0.4019148, 1, 1])

covTNorm = np.divide(np.subtract(cov_T,cov_T_mean),cov_T_std)
covTNormalize = np.multiply(covTNorm,multCovT)

# covTSimNorm = np.divide(np.subtract(cov_T_sim,cov_T_mean),cov_T_std)
# covTSimNormalize = np.multiply(covTSimNorm,multCovT)

# KMA related covars starting at KMA period
i0 = d_covars_fit.index(x2d(xds_KMA_fit.time[0]))
cov_KMA = cov_T[i0:,:]
d_covars_fit = d_covars_fit[i0:]

# generate xarray.Dataset
cov_names = ['PC1', 'PC2', 'PC3', 'MJO1', 'MJO2']
xds_cov_fit = xr.Dataset(
    {
        'cov_values': (('time','cov_names'), covTNormalize),
    },
    coords = {
        'time': d_covars_fit,
        'cov_names': cov_names,
    }
)





# --------------------------------------
# Autoregressive Logistic Regression

# available data:
# model fit: xds_KMA_fit, xds_cov_sim, num_clusters
# model sim: xds_cov_sim, sim_num, sim_years

# use bmus inside covariate time frame
xds_bmus_fit = xds_KMA_fit.sel(
    time=slice(d_covars_fit[0], d_covars_fit[-1])
)


# Autoregressive logistic wrapper
num_clusters = 70
sim_num = 100
fit_and_save = True # False for loading
p_test_ALR = '/media/dylananderson/Elements/NC_climate/testALR/'

# ALR terms
d_terms_settings = {
    'mk_order'  : 2,
    'constant' : False,
    'long_term' : False,
    'seasonality': (True, [2, 4]),
    'covariates': (True, xds_cov_fit),
}


# Autoregressive logistic wrapper
ALRW = ALR_WRP(p_test_ALR)
ALRW.SetFitData(
    num_clusters, xds_bmus_fit, d_terms_settings)

ALRW.FitModel(max_iter=20000)


#p_report = op.join(p_data, 'r_{0}'.format(name_test))

ALRW.Report_Fit() #'/media/dylananderson/Elements/NC_climate/testALR/r_Test', terms_fit==False)


# ALR model simulations
sim_years = 42
# start simulation at PCs available data
d1 = x2d(xds_cov_fit.time[0])
d2 = datetime(d1.year+sim_years, d1.month, d1.day)
dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

# print some info
print('ALR model fit   : {0} --- {1}'.format(
    d_covars_fit[0], d_covars_fit[-1]))
print('ALR model sim   : {0} --- {1}'.format(
    dates_sim[0], dates_sim[-1]))

# launch simulation
xds_ALR = ALRW.Simulate(
    sim_num, dates_sim[0:-2], xds_cov_fit)

dates_sim = dates_sim[0:-2]


# Save results for matlab plot
evbmus_sim = xds_ALR.evbmus_sims.values
# evbmus_probcum = xds_ALR.evbmus_probcum.values

p_mat_output = ('/media/dylananderson/Elements/NC_climate/testALR/testWithAWTNormedAgain_y{0}s{1}.h5'.format(
        sim_years, sim_num))
import h5py
with h5py.File(p_mat_output, 'w') as hf:
    hf['bmusim'] = evbmus_sim
    # hf['probcum'] = evbmus_probcum
    hf['dates'] = np.vstack(
        ([d.year for d in dates_sim],
        [d.month for d in dates_sim],
        [d.day for d in dates_sim])).T


samplesPickle = 'dwtHistoricalSimulations100.pickle'
outputSamples = {}
outputSamples['evbmus_sim'] = evbmus_sim
# outputSamples['evbmus_probcum'] = evbmus_probcum
outputSamples['sim_years'] = sim_num
outputSamples['dates_sim'] = dates_sim

with open(samplesPickle,'wb') as f:
    pickle.dump(outputSamples, f)



def GenOneYearDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime(yy, month_ini, 2)
   dp2 = dp1 + timedelta(days=364)

   return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]


def GenOneSeasonDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime(yy, month_ini, 1)
   dp2 = dp1 + timedelta(3*365/12)

   return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]



import matplotlib.pyplot as plt



bmus_dates_months = np.array([d.month for d in dates_sim])
bmus_dates_days = np.array([d.day for d in dates_sim])


# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=6)
m_plot = np.zeros((70, len(list_pyear))) * np.nan
num_clusters=70
num_sim=1
# sort data
for i, dpy in enumerate(list_pyear):
   _, s = np.where(
      [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
   )
   b = evbmus_sim[s,:]
   # b = bmus[s]
   b = b.flatten()

   for j in range(num_clusters):
      _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!

      m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

import matplotlib.cm as cm
etcolors = cm.viridis(np.linspace(0, 1, 70-20))
tccolors = np.flipud(cm.autumn(np.linspace(0,1,21)))
dwtcolors = np.vstack((etcolors,tccolors[1:,:]))




fig = plt.figure(figsize=(10,4))
ax = plt.subplot2grid((1,1),(0,0))
# plot stacked bars
bottom_val = np.zeros(m_plot[1, :].shape)
for r in range(num_clusters):
   row_val = m_plot[r, :]
   ax.bar(list_pyear, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
   # store bottom
   bottom_val += row_val

import matplotlib.dates as mdates
# customize  axis
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')
ax.set_xlim(list_pyear[0], list_pyear[-1])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_ylim(0, 100)
ax.set_ylabel('Probability')




# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=6)
m_plot = np.zeros((70, len(list_pyear))) * np.nan
num_clusters=70
num_sim=1
# sort data
for i, dpy in enumerate(list_pyear):
   _, s = np.where(
      [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
   )
   b = evbmus_sim[s,2]
   # b = bmus[s]
   b = b.flatten()

   for j in range(num_clusters):
      _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!

      m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

import matplotlib.cm as cm
etcolors = cm.viridis(np.linspace(0, 1, 70-20))
tccolors = np.flipud(cm.autumn(np.linspace(0,1,21)))
dwtcolors = np.vstack((etcolors,tccolors[1:,:]))

fig = plt.figure(figsize=(10,4))
ax = plt.subplot2grid((1,1),(0,0))
# plot stacked bars
bottom_val = np.zeros(m_plot[1, :].shape)
for r in range(num_clusters):
   row_val = m_plot[r, :]
   ax.bar(list_pyear, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
   # store bottom
   bottom_val += row_val

import matplotlib.dates as mdates
# customize  axis
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')
ax.set_xlim(list_pyear[0], list_pyear[-1])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
# ax.set_ylim(0, 100)
ax.set_ylim(0, 1)
ax.set_ylabel('Probability')



# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=6)
m_plot = np.zeros((70, len(list_pyear))) * np.nan
num_clusters=70
num_sim=1
# sort data
for i, dpy in enumerate(list_pyear):
   _, s = np.where([(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)])
   # b = evbmus_sim[s,:]
   b = bmus[s]
   b = b.flatten()

   for j in range(num_clusters):
      _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!

      m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

plt.style.use('dark_background')
fig = plt.figure(figsize=(10,4))
ax = plt.subplot2grid((1,1),(0,0))
# plot stacked bars
bottom_val = np.zeros(m_plot[1, :].shape)
for r in range(num_clusters):
   row_val = m_plot[r, :]
   ax.bar(
      list_pyear, row_val, bottom=bottom_val,
      width=1, color=np.array([dwtcolors[r]]))

   # store bottom
   bottom_val += row_val

import matplotlib.dates as mdates

# customize  axis
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')

ax.set_xlim(list_pyear[0], list_pyear[-1])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_ylim(0, 1)
ax.set_ylabel('Probability')


from matplotlib import gridspec


#
# Lets get complicated...
# a grid, 8 x 4 for the 8 SWTs and the 4 seasons?
# generate perpetual seasonal list
fig = plt.figure()
gs = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.15)

# onlyDo = [0,1,2,3,4,5]
onlyDo = [0,5]

c = 0
for m in np.unique(onlyDo):

    list_pSeason = GenOneYearDaily(month_ini=6)
    m_plot = np.zeros((70, len(list_pSeason))) * np.nan
    num_clusters=70
    num_sim=1
    # Identify on the data that occurs in this awt
    awtIndex = np.where(dailyAWT[0:-2] == m)
    subsetMonths = bmus_dates_months[awtIndex]
    subsetDays = bmus_dates_days[awtIndex]
    subsetBmus = bmus[awtIndex]
    #subsetBmus = evbmus_sim[awtIndex[0], :]
    # sort data

    for i, dpy in enumerate(list_pSeason):
        _, s = np.where([(subsetMonths == dpy.month) & (subsetDays == dpy.day)])
        # _, s = np.where([(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)])
        # b = evbmus_sim[s,:]
        # b = bmus[s]
        b = subsetBmus[s]
        b = b.flatten()

        for j in range(num_clusters):
            _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!

            m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

    ax = plt.subplot(gs[c])
    # plot stacked bars
    bottom_val = np.zeros(m_plot[1, :].shape)
    for r in range(num_clusters):
        row_val = m_plot[r, :]
        ax.bar(list_pSeason, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))

        # store bottom
        bottom_val += row_val
    # customize  axis
    months = mdates.MonthLocator()
    monthsFmt = mdates.DateFormatter('%b')
    ax.set_xlim(list_pSeason[0], list_pSeason[-1])
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    #ax.set_ylim(0, 100)
    ax.set_ylabel('')
    c = c + 1


# dailyMWT = dailyMWT[0:-2]
#
# # evbmus_sim = evbmus_sim - 1
# # bmus = bmus + 1
# fig = plt.figure()
# gs = gridspec.GridSpec(9, 4, wspace=0.1, hspace=0.15)
# c = 0
# for awt in np.unique(awt_bmus):
#
#     ind = np.where((dailyMWT == awt))[0][:]
#     timeSubDays = bmus_dates_days[ind]
#     timeSubMonths = bmus_dates_months[ind]
#     a = evbmus_sim[ind,:]
#
#     monthsIni = [3,6,9,12]
#     for m in monthsIni:
#
#         list_pSeason = GenOneSeasonDaily(month_ini=m)
#         m_plot = np.zeros((70, len(list_pSeason))) * np.nan
#         num_clusters=70
#         num_sim=1
#         # sort data
#         for i, dpy in enumerate(list_pSeason):
#             _, s = np.where([(timeSubMonths == dpy.month) & (timeSubDays == dpy.day)])
#             b = a[s,:]
#             # b = bmus[s]
#             b = b.flatten()
#             if len(b) > 0:
#                 for j in range(num_clusters):
#                     _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
#                     # _, bb = np.where([(j == b)])  # j starts at 0 bmus value!
#
#                     m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)
#
#
#         # indNan = np.where(np.isnan(m_plot))[0][:]
#         # if len(indNan) > 0:
#         #     m_plot[indNan] = np.ones((len(indNan),))
#         #m_plot = m_plot[1:,:]
#         ax = plt.subplot(gs[c])
#         # plot stacked bars
#         bottom_val = np.zeros(m_plot[1, :].shape)
#         for r in range(num_clusters):
#             row_val = m_plot[r, :]
#             indNan = np.where(np.isnan(row_val))[0][:]
#             if len(indNan) > 0:
#                 row_val[indNan] = 0
#             ax.bar(list_pSeason, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
#
#             # store bottom
#             bottom_val += row_val
#         # customize  axis
#         months = mdates.MonthLocator()
#         monthsFmt = mdates.DateFormatter('%b')
#         ax.set_xlim(list_pSeason[0], list_pSeason[-1])
#         ax.xaxis.set_major_locator(months)
#         ax.xaxis.set_major_formatter(monthsFmt)
#         ax.set_ylim(0, 100)
#         ax.set_ylabel('')
#         c = c + 1
#



# Lets make a plot comparing probabilities in sim vs. historical
probH = np.nan*np.ones((num_clusters,))
probS = np.nan * np.ones((sim_num,num_clusters))

for h in np.unique(bmus):
    findH = np.where((bmus == h))[0][:]
    probH[int(h-1)] = len(findH)/len(bmus)

    for s in range(sim_num):
        findS = np.where((evbmus_sim[:,s] == h))[0][:]
        probS[s,int(h-1)] = len(findS)/len(bmus)



plt.figure()
# plt.plot(probH,np.mean(probS,axis=0),'.')
# plt.plot([0,0.03],[0,0.03],'.--')
ax = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
for i in range(70):
    temp = probS[:,i]
    temp2 = probH[i]
    box1 = ax.boxplot(temp,positions=[temp2],widths=.0006,notch=True,patch_artist=True,showfliers=False)
    plt.setp(box1['boxes'],color=dwtcolors[i])
    plt.setp(box1['means'],color=dwtcolors[i])
    plt.setp(box1['fliers'],color=dwtcolors[i])
    plt.setp(box1['whiskers'],color=dwtcolors[i])
    plt.setp(box1['caps'],color=dwtcolors[i])
    plt.setp(box1['medians'],color=dwtcolors[i],linewidth=0)

    #box1['boxes'].set(facecolor=dwtcolors[i])
    #plt.set(box1['fliers'],markeredgecolor=dwtcolors[i])
ax.plot([0,0.03],[0,0.03],'k.--', zorder=10)
plt.xlim([0,0.03])
plt.ylim([0,0.03])
plt.xticks([0,0.01,0.02,0.03], ['0','0.01','0.02','0.03'])
plt.xlabel('Historical Probability')
plt.ylabel('Simulated Probability')
plt.title('Validation of ALR DWT Simulations')











