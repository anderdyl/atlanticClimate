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
from matplotlib import gridspec
import matplotlib.pyplot as plt
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

#
#
# # with open(r"dwtsAll6TCTracksClusters.pickle", "rb") as input_file:
# with open(r"dwts49Clusters.pickle", "rb") as input_file:
#    historicalDWTs = pickle.load(input_file)
#
# timeDWTs = historicalDWTs['SLPtime']
# # outputDWTs['slpDates'] = slpDates
# dwtBmus = historicalDWTs['bmus_corrected']
#
# # with open(r"dwtsOfExtraTropicalDays.pickle", "rb") as input_file:
# with open(r"dwtsOfExtraTropicalDays21Clusters.pickle", "rb") as input_file:
#    historicalTWTs = pickle.load(input_file)
# #timeTCs = historicalTWTs['tcDates']
# twtBmus = historicalTWTs['bmus_corrected']
# twtOrder = historicalTWTs['kma_order']
# tcIndices = historicalTWTs['tcIndices']
# TIMEtcs = historicalTWTs['TIMEtcs']
#
#
# timeSLPs = dateDay2datetimeDate(timeDWTs)
# timeTCs = dateDay2datetimeDate(TIMEtcs)
#
# overlap = [x for x in timeSLPs if x in timeTCs]
# ind_dict = dict((k,i) for i,k in enumerate(timeSLPs))
# inter = set(timeSLPs).intersection(timeTCs)
# indices = [ ind_dict[x] for x in inter ]
# indices.sort()
#
# mask = np.ones(len(timeDWTs), np.bool)
# mask[indices] = 0
# timeEWTs = timeDWTs[mask,:]
#
# mask2 = np.zeros(len(timeDWTs), np.bool)
# mask2[indices] = 1
# timeTWTs = timeDWTs[mask2,:]
#
# bmus = np.nan * np.ones((len(timeDWTs)))
# bmus[mask] = dwtBmus+0
# bmus[mask2] = twtBmus+49
#
# bmus_dates = dateDay2datetimeDate(timeDWTs)
# bmus_dates_months = np.array([d.month for d in bmus_dates])
# bmus_dates_days = np.array([d.day for d in bmus_dates])
#
# bmus = bmus[120:]+1
# timeDWTs = timeDWTs[120:]
# bmus_dates = bmus_dates[120:]
#
# xds_KMA_fit = xr.Dataset(
#     {
#         'bmus':(('time',), bmus),
#     },
#     coords = {'time': [datetime(r[0],r[1],r[2]) for r in timeDWTs]}
# )
#
#
#


def MJO_Categories(rmm1, rmm2, phase):
    '''
    Divides MJO data in 25 categories.

    rmm1, rmm2, phase - MJO parameters

    returns array with categories time series
    and corresponding rmm
    '''

    rmm = np.sqrt(rmm1**2 + rmm2**2)
    categ = np.empty(rmm.shape) * np.nan

    for i in range(1,9):
        s = np.squeeze(np.where(phase == i))
        rmm_p = rmm[s]

        # categories
        categ_p = np.empty(rmm_p.shape) * np.nan
        categ_p[rmm_p <=1] =  25
        categ_p[rmm_p > 1] =  i + 8*2
        categ_p[rmm_p > 1.5] =  i + 8
        categ_p[rmm_p > 2.5] =  i
        categ[s] = categ_p

    # get rmm_categ
    rmm_categ = {}
    for i in range(1,26):
        s = np.squeeze(np.where(categ == i))
        rmm_categ['cat_{0}'.format(i)] = np.column_stack((rmm1[s],rmm2[s]))

    return categ.astype(int), rmm_categ



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
xds_MJO_fit = xr_daily(xds_MJO_fit, datetime(1979, 6, 1))


mjoBmus, mjoGroups = MJO_Categories(dataMJO['rmm1'],dataMJO['rmm2'],dataMJO['phase'])


# xds_KMA_fit = xr.Dataset(
#     {
#         'bmus':(('time',), mjoBmus),
#     },
#     coords = {'time': [datetime(r[0],r[1],r[2]) for r in Dates.T]}
# )

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

import pickle
with open(r"mwtPCs2.pickle", "rb") as input_file:
    historicalMWTs = pickle.load(input_file)
dailyPC1 = historicalMWTs['dailyPC1']
dailyPC2 = historicalMWTs['dailyPC2']
dailyPC3 = historicalMWTs['dailyPC3']
dailyPC4 = historicalMWTs['dailyPC4']
dailyDates = historicalMWTs['dailyDates']
mwt_bmus = historicalMWTs['mwt_bmus']
seasonalTime = historicalMWTs['seasonalTime']
dailyMWT = historicalMWTs['dailyMWT']

PC1 = historicalMWTs['PC1']
PC2 = historicalMWTs['PC2']
PC3 = historicalMWTs['PC3']
PC4 = historicalMWTs['PC4']

seasonTime = [datetime(y.year,y.month,y.day) for y in seasonalTime]

xds_KMA_fit = xr.Dataset(
    {
        'bmus':(('time',), mwt_bmus),
    },
    coords = {'time': seasonTime}
    #coords = {'time': [datetime(r[0], r[1], r[2]) for r in dailyDates]}

)



# p_transition = np.array(
#     [[0.90, 0.05, 0.05],
#      [0.01, 0.90, 0.09],
#      [0.07, 0.03, 0.9]]
# )

T = mwt_bmus

M = [[0]*9 for _ in range(9)]

for (i,j) in zip(T,T[1:]):
    M[i][j] += 1

#now convert to probabilities:
for row in M:
    n = sum(row)
    if n > 0:
        row[:] = [f/sum(row) for f in row]

transistionMatrix = np.nan*np.ones((9,9))
c = 0
for row in M:
    print(row)
    transistionMatrix[c,:] = row
    c = c+1

def equilibrium_distribution(p_transition):
    n_states = p_transition.shape[0]
    A = np.append(
        arr=p_transition.T - np.eye(n_states),
        values=np.ones(n_states).reshape(1, -1),
        axis=0
    )
    b = np.transpose(np.array([0] * n_states + [1]))
    p_eq = np.linalg.solve(
        a=np.transpose(A).dot(A),
        b=np.transpose(A).dot(b)
    )
    return p_eq



from scipy.stats import multinomial
from typing import List

def markov_sequence(p_init: np.array, p_transition: np.array, sequence_length: int) -> List[int]:
    """
    Generate a Markov sequence based on p_init and p_transition.
    """
    if p_init is None:
        p_init = equilibrium_distribution(p_transition)
    initial_state = list(multinomial.rvs(1, p_init)).index(1)

    states = [1]#[initial_state]
    for _ in range(sequence_length - 1):
        p_tr = p_transition[states[-1]]
        new_state = list(multinomial.rvs(1, p_tr)).index(1)
        states.append(new_state)
    return states


import seaborn as sns
sim_num = 100
simbmus = np.nan*np.ones((sim_num,400))
for hh in range(sim_num):
    states = markov_sequence(None, transistionMatrix, sequence_length=100*4)
    # fig, ax = plt.subplots(figsize=(12, 4))
    # plt.plot(states)
    # plt.xlabel("time step")
    # plt.ylabel("state")
    # plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # sns.despine()
    simbmus[hh,:] = states
# ALR model simulations
sim_years = 400/4
# start simulation at PCs available data
d1 = datetime(2022,6,1)#x2d(xds_cov_fit.time[0])
d2 = datetime(2022+int(sim_years),6,1)#datetime(d1.year+sim_years, d1.month, d1.day)
# dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]
dt = date(2022, 6, 1)
end = date(2022+int(sim_years), 6, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(months=3)
dates_sim = []
while dt < end:
    dates_sim.append(dt)#.strftime('%Y-%m-%d'))
    dt += step






#
#
#
# # AWT: PCs (Generated with copula simulation. Annual data, parse to daily)
# xds_PCs_fit = xr.Dataset(
#     {
#         'PC1': (('time',), PC1),
#         'PC2': (('time',), PC3),
#         'PC3': (('time',), PC3),
#         'PC4': (('time',), PC4),
#     },
#     coords = {'time': seasonTime}
#     #coords = {'time': [datetime(r[0], r[1], r[2]) for r in dailyDates]}
#
# )
# # # reindex annual data to daily data
# # xds_PCs_fit = xr_daily(xds_PCs_fit)
#
#
# ### NAO AS AN INDEX
# #
# # with open('/home/dylananderson/projects/duckGeomorph/NAO2021.txt', 'r') as fd:
# #     c = 0
# #     dataNAO = list()
# #     for line in fd:
# #         splitLine = line.split(',')
# #         secondSplit = splitLine[1].split('/')
# #         dataNAO.append(float(secondSplit[0]))
# # nao = np.asarray(dataNAO)
# #
# # dt = datetime.date(1950, 1, 1)
# # end = datetime.date(2021, 6, 1)
# # #step = datetime.timedelta(months=1)
# # step = relativedelta(months=1)
# # naoTime = []
# # while dt < end:
# #     naoTime.append(dt)#.strftime('%Y-%m-%d'))
# #     dt += step
# #
# # naoTIME = naoTime[353:]
# # data = nao[353:]
# # naoShort = data
# #
# # bins = np.linspace(np.min(data)-.05, np.max(data)+.05, 7)
# # digitized = np.digitize(data, bins)
# # bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]
# #
# #
# # years = np.arange(1979,2022)
# # months = np.arange(1,13)
# # awtYears = np.arange(1880,2021)
# #
# #
# # digitShort = digitized#[353:]
# #
# #
# # naoTIME.append(datetime.date(2021,6,1))
# # naoDailyBmus = np.nan * np.ones(np.shape(bmus))
# # naoDaily = np.nan * np.ones(np.shape(bmus))
# # for hh in range(len(naoTIME)-1):
# #     #for mm in months:
# #         # indexDWT = np.where((np.asarray(bmus_dates) >= datetime.date(hh,6,1)) & (np.asarray(bmus_dates) <= datetime.date(hh+1,6,1)))
# #     indexDWT = np.where((np.asarray(bmus_dates) >= naoTIME[hh]) & (np.asarray(bmus_dates) <= naoTIME[hh+1]))
# #     #indexAWT = np.where((awtYears == hh))
# #     naoDaily[indexDWT] = naoShort[hh]*np.ones(len(indexDWT[0]))
# #     naoDailyBmus[indexDWT] = digitShort[hh]*np.ones(len(indexDWT[0]))
# #
# #
#
#
#
#
# #
# # # --------------------------------------
# # # Mount covariates matrix
# #
# # # available data:
# # # model fit: xds_KMA_fit, xds_MJO_fit, xds_PCs_fit
# # # model sim: xds_MJO_sim, xds_PCs_sim
# #
# # # covariates: FIT
# # #d_covars_fit = xcd_daily([xds_MJO_fit, xds_PCs_fit, xds_KMA_fit])
# # d_covars_fit = xds_PCs_fit
# # # PCs covar
# # cov_PCs = xds_PCs_fit.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
# # cov_1 = cov_PCs.PC1.values.reshape(-1,1)
# # cov_2 = cov_PCs.PC2.values.reshape(-1,1)
# # cov_3 = cov_PCs.PC3.values.reshape(-1,1)
# # cov_4 = cov_PCs.PC4.values.reshape(-1,1)
# #
# # # # MJO covars
# # # cov_MJO = xds_MJO_fit.sel(time=slice(d_covars_fit[0],d_covars_fit[-1]))
# # # cov_5 = cov_MJO.rmm1.values.reshape(-1,1)
# # # cov_6 = cov_MJO.rmm2.values.reshape(-1,1)
# #
# # # join covars and norm.
# # cov_T = np.hstack((cov_1, cov_2, cov_3, cov_4))#, cov_5, cov_6))
#
# # KMA related covars starting at KMA period
# # i0 = d_covars_fit.index(x2d(xds_KMA_fit.time[0]))
# # #i0 = d_covars_fit.index(x2d(xds_PCs_fit.time[0]))
# # cov_KMA = cov_T[i0:,:]
# # d_covars_fit = d_covars_fit[i0:]
#
# # i0 = d_covars_fit.index(x2d(xds_KMA_fit.time[0]))
# # #i0 = d_covars_fit.index(x2d(xds_PCs_fit.time[0]))
# # cov_KMA = cov_T[i0:,:]
# # d_covars_fit = d_covars_fit[i0:]
#
#
# # # generate xarray.Dataset
# # cov_names = ['PC1', 'PC2', 'PC3', 'PC4']#, 'MJO1', 'MJO2']
# # xds_cov_fit = xr.Dataset(
# #     {
# #         'cov_values': (('time','cov_names'), cov_T),
# #     },
# #     coords = {
# #         'time': d_covars_fit,
# #         'cov_names': cov_names,
# #     }
# # )
# cov_names = ['PC1', 'PC2', 'PC3', 'PC4']
# cov_T = np.vstack((PC1, PC2, PC3, PC4))#
# xds_cov_fit = xr.Dataset(
#     {
#         'PC1': (('time',), PC1),
#         'PC2': (('time',), PC3),
#         'PC3': (('time',), PC3),
#         'PC4': (('time',), PC4),
#         'cov_values': (('time','cov_names'),cov_T.T),
#     },
#     coords = {'time': seasonTime,
#               'cov_names': cov_names,}
#     #coords = {'time': [datetime(r[0], r[1], r[2]) for r in dailyDates]}
#
# )
#
# # # use bmus inside covariate time frame
# # xds_bmus_fit = xds_KMA_fit.sel(
# #     time=slice(d_covars_fit[0], d_covars_fit[-1])
# # )
#
# #bmus = xds_bmus_fit.bmus
# xds_bmus_fit = xds_KMA_fit
# #xds_cov_fit = xds_KMA_fit
# # --------------------------------------
# # Autoregressive Logistic Regression
#
# # available data:
# # model fit: xds_KMA_fit, xds_cov_sim, num_clusters
# # model sim: xds_cov_sim, sim_num, sim_years
#
# bmus = mwt_bmus
#
# # Autoregressive logistic wrapper
# num_clusters = 9
# sim_num = 10
# fit_and_save = True # False for loading
# p_test_ALR = '/media/dylananderson/Elements/NC_climate/testSWTALR/'
#
# # ALR terms
# d_terms_settings = {
#     'mk_order'  : 2,
#     'constant' : True,
#     'long_term' : False,
#     'seasonality': (False,),# [2, 4, 6]),
#     'covariates': (False,),# xds_cov_fit),
# }
#
#
# # Autoregressive logistic wrapper
# ALRW = ALR_WRP(p_test_ALR)
# ALRW.SetFitData(
#     num_clusters, xds_bmus_fit, d_terms_settings)
#
# ALRW.FitModel(max_iter=20000)
#
#
# #p_report = op.join(p_data, 'r_{0}'.format(name_test))
#
# ALRW.Report_Fit() #'/media/dylananderson/Elements/NC_climate/testALR/r_Test', terms_fit==False)
#
#
# # ALR model simulations
# sim_years = 100
# # start simulation at PCs available data
# d1 = datetime(2022,1,1)#x2d(xds_cov_fit.time[0])
# d2 = datetime(2122,1,1)#datetime(d1.year+sim_years, d1.month, d1.day)
# # dates_sim = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]
# dt = date(2022, 1, 1)
# end = date(2123, 1, 1)
# #step = datetime.timedelta(months=1)
# step = relativedelta(months=3)
# dates_sim = []
# while dt < end:
#     dates_sim.append(dt)#.strftime('%Y-%m-%d'))
#     dt += step
#
#
# # print some info
# #print('ALR model fit   : {0} --- {1}'.format(
# #    d_covars_fit[0], d_covars_fit[-1]))
# print('ALR model sim   : {0} --- {1}'.format(
#     dates_sim[0], dates_sim[-1]))
#
# # launch simulation
# xds_ALR = ALRW.Simulate(
#     sim_num, dates_sim, xds_cov_fit)
#
# dates_sim = dates_sim
#
#
# # Save results for matlab plot
# evbmus_sim = xds_ALR.evbmus_sims.values
# # evbmus_probcum = xds_ALR.evbmus_probcum.values
#
# p_mat_output = ('/media/dylananderson/Elements/NC_climate/testSWTALR/testSWT_y{0}s{1}.h5'.format(
#         sim_years, sim_num))
# import h5py
# with h5py.File(p_mat_output, 'w') as hf:
#     hf['bmusim'] = evbmus_sim
#     # hf['probcum'] = evbmus_probcum
#     hf['dates'] = np.vstack(
#         ([d.year for d in dates_sim],
#         [d.month for d in dates_sim],
#         [d.day for d in dates_sim])).T
#




from alrPlotting import colors_mjo
from alrPlotting import colors_awt

def GenOneYearDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime(yy, month_ini, 1)
   dp2 = dp1 + timedelta(days=365)

   return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]


def GenOneSeasonDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime(yy, month_ini, 1)
   dp2 = dp1 + timedelta(3*365/12)

   return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]






bmus_dates_months = np.array([d.month for d in dates_sim])
bmus_dates_days = np.array([d.day for d in dates_sim])


# # generate perpetual year list
# list_pyear = GenOneYearDaily(month_ini=6)
# m_plot = np.zeros((25, len(list_pyear))) * np.nan
# num_clusters=25
# num_sim=1
# # sort data
# for i, dpy in enumerate(list_pyear):
#    _, s = np.where(
#       [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
#    )
#    b = evbmus_sim[s,:]
#    # b = bmus[s]
#    b = b.flatten()
#
#    for j in range(num_clusters):
#       _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
#
#       m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

import matplotlib.cm as cm
etcolors = cm.jet(np.linspace(0, 1, 24))#70-20))
tccolors = np.flipud(cm.autumn(np.linspace(0,1,2)))#21)))
dwtcolors = np.vstack((etcolors,tccolors[1:,:]))

dwtcolors = colors_mjo()













#
# fig = plt.figure()
# ax = plt.subplot2grid((1,1),(0,0))
# # plot stacked bars
# bottom_val = np.zeros(m_plot[1, :].shape)
# for r in range(num_clusters):
#    row_val = m_plot[r, :]
#    ax.bar(list_pyear, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
#    # store bottom
#    bottom_val += row_val
#
# import matplotlib.dates as mdates
# # customize  axis
# months = mdates.MonthLocator()
# monthsFmt = mdates.DateFormatter('%b')
# ax.set_xlim(list_pyear[0], list_pyear[-1])
# ax.xaxis.set_major_locator(months)
# ax.xaxis.set_major_formatter(monthsFmt)
# ax.set_ylim(0, 10)
# ax.set_ylabel('')



#
#
# # generate perpetual year list
# list_pyear = GenOneYearDaily(month_ini=6)
# m_plot = np.zeros((70, len(list_pyear))) * np.nan
# num_clusters=70
# num_sim=1
# # sort data
# for i, dpy in enumerate(list_pyear):
#    _, s = np.where([(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)])
#    # b = evbmus_sim[s,:]
#    b = bmus[s]
#    b = b.flatten()
#
#    for j in range(num_clusters):
#       _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
#
#       m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)
#
# fig = plt.figure()
# ax = plt.subplot2grid((1,1),(0,0))
# # plot stacked bars
# bottom_val = np.zeros(m_plot[1, :].shape)
# for r in range(num_clusters):
#    row_val = m_plot[r, :]
#    ax.bar(
#       list_pyear, row_val, bottom=bottom_val,
#       width=1, color=np.array([dwtcolors[r]]))
#
#    # store bottom
#    bottom_val += row_val
#
# import matplotlib.dates as mdates
#
# # customize  axis
# months = mdates.MonthLocator()
# monthsFmt = mdates.DateFormatter('%b')
#
# ax.set_xlim(list_pyear[0], list_pyear[-1])
# ax.xaxis.set_major_locator(months)
# ax.xaxis.set_major_formatter(monthsFmt)
# ax.set_ylim(0, 1)
# ax.set_ylabel('')


#
# # Lets get complicated...
# # a grid, 8 x 4 for the 8 SWTs and the 4 seasons?
# # generate perpetual seasonal list
# fig = plt.figure()
# gs = gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.15)
#
# monthsIni = [3,6,9,12]
# c = 0
# for m in monthsIni:
#
#     list_pSeason = GenOneSeasonDaily(month_ini=m)
#     m_plot = np.zeros((70, len(list_pSeason))) * np.nan
#     num_clusters=70
#     num_sim=1
#     # sort data
#     for i, dpy in enumerate(list_pSeason):
#         _, s = np.where([(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)])
#         b = evbmus_sim[s,:]
#         # b = bmus[s]
#         b = b.flatten()
#
#         for j in range(num_clusters):
#             _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!
#
#             m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)
#
#     ax = plt.subplot(gs[c])
#     # plot stacked bars
#     bottom_val = np.zeros(m_plot[1, :].shape)
#     for r in range(num_clusters):
#         row_val = m_plot[r, :]
#         ax.bar(list_pSeason, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
#
#         # store bottom
#         bottom_val += row_val
#     # customize  axis
#     months = mdates.MonthLocator()
#     monthsFmt = mdates.DateFormatter('%b')
#     ax.set_xlim(list_pSeason[0], list_pSeason[-1])
#     ax.xaxis.set_major_locator(months)
#     ax.xaxis.set_major_formatter(monthsFmt)
#     ax.set_ylim(0, 100)
#     ax.set_ylabel('')
#     c = c + 1

#
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






num_clusters = 9
sim_num = 20
bmus = mwt_bmus
evbmus_sim = simbmus

# Lets make a plot comparing probabilities in sim vs. historical
probH = np.nan*np.ones((num_clusters,))
probS = np.nan * np.ones((sim_num,num_clusters))
bmus = bmus
for h in np.unique(bmus):
    findH = np.where((bmus == h))[0][:]
    probH[int(h-1)] = len(findH)/len(bmus)

    for s in range(sim_num):
        findS = np.where((evbmus_sim[s,:] == h))[0][:]
        probS[s,int(h-1)] = len(findS)/len(evbmus_sim[s,:])



plt.figure()
# plt.plot(probH,np.mean(probS,axis=0),'.')
# plt.plot([0,0.03],[0,0.03],'.--')
ax = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
tempPs = np.nan*np.ones((9,))
for i in range(9):
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
ax.plot([0,0.2],[0,0.2],'k--', zorder=10)
plt.xlim([0,0.2])
plt.ylim([0,0.2])
plt.xticks([0,0.05,0.10,0.15,0.20], ['0','0.05','0.10','0.15','0.20'])
plt.xlabel('Historical Probability')
plt.ylabel('Simulated Probability')
plt.title('Validation of ALR SWT Simulations')



from itertools import groupby

a = list(bmus)
seq = list()
for i in np.arange(1,10):
    temp = [len(list(v)) for k,v in groupby(a) if k==i-1]
    seq.append(temp)

simseqPers = list()
for hhh in range(sim_num):
    b = list(evbmus_sim[hhh,:])
    seq_sim = list()
    for i in np.arange(1,10):
        temp2 = [len(list(v)) for k,v in groupby(b) if k==i-1]
        seq_sim.append(temp2)
    simseqPers.append(seq_sim)

persistReal = np.nan * np.ones((9,3))
for dwt in np.arange(1,10):
    sortDurs = np.sort(seq[dwt-1])
    realPercent = np.nan*np.ones((3,))
    for qq in np.arange(1,4):
        realInd = np.where((sortDurs <= qq))
        realPercent[qq-1] = len(realInd[0])/len(sortDurs)
    persistReal[dwt-1,:] = realPercent

persistSim = list()
for dwt in np.arange(1,10):
    persistDWT = np.nan * np.ones((sim_num, 3))
    for simInd in range(sim_num):

        sortDursSim = np.sort(simseqPers[simInd][dwt-1])
        simPercent = np.nan*np.ones((3,))
        for qq in np.arange(1,4):
            simIndex = np.where((sortDursSim <= qq))
            simPercent[qq-1] = len(simIndex[0])/len(sortDursSim)
        persistDWT[simInd,:] = simPercent
    persistSim.append(persistDWT)


# x = [0.5,1.5,1.5,2.5,2.5,3.5,3.5,4.5,4.5,5.5]
# num = 20
# plt.figure()
# ax1 = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
# ax1.boxplot(persistSim[num])
# y = [persistReal[num,0],persistReal[num,0],persistReal[num,1],persistReal[num,1],persistReal[num,2],persistReal[num,2],
#      persistReal[num,3],persistReal[num,3],persistReal[num,4],persistReal[num,4],]
# ax1.plot(x,y,color=dwtcolors[num])


x = [0.5,1.5,1.5,2.5,2.5,3.5]
plt.figure()
gs2 = gridspec.GridSpec(3, 3)
for xx in range(9):
    ax = plt.subplot(gs2[xx])
    ax.boxplot(persistSim[xx])
    y = [persistReal[xx, 0], persistReal[xx, 0], persistReal[xx, 1], persistReal[xx, 1], persistReal[xx, 2],
         persistReal[xx, 2],]
    ax.plot(x, y, color=dwtcolors[xx])
    ax.set_ylim([0.55, 1.05])






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
        dataCop.append(list([PC1[tempInd[0][kk]],PC2[tempInd[0][kk]],PC3[tempInd[0][kk]],PC4[tempInd[0][kk]]]))
    copulaData.append(dataCop)



gevCopulaSims = list()
for i in range(len(np.unique(bmus))):
    tempCopula = np.asarray(copulaData[i])
    kernels = ['GEV','GEV','GEV','GEV']
    samples = CopulaSimulation(tempCopula,kernels,100000)
    print('generating samples for DWT {}'.format(i))
    gevCopulaSims.append(samples)


import random
### TODO: Fill in the Markov chain bmus with RMM vales
pc1Sims = list()
pc2Sims = list()
pc3Sims = list()
pc4Sims = list()
for kk in range(sim_num):
    tempSimulation = evbmus_sim[kk,:]
    tempPC1 = np.nan*np.ones((np.shape(tempSimulation)))
    tempPC2 = np.nan*np.ones((np.shape(tempSimulation)))
    tempPC3 = np.nan*np.ones((np.shape(tempSimulation)))
    tempPC4 = np.nan*np.ones((np.shape(tempSimulation)))

    groups = [list(j) for i, j in groupby(tempSimulation)]
    c = 0
    for gg in range(len(groups)):
        getInds = random.sample(range(1, 10000), len(groups[gg]))
        tempPC1s = gevCopulaSims[int(groups[gg][0])][getInds[0], 0]
        tempPC2s = gevCopulaSims[int(groups[gg][0])][getInds[0], 1]
        tempPC3s = gevCopulaSims[int(groups[gg][0])][getInds[0], 2]
        tempPC4s = gevCopulaSims[int(groups[gg][0])][getInds[0], 3]
        tempPC1[c:c + len(groups[gg])] = tempPC1s
        tempPC2[c:c + len(groups[gg])] = tempPC2s
        tempPC3[c:c + len(groups[gg])] = tempPC3s
        tempPC4[c:c + len(groups[gg])] = tempPC4s
        c = c + len(groups[gg])
    pc1Sims.append(tempPC1)
    pc2Sims.append(tempPC2)
    pc3Sims.append(tempPC3)
    pc4Sims.append(tempPC4)


samplesPickle = 'swtSimulations.pickle'
outputSamples = {}
outputSamples['pc1Sims'] = pc1Sims
outputSamples['pc2Sims'] = pc2Sims
outputSamples['pc3Sims'] = pc3Sims
outputSamples['pc4Sims'] = pc4Sims
outputSamples['evbmus_sim'] = evbmus_sim
outputSamples['dates_sim'] = dates_sim
with open(samplesPickle,'wb') as f:
    pickle.dump(outputSamples, f)



