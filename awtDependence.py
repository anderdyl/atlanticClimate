import pandas as pd
from dateutil.relativedelta import relativedelta
import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import datetime
import pickle
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib import gridspec

# # import constants
# from .config import _faspect, _fsize, _fdpi
_faspect = 1.618
_fsize = 9.8
_fdpi = 128


def GenOneYearDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime.datetime(yy, month_ini, 1)
   dp2 = dp1 + datetime.timedelta(days=365)

   return [dp1 + datetime.timedelta(days=i) for i in range((dp2 - dp1).days)]



def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [datetime.date(d[0], d[1], d[2]) for d in d_vec]



def ClusterProbabilities(series, set_values):
    'return series probabilities for each item at set_values'

    us, cs = np.unique(series, return_counts=True)
    d_count = dict(zip(us,cs))

    # cluster probabilities
    cprobs = np.zeros((len(set_values)))
    for i, c in enumerate(set_values):
       cprobs[i] = 1.0*d_count[c]/len(series) if c in d_count.keys() else 0.0

    return cprobs



def axplot_WT_Probs(ax, wt_probs,
                     ttl = '', vmin = 0, vmax = 0.1,
                     cmap = 'Blues', caxis='black'):
    'axes plot WT cluster probabilities'

    # clsuter transition plot
    pc = ax.pcolor(
        np.flipud(wt_probs),
        cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors='k',
    )

    # customize axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(ttl, {'fontsize':10, 'fontweight':'bold'})

    # axis color
    plt.setp(ax.spines.values(), color=caxis)
    plt.setp(
        [ax.get_xticklines(), ax.get_yticklines()],
        color=caxis,
    )

    # axis linewidth
    if caxis != 'black':
        plt.setp(ax.spines.values(), linewidth=3)

    return pc






with open(r"dwtsAll6TCTracksClusters.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

timeDWTs = historicalDWTs['SLPtime']
# outputDWTs['slpDates'] = slpDates
dwtBmus = historicalDWTs['bmus_corrected']


with open(r"dwtsOfExtraTropicalDays.pickle", "rb") as input_file:
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
bmus[mask2] = twtBmus+36

bmus_dates = dateDay2datetimeDate(timeDWTs)
bmus_dates_months = np.array([d.month for d in bmus_dates])
bmus_dates_days = np.array([d.day for d in bmus_dates])



# etcolors = cm.rainbow(np.linspace(0, 1, 48-11))
# tccolors = np.flipud(cm.gray(np.linspace(0,1,12)))
etcolors = cm.viridis(np.linspace(0, 1, 48-11))
tccolors = np.flipud(cm.autumn(np.linspace(0,1,12)))

dwtcolors = np.vstack((etcolors,tccolors[1:,:]))



with open(r"AWT1880to2020.pickle", "rb") as input_file:
   historicalAWTs = pickle.load(input_file)
awtClusters = historicalAWTs['clusters']
awtPredictor = historicalAWTs['predictor']

awtBmus = awtClusters.bmus.values


dt = datetime.datetime(1880, 6, 1)
end = datetime.datetime(2021, 6, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(years=1)
sstTime = []
while dt < end:
    sstTime.append(dt)
    dt += step

years = np.arange(1979,2021)
awtYears = np.arange(1880,2021)

awtDailyBmus = np.nan * np.ones(np.shape(bmus))
for hh in years:
   indexDWT = np.where((np.asarray(bmus_dates) >= datetime.date(hh,6,1)) & (np.asarray(bmus_dates) <= datetime.date(hh+1,6,1)))
   indexAWT = np.where((awtYears == hh))
   awtDailyBmus[indexDWT] = awtBmus[indexAWT]*np.ones(len(indexDWT[0]))



fig10 = plt.figure()

gs = gridspec.GridSpec(2, 3, wspace=0.10, hspace=0.15)

for ic in range(6):
    ax = plt.subplot(gs[ic])

    # select DWT bmus at current AWT indexes
    index_1 = np.where(awtDailyBmus == ic)[0][:]
    sel_2 = bmus[index_1]
    set_2 = np.arange(48)
    # get DWT cluster probabilities
    cps = ClusterProbabilities(sel_2, set_2)
    C_T = np.reshape(cps, (8, 6))

    # # axis colors
    # if wt_colors:
    #     caxis = cs_wt[ic]
    # else:
    caxis = 'black'

    # plot axes
    ax = plt.subplot(gs[ic])
    axplot_WT_Probs(
        ax, C_T,
        ttl='WT {0}'.format(ic + 1),
        cmap='Reds', caxis=caxis,
    )
    ax.set_aspect('equal')




