import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import matplotlib.pyplot as plt


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



with open('/home/dylananderson/projects/duckGeomorph/NAO2021.txt', 'r') as fd:
    c = 0
    dataNAO = list()
    for line in fd:
        splitLine = line.split(',')
        secondSplit = splitLine[1].split('/')
        dataNAO.append(float(secondSplit[0]))
nao = np.asarray(dataNAO)

dt = datetime.date(1950, 1, 1)
end = datetime.date(2021, 6, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(months=1)
naoTime = []
while dt < end:
    naoTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step




import pickle

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



data = nao
bins = np.linspace(np.min(data)-.05, np.max(data)+.05, 9)
digitized = np.digitize(data, bins)
bin_means = [data[digitized == i].mean() for i in range(1, len(bins))]


years = np.arange(1979,2022)
months = np.arange(1,13)
awtYears = np.arange(1880,2021)

naoTIME = naoTime[353:]
naoTIME.append(datetime.date(2021,6,1))
naoDailyBmus = np.nan * np.ones(np.shape(bmus))
for hh in range(len(naoTime)):
    #for mm in months:
        # indexDWT = np.where((np.asarray(bmus_dates) >= datetime.date(hh,6,1)) & (np.asarray(bmus_dates) <= datetime.date(hh+1,6,1)))
    indexDWT = np.where((np.asarray(bmus_dates) >= naoTIME[hh]) & (np.asarray(bmus_dates) <= naoTIME[hh+1]))
    #indexAWT = np.where((awtYears == hh))
    naoDailyBmus[indexDWT] = nao[hh]*np.ones(len(indexDWT[0]))



figClimate = plt.figure()
ax3Cl = plt.subplot2grid((2,1),(0,0),rowspan=1,colspan=1)
ax3Cl.plot(naoTime,nao)
ax3Cl.set_xlim([datetime.date(1979,1,1),datetime.date(2021,5,1)])
ax3Cl.set_ylabel('NAO')
ax4Cl = plt.subplot2grid((2,1),(1,0),rowspan=1,colspan=1)
ax4Cl.plot(naoTime,digitized)
ax4Cl.set_xlim([datetime.date(1979,1,1),datetime.date(2021,5,1)])
ax4Cl.set_ylabel('Bins')









