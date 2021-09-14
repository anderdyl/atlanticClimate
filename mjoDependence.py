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
bmus[mask] = dwtBmus+1
bmus[mask2] = twtBmus+50

bmus_dates = dateDay2datetimeDate(timeDWTs)
bmus_dates_months = np.array([d.month for d in bmus_dates])
bmus_dates_days = np.array([d.day for d in bmus_dates])


dataMJO = ReadMatfile('/media/dylananderson/Elements/NC_climate/mjo_australia_2021.mat')
mjoPhase = dataMJO['phase']
mjoRmm1 = dataMJO['rmm1']
mjoRmm2 = dataMJO['rmm2']
dt = datetime.date(1974, 6, 1)
end = datetime.date(2021, 6, 18)
step = relativedelta(days=1)
mjoTime = []
while dt < end:
    mjoTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step



index = np.where((np.asarray(mjoTime) >= datetime.date(1979,2,1)) & (np.asarray(mjoTime) < datetime.date(2021,6,1)))

mjoRmm1 = mjoRmm1[index]
mjoRmm2 = mjoRmm2[index]
mjoPhase = mjoPhase[index]
#mjoTime = mjoTime[index]





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

def Plot_MJO_phases(rmm1, rmm2, phase, show=True):
    'Plot MJO data separated by phase'

    # parameters for custom plot
    size_points = 0.2
    size_lines = 0.8
    l_colors_phase = np.array(
        [
            [1, 0, 0],
            [0.6602, 0.6602, 0.6602],
            [1.0, 0.4961, 0.3125],
            [0, 1, 0],
            [0.2539, 0.4102, 0.8789],
            [0, 1, 1],
            [1, 0.8398, 0],
            [0.2930, 0, 0.5078]
        ]
    )

    color_lines_1 = (0.4102, 0.4102, 0.4102)


    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_fsize, _fsize))
    ax.scatter(rmm1, rmm2, c='b', s=size_points)

    # plot data by phases
    for i in range(1,9):
        ax.scatter(
            rmm1.where(phase==i),
            rmm2.where(phase==i),
            c=np.array([l_colors_phase[i-1]]),
            s=size_points)

    # plot sectors
    ax.plot([-4,4],[-4,4], color='k', linewidth=size_lines)
    ax.plot([-4,4],[4,-4], color='k', linewidth=size_lines)
    ax.plot([-4,4],[0,0],  color='k', linewidth=size_lines)
    ax.plot([0,0], [-4,4], color='k', linewidth=size_lines)

    # axis
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('RMM1')
    plt.ylabel('RMM2')
    ax.set_aspect('equal')

    # show and return figure
    if show: plt.show()
    return fig


#xds = xr.open_dataset(self.paths.site.MJO.hist)



mjoBmus, mjoGroups = MJO_Categories(mjoRmm1,mjoRmm2,mjoPhase)

rm1 = pd.DataFrame(mjoRmm1)
rm2 = pd.DataFrame(mjoRmm2)
ph = pd.DataFrame(mjoPhase)
bmusMJO = pd.DataFrame(mjoBmus)
axMJO = Plot_MJO_phases(rm1,rm2,ph)



def colors_mjo():
    'custom colors for MJO 25 categories'

    l_named_colors = [
        'lightskyblue', 'deepskyblue', 'royalblue', 'mediumblue',
        'darkblue', 'darkblue', 'darkturquoise', 'turquoise',
        'maroon', 'saddlebrown', 'chocolate', 'gold', 'orange',
        'orangered', 'red', 'firebrick', 'Purple', 'darkorchid',
        'mediumorchid', 'magenta', 'mediumslateblue', 'blueviolet',
        'darkslateblue', 'indigo', 'darkgray',
    ]

    # get rgb colors as numpy array
    np_colors_rgb = np.array(
        [mcolors.to_rgb(c) for c in l_named_colors]
    )

    return np_colors_rgb


np_colors_rgb_categ = colors_mjo()


def Plot_MJO_Categories(rmm1, rmm2, categ, show=True):
    'Plot MJO data separated by 25 categories'

    # parameters for custom plot
    size_lines = 0.8
    color_lines_1 = (0.4102, 0.4102, 0.4102)

    # custom colors for mjo 25 categories
    np_colors_rgb_categ = colors_mjo()

    # plot figure
    fig, ax = plt.subplots(1,1, figsize=(_fsize,_fsize))

    # plot sectors
    ax.plot([-4,4],[-4,4], color='k', linewidth=size_lines, zorder=9)
    ax.plot([-4,4],[4,-4], color='k', linewidth=size_lines, zorder=9)
    ax.plot([-4,4],[0,0],  color='k', linewidth=size_lines, zorder=9)
    ax.plot([0,0], [-4,4], color='k', linewidth=size_lines, zorder=9)

    # plot circles
    R = [1, 1.5, 2.5]

    for rr in R:
        ax.add_patch(
            patches.Circle(
                (0,0),
                rr,
                color='k',
                linewidth=size_lines,
                fill=False,
                zorder=9)
        )
    ax.add_patch(
        patches.Circle((0,0),R[0],fc='w',fill=True, zorder=10))

    # plot data by categories
    for i in range(1,25):
        if i>8:
            size_points = 0.2
        else:
            size_points = 1.7

        ax.scatter(
            rmm1.where(categ==i),
            rmm2.where(categ==i),
            c=[np_colors_rgb_categ[i-1]],
            s=size_points
        )

    # last category on top (zorder)
    ax.scatter(
        rmm1.where(categ==25),
        rmm2.where(categ==25),
        c=[np_colors_rgb_categ[-1]],
        s=0.2,
        zorder=12
    )

    # TODO: category number
    rr = 0.3
    ru = 0.2
    l_pn = [
        (-3, -1.5, '1'),
        (-1.5, -3, '2'),
        (1.5-rr, -3, '3'),
        (3-rr, -1.5, '4'),
        (3-rr, 1.5-ru, '5'),
        (1.5-rr, 3-ru, '6'),
        (-1.5, 3-ru, '7'),
        (-3, 1.5-ru, '8'),
        (-2, -1, '9'),
        (-1, -2, '10'),
        (1-rr, -2, '11'),
        (2-rr, -1, '12'),
        (2-rr, 1-ru, '13'),
        (1-rr, 2-ru, '14'),
        (-1, 2-ru, '15'),
        (-2, 1-ru, '16'),
        (-1.3, -0.6, '17'),
        (-0.6, -1.3, '18'),
        (0.6-rr, -1.3, '19'),
        (1.3-rr, -0.6, '20'),
        (1.3-rr, 0.6-ru, '21'),
        (0.6-rr, 1.3-ru, '22'),
        (-0.6, 1.3-ru, '23'),
        (-1.3, 0.6-ru, '24'),
        (0-rr/2, 0-ru/2, '25'),
    ]
    for xt, yt, tt in l_pn:
        ax.text(xt, yt, tt, fontsize=15, fontweight='bold', zorder=11)

    # axis
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('RMM1')
    plt.ylabel('RMM2')
    ax.set_aspect('equal')

    # show and return figure
    if show: plt.show()
    return fig


axMJO2 = Plot_MJO_Categories(rm1,rm2,bmusMJO)


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

fig10 = plt.figure()

gs = gridspec.GridSpec(5, 5, wspace=0.10, hspace=0.15)

for ic in range(25):
    ax = plt.subplot(gs[ic])

    # select DWT bmus at current AWT indexes
    index_1 = np.where(mjoBmus == ic+1)[0][:]
    sel_2 = bmus[index_1]
    set_2 = np.arange(70)
    # get DWT cluster probabilities
    cps = ClusterProbabilities(sel_2, set_2)
    C_T = np.reshape(cps, (10, 7))

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



rmm = np.sqrt(mjoRmm1**2 + mjoRmm2**2)

bigInd = np.where((rmm > 1))
mjoRmm1Big = mjoRmm1[bigInd]
mjoRmm2Big = mjoRmm2[bigInd]
mjoPhaseBig = mjoPhase[bigInd]
bmusBig = bmus[bigInd]

plt.style.use('dark_background')
fig10 = plt.figure()

gs = gridspec.GridSpec(1, 8, wspace=0.10, hspace=0.15)

for ic in range(8):
    ax = plt.subplot(gs[ic])

    # select DWT bmus at current AWT indexes
    index_1 = np.where(mjoPhaseBig == ic+1)[0][:]
    sel_2 = bmusBig[index_1]
    set_2 = np.arange(70)
    # get DWT cluster probabilities
    cps = ClusterProbabilities(sel_2, set_2)
    C_T = np.reshape(cps, (10, 7))

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
        cmap='Reds', caxis=caxis, vmin=0, vmax=0.07
    )
    ax.set_aspect('equal')
