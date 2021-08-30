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



with open(r"filteredStorms.pickle", "rb") as input_file:
   filteredStorms = pickle.load(input_file)

filteredHs = filteredStorms['filteredHs']
filteredTp = filteredStorms['filteredTp']
filteredDm = filteredStorms['filteredDm']
filteredNTR = filteredStorms['filteredNTR']
filteredDur = filteredStorms['filteredDur']
















