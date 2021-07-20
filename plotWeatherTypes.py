import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import numpy as np
import os
import datetime
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import mda
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import datetime
from dateutil.relativedelta import relativedelta
import random
import xarray as xr
import geopy.distance
from dipy.segment.metric import Metric
from dipy.segment.metric import ResampleFeature
from dipy.segment.clustering import QuickBundles
from mpl_toolkits.basemap import Basemap


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color


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

class GPSDistance(Metric):
    """computer the average GPS distance between two streamlines"""
    def __init__(self):
        super(GPSDistance, self).__init__(feature=ResampleFeature(nb_points=20))

    def are_compatible(self, shape1, shape2):
        return len(shape1) == len(shape2)

    # def dist(self,v1,v2):
    #     x = [geopy.distance.vincenty([p[0][0],p[0][1]], [p[1][0],p[1][1]]).km for p in list(zip(v1,v2))]
    #     currD = np.mean(x)
    #     return currD
    def dist(self, v1, v2):
        x = [geopy.distance.distance([p[0][0], p[0][1]], [p[1][0], p[1][1]]).kilometers for p in list(zip(v1, v2))]
        currD = np.mean(x)
        return currD






#DWT = ReadMatfile('/media/dylananderson/Elements1/NC_climate/Nags_Head_DWTs_49_w20minDates_2degree_plus5TCs3dayafterentering2.mat')
#DWT = ReadMatfile('/media/dylananderson/Elements1/NC_climate/Nags_Head_DWTS_25_w50minDates_plus5TCs_goodorder.mat')
DWT = ReadMatfile('/media/dylananderson/Elements/NC_climate/Nags_Head_DWTs_25_w50minDates_2degree_plus5TCs6daysAroundEntering.mat')
PCAmat = ReadMatfile('/media/dylananderson/Elements/NC_climate/Nags_Head_SLPs_2degree_memory_July2020.mat')
SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/NorthAtlanticSLPs_June2021.mat')


numDWTs = 30
mycols = ReadMatfile('/media/dylananderson/Elements/shusin6_contents/codes/mycmap_col.mat')
mycmap = mycols['mycmap_col']
# Need to get the dates for the bmus into the correct format (datetime)
def datevec2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime.datetime(d[0], d[1], d[2], d[3], d[4]) for d in d_vec]

dwtDates = np.array(datevec2datetime(DWT['DWT']['dates']))
dwtBmus = DWT['DWT']['bmus']
dwtBMUS = dwtBmus
order = DWT['DWT']['order']

import matplotlib.cm as cm

etcolors = cm.rainbow(np.linspace(0, 1, numDWTs-5))
tccolors = np.flipud(cm.gray(np.linspace(0,1,6)))

dwtcolors = np.vstack((etcolors,tccolors[1:,:]))


# etcolors = cm.viridis(np.linspace(0, 1, 48-11))
# tccolors = np.flipud(cm.autumn(np.linspace(0,1,12)))
#
# dwtcolors = np.vstack((etcolors,tccolors[1:,:]))
#




repmatDesviacion = np.tile(DWT['PCA']['Desviacion'], (25,1))
repmatMedia = np.tile(DWT['PCA']['Media'], (25,1))
Km_ = np.multiply(DWT['KMA']['centroids'],repmatDesviacion) + repmatMedia



[mK, nK] = np.shape(Km_)

Km_slp = Km_[:,0:int(nK/2)]
Km_grd = Km_[:,int(nK/2):]
X_B = DWT['X_B']
Y_B = DWT['Y_B']
SLP_C = DWT['SLP_C']
Km_slp = Km_slp[:,0:len(X_B)]
Km_grd = Km_grd[:,1:len(X_B)]

Xs = np.arange(np.min(X_B),np.max(X_B),2)
#Xs = np.arange(-(360-np.min(X_B)),(np.max(X_B)-360))
Ys = np.arange(np.min(Y_B),np.max(Y_B),2)
lenXB = len(X_B)
[XR,YR] = np.meshgrid(Xs,Ys)
sea_nodes = []
for qq in range(lenXB-1):
    sea_nodes.append(np.where((XR == X_B[qq]) & (YR == Y_B[qq])))
    #print(indexTest[0])


flat_list = [item for sublist in sea_nodes for item in sublist]


plt.style.use('default')
fig2 = plt.figure(figsize=(8,10))
gs1 = gridspec.GridSpec(6, 5)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
counter = 0
plotIndx = 0
plotIndy = 0
for qq in range(numDWTs):
#qq = 0

    #ax = plt.subplot2grid((7, 7), (plotIndx, plotIndy), rowspan=1, colspan=1)
    ax = plt.subplot(gs1[qq])
    #num = qq
    num = order[qq]
    #num = newOrder[qq]+1
    wt = np.ones((np.shape(XR))) * np.nan

    if num < 25:
        num_index = np.where((dwtBMUS == num))
        temp = Km_slp[(num-1),:]/100 - np.nanmean(SLP_C,axis=0)/100
    else:
        num_index = np.where((dwtBmus == num))
        temp = np.nanmean(SLP_C[num_index,:],axis=1)/100 - np.nanmean(SLP_C,axis=0)/100
        temp= temp.flatten()
    for tt in range(len(sea_nodes)):
        wt[sea_nodes[tt]] = temp[tt]
#reshaped = np.tile(wt,(np.shape(XR)))


#m = Basemap(projection='cyl',lon_0=320,lat_0=0)
    m = Basemap(projection='merc',llcrnrlat=-40,urcrnrlat=50,llcrnrlon=275,urcrnrlon=370,lat_ts=10,resolution='c')
#m = Basemap(projection='stere',lon_0=-120,lat_0=20,llcrnrlat=-10,llcrnrlon=50,urcrnrlat=50, urcrnrlon=190)
#m = Basemap(width=12000000,height=8000000,resolution='l',projection='stere',lat_ts=30,lat_0=20,lon_0=-70.)
    m.fillcontinents(color=dwtcolors[qq])
    m.drawcoastlines()
#parallels = np.arange(-90.,90,30.)
#meridians = np.arange(0.,360.,20.)
#m.drawparallels(parallels,labels=[1,1,0,0],fontsize=10)  # 緯度度線、在左右兩邊加緯度標籤
#m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    clevels = np.arange(-17,17,1)
    cx,cy =m(XR,YR)  # convert to map projection coordinate
#CS = m.pcolormesh(cx,cy,wt,clevels,cmap=cm.jet,shading='gouraud')
    CS = m.contourf(cx,cy,wt,clevels,vmin=-7.5,vmax=7.5,cmap=cm.RdBu_r,shading='gouraud')

    #plt.colorbar(CS,orientation='horizontal')
    if plotIndx < 7:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    counter = counter + 1
    if plotIndy < 7:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
#date=time_d[0].strftime('%Y/%m/%d')
#plt.title('Cylindrical, T at 1000 hPa, GFS'+date)
#plt.tight_layout()
colormap = cm.RdBu_r
normalize = mcolors.Normalize(vmin=-7.5, vmax=7.5)

s_map2 = cm.ScalarMappable(norm=normalize, cmap=colormap)
#s_map2.set_array(colorparam)
fig2.subplots_adjust(right=0.85)
cbar_ax2 = fig2.add_axes([0.91, 0.15, 0.02, 0.7])
cbar2 = fig2.colorbar(s_map2, cax=cbar_ax2)
cbar2.set_label('slp anom (mbar)')

plt.show()
