import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from dipy.segment.clustering import QuickBundles
from mpl_toolkits.basemap import Basemap
import geopy.distance
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import Metric
import datetime
import random
import matplotlib.cm as cm


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


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color



def datevec2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime.datetime(d[0], d[1], d[2], d[3], d[4]) for d in d_vec]

def dateDay2datetime(d_vec):
    '''
    Returns datetime list from a datevec matrix
    d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
    '''
    return [datetime.datetime(d[0], d[1], d[2]) for d in d_vec]


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


file = '/media/dylananderson/Elements/SERDP/Data/TC_tracks/IBTrACS.NA.v04r00.nc'
data = xr.open_dataset(file)
#merged = data['merged'].values
TCtime = data['time'].values
TClon = data['lon'].values
TClat = data['lat'].values
# TCpres = data['wmo_pres']
# TCwind = data['wmo_wind']
TCpres = data['usa_pres']
TCwind = data['usa_wind']

indexTC = np.arange(1,len(TCtime))

streams = []
streamTime = []
streamWind = []
streamPres = []
for hh in range(len(TCtime)):
    indexReal = np.where(TClat[hh,:] > 0)
    if len(indexReal[0])>1:
        lat_lng_data = np.c_[TClat[hh,indexReal[0]], TClon[hh,indexReal[0]]]
        streams.append(lat_lng_data)
        streamTime.append(TCtime[hh,indexReal[0]])
        streamPres.append(TCpres[hh,indexReal[0]])
        streamWind.append(TCwind[hh,indexReal[0]])

metric = GPSDistance()
qb = QuickBundles(threshold=2750,metric=metric)
clusters = qb.cluster(streams)
print("Nb. clusters:",len(clusters))

fig = plt.figure()
#m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=60, llcrnrlon=250, urcrnrlon=-10, lat_ts=10,resolution='c')
#m = Basemap(projection='ortho',lat_0=0, lon_0=0)
# #m.fillcontinents(color=dwtcolors[qq])
plotIndy=0
plotIndx=0
for clustersIndex in range(6):
    p1 = plt.subplot2grid((2, 3), (plotIndx, plotIndy), rowspan=1, colspan=1)
    m = Basemap(llcrnrlon=-120.7, llcrnrlat=0., urcrnrlon=-10.1, urcrnrlat=60, projection='merc', lat_1=30., lat_2=60.,
                lat_0=34.83158, lon_0=-98.)

    color = randomcolor()
    for i in clusters[clustersIndex].indices:
        cx, cy = m(streams[i][:,1], streams[i][:,0])  # convert to map projection coordinate
        # cx = streams[i][:,1]
        # cy = streams[i][:,0]
        m.plot(cx, cy, marker=None, color=color)
    m.drawcoastlines()
    if plotIndy < 2:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
for i in clusters[6].indices:
    cx, cy = m(streams[i][:,1], streams[i][:,0])  # convert to map projection coordinate
        # cx = streams[i][:,1]
        # cy = streams[i][:,0]
    m.plot(cx, cy, marker=None, color=color)






fig = plt.figure()
#m = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=60, llcrnrlon=250, urcrnrlon=-10, lat_ts=10,resolution='c')
#m = Basemap(projection='ortho',lat_0=0, lon_0=0)
# #m.fillcontinents(color=dwtcolors[qq])
plotIndy=0
plotIndx=0
for clustersIndex in range(6):
    p1 = plt.subplot2grid((2, 3), (plotIndx, plotIndy), rowspan=1, colspan=1)
    m = Basemap(llcrnrlon=-120.7, llcrnrlat=0., urcrnrlon=-10.1, urcrnrlat=60, projection='merc', lat_1=30., lat_2=60.,
                lat_0=34.83158, lon_0=-98.)

    color = randomcolor()
    for i in clusters[clustersIndex].indices:
        cx, cy = m(streams[i][:,1], streams[i][:,0])  # convert to map projection coordinate
        m.plot(cx[0], cy[0], marker='.', color=color)
    m.drawcoastlines()
    if plotIndy < 2:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
for i in clusters[6].indices:
    cx, cy = m(streams[i][:, 1], streams[i][:, 0])  # convert to map projection coordinate
    m.plot(cx[0], cy[0], marker='.', color=color)









fig = plt.figure()

p1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
m = Basemap(llcrnrlon=-120.7, llcrnrlat=0., urcrnrlon=-10.1, urcrnrlat=60, projection='merc', lat_1=30., lat_2=60.,lat_0=34.83158, lon_0=-98.)
for clustersIndex in range(6):
    color = randomcolor()
    cx, cy = m(clusters.centroids[clustersIndex][:,1], clusters.centroids[clustersIndex][:,0])  # convert to map projection coordinate
    m.plot(cx, cy, marker=None, color=color)

# cx, cy = m(clusters.centroids[6][:,1], clusters.centroids[6][:,0])  # convert to map projection coordinate
# m.plot(cx, cy, marker=None, color=color)
m.drawcoastlines()

cluster1minTime = []
cluster1Time = []
cluster1Pres = []
cluster1Wind = []
for i in clusters[0].indices:
    arrayYear = dt2cal(streamTime[i][0])[0]
    if arrayYear > 1978:
        arrayTime = [dt2cal(dt) for dt in streamTime[i]]
        presTemp = streamPres[i].values
        windTemp = streamWind[i].values

        minPresind = np.where((np.nanmin(presTemp)==presTemp))
        if len(minPresind[0]) > 0:
            datesToAppendMin = arrayTime[minPresind[0][0]]
            print('chose {}, started {}, ended {}'.format(datesToAppendMin,arrayTime[0],arrayTime[-1]))
            cluster1minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
            cluster1Pres.append(np.unique(presTemp[minPresind]))
        else:
            minWindind = np.where((np.nanmax(windTemp)==windTemp))
            if len(minWindind[0]) > 0:
                datesToAppendMin = arrayTime[minWindind[0][0]]
                cluster1minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster1Pres.append(np.array(0))

            else:
                midStorm = np.round((len(arrayTime)/2))
                datesToAppendMin = arrayTime[int(midStorm)]
                cluster1minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster1Pres.append(np.array(500))

        arrayDay = np.asarray([list((hh[0],hh[1],hh[2])) for hh in arrayTime])
        uniqueDay = np.unique(arrayDay[:,2],return_index=True)
        datesToAppend = arrayDay[uniqueDay[1],:]
        cluster1Time.append(datesToAppend)


cluster2Time = []
cluster2minTime = []
cluster2Pres = []
for i in clusters[1].indices:
    arrayYear = dt2cal(streamTime[i][0])[0]
    if arrayYear > 1978:
        arrayTime = [dt2cal(dt) for dt in streamTime[i]]
        presTemp = streamPres[i].values
        minPresind = np.where((np.nanmin(presTemp)==presTemp))
        if len(minPresind[0]) > 0:
            datesToAppendMin = arrayTime[minPresind[0][0]]
            cluster2minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
            cluster2Pres.append(np.unique(presTemp[minPresind]))

        else:
            windTemp = streamWind[i].values
            minWindind = np.where((np.nanmax(windTemp)==windTemp))
            if len(minWindind[0]) > 0:
                datesToAppendMin = arrayTime[minWindind[0][0]]
                cluster2minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster2Pres.append(np.array(0))

            else:
                midStorm = np.round((len(arrayTime)/2))
                datesToAppendMin = arrayTime[int(midStorm)]
                cluster2minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster2Pres.append(np.array(500))

        arrayDay = np.asarray([list((hh[0],hh[1],hh[2])) for hh in arrayTime])
        uniqueDay = np.unique(arrayDay[:,2],return_index=True)
        datesToAppend = arrayDay[uniqueDay[1],:]
        cluster2Time.append(datesToAppend)


cluster3Time = []
cluster3minTime = []
cluster3Pres = []
for i in clusters[2].indices:
    arrayYear = dt2cal(streamTime[i][0])[0]
    if arrayYear > 1978:
        arrayTime = [dt2cal(dt) for dt in streamTime[i]]
        presTemp = streamPres[i].values
        minPresind = np.where((np.nanmin(presTemp)==presTemp))
        if len(minPresind[0]) > 0:
            datesToAppendMin = arrayTime[minPresind[0][0]]
            cluster3minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
            cluster3Pres.append(np.unique(presTemp[minPresind]))

        else:
            windTemp = streamWind[i].values
            minWindind = np.where((np.nanmax(windTemp)==windTemp))
            if len(minWindind[0]) > 0:
                datesToAppendMin = arrayTime[minWindind[0][0]]
                cluster3minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster3Pres.append(np.array(0))

            else:
                midStorm = np.round((len(arrayTime)/2))
                datesToAppendMin = arrayTime[int(midStorm)]
                cluster3minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster3Pres.append(np.array(500))

        arrayDay = np.asarray([list((hh[0],hh[1],hh[2])) for hh in arrayTime])
        uniqueDay = np.unique(arrayDay[:,2],return_index=True)
        datesToAppend = arrayDay[uniqueDay[1],:]
        cluster3Time.append(datesToAppend)


cluster4Time = []
cluster4minTime = []
cluster4Pres = []
for i in clusters[3].indices:
    arrayYear = dt2cal(streamTime[i][0])[0]
    if arrayYear > 1978:
        arrayTime = [dt2cal(dt) for dt in streamTime[i]]
        presTemp = streamPres[i].values
        minPresind = np.where((np.nanmin(presTemp)==presTemp))
        if len(minPresind[0]) > 0:
            datesToAppendMin = arrayTime[minPresind[0][0]]
            cluster4minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
            cluster4Pres.append(np.unique(presTemp[minPresind]))

        else:
            windTemp = streamWind[i].values
            minWindind = np.where((np.nanmax(windTemp)==windTemp))
            if len(minWindind[0]) > 0:
                datesToAppendMin = arrayTime[minWindind[0][0]]
                cluster4minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster4Pres.append(np.array(0))

            else:
                midStorm = np.round((len(arrayTime)/2))
                datesToAppendMin = arrayTime[int(midStorm)]
                cluster4minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster4Pres.append(np.array(500))

        arrayDay = np.asarray([list((hh[0],hh[1],hh[2])) for hh in arrayTime])
        uniqueDay = np.unique(arrayDay[:,2],return_index=True)
        datesToAppend = arrayDay[uniqueDay[1],:]
        cluster4Time.append(datesToAppend)

cluster5Time = []
cluster5minTime = []
cluster5Pres = []
for i in clusters[4].indices:
    arrayYear = dt2cal(streamTime[i][0])[0]
    if arrayYear > 1978:
        arrayTime = [dt2cal(dt) for dt in streamTime[i]]
        presTemp = streamPres[i].values
        minPresind = np.where((np.nanmin(presTemp)==presTemp))
        if len(minPresind[0]) > 0:
            datesToAppendMin = arrayTime[minPresind[0][0]]
            cluster5minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
            cluster5Pres.append(np.unique(presTemp[minPresind]))

        else:
            windTemp = streamWind[i].values
            minWindind = np.where((np.nanmax(windTemp)==windTemp))
            if len(minWindind[0]) > 0:
                datesToAppendMin = arrayTime[minWindind[0][0]]
                cluster5minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster5Pres.append(np.array(0))

            else:
                midStorm = np.round((len(arrayTime)/2))
                datesToAppendMin = arrayTime[int(midStorm)]
                cluster5minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster5Pres.append(np.array(500))

        arrayDay = np.asarray([list((hh[0],hh[1],hh[2])) for hh in arrayTime])
        uniqueDay = np.unique(arrayDay[:,2],return_index=True)
        datesToAppend = arrayDay[uniqueDay[1],:]
        cluster5Time.append(datesToAppend)

for i in clusters[6].indices:
    arrayYear = dt2cal(streamTime[i][0])[0]
    if arrayYear > 1978:
        arrayTime = [dt2cal(dt) for dt in streamTime[i]]
        presTemp = streamPres[i].values
        minPresind = np.where((np.nanmin(presTemp)==presTemp))
        if len(minPresind[0]) > 0:
            datesToAppendMin = arrayTime[minPresind[0][0]]
            cluster5minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
            cluster5Pres.append(np.unique(presTemp[minPresind]))

        else:
            windTemp = streamWind[i].values
            minWindind = np.where((np.nanmax(windTemp)==windTemp))
            if len(minWindind[0]) > 0:
                datesToAppendMin = arrayTime[minWindind[0][0]]
                cluster5minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster5Pres.append(np.array(0))

            else:
                midStorm = np.round((len(arrayTime)/2))
                datesToAppendMin = arrayTime[int(midStorm)]
                cluster5minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster5Pres.append(np.array(500))

        arrayDay = np.asarray([list((hh[0],hh[1],hh[2])) for hh in arrayTime])
        uniqueDay = np.unique(arrayDay[:,2],return_index=True)
        datesToAppend = arrayDay[uniqueDay[1],:]
        cluster5Time.append(datesToAppend)






cluster6Time = []
cluster6minTime = []
cluster6Pres = []
for i in clusters[5].indices:
    arrayYear = dt2cal(streamTime[i][0])[0]
    if arrayYear > 1978:
        arrayTime = [dt2cal(dt) for dt in streamTime[i]]
        presTemp = streamPres[i].values
        minPresind = np.where((np.nanmin(presTemp)==presTemp))
        if len(minPresind[0]) > 0:
            datesToAppendMin = arrayTime[minPresind[0][0]]
            cluster6minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
            cluster6Pres.append(np.unique(presTemp[minPresind]))

        else:
            windTemp = streamWind[i].values
            minWindind = np.where((np.nanmax(windTemp)==windTemp))
            if len(minWindind[0]) > 0:
                datesToAppendMin = arrayTime[minWindind[0][0]]
                cluster6minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster6Pres.append(np.array(0))

            else:
                midStorm = np.round((len(arrayTime)/2))
                datesToAppendMin = arrayTime[int(midStorm)]
                cluster6minTime.append(np.array([datesToAppendMin[0],datesToAppendMin[1],datesToAppendMin[2]]))
                cluster6Pres.append(np.array(500))

        arrayDay = np.asarray([list((hh[0],hh[1],hh[2])) for hh in arrayTime])
        uniqueDay = np.unique(arrayDay[:,2],return_index=True)
        datesToAppend = arrayDay[uniqueDay[1],:]
        cluster6Time.append(datesToAppend)



plt.figure()
p1 = plt.subplot2grid((6,1),(0,0))
p1.plot(dateDay2datetime(cluster1minTime),cluster1Pres,'o')
p2 = plt.subplot2grid((6,1),(1,0))
p2.plot(dateDay2datetime(cluster2minTime),cluster2Pres,'o')
p3 = plt.subplot2grid((6,1),(2,0))
p3.plot(dateDay2datetime(cluster3minTime),cluster3Pres,'o')
p4 = plt.subplot2grid((6,1),(3,0))
p4.plot(dateDay2datetime(cluster4minTime),cluster4Pres,'o')
p5 = plt.subplot2grid((6,1),(4,0))
p5.plot(dateDay2datetime(cluster5minTime),cluster5Pres,'o')
p6 = plt.subplot2grid((6,1),(5,0))
p6.plot(dateDay2datetime(cluster6minTime),cluster6Pres,'o')






SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/NorthAtlanticSLPs_June2021.mat')
# SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/Nags_Head_SLPs_2degree_memory_June2021.mat')

X_in = SLPs['X_in']
Y_in = SLPs['Y_in']
SLP = SLPs['slp_mem']
SLPtime = SLPs['time']

DWT = ReadMatfile('/media/dylananderson/Elements/NC_climate/Nags_Head_DWTs_25_w50minDates_2degree_plus5TCs6daysAroundEntering.mat')

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
Ys = np.arange(np.min(Y_B),np.max(Y_B),2)
lenXB = len(X_B)
[XR,YR] = np.meshgrid(Xs,Ys)
sea_nodes = []
for qq in range(lenXB-1):
    sea_nodes.append(np.where((XR == X_B[qq]) & (YR == Y_B[qq])))


#
# Dindices1 = []
# for hh in range(len(cluster1Time)):
#     times = cluster1Time[hh]
#     for qq in range(len(times)):
#         dIndex = np.where((times[qq][0]==SLPtime[:,0]) & (times[qq][1]==SLPtime[:,1]) & (times[qq][2]==SLPtime[:,2]))
#         Dindices1.append(dIndex[0][0])
# tc1Dindices = np.asarray(Dindices1)
# cluster1SLPIndex = np.unique(tc1Dindices)
# cluster1SLPs = np.nanmean(SLP[cluster1SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
# Dindices2 = []
# for hh in range(len(cluster2Time)):
#     times = cluster2Time[hh]
#     for qq in range(len(times)):
#         dIndex = np.where((times[qq][0]==SLPtime[:,0]) & (times[qq][1]==SLPtime[:,1]) & (times[qq][2]==SLPtime[:,2]))
#         Dindices2.append(dIndex[0][0])
# tc2Dindices = np.asarray(Dindices2)
# cluster2SLPIndex = np.unique(tc2Dindices)
# cluster2SLPs = np.nanmean(SLP[cluster2SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
# Dindices3 = []
# for hh in range(len(cluster3Time)):
#     times = cluster3Time[hh]
#     for qq in range(len(times)):
#         dIndex = np.where((times[qq][0]==SLPtime[:,0]) & (times[qq][1]==SLPtime[:,1]) & (times[qq][2]==SLPtime[:,2]))
#         Dindices3.append(dIndex[0][0])
# tc3Dindices = np.asarray(Dindices3)
# cluster3SLPIndex = np.unique(tc3Dindices)
# cluster3SLPs = np.nanmean(SLP[cluster3SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
# Dindices4 = []
# for hh in range(len(cluster4Time)):
#     times = cluster4Time[hh]
#     for qq in range(len(times)):
#         dIndex = np.where((times[qq][0]==SLPtime[:,0]) & (times[qq][1]==SLPtime[:,1]) & (times[qq][2]==SLPtime[:,2]))
#         Dindices4.append(dIndex[0][0])
# tc4Dindices = np.asarray(Dindices4)
# cluster4SLPIndex = np.unique(tc4Dindices)
# cluster4SLPs = np.nanmean(SLP[cluster4SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
# Dindices5 = []
# for hh in range(len(cluster5Time)):
#     times = cluster5Time[hh]
#     for qq in range(len(times)):
#         dIndex = np.where((times[qq][0]==SLPtime[:,0]) & (times[qq][1]==SLPtime[:,1]) & (times[qq][2]==SLPtime[:,2]))
#         Dindices5.append(dIndex[0][0])
# tc5Dindices = np.asarray(Dindices5)
# cluster5SLPIndex = np.unique(tc5Dindices)
# cluster5SLPs = np.nanmean(SLP[cluster5SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
# Dindices6 = []
# for hh in range(len(cluster6Time)):
#     times = cluster6Time[hh]
#     for qq in range(len(times)):
#         dIndex = np.where((times[qq][0]==SLPtime[:,0]) & (times[qq][1]==SLPtime[:,1]) & (times[qq][2]==SLPtime[:,2]))
#         Dindices6.append(dIndex[0][0])
# tc6Dindices = np.asarray(Dindices6)
# cluster6SLPIndex = np.unique(tc6Dindices)
# cluster6SLPs = np.nanmean(SLP[cluster6SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
#
# ## IF USING THE ESTELA SLP PATTERNS
# Dindices1 = []
# for hh in range(len(cluster1minTime)):
#     times = cluster1minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & ((times[1])==SLPtime[:,1]) & ((int(times[2]))==SLPtime[:,2]))
#     Dindices1.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
# tc1Dindices = np.asarray(Dindices1)
# cluster1SLPIndex = np.unique(tc1Dindices)
# cluster1SLPs = np.ones((np.shape(XR))) * np.nan
# temp = np.nanmean(SLP[cluster1SLPIndex,:],axis=0)/100 - np.nanmean(SLP, axis=0) / 100
# temp = temp.flatten()
# for tt in range(len(sea_nodes)):
#     cluster1SLPs[sea_nodes[tt]] = temp[tt]
#
# Dindices2 = []
# for hh in range(len(cluster2minTime)):
#     times = cluster2minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & ((times[1])==SLPtime[:,1]) & ((int(times[2]))==SLPtime[:,2]))
#     Dindices2.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
# tc2Dindices = np.asarray(Dindices2)
# cluster2SLPIndex = np.unique(tc2Dindices)
# cluster2SLPs = np.ones((np.shape(XR))) * np.nan
# temp = np.nanmean(SLP[cluster2SLPIndex,:],axis=0)/100 - np.nanmean(SLP, axis=0) / 100
# temp = temp.flatten()
# for tt in range(len(sea_nodes)):
#     cluster2SLPs[sea_nodes[tt]] = temp[tt]
#
# Dindices3 = []
# for hh in range(len(cluster3minTime)):
#     times = cluster3minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & ((times[1])==SLPtime[:,1]) & ((int(times[2]))==SLPtime[:,2]))
#     Dindices3.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
# tc3Dindices = np.asarray(Dindices3)
# cluster3SLPIndex = np.unique(tc3Dindices)
# cluster3SLPs = np.ones((np.shape(XR))) * np.nan
# temp = np.nanmean(SLP[cluster3SLPIndex,:],axis=0)/100 - np.nanmean(SLP, axis=0) / 100
# temp = temp.flatten()
# for tt in range(len(sea_nodes)):
#     cluster3SLPs[sea_nodes[tt]] = temp[tt]
#
# Dindices4 = []
# for hh in range(len(cluster4minTime)):
#     times = cluster4minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & ((times[1])==SLPtime[:,1]) & ((int(times[2]))==SLPtime[:,2]))
#     Dindices4.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
# tc4Dindices = np.asarray(Dindices4)
# cluster4SLPIndex = np.unique(tc4Dindices)
# cluster4SLPs = np.ones((np.shape(XR))) * np.nan
# temp = np.nanmean(SLP[cluster4SLPIndex,:],axis=0)/100 - np.nanmean(SLP, axis=0) / 100
# temp = temp.flatten()
# for tt in range(len(sea_nodes)):
#     cluster4SLPs[sea_nodes[tt]] = temp[tt]
#
# Dindices5 = []
# for hh in range(len(cluster5minTime)):
#     times = cluster5minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & ((times[1])==SLPtime[:,1]) & ((int(times[2]))==SLPtime[:,2]))
#     Dindices5.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
# tc5Dindices = np.asarray(Dindices5)
# cluster5SLPIndex = np.unique(tc5Dindices)
# cluster5SLPs = np.ones((np.shape(XR))) * np.nan
# temp = np.nanmean(SLP[cluster5SLPIndex,:],axis=0)/100 - np.nanmean(SLP, axis=0) / 100
# temp = temp.flatten()
# for tt in range(len(sea_nodes)):
#     cluster5SLPs[sea_nodes[tt]] = temp[tt]
#
# Dindices6 = []
# for hh in range(len(cluster6minTime)):
#     times = cluster6minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & ((times[1])==SLPtime[:,1]) & ((int(times[2]))==SLPtime[:,2]))
#     Dindices6.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
# tc6Dindices = np.asarray(Dindices6)
# cluster6SLPIndex = np.unique(tc6Dindices)
# cluster6SLPs = np.ones((np.shape(XR))) * np.nan
# temp = np.nanmean(SLP[cluster6SLPIndex,:],axis=0)/100 - np.nanmean(SLP, axis=0) / 100
# temp = temp.flatten()
# for tt in range(len(sea_nodes)):
#     cluster6SLPs[sea_nodes[tt]] = temp[tt]
#


# IF USING THE STRAIGHT SLP PATTERNS
Dindices1 = []
for hh in range(len(cluster1minTime)):
    times = cluster1minTime[hh]
    dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
    Dindices1.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
tc1Dindices = np.asarray(Dindices1)
cluster1SLPIndex = np.unique(tc1Dindices)
cluster1SLPs = np.nanmean(SLP[cluster1SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
cluster1PeakTimes = SLPtime[cluster1SLPIndex,:]

Dindices2 = []
for hh in range(len(cluster2minTime)):
    times = cluster2minTime[hh]
    dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
    Dindices2.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
tc2Dindices = np.asarray(Dindices2)
cluster2SLPIndex = np.unique(tc2Dindices)
cluster2SLPs = np.nanmean(SLP[cluster2SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
cluster2PeakTimes = SLPtime[cluster2SLPIndex,:]

Dindices3 = []
for hh in range(len(cluster3minTime)):
    times = cluster3minTime[hh]
    dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
    Dindices3.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
tc3Dindices = np.asarray(Dindices3)
cluster3SLPIndex = np.unique(tc3Dindices)
cluster3SLPs = np.nanmean(SLP[cluster3SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
cluster3PeakTimes = SLPtime[cluster3SLPIndex,:]

Dindices4 = []
for hh in range(len(cluster4minTime)):
    times = cluster4minTime[hh]
    dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
    Dindices4.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
tc4Dindices = np.asarray(Dindices4)
cluster4SLPIndex = np.unique(tc4Dindices)
cluster4SLPs = np.nanmean(SLP[cluster4SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
cluster4PeakTimes = SLPtime[cluster4SLPIndex,:]

Dindices5 = []
for hh in range(len(cluster5minTime)):
    times = cluster5minTime[hh]
    dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
    Dindices5.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
tc5Dindices = np.asarray(Dindices5)
cluster5SLPIndex = np.unique(tc5Dindices)
cluster5SLPs = np.nanmean(SLP[cluster5SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
cluster5PeakTimes = SLPtime[cluster5SLPIndex,:]

Dindices6 = []
for hh in range(len(cluster6minTime)):
    times = cluster6minTime[hh]
    dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
    Dindices6.append(np.arange((dIndex[0][0]-1),(dIndex[0][0]+2)))
tc6Dindices = np.asarray(Dindices6)
cluster6SLPIndex = np.unique(tc6Dindices)
cluster6SLPs = np.nanmean(SLP[cluster6SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
cluster6PeakTimes = SLPtime[cluster6SLPIndex,:]





#
#
# Dindices1 = []
# for hh in range(len(cluster1minTime)):
#     times = cluster1minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
#     Dindices1.append(dIndex[0][0])
# tc1Dindices = np.asarray(Dindices1)
# cluster1SLPIndex = np.unique(tc1Dindices)
# cluster1SLPs = np.nanmean(SLP[cluster1SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
# Dindices2 = []
# for hh in range(len(cluster2minTime)):
#     times = cluster2minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
#     Dindices2.append(dIndex[0][0])
# tc2Dindices = np.asarray(Dindices2)
# cluster2SLPIndex = np.unique(tc2Dindices)
# cluster2SLPs = np.nanmean(SLP[cluster2SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
#
# Dindices3 = []
# for hh in range(len(cluster3minTime)):
#     times = cluster3minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
#     Dindices3.append(dIndex[0][0])
# tc3Dindices = np.asarray(Dindices3)
# cluster3SLPIndex = np.unique(tc3Dindices)
# cluster3SLPs = np.nanmean(SLP[cluster3SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
#
# Dindices4 = []
# for hh in range(len(cluster4minTime)):
#     times = cluster4minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
#     Dindices4.append(dIndex[0][0])
# tc4Dindices = np.asarray(Dindices4)
# cluster4SLPIndex = np.unique(tc4Dindices)
# cluster4SLPs = np.nanmean(SLP[cluster4SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
#
# Dindices5 = []
# for hh in range(len(cluster5minTime)):
#     times = cluster5minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
#     Dindices5.append(dIndex[0][0])
# tc5Dindices = np.asarray(Dindices5)
# cluster5SLPIndex = np.unique(tc5Dindices)
# cluster5SLPs = np.nanmean(SLP[cluster5SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100
#
# Dindices6 = []
# for hh in range(len(cluster6minTime)):
#     times = cluster6minTime[hh]
#     dIndex = np.where((times[0]==SLPtime[:,0]) & (times[1]==SLPtime[:,1]) & (times[2]==SLPtime[:,2]))
#     Dindices6.append(dIndex[0][0])
# tc6Dindices = np.asarray(Dindices6)
# cluster6SLPIndex = np.unique(tc6Dindices)
# cluster6SLPs = np.nanmean(SLP[cluster6SLPIndex,:],axis=0).reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100


#temp = np.nanmean(SLP_C[num_index, :], axis=1) / 100 - np.nanmean(SLP_C, axis=0) / 100

#wt = SLP[2000,:].reshape(73,43)/100 - np.nanmean(SLP, axis=0).reshape(73,43) / 100

fig3 = plt.figure()
clevels = np.arange(-20,20,1)
 # convert to map projection coordinate
# cx,cy =m(XR,YR)  # convert to map projection coordinate

ax1 = plt.subplot2grid((2,3),(0,0),rowspan=1,colspan=1)
m = Basemap(projection='merc',llcrnrlat=-10,urcrnrlat=70,llcrnrlon=245,urcrnrlon=370,lat_ts=10,resolution='c')
#m.fillcontinents(color=dwtcolors[qq])
cx,cy =m(X_in,Y_in)
m.drawcoastlines()
CS = m.contourf(cx,cy,cluster1SLPs.T,clevels,vmin=-5,vmax=7,cmap=cm.RdBu_r,shading='gouraud')

ax2 = plt.subplot2grid((2,3),(0,1),rowspan=1,colspan=1)
m = Basemap(projection='merc',llcrnrlat=-10,urcrnrlat=70,llcrnrlon=245,urcrnrlon=370,lat_ts=10,resolution='c')
m.drawcoastlines()
CS = m.contourf(cx,cy,cluster2SLPs.T,clevels,vmin=-5,vmax=5,cmap=cm.RdBu_r,shading='gouraud')

ax3 = plt.subplot2grid((2,3),(0,2),rowspan=1,colspan=1)
m = Basemap(projection='merc',llcrnrlat=-10,urcrnrlat=70,llcrnrlon=245,urcrnrlon=370,lat_ts=10,resolution='c')
m.drawcoastlines()
CS = m.contourf(cx,cy,cluster3SLPs.T,clevels,vmin=-5,vmax=5,cmap=cm.RdBu_r,shading='gouraud')

ax4 = plt.subplot2grid((2,3),(1,0),rowspan=1,colspan=1)
m = Basemap(projection='merc',llcrnrlat=-10,urcrnrlat=70,llcrnrlon=245,urcrnrlon=370,lat_ts=10,resolution='c')
m.drawcoastlines()
CS = m.contourf(cx,cy,cluster4SLPs.T,clevels,vmin=-5,vmax=5,cmap=cm.RdBu_r,shading='gouraud')

ax5 = plt.subplot2grid((2,3),(1,1),rowspan=1,colspan=1)
m = Basemap(projection='merc',llcrnrlat=-10,urcrnrlat=70,llcrnrlon=245,urcrnrlon=370,lat_ts=10,resolution='c')
m.drawcoastlines()
CS = m.contourf(cx,cy,cluster5SLPs.T,clevels,vmin=-5,vmax=5,cmap=cm.RdBu_r,shading='gouraud')

ax6 = plt.subplot2grid((2,3),(1,2),rowspan=1,colspan=1)
m = Basemap(projection='merc',llcrnrlat=-10,urcrnrlat=70,llcrnrlon=245,urcrnrlon=370,lat_ts=10,resolution='c')
m.drawcoastlines()
CS = m.contourf(cx,cy,cluster6SLPs.T,clevels,vmin=-5,vmax=5,cmap=cm.RdBu_r,shading='gouraud')




cluster1DateVectors = np.concatenate(cluster1Time)
cluster2DateVectors = np.concatenate(cluster2Time)
cluster3DateVectors = np.concatenate(cluster3Time)
cluster4DateVectors = np.concatenate(cluster4Time)
cluster5DateVectors = np.concatenate(cluster5Time)
cluster6DateVectors = np.concatenate(cluster6Time)




import pickle

tcPickle = 'historicalTCs.pickle'
outputTCs = {}
outputTCs['c1times'] = cluster1DateVectors
outputTCs['c2times'] = cluster2DateVectors
outputTCs['c3times'] = cluster3DateVectors
outputTCs['c4times'] = cluster4DateVectors
outputTCs['c5times'] = cluster5DateVectors
outputTCs['c6times'] = cluster6DateVectors
outputTCs['c1minTimes'] = cluster1minTime
outputTCs['c2minTimes'] = cluster2minTime
outputTCs['c3minTimes'] = cluster3minTime
outputTCs['c4minTimes'] = cluster4minTime
outputTCs['c5minTimes'] = cluster5minTime
outputTCs['c6minTimes'] = cluster6minTime

with open(tcPickle,'wb') as f:
    pickle.dump(outputTCs, f)









