import xarray as xr
import os
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.decomposition import PCA
import cftime
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import itertools
from mpl_toolkits.basemap import Basemap
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression

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


def RunnningMean_Monthly(xds, var_name, window=5):
    '''
    Calculate running average grouped by months

    xds:
        (longitude, latitude, time) variables: var_name

    returns xds with new variable "var_name_runavg"
    '''

    tempdata_runavg = np.empty(xds[var_name].shape)

    for lon in xds.longitude.values:
       for lat in xds.latitude.values:
          for mn in range(1, 13):

             # indexes
             ix_lon = np.where(xds.longitude == lon)
             ix_lat = np.where(xds.latitude == lat)
             ix_mnt = np.where(xds['time.month'] == mn)

             # point running average
             time_mnt = xds.time[ix_mnt]
             data_pnt = xds[var_name].loc[lon, lat, time_mnt]

             tempdata_runavg[ix_lon[0], ix_lat[0], ix_mnt[0]] = running_mean(
                 data_pnt.values, window)

    # store running average
    xds['{0}_runavg'.format(var_name)]= (
        ('longitude', 'latitude', 'time'),
        tempdata_runavg)

    return xds


def PCA_LatitudeAverage(xds, var_name, y1, y2, m1, m2):
    '''
    Principal component analysis
    method: remove monthly running mean and latitude average

    xds:
        (longitude, latitude, time), pred_name | pred_name_runavg

    returns a xarray.Dataset containing PCA data: PCs, EOFs, variance
    '''

    # calculate monthly running mean
    xds = RunnningMean_Monthly(xds, var_name)

    # predictor variable and variable_runnavg from dataset
    var_val = xds[var_name]
    var_val_ra = xds['{0}_runavg'.format(var_name)]

    # use datetime for indexing
    dt1 = datetime.datetime(y1, m1, 1)
    dt2 = datetime.datetime(y2+1, m2, 28)
    time_PCA = [datetime.datetime(y, m1, 1) for y in range(y1, y2+1)]

    # use data inside timeframe
    data_ss = var_val.loc[:,:,dt1:dt2]
    data_ss_ra = var_val_ra.loc[:,:,dt1:dt2]

    # anomalies: remove the monthly running mean
    data_anom = data_ss - data_ss_ra

    # average across all latitudes
    data_avg_lat = data_anom.mean(dim='latitude')

    # collapse 12 months of data to a single vector
    nlon = data_avg_lat.longitude.shape[0]
    ntime = data_avg_lat.time.shape[0]
    hovmoller = xr.DataArray(
        np.reshape(data_avg_lat.values, (12*nlon, ntime//12), order='F')
    )
    hovmoller = hovmoller.transpose()

    # mean and standard deviation
    var_anom_mean = hovmoller.mean(axis=0)
    var_anom_std = hovmoller.std(axis=0)

    # remove means and normalize by the standard deviation at anomaly
    # rows = time, columns = longitude
    nk_m = np.kron(np.ones((y2-y1+1,1)), var_anom_mean)
    nk_s = np.kron(np.ones((y2-y1+1,1)), var_anom_std)
    var_anom_demean = (hovmoller - nk_m) / nk_s

    # sklearn principal components analysis
    ipca = PCA(n_components=var_anom_demean.shape[0])
    PCs = ipca.fit_transform(var_anom_demean)

    pred_lon = xds.longitude.values[:]

    return xr.Dataset(
        {
            'PCs': (('n_components', 'n_components'), PCs),
            'EOFs': (('n_components','n_features'), ipca.components_),
            'variance': (('n_components',), ipca.explained_variance_),

            'var_anom_std': (('n_features',), var_anom_std),
            'var_anom_mean': (('n_features',), var_anom_mean),

            'time': (('n_components'), time_PCA),
            'pred_lon': (('n_lon',), pred_lon),
        },

        # store PCA algorithm metadata
        attrs = {
            'method': 'anomalies, latitude averaged',
        }
    )






data_folder="/media/dylananderson/Elements/shusin6_contents/AWT/ERSSTV5/"


years = np.arange(1880,2022)
months = np.arange(1,13)
ogTime = []
for ii in years:
    for hh in months:
        if hh < 10:
            date = str(ii) + "0" + str(hh)
        else:
            date = str(ii) + str(hh)

        file = "ersst.v5." + date + ".nc"
        print(file)
        if ii == 1880 and hh < 6:
            print("skipping {}/{}".format(ii,hh))
        else:
            if ii == 1880 and hh == 6:
                with xr.open_dataset(os.path.join(data_folder, file)) as ds:
                    temp = ds
                    SST = ds['sst']
                    ogTime.append(datetime.datetime(ii,hh,1))
            elif ii == 2021 and hh > 5:
                print("skipping {}/{}".format(ii,hh))
            else:
                with xr.open_dataset(os.path.join(data_folder,file)) as ds:
                    SST = xr.concat([SST,ds['sst']],dim="time")
                    ogTime.append(datetime.datetime(ii,hh,1))




dt = datetime.datetime(1880, 6, 1)
end = datetime.datetime(2021, 6, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(years=1)
sstTime = []
while dt < end:
    sstTime.append(dt)
    dt += step

data = SST.squeeze("lev")

# parse data to xr.Dataset
xds_predictor = xr.Dataset(
    {
        'SST': (('longitude','latitude','time'), data.T),
    },
    coords = {
        'longitude': SST.lon.values,
        'latitude': SST.lat.values,
        'time': ogTime,
    }
)


var_name = "SST"
y1 = 1880
y2 = 2021
m1 = 6
m2 = 5
subset = xds_predictor.sel(longitude=slice(280,350),latitude=slice(0,65))
#
# plt.figure()
# p1 = plt.subplot2grid((1, 1), (0, 0))
# spatialField = subset["SST"][:,:,0]
# Xs = subset.longitude.values
# Ys = subset.latitude.values
# [XR, YR] = np.meshgrid(Xs, Ys)
# m = Basemap(projection='merc', llcrnrlat=-40, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10, resolution='c')
# m.drawcoastlines()
# cx, cy = m(XR, YR)
# CS = m.contour(cx, cy, spatialField.values.T, np.arange(0,30,1), vmin=20, vmax=30, cmap=cm.RdBu_r, shading='gouraud')
#
d1 = datetime.datetime(1880, 6, 1)
dt = datetime.datetime(1880, 6, 1)
end = datetime.datetime(2021, 6, 1)
# step = datetime.timedelta(months=1)
step = relativedelta(months=1)
sstTime = []
while dt < end:
    sstTime.append(dt)
    dt += step


timeDelta = np.array([(d - d1).days/365.25 for d in sstTime])

tempdata_runavg = np.nan*np.ones(subset["SST"].shape)

for lon in subset.longitude.values:
    for lat in subset.latitude.values:
        # indexes
        ix_lon = np.where(subset.longitude == lon)
        ix_lat = np.where(subset.latitude == lat)
        data_pnt = subset["SST"].loc[lon, lat, :]
        if ~np.any(np.isnan(data_pnt.values)):
            model = LinearRegression()
            X = np.reshape(timeDelta, (len(timeDelta), 1))
            model.fit(X, data_pnt.values)
            trend = model.predict(X)
            detrended = [data_pnt.values[i] - trend[i] for i in range(0,len(data_pnt.values))]
            tempdata_runavg[ix_lon,ix_lat,:] = detrended
    # for mn in range(1, 13):

# test = detrendAllTime(subset, "SST")

d1 = datetime.datetime(1880, 6, 1)
dt = datetime.datetime(1880, 6, 1)
end = datetime.datetime(2021, 6, 1)
# step = datetime.timedelta(months=1)
step = relativedelta(years=1)
annualTime = []
while dt < end:
    annualTime.append(dt)
    dt += step


#
# xds = RunnningMean_Monthly(subset, var_name)
#
# # predictor variable and variable_runnavg from dataset
# var_val = xds[var_name]
# var_val_ra = xds['{0}_runavg'.format(var_name)]

# use datetime for indexing
dt1 = datetime.datetime(y1, m1, 1)
dt2 = datetime.datetime(y2 + 1, m2, 28)
time_PCA = [datetime.datetime(y, m1, 1) for y in range(y1, y2 + 1)]

# # use data inside timeframe
# data_ss = var_val.loc[:, :, dt1:dt2]
# data_ss_ra = var_val_ra.loc[:, :, dt1:dt2]
#
# # anomalies: remove the monthly running mean
# data_anom = data_ss - data_ss_ra

plt.figure()
p1 = plt.subplot2grid((1, 1), (0, 0))
spatialField = tempdata_runavg[:,:,-1]#np.reshape(var_anom_mean.values,(33,36))
Xs = subset.longitude.values
Ys = subset.latitude.values
[XR, YR] = np.meshgrid(Xs, Ys)
m = Basemap(projection='merc', llcrnrlat=-40, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10, resolution='c')
m.drawcoastlines()
cx, cy = m(XR, YR)
CS = m.contour(cx, cy, spatialField.T),# np.arange(0,0.023,.003), cmap=cm.RdBu_r, shading='gouraud')



nlon,nlat,ntime = np.shape(tempdata_runavg)

collapsed = np.reshape(tempdata_runavg,(nlon*nlat, ntime))

annual = np.nan*np.ones((int(nlon*nlat),int(ntime/12)))
c = 0
for hh in range(int(ntime/12)):
    annual[:,hh] = np.nanmean(collapsed[:,c:c+12],axis=1)
    c = c + 12


# # average across all latitudes
# data_avg_lat = data_anom.mean(dim='latitude')


# # collapse 12 months of data to a single vector
# nlon = data_anom.longitude.shape[0]
# nlat = data_anom.latitude.shape[0]
# ntime = data_anom.time.shape[0]
# collapsed = xr.DataArray(
#     np.reshape(data_anom.values, (nlon*nlat, ntime), order='F')
# )
# collapsed = collapsed.transpose()

# mean and standard deviation


index = ~np.isnan(annual[:,0])
badIndex = np.isnan(annual[:,0])
ocean = [i for i, x in enumerate(index) if x]
land = [i for i, x in enumerate(badIndex) if x]
realDataAnoms = annual[index,:]

var_anom_mean = np.nanmean(realDataAnoms.T,axis=0)
var_anom_std = np.nanstd(realDataAnoms.T,axis=0)
timeSeries_mean = np.nanmean(realDataAnoms,axis=0)
plt.figure()
plt.plot(annualTime,timeSeries_mean)






# plt.figure()
# p1 = plt.subplot2grid((1, 1), (0, 0))
# spatialField = np.reshape(var_anom_mean.values,(33,36))
# Xs = subset.longitude.values
# Ys = subset.latitude.values
# [XR, YR] = np.meshgrid(Xs, Ys)
# m = Basemap(projection='merc', llcrnrlat=-40, urcrnrlat=55, llcrnrlon=255, urcrnrlon=375, lat_ts=10, resolution='c')
# m.drawcoastlines()
# cx, cy = m(XR, YR)
# CS = m.contour(cx, cy, spatialField.T, np.arange(0,0.023,.003), cmap=cm.RdBu_r, shading='gouraud')


#  remove means and normalize by the standard deviation at anomaly
# rows = time, columns = longitude
# nk_m = np.kron(np.ones((y2 - y1 + 1, 1)), var_anom_mean)
# nk_s = np.kron(np.ones((y2 - y1 + 1, 1)), var_anom_std)
# var_anom_demean = (reaDataAnoms - nk_m) / nk_s
nk_m = np.kron(np.ones(((y2 - y1), 1)), var_anom_mean)
nk_s = np.kron(np.ones(((y2 - y1), 1)), var_anom_std)
var_anom_demean = (realDataAnoms.T - nk_m) / nk_s
# var_anom_demean = var_anom_demean.T
# sklearn principal components analysis
ipca = PCA()#n_components=var_anom_demean.shape[0])
PCs = ipca.fit_transform(var_anom_demean)

plt.figure()
plt.plot(PCs[:,1])

EOFs = ipca.components_
variance = ipca.explained_variance_
nPercent = variance / np.sum(variance)
APEV = np.cumsum(variance) / np.sum(variance) * 100.0
nterm = np.where(APEV <= 0.95 * 100)[0][-1]





n_clusters = 6

kmeans = KMeans(n_clusters, init='k-means++', random_state=100)  # 80

n_components = 3 # !!!!
data = PCs[:, 0:n_components]

#    data1=data/np.std(data,axis=0)

awt_bmus = kmeans.fit_predict(data)

plt.plot(annualTime,awt_bmus)



import pickle

mwtPickle = 'sstWTsPCsAndAllData.pickle'
outputMWTs = {}
outputMWTs['PCs'] = PCs
outputMWTs['EOFs'] = EOFs
outputMWTs['nPercent'] = nPercent
outputMWTs['awt_bmus'] = awt_bmus
outputMWTs['n_components'] = n_components
outputMWTs['variance'] = variance
outputMWTs['ocean'] = ocean
outputMWTs['land'] = land
outputMWTs['realDataAnoms'] = realDataAnoms
outputMWTs['tempdata_runavg'] = tempdata_runavg
outputMWTs['collapsed'] = collapsed
outputMWTs['annual'] = annual
outputMWTs['annualTime'] = annualTime
outputMWTs['subset'] = subset
outputMWTs['data'] = data

with open(mwtPickle,'wb') as f:
    pickle.dump(outputMWTs, f)



#
# pred_lon = xds.longitude.values[:]
#
# xr.Dataset(
#     {
#         'PCs': (('n_components', 'n_components'), PCs),
#         'EOFs': (('n_components', 'n_features'), ipca.components_),
#         'variance': (('n_components',), ipca.explained_variance_),
#
#         'var_anom_std': (('n_features',), var_anom_std),
#         'var_anom_mean': (('n_features',), var_anom_mean),
#
#         'time': (('n_components'), time_PCA),
#         'pred_lon': (('n_lon',), pred_lon),
#     })
