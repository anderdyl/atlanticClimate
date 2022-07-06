import xarray as xr
import os
import numpy as np
import datetime
from sklearn.decomposition import PCA
import cftime
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import itertools


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



def KMA_simple(xds_PCA, num_clusters, repres=0.95):
    '''
    KMeans Classification for PCA data

    xds_PCA:
        (n_components, n_components) PCs
        (n_components, n_features) EOFs
        (n_components, ) variance
    num_clusters
    repres

    returns a xarray.Dataset containing KMA data
    '''

    # PCA data
    variance = xds_PCA.variance.values[:]
    EOFs = xds_PCA.EOFs.values[:]
    PCs = xds_PCA.PCs.values[:]

    var_anom_std = xds_PCA.var_anom_std.values[:]
    var_anom_mean = xds_PCA.var_anom_mean.values[:]
    time = xds_PCA.time.values[:]

    # APEV: the cummulative proportion of explained variance by ith PC
    APEV = np.cumsum(variance) / np.sum(variance)*100.0
    nterm = np.where(APEV <= repres*100)[0][-1]

    PCsub = PCs[:, :nterm+1]
    EOFsub = EOFs[:nterm+1, :]

    # KMEANS
    kma = KMeans(n_clusters=num_clusters, n_init=2000).fit(PCsub)

    # groupsize
    _, group_size = np.unique(kma.labels_, return_counts=True)

    # groups
    d_groups = {}
    for k in range(num_clusters):
        d_groups['{0}'.format(k)] = np.where(kma.labels_==k)
    # TODO: STORE GROUPS WITHIN OUTPUT DATASET

    # centroids
    centroids = np.dot(kma.cluster_centers_, EOFsub)

    # km, x and var_centers
    km = np.multiply(
        centroids,
        np.tile(var_anom_std, (num_clusters, 1))
    ) + np.tile(var_anom_mean, (num_clusters, 1))

    # sort kmeans
    kma_order = np.argsort(np.mean(-km, axis=1))

    # reorder clusters: bmus, km, cenEOFs, centroids, group_size
    sorted_bmus = np.zeros((len(kma.labels_),),)*np.nan
    for i in range(num_clusters):
        posc = np.where(kma.labels_ == kma_order[i])
        sorted_bmus[posc] = i
    sorted_km = km[kma_order]
    sorted_cenEOFs = kma.cluster_centers_[kma_order]
    sorted_centroids = centroids[kma_order]
    sorted_group_size = group_size[kma_order]

    return xr.Dataset(
        {
            'bmus': (('n_pcacomp'), sorted_bmus.astype(int)),
            'cenEOFs': (('n_clusters', 'n_features'), sorted_cenEOFs),
            'centroids': (('n_clusters','n_pcafeat'), sorted_centroids),
            'Km': (('n_clusters','n_pcafeat'), sorted_km),
            'group_size': (('n_clusters'), sorted_group_size),

            # PCA data
            'PCs': (('n_pcacomp','n_features'), PCsub),
            'variance': (('n_pcacomp',), variance),
            'time': (('n_pcacomp',), time),
        }
    )


def axplot_AWT_2D(ax, var_2D, num_wts, id_wt, color_wt):
    'axes plot AWT variable (2D)'

    # plot 2D AWT
    ax.pcolormesh(
        var_2D,
        cmap='RdBu_r', shading='gouraud',
        vmin=-1.5, vmax=+1.5,
    )

    # title and axis labels/ticks
    ax.set_title(
        'WT #{0} --- {1} years'.format(id_wt, num_wts),
        {'fontsize': 14, 'fontweight':'bold'}
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('month', {'fontsize':8})
    ax.set_xlabel('lon', {'fontsize':8})

    # set WT color on axis frame
    plt.setp(ax.spines.values(), color=color_wt, linewidth=4)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=color_wt)


def colors_awt():

    # 6 AWT colors
    l_colors_dwt = [
        (155/255.0, 0, 0),
        (1, 0, 0),
        (255/255.0, 216/255.0, 181/255.0),
        (164/255.0, 226/255.0, 231/255.0),
        (0/255.0, 190/255.0, 255/255.0),
        (51/255.0, 0/255.0, 207/255.0),
    ]

    return np.array(l_colors_dwt)

def Plot_AWTs(bmus, Km, n_clusters, lon, show=True):
    '''
    Plot Annual Weather Types

    bmus, Km, n_clusters, lon - from KMA_simple()
    '''

    # get number of rows and cols for gridplot
    #n_cols, n_rows = GetBestRowsCols(n_clusters)
    n_rows = 2
    n_cols = 3
    # get cluster colors
    cs_awt = colors_awt()

    # plot figure
    fig = plt.figure()#figsize=(_faspect*_fsize, _fsize))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.10, hspace=0.15)
    gr, gc = 0, 0

    for ic in range(n_clusters):

        id_AWT = ic + 1           # cluster ID
        index = np.where(bmus==ic)[0][:]
        var_AWT = Km[ic,:]
        var_AWT_2D = var_AWT.reshape(-1, len(lon))
        num_WTs = len(index)
        clr = cs_awt[ic]          # cluster color

        # AWT var 2D
        ax = plt.subplot(gs[gr, gc])
        axplot_AWT_2D(ax, var_AWT_2D, num_WTs, id_AWT, clr)

        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    # show and return figure
    if show: plt.show()
    return fig



def axplot_AWT_years(ax, dates_wt, bmus_wt, color_wt, xticks_clean=False,
                     ylab=None, xlims=None):
    'axes plot AWT dates'

    # date axis locator
    yloc5 = mdates.YearLocator(5)
    yloc1 = mdates.YearLocator(1)
    yfmt = mdates.DateFormatter('%Y')

    # get years string
    ys_str = np.array([str(d).split('-')[0] for d in dates_wt])

    # use a text bottom - top cycler
    text_cycler_va = itertools.cycle(['bottom', 'top'])
    text_cycler_ha = itertools.cycle(['left', 'right'])

    # plot AWT dates and bmus
    ax.plot(
        dates_wt, bmus_wt,
        marker='+',markersize=9, linestyle='', color=color_wt,
    )
    va = 'bottom'
    for tx,ty,tt in zip(dates_wt, bmus_wt, ys_str):
        ax.text(
            tx, ty, tt,
            {'fontsize':8},
            verticalalignment = next(text_cycler_va),
            horizontalalignment = next(text_cycler_ha),
            rotation=45,
        )

    # configure axis
    ax.set_yticks([])
    ax.xaxis.set_major_locator(yloc5)
    ax.xaxis.set_minor_locator(yloc1)
    ax.xaxis.set_major_formatter(yfmt)
    #ax.grid(True, which='both', axis='x', linestyle='--', color='grey')
    ax.tick_params(axis='x', which='major', labelsize=8)

    # optional parameters
    if xticks_clean:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Year', {'fontsize':8})

    if ylab: ax.set_ylabel(ylab)

    if xlims is not None:
        ax.set_xlim(xlims[0], xlims[1])


def Plot_AWTs_Dates(bmus, dates, n_clusters, show=True):
    '''
    Plot Annual Weather Types dates

    bmus, dates, n_clusters - from KMA_simple()
    '''

    # get cluster colors
    cs_awt = colors_awt()

    # plot figure
    fig, axs = plt.subplots(nrows=n_clusters)#, figsize=(_faspect*_fsize, _fsize))

    # each cluster has a figure
    for ic in range(n_clusters):

        id_AWT = ic + 1           # cluster ID
        index = np.where(bmus==ic)[0][:]
        dates_AWT = dates[index]  # cluster dates
        bmus_AWT = bmus[index]    # cluster bmus
        clr = cs_awt[ic]          # cluster color

        ylabel = "WT #{0}".format(id_AWT)
        #xlims = [dates[0].astype('datetime64[Y]')-np.timedelta64(3, 'Y'), dates[-1].astype('datetime64[Y]')+np.timedelta64(3, 'Y')]
        xlims = [datetime.datetime(1877,1,1),datetime.datetime(2024,1,1)]

        xaxis_clean=True
        if ic == n_clusters-1:
            xaxis_clean=False

        # axs plot
        axplot_AWT_years(
            axs[ic], dates_AWT, bmus_AWT,
            clr, xaxis_clean, ylabel, xlims
        )
        #axs[ic].set_xticks(dates_AWT)
    # show and return figure
    if show: plt.show()
    return fig


data_folder="/media/dylananderson/Elements1/shusin6_contents/AWT/ERSSTV5/"


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


#time = SST.time.values.date2num()


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



subset = xds_predictor.sel(longitude=slice(120,280),latitude=slice(-5,5))


predictor = PCA_LatitudeAverage(subset, "SST", 1880, 2020, 6, 5)


clusters = KMA_simple(predictor, 6, repres=0.95)

Plot_AWTs(clusters.bmus.values, clusters.Km.values, 6, subset.longitude.values, show=True)
Plot_AWTs_Dates(clusters.bmus.values, np.asarray(sstTime), 6, show=True)



import pickle

dwtPickle = 'AWT1880to2020.pickle'
outputAWT = {}
outputAWT['clusters'] = clusters
outputAWT['predictor'] = predictor
outputAWT['sstSubset'] = subset
outputAWT['xds_predictor'] = xds_predictor

with open(dwtPickle,'wb') as f:
    pickle.dump(outputAWT, f)



