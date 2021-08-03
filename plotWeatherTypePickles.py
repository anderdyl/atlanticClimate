
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap
import pickle



with open(r"dwtsAll6TCTracksALLDATA.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

SlpGrdMeanET = historicalDWTs['SlpGrdMean']
SlpGrdStdET = historicalDWTs['SlpGrdStd']
sorted_centroidsET = historicalDWTs['sorted_centroids']
X_inET = historicalDWTs['X_in']
Y_inET = historicalDWTs['Y_in']
kma_orderET = historicalDWTs['kma_order']
SLPET = historicalDWTs['SLP']
group_sizeET = historicalDWTs['group_size']


with open(r"dwtsOfExtraTropicalDays.pickle", "rb") as input_file:
   historicalTWTs = pickle.load(input_file)
SlpGrdMeanTC = historicalTWTs['SlpGrdMean']
SlpGrdStdTC = historicalTWTs['SlpGrdStd']
sorted_centroidsTC = historicalTWTs['sorted_centroids']
X_inTC = historicalTWTs['X_in']
Y_inTC = historicalTWTs['Y_in']
kma_orderTC = historicalTWTs['kma_order']
SLPTC = historicalTWTs['SLP']
group_sizeTC = historicalTWTs['group_size']


repmatDesviacionET = np.tile(SlpGrdStdET, (36,1))
repmatMediaET = np.tile(SlpGrdMeanET, (36,1))
Km_ET = np.multiply(sorted_centroidsET,repmatDesviacionET) + repmatMediaET
[mK, nK] = np.shape(Km_ET)
Km_slpET = Km_ET[:,0:int(nK/2)]
Km_grdET = Km_ET[:,int(nK/2):]
X_BET = X_inET
Y_BET = Y_inET
#SLP_C = SLP
Km_slpET = Km_slpET[:,0:len(X_BET)]
Km_grdET = Km_grdET[:,0:len(X_BET)]





repmatDesviacionTC = np.tile(SlpGrdStdTC, (12,1))
repmatMediaTC = np.tile(SlpGrdMeanTC, (12,1))
Km_slpTC = np.multiply(sorted_centroidsTC,repmatDesviacionTC) + repmatMediaTC
XsTC = np.arange(np.min(X_inTC),np.max(X_inTC),2)
YsTC = np.arange(np.min(Y_inTC),np.max(Y_inTC),2)
lenXBTC = len(X_inTC)
[XRTC,YRTC] = np.meshgrid(XsTC,YsTC)








etcolors = cm.viridis(np.linspace(0, 1, 48-11))
tccolors = np.flipud(cm.autumn(np.linspace(0,1,12)))
dwtcolors = np.vstack((etcolors,tccolors[1:,:]))





# plotting the EOF patterns
fig2 = plt.figure(figsize=(10,10))
gs1 = gridspec.GridSpec(8, 6)
gs1.update(wspace=0.00, hspace=0.00) # set the spacing between axes.
c1 = 0
c2 = 0
counter = 0
plotIndx = 0
plotIndy = 0
for hh in range(48):
    #p1 = plt.subplot2grid((6,6),(c1,c2))
    ax = plt.subplot(gs1[hh])

    if hh <= 35:
        num = kma_orderET[hh]

        #spatialField = Km_slpET[(num - 1), :] / 100 - np.nanmean(SLPET, axis=0) / 100
        spatialField = Km_slpET[(hh), :] / 100 - np.nanmean(SLPET, axis=0) / 100

        #spatialField = np.multiply(EOFs[hh,0:len(X_in)],np.sqrt(variance[hh]))
        Xs = np.arange(np.min(X_inET),np.max(X_inET),2)
        Ys = np.arange(np.min(Y_inET),np.max(Y_inET),2)
        lenXB = len(X_inET)
        [XR,YR] = np.meshgrid(Xs,Ys)
        sea_nodes = []
        for qq in range(lenXB-1):
            sea_nodes.append(np.where((XR == X_inET[qq]) & (YR == Y_inET[qq])))

        rectField = np.ones((np.shape(XR))) * np.nan
        for tt in range(len(sea_nodes)):
            rectField[sea_nodes[tt]] = spatialField[tt]

        clevels = np.arange(-27,27,1)
        m = Basemap(projection='merc',llcrnrlat=-30,urcrnrlat=48,llcrnrlon=260,urcrnrlon=370,lat_ts=10,resolution='c')

        # m = Basemap(projection='merc',llcrnrlat=-40,urcrnrlat=55,llcrnrlon=255,urcrnrlon=375,lat_ts=10,resolution='c')
        m.fillcontinents(color=dwtcolors[hh])
        cx,cy =m(XR,YR)
        m.drawcoastlines()
        CS = m.contourf(cx,cy,rectField,clevels,vmin=-12,vmax=12,cmap=cm.RdBu_r,shading='gouraud')
        #p1.set_title('EOF {} = {}%'.format(hh+1,np.round(nPercent[hh]*10000)/100))
        tx,ty = m(323,-27)
        ax.text(tx,ty,'{}'.format(group_sizeET[num]))

    else:
        clevels = np.arange(-27, 27, 1)
        num = kma_orderTC[hh-36]
        #spatialField = Km_slpTC[(num - 1), :] / 100 - np.nanmean(SLPTC, axis=0) / 100
        spatialField = Km_slpTC[(hh-36), :] / 100 - np.nanmean(SLPTC, axis=0) / 100

        rectField = spatialField.reshape(63, 32)

        # cluster2SLPs = np.nanmean(SLP[cluster2SLPIndex, :], axis=0).reshape(73, 43) / 100 - np.nanmean(SLP, axis=0).reshape(73, 43) / 100

        # rectField = np.ones((np.shape(X_in))) * np.nan
        # # temp = np.nanmean(SLP[cluster6SLPIndex,:],axis=0)/100 - np.nanmean(SLP, axis=0) / 100
        # #temp = spatialField.flatten()
        # for tt in range(len(sea_nodes)):
        #     rectField[sea_nodes[tt]] = spatialField[tt]
        m = Basemap(projection='merc', llcrnrlat=-5, urcrnrlat=55, llcrnrlon=255, urcrnrlon=360, lat_ts=10,
                    resolution='c')
        m.fillcontinents(color=dwtcolors[hh])
        cx, cy = m(X_inTC, Y_inTC)
        m.drawcoastlines()#color=dwtcolors[hh])
        CS = m.contourf(cx, cy, rectField.T, clevels, vmin=-12, vmax=12, cmap=cm.RdBu_r, shading='gouraud')
        tx, ty = m(320, -0)
        ax.text(tx, ty, '{}'.format((group_sizeET[num])))

    c2 += 1
    if c2 == 6:
        c1 += 1
        c2 = 0

    if plotIndx < 8:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    if plotIndy > 0:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
    counter = counter + 1
    if plotIndy < 8:
        plotIndy = plotIndy + 1
    else:
        plotIndy = 0
        plotIndx = plotIndx + 1
