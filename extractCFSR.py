from netCDF4 import Dataset
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import h5py


lonLeft = 250
lonRight = 20
latBot = -10
latTop = 60
year = 2019
month = 1
avgTime = 2
resolution = 0.5
year2 = 2019
month2 = 12
estelaMat = '/media/dylananderson/Elements1/ESTELA/out/NagsHead2/NagsHead2_obj.mat'
mem = True

dt = datetime.datetime(year, month, 1)
if month2 == 12:
    end = datetime.datetime(year2+1, 1, 1)
else:
    end = datetime.datetime(year2, month2+1, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(months=1)
extractTime = []
while dt < end:
    extractTime.append(dt)
    dt += step

# Let's get all of the relevant lat/lon indices
file = '/media/dylananderson/Elements1/CFS/prmsl/prmsl.cdas1.201104.grb2.nc'
data = Dataset(file)
lat = data.variables['lat'][:]
lon = data.variables['lon'][:]

pos_lon1 = np.where((lon == lonLeft - 2))
pos_lon2 = np.where((lon == lonRight + 2))
pos_lat2 = np.where((lat == latTop + 2))
pos_lat1 = np.where((lat == latBot - 2))

latitud = lat[(pos_lat2[0][0]):(pos_lat1[0][0]+1)]
if lonLeft > lonRight:
    longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0]+1)]))
else:
    longitud = lon[pos_lon1[0][0]:(pos_lon2[0][0]+1)]
[x, y] = np.meshgrid(longitud, latitud)

counter = 0
# Now need to loop through the number of files we're extract data from
for tt in extractTime:

    for mm in range(2):
        if mm == 0:
            monthExtract = tt.month-1
            if monthExtract == 0:
                monthExtract = 12
                yearExtract = tt.year-1
            else:
                yearExtract = tt.year
        else:
            yearExtract = tt.year
            monthExtract = tt.month


        if (yearExtract == 2011 and monthExtract >= 4) or yearExtract > 2012:
            file = '/media/dylananderson/Elements1/CFS/prmsl/prmsl.cdas1.{}{:02d}.grb2.nc'.format(yearExtract,monthExtract)
        else:
            file = '/media/dylananderson/Elements1/CFS/prmsl/prmsl.gdas.{}{:02d}.grb2.nc'.format(yearExtract,monthExtract)

        data = Dataset(file)
        time = data.variables['valid_date_time']
        print('{}-{}'.format(yearExtract,monthExtract))

        # extract times and turn them into datetimes
        #years = np.empty((len(time),))
        year = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[0:4]) for i in time]
        month = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[4:6]) for i in time]
        day = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[6:8]) for i in time]
        hour = [int(''.join(list(map(lambda x: x.decode('utf-8'), i))).strip()[8:]) for i in time]
        d_vec = np.vstack((year,month,day,hour)).T
        dates = [datetime.datetime(d[0], d[1], d[2], d[3], 0, 0) for d in d_vec]

        # Extracting SLP fields, need to account for wrap around international date line
        if lonLeft > lonRight:
            #longitud = np.hstack((lon[pos_lon1[0][0]:], lon[0:(pos_lon2[0][0] + 1)]))
            slp1 = data.variables['PRMSL_L101'][:,(pos_lat2[0][0]):(pos_lat1[0][0]+1),(pos_lon1[0][0]):]
            slp2 = data.variables['PRMSL_L101'][:,(pos_lat2[0][0]):(pos_lat1[0][0]+1),0:(pos_lon2[0][0]+1)]
            slp = np.concatenate((slp1,slp2),axis=2)
        else:
            slp = data.variables['PRMSL_L101'][:,(pos_lat2[0][0]):(pos_lat1[0][0]+1),(pos_lon1[0][0]):(pos_lon2[0][0]+1)]



        # are we averaging to a shorter time window?
        m,n,p = np.shape(slp)
        slp_ = np.zeros((n*p,m))
        for mmm in range(m):
            slp_[:,mmm] = slp[mmm,:,:].flatten()
        # slp_ = slp.reshape(n*m,p)

        if avgTime == 0:
            print('returning hourly values')
            slpAvg = slp_
            datesAvg = dates
        else:
            numWindows = int(len(time)/avgTime)
            c = 0
            datesAvg = list()
            slpAvg = np.empty((n*p,numWindows))

            for t in range(numWindows):
                slpAvg[:,t] = np.nanmean(slp_[:,c:c+avgTime],axis=1)
                datesAvg.append(dates[c])
                c = c+avgTime


        # are we reducing the resolution of the grid?
        if resolution == 0.5:
            print('keeping full 0.5 degree resolution')
            slpDownscaled = slpAvg
            xDownscaled = x.flatten()
            yDownscaled = y.flatten()
            x2 = x
            y2 = y
        else:
            xFlat = x.flatten()
            yFlat = y.flatten()
            xRem = np.fmod(xFlat,resolution)
            yRem = np.fmod(yFlat,resolution)
            cc = 0
            for pp in range(len(xFlat)):
                if xRem[pp] == 0:
                    if yRem[pp] == 0:
                        if cc == 0:
                            ind2deg = int(pp)
                            cc = cc + 1
                        else:
                            ind2deg = np.hstack((ind2deg,int(pp)))
                            cc = cc + 1
            slpDownscaled = slpAvg[ind2deg,:]
            xDownscaled = xFlat[ind2deg]
            yDownscaled = yFlat[ind2deg]

            reshapeIndX = np.where((np.diff(xDownscaled[0:100]) > 2))
            reshapeIndY = np.where((np.diff(yDownscaled) < 0))

            x2 = xDownscaled.reshape(int(len(reshapeIndY[0])+1),int(reshapeIndX[0]+1))
            y2 = yDownscaled.reshape(int(len(reshapeIndY[0])+1),int(reshapeIndX[0]+1))

        My, Mx = np.shape(y2)
        # dim, Ntime = np.shape(slpDownscaled)
        # GrdSlp = np.zeros((np.shape(slpDownscaled)))
        # for ttt in range(Ntime):
        #     p = slpDownscaled[:,ttt].reshape(My,Mx)
        #     dp = np.zeros((np.shape(p)))
        #     ii = np.arange(1,Mx-1)
        #     jj = np.arange(1,My-1)
        #     for iii in ii:
        #         for jjj in jj:
        #             phi = np.pi*np.abs(y2[jjj,iii])/180
        #             dpx1 = (p[jjj,iii] - p[jjj,iii-1])/np.cos(phi)
        #             dpx2 = (p[jjj,iii+1] - p[jjj,iii])/np.cos(phi)
        #             dpy1 = p[jjj,iii] - p[jjj-1,iii]
        #             dpy2 = p[jjj+1,iii] - p[jjj,iii]
        #             dp[jjj,iii] = (dpx1**2 + dpx2**2 )/2+(dpy1**2 + dpy2**2)/2
        #     GrdSlp[:,ttt] = dp.flatten()

        # if mem == True:
        #
        #     with h5py.File(estelaMat, 'r') as f:
        #         # for k in f.keys():
        #         #     print(k)
        #         Xraw = f['full/Xraw'][:]
        #         Y = f['full/Y'][:]
        #         DJF = f['C/traveldays_interp/DJF'][:]
        #         MAM = f['C/traveldays_interp/MAM'][:]
        #         JJA = f['C/traveldays_interp/JJA'][:]
        #         SON = f['C/traveldays_interp/SON'][:]
        #
        #
        #
        #     X_estela = Xraw
        #     tempX = np.where(X_estela < 120)
        #     X_estela[tempX] = X_estela[tempX]+(X_estela[tempX]*0+360)
        #     estelaX = np.hstack((X_estela[:,600:],X_estela[:,0:600]))
        #
        #     Y_estela = Y
        #     estelaY = np.hstack((Y_estela[:,600:],Y_estela[:,0:600]))
        #
        #     #dateTemp = days
        if mm == 0:
            slpMem = slpDownscaled
            datesMem = datesAvg
        # else:
        #     slpMem = np.hstack((slpMem, slpDownscaled))
        #     datesMem = np.append(datesMem, datesAvg)

    if counter == 0:
        SLPS = slpMem
        DATES = datesMem
    else:
        SLPS = np.hstack((SLPS,slpMem))
        DATES = np.append(DATES,datesMem)

    counter = counter + 1


dataPickle = 'tempAugSept2019slps.pickle'
output = {}
output['SLPS'] = SLPS
output['DATES'] = DATES
output['x2'] = x2
output['y2'] = y2
output['Mx'] = Mx
output['My'] = My

import pickle
with open(dataPickle,'wb') as f:
    pickle.dump(output, f)


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap

i = 0
fig = plt.figure(figsize=(10, 6))

ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
# clevels = np.arange(-40, 40, 1)
clevels = np.arange(980, 1040, 1)

spatialField = SLPS[:,i]/100 #SLPS[:, i] / 100  # - np.nanmean(SLP, axis=1) / 100
rectField = spatialField.reshape(My, Mx)
m = Basemap(projection='merc', llcrnrlat=-5, urcrnrlat=55, llcrnrlon=255, urcrnrlon=360, lat_ts=10,
            resolution='c')
m.fillcontinents(color=[0.5, 0.5, 0.5])
cx, cy = m(x2, y2)

m.drawcoastlines()
# m.bluemarble()
# CS = m.contourf(cx, cy, rectField.T, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r, shading='gouraud')
CS = m.contourf(cx.T, cy.T, rectField.T, clevels, vmin=980, vmax=1040, cmap=cm.RdBu_r, shading='gouraud')

tx, ty = m(320, -0)
parallels = np.arange(0, 360, 10)
m.drawparallels(parallels, labels=[True, True, True, False], textcolor='white')
# ax.text(tx, ty, '{}'.format((group_size[num])))
meridians = np.arange(0, 360, 20)
m.drawmeridians(meridians, labels=[True, True, True, True], textcolor='white')

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
cbar = fig.colorbar(CS, cax=cbar_ax)
cbar.set_label('SLP (mbar)')