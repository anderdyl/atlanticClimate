import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
from mpl_toolkits.basemap import Basemap
import os
import pickle

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

# plt.rcParams.update({
#     "lines.color": "white",
#     "patch.edgecolor": "white",
#     "text.color": "white",
#     "axes.facecolor": "black",
#     "axes.edgecolor": "lightgray",
#     "axes.labelcolor": "white",
#     "xtick.color": "white",
#     "ytick.color": "white",
#     "grid.color": "lightgray",
#     "figure.facecolor": "black",
#     "figure.edgecolor": "black",
#     "savefig.facecolor": "black",
#     "savefig.edgecolor": "black"})

plt.style.use('dark_background')

pickleFile = True


if pickleFile == True:

    pickleName = 'tempAugSept2019slps.pickle'
    dbfile = open(pickleName, 'rb')
    SLPs = pickle.load(dbfile)
    dbfile.close()
    #SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/NorthAtlanticSLPs_June2021_smaller.mat')
    # SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/Nags_Head_SLPs_2degree_memory_June2021.mat')

    X_in = SLPs['x2']
    Y_in = SLPs['y2']
    SLP = SLPs['SLPS']
    SLPtime = SLPs['DATES']
    Mx = SLPs['Mx']
    My = SLPs['My']


    temp = np.where((X_in < 25))
    X_in[temp] = X_in[temp] + X_in[temp]*0+360
    subset = np.arange(3100, 3500, 1)
    for i in range(len(subset)):
        plt.ioff()
        fig = plt.figure(figsize=(10, 6))

        ax = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
        clevels = np.arange(-45, 45, 1)
        # clevels = np.arange(990, 1035, 1)

        spatialField = SLP[:, subset[i]] / 100 - np.nanmean(SLP, axis=1) / 100
        rectField = spatialField.reshape(My, Mx)
        m = Basemap(projection='merc', llcrnrlat=-5, urcrnrlat=55, llcrnrlon=255, urcrnrlon=360, lat_ts=10,
                    resolution='c')
        m.fillcontinents(color=[0.5, 0.5, 0.5])
        cx, cy = m(X_in, Y_in)

        m.drawcoastlines()
        # m.bluemarble()
        CS = m.contourf(cx, cy, rectField, clevels, vmin=-20, vmax=20, cmap=cm.RdBu_r, shading='gouraud')
        # CS = m.contourf(cx.T, cy.T, rectField.T, clevels, vmin=990, vmax=1035, cmap=cm.RdBu_r, shading='gouraud')

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
        #    cb = plt.colorbar(CS)
        # cbar.set_clim(-20.0, 20.0)

        if i < 10:
            plt.savefig('/home/dylananderson/projects/slpImages/frame00' + str(i) + '.png')
        elif i > 9 and i < 100:
            plt.savefig('/home/dylananderson/projects/slpImages/frame0' + str(i) + '.png')
        else:
            plt.savefig('/home/dylananderson/projects/slpImages/frame' + str(i) + '.png')
        plt.close()
else:

    SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/NorthAtlanticSLPs_June2021_smaller.mat')
    # SLPs = ReadMatfile('/media/dylananderson/Elements/NC_climate/Nags_Head_SLPs_2degree_memory_June2021.mat')

    X_in = SLPs['X_in']
    Y_in = SLPs['Y_in']
    SLP = SLPs['slp_mem']
    SLPtime = SLPs['time']

    subset = np.arange(14480,14540,1)
    for i in range(len(subset)):
        fig = plt.figure(figsize=(10,6))

        ax = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
        clevels = np.arange(-40, 40, 1)
        spatialField = SLP[(i), :] / 100 - np.nanmean(SLP, axis=0) / 100
        rectField = spatialField.reshape(63, 32)
        m = Basemap(projection='merc', llcrnrlat=-5, urcrnrlat=55, llcrnrlon=255, urcrnrlon=360, lat_ts=10, resolution='c')
        m.fillcontinents(color=[0.5,0.5,0.5])
        cx, cy = m(X_in, Y_in)
        m.drawcoastlines()
        # m.bluemarble()
        CS = m.contourf(cx, cy, rectField.T, clevels, vmin=990, vmax=135, cmap=cm.RdBu_r, shading='gouraud')
        tx, ty = m(320, -0)
        parallels = np.arange(0,360,10)
        m.drawparallels(parallels,labels=[True,True,True,False],textcolor='white')
        # ax.text(tx, ty, '{}'.format((group_size[num])))
        meridians = np.arange(0,360,20)
        m.drawmeridians(meridians,labels=[True,True,True,True],textcolor='white')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.88, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(CS, cax=cbar_ax)
        cbar.set_label('SLP (mbar)')
    #    cb = plt.colorbar(CS)
        cbar.set_clim(-20.0, 20.0)


        if i < 10:
            plt.savefig('/home/dylananderson/projects/slpImages/frame00' + str(i) + '.png')
        elif i > 9 and i < 100:
            plt.savefig('/home/dylananderson/projects/slpImages/frame0' + str(i) + '.png')
        else:
            plt.savefig('/home/dylananderson/projects/slpImages/frame' + str(i) + '.png')
        plt.close()



geomorphdir = '/home/dylananderson/projects/slpImages/'

files = os.listdir(geomorphdir)

files.sort()

files_path = [os.path.join(geomorphdir,x) for x in os.listdir(geomorphdir)]

files_path.sort()

import cv2

frame = cv2.imread(files_path[0])
height, width, layers = frame.shape
forcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('slpTestMovie.avi', forcc, 16, (width, height))
for image in files_path:
    video.write(cv2.imread(image))
cv2.destroyAllWindows()
video.release()

