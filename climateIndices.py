import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import matplotlib.pyplot as plt

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


with open('/home/dylananderson/projects/duckGeomorph/atlanticMultiDecadalOscillation.txt', 'r') as fd:
    c = 0
    dataAMO = list()
    for line in fd:
        splitLine = line.split()
        print(splitLine[1:])
        for t in splitLine[1:]:
            dataAMO.append(float(t))

amo = np.asarray(dataAMO)
amo = amo[0:-7]
dt = datetime.date(1856, 1, 1)
end = datetime.date(2021, 6, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(months=1)
amoTime = []
while dt < end:
    amoTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step



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


with open('/home/dylananderson/projects/duckGeomorph/aceNorthAtlanticHurricanes.csv', 'r') as fd:
    c = 0
    dataACE = list()
    for line in fd:
        splitLine = line.split(',')
        secondSplit = splitLine[3].split('/')
        print(secondSplit[0])
        dataACE.append(float(secondSplit[0]))
ace = np.asarray(dataACE)
dt = datetime.date(1851, 1, 1)
end = datetime.date(2020, 5, 1)
#step = datetime.timedelta(months=1)
step = relativedelta(years=1)
aceTime = []
while dt < end:
    aceTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step


dataMJO = ReadMatfile('/media/dylananderson/Elements/SERDP/Data/MJO/mjo_australia_2021.mat')
mjoPhase = dataMJO['phase']
dt = datetime.date(1974, 6, 1)
end = datetime.date(2021, 6, 18)
step = relativedelta(days=1)
mjoTime = []
while dt < end:
    mjoTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step


figClimate = plt.figure()
ax1Cl = plt.subplot2grid((4,1),(0,0),rowspan=1,colspan=1)
ax1Cl.plot(mjoTime,mjoPhase,'.')
ax1Cl.set_xlim([datetime.date(1979,1,1),datetime.date(2021,5,1)])
ax1Cl.set_ylim([0,9])
ax1Cl.set_ylabel('MJO')
ax2Cl = plt.subplot2grid((4,1),(1,0),rowspan=1,colspan=1)
ax2Cl.plot(aceTime,ace)
ax2Cl.set_xlim([datetime.date(1979,1,1),datetime.date(2021,5,1)])
ax2Cl.set_ylabel('ACE')
ax3Cl = plt.subplot2grid((4,1),(2,0),rowspan=1,colspan=1)
ax3Cl.plot(naoTime,nao)
ax3Cl.set_xlim([datetime.date(1979,1,1),datetime.date(2021,5,1)])
ax3Cl.set_ylabel('NAO')
ax4Cl = plt.subplot2grid((4,1),(3,0),rowspan=1,colspan=1)
ax4Cl.plot(amoTime,amo)
ax4Cl.set_xlim([datetime.date(1979,1,1),datetime.date(2021,5,1)])
ax4Cl.set_ylabel('AMO')




