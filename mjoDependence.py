from dateutil.relativedelta import relativedelta
import scipy.io as sio
from scipy.io.matlab.mio5_params import mat_struct
import datetime


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




dataMJO = ReadMatfile('/media/dylananderson/Elements/SERDP/Data/MJO/mjo_australia_2021.mat')
mjoPhase = dataMJO['phase']
dt = datetime.date(1974, 6, 1)
end = datetime.date(2021, 6, 18)
step = relativedelta(days=1)
mjoTime = []
while dt < end:
    mjoTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step







