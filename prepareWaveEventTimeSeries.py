
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
#Getting main packages
from scipy.stats import norm
import seaborn as sns; sns.set(style = 'whitegrid')
from scipy.stats import genpareto
import math as mt
import scipy.special as sm
import more_itertools as mit

bump = 57
for hh in range(250):

   print('working on {}'.format(hh))
   # file = r"/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle".format(hh)
   file = r"/media/dylananderson/Elements/Sims/simulation{}.pickle".format(bump+hh)

   with open(file, "rb") as input_file:
      simsInput = pickle.load(input_file)
   simulationData = simsInput['simulationData']
   df = simsInput['df']
   time = simsInput['time']
   year = np.array([tt.year for tt in time])
   df['year'] = year
   month = np.array([tt.month for tt in time])
   df['month'] = month
   time = np.asarray(time)
   hs = simulationData[:,0]
   tp = simulationData[:,1]
   dm = simulationData[:,2]
   wl = simulationData[:,3]

   stormHsInd = np.where((hs > 1.9))
   stormHsList = [list(group) for group in mit.consecutive_groups(stormHsInd[0])]


   keepList = []
   skipList = []
   hsStormList = []
   tpStormList = []
   dmStormList = []
   ntrStormList = []
   timeStormList = []
   hourStormList = []
   indStormList = []

   for xx in range(len(stormHsList)-2):

       i1 = stormHsList[xx][0]
       i2 = stormHsList[xx][-1]
       t1 = time[i1]
       t2 = time[i2]
       nexti1 = stormHsList[xx+1][0]
       diff = nexti1 - i2
       if diff < 24:
           i2 = stormHsList[xx+1][-1]
           t2 = time[i2]
           nexti1 = stormHsList[xx+2][0]
           diff2 = nexti1-i2
           skipList.append(xx+1)
           if diff2 < 24:
               i2 = stormHsList[xx + 2][-1]
               t2 = time[i2]
               skipList.append(xx+2)

       tempWave = np.where((time < t2) & (time > t1))
       if len(tempWave[0]) > 12:
           if i1 < 12:
              t1 = time[0]
           else:
              t1 = time[i1-12]

           t2 = time[i2+12]

           tempWave = np.where((time < t2) & (time > t1))
           indices = np.arange(i1-12,i2+12)
           keepList.append(xx)
           hsStormList.append(hs[tempWave])
           tpStormList.append(tp[tempWave])
           dmStormList.append(dm[tempWave])
           ntrStormList.append(wl[tempWave])
           timeStormList.append(time[tempWave])
           indStormList.append(indices)



   hsMaxStorm = []
   hsMaxTimeInStorm = []
   bmuMaxTimeInStorm = []
   durationHoursStorm = []
   dmAvgStorm = []
   ntrMaxStorm = []
   tpMaxStorm = []
   timeStorm = []
   timeStormEnd = []


   for x in range(len(hsStormList)):
        if keepList[x] in skipList:
             print('skipping storm {}'.format(x))
        else:
             tempHs = hsStormList[x]
             tempTp = tpStormList[x]
             timeStorm.append(timeStormList[x][12])
             timeStormEnd.append(timeStormList[x][-12])
             tempMaxInds = np.where((np.nanmax(tempHs) == tempHs))
             hsMaxStorm.append(np.nanmax(tempHs))
             dayOf = int(np.floor(tempMaxInds[0][-1]/24))
             hsMaxTimeInStorm.append(tempMaxInds[0][-1])
             tpMaxStorm.append(tempTp[tempMaxInds[0][-1]])
             durationHoursStorm.append(len(tempHs))
             tempDm = dmStormList[x]
             dmAvgStorm.append(np.nanmean(tempDm))
             tempNTR = ntrStormList[x]
             ntrMaxStorm.append(np.nanmax(tempNTR))

   hsStormArray = np.array(hsMaxStorm)
   tpStormArray = np.array(tpMaxStorm)
   dmStormArray = np.array(dmAvgStorm)
   ntrStormArray = np.array(ntrMaxStorm)
   durationStormArray = np.array(durationHoursStorm)
   timeStormArray = np.array(timeStorm)
   timeStormEndArray = np.array(timeStormEnd)

   filteredHs = hsStormArray[~np.isnan(ntrStormArray)]
   filteredTp = tpStormArray[~np.isnan(ntrStormArray)]
   filteredDm = dmStormArray[~np.isnan(ntrStormArray)]
   filteredDur = durationStormArray[~np.isnan(ntrStormArray)]
   filteredTime = timeStormArray[~np.isnan(ntrStormArray)]
   filteredNTR = ntrStormArray[~np.isnan(ntrStormArray)]
   filteredTimeEnd = timeStormEndArray[~np.isnan(ntrStormArray)]

   simsPickle = ('/media/dylananderson/Elements/waveSims/waveSims{}.pickle'.format(bump+hh))

   outputSims = {}
   outputSims['filteredHs'] = filteredHs
   outputSims['filteredTp'] = filteredTp
   outputSims['filteredDm'] = filteredDm
   outputSims['filteredDur'] = filteredDur
   outputSims['filteredTime'] = filteredTime
   outputSims['filteredTimeEnd'] = filteredTimeEnd
   outputSims['filteredNTR'] = filteredNTR
   outputSims['time'] = time
   outputSims['hs'] = hs
   outputSims['tp'] = tp
   outputSims['dm'] = dm
   outputSims['wl'] = wl

   with open(simsPickle, 'wb') as f:
      pickle.dump(outputSims, f)


   def datetime2datevec(dtime):
       'Return matlab date vector from datetimes'
       return [dtime.year, dtime.month, dtime.day]


   mdateVecTime = [datetime2datevec(x) for x in filteredTime]
   mdateVecTimeEnd = [datetime2datevec(x) for x in filteredTimeEnd]
   mdateVecTC = [datetime2datevec(x) for x in time]

   filteredOutput = dict()
   filteredOutput['filteredHs'] = filteredHs
   filteredOutput['filteredTp'] = filteredTp
   filteredOutput['filteredDm'] = filteredDm
   filteredOutput['filteredNTR'] = filteredNTR
   filteredOutput['filteredTimeEnd'] = mdateVecTimeEnd
   filteredOutput['filteredTime'] = mdateVecTime
   filteredOutput['filteredDur'] = filteredDur
   # filteredOutput['timeStormList'] = timeStormList
   filteredOutput['hs'] = hs
   filteredOutput['tp'] = tp
   filteredOutput['dm'] = dm
   filteredOutput['wl'] = wl
   filteredOutput['time'] = mdateVecTC

   import scipy.io
   fileMat = ('/media/dylananderson/Elements/waveSims/stormClimateSims{}.mat'.format(bump+hh))
   scipy.io.savemat(fileMat, filteredOutput)