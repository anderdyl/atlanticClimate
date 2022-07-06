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
#Getting main packages from R in order to apply the maximum likelihood function
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

POT = importr('POT') #importing POT package


def return_value(sample_real, threshold, alpha, block_size, return_period,
                 fit_method):  # return value plot and return value estimative
   sample = np.sort(sample_real)
   sample_excess = []
   sample_over_thresh = []
   for data in sample:
      if data > threshold + 0.00001:
         sample_excess.append(data - threshold)
         sample_over_thresh.append(data)

   rdata = FloatVector(sample)
   fit = POT.fitgpd(rdata, threshold, est=fit_method)  # fit data
   shape = fit[0][1]
   scale = fit[0][0]

   # Computing the return value for a given return period with the confidence interval estimated by the Delta Method
   m = return_period
   Eu = len(sample_over_thresh) / len(sample)
   x_m = threshold + (scale / shape) * (((m * Eu) ** shape) - 1)

   # Solving the delta method
   d = Eu * (1 - Eu) / len(sample)
   e = fit[3][0]
   f = fit[3][1]
   g = fit[3][2]
   h = fit[3][3]
   a = (scale * (m ** shape)) * (Eu ** (shape - 1))
   b = (shape ** -1) * (((m * Eu) ** shape) - 1)
   c = (-scale * (shape ** -2)) * ((m * Eu) ** shape - 1) + (scale * (shape ** -1)) * ((m * Eu) ** shape) * mt.log(
      m * Eu)
   CI = (norm.ppf(1 - (alpha / 2)) * ((((a ** 2) * d) + (b * ((c * g) + (e * b))) + (c * ((b * f) + (c * h)))) ** 0.5))

   print('The return value for the given return period is {} \u00B1 {}'.format(x_m, CI))

   ny = block_size  # defining how much observations will be a block (usually anual)
   N_year = return_period / block_size  # N_year represents the number of years based on the given return_period

   for i in range(0, len(sample)):
      if sample[i] > threshold + 0.0001:
         i_initial = i
         break

   p = np.arange(i_initial, len(sample)) / (len(sample))  # Getting Plotting Position points
   N = 1 / (ny * (1 - p))  # transforming plotting position points to years

   year_array = np.arange(min(N), N_year + 0.1, 0.1)  # defining a year array

   # Algorithm to compute the return value and the confidence intervals for plotting
   z_N = []
   CI_z_N_high_year = []
   CI_z_N_low_year = []
   for year in year_array:
      z_N.append(threshold + (scale / shape) * (((year * ny * Eu) ** shape) - 1))
      a = (scale * ((year * ny) ** shape)) * (Eu ** (shape - 1))
      b = (shape ** -1) * ((((year * ny) * Eu) ** shape) - 1)
      c = (-scale * (shape ** -2)) * (((year * ny) * Eu) ** shape - 1) + (scale * (shape ** -1)) * (
                 ((year * ny) * Eu) ** shape) * mt.log((year * ny) * Eu)
      CIyear = (norm.ppf(1 - (alpha / 2)) * (
                 (((a ** 2) * d) + (b * ((c * g) + (e * b))) + (c * ((b * f) + (c * h)))) ** 0.5))
      CI_z_N_high_year.append(threshold + (scale / shape) * (((year * ny * Eu) ** shape) - 1) + CIyear)
      CI_z_N_low_year.append(threshold + (scale / shape) * (((year * ny * Eu) ** shape) - 1) - CIyear)

   # Plotting Return Value
   # plt.figure(8)
   # plt.plot(year_array, CI_z_N_high_year, linestyle='--', color='red', alpha=0.8, lw=0.9, label='Confidence Bands')
   # plt.plot(year_array, CI_z_N_low_year, linestyle='--', color='red', alpha=0.8, lw=0.9)
   # plt.plot(year_array, z_N, color='black', label='Theoretical Return Level')
   # plt.scatter(N, sample_over_thresh, label='Empirical Return Level')
   # plt.xscale('log')
   # plt.xlabel('Return Period')
   # plt.ylabel('Return Level')
   # plt.title('Return Level Plot')
   # plt.legend()
   #
   # plt.show()

   output = dict()
   output['year_array'] = year_array
   output['N'] = N
   output['sample_over_thresh'] = sample_over_thresh
   output['CI_z_N_high_year'] = CI_z_N_high_year
   output['CI_z_N_low_year'] = CI_z_N_low_year
   output['z_N'] = z_N
   output['CI'] = CI
   return output


def moving_average(a, n=3):
   ret = np.cumsum(a, dtype=float)
   ret[n:] = ret[n:] - ret[:-n]
   return ret[n - 1:] / n


with open(r"realWaves.pickle", "rb") as input_file:
   wavesInput = pickle.load(input_file)
tWave = wavesInput['tWave'][5:]
tC = wavesInput['tC'][5:]
hsCombined = wavesInput['hsCombined']
hsCombined = moving_average(hsCombined,3)
hsCombined = hsCombined[3:]
tpCombined = wavesInput['tpCombined'][5:]
dmCombined = wavesInput['dmCombined'][5:]
waveNorm = wavesInput['waveNorm']
wlFRF = wavesInput['wlFRF']
tFRF = wavesInput['tFRF']
resFRF = wavesInput['resFRF']

data = np.array([hsCombined,tpCombined,dmCombined])
ogdf = pd.DataFrame(data=data.T, index=tC, columns=["hs", "tp", "dm"])
year = np.array([tt.year for tt in tC])
ogdf['year'] = year
month = np.array([tt.month for tt in tC])
ogdf['month'] = month

dailyMaxHs = ogdf.resample("d")['hs'].max()

seasonalMean = ogdf.groupby('month').mean()
seasonalStd = ogdf.groupby('month').std()
yearlyMax = ogdf.groupby('year').max()

g2 = ogdf.groupby(pd.Grouper(freq="M")).mean()
c = 0
threeDayMax = []
while c < len(hsCombined):
    threeDayMax.append(np.nanmax(hsCombined[c:c+72]))
    c = c + 72
threeDayMaxHs = np.asarray(threeDayMax)

c = 0
fourDayMax = []
while c < len(hsCombined):
   fourDayMax.append(np.nanmax(hsCombined[c:c + 96]))
   c = c + 96
fourDayMaxHs = np.asarray(fourDayMax)

simSeasonalMean = np.nan * np.ones((250,12))
simSeasonalStd = np.nan * np.ones((250,12))
simYearlyMax = np.nan * np.ones((250,101))
yearArray = []
zNArray = []
ciArray = []
for hh in range(250):
   # file = r"/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle".format(hh)
   file = r"/media/dylananderson/Elements/Sims/simulation{}.pickle".format(hh)

   with open(file, "rb") as input_file:
      simsInput = pickle.load(input_file)
   simulationData = simsInput['simulationData']
   df = simsInput['df']
   time = simsInput['time']
   year = np.array([tt.year for tt in time])
   df['year'] = year
   month = np.array([tt.month for tt in time])
   df['month'] = month

   g1 = df.groupby(pd.Grouper(freq="M")).mean()
   simSeasonalMean[hh,:] = df.groupby('month').mean()['hs']
   simSeasonalStd[hh,:] = df.groupby('month').std()['hs']
   simYearlyMax[hh,:] = df.groupby('year').max()['hs']
   dailyMaxHsSim = df.resample("d")['hs'].max()
   c = 0
   threeDayMaxSim = []
   while c < len(simulationData):
      threeDayMaxSim.append(np.nanmax(simulationData[c:c + 72, 0]))
      c = c + 72
   threeDayMaxHsSim = np.asarray(threeDayMaxSim)

   c = 0
   fourDayMaxSim = []
   while c < len(simulationData):
      fourDayMaxSim.append(np.nanmax(simulationData[c:c + 96, 0]))
      c = c + 96
   fourDayMaxHsSim = np.asarray(fourDayMaxSim)

   sim = return_value(np.asarray(fourDayMaxHsSim)[0:365*40], 4, 0.05, 365/4, 36525/4, 'mle')

   yearArray.append(sim['year_array'])
   zNArray.append(sim['z_N'])
   ciArray.append(sim['CI'])


#
# import pandas
# file = r"/home/dylananderson/projects/atlanticClimate/Sims/allSimulations.pickle"
#
# with open(file, "rb") as input_file:
#    simsInput = pickle.load(input_file)
# simulationHs = simsInput['simulationsHs']
# simulationTp = simsInput['simulationsTp']
# simulationTimes = simsInput['simulationsTime']
#
# simYearlyMaxNotInterped = np.nan * np.ones((50,101))
#
# for hh in range(50):
#    simData = np.array([simulationHs[hh],simulationTp[hh]])
#
#    simdf = pandas.DataFrame(data=simData.T, index=simulationTimes[hh], columns=["hs","tp"])
#    year = np.array([tt.year for tt in simulationTimes[hh]])
#    simdf['year'] = year
#    month = np.array([tt.month for tt in simulationTimes[hh]])
#    simdf['month'] = month
#
#    #g1 = df.groupby(pd.Grouper(freq="M")).mean()
#    #simSeasonalMean[hh,:] = df.groupby('month').mean()['hs']
#    #simSeasonalStd[hh,:] = df.groupby('month').std()['hs']
#    simYearlyMaxNotInterped[hh,:] = simdf.groupby('year').max()['hs']
#

dt = datetime(2022, 1, 1)
end = datetime(2023, 1, 1)
step = relativedelta(months=1)
plotTime = []
while dt < end:
    plotTime.append(dt)#.strftime('%Y-%m-%d'))
    dt += step

plt.figure()
ax1 = plt.subplot2grid((1,1),(0,0),rowspan=1,colspan=1)
ax1.plot(plotTime,seasonalMean['hs'],label='WIS record (41 years)')
ax1.fill_between(plotTime, seasonalMean['hs'] - seasonalStd['hs'], seasonalMean['hs'] + seasonalStd['hs'], color='b', alpha=0.2)
ax1.plot(plotTime,df.groupby('month').mean()['hs'],label='Synthetic record (100 years)')
ax1.fill_between(plotTime, df.groupby('month').mean()['hs'] - df.groupby('month').std()['hs'], df.groupby('month').mean()['hs'] + df.groupby('month').std()['hs'], color='orange', alpha=0.2)
# ax1.fill_between(plotTime, simSeasonalMean['hs'] - simSeasonalStd['hs'], simSeasonalMean['hs'] + simSeasonalStd['hs'], color='orange', alpha=0.2)
ax1.set_xticks([plotTime[0],plotTime[1],plotTime[2],plotTime[3],plotTime[4],plotTime[5],plotTime[6],plotTime[7],plotTime[8],plotTime[9],plotTime[10],plotTime[11]])
ax1.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax1.legend()

simMaxHs = np.nan*np.ones((250,100))
# simMaxHsNotInterped = np.nan*np.ones((50,100))
for hh in range(250):
   simMaxHs[hh,:] = np.sort(simYearlyMax[hh,0:100])
   # simMaxHsNotInterped[hh,:] = np.sort(simYearlyMaxNotInterped[hh,0:-1])

plt.figure()
ax2 = plt.subplot2grid((1,1),(0,0))
maxHs = np.sort(yearlyMax['hs'])
# maxHsNotInterped = np.sort(yearlyMax['hs'])

returnPeriod = np.flipud((len(maxHs)+1)/np.arange(1,len(maxHs)+1))
# simMaxHs = np.sort(simYearlyMax['hs'][0:-1])
# simReturnPeriod = np.flipud(100/np.arange(1,101))
simReturnPeriod = np.flipud(100/np.arange(1,101))

ax2.fill_between(simReturnPeriod, np.min(simMaxHs,axis=0), np.max(simMaxHs,axis=0), color='orange', alpha=0.2)
ax2.plot(returnPeriod,maxHs,'o')
ax2.plot(simReturnPeriod,np.mean(simMaxHs,axis=0),'.-')

# ax2.fill_between(simReturnPeriod, np.min(simMaxHsNotInterped,axis=0), np.max(simMaxHsNotInterped,axis=0), color='green', alpha=0.2)
# ax2.plot(simReturnPeriod,np.mean(simMaxHsNotInterped,axis=0),'.-')
# ax2.plot(simReturnPeriod,simMaxHs,'.')
ax2.set_xscale('log')
# ax1.set_xticks([plotTime[0],plotTime[2],plotTime[4],plotTime[6],plotTime[8],plotTime[10]])
# ax1.set_xticklabels(['Jan','Mar','May','Jul','Sep','Nov'])


# from rpy2.robjects.packages import importr
# import rpy2.robjects.packages as rpackages
#
# base = importr('base')
# utils = importr('utils')
# utils.chooseCRANmirror(ind=1)
# utils.install_packages('POT') #installing POT package
# from thresholdmodeling import thresh_modeling #importing package
# import pandas as pd #importing pandas
#
# #url = 'https://raw.githubusercontent.com/iagolemos1/thresholdmodeling/master/dataset/rain.csv' #saving url
# #df =  pd.read_csv(url, error_bad_lines=False) #getting data
# data = df['hs'].values.ravel() #turning data into an array
#
data = np.asarray(dailyMaxHs)
# #data =
#
# # thresh_modeling.MRL(data, 0.05)
# # thresh_modeling.Parameter_Stability_plot(data, 0.05)
# # thresh_modeling.return_value(data, 30, 0.05, 365, 36500, 'mle')
#
# dataDecluster, data2 = thresh_modeling.decluster(data,3,24*2)
# thresh_modeling.return_value(data, 3, 0.05, 365.25*24, 36525*24, 'mle')
#


import matplotlib.cm as cm
import matplotlib.colors as mcolors

historical = return_value(np.asarray(fourDayMax), 4, 0.05, 365/4, 36525/4, 'mle')

plt.style.use('dark_background')


# to do order this by uncertainty
plt.figure(8)
colorparam = np.zeros((len(zNArray),))
for qq in range(len(zNArray)):
   normalize = mcolors.Normalize(vmin=0, vmax=5)
   colorparam[qq] = ciArray[qq]
   colormap = cm.Greys
   color = colormap(normalize(colorparam[qq]))
   plt.plot(yearArray[qq],zNArray[qq],color=color,alpha=0.75)#color=[0.5,0.5,0.5],alpha=0.5)

plt.plot(historical['year_array'], historical['CI_z_N_high_year'], linestyle='--', color='red', alpha=0.8, lw=0.9, label='Confidence Bands')
plt.plot(historical['year_array'], historical['CI_z_N_low_year'], linestyle='--', color='red', alpha=0.8, lw=0.9)
plt.plot(historical['year_array'], historical['z_N'], color='orange', label='Theoretical Return Level')
plt.scatter(historical['N'], historical['sample_over_thresh'], color='orange',label='Empirical Return Level',zorder=10)
plt.xscale('log')
plt.xlabel('Return Period')
plt.ylabel('Return Level')
plt.title('Return Level Plot')
plt.legend()

plt.show()

