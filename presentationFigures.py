import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


#### Wave time series plots ##########
# with open(r"realWaves.pickle", "rb") as input_file:
#    wavesInput = pickle.load(input_file)
# tWave = wavesInput['tWave'][5:]
# tC = wavesInput['tC'][5:]
# hsCombined = wavesInput['hsCombined'][5:]
# tpCombined = wavesInput['tpCombined'][5:]
# dmCombined = wavesInput['dmCombined'][5:]
# waveNorm = wavesInput['waveNorm']
# wlFRF = wavesInput['wlFRF']
# tFRF = wavesInput['tFRF']
# resFRF = wavesInput['resFRF']
# hh = 1
# # file = r"/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle".format(hh)
# file = r"/media/dylananderson/Elements/Sims/simulation{}.pickle".format(hh)
#
# with open(file, "rb") as input_file:
#     simsInput = pickle.load(input_file)
# simulationData = simsInput['simulationData']
# time = simsInput['time']
#
# plt.style.use('dark_background')
# plt.figure(figsize=(10,2))
# ax1 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
# ax1.plot(tC,hsCombined,'w-')
# ax1.set_ylim([0,10])
# plt.savefig('realWaves.png')
#
# plt.figure(figsize=(10,2))
# ax2 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
# ax2.plot(time,simulationData[:,0],'w-')
# ax2.set_ylim([0,10])
# plt.savefig('simWaves1.png')
#
# hh = 2
# # file = r"/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle".format(hh)
# file = r"/media/dylananderson/Elements/Sims/simulation{}.pickle".format(hh)
#
# with open(file, "rb") as input_file:
#     simsInput = pickle.load(input_file)
# simulationData = simsInput['simulationData']
# time = simsInput['time']
#
# plt.figure(figsize=(10,2))
# ax3 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
# ax3.plot(time,simulationData[:,0],'w-')
# ax3.plot(time,simulationData[:,0],'w-')
# ax3.set_ylim([0,10])
# plt.savefig('simWaves2.png')
#
# hh = 3
# # file = r"/home/dylananderson/projects/atlanticClimate/Sims/simulation{}.pickle".format(hh)
# file = r"/media/dylananderson/Elements/Sims/simulation{}.pickle".format(hh)
#
# with open(file, "rb") as input_file:
#     simsInput = pickle.load(input_file)
# simulationData = simsInput['simulationData']
# time = simsInput['time']
#
# plt.figure(figsize=(10,2))
# ax4 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
# ax4.plot(time,simulationData[:,0],'w-')
# ax4.set_ylim([0,10])
# plt.savefig('simWaves3.png')

from datetime import date

def dateDay2datetimeDate(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [date(d[0], d[1], d[2]) for d in d_vec]

def dateDay2datetime(d_vec):
   '''
   Returns datetime list from a datevec matrix
   d_vec = [[y1 m1 d1 H1 M1],[y2 ,2 d2 H2 M2],..]
   '''
   return [date(d[0], d[1], d[2]) for d in d_vec]




##### LOADING ALL OF THE DWTS THAT WILL BECOME THE TARGET ########
with open(r"dwts49Clusters.pickle", "rb") as input_file:
   historicalDWTs = pickle.load(input_file)

timeDWTs = historicalDWTs['SLPtime']
dwtBmus = historicalDWTs['bmus_corrected']

with open(r"dwtsOfExtraTropicalDays21Clusters.pickle", "rb") as input_file:
   historicalTWTs = pickle.load(input_file)
twtBmus = historicalTWTs['bmus_corrected']
twtOrder = historicalTWTs['kma_order']
tcIndices = historicalTWTs['tcIndices']
TIMEtcs = historicalTWTs['TIMEtcs']

timeSLPs = dateDay2datetimeDate(timeDWTs)
timeTCs = dateDay2datetimeDate(TIMEtcs)

overlap = [x for x in timeSLPs if x in timeTCs]
ind_dict = dict((k,i) for i,k in enumerate(timeSLPs))
inter = set(timeSLPs).intersection(timeTCs)
indices = [ ind_dict[x] for x in inter ]
indices.sort()

mask = np.ones(len(timeDWTs), np.bool)
mask[indices] = 0
timeEWTs = timeDWTs[mask,:]

mask2 = np.zeros(len(timeDWTs), np.bool)
mask2[indices] = 1
timeTWTs = timeDWTs[mask2,:]

bmus = np.nan * np.ones((len(timeDWTs)))
bmus[mask] = dwtBmus+0
bmus[mask2] = twtBmus+49

bmus_dates = dateDay2datetimeDate(timeDWTs)
bmus_dates_months = np.array([d.month for d in bmus_dates])
bmus_dates_days = np.array([d.day for d in bmus_dates])


########## BMUS FROM JUNE 1 TO MAY 31
bmus = bmus[120:]+1
timeDWTs = timeDWTs[120:]
bmus_dates = bmus_dates[120:]


with open(r"dwtFutureSimulations1000.pickle", "rb") as input_file:
   simsInput = pickle.load(input_file)
evbmus_sim = simsInput['evbmus_sim']
sim_num = simsInput['sim_years']
dates_sim = simsInput['dates_sim']

num_clusters = 70


def GenOneYearDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime(yy, month_ini, 2)
   dp2 = dp1 + timedelta(days=364)

   return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]


def GenOneSeasonDaily(yy=1981, month_ini=1):
   'returns one generic year in a list of datetimes. Daily resolution'

   dp1 = datetime(yy, month_ini, 1)
   dp2 = dp1 + timedelta(3*365/12)

   return [dp1 + timedelta(days=i) for i in range((dp2 - dp1).days)]


bmus_dates_months = np.array([d.month for d in dates_sim])
bmus_dates_days = np.array([d.day for d in dates_sim])



plt.style.use('dark_background')

# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=6)
m_plot = np.zeros((num_clusters, len(list_pyear))) * np.nan
numberOfSims = 1
# sort data
for i, dpy in enumerate(list_pyear):
   _, s = np.where(
      [(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)]
   )
   b = evbmus_sim[s,:]
   # b = bmus[s]
   b = b.flatten()

   for j in range(num_clusters):
      _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!

      m_plot[j, i] = float(len(bb) / float(numberOfSims)) / len(s)

import matplotlib.cm as cm
etcolors = cm.viridis(np.linspace(0, 1, 70-20))
tccolors = np.flipud(cm.autumn(np.linspace(0,1,21)))
dwtcolors = np.vstack((etcolors,tccolors[1:,:]))


fig = plt.figure()
ax = plt.subplot2grid((1,1),(0,0))
# plot stacked bars
bottom_val = np.zeros(m_plot[1, :].shape)
for r in range(num_clusters):
   row_val = m_plot[r, :]
   ax.bar(list_pyear, row_val, bottom=bottom_val,width=1, color=np.array([dwtcolors[r]]))
   # store bottom
   bottom_val += row_val

import matplotlib.dates as mdates
# customize  axis
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')
ax.set_xlim(list_pyear[0], list_pyear[-1])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_ylim(0, 1000)
ax.set_ylabel('')




d1 = datetime(1979,6,1)
d2 = datetime(2021, 5, 31)
dates_hist = [d1 + timedelta(days=i) for i in range((d2-d1).days+1)]

bmus_dates_months = np.array([d.month for d in dates_hist])
bmus_dates_days = np.array([d.day for d in dates_hist])
# generate perpetual year list
list_pyear = GenOneYearDaily(month_ini=6)
m_plot = np.zeros((70, len(list_pyear))) * np.nan
num_clusters=70
num_sim=1
# sort data
for i, dpy in enumerate(list_pyear):
   _, s = np.where([(bmus_dates_months == dpy.month) & (bmus_dates_days == dpy.day)])
   # b = evbmus_sim[s,:]
   b = bmus[s]
   b = b.flatten()

   for j in range(num_clusters):
      _, bb = np.where([(j + 1 == b)])  # j+1 starts at 1 bmus value!

      m_plot[j, i] = float(len(bb) / float(num_sim)) / len(s)

fig = plt.figure()
ax = plt.subplot2grid((1,1),(0,0))
# plot stacked bars
bottom_val = np.zeros(m_plot[1, :].shape)
for r in range(num_clusters):
   row_val = m_plot[r, :]
   ax.bar(
      list_pyear, row_val, bottom=bottom_val,
      width=1, color=np.array([dwtcolors[r]]))

   # store bottom
   bottom_val += row_val

import matplotlib.dates as mdates

# customize  axis
months = mdates.MonthLocator()
monthsFmt = mdates.DateFormatter('%b')

ax.set_xlim(list_pyear[0], list_pyear[-1])
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.set_ylim(0, 1)
ax.set_ylabel('')




