import os
import numpy as np
import datetime
from netCDF4 import Dataset
from scipy.stats.kde import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import gridspec
import pickle
from scipy.io.matlab.mio5_params import mat_struct
from datetime import datetime, date, timedelta
import random
import itertools
import operator
import scipy.io as sio
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.interpolate import interp1d
from scipy.stats import norm, genpareto, t
from scipy.special import ndtri  # norm inv
import matplotlib.dates as mdates
from scipy.stats import  genextreme, gumbel_l, spearmanr, norm, weibull_min

from scipy.spatial import distance
import xarray as xr




with open(r"waveHydrographs.pickle", "rb") as input_file:
   waveHydrographs = pickle.load(input_file)

hydros = waveHydrographs['hydros']


with open(r"hydrographCopulaData.pickle", "rb") as input_file:
   hydrographCopulaData = pickle.load(input_file)

copulaData = hydrographCopulaData['copulaData']

with open(r"historicalData.pickle", "rb") as input_file:
   historicalData = pickle.load(input_file)

grouped = historicalData['grouped']
groupLength = historicalData['groupLength']
bmuGroup = historicalData['bmuGroup']
timeGroup = historicalData['timeGroup']


### TODO: Need to fit copulas
#       - first requires a function looping through DWTs
#       - inside that a cdf needs to be fit for each
#       - which requires GEVs to be fit BEFOREHAND
#       - a correlation sigma is then calculated



def CDF_Distribution(self, vn, vv, xds_GEV_Par, d_shape, i_wt):
    '''
    Switch function: GEV / Empirical / Weibull

    Check variable distribution and calculates CDF

    vn - var name
    vv - var value
    i_wt - Weather Type index
    xds_GEV_Par , d_shape: GEV data used in sigma correlation
    '''

    # get GEV / EMPIRICAL / WEIBULL variables list
    vars_GEV = self.vars_GEV
    vars_EMP = self.vars_EMP
    vars_WBL = self.vars_WBL

    # switch variable name
    if vn in vars_GEV:

        # gev CDF
        sha_g = d_shape[vn][i_wt]
        loc_g = xds_GEV_Par.sel(parameter='location')[vn].values[i_wt]
        sca_g = xds_GEV_Par.sel(parameter='scale')[vn].values[i_wt]
        norm_VV = genextreme.cdf(vv, -1 * sha_g, loc_g, sca_g)

    elif vn in vars_EMP:

        # empirical CDF
        ecdf = ECDF(vv)
        norm_VV = ecdf(vv)

    elif vn in vars_WBL:

        # Weibull CDF
        norm_VV = weibull_min.cdf(vv, *weibull_min.fit(vv))

    return norm_VV


def ICDF_Distribution(self, vn, vv, pb, xds_GEV_Par, i_wt):
    '''
    Switch function: GEV / Empirical / Weibull

    Check variable distribution and calculates ICDF

    vn - var name
    vv - var value
    pb - var simulation probs
    i_wt - Weather Type index
    xds_GEV_Par: GEV parameters
    '''

    # optional empirical var_wt override
    fv = '{0}_{1}'.format(vn, i_wt + 1)
    if fv in self.sim_icdf_empirical_override:
        ppf_VV = Empirical_ICDF(vv, pb)
        return ppf_VV

    # get GEV / EMPIRICAL / WEIBULL variables list
    vars_GEV = self.vars_GEV
    vars_EMP = self.vars_EMP
    vars_WBL = self.vars_WBL

    # switch variable name
    if vn in vars_GEV:

        # gev ICDF
        sha_g = xds_GEV_Par.sel(parameter='shape')[vn].values[i_wt]
        loc_g = xds_GEV_Par.sel(parameter='location')[vn].values[i_wt]
        sca_g = xds_GEV_Par.sel(parameter='scale')[vn].values[i_wt]
        ppf_VV = genextreme.ppf(pb, -1 * sha_g, loc_g, sca_g)

    elif vn in vars_EMP:

        # empirical ICDF
        ppf_VV = Empirical_ICDF(vv, pb)

    elif vn in vars_WBL:

        # Weibull ICDF
        ppf_VV = weibull_min.ppf(pb, *weibull_min.fit(vv))

    return ppf_VV


def Calc_SigmaCorrelation_AllOn_Chromosomes(self, xds_KMA_MS, xds_WVS_MS, xds_GEV_Par):
    'Calculate Sigma Pearson correlation for each WT, all on chrom combo'

    bmus = xds_KMA_MS.bmus.values[:]
    cenEOFs = xds_KMA_MS.cenEOFs.values[:]
    n_clusters = len(xds_KMA_MS.n_clusters)
    wvs_fams = self.fams
    vars_extra = self.extra_variables
    vars_GEV = self.vars_GEV

    # smooth GEV shape parameter
    d_shape = {}
    for vn in vars_GEV:
        sh_GEV = xds_GEV_Par.sel(parameter='shape')[vn].values[:]
        d_shape[vn] = Smooth_GEV_Shape(cenEOFs, sh_GEV)

    # Get sigma correlation for each KMA cluster
    d_sigma = {}  # nested dict [WT][crom]
    for iwt in range(n_clusters):
        c = iwt+1
        pos = np.where((bmus==c))[0]
        d_sigma[c] = {}

        # current cluster waves
        xds_K_wvs = xds_WVS_MS.isel(time=pos)

        # append data for spearman correlation
        to_corr = np.empty((0, len(xds_K_wvs.time)))

        # solve normal inverse GEV/EMP/WBL CDF for each waves family
        for fam_n in wvs_fams:

            # get wave family variables
            vn_Hs = '{0}_Hs'.format(fam_n)
            vn_Tp = '{0}_Tp'.format(fam_n)
            vn_Dir = '{0}_Dir'.format(fam_n)

            vv_Hs = xds_K_wvs[vn_Hs].values[:]
            vv_Tp = xds_K_wvs[vn_Tp].values[:]
            vv_Dir = xds_K_wvs[vn_Dir].values[:]

            # fix fams nan: Hs 0, Tp mean, dir mean
            p_nans = np.where(np.isnan(vv_Hs))[0]
            vv_Hs[p_nans] = 0
            vv_Tp[p_nans] = np.nanmean(vv_Tp)
            vv_Dir[p_nans] = np.nanmean(vv_Dir)

            # Hs
            norm_Hs = self.CDF_Distribution(vn_Hs, vv_Hs, xds_GEV_Par, d_shape, iwt)

            # Tp
            norm_Tp = self.CDF_Distribution(vn_Tp, vv_Tp, xds_GEV_Par, d_shape, iwt)

            # Dir
            norm_Dir = self.CDF_Distribution(vn_Dir, vv_Dir, xds_GEV_Par, d_shape, iwt)

            # normal inverse CDF
            u_cdf = np.column_stack([norm_Hs, norm_Tp, norm_Dir])
            u_cdf[u_cdf>=1.0] = 0.999999
            inv_n = ndtri(u_cdf)

            # concatenate data for correlation
            to_corr = np.concatenate((to_corr, inv_n.T), axis=0)

        # concatenate extra variables for correlation
        for vn in vars_extra:
            vv = xds_K_wvs[vn].values[:]

            norm_vn = self.CDF_Distribution(vn, vv, xds_GEV_Par, d_shape, iwt)
            norm_vn[norm_vn>=1.0] = 0.999999

            inv_n = ndtri(norm_vn)
            to_corr = np.concatenate((to_corr, inv_n[:, None].T), axis=0)

        # sigma: spearman correlation
        corr, pval = spearmanr(to_corr, axis=1)

        # store data at dict (keep cromosomes structure)
        d_sigma[c][0] = {
            'corr': corr, 'data': len(xds_K_wvs.time), 'wt_crom': 1
        }

    return d_sigma




def Calc_GEVParams(self, xds_KMA_MS, xds_WVS_MS):
    '''
    Fits each WT (KMA.bmus) waves families data to a GEV distribtion
    Requires KMA and WVS families at storms max. TWL

    Returns xarray.Dataset with GEV shape, location and scale parameters
    '''

    vars_gev = self.vars_GEV
    bmus = xds_KMA_MS.bmus.values[:]
    cenEOFs = xds_KMA_MS.cenEOFs.values[:]
    n_clusters = len(xds_KMA_MS.n_clusters)

    xds_GEV_Par = xr.Dataset(
        coords = {
            'n_cluster' : np.arange(n_clusters)+1,
            'parameter' : ['shape', 'location', 'scale'],
        }
    )

    # Fit each wave family var to GEV distribution (using KMA bmus)
    for vn in vars_gev:
        gp_pars = FitGEV_KMA_Frechet(
            bmus, n_clusters, xds_WVS_MS[vn].values[:])

        xds_GEV_Par[vn] = (('n_cluster', 'parameter',), gp_pars)

    return xds_GEV_Par


def fitGEVparams(var):
    '''
    Returns stationary GEV/Gumbel_L params for KMA bmus and varible series

    bmus        - KMA bmus (time series of KMA centroids)
    n_clusters  - number of KMA clusters
    var         - time series of variable to fit to GEV/Gumbel_L

    returns np.array (n_clusters x parameters). parameters = (shape, loc, scale)
    for gumbel distributions shape value will be ~0 (0.0000000001)
    '''

    param_GEV = np.empty((3,))

    # get variable at cluster position
    var_c = var
    var_c = var_c[~np.isnan(var_c)]

    # fit to Gumbel_l and get negative loglikelihood
    loc_gl, scale_gl = gumbel_l.fit(-var_c)
    theta_gl = (0.0000000001, -1*loc_gl, scale_gl)
    nLogL_gl = genextreme.nnlf(theta_gl, var_c)

    # fit to GEV and get negative loglikelihood
    c = -0.1
    shape_gev, loc_gev, scale_gev = genextreme.fit(var_c, c)
    theta_gev = (shape_gev, loc_gev, scale_gev)
    nLogL_gev = genextreme.nnlf(theta_gev, var_c)

    # store negative shape
    theta_gev_fix = (-shape_gev, loc_gev, scale_gev)

    # apply significance test if Frechet
    if shape_gev < 0:

        # TODO: cant replicate ML exact solution
        if nLogL_gl - nLogL_gev >= 1.92:
            param_GEV = list(theta_gev_fix)
        else:
            param_GEV = list(theta_gl)
    else:
        param_GEV = list(theta_gev_fix)

    return param_GEV

def Smooth_GEV_Shape(cenEOFs, param):
    '''
    Smooth GEV shape parameter (for each KMA cluster) by promediation
    with neighbour EOFs centroids

    cenEOFs  - (n_clusters, n_features) KMA centroids
    param    - GEV shape parameter for each KMA cluster

    returns smoothed GEV shape parameter as a np.array (n_clusters)
    '''

    # number of clusters
    n_cs = cenEOFs.shape[0]

    # calculate distances (optimized)
    cenEOFs_b = cenEOFs.reshape(cenEOFs.shape[0], 1, cenEOFs.shape[1])
    D = np.sqrt(np.einsum('ijk, ijk->ij', cenEOFs-cenEOFs_b, cenEOFs-cenEOFs_b))
    np.fill_diagonal(D, np.nan)

    # sort distances matrix to find neighbours
    sort_ord = np.empty((n_cs, n_cs), dtype=int)
    D_sorted = np.empty((n_cs, n_cs))
    for i in range(n_cs):
        order = np.argsort(D[i,:])
        sort_ord[i,:] = order
        D_sorted[i,:] = D[i, order]

    # calculate smoothed parameter
    denom = np.sum(1/D_sorted[:,:4], axis=1)
    param_c = 0.5 * np.sum(np.column_stack(
        [
            param[:],
            param[sort_ord[:,:4]] * (1/D_sorted[:,:4])/denom[:,None]
        ]
    ), axis=1)

    return param_c




def gev_CDF(x):
    '''
    :param x: observations
    :return: normalized cdf
    '''

    shape, loc, scale = fitGEVparams(x)
    # # gev CDF
    # sha_g = d_shape[vn][i_wt]
    # loc_g = xds_GEV_Par.sel(parameter='location')[vn].values[i_wt]
    # sca_g = xds_GEV_Par.sel(parameter='scale')[vn].values[i_wt]
    cdf = genextreme.cdf(x, -1 * shape, loc, scale)

    return cdf

def gev_ICDF(x,y):
    '''
    :param x: observations
    :param y: simulated probabilities
    :return: simulated values
    '''
    shape, loc, scale = fitGEVparams(x)
    ppf_VV = genextreme.ppf(y, -1 * shape, loc, scale)
    return ppf_VV


def ksdensity_CDF(x):
    '''
    Kernel smoothing function estimate.
    Returns cumulative probability function at x.
    '''

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF at x position (kde.support = x)
    fint = interp1d(kde.support, kde.cdf)

    return fint(x)

def ksdensity_ICDF(x, p):
    '''
    Returns Inverse Kernel smoothing function at p points
    '''

    # fit a univariate KDE
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit()

    # interpolate KDE CDF to get support values 
    fint = interp1d(kde.cdf, kde.support)

    # ensure p inside kde.cdf
    p[p<np.min(kde.cdf)] = kde.cdf[0]
    p[p>np.max(kde.cdf)] = kde.cdf[-1]

    return fint(p)

def GeneralizedPareto_CDF(x):
    '''
    Generalized Pareto fit
    Returns cumulative probability function at x.
    '''

    # fit a generalized pareto and get params
    shape, _, scale = genpareto.fit(x)

    # get generalized pareto CDF
    cdf = genpareto.cdf(x, shape, scale=scale)

    return cdf

def GeneralizedPareto_ICDF(x, p):
    '''
    Generalized Pareto fit
    Returns inverse cumulative probability function at p points
    '''

    # fit a generalized pareto and get params
    shape, _, scale = genpareto.fit(x)

    # get percent points (inverse of CDF)
    icdf = genpareto.ppf(p, shape, scale=scale)

    return icdf

def Empirical_CDF(x):
    '''
    Returns empirical cumulative probability function at x.
    '''

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    return cdf

def Empirical_ICDF(x, p):
    '''
    Returns inverse empirical cumulative probability function at p points
    '''

    # TODO: build in functionality for a fill_value?

    # fit ECDF
    ecdf = ECDF(x)
    cdf = ecdf(x)

    # interpolate KDE CDF to get support values 
    fint = interp1d(
        cdf, x,
        fill_value=(np.nanmin(x), np.nanmax(x)),
        #fill_value=(np.min(x), np.max(x)),
        bounds_error=False
    )
    return fint(p)


def copulafit(u, family='gaussian'):
    '''
    Fit copula to data.
    Returns correlation matrix and degrees of freedom for t student
    '''

    rhohat = None  # correlation matrix
    nuhat = None  # degrees of freedom (for t student)

    if family=='gaussian':
        u[u>=1.0] = 0.999999
        inv_n = ndtri(u)
        rhohat = np.corrcoef(inv_n.T)

    elif family=='t':
        raise ValueError("Not implemented")

        # TODO:
        x = np.linspace(np.min(u), np.max(u),100)
        inv_t = np.ndarray((len(x), u.shape[1]))

        for j in range(u.shape[1]):
            param = t.fit(u[:,j])
            t_pdf = t.pdf(x,loc=param[0],scale=param[1],df=param[2])
            inv_t[:,j] = t_pdf

        # TODO CORRELATION? NUHAT?
        rhohat = np.corrcoef(inv_n.T)
        nuhat = None

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return rhohat, nuhat

def copularnd(family, rhohat, n):
    '''
    Random vectors from a copula
    '''

    if family=='gaussian':
        mn = np.zeros(rhohat.shape[0])
        np_rmn = np.random.multivariate_normal(mn, rhohat, n)
        u = norm.cdf(np_rmn)

    elif family=='t':
        # TODO
        raise ValueError("Not implemented")

    else:
        raise ValueError("Wrong family parameter. Use 'gaussian' or 't'")

    return u




def CopulaSimulation(U_data, kernels, num_sim):
    '''
    Fill statistical space using copula simulation

    U_data: 2D nump.array, each variable in a column
    kernels: list of kernels for each column at U_data (KDE | GPareto | Empirical | GEV)
    num_sim: number of simulations
    '''



    # kernel CDF dictionary
    d_kf = {
        'KDE' : (ksdensity_CDF, ksdensity_ICDF),
        'GPareto' : (GeneralizedPareto_CDF, GeneralizedPareto_ICDF),
        'ECDF' : (Empirical_CDF, Empirical_ICDF),
        'GEV': (gev_CDF, gev_ICDF),
    }


    # check kernel input
    if any([k not in d_kf.keys() for k in kernels]):
        raise ValueError(
            'wrong kernel: {0}, use: {1}'.format(
                kernel, ' | '.join(d_kf.keys())
            )
        )


    # normalize: calculate data CDF using kernels
    U_cdf = np.zeros(U_data.shape) * np.nan
    ic = 0
    for d, k in zip(U_data.T, kernels):
        cdf, _ = d_kf[k]  # get kernel cdf
        U_cdf[:, ic] = cdf(d)
        ic += 1

    # fit data CDFs to a gaussian copula
    rhohat, _ = copulafit(U_cdf, 'gaussian')

    # simulate data to fill probabilistic space
    U_cop = copularnd('gaussian', rhohat, num_sim)

    # de-normalize: calculate data ICDF
    U_sim = np.zeros(U_cop.shape) * np.nan
    ic = 0
    for d, c, k in zip(U_data.T, U_cop.T, kernels):
        _, icdf = d_kf[k]  # get kernel icdf
        U_sim[:, ic] = icdf(d, c)
        ic += 1

    return U_sim


### TODO: copula simulation using GEV params


gevCopulaSims = list()
for i in range(len(np.unique(bmuGroup))):
    tempCopula = np.asarray(copulaData[i])
    kernels = ['GEV','GEV','KDE','KDE','KDE','KDE',]
    samples = CopulaSimulation(tempCopula,kernels,10000)
    print('generating samples for DWT {}'.format(i))
    gevCopulaSims.append(samples)


samplesPickle = 'copulaSamplesTest.pickle'
outputSamples = {}
outputSamples['gevCopulaSims'] = gevCopulaSims
outputSamples['kernels'] = kernels

with open(samplesPickle,'wb') as f:
    pickle.dump(outputSamples, f)
