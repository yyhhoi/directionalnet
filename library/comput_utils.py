#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" General helper functions
"""
import warnings

import numpy as np
import pandas as pd
import random

from scipy.interpolate import interp1d
from scipy.special import comb
from scipy.stats import pearsonr, vonmises, norm, ranksums, f as fdist, fisher_exact
from scipy import interpolate

import scipy.ndimage as ndimage
from pycircstat.descriptive import resultant_vector_length, center, cdiff
from pycircstat.descriptive import mean as circmean

from library.correlogram import ThetaEstimator


def dist_overlap(rate_map1, rate_map2, mask1, mask2):
    """

    Parameters
    ----------
    rate_map1 : ndarray
        2D rate map of first place field.
    rate_map2 : ndarray
        2D rate map of second place field.
    mask1 : ndarray
        2D boolean mask of first place field.
    mask2 : ndarray
        2D boolean mask of second place field.

    Returns
    -------
        pr : float
            Pearson R correlation coefficient from 0 (lowest overlap) to 1 (highest overlap).
        ks_dist : float
            Kolmogorov-Smirnov distance from 0 (lowest overlap) to 1 (highest overlap).
        min_r : float
            Minimum sum between the two distribution, from 0 (lowest overlap) to 1 (highest overlap).

    """

    assert len(rate_map1.shape) == 2 and len(rate_map2.shape) == 2
    assert rate_map1.shape == rate_map2.shape

    rate_map1 = rate_map1 * mask1
    rate_map2 = rate_map2 * mask2

    rate1_clipped = rate_map1
    rate2_clipped = rate_map2

    # Normalized
    r1_n, r2_n = rate1_clipped / np.sum(rate1_clipped), rate2_clipped / np.sum(rate2_clipped)

    # Pearson R
    pr, _ = pearsonr(r1_n.flatten(), r2_n.flatten())
    pr = (pr + 1) / 2

    # ks distance
    r1_cumsum = np.cumsum(np.cumsum(r1_n, 0), 1)
    r2_cumsum = np.cumsum(np.cumsum(r2_n, 0), 1)
    ks_dist = 1 - np.max(np.abs(r1_cumsum - r2_cumsum))  # s.t. overlap increases from 0 to 1

    # min overlap
    min_r = np.sum(np.min(np.stack([r1_n, r2_n]), axis=0))

    return pr, ks_dist, min_r



def get_numpass_at_angle(target_angle, aedge, all_passbins):
    binstmp, _ = np.histogram(shiftcyc_full2half(target_angle), bins=aedge)
    binsidxtmp = np.argmax(binstmp)
    numpass_at_precess = all_passbins[binsidxtmp]
    return numpass_at_precess


def compute_straightness(pass_angles):
    mean_r = np.abs(np.mean(np.exp(1j * pass_angles)))
    m = pass_angles.shape[0]
    straightness = ((m * np.square(mean_r)) - 1) / np.sqrt(1 - (1 / m))
    return straightness


def correlate(arr1, arr2):
    try:
        assert (len(arr1.shape) == 1) and (len(arr2.shape) == 1)
    except AssertionError:
        print('Input arrays shall be one-dimensional numpy.darray')

    dim_n, dim_m = arr1.shape[0], arr2.shape[0]
    out = np.correlate(arr1, arr2, mode="full")
    min_length = np.min([dim_n, dim_m]) - 1
    lags = np.arange(-min_length, min_length + 1, step=1)
    try:
        assert lags.shape[0] == dim_n + dim_m - 1
    except AssertionError:
        print('Unknown error causing output with unmatched length')
    return out, lags


def custum_interpolate(lags, r, n_order=5):
    tck = interpolate.splrep(lags, r, s=0)
    lags_inter = np.linspace(np.min(lags), np.max(lags), lags.shape[0] * n_order)
    r_inter = interpolate.splev(lags_inter, tck, der=0)
    return lags_inter, r_inter


def smooth_xy(x, y, num_bins=50, sigma=(2, 2)):
    H, xedges, yedges = np.histogram2d(x, y, bins=num_bins)
    img = ndimage.gaussian_filter(H, sigma=sigma, order=0)
    return img, xedges, yedges


def normsig(signal_filt, mode='max'):
    signal_filt_np = np.asarray(signal_filt)
    with_offset = signal_filt_np - np.nanmin(signal_filt_np)

    if mode == 'max':
        norm = with_offset / np.nanmax(with_offset)
    elif mode == 'area':
        norm = with_offset / np.nansum(with_offset)

    return norm


def comput_overlap(signal1, signal2, method='pearsonr'):
    """
    Args:
        signal1 (numpy.darray):
            First Cross-correlation signal (from A to B), with shape (m, ), normalized with respect to maximum or total area.
        signal2 (numpy.darray):
            Second Cross-correlation signal (from B to A), with shape (m, ), normalized with respect to maximum or total area. It would be flipped.
    Returns:
        overlap_plus (numpy.float64):
            Degree of overlap (by minimum signal between the two) between signal1 and signal 2. Not yet normalized.
        
        overlap_minus (numpy.float64):
            Degree of overlap between signal1 and flipped signal 2. Not yet normalized.
        
        signal1, signal2, flipped_signal2, min_12, min_1flip2:
            Intermediate processed signals stored for examination. 
            
    """

    flipped_signal2 = np.flip(signal2)
    if method == 'min':
        min_12 = np.min(np.stack([signal1, signal2]), axis=0)
        min_1flip2 = np.min(np.stack([signal1, flipped_signal2]), axis=0)
    elif method == 'pearsonr':
        min_12, _ = pearsonr(signal1, signal2)
        min_12 = (min_12 + 1) / 2
        min_1flip2, _ = pearsonr(signal1, flipped_signal2)
        min_1flip2 = (min_1flip2 + 1) / 2
    else:
        raise AssertionError('method argument has to be either "pearsonr" or "min"')
    overlap_plus = np.sum(min_12)
    overlap_minus = np.sum(min_1flip2)

    return (overlap_plus, overlap_minus), (signal1, signal2, flipped_signal2, min_12, min_1flip2)


def append_extrinsicity(df):
    '''
    Cross-correlograms are normalized by their areas. 'Overlap' between two cross-correlograms is calculated by
    Pearson's R.
    Parameters
    ----------
    df : dataframe

    Returns
    -------
    df : dataframe
        Input dataframe with appended columns "overlap_plus", "overlap_minus" and "overlap_ratio".
        Also, with the variants of "overlap_plus_phase", "overlap_minus_phase" and "overlap_ratio_phase"
    '''

    overlap_plus_list, overlap_minus_list = [], []
    overlap_plus_phase_list, overlap_minus_phase_list = [], []
    alpha_edges = np.linspace(-np.pi, np.pi, 36)
    for i in range(df.shape[0]):
        corr_info_AB, corr_info_BA = df.loc[i, ['corr_info_AB', 'corr_info_BA']]
        if hasattr(corr_info_AB, '__iter__') and hasattr(corr_info_BA, '__iter__'):
            (isibins_AB, isiedges_AB, signal_filt_AB, zAB, alphasAB, isiAB) = corr_info_AB
            (isibins_BA, isiedges_BA, signal_filt_BA, zBA, alphasBA, isiBA) = corr_info_BA

            # Overlap of correlograms
            norm_AB, norm_BA = normsig(isibins_AB, mode='area'), normsig(isibins_BA, mode='area')
            (overlap_plus, overlap_minus), _ = comput_overlap(norm_AB, norm_BA, method='pearsonr')

            # Overlap of phase correlograms
            midedgesAB, midedgesBA = midedges(isiedges_AB), midedges(isiedges_BA)
            interAB, interBA = interp1d(midedgesAB, alphasAB), interp1d(midedgesBA, alphasBA)
            isiphaseAB, isiphaseBA = interAB(isiAB), interBA(isiBA)
            phasebinsAB, _ = np.histogram(isiphaseAB, bins=alpha_edges)
            phasebinsBA, _ = np.histogram(isiphaseBA, bins=alpha_edges)
            norm_AB_phase, norm_BA_phase = normsig(phasebinsAB, mode='area'), normsig(phasebinsBA, mode='area')
            (overlap_plus_phase, overlap_minus_phase), _ = comput_overlap(norm_AB_phase, norm_BA_phase, method='pearsonr')

            overlap_plus_list.append(overlap_plus)
            overlap_minus_list.append(overlap_minus)
            overlap_plus_phase_list.append(overlap_plus_phase)
            overlap_minus_phase_list.append(overlap_minus_phase)

        else:

            overlap_plus_list.append(np.nan)
            overlap_minus_list.append(np.nan)
            overlap_plus_phase_list.append(np.nan)
            overlap_minus_phase_list.append(np.nan)
    df['overlap_plus'] = overlap_plus_list
    df['overlap_minus'] = overlap_minus_list
    df['overlap_ratio'] = df['overlap_minus'] - df['overlap_plus']
    df['overlap_plus_phase'] = overlap_plus_phase_list
    df['overlap_minus_phase'] = overlap_minus_phase_list
    df['overlap_ratio_phase'] = df['overlap_minus_phase'] - df['overlap_plus_phase']
    return df




def circular_circular_gauss_density(x_cir, y_cir, cir_kappax=8 * np.pi, cir_kappay=8 * np.pi, xbins=300, ybins=1200,
                                    xbound=(0, 1), ybound=(-np.pi, 2 * np.pi)):
    '''
    Args:
        x_cir (numpy.darray) : 1-d array of linear variables
        y_cir (numpy.darray) : 1-d array of circular variables with same shape as x_lin
        cir_kappax (float) : Concentration of Von Mise distribotion
        cir_kappay (float) : Concentration of Von Mise distribotion
        xbins (int) : Number of bins for x axis
        ybins (int) : Number of bins for y axis
        
    '''

    num_points = x_cir.shape[0]

    xaxis = np.linspace(xbound[0], xbound[1], xbins)
    yaxis = np.linspace(ybound[0], ybound[1], ybins)
    xx, yy = np.meshgrid(xaxis, yaxis)
    zz = np.zeros(xx.shape)

    for npt in range(num_points):
        x_each, y_each = x_cir[npt], y_cir[npt]
        y_cir_pdf = vonmises.pdf(yy, cir_kappay, loc=y_each, scale=1)
        y_cir_pdf = y_cir_pdf / np.max(y_cir_pdf)
        x_cir_pdf = vonmises.pdf(xx, cir_kappax, loc=x_each, scale=1)
        x_cir_pdf = x_cir_pdf / np.max(x_cir_pdf)
        norm_pdf = y_cir_pdf * x_cir_pdf
        norm_pdf = norm_pdf / np.sum(norm_pdf)
        zz += norm_pdf
    return xx, yy, zz


def circular_density_1d(alpha, kappa, bins, bound, w=None):
    alpha_ax = np.linspace(bound[0], bound[1], num=bins)
    density = np.zeros(alpha_ax.shape[0])
    if w is None:
        w = np.ones(alpha.shape[0])
    for id, val in enumerate(alpha):
        if np.isnan(w[id]):
            continue
        density += w[id] * vonmises.pdf(alpha_ax, kappa, loc=val, scale=1)
    total = np.sum(density)
    if total > 0:
        density = density / np.sum(density)
    return alpha_ax, density


def linear_density_1d(x, std, bins, bound):
    x_ax = np.linspace(bound[0], bound[1], num=bins)
    density = np.zeros(x_ax.shape[0])
    for val in x:
        density += norm.pdf(x_ax, loc=val, scale=std)
    density = density / np.sum(density)
    return x_ax, density


def linear_circular_gauss_density(x_lin, y_cir, cir_kappa=8 * np.pi, lin_std=0.025, xbins=300, ybins=1200,
                                  xbound=(0, 1), ybound=(-np.pi, 2 * np.pi)):
    '''

    Parameters
    ----------
    x_lin : ndarray
        1D array of linear variables.
    y_cir : ndarray
        1D array of circular variables with same shape as x_lin
    cir_kappa : scalar
        Concentration of Von Mise distribotion.
    lin_std : scalar
        Standard deviation of linear Gaussian distribution
    xbins : int
        Number of bins for x axis
    ybins : int
        Number of bins for y axis
    xbound : tuple
        (lower_bound, upper_bounad) of x axis
    ybound : tuple
        (lower_bound, upper_bounad) of y axis
    Returns
    -------
    xx : ndarray
        x-meshgrid. 2D array with shape (xbins, ybins).
    yy : ndarray
        y-meshgrid. 2D array with shape (xbins, ybins).
    zz : ndarray
        Density. 2D array with shape (xbins, ybins).
    '''

    num_points = x_lin.shape[0]

    xaxis = np.linspace(xbound[0], xbound[1], xbins)
    yaxis = np.linspace(ybound[0], ybound[1], ybins)
    xx, yy = np.meshgrid(xaxis, yaxis)
    zz = np.zeros(xx.shape)

    import pdb

    for npt in range(num_points):
        x_each, y_each = x_lin[npt], y_cir[npt]
        y_cir_pdf = vonmises.pdf(yy, cir_kappa, loc=y_each, scale=1)
        x_lin_pdf = norm.pdf(xx, loc=x_each, scale=lin_std)
        if np.max(y_cir_pdf) != 0:
            y_cir_pdf = y_cir_pdf / np.max(y_cir_pdf)
        if np.max(x_lin_pdf) != 0:
            x_lin_pdf = x_lin_pdf / np.max(x_lin_pdf)
        norm_pdf = y_cir_pdf * x_lin_pdf
        if np.sum(norm_pdf) != 0:
            norm_pdf = norm_pdf / np.sum(norm_pdf)
        zz += norm_pdf
    return xx, yy, zz


def linear_gauss_density(x, y, xstd=0.025, ystd=0.025, xbins=300, ybins=300, xbound=None, ybound=None):
    '''
    Args:
        x (numpy.darray) : 1-d array of linear variables
        y (numpy.darray) : 1-d array of linear variables
        std (float) : Standard deviation of linear Gaussian distribution
        xbins (int) : Number of bins for x axis
        ybins (int) : Number of bins for y axis
        
    '''
    if xbound is None:
        xbound = (x.min(), x.max())
    if ybound is None:
        ybound = (y.min(), y.max())

    num_points = x.shape[0]
    xaxis = np.linspace(xbound[0], xbound[1], xbins)
    yaxis = np.linspace(ybound[0], ybound[1], ybins)
    xx, yy = np.meshgrid(xaxis, yaxis)
    zz = np.zeros(xx.shape)

    for npt in range(num_points):
        x_each, y_each = x[npt], y[npt]
        x_pdf = norm.pdf(xx, loc=x_each, scale=xstd)
        x_pdf = x_pdf / np.max(x_pdf)

        y_pdf = norm.pdf(yy, loc=y_each, scale=ystd)
        y_pdf = y_pdf / np.max(y_pdf)

        norm_pdf = y_pdf * x_pdf
        norm_pdf = norm_pdf / np.sum(norm_pdf)
        zz += norm_pdf
    return xx, yy, zz


def find_pair_angles_withintheta(tsp1_list, spike1_angles_list, tsp2_list, spike2_angles_list):
    if (len(tsp1_list) == 0) or (len(spike1_angles_list) == 0) or (len(tsp2_list) == 0) or (
            len(spike2_angles_list) == 0):
        return np.array([])
    tsp1_all = np.concatenate(tsp1_list)
    tsp2_all = np.concatenate(tsp2_list)
    s1angles_all = np.concatenate(spike1_angles_list)
    s2angles_all = np.concatenate(spike2_angles_list)
    pair_angles_list = []

    for idx1 in range(tsp1_all.shape[0]):
        within_cycle_idx = np.where(np.abs(tsp1_all[idx1] - tsp2_all) < 0.08)[0]
        s2angles_within = s2angles_all[within_cycle_idx]
        if s2angles_within.shape[0] > 0:
            pair_angles_list.append(s2angles_within)

    for idx2 in range(tsp2_all.shape[0]):
        within_cycle_idx = np.where(np.abs(tsp2_all[idx2] - tsp1_all) < 0.08)[0]
        s1angles_within = s1angles_all[within_cycle_idx]
        if s1angles_within.shape[0] > 0:
            pair_angles_list.append(s1angles_within)
    if len(pair_angles_list) < 1:
        pair_angles = np.array([])
    else:
        pair_angles = np.concatenate(pair_angles_list)
    return pair_angles


def find_pair_angles_np(tsp1, spike1angles, tsp2, spike2angles):
    pair_angles_list = []

    for idx1 in range(tsp1.shape[0]):
        within_cycle_idx = np.where(np.abs(tsp1[idx1] - tsp2) < 0.08)[0]
        s2angles_within = spike2angles[within_cycle_idx]
        if s2angles_within.shape[0] > 0:
            pair_angles_list.append(s2angles_within)

    for idx2 in range(tsp2.shape[0]):
        within_cycle_idx = np.where(np.abs(tsp2[idx2] - tsp1) < 0.08)[0]
        s1angles_within = spike1angles[within_cycle_idx]
        if s1angles_within.shape[0] > 0:
            pair_angles_list.append(s1angles_within)
    if len(pair_angles_list) < 1:
        pair_angles = np.array([])
    else:
        pair_angles = np.concatenate(pair_angles_list)
    return pair_angles




def find_pair_times(tsp1, tsp2, tdiff=0.08):
    tsp1_idx = []
    tsp2_idx = []
    for idx1 in range(tsp1.shape[0]):
        spiketime1 = tsp1[idx1]
        within_cycle_idx = np.where(np.abs(spiketime1 - tsp2) < tdiff)[0]
        if within_cycle_idx.shape[0] > 0:
            tsp2_idx += list(within_cycle_idx)
    for idx2 in range(tsp2.shape[0]):
        spiketime2 = tsp2[idx2]
        within_cycle_idx = np.where(np.abs(spiketime2 - tsp1) < tdiff)[0]
        if within_cycle_idx.shape[0] > 0:
            tsp1_idx += list(within_cycle_idx)
    tsp1_idx, tsp2_idx = np.unique(tsp1_idx).astype(int), np.unique(tsp2_idx).astype(int)
    return tsp1_idx, tsp2_idx


def corr_cc(alpha1, alpha2):
    if (len(alpha1.shape) > 1) or (len(alpha2.shape) > 1):
        raise TypeError('alpha 1 and 2 must be one-dimensional numpy array.')
    if alpha1.shape[0] != alpha2.shape[0]:
        raise ValueError('alpha 1 and 2 must have the same size.')

    n = alpha1.shape[0]
    alpha1_bar = circmean(alpha1)
    alpha2_bar = circmean(alpha2)

    num = np.sum(np.sin(alpha1 - alpha1_bar) * np.sin(alpha2 - alpha2_bar))
    den = np.sqrt(np.sum(np.square(np.sin(alpha1 - alpha1_bar))) * np.sum(np.square(np.sin(alpha2 - alpha2_bar))))
    rho = num / den

    l20 = np.mean(np.square(np.sin(alpha1 - alpha1_bar)))
    l02 = np.mean(np.square(np.sin(alpha2 - alpha2_bar)))
    l22 = np.mean(np.square(np.sin(alpha1 - alpha1_bar)) * np.square(np.sin(alpha2 - alpha2_bar)))
    ts = np.sqrt((n * l20 * l02) / l22) * rho
    pval = 2 * (1 - norm.cdf(np.abs(ts)))
    return rho, pval


def pair_dist_self(x):
    return np.matmul(x.reshape(x.shape[0], 1), np.ones((1, x.shape[0]))) - np.matmul(np.ones((x.shape[0], 1)),
                                                                                     x.reshape(1, x.shape[0]))


def pair_diff(a, b):
    """
    Calculate a - b pairwise.
    Parameters
    ----------
    a : ndarray
        Shape (N, )
    b : ndarray
        Shape (M, )

    Returns
    -------
    c : ndarray
        Shape (N, M). Expand a along rows, minus b for each extended column.
    """

    return a.reshape(a.shape[0], 1) - b.reshape(1, b.shape[0])


def heading(pos):
    """
    Args:
        pos (numpy.ndarray): shape (N, 2). Samples of x, y coordinates
    Returns:
        hd (numpy.ndarray): shape (N, ). Samples of heading angles. [-pi, pi)
    """
    dpos = pos[1:, :] - pos[:-1, :]
    hd = np.angle(dpos[:, 0] + 1j * dpos[:, 1])
    hd = np.append(hd, hd[-1])
    return hd


class IndataProcessor:
    def __init__(self, indata, vthresh=5, sthresh=3, minpasstime=0.4, smooth=None):
        self.x, self.y, self.t = np.array(indata['x']), np.array(indata['y']), np.array(indata['t'])
        self.sthresh, self.vthresh = sthresh, vthresh
        self.smooth = smooth
        self.minpasstime = minpasstime
        self.speed, self.angle = self._get_speed_angles()
        self.speedmask_idx = np.where(self.speed < self.vthresh)[0]
        self.v_interp = interp1d(self.t, self.speed)

    def get_idin(self, mask, xaxis, yaxis):
        ind_x = np.argmin(np.square(pair_diff(self.x, xaxis)), axis=1)
        ind_y = np.argmin(np.square(pair_diff(self.y, yaxis)), axis=1)
        tok = np.zeros(ind_x.shape[0]).astype(bool)
        for i in range(ind_x.shape[0]):
            tok[i] = mask[ind_y[i], ind_x[i]]  # Double check for the order of ind_y and ind_x
        idin = np.where(tok)[0]
        return tok, idin

    def construct_singlefield_passdf(self, tok, tsp, interp_x, interp_y, interp_angle):

        pass_dict = dict(x=[], y=[], t=[], v=[], angle=[], tsp=[], spikex=[], spikey=[], spikeangle=[],
                         straightrank=[], chunked=[], rejected=[])
        all_passidx = segment_passes(tok)

        for pid1, pid2 in all_passidx:
            # segment by speed
            belowspeed_idx = self.speedmask_idx[(self.speedmask_idx >= pid1) & (self.speedmask_idx < pid2)]
            if belowspeed_idx.shape[0] >= 1:  # If there is at least one split (based on speed)
                all_segmentidx = [pid1] + list(belowspeed_idx) + [pid2]
                for j in range(len(all_segmentidx)-1):
                    chunkid = j+1
                    jid1, jid2 = all_segmentidx[j], all_segmentidx[j+1]
                    x_in = self.x[jid1:jid2]
                    y_in = self.y[jid1:jid2]
                    t_in = self.t[jid1:jid2]
                    v_in = self.speed[jid1:jid2]
                    angle_in = self.angle[jid1:jid2]

                    if t_in.shape[0] < 3:
                        continue

                    rejected, (tsp_in, straightrank) = self.rejection_singlefield(tsp, t_in, angle_in)

                    pass_dict['x'].append(x_in)
                    pass_dict['y'].append(y_in)
                    pass_dict['t'].append(t_in)
                    pass_dict['v'].append(v_in)
                    pass_dict['angle'].append(angle_in)
                    pass_dict['tsp'].append(tsp_in)
                    pass_dict['spikex'].append(interp_x(tsp_in))
                    pass_dict['spikey'].append(interp_y(tsp_in))
                    pass_dict['spikeangle'].append(interp_angle(tsp_in))
                    pass_dict['straightrank'].append(straightrank)
                    pass_dict['chunked'].append(chunkid)
                    pass_dict['rejected'].append(rejected)


            else:  # If all speed is above threshold and no further spilting is needed

                chunkid = 0
                x_in = self.x[pid1:pid2]
                y_in = self.y[pid1:pid2]
                t_in = self.t[pid1:pid2]
                v_in = self.speed[pid1:pid2]
                angle_in = self.angle[pid1:pid2]

                if t_in.shape[0] < 3:
                    continue

                rejected, (tsp_in, straightrank) = self.rejection_singlefield(tsp, t_in, angle_in)

                pass_dict['x'].append(x_in)
                pass_dict['y'].append(y_in)
                pass_dict['t'].append(t_in)
                pass_dict['v'].append(v_in)
                pass_dict['angle'].append(angle_in)
                pass_dict['tsp'].append(tsp_in)
                pass_dict['spikex'].append(interp_x(tsp_in))
                pass_dict['spikey'].append(interp_y(tsp_in))
                pass_dict['spikeangle'].append(interp_angle(tsp_in))
                pass_dict['straightrank'].append(straightrank)
                pass_dict['chunked'].append(chunkid)
                pass_dict['rejected'].append(rejected)
        pass_df = pd.DataFrame(pass_dict)
        return pass_df

    def construct_singlefield_passdf_all(self, tok, tsp, interp_x, interp_y, interp_angle):

        pass_dict = dict(x=[], y=[], t=[], v=[], angle=[], tsp=[], spikex=[], spikey=[], spikeangle=[],
                         straightrank=[], chunked=[], duration=[], num_spikes=[])
        all_passidx = segment_passes(tok)

        for pid1, pid2 in all_passidx:
            # segment by speed
            belowspeed_idx = self.speedmask_idx[(self.speedmask_idx >= pid1) & (self.speedmask_idx < pid2)]
            if belowspeed_idx.shape[0] >= 1:  # If there is at least one split (based on speed)
                all_segmentidx = [pid1] + list(belowspeed_idx) + [pid2]
                for j in range(len(all_segmentidx)-1):
                    chunkid = j+1
                    jid1, jid2 = all_segmentidx[j], all_segmentidx[j+1]
                    x_in = self.x[jid1:jid2]
                    y_in = self.y[jid1:jid2]
                    t_in = self.t[jid1:jid2]
                    v_in = self.speed[jid1:jid2]
                    angle_in = self.angle[jid1:jid2]

                    if t_in.shape[0] < 2:
                        continue

                    # Spike counts
                    tsp_in = tsp[(tsp < t_in.max()) & (tsp >= t_in.min())]
                    num_spikes = tsp_in.shape[0]

                    # Straightness constraint
                    straightrank = compute_straightness(angle_in)

                    # Minimum duration constraint
                    duration = t_in.max() - t_in.min()

                    pass_dict['x'].append(x_in)
                    pass_dict['y'].append(y_in)
                    pass_dict['t'].append(t_in)
                    pass_dict['v'].append(v_in)
                    pass_dict['angle'].append(angle_in)
                    pass_dict['tsp'].append(tsp_in)
                    pass_dict['spikex'].append(interp_x(tsp_in))
                    pass_dict['spikey'].append(interp_y(tsp_in))
                    pass_dict['spikeangle'].append(interp_angle(tsp_in))
                    pass_dict['straightrank'].append(straightrank)
                    pass_dict['chunked'].append(chunkid)
                    pass_dict['num_spikes'].append(num_spikes)
                    pass_dict['duration'].append(duration)


            else:  # If all speed is above threshold and no further spilting is needed

                chunkid = 0
                x_in = self.x[pid1:pid2]
                y_in = self.y[pid1:pid2]
                t_in = self.t[pid1:pid2]
                v_in = self.speed[pid1:pid2]
                angle_in = self.angle[pid1:pid2]

                if t_in.shape[0] < 3:
                    continue

                # Spike counts
                tsp_in = tsp[(tsp < t_in.max()) & (tsp >= t_in.min())]
                num_spikes = tsp_in.shape[0]

                # Straightness constraint
                straightrank = compute_straightness(angle_in)

                # Minimum duration constraint
                duration = t_in.max() - t_in.min()

                pass_dict['x'].append(x_in)
                pass_dict['y'].append(y_in)
                pass_dict['t'].append(t_in)
                pass_dict['v'].append(v_in)
                pass_dict['angle'].append(angle_in)
                pass_dict['tsp'].append(tsp_in)
                pass_dict['spikex'].append(interp_x(tsp_in))
                pass_dict['spikey'].append(interp_y(tsp_in))
                pass_dict['spikeangle'].append(interp_angle(tsp_in))
                pass_dict['straightrank'].append(straightrank)
                pass_dict['chunked'].append(chunkid)
                pass_dict['num_spikes'].append(num_spikes)
                pass_dict['duration'].append(duration)

        pass_df = pd.DataFrame(pass_dict)
        return pass_df


    def construct_pairfield_passdf(self, tok_pair, tok1, tok2, tsp1, tsp2, interp_x, interp_y, interp_angle):

        pass_dict = dict(x=[], y=[], t=[], v=[], angle=[], passidx=[], straightrank=[], infield1=[], infield2=[],
                         chunked=[], direction=[], rejected=[],
                         tsp1=[], spike1x=[], spike1y=[], spike1angle=[],
                         tsp2=[], spike2x=[], spike2y=[], spike2angle=[])
        all_passidx = segment_passes(tok_pair)
        for pid1, pid2 in all_passidx:
            # segment by speed
            belowspeed_idx = self.speedmask_idx[(self.speedmask_idx >= pid1) & (self.speedmask_idx < pid2)]
            if belowspeed_idx.shape[0] >= 1:  # If there is at least one split (based on speed)
                all_segmentidx = [pid1] + list(belowspeed_idx) + [pid2]
                for j in range(len(all_segmentidx)-1):

                    jid1, jid2 = all_segmentidx[j], all_segmentidx[j+1]
                    chunkid = j+1
                    rejected = False

                    x_in = self.x[jid1:jid2]
                    y_in = self.y[jid1:jid2]
                    t_in = self.t[jid1:jid2]
                    v_in = self.speed[jid1:jid2]
                    angle_in = self.angle[jid1:jid2]

                    if t_in.shape[0] < 3:
                        continue

                    # Spike count threshold, at least 2
                    tsp1_in = tsp1[(tsp1 < t_in.max()) & (tsp1 >= t_in.min())]
                    tsp2_in = tsp2[(tsp2 < t_in.max()) & (tsp2 >= t_in.min())]

                    if (tsp1_in.shape[0] < 2) and ((tsp2_in.shape[0] < 2)):
                        rejected = True

                    # Straightness constraint
                    straightrank = compute_straightness(angle_in)
                    if straightrank < self.sthresh:
                        rejected = True

                    # Minimum duration constraint
                    if (t_in.shape[0] < 5) or ((t_in[-1]-t_in[0]) < self.minpasstime):
                        rejected = True

                    # Border crossing
                    infield1 = tok1[jid1:jid2]
                    infield2 = tok2[jid1:jid2]
                    loc, direction = ThetaEstimator.find_direction(infield1, infield2)
                    cross1 = np.sum(np.abs(np.diff(infield1)))
                    cross2 = np.sum(np.abs(np.diff(infield2)))
                    if (cross1 > 1) or (cross2 > 1):
                        direction = None


                    pass_dict['x'].append(x_in)
                    pass_dict['y'].append(y_in)
                    pass_dict['t'].append(t_in)
                    pass_dict['v'].append(v_in)
                    pass_dict['angle'].append(angle_in)
                    pass_dict['passidx'].append((jid1, jid2))
                    pass_dict['straightrank'].append(straightrank)
                    pass_dict['infield1'].append(infield1)
                    pass_dict['infield2'].append(infield2)
                    pass_dict['chunked'].append(chunkid)
                    pass_dict['direction'].append(direction)
                    pass_dict['rejected'].append(rejected)


                    pass_dict['tsp1'].append(tsp1_in)
                    pass_dict['spike1x'].append(interp_x(tsp1_in))
                    pass_dict['spike1y'].append(interp_y(tsp1_in))
                    pass_dict['spike1angle'].append(interp_angle(tsp1_in))
                    pass_dict['tsp2'].append(tsp2_in)
                    pass_dict['spike2x'].append(interp_x(tsp2_in))
                    pass_dict['spike2y'].append(interp_y(tsp2_in))
                    pass_dict['spike2angle'].append(interp_angle(tsp2_in))


            else:  # If all speed is above threshold and no further spilting is needed
                rejected = False
                chunkid = 0
                x_in = self.x[pid1:pid2]
                y_in = self.y[pid1:pid2]
                t_in = self.t[pid1:pid2]
                v_in = self.speed[pid1:pid2]
                angle_in = self.angle[pid1:pid2]

                if t_in.shape[0] < 3:
                    continue

                # Spike count threshold, at least 2
                tsp1_in = tsp1[(tsp1 < t_in.max()) & (tsp1 >= t_in.min())]
                tsp2_in = tsp2[(tsp2 < t_in.max()) & (tsp2 >= t_in.min())]

                if (tsp1_in.shape[0] < 2) and ((tsp2_in.shape[0] < 2)):
                    rejected = True

                # Straightness constraint
                straightrank = compute_straightness(angle_in)
                if straightrank < self.sthresh:
                    rejected = True

                # Minimum duration constraint
                if (t_in.shape[0] < 5) or ((t_in[-1]-t_in[0]) < self.minpasstime):
                    rejected = True

                # Border crossing
                infield1 = tok1[pid1:pid2]
                infield2 = tok2[pid1:pid2]
                loc, direction = ThetaEstimator.find_direction(infield1, infield2)
                cross1 = np.sum(np.abs(np.diff(infield1)))
                cross2 = np.sum(np.abs(np.diff(infield2)))
                if (cross1 > 1) or (cross2 > 1):
                    rejected = True


                pass_dict['x'].append(x_in)
                pass_dict['y'].append(y_in)
                pass_dict['t'].append(t_in)
                pass_dict['v'].append(v_in)
                pass_dict['angle'].append(angle_in)
                pass_dict['passidx'].append((pid1, pid2))
                pass_dict['straightrank'].append(straightrank)
                pass_dict['infield1'].append(infield1)
                pass_dict['infield2'].append(infield2)
                pass_dict['chunked'].append(chunkid)
                pass_dict['direction'].append(direction)
                pass_dict['rejected'].append(rejected)

                pass_dict['tsp1'].append(tsp1_in)
                pass_dict['spike1x'].append(interp_x(tsp1_in))
                pass_dict['spike1y'].append(interp_y(tsp1_in))
                pass_dict['spike1angle'].append(interp_angle(tsp1_in))
                pass_dict['tsp2'].append(tsp2_in)
                pass_dict['spike2x'].append(interp_x(tsp2_in))
                pass_dict['spike2y'].append(interp_y(tsp2_in))
                pass_dict['spike2angle'].append(interp_angle(tsp2_in))
        pass_df = pd.DataFrame(pass_dict)
        return pass_df



    def construct_pairfield_passdf(self, tok_pair, tok1, tok2, tsp1, tsp2, interp_x, interp_y, interp_angle):

        pass_dict = dict(x=[], y=[], t=[], v=[], angle=[], passidx=[], straightrank=[], infield1=[], infield2=[],
                         chunked=[], direction=[], rejected=[],
                         tsp1=[], spike1x=[], spike1y=[], spike1angle=[],
                         tsp2=[], spike2x=[], spike2y=[], spike2angle=[])
        all_passidx = segment_passes(tok_pair)
        for pid1, pid2 in all_passidx:
            # segment by speed
            belowspeed_idx = self.speedmask_idx[(self.speedmask_idx >= pid1) & (self.speedmask_idx < pid2)]
            if belowspeed_idx.shape[0] >= 1:  # If there is at least one split (based on speed)
                all_segmentidx = [pid1] + list(belowspeed_idx) + [pid2]
                for j in range(len(all_segmentidx)-1):

                    jid1, jid2 = all_segmentidx[j], all_segmentidx[j+1]
                    chunkid = j+1
                    rejected = False

                    x_in = self.x[jid1:jid2]
                    y_in = self.y[jid1:jid2]
                    t_in = self.t[jid1:jid2]
                    v_in = self.speed[jid1:jid2]
                    angle_in = self.angle[jid1:jid2]

                    if t_in.shape[0] < 3:
                        continue

                    # Spike count threshold, at least 2
                    tsp1_in = tsp1[(tsp1 < t_in.max()) & (tsp1 >= t_in.min())]
                    tsp2_in = tsp2[(tsp2 < t_in.max()) & (tsp2 >= t_in.min())]

                    if (tsp1_in.shape[0] < 2) and ((tsp2_in.shape[0] < 2)):
                        rejected = True

                    # Straightness constraint
                    straightrank = compute_straightness(angle_in)
                    if straightrank < self.sthresh:
                        rejected = True

                    # Minimum duration constraint
                    if (t_in.shape[0] < 5) or ((t_in[-1]-t_in[0]) < self.minpasstime):
                        rejected = True

                    # Border crossing
                    infield1 = tok1[jid1:jid2]
                    infield2 = tok2[jid1:jid2]
                    loc, direction = ThetaEstimator.find_direction(infield1, infield2)
                    cross1 = np.sum(np.abs(np.diff(infield1)))
                    cross2 = np.sum(np.abs(np.diff(infield2)))
                    if (cross1 > 1) or (cross2 > 1):
                        direction = None


                    pass_dict['x'].append(x_in)
                    pass_dict['y'].append(y_in)
                    pass_dict['t'].append(t_in)
                    pass_dict['v'].append(v_in)
                    pass_dict['angle'].append(angle_in)
                    pass_dict['passidx'].append((jid1, jid2))
                    pass_dict['straightrank'].append(straightrank)
                    pass_dict['infield1'].append(infield1)
                    pass_dict['infield2'].append(infield2)
                    pass_dict['chunked'].append(chunkid)
                    pass_dict['direction'].append(direction)
                    pass_dict['rejected'].append(rejected)


                    pass_dict['tsp1'].append(tsp1_in)
                    pass_dict['spike1x'].append(interp_x(tsp1_in))
                    pass_dict['spike1y'].append(interp_y(tsp1_in))
                    pass_dict['spike1angle'].append(interp_angle(tsp1_in))
                    pass_dict['tsp2'].append(tsp2_in)
                    pass_dict['spike2x'].append(interp_x(tsp2_in))
                    pass_dict['spike2y'].append(interp_y(tsp2_in))
                    pass_dict['spike2angle'].append(interp_angle(tsp2_in))


            else:  # If all speed is above threshold and no further spilting is needed
                rejected = False
                chunkid = 0
                x_in = self.x[pid1:pid2]
                y_in = self.y[pid1:pid2]
                t_in = self.t[pid1:pid2]
                v_in = self.speed[pid1:pid2]
                angle_in = self.angle[pid1:pid2]

                if t_in.shape[0] < 3:
                    continue

                # Spike count threshold, at least 2
                tsp1_in = tsp1[(tsp1 < t_in.max()) & (tsp1 >= t_in.min())]
                tsp2_in = tsp2[(tsp2 < t_in.max()) & (tsp2 >= t_in.min())]

                if (tsp1_in.shape[0] < 2) and ((tsp2_in.shape[0] < 2)):
                    rejected = True

                # Straightness constraint
                straightrank = compute_straightness(angle_in)
                if straightrank < self.sthresh:
                    rejected = True

                # Minimum duration constraint
                if (t_in.shape[0] < 5) or ((t_in[-1]-t_in[0]) < self.minpasstime):
                    rejected = True

                # Border crossing
                infield1 = tok1[pid1:pid2]
                infield2 = tok2[pid1:pid2]
                loc, direction = ThetaEstimator.find_direction(infield1, infield2)
                cross1 = np.sum(np.abs(np.diff(infield1)))
                cross2 = np.sum(np.abs(np.diff(infield2)))
                if (cross1 > 1) or (cross2 > 1):
                    rejected = True


                pass_dict['x'].append(x_in)
                pass_dict['y'].append(y_in)
                pass_dict['t'].append(t_in)
                pass_dict['v'].append(v_in)
                pass_dict['angle'].append(angle_in)
                pass_dict['passidx'].append((pid1, pid2))
                pass_dict['straightrank'].append(straightrank)
                pass_dict['infield1'].append(infield1)
                pass_dict['infield2'].append(infield2)
                pass_dict['chunked'].append(chunkid)
                pass_dict['direction'].append(direction)
                pass_dict['rejected'].append(rejected)

                pass_dict['tsp1'].append(tsp1_in)
                pass_dict['spike1x'].append(interp_x(tsp1_in))
                pass_dict['spike1y'].append(interp_y(tsp1_in))
                pass_dict['spike1angle'].append(interp_angle(tsp1_in))
                pass_dict['tsp2'].append(tsp2_in)
                pass_dict['spike2x'].append(interp_x(tsp2_in))
                pass_dict['spike2y'].append(interp_y(tsp2_in))
                pass_dict['spike2angle'].append(interp_angle(tsp2_in))
        pass_df = pd.DataFrame(pass_dict)
        return pass_df


    def construct_pairfield_passdf_all(self, tok_pair, tok1, tok2, tsp1, tsp2, interp_x, interp_y, interp_angle):

        pass_dict = dict(x=[], y=[], t=[], v=[], angle=[], passidx=[], straightrank=[],
                         chunked=[], direction=[], duration=[],
                         tsp1=[], spike1x=[], spike1y=[], spike1angle=[], num_spikes1=[], infield1=[],
                         tsp2=[], spike2x=[], spike2y=[], spike2angle=[], num_spikes2=[], infield2=[],
                         tsp_pair=[], num_spikes_pair=[])
        all_passidx = segment_passes(tok_pair)
        for pid1, pid2 in all_passidx:
            # segment by speed
            belowspeed_idx = self.speedmask_idx[(self.speedmask_idx >= pid1) & (self.speedmask_idx < pid2)]
            if belowspeed_idx.shape[0] >= 1:  # If there is at least one split (based on speed)
                all_segmentidx = [pid1] + list(belowspeed_idx) + [pid2]
                for j in range(len(all_segmentidx)-1):

                    jid1, jid2 = all_segmentidx[j], all_segmentidx[j+1]
                    chunkid = j+1

                    x_in = self.x[jid1:jid2]
                    y_in = self.y[jid1:jid2]
                    t_in = self.t[jid1:jid2]
                    v_in = self.speed[jid1:jid2]
                    angle_in = self.angle[jid1:jid2]

                    if t_in.shape[0] < 2:
                        continue

                    # Spike count
                    tsp1_in = tsp1[(tsp1 < t_in.max()) & (tsp1 >= t_in.min())]
                    tsp2_in = tsp2[(tsp2 < t_in.max()) & (tsp2 >= t_in.min())]
                    num_spikes1 = tsp1_in.shape[0]
                    num_spikes2 = tsp2_in.shape[0]

                    # Paired spikes
                    tsp1_idx, tsp2_idx = find_pair_times(tsp1_in, tsp2_in)
                    tsp_pair = np.concatenate([tsp1_in[tsp1_idx], tsp2_in[tsp2_idx]])
                    num_spikes_pair = tsp_pair.shape[0]


                    # Straightness constraint
                    straightrank = compute_straightness(angle_in)

                    # Minimum duration constraint
                    duration = t_in.max() - t_in.min()


                    # Border crossing
                    infield1 = tok1[jid1:jid2]
                    infield2 = tok2[jid1:jid2]
                    loc, direction = ThetaEstimator.find_direction(infield1, infield2)
                    cross1 = np.sum(np.abs(np.diff(infield1)))
                    cross2 = np.sum(np.abs(np.diff(infield2)))
                    if (cross1 > 1) or (cross2 > 1):
                        direction = None


                    pass_dict['x'].append(x_in)
                    pass_dict['y'].append(y_in)
                    pass_dict['t'].append(t_in)
                    pass_dict['v'].append(v_in)
                    pass_dict['angle'].append(angle_in)
                    pass_dict['passidx'].append((jid1, jid2))
                    pass_dict['straightrank'].append(straightrank)
                    pass_dict['chunked'].append(chunkid)
                    pass_dict['direction'].append(direction)
                    pass_dict['duration'].append(duration)

                    pass_dict['tsp1'].append(tsp1_in)
                    pass_dict['spike1x'].append(interp_x(tsp1_in))
                    pass_dict['spike1y'].append(interp_y(tsp1_in))
                    pass_dict['spike1angle'].append(interp_angle(tsp1_in))
                    pass_dict['num_spikes1'].append(num_spikes1)
                    pass_dict['infield1'].append(infield1)

                    pass_dict['tsp2'].append(tsp2_in)
                    pass_dict['spike2x'].append(interp_x(tsp2_in))
                    pass_dict['spike2y'].append(interp_y(tsp2_in))
                    pass_dict['spike2angle'].append(interp_angle(tsp2_in))
                    pass_dict['num_spikes2'].append(num_spikes2)
                    pass_dict['infield2'].append(infield2)

                    pass_dict['tsp_pair'].append(tsp_pair)
                    pass_dict['num_spikes_pair'].append(num_spikes_pair)

            else:  # If all speed is above threshold and no further spilting is needed
                chunkid = 0
                x_in = self.x[pid1:pid2]
                y_in = self.y[pid1:pid2]
                t_in = self.t[pid1:pid2]
                v_in = self.speed[pid1:pid2]
                angle_in = self.angle[pid1:pid2]

                if t_in.shape[0] < 2:
                    continue

                # Spike count threshold
                tsp1_in = tsp1[(tsp1 < t_in.max()) & (tsp1 >= t_in.min())]
                tsp2_in = tsp2[(tsp2 < t_in.max()) & (tsp2 >= t_in.min())]
                num_spikes1 = tsp1_in.shape[0]
                num_spikes2 = tsp2_in.shape[0]

                # Paired spikes
                tsp1_idx, tsp2_idx = find_pair_times(tsp1_in, tsp2_in)
                tsp_pair = np.concatenate([tsp1_in[tsp1_idx], tsp2_in[tsp2_idx]])
                num_spikes_pair = tsp_pair.shape[0]

                # Straightness constraint
                straightrank = compute_straightness(angle_in)

                # Minimum duration constraint
                duration = t_in.max() - t_in.min()


                # Border crossing
                infield1 = tok1[pid1:pid2]
                infield2 = tok2[pid1:pid2]
                loc, direction = ThetaEstimator.find_direction(infield1, infield2)
                cross1 = np.sum(np.abs(np.diff(infield1)))
                cross2 = np.sum(np.abs(np.diff(infield2)))
                if (cross1 > 1) or (cross2 > 1):
                    direction = None


                pass_dict['x'].append(x_in)
                pass_dict['y'].append(y_in)
                pass_dict['t'].append(t_in)
                pass_dict['v'].append(v_in)
                pass_dict['angle'].append(angle_in)
                pass_dict['passidx'].append((pid1, pid2))
                pass_dict['straightrank'].append(straightrank)
                pass_dict['chunked'].append(chunkid)
                pass_dict['direction'].append(direction)
                pass_dict['duration'].append(duration)

                pass_dict['tsp1'].append(tsp1_in)
                pass_dict['spike1x'].append(interp_x(tsp1_in))
                pass_dict['spike1y'].append(interp_y(tsp1_in))
                pass_dict['spike1angle'].append(interp_angle(tsp1_in))
                pass_dict['num_spikes1'].append(num_spikes1)
                pass_dict['infield1'].append(infield1)

                pass_dict['tsp2'].append(tsp2_in)
                pass_dict['spike2x'].append(interp_x(tsp2_in))
                pass_dict['spike2y'].append(interp_y(tsp2_in))
                pass_dict['spike2angle'].append(interp_angle(tsp2_in))
                pass_dict['num_spikes2'].append(num_spikes2)
                pass_dict['infield2'].append(infield2)

                pass_dict['tsp_pair'].append(tsp_pair)
                pass_dict['num_spikes_pair'].append(num_spikes_pair)
        pass_df = pd.DataFrame(pass_dict)
        return pass_df


    def rejection_singlefield(self, tsp, t_in, angle_in):
        rejected = False

        # Spike counts
        tsp_in = tsp[(tsp < t_in.max()) & (tsp >= t_in.min())]
        if tsp_in.shape[0] < 2:
            rejected = True

        # Straightness constraint
        straightrank = compute_straightness(angle_in)
        if straightrank < self.sthresh:
            rejected = True

        # Minimum duration constraint
        if (t_in.shape[0] < 5) or ((t_in[-1]-t_in[0]) < self.minpasstime):
            rejected = True
        return rejected, (tsp_in, straightrank)



    def _get_speed_angles(self):
        dx = self.x[1:] - self.x[:-1]
        dy = self.y[1:] - self.y[:-1]
        dt = self.t[1:] - self.t[:-1]
        if self.smooth is not None:
            dx = np.convolve(dx, self.smooth, mode='same')
            dy = np.convolve(dy, self.smooth, mode='same')
        speed = np.sqrt(dx ** 2 + dy ** 2) / dt
        speed = np.append(speed, speed[-1])
        angle = np.angle(dx + 1j * dy)
        angle = np.append(angle, angle[-1])
        return speed, angle



def normalize_distr(up_distr, down_distr):
    """
    Compute normalized_distr = up_distr / down_distr, with nan/inf correction and warning suppression.

    Parameters
    ----------
    up_distr : ndarray
        1-D array for bins/probability as numerator.
    down_distr : ndarray
        1-D array for bins/probability as denominator.
    Returns
    -------
    norm_prob : ndarray
        1-D array of up_distr/down_distr
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_prob = up_distr / down_distr
        norm_prob[np.isnan(norm_prob)] = 0
        norm_prob[np.isinf(norm_prob)] = 0
        total = np.sum(norm_prob)
        if total > 0:
            norm_prob = norm_prob / total
    return norm_prob





def check_border(mask, margin=2):
    maridx = np.array(list(range(0, margin)) + list(range(-margin, 0)))
    # Check for border
    edges_lr = np.sum(mask[:, maridx])
    edges_tb = np.sum(mask[maridx, :])
    if (edges_lr + edges_tb) > 0:
        border = True
    else:
        border = False
    return border


def check_border_sim(x, y, radius, bound=(0, 2 * np.pi)):
    minx = x - radius
    maxx = x + radius
    miny = y - radius
    maxy = y + radius

    if (minx < bound[0]) or (maxx > bound[1]) or (miny < bound[0]) or (maxy > bound[1]):
        return True
    else:
        return False


def segment_passes(tok):
    nstep = 0
    all_passidx = []
    while nstep < tok.shape[0]:
        trest = tok[nstep:]

        # If it doesn't start inside the field, jump to the start
        if ~trest[0]:
            i0 = np.where(trest == True)[0]
            if i0.shape[0] == 0:  # nstep is inside the field
                break
            else:
                i0 = i0[0]
                nstep = nstep + i0
                trest = tok[nstep:]

        start_idx = nstep

        # Looking for closest end
        ie = np.where(trest == False)[0]
        if ie.shape[0] == 0:  # The rest are all inside the field
            ie = trest.shape[0]
        else:
            ie = ie[0]

        end_idx = nstep + ie
        all_passidx.append((start_idx, end_idx - 1))

        # Update nstep
        nstep = end_idx
    return all_passidx


def window_shuffle(tsp, seed=None, windowsize=0.5):


    # Create time windows
    maxt, mint = tsp.max(), tsp.min()
    tsp_duration = maxt-mint
    if tsp_duration > windowsize:
        raise ValueError('Width of shuffling window has to be larger than the whole spike train.')
    windows = np.arange(mint, maxt, step=windowsize)  # 500ms window
    windows = np.append(windows, windows[-1] + 0.5)

    tsp_windows = []
    win_starts = []
    for wid in range(windows.shape[0] - 1):
        wstart, wend = windows[wid], windows[wid + 1]
        tsp_inside_window = tsp[(tsp < wend) & (tsp >= wstart)]
        if tsp_inside_window.shape[0] < 1:
            continue
        tsp_inside_window = tsp_inside_window - wstart
        win_starts.append(wstart)
        tsp_windows.append(tsp_inside_window)
    if seed is not None:
        random.seed(seed)
    random.shuffle(tsp_windows)
    tsp_shuffled = np.concatenate([tsp_windows[x] + win_starts[x] for x in range(len(tsp_windows))])
    return tsp_shuffled

def window_shuffle_gen(tsp, NShuffles, trange, windowsize=0.5):
    # Create time windows
    all_maxt, all_mint = trange
    maxt, mint = tsp.max(), tsp.min()
    windows = np.arange(mint, maxt, step=windowsize)  # 500ms window
    windows = np.append(windows, windows[-1] + 0.5)

    tsp_windows = []
    win_starts = []
    for wid in range(windows.shape[0] - 1):
        wstart, wend = windows[wid], windows[wid + 1]
        tsp_inside_window = tsp[(tsp < wend) & (tsp >= wstart)]
        if tsp_inside_window.shape[0] < 1:
            continue
        tsp_inside_window = tsp_inside_window - wstart
        win_starts.append(wstart)
        tsp_windows.append(tsp_inside_window)

    # Shuffling windows
    for shufi in range(NShuffles):
        random.seed(shufi)
        random.shuffle(tsp_windows)
        tsp_shuffled = np.concatenate([tsp_windows[x] + win_starts[x] for x in range(len(tsp_windows))])
        tsp_shuffled = tsp_shuffled[(tsp_shuffled < all_maxt) & (tsp_shuffled > all_mint)]
        yield shufi, tsp_shuffled

def window_shuffle_wrapper(tsp, fieldR_mlm, NShuffles, direction_mlmer,
                           interpolater_x, interpolater_y, interpolater_angle, trange):

    all_shuf_fieldR_mlm = np.zeros(NShuffles)
    for shufi, tsp_shuffled in window_shuffle_gen(tsp, NShuffles, trange):
        shuffled_angles = interpolater_angle(tsp_shuffled)
        shuffled_x = interpolater_x(tsp_shuffled)
        shuffled_y = interpolater_y(tsp_shuffled)
        possp = np.stack([shuffled_x, shuffled_y]).T
        _, shuf_fieldR_mlm, _ = direction_mlmer.get_directionality(possp, shuffled_angles)
        all_shuf_fieldR_mlm[shufi] = shuf_fieldR_mlm

    win_pval_mlm = 1 - np.mean(fieldR_mlm > all_shuf_fieldR_mlm)

    return win_pval_mlm


def timeshift_shuffle_exp_wrapper(all_tsp_list, all_t_list, fieldR_mlm, NShuffles,
                                  direction_mlmer, interpolater_x, interpolater_y, interpolater_angle, trange):
    all_maxt, all_mint = trange
    all_shuf_fieldR_mlm = np.zeros(NShuffles)
    shuffler = PassShuffler(all_t_list, all_tsp_list)
    for shufi in range(NShuffles):
        tsp_shuffled = shuffler.timeshift_shuffle(seed=shufi, return_concat=True)
        # tsp_shuffled = passes_spikes_shuffle(all_t_list, all_tsp_list, return_concat=True, seed=shufi)
        tsp_shuffled = tsp_shuffled[(tsp_shuffled < all_maxt) & (tsp_shuffled > all_mint)]
        shuffled_angles = interpolater_angle(tsp_shuffled)
        shuffled_x = interpolater_x(tsp_shuffled)
        shuffled_y = interpolater_y(tsp_shuffled)
        possp = np.stack([shuffled_x, shuffled_y]).T
        _, shuf_fieldR_mlm, _ = direction_mlmer.get_directionality(possp, shuffled_angles)
        all_shuf_fieldR_mlm[shufi] = shuf_fieldR_mlm
    shift_pval_mlm = 1 - np.mean(fieldR_mlm > all_shuf_fieldR_mlm)
    return shift_pval_mlm


def cos_scaled(t):
    y = np.cos(t) + 1
    cos_sum = np.sum(y)
    y[np.abs(t) > np.pi] = 0
    y = y * (cos_sum / np.sum(y))
    y = y - 1
    return y


def cos_scaled_2d(pairdist1, pairdist2):
    '''

    Parameters
    ----------
    pairdist1 : ndarray
        Delta x, with shape (n, ) or (n, m)
    pairdist2 : ndarray
        Delta y, with shape (n, ) or (n, m)
    Returns
    -------
    out : ndarray
        output with shape (n, ) or (n, m)
    '''

    dist = np.sqrt(pairdist1 ** 2 + pairdist2 ** 2)
    out = 2 * cos_scaled(dist)  # multiplied by 2 because it is cos(x)+cos(y) in Romani's paper
    return out


def midedges(e):
    return (e[1:] + e[:-1]) / 2


def linear_transform(x, trans, scaling_const):
    return (x - trans) * scaling_const


def get_transform_params(x, y):
    trans_x = np.min(x)
    trans_y = np.min(y)
    scaling_const = 2 * np.pi / np.max([x - trans_x, y - trans_y])
    return trans_x, trans_y, scaling_const


def append_info_from_passes(passdf, vthresh, sthresh, trange):
    all_maxt, all_mint = trange
    num_passes = passdf.shape[0]
    all_passangles, all_tsp = [], []
    all_x, all_y, all_t = [], [], []
    for npass in range(num_passes):
        pass_angles, v, tsp, vsp = passdf.loc[npass, ['angle', 'v', 'tsp', 'spikev']]
        x, y, t = passdf.loc[npass, ['x', 'y', 't']]


        # Straightness criterion
        straightrank = compute_straightness(pass_angles)
        if straightrank < sthresh:
            continue

        # Speed criterion
        if (vsp.shape[0] != tsp.shape[0]):
            continue
        mask = v > vthresh
        masksp = (vsp > vthresh) & (tsp > all_mint) & (tsp < all_maxt)
        if (mask.sum() < 2) or (masksp.sum() < 2):  # Threshold = 2 because we want both upper & lower boundary.
            continue

        # Concatenate info
        all_tsp.append(tsp[masksp])
        all_t.append(t[mask])
        all_passangles.append(pass_angles[mask])
        all_x.append(x[mask])
        all_y.append(y[mask])

    return (all_x, all_y, all_t, all_passangles), all_tsp



def calc_kld(norm_prob1, norm_prob2, norm_prob_pair):
    # KLD
    indep_prob = norm_prob1 * norm_prob2
    indep_prob = indep_prob / np.sum(indep_prob)
    with np.errstate(divide='ignore', invalid='ignore'):  # division by 0 expected
        logscale = np.log(norm_prob_pair / indep_prob)
        logscale[np.isnan(logscale)] = 0  # setting as 0 is equivalent to ignoring the data point
        logscale[np.isinf(logscale)] = 0
        kld = np.nansum(norm_prob_pair * logscale)
    return kld


def unfold_binning_2d(bins, edgesm1, edgesm2):
    bins = np.around(bins).astype(int)
    all_x = []
    all_y = []
    for id1, ed1 in enumerate(edgesm1):
        for id2, ed2 in enumerate(edgesm2):
            times = bins[id1, id2]
            if times == 0:
                continue
            all_x.append(np.array([ed1] * times))
            all_y.append(np.array([ed2] * times))
    all_x_np = np.concatenate(all_x)
    all_y_np = np.concatenate(all_y)
    return all_x_np, all_y_np


def custom_corrcc(alpha1, alpha2, axis=None):
    """

    Parameters
    ----------
    alpha1 : ndarray
        First 1d circular data.
    alpha2 : ndarray
        Second 1d circular data. Array size must be the same as alpha1.
    axis : int
        Axis to compute the data.
    Returns
    -------
    rho, pval
    """

    assert alpha1.shape == alpha2.shape, 'Input dimensions do not match.'
    n = alpha1.shape[0]
    # center data on circular mean
    alpha1, alpha2 = center(alpha1, alpha2, axis=axis)

    # compute correlation coeffcient from p. 176
    num = np.sum(np.sin(alpha1) * np.sin(alpha2), axis=axis)
    den = np.sqrt(np.sum(np.sin(alpha1) ** 2, axis=axis) *
                  np.sum(np.sin(alpha2) ** 2, axis=axis))
    rho = num / den

    l20 = np.mean(np.sin(alpha1) ** 2)
    l02 = np.mean(np.sin(alpha2) ** 2)
    l22 = np.mean((np.sin(alpha1) ** 2) * (np.sin(alpha2) ** 2))
    ts = np.sqrt((n * l20 * l02) / l22) * rho
    pval = 2 * (1 - norm.cdf(np.abs(ts)))
    return rho, pval


class DirectionerBining:
    def __init__(self, aedges, passangles):
        self.aedges = aedges
        self.aedm = midedges(self.aedges)
        self.abind = self.aedges[1] - self.aedges[0]
        self.occ_bins, _ = np.histogram(passangles, bins=self.aedges)

    def get_directionality(self, spikeangles):
        """
        Take spike angles, return directionality information (nan excluded).
        Parameters
        ----------
        spikeangles : ndarray
            Heading direction at spikes, in range (-pi, pi). Shape = (N, ).
        Returns
        -------
        fieldangle, fieldR, (spike_bins, occ_bins, norm_prob)

        """
        total_nspks = spikeangles.shape[0]
        spike_bins, _ = np.histogram(spikeangles, bins=self.aedges)

        norm_prob = normalize_distr(spike_bins, self.occ_bins)
        fieldangle = circmean(self.aedm, norm_prob, d=self.abind)
        fieldR = resultant_vector_length(self.aedm, norm_prob, d=self.abind)
        return fieldangle, fieldR, (spike_bins, self.occ_bins, norm_prob)


class DirectionerMLM:
    def __init__(self, pos, hd, dt, sp_binwidth, a_binwidth, minerr=0.001, verbose=False):
        """
        Parameters
        ----------
        pos : ndarray
            xy coordinates of trajectory. Shape = (time, 2).
        hd : ndarray
            Heading in range (-pi, pi). Shape = (time, )
        dt : scalar
            Time duration represented by each sample of occupancy.

        sp_binwidth : scalar
            Bin width of xy space. Recommended 0.05 of the range.
        a_binwidth : scalar
            Bin width of angular distribution. Should be 2pi/36
        minerr : scalar
            Error tolerance of MLM iteration. Default 0.01.
        """
        self.minerr = minerr
        self.verbose = verbose
        # Binning
        self.xbins = np.arange(pos[:, 0].min(), pos[:, 0].max() + sp_binwidth, step=sp_binwidth)
        self.ybins = np.arange(pos[:, 1].min(), pos[:, 1].max() + sp_binwidth, step=sp_binwidth)
        self.abins = np.arange(-np.pi, np.pi + a_binwidth, step=a_binwidth)
        self.aedm = midedges(self.abins)
        self.abind = self.abins[1] - self.abins[0]

        # Behavioral
        data3d = np.concatenate([pos, hd.reshape(-1, 1)], axis=1)
        bins3d, edges3d = np.histogramdd(data3d, bins=[self.xbins, self.ybins, self.abins])
        self.tocc = bins3d * dt

    def get_directionality(self, possp, hdsp):
        """

        Parameters
        ----------
        possp : ndarray
            xy coordinates at spike times. Shape = (spiketime, 2)
        hdsp : ndarray
            Heading at spike times. Shape = (spiketime, )

        Returns
        -------

        """
        datasp3d = np.concatenate([possp, hdsp.reshape(-1, 1)], axis=1)
        nspk, edgessp3d = np.histogramdd(datasp3d, bins=[self.xbins, self.ybins, self.abins])
        totspks = np.sum(nspk)

        directionality = np.ones(self.tocc.shape[2]) / self.tocc.shape[2] * np.sqrt(totspks)
        positionality = np.ones((self.tocc.shape[0], self.tocc.shape[1])) / self.tocc.shape[0] / self.tocc.shape[1] * np.sqrt(totspks)

        err = 2
        iter = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            while err > self.minerr:

                # pi
                tmp = np.nansum(directionality.reshape(1, 1, -1) * self.tocc, axis=2)
                ptmp = np.nansum(nspk, axis=2) / tmp
                ptmp[np.isinf(ptmp)] = np.nan

                # dj
                tmp = np.nansum(
                    np.nansum(positionality.reshape(positionality.shape[0], positionality.shape[1], 1) * self.tocc,
                              axis=0), axis=0)
                dtmp = np.nansum(np.nansum(nspk, axis=0), axis=0) / tmp
                dtmp[np.isinf(dtmp)] = np.nan

                # nfac
                # nfac = np.nansum(np.nansum(dtmp.reshape(1, 1, -1) * self.tocc, axis=2) * ptmp)
                # dtmp_norm = dtmp * np.sqrt(totspks / nfac)
                # ptmp_norm = ptmp * np.sqrt(totspks / nfac)

                dtmp_norm = dtmp / np.nansum(dtmp)
                ptmp_norm = ptmp / np.nansum(ptmp)

                # error
                errd = np.nanmean(directionality - dtmp_norm) ** 2
                errp = np.nanmean(positionality - ptmp_norm) ** 2
                err = np.sqrt(errd + errp)
                if self.verbose:
                    print('\r Error = %0.5f' % err, end="", flush=True)
                # update
                positionality = ptmp_norm
                directionality = dtmp_norm
                iter += 1

        directionality = directionality/np.nansum(directionality)
        nonanmask = ~np.isnan(directionality)
        fieldangle = circmean(self.aedm[nonanmask], directionality[nonanmask], d=self.abind)
        fieldR = resultant_vector_length(self.aedm[nonanmask], directionality[nonanmask], d=self.abind)
        return fieldangle, fieldR, directionality


def passes_spikes_shuffle(passes_list, sptimes_list, return_concat=False, seed=None):
    """
    Random circular shift the spike times in different pass compartments.

    Parameters
    ----------
    passes_list : iterable
        Each element contains the times of a pass.
    sptimes_list : iterable
        Each element contains the spike times of a pass.
    return_concat : bool
        Default False, returning a list of ndarray. Each element contains shuffled spike times in the corresponding pass compartment. If True, return a concatenated array of spike times instead.

    Returns
    -------
    shifted_tsp_boxes : list or ndarray.

    """

    assert len(passes_list) == len(sptimes_list)

    passbox_list = [(x.min(), x.max()) for x in passes_list]

    # Concatenate spike trains across different passes.
    offseted = []
    all_gaps = np.zeros(len(passbox_list))
    boxes = []
    gap = passbox_list[0][0]  # Remove time gap between each pass compartment. The 1st gap is the starting time
    for i in range(len(passbox_list)):
        passeach, sptime = passbox_list[i], sptimes_list[i]
        if i > 0:
            gap += passbox_list[i][0] - passbox_list[i - 1][1]
            tsp_tmp = sptime - gap
        else:
            tsp_tmp = sptime - gap
        boxes.append((passeach - gap))
        offseted.append(tsp_tmp)
        all_gaps[i] = gap

    # Random time shift the concatenated spike trains
    all_tsp_cat = np.concatenate(offseted)
    maxduration = boxes[-1][1]
    if seed is not None:
        np.random.seed(seed)
    shift_amount = np.random.rand() * maxduration
    all_tsp_shifted = np.mod(all_tsp_cat + shift_amount, maxduration)

    # Separate the spikes to original pass compartments
    shifted_tsp_boxes = []
    for i in range(len(boxes)):
        start, end = boxes[i]
        tsp_inside = all_tsp_shifted[(all_tsp_shifted < end) & (all_tsp_shifted >= start)] + all_gaps[i]
        shifted_tsp_boxes.append(np.sort(tsp_inside))

    if return_concat:
        shifted_tsp_boxes = np.concatenate(shifted_tsp_boxes)
    return shifted_tsp_boxes


class PassShuffler:
    def __init__(self, passes_list, sptimes_list):
        self.boxes, self.tsp_cat, self.gaps = self.concat_passes(passes_list, sptimes_list)
        self.tsp_windows, self.win_starts = self.split_to_windows(self.tsp_cat)


    def timeshift_shuffle(self, seed=None, return_concat=False):
        # Random time shift the concatenated spike trains
        maxduration = self.boxes[-1][1]
        if seed is not None:
            np.random.seed(seed)
        shift_amount = np.random.rand() * maxduration
        all_tsp_shifted = np.mod(self.tsp_cat + shift_amount, maxduration)

        # Separate the spikes to original pass compartments
        shifted_tsp_boxes = self.separate_to_passboxes(all_tsp_shifted)

        if return_concat:
            shifted_tsp_boxes = np.concatenate(shifted_tsp_boxes)
        return shifted_tsp_boxes

    def window_shuffle(self, seed=None, return_concat=False):

        if seed is not None:
            random.seed(seed)
        random.shuffle(self.tsp_windows)
        tsp_shuffled = np.concatenate([self.tsp_windows[x] + self.win_starts[x] for x in range(len(self.tsp_windows))])

        # Separate the spikes to original pass compartments
        shifted_tsp_boxes = self.separate_to_passboxes(tsp_shuffled)

        if return_concat:
            shifted_tsp_boxes = np.concatenate(shifted_tsp_boxes)
        return shifted_tsp_boxes

    def separate_to_passboxes(self, tsp):

        # Separate the spikes to original pass compartments
        shifted_tsp_boxes = []
        for i in range(len(self.boxes)):
            start, end = self.boxes[i]
            tsp_inside = tsp[(tsp < end) & (tsp >= start)] + self.gaps[i]
            shifted_tsp_boxes.append(np.sort(tsp_inside))
        return shifted_tsp_boxes


    @staticmethod
    def concat_passes(passes_list, sptimes_list):

        assert len(passes_list) == len(sptimes_list)

        passbox_list = [(x.min(), x.max()) for x in passes_list]

        # Concatenate spike trains across different passes.
        offseted = []
        all_gaps = np.zeros(len(passbox_list))
        boxes = []
        gap = passbox_list[0][0]  # Remove time gap between each pass compartment. The 1st gap is the starting time
        for i in range(len(passbox_list)):
            passeach, sptime = passbox_list[i], sptimes_list[i]
            if i > 0:
                gap += passbox_list[i][0] - passbox_list[i - 1][1]
                tsp_tmp = sptime - gap
            else:
                tsp_tmp = sptime - gap
            boxes.append((passeach - gap))
            offseted.append(tsp_tmp)
            all_gaps[i] = gap
        all_tsp_cat = np.concatenate(offseted)
        return boxes, all_tsp_cat, all_gaps

    @staticmethod
    def split_to_windows(tsp, windowsize=0.5):
        # Create time windows
        maxt, mint = tsp.max(), tsp.min()
        tsp_duration = maxt-mint
        if tsp_duration < windowsize:
            windows = np.arange(mint, maxt, step=(maxt-mint)/5)
            warnings.warn("Shuffling window is smaller than the spike train! Using window size = range/5 instead.")
        else:
            windows = np.arange(mint, maxt, step=windowsize)  # 500ms window
            windows = np.append(windows, windows[-1] + 0.5)

        tsp_windows = []
        win_starts = []
        for wid in range(windows.shape[0] - 1):
            wstart, wend = windows[wid], windows[wid + 1]
            tsp_inside_window = tsp[(tsp < wend) & (tsp >= wstart)]
            if tsp_inside_window.shape[0] < 1:
                continue
            tsp_inside_window = tsp_inside_window - wstart
            win_starts.append(wstart)
            tsp_windows.append(tsp_inside_window)
        return tsp_windows, win_starts

class RegressionCC:
    """
    Topics in Circular Statistics. Jammalamadaka, S Rao. Seagupta, A. P-187. Section 8.6.

    """

    def __init__(self, m):
        """

        Parameters
        ----------
        m : int
            Order of the trigonometric polynomials of the regression model.
        """
        self.m = m
        self.A0 = None
        self.C0 = None
        self.A_dict = {}
        self.B_dict = {}
        self.C_dict = {}
        self.D_dict = {}

    def fit(self, alpha, beta):
        """
        Fit the regression model.
        Parameters
        ----------
        alpha : ndarray
            Independent circular variable. Shape = (n, ).
        beta : ndarray
            Dependent circular variable. Same shape as alpha.
        Returns
        -------
        None
        """
        assert alpha.shape[0] == beta.shape[0]
        self.A0 = np.mean(np.cos(beta))
        self.C0 = np.mean(np.sin(beta))
        for i in range(self.m):
            j = i + 1
            self.A_dict[j] = np.mean(np.cos(beta) * np.cos(j * alpha))
            self.B_dict[j] = np.mean(np.cos(beta) * np.sin(j * alpha))
            self.C_dict[j] = np.mean(np.sin(beta) * np.cos(j * alpha))
            self.D_dict[j] = np.mean(np.sin(beta) * np.sin(j * alpha))

    def predict(self, alpha):
        """

        Parameters
        ----------
        alpha : ndarray
            Independent circular variable. Shape = (n, ).
        Returns
        -------
        beta_pred : ndarray
            Predicted dependent circular variable given alpha. Same shape as alpha.
        """
        nsamples = alpha.shape[0]
        g1 = np.zeros(nsamples) * self.A0
        g2 = np.zeros(nsamples) * self.C0
        for i in range(self.m):
            k = i + 1
            g1 += self.A_dict[k] * np.cos(k * alpha) + self.B_dict[k] * np.sin(k * alpha)
            g2 += self.C_dict[k] * np.cos(k * alpha) + self.D_dict[k] * np.sin(k * alpha)
        beta_pred = np.arctan2(g2, g1)
        return beta_pred


def repeat_arr(val_arr, n_arr):
    '''
    Repeat the element in val_arr for n times as in n_arr.

    Parameters
    ----------
    val_arr : ndarray
        1D numpy array. Each element is a value.
    n_arr : ndarray
        1D numpy array. Each element is number of times you want to repeat the corresponding value in val_arr.

    Returns
    -------
    expanded_arr : ndarray
        1D numpy array. The array contains each val in val_arr repeated for the corresponding times in n_arr.
    '''
    # Spike Counts
    expanded_list = []
    for n, val in zip(n_arr, val_arr):
        tmp_val = np.array([val] * n)
        expanded_list.append(tmp_val)
    expanded_arr = np.concatenate(expanded_list)
    return expanded_arr


def angular_dispersion_test(alpha1, alpha2):
    mean1 = circmean(alpha1)
    mean2 = circmean(alpha2)
    adv1 = np.abs(cdiff(alpha1, mean1))
    adv2 = np.abs(cdiff(alpha2, mean2))
    z, pval = ranksums(adv1, adv2)
    return z, pval, (adv1, adv2)

def circ_ktest(alpha1, alpha2):
    alpha1 = np.asarray(alpha1)
    alpha2 = np.asarray(alpha2)

    n1 = alpha1.shape[0]
    n2 = alpha2.shape[0]

    R1 = resultant_vector_length(alpha1)
    R2 = resultant_vector_length(alpha2)

    f_stat = ( (n2 - 1) * (n1- R1) ) / ( (n1 - 1) * (n2 - R2) )

    if f_stat > 1:
        pval = 2 * (1 - fdist.cdf(f_stat, n1, n2))
    else:
        f_stat = 1/f_stat
        pval = 2 * (1 - fdist.cdf(f_stat, n2, n1))

    return f_stat, pval


def shiftcyc_full2half(angle):
    newangle = np.mod(angle + np.pi, 2 * np.pi) - np.pi
    return newangle



def fisherexact(arr):
    """

    Parameters
    ----------
    arr : ndarray
        2 x 2 numpy array, as [[a, b], [c, d]]

    Returns
    -------
    p : float
        p-value or Fisher's exact test.
    """
    a, b = arr[0, 0], arr[0, 1]
    c, d = arr[1, 0], arr[1, 1]
    n = a+b+c+d
    up = comb(a+b, a) * comb(c+d, c)
    down = comb(n, a+c)
    p = up/down
    if np.isnan(p):
        _, p = fisher_exact(arr)
    return p


def cal_hd(x, y):
    hd = torch.angle(torch.diff(x) + 1j * np.diff(y))  # angle is converted to (-pi, pi)
    hd = torch.cat([hd, hd[[-1]]])
    return hd

def get_nidx(x, y, a, xxtun, yytun, aatun):
    nidx = torch.argmin((xxtun - x) ** 2 + (yytun - y) ** 2 + (aatun - a) ** 2)
    return nidx.item()

def fr_transfer(I):
    """for alpha = 1, In-place operation
    """
    I[I <= 4] = torch.log(1 + torch.exp(I[I <= 4]))
    return I
