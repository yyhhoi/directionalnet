#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Wrappers enclosing scripts which are invoked more than once

"""

import numpy as np
import pandas as pd
from pycircstat.tests import rayleigh
from scipy.interpolate import interp1d
from pycircstat.descriptive import mean as circmean, cdiff, resultant_vector_length
from library.linear_circular_r import rcc
from library.comput_utils import DirectionerBining, DirectionerMLM, compute_straightness, circular_density_1d, \
    repeat_arr, shiftcyc_full2half, midedges, normalize_distr, passes_spikes_shuffle


class DirectionalityStatsByThresh:
    def __init__(self, nspikes_key, shiftp_key, fieldR_key):
        self.nspikes_key = nspikes_key
        self.shiftp_key = shiftp_key
        self.fieldR_key = fieldR_key

    def gen_directionality_stats_by_thresh(self, df, spike_threshs=None):
        if spike_threshs is None:
            max_nspikes = df[self.nspikes_key].max()
            spike_threshs = np.linspace(0, max_nspikes, 20)
            spike_threshs = spike_threshs[:-1]

        all_n = df.shape[0]
        stats_dict = dict(
            spike_threshs=spike_threshs,
            sigfrac_shift=np.zeros(spike_threshs.shape[0]),
            medianR=np.zeros(spike_threshs.shape[0]),
            datafrac=np.zeros(spike_threshs.shape[0]),
            allR=[],
            shift_signum=np.zeros(spike_threshs.shape[0]),
            shift_nonsignum=np.zeros(spike_threshs.shape[0]),
            n=np.zeros(spike_threshs.shape[0])

        )
        for idx, thresh in enumerate(spike_threshs):
            thresh_df = df[df[self.nspikes_key] > thresh]

            stats_dict['sigfrac_shift'][idx] = np.mean(thresh_df[self.shiftp_key] < 0.05)
            stats_dict['medianR'][idx] = np.median(thresh_df[self.fieldR_key])
            stats_dict['datafrac'][idx] = thresh_df.shape[0] / all_n
            stats_dict['allR'].append(thresh_df[self.fieldR_key].to_numpy())
            stats_dict['shift_signum'][idx] = np.sum(thresh_df[self.shiftp_key] < 0.05)
            stats_dict['shift_nonsignum'][idx] = np.sum(thresh_df[self.shiftp_key] >= 0.05)
            stats_dict['n'][idx] = thresh_df.shape[0]

        return stats_dict


class PrecessionProcesser:
    def __init__(self, wave):
        self.wave = wave
        self.trange = None
        self.phase_inter = interp1d(wave['tax'], wave['phase'])
        self.theta_inter = interp1d(wave['tax'], wave['theta'])
        self.wave_maxt, self.wave_mint = wave['tax'].max(), wave['tax'].min()

    def get_single_precession(self, passes_df, neuro_keys_dict, field_dia, tag=''):
        """Receive passdf and append columns containing precession information

        Parameters
        ----------
        passes_df : dataframe
            Must contain behavioural columns - 'x', 'y', 't', 'angle', and spike columns 'tsp', 'spikex', 'spikey', 'spikev', 'spikeangle'
        neuro_keys_dict : dict
            Dictionary defining the keynames of the spike columns.

        Returns
        -------
        dict
            Dictionay containing all the precession information.
        """
        assert self.trange is not None

        data_dict = dict(
            dsp=[], pass_nspikes=[], phasesp=[], tsp_withtheta=[], mean_angle=[], mean_anglesp=[],
            rcc_m=[], rcc_c=[], rcc_rho=[], rcc_p=[],
            wave_t=[], wave_phase=[], wave_theta=[], wave_totalcycles=[], wave_truecycles=[], wave_maxperiod=[],
            cycfrac=[], fitted=[]
        )
        data_dict = {k+tag: val for k, val in data_dict.items()}


        all_maxt, all_mint = self.trange
        tsp_k = neuro_keys_dict['tsp']
        spikex_k = neuro_keys_dict['spikex']
        spikey_k = neuro_keys_dict['spikey']
        spikeangle_k = neuro_keys_dict['spikeangle']
        for npass in range(passes_df.shape[0]):

            excluded_for_precess = passes_df.loc[npass, 'excluded_for_precess']
            if excluded_for_precess:
                for k in data_dict.keys():
                    if k !=('fitted'+tag):
                        data_dict[k].append(None)
                data_dict['fitted'+tag].append(False)
                continue

            # Behavioural
            x, y, t, angle, chunked = passes_df.loc[npass, ['x', 'y', 't', 'angle', 'chunked']]

            # Neural
            tsp, xsp, ysp, anglesp = passes_df.loc[npass, [tsp_k, spikex_k, spikey_k, spikeangle_k]]

            # Filtering
            inidx = np.where((tsp > self.wave_mint) & (tsp <= self.wave_maxt) &
                             (tsp > all_mint) & (tsp <= all_maxt))[0]
            tsp_in = tsp[inidx]
            thetasp_in = self.theta_inter(tsp_in)
            inthetaidx = np.where(np.abs(thetasp_in) > 1e-5)[0]
            if (inidx.shape[0] < 2) or (inthetaidx.shape[0] < 2):
                for k in data_dict.keys():
                    if k !=('fitted'+tag):
                        data_dict[k].append(None)
                data_dict['fitted'+tag].append(False)
                continue


            # Compute pass direction
            anglesp_in = anglesp[inidx]
            mean_angle = shiftcyc_full2half(circmean(angle))
            mean_anglesp = shiftcyc_full2half(circmean(anglesp_in))

            # Pass length
            d = np.sqrt(np.square(np.diff(x)) + np.square(np.diff(y)))
            cumd = np.append(0, np.cumsum(d))
            if chunked==1:
                cumd_norm = cumd / field_dia
            elif chunked==0:
                cumd_norm = cumd / cumd.max()
            else:
                raise  # chunked must be either 0 or 1. Please exclude the rows which have other values.
            dinter = interp1d(t, cumd_norm)

            # RCC
            tsp_intheta = tsp_in[inthetaidx]
            phasesp = self.phase_inter(tsp_intheta)
            dsp = dinter(tsp_intheta)
            regress = rcc(dsp, phasesp, abound=(-2, 2))
            rcc_m, rcc_c, rcc_rho, rcc_p = regress['aopt'], regress['phi0'], regress['rho'], regress['p']

            # Wave
            wave_mint, wave_maxt = self.wave['tax'].min(), self.wave['tax'].max()
            if (wave_mint < t.min()) and (wave_maxt > t.max()):
                wid1 = np.where(self.wave['tax'] < t.min())[0][-1]
                wid2 = np.where(self.wave['tax'] > t.max())[0][0]
            else:
                wid1 = np.argmin( np.abs(self.wave['tax']-t.min()))
                wid2 = np.argmin( np.abs(self.wave['tax']-t.max()))

            wave_t = self.wave['tax'][wid1:wid2]
            wave_phase = self.wave['phase'][wid1:wid2]
            wave_theta = self.wave['theta'][wid1:wid2]


            # Number of theta cycles, and cycles that have spikes
            cycidx = np.where(np.diff(wave_phase) < -(np.pi))[0]
            if cycidx.shape[0] == 0:
                wave_totalcycles = 1
                wave_truecycles = 0
                wave_maxperiod = t.max() - t.min()
            else:
                cyc_twindows = np.concatenate([wave_t[[0]], wave_t[cycidx], wave_t[[-1]]])
                presence = 0
                num_windows = cyc_twindows.shape[0] - 1
                for i in range(num_windows):
                    t1, t2 = cyc_twindows[i], cyc_twindows[i + 1]
                    if np.sum((tsp_intheta <= t2) & (tsp_intheta > t1)) > 0:
                        presence += 1
                wave_totalcycles = num_windows
                wave_truecycles = presence
                wave_maxperiod = np.max(np.diff(cyc_twindows))

            # Pass info
            data_dict['dsp'+tag].append(dsp)
            data_dict['pass_nspikes'+tag].append(dsp.shape[0])
            data_dict['phasesp'+tag].append(phasesp)
            data_dict['tsp_withtheta'+tag].append(tsp_intheta)
            data_dict['mean_angle'+tag].append(mean_angle)
            data_dict['mean_anglesp'+tag].append(mean_anglesp)

            # Pass regression
            data_dict['rcc_m'+tag].append(rcc_m)
            data_dict['rcc_c'+tag].append(rcc_c)
            data_dict['rcc_rho'+tag].append(rcc_rho)
            data_dict['rcc_p'+tag].append(rcc_p)

            # Pass LFP
            data_dict['wave_t'+tag].append(wave_t)
            data_dict['wave_phase'+tag].append(wave_phase)
            data_dict['wave_theta'+tag].append(wave_theta)
            data_dict['wave_totalcycles'+tag].append(wave_totalcycles)
            data_dict['wave_truecycles'+tag].append(wave_truecycles)
            data_dict['wave_maxperiod'+tag].append(wave_maxperiod)
            data_dict['cycfrac'+tag].append(wave_truecycles / wave_totalcycles)
            data_dict['fitted'+tag].append(True)

        datadf = pd.DataFrame(data_dict)
        appended_df = pd.concat([passes_df, datadf], axis=1)
        return appended_df



    def set_trange(self, trange):
        """Set the min and max of time

        Parameters
        ----------
        trange : tuple
            Tuple which is (max_t, min_t).
        """
        self.trange = trange



class PrecessionFilter:
    def __init__(self):
        self.cycfrac_thresh = 0.2  # set 0.5
        self.min_num_truecycle = 2  # set 3
        self.maxperiod_thresh = 0.27  # in s, about 3 theta cycle, set 0.27

    def filter_single(self, precess_df):

        total_mask = (precess_df['cycfrac'] > self.cycfrac_thresh) & \
                     (precess_df['wave_truecycles'] > self.min_num_truecycle) & \
                     (precess_df['wave_maxperiod'] < self.maxperiod_thresh) & \
                     (precess_df['rcc_m'] < 0) & \
                     (precess_df['rcc_m'] > -1.8)


        precess_df['precess_exist'] = total_mask
        return precess_df

    def filter_pair(self, precess_df):

        cycfrac_mask1 = precess_df['cycfrac1'] > self.cycfrac_thresh
        cycfrac_mask2 = precess_df['cycfrac2'] > self.cycfrac_thresh
        true_cycle_mask1 = precess_df['wave_truecycles1'] > self.min_num_truecycle
        true_cycle_mask2 = precess_df['wave_truecycles2'] > self.min_num_truecycle
        max_period_mask1 = precess_df['wave_maxperiod1'] < self.maxperiod_thresh
        max_period_mask2 = precess_df['wave_maxperiod2'] < self.maxperiod_thresh
        slope_mask1 = (precess_df['rcc_m1'] < 0) & (precess_df['rcc_m1'] > -1.8)
        slope_mask2 = (precess_df['rcc_m2'] < 0) & (precess_df['rcc_m2'] > -1.8)
        total_mask1 = cycfrac_mask1 & true_cycle_mask1 & max_period_mask1 & slope_mask1
        total_mask2 = cycfrac_mask2 & true_cycle_mask2 & max_period_mask2 & slope_mask2
        total_mask = total_mask1 & total_mask2


        precess_df['precess_exist1'] = total_mask1
        precess_df['precess_exist2'] = total_mask2
        precess_df['precess_exist'] = total_mask

        # precess_df = precess_df[straightrank_mask1 & straightrank_mask2].reset_index(drop=True)
        return precess_df




def compute_precessangle(pass_angles, pass_nspikes, precess_mask, kappa=None, bins=None):
    """

    Parameters
    ----------
    pass_angles : ndarray
        1d array with pass angles in range -pi and pi.
    pass_nspikes : ndarray
        1d array with number of spikes in the corresponding pass as "pass_angles".
    precess_mask : ndarray
        1d bool-array specifying which pass in "pass_angles" is precessing.
    kappa : int or float or None
        Concentration of von-mise distribution for KDE. int or float for enabling KDE. None for disabling KDE.
    bins : ndarray or None
        1d array of edges for binning the pass_angles.

    Returns
    -------

    """

    if (kappa is None) and (bins is None):
        raise AssertionError('Either one of the arguments kappa and bins must be specified.')
    if (kappa is not None) and (bins is not None):
        raise AssertionError('Argument kappa and bins cannot be both specified.')
    numpass = pass_angles[precess_mask].shape[0]
    if kappa is not None:  # Use KDE with concentration kappa
        pass_ax, passbins_p = circular_density_1d(pass_angles[precess_mask], kappa, 100, (-np.pi, np.pi))
        pass_ax, passbins_np = circular_density_1d(pass_angles[~precess_mask], kappa, 100, (-np.pi, np.pi))
        spike_ax, spikebins = circular_density_1d(pass_angles[precess_mask], kappa, 100, (-np.pi, np.pi), w=pass_nspikes[precess_mask])
    elif bins is not None:  # Use binning with certain binsize
        passbins_p, pass_ax = np.histogram(pass_angles[precess_mask], bins=bins)
        passbins_np, pass_ax = np.histogram(pass_angles[~precess_mask], bins=bins)
        spikebins, pass_ax = np.histogram(repeat_arr(pass_angles[precess_mask], pass_nspikes[precess_mask].astype(int)), bins=bins)
        pass_ax = midedges(pass_ax)
    else:
        raise

    norm_density = normalize_distr(passbins_p, spikebins)

    # Best direction
    bestangle = circmean(pass_ax, w=norm_density, d=pass_ax[1] - pass_ax[0])
    R = resultant_vector_length(pass_ax, w=norm_density, d=pass_ax[1] - pass_ax[0])
    return bestangle, R, (norm_density, passbins_p, passbins_np, spikebins)


def get_single_precessdf(passdf, precesser, precess_filter, neuro_keys_dict, field_d, kappa=None, bins=None):
    """
        1. Converting dataframe of passes to dataframe of precession.
        2. Computing best angle and R for precession.
    Parameters
    ----------
    passdf
    precesser
    precess_filter

    Returns
    -------

    """
    # Convert pass information to precession information
    precess_df = precesser.get_single_precession(passdf, neuro_keys_dict, field_dia=field_d)
    precess_df = precess_filter.filter_single(precess_df)
    if (precess_df.shape[0] < 1) or (precess_df['precess_exist'].sum() < 1):
        return precess_df, None, None, None
    else:

        # Compute statistics for precession (bestangle, R, norm_distr)
        bestangle, R, densities = compute_precessangle(pass_angles=precess_df['mean_anglesp'].to_numpy(),
                                                       pass_nspikes=precess_df['pass_nspikes'].to_numpy(),
                                                       precess_mask=precess_df['precess_exist'].to_numpy(),
                                                       kappa=kappa, bins=bins)

        return precess_df, bestangle, R, densities



def combined_average_curve(slopes, offsets, xrange=(0, 1), xbins=10):
    """

    Parameters
    ----------
    slopes : ndarray
        with shape (n, ). n is number of samples. In unit rad
    offsets : ndarray
        with shape (n, ). In unit rad.
    xrange : tuple
    xbins : int

    Returns
    -------

    """
    n = slopes.shape[0]
    assert n == offsets.shape[0]
    xdum = np.linspace(xrange[0], xrange[1], xbins)
    all_xdum = [xdum] * n
    all_ydum = []
    for i in range(n):
        ydum = xdum * slopes[i] + offsets[i]
        all_ydum.append(ydum)
    all_xdum = np.concatenate(all_xdum)
    all_ydum = np.concatenate(all_ydum)
    regress = rcc(all_xdum, all_ydum)
    return regress



def permutation_test_average_slopeoffset(slopes_high, offsets_high, slopes_low, offsets_low, NShuffles=200):
    """

    Parameters
    ----------
    slopes_high : ndarray
        Shape (N, ). In unit rad.
    offsets_high : ndarray
        Shape (N, ). In unit rad.
    slopes_low : ndarray
        Shape (K, ). In unit rad.
    offsets_low : ndarray
        Shape (K, ). In unit rad.
    NShuffles : int
        Number of shuffling times.

    Returns
    -------
    regress_high : dict
    regress_low : dict
    pval_slope : float
    pval_offset : float

    """

    nhigh = slopes_high.shape[0]
    nlow = slopes_low.shape[0]
    ntotal = nhigh + nlow
    assert nhigh == offsets_high.shape[0]
    assert nlow == offsets_low.shape[0]

    regress_high = combined_average_curve(slopes_high, offsets_high)
    regress_low = combined_average_curve(slopes_low, offsets_low)

    # slope_diff = cdiff(regress_high['aopt']*2*np.pi, regress_low['aopt']*2*np.pi)
    slope_diff = np.abs(regress_high['aopt'] - regress_low['aopt'])
    offset_diff = np.abs(cdiff(regress_high['phi0'], regress_low['phi0']))

    pooled_slopes = np.append(slopes_high, slopes_low)
    pooled_offsets = np.append(offsets_high, offsets_low)

    all_shuf_slope_diff = np.zeros(NShuffles)
    all_shuf_offset_diff = np.zeros(NShuffles)
    for i in range(NShuffles):
        if i % 10 == 0:
            print('\rShuffling %d/%d'%(i, NShuffles), flush=True, end='')
        np.random.seed(i)
        ran_vec = np.random.permutation(ntotal)

        shuf_slopes = pooled_slopes[ran_vec]
        shuf_offsets = pooled_offsets[ran_vec]

        shuf_regress_high = combined_average_curve(shuf_slopes[0:nhigh], shuf_offsets[0:nhigh])
        shuf_regress_low = combined_average_curve(shuf_slopes[nhigh:], shuf_offsets[nhigh:])

        all_shuf_slope_diff[i] = np.abs(shuf_regress_high['aopt'] - shuf_regress_low['aopt'])
        all_shuf_offset_diff[i] = np.abs(cdiff(shuf_regress_high['phi0'], shuf_regress_low['phi0']))
    print()
    pval_slope = 1- np.mean(np.abs(slope_diff) > np.abs(all_shuf_slope_diff))
    pval_offset = 1 - np.mean(np.abs(offset_diff) > np.abs(all_shuf_offset_diff))

    return regress_high, regress_low, pval_slope, pval_offset

def permutation_test_arithmetic_average_slopeoffset(slopes_high, offsets_high, slopes_low, offsets_low, NShuffles=200):
    """

    Parameters
    ----------
    slopes_high : ndarray
        Shape (N, ). In unit rad.
    offsets_high : ndarray
        Shape (N, ). In unit rad.
    slopes_low : ndarray
        Shape (K, ). In unit rad.
    offsets_low : ndarray
        Shape (K, ). In unit rad.
    NShuffles : int
        Number of shuffling times.

    Returns
    -------
    regress_high : dict
    regress_low : dict
    pval_slope : float
    pval_offset : float

    """

    nhigh = slopes_high.shape[0]
    nlow = slopes_low.shape[0]
    ntotal = nhigh + nlow
    assert nhigh == offsets_high.shape[0]
    assert nlow == offsets_low.shape[0]

    mean_slope_high, mean_slope_low = np.mean(slopes_high), np.mean(slopes_low)
    mean_offset_high, mean_offset_low = circmean(offsets_high), circmean(offsets_low)

    slope_diff = np.abs(mean_slope_high - mean_slope_low)
    offset_diff = np.abs(cdiff(mean_offset_high, mean_offset_low))

    pooled_slopes = np.append(slopes_high, slopes_low)
    pooled_offsets = np.append(offsets_high, offsets_low)

    all_shuf_slope_diff = np.zeros(NShuffles)
    all_shuf_offset_diff = np.zeros(NShuffles)
    for i in range(NShuffles):
        if i % 10 == 0:
            print('\rShuffling %d/%d'%(i, NShuffles), flush=True, end='')
        np.random.seed(i)
        ran_vec = np.random.permutation(ntotal)

        shuf_slopes = pooled_slopes[ran_vec]
        shuf_offsets = pooled_offsets[ran_vec]

        shufmean_slope_high, shufmean_slope_low = np.mean(shuf_slopes[0:nhigh]), np.mean(shuf_slopes[nhigh:])
        shufmean_offset_high, shufmean_offset_low = circmean(shuf_offsets[0:nhigh]), circmean(shuf_offsets[nhigh:])


        all_shuf_slope_diff[i] = np.abs(shufmean_slope_high - shufmean_slope_low)
        all_shuf_offset_diff[i] = np.abs(cdiff(shufmean_offset_high, shufmean_offset_low))
    print()
    pval_slope = 1- np.mean(np.abs(slope_diff) > np.abs(all_shuf_slope_diff))
    pval_offset = 1 - np.mean(np.abs(offset_diff) > np.abs(all_shuf_offset_diff))

    regress_high = {'aopt':mean_slope_high/(2*np.pi), 'phi0':mean_offset_high}
    regress_low = {'aopt':mean_slope_low/(2*np.pi), 'phi0':mean_offset_low}

    return regress_high, regress_low, pval_slope, pval_offset




def construct_passdf_sim(analyzer, all_passidx, tidxsp, minpasstime=0.6):

    # x, y, t, v, angle, tsp, spikev, spikex, spikey, spikeangle
    pass_dict = dict(x=[], y=[], t=[], v=[], angle=[],
                     spikex=[], spikey=[], tsp=[], spikev=[], spikeangle=[])
    for pid1, pid2 in all_passidx:
        pass_t = analyzer.t[pid1:pid2]

        # Pass duration threshold
        if pass_t.shape[0] < 5:
            continue
        if (pass_t[-1] - pass_t[0]) < minpasstime:
            continue


        pass_dict['x'].append(analyzer.x[pid1:pid2])
        pass_dict['y'].append(analyzer.y[pid1:pid2])
        pass_dict['t'].append(pass_t)
        pass_dict['v'].append(analyzer.speed[pid1:pid2])
        pass_dict['angle'].append(analyzer.angle[pid1:pid2])

        tidxsp_within = tidxsp[(tidxsp >= pid1) & (tidxsp < pid2)]
        pass_dict['spikex'].append(analyzer.x[tidxsp_within])
        pass_dict['spikey'].append(analyzer.y[tidxsp_within])
        pass_dict['spikev'].append(analyzer.speed[tidxsp_within])
        pass_dict['tsp'].append(analyzer.t[tidxsp_within])
        pass_dict['spikeangle'].append(analyzer.angle[tidxsp_within])

    pass_df = pd.DataFrame(pass_dict)
    return pass_df



def construct_pairedpass_df_sim(analyzer, all_passidx_pair, tok1, tok2, tidxsp1, tidxsp2, minpasstime=0.6):

    # x, y, t, v, angle,
    # tsp1, spike1v, spike1x, spike1y, spike1angle
    # tsp2, spike2v, spike2x, spike2y, spike2angle
    pass_dict = dict(x=[], y=[], t=[], v=[], angle=[], infield1=[], infield2=[],
                     spike1x=[], spike1y=[], tsp1=[], spike1v=[], spike1angle=[],
                     spike2x=[], spike2y=[], tsp2=[], spike2v=[], spike2angle=[])

    for pid1, pid2 in all_passidx_pair:
        pass_t = analyzer.t[pid1:pid2]

        # Pass duration threshold
        if pass_t.shape[0] < 5:
            continue
        if (pass_t[-1] - pass_t[0]) < minpasstime:
            continue

        # Border crossing
        cross1 = np.sum(np.abs(np.diff(tok1[pid1:pid2])))
        cross2 = np.sum(np.abs(np.diff(tok2[pid1:pid2])))
        if (cross1 > 1) or (cross2 > 1):
            continue


        pass_dict['x'].append(analyzer.x[pid1:pid2])
        pass_dict['y'].append(analyzer.y[pid1:pid2])
        pass_dict['t'].append(pass_t)
        pass_dict['v'].append(analyzer.velocity[pid1:pid2])
        pass_dict['angle'].append(analyzer.angle[pid1:pid2])
        pass_dict['infield1'].append(tok1[pid1:pid2])
        pass_dict['infield2'].append(tok2[pid1:pid2])

        tidxsp1_within = tidxsp1[(tidxsp1 >= pid1) & (tidxsp1 < pid2)]
        pass_dict['spike1x'].append(analyzer.x[tidxsp1_within])
        pass_dict['spike1y'].append(analyzer.y[tidxsp1_within])
        pass_dict['spike1v'].append(analyzer.velocity[tidxsp1_within])
        pass_dict['tsp1'].append(analyzer.t[tidxsp1_within])
        pass_dict['spike1angle'].append(analyzer.angle[tidxsp1_within])

        tidxsp2_within = tidxsp2[(tidxsp2 >= pid1) & (tidxsp2 < pid2)]
        pass_dict['spike2x'].append(analyzer.x[tidxsp2_within])
        pass_dict['spike2y'].append(analyzer.y[tidxsp2_within])
        pass_dict['spike2v'].append(analyzer.velocity[tidxsp2_within])
        pass_dict['tsp2'].append(analyzer.t[tidxsp2_within])
        pass_dict['spike2angle'].append(analyzer.angle[tidxsp2_within])

    pass_df = pd.DataFrame(pass_dict)
    return pass_df


