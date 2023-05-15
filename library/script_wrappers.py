#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Wrappers enclosing scripts which are invoked more than once

"""

import numpy as np
import pandas as pd
from pycircstat.descriptive import cdiff
from library.comput_utils import rcc_wrapper, get_tspdiff, calc_exin_samepath, acc_metrics


def best_worst_analysis(SpikeDF, ref_a, ca3nidx, t, theta_phase, traj_d, xxtun1d, aatun1d, abs_xlim=20):
    precessdf_dict = dict(nidx=[], adiff=[], phasesp=[], onset=[], slope=[])
    for neuronid in ca3nidx:
        neuronxtmp = xxtun1d[neuronid]
        if np.abs(neuronxtmp) > abs_xlim:
            continue

        tidxsp_tmp = SpikeDF.loc[SpikeDF['neuronid']==neuronid, 'tidxsp'].to_numpy()
        if tidxsp_tmp.shape[0] < 5:
            continue
        atun_this = aatun1d[neuronid]
        adiff = np.abs(cdiff(atun_this, ref_a))
        tsp_eg, phasesp_eg = t[tidxsp_tmp], theta_phase[tidxsp_tmp]
        dsp_eg = traj_d[tidxsp_tmp]
        (dsp_norm, phasesp), (onset, slope), _ = rcc_wrapper(dsp_eg, phasesp_eg)
        if (onset > (1.9 * np.pi)) or (onset < (-1.9 * np.pi)):
            continue
        precessdf_dict['nidx'].append(neuronid)
        precessdf_dict['adiff'].append(adiff)
        precessdf_dict['phasesp'].append(phasesp_eg)
        precessdf_dict['onset'].append(onset)
        precessdf_dict['slope'].append(slope)
    precessdf = pd.DataFrame(precessdf_dict)

    bestprecessdf = precessdf[precessdf['adiff'] <= (np.pi/6)].reset_index(drop=True)
    worstprecessdf = precessdf[precessdf['adiff'] >= (np.pi - np.pi/6)].reset_index(drop=True)

    phasesp_best, phasesp_worst = np.concatenate(bestprecessdf['phasesp'].to_numpy()), np.concatenate(worstprecessdf['phasesp'].to_numpy())
    onset_best, onset_worst = np.array(bestprecessdf['onset'].to_numpy()), np.array(worstprecessdf['onset'].to_numpy())
    slope_best, slope_worst = np.array(bestprecessdf['slope'].to_numpy()), np.array(worstprecessdf['slope'].to_numpy())
    nidx_best, nidx_worst = np.array(bestprecessdf['nidx'].to_numpy()), np.array(worstprecessdf['nidx'].to_numpy())

    info_best = (phasesp_best, onset_best, slope_best, nidx_best)
    info_worst = (phasesp_worst, onset_worst, slope_worst, nidx_worst)
    return precessdf, info_best, info_worst



def exin_analysis(SpikeDF1, SpikeDF2, t, all_nidx, xxtun1d, yytun1d, aatun1d, sortx=True, sorty=True, sampfrac=0.6):
    edges = np.arange(-100, 100, 5)
    exindf_dict = {
        'Sim':[], 'Dissim':[], 'Best':[], 'Worst':[],
        'pairidx':[], 'ex':[], 'in':[], 'ex_bias':[],
    }
    all_sampled_xcoords = xxtun1d[all_nidx]
    all_sampled_ycoords = yytun1d[all_nidx]

    if sortx:
        sorted_idx = all_sampled_xcoords.argsort()
        sorted_sampled_nidx = all_nidx[sorted_idx]
        sorted_sampled_xcoords = all_sampled_xcoords[sorted_idx]
        sorted_sampled_ycoords = all_sampled_ycoords[sorted_idx]
    else:
        sorted_sampled_nidx = all_nidx
        sorted_sampled_xcoords = all_sampled_xcoords
        sorted_sampled_ycoords = all_sampled_xcoords

    for i in range(sorted_sampled_nidx.shape[0]):
        for j in range(i, sorted_sampled_nidx.shape[0]):
            x1, x2 = sorted_sampled_xcoords[i], sorted_sampled_xcoords[j]
            y1, y2 = sorted_sampled_ycoords[i], sorted_sampled_ycoords[j]
            if (x1 == x2) and (y1 == y2):
                continue
            nidx1, nidx2 = sorted_sampled_nidx[i], sorted_sampled_nidx[j]

            if (x1 == x2):
                if sorty:
                    if y2 < y1:
                        nidx1, nidx2 = sorted_sampled_nidx[j], sorted_sampled_nidx[i]

            # Calculate Ex/In
            tsp_diff0 = get_tspdiff(SpikeDF1.sample(frac=sampfrac, random_state=i*j, ignore_index=True), t, nidx1, nidx2)
            if (tsp_diff0.shape[0] < 2):
                continue
            tsp_diff180 = get_tspdiff(SpikeDF2.sample(frac=sampfrac, random_state=i*j, ignore_index=True), t, nidx1, nidx2)
            if (tsp_diff180.shape[0]< 2):
                continue


            # Assign categories (Sim, Dissim, Best, Worst)
            a1, a2 = aatun1d[nidx1], aatun1d[nidx2]
            absadiff = np.abs(cdiff(a1, a2))
            absadiff_a1pass = np.abs(cdiff(a1, 0))
            absadiff_a2pass = np.abs(cdiff(a2, 0))
            labels = [False, False, False, False]  # Sim, Dissim, Best, Worst
            if absadiff < (np.pi/2):  # Similar
                labels[0] = True
            if absadiff > (np.pi - np.pi/2):  # dismilar
                labels[1] = True
            if (absadiff_a1pass < (np.pi/6)) and (absadiff_a2pass < (np.pi/6)):  # Both best
                labels[2] = True
            if (absadiff_a1pass > (np.pi - np.pi/6)) and (absadiff_a2pass > (np.pi - np.pi/6)):  # Both worst
                labels[3] = True

            if np.sum(labels) < 1:
                continue
            tspdiff_bins0, _ = np.histogram(tsp_diff0, bins=edges)
            tspdiff_bins180, _ = np.histogram(tsp_diff180, bins=edges)
            ex_val, in_val, ex_bias = calc_exin_samepath(tspdiff_bins0, tspdiff_bins180)
            if ex_bias == 0:
                continue

            # Store data
            exindf_dict['Sim'].append(labels[0])
            exindf_dict['Dissim'].append(labels[1])
            exindf_dict['Best'].append(labels[2])
            exindf_dict['Worst'].append(labels[3])
            exindf_dict['pairidx'].append((nidx1, nidx2))
            exindf_dict['ex'].append(ex_val)
            exindf_dict['in'].append(in_val)
            exindf_dict['ex_bias'].append(ex_bias)

    exindf = pd.DataFrame(exindf_dict)


    exin_statdict = dict()
    for pairtype in ['Sim', 'Dissim', 'Best', 'Worst']:

        exindf_tmp = exindf[exindf[pairtype]].reset_index(drop=True)
        exin_statdict[pairtype] = dict()
        exin_statdict[pairtype]['ex_n'] = (exindf_tmp['ex_bias'] > 0).sum()
        exin_statdict[pairtype]['in_n'] = (exindf_tmp['ex_bias'] < 0).sum()
        exin_statdict[pairtype]['exin_ratio'] = exin_statdict[pairtype]['ex_n']/exin_statdict[pairtype]['in_n']
        exin_statdict[pairtype]['ex_bias_mu'] = np.mean(exindf_tmp['ex_bias'])

    return exindf, exin_statdict


def find_nidx_along_traj(traj_x, traj_y, xxtun1d, yytun1d):
    # Unique neuron indices along the trajectory
    # Sorted accordig to the first encounter
    all_nidx = np.zeros(traj_x.shape[0])
    for i in range(traj_x.shape[0]):
        run_x, run_y = traj_x[i], traj_y[i]
        nidx = np.argmin(np.square(run_x - xxtun1d) + np.square(run_y - yytun1d))
        all_nidx[i] = nidx
    uni_index = np.unique(all_nidx, return_index=True)[1]
    all_nidx_uni = all_nidx[np.sort(uni_index)]
    all_nidx_uni = all_nidx_uni.astype(int)
    return all_nidx_uni, all_nidx


def datagen_jitter(X, Y, trajtype, jitter_num, jitter_ms=2.5, startseed=None):

    M = len(X)
    N = len(X[0])
    new_X = []
    new_Y = []
    new_trajtype = []
    new_M = []
    new_jitbatch = []
    if startseed is None:
        seed = None
    else:
        seed = startseed


    for mi in range(M):

        for jitter_i in range(jitter_num):
            new_X_eachM = []
            for nj in range(N):
                tsp_ori = X[mi][nj]

                # Jittering
                num_spikes = tsp_ori.shape[0]
                if seed is not None:
                    np.random.seed(seed)
                    seed += 1
                # tsp_jittered = tsp_ori + np.random.uniform(-jitter_ms, jitter_ms, size=num_spikes)
                tsp_jittered = tsp_ori + np.random.normal(0, jitter_ms, size=num_spikes)
                new_X_eachM.append(tsp_jittered)


            new_X.append(new_X_eachM)
            new_Y.append(Y[mi])
            new_trajtype.append(trajtype[mi])
            new_M.append(mi)
            new_jitbatch.append(jitter_i)

    return np.array(new_X, dtype=object), np.array(new_Y), np.array(new_trajtype), np.array(new_M), np.array(new_jitbatch)


def directional_acc_metrics(Y, Y_pred, trajtype, num_trajtypes):
    """

    Parameters
    ----------
    Y : ndarray
        Ground true labels. One-dimensional boolean array with shape (M, ) where M = number of samples.
    Y_pred : ndarray
        Boolean array of predicted labels with shape (M, ).
    trajtype : ndarray
        Integer array with shape (M, ). Trajectory types in range [0, NT) where NT = num_trajtypes-1
    num_trajtypes : int
        Total number of trajectory types in the daya

    Returns
    -------
    Accuracy, true positive rate, true negative rate : ndarray
        Metrics for each trajectory type. Each with shape (NT, )
    Standard errors of ACC, TPR and TNR : ndarray
        Standard errors of the above 3 metrics for each trajectory type. Each with shape (NT, )
    """

    trajtype_ax = np.arange(num_trajtypes)
    acc_pera = np.zeros(num_trajtypes)
    acc_se_pera = np.zeros(num_trajtypes)
    tpr_pera = np.zeros(num_trajtypes)
    tpr_se_pera = np.zeros(num_trajtypes)
    tnr_pera = np.zeros(num_trajtypes)
    tnr_se_pera = np.zeros(num_trajtypes)
    firepercent_pera = np.zeros(num_trajtypes)

    for trajtype_i in trajtype_ax:
        mask = trajtype == trajtype_i
        masked_num = mask.sum()
        if masked_num < 1:
            acc_pera[trajtype_i], tpr_pera[trajtype_i], tnr_pera[trajtype_i] = None, None, None
            acc_se_pera[trajtype_i], tpr_se_pera[trajtype_i], tnr_se_pera[trajtype_i] = None, None, None
            firepercent_pera[trajtype_i] = None
        else:
            acc_pera[trajtype_i], tpr_pera[trajtype_i], tnr_pera[trajtype_i] = acc_metrics(Y[mask], Y_pred[mask])
            acc_se_pera[trajtype_i] = np.sqrt(acc_pera[trajtype_i] * (1-acc_pera[trajtype_i]) / masked_num)
            tpr_se_pera[trajtype_i] = np.sqrt(tpr_pera[trajtype_i] * (1-tpr_pera[trajtype_i]) / masked_num)
            tnr_se_pera[trajtype_i] = np.sqrt(tnr_pera[trajtype_i] * (1-tnr_pera[trajtype_i]) / masked_num)
            firepercent_pera[trajtype_i] = Y_pred[mask].mean()


    return (acc_pera, tpr_pera, tnr_pera), (acc_se_pera, tpr_se_pera, tnr_se_pera), firepercent_pera