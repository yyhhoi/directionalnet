#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Wrappers enclosing scripts which are invoked more than once

"""

import numpy as np
import pandas as pd
from pycircstat.tests import rayleigh
from scipy.interpolate import interp1d
from pycircstat.descriptive import mean as cmean, cdiff, resultant_vector_length

from library.comput_utils import rcc_wrapper
from library.linear_circular_r import rcc



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
