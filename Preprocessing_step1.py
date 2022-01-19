# This script performs the first (stable) step of preprocessing:
# 1. Sample spikes from firing rates to a right number (~500, determined by experimental data), with
# 5ms refractory period.
# 2. Compute occupancy map
# 3. Compute firing rate map and segment place fields
# 4. Identify pairs based on number of paired spikes (|delta t|<theta_T) and field mask overlap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch

from os.path import join
from itertools import combinations
from scipy.interpolate import interp1d
from library.preprocess import get_occupancy, placefield, segment_fields
from library.comput_utils import find_pair_times, poisson_sampling
from library.utils import load_pickle



def PreprocessStep1(simdata, save_pth):
    NeuronDF = simdata['NeuronDF']
    BehDF = simdata['BehDF']
    Activity = simdata['Activity']
    theta_freq = simdata['theta_freq']

    # # Sample spikes
    print('Sampling spikes')
    dt = np.diff(BehDF['t']).mean()
    seedcount = 0
    spdict = {i: [] for i in NeuronDF.index}
    median_nsp = 0
    max_nsp = 400
    while median_nsp < max_nsp:
        for nidx in NeuronDF.index:
            rate_slice = Activity[:, nidx]
            tidxsp = poisson_sampling(rate_slice, dt, refrac=0, seed=seedcount)
            spdict[nidx].extend(tidxsp)
            seedcount += 1
        median_nsp = np.median([len(val) for key, val in spdict.items()])
        print('Median spike count =%d' % median_nsp)
    subsample_frac = max_nsp/median_nsp
    sorted_tidxsp_list = []
    sorted_tsp_list = []
    for i in NeuronDF.index:
        tidxsp_tmp = np.array(spdict[i])
        subsampled_n = int(tidxsp_tmp.shape[0] * subsample_frac)
        np.random.seed(i)
        tidxsp_tmp2 = np.random.choice(tidxsp_tmp, size=subsampled_n, replace=False)
        tidxsp = np.sort(tidxsp_tmp2)
        sorted_tidxsp_list.append(tidxsp)
        sorted_tsp_list.append(BehDF.loc[tidxsp, 't'].to_numpy())
    NeuronDF['tidxsp'] = sorted_tidxsp_list
    NeuronDF['tsp'] = sorted_tsp_list
    print('Final median spike count = %0.2f'%(NeuronDF['tidxsp'].apply(lambda x: x.shape[0]).median()))

    # # Occupancy
    print('Computing Occupancy')
    x = BehDF['x'].to_numpy()
    y = BehDF['y'].to_numpy()
    t = BehDF['t'].to_numpy()
    xx, yy, occupancy = get_occupancy(x, y, t)
    occ_dict = dict(xx=xx, yy=yy, occ=occupancy)
    xbound = (0, xx.shape[1] - 1)
    ybound = (0, yy.shape[0] - 1)
    x_ax, y_ax = xx[0, :], yy[:, 0]

    # # Rate map and field segmentation
    pf_list = []
    fields_list = []
    field_id = 0
    for nidx in NeuronDF.index:
        print("\r%d/%d Neurons - Computing rate map" % (nidx, NeuronDF.shape[0]), flush=True, end='')

        # Rate map
        fielddf_dict = dict(cellid=[], field_id=[], mask=[], xyval=[])
        tidxsp = NeuronDF.loc[nidx, 'tidxsp']

        if tidxsp.shape[0] < 1:
            fields_list.append(pd.DataFrame(fielddf_dict))
            pf_list.append(None)
            continue
        else:
            tsp = t[tidxsp]
            xsp = x[tidxsp]
            ysp = y[tidxsp]
            freq, rate = placefield(xx, yy, occupancy, xsp, ysp, tsp)
            pf = dict(freq=freq, rate=rate)
            pf_list.append(pf)
            # Segment placefields
            for mask, xyval in segment_fields(xx, yy, freq, rate, 25, freq_thresh=0):
                fielddf_dict['cellid'].append(nidx)
                fielddf_dict['field_id'].append(field_id)
                fielddf_dict['mask'].append(mask)
                fielddf_dict['xyval'].append(xyval)
                field_id += 1
            fielddf = pd.DataFrame(fielddf_dict)
            fields_list.append(fielddf)
    NeuronDF['pf'] = pf_list
    NeuronDF['fields'] = fields_list

    # # Pair identification
    print('\nIdentifying pairs')
    max_samples = 3600
    attempt_count = 0
    all_pairids = []
    allfields_df = pd.concat(NeuronDF['fields'].to_list(), axis=0, ignore_index=True)
    num_fields = allfields_df.shape[0]

    while len(all_pairids) < max_samples:
        # Sample apair
        np.random.seed(attempt_count)
        sampled_pairs = np.random.choice(num_fields, 2, replace=False)

        # Check if the same pair has been sampled before
        if len(all_pairids) > 2:
            admitted_pairs = np.array(all_pairids)
            samemask1 = (np.abs(admitted_pairs[:, [2, 3]] - sampled_pairs.reshape(1, 2)) == 0)
            samemask2 = (np.abs(admitted_pairs[:, [3, 2]] - sampled_pairs.reshape(1, 2)) == 0)
            num_samepair1 = (np.sum(samemask1, axis=1) > 1).sum()
            num_samepair2 = (np.sum(samemask2, axis=1) > 1).sum()
            if (num_samepair1 > 0) or (num_samepair2 > 0):
                print('\nSame pair is found, skipe')
                attempt_count += 1
                continue

        # Get data
        field_rowid1, field_rowid2 = sampled_pairs[0], sampled_pairs[1]
        print('\rid1=%d, id2=%d, admitted=%d, attempted=%d' % (field_rowid1, field_rowid2, len(all_pairids), attempt_count),
              flush=True, end=' ' * 10)
        cellid1, field_id1, mask1 = allfields_df.loc[field_rowid1, ['cellid', 'field_id', 'mask']]
        tsp1, tidxsp1, pf1, neu1x, neu1y = NeuronDF.loc[cellid1, ['tsp', 'tidxsp', 'pf', 'neuronx', 'neurony']]
        cellid2, field_id2, mask2 = allfields_df.loc[field_rowid2, ['cellid', 'field_id', 'mask']]
        tsp2, tidxsp2, pf2, neu2x, neu2y = NeuronDF.loc[cellid2, ['tsp', 'tidxsp', 'pf', 'neuronx', 'neurony']]
        if (neu1x == neu2x) & (neu1y == neu2y):  # neurons with same x-, y- positions are not considered
            attempt_count += 1
            continue

        # Criteria
        theta_T = 1 / theta_freq
        enough_npairs = False
        total_npairs = 0
        for tsp1_each in tsp1:
            masktmp = np.abs(tsp2 - tsp1_each) < theta_T
            total_npairs += masktmp.sum()
            if total_npairs > 15:
                enough_npairs = True
                break
        mask_and = mask1 & mask2
        if enough_npairs & (mask_and.sum() > 0):
            all_pairids.append([cellid1, cellid2, field_id1, field_id2])
        attempt_count += 1


    # # Save
    simdata['NeuronDF'] = NeuronDF
    simdata['occupancy'] = occ_dict
    simdata['PairDF'] = pd.DataFrame(all_pairids, columns=['cell_id1', 'cell_id2', 'field_id1', 'field_id2'])
    if save_pth:
        with open(join(save_pth), 'wb') as fh:
            pickle.dump(simdata, fh)
    return simdata


