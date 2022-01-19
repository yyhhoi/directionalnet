import time
import numpy as np
import pandas as pd
import os
from os.path import join
from scipy.interpolate import interp1d
from pycircstat.descriptive import resultant_vector_length, cdiff
from pycircstat.descriptive import mean as circmean
import warnings

from library.correlogram import ThetaEstimator
from library.utils import load_pickle
from library.comput_utils import check_border, IndataProcessor, midedges, segment_passes, \
    check_border_sim, append_info_from_passes, \
    DirectionerBining, DirectionerMLM, timeshift_shuffle_exp_wrapper, circular_density_1d, get_numpass_at_angle, \
    PassShuffler, find_pair_times, append_extrinsicity, dist_overlap

from library.visualization import color_wheel, directionality_polar_plot, customlegend

from library.script_wrappers import DirectionalityStatsByThresh, PrecessionProcesser, PrecessionFilter, \
    construct_passdf_sim, get_single_precessdf, compute_precessangle
from library.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw


def single_field_processing_wrapper(tunner, precesser, precess_filter,
                                    pf, mask, x_ax, y_ax, tsp, trange, NShuffles,
                                    interpolater_x, interpolater_y, interpolater_angle,
                                    dt, aedges, sp_binwidth,
                                    aedges_precess, kappa_precess):
    # Get field info
    field_area = np.sum(mask)
    field_d = np.sqrt(field_area / np.pi) * 2
    border = check_border(mask, margin=2)
    abind = aedges[1]-aedges[0]

    # Construct passes (segment & chunk)
    tok, idin = tunner.get_idin(mask, x_ax, y_ax)
    passdf = tunner.construct_singlefield_passdf(tok, tsp, interpolater_x, interpolater_y, interpolater_angle)
    allchunk_df = passdf[(~passdf['rejected'])].reset_index(drop=True)

    # Get info from passdf and interpolate
    if allchunk_df.shape[0] < 1:
        return None
    all_x_list, all_y_list = allchunk_df['x'].to_list(), allchunk_df['y'].to_list()
    all_t_list, all_passangles_list = allchunk_df['t'].to_list(), allchunk_df['angle'].to_list()
    all_tsp_list, all_chunked_list = allchunk_df['tsp'].to_list(), allchunk_df['chunked'].to_list()
    all_x = np.concatenate(all_x_list)
    all_y = np.concatenate(all_y_list)
    all_passangles = np.concatenate(all_passangles_list)
    all_tsp = np.concatenate(all_tsp_list)
    all_anglesp = np.concatenate(allchunk_df['spikeangle'].to_list())
    xsp, ysp = np.concatenate(allchunk_df['spikex'].to_list()), np.concatenate(allchunk_df['spikey'].to_list())
    pos = np.stack([all_x, all_y]).T
    possp = np.stack([xsp, ysp]).T

    # Average firing rate
    aver_rate = all_tsp.shape[0] / (all_x.shape[0] * dt)
    peak_rate = np.max(pf['rate'] * mask)

    # Field's directionality - need angle, anglesp, pos
    num_spikes = all_tsp.shape[0]
    occ_bins, _ = np.histogram(all_passangles, bins=aedges)
    mlmer = DirectionerMLM(pos, all_passangles, dt, sp_binwidth=sp_binwidth, a_binwidth=abind)
    rate_angle, rate_R, norm_prob_mlm = mlmer.get_directionality(possp, all_anglesp)

    # Time shift shuffling for rate directionality
    if np.isnan(rate_R) or (NShuffles is None):
        rate_R_pval = np.nan
    else:
        rate_R_pval = timeshift_shuffle_exp_wrapper(all_tsp_list, all_t_list, rate_R,
                                                    NShuffles, mlmer, interpolater_x,
                                                    interpolater_y, interpolater_angle, trange)

    # Precession per pass
    neuro_keys_dict = dict(tsp='tsp', spikev='spikev', spikex='spikex', spikey='spikey',
                           spikeangle='spikeangle')
    accept_mask = (~passdf['rejected']) & (passdf['chunked'] < 2)
    passdf['excluded_for_precess'] = ~accept_mask
    precessdf, precess_angle, precess_R, _ = get_single_precessdf(passdf, precesser, precess_filter,
                                                                  neuro_keys_dict,
                                                                  field_d=field_d, kappa=kappa_precess, bins=None)
    fitted_precessdf = precessdf[precessdf['fitted']].reset_index(drop=True)
    # Proceed only if precession exists
    if (precess_angle is not None) and (fitted_precessdf['precess_exist'].sum() > 1):

        # Post-hoc precession exclusion
        _, binR, postdoc_dens = compute_precessangle(pass_angles=fitted_precessdf['mean_anglesp'].to_numpy(),
                                                     pass_nspikes=fitted_precessdf['pass_nspikes'].to_numpy(),
                                                     precess_mask=fitted_precessdf['precess_exist'].to_numpy(),
                                                     kappa=None, bins=aedges_precess)
        (_, passbins_p, passbins_np, _) = postdoc_dens
        all_passbins = passbins_p + passbins_np
        numpass_at_precess = get_numpass_at_angle(target_angle=precess_angle, aedge=aedges_precess,
                                                  all_passbins=all_passbins)

        precess_R_pval = 1
    else:
        numpass_at_precess = None
        precess_R_pval = None

    normal_info = (num_spikes, field_area, border, aver_rate, peak_rate, tok)
    direct_info = (rate_angle, rate_R, rate_R_pval)
    precess_info = (fitted_precessdf, precess_angle, precess_R, precess_R_pval, numpass_at_precess)
    return normal_info, direct_info, precess_info

def compute_lowspikeprecession(df, lqr=11):
    aedges_precess = np.linspace(-np.pi, np.pi, 6)
    kappa_precess = 1

    precess_angle_low_list = []
    numpass_at_precess_low_list = []
    for i in range(df.shape[0]):
        print('\rComputing low-spike precession %d/%d' % (i, df.shape[0]), flush=True, end='')
        precess_df = df.loc[i, 'precess_df']

        # Precession - low-spike passes
        ldf = precess_df[precess_df['pass_nspikes'] < lqr]  # 25% quantile
        if (ldf.shape[0] > 0) and (ldf['precess_exist'].sum() > 1):
            precess_angle_low, _, _ = compute_precessangle(pass_angles=ldf['mean_anglesp'].to_numpy(),
                                                           pass_nspikes=ldf['pass_nspikes'].to_numpy(),
                                                           precess_mask=ldf['precess_exist'].to_numpy(),
                                                           kappa=kappa_precess, bins=None)
            _, _, postdoc_dens_low = compute_precessangle(pass_angles=ldf['mean_anglesp'].to_numpy(),
                                                          pass_nspikes=ldf['pass_nspikes'].to_numpy(),
                                                          precess_mask=ldf['precess_exist'].to_numpy(),
                                                          kappa=None, bins=aedges_precess)
            (_, passbins_p_low, passbins_np_low, _) = postdoc_dens_low
            all_passbins_low = passbins_p_low + passbins_np_low
            numpass_at_precess_low = get_numpass_at_angle(target_angle=precess_angle_low, aedge=aedges_precess,
                                                          all_passbins=all_passbins_low)
        else:
            precess_angle_low = None
            numpass_at_precess_low = None

        precess_angle_low_list.append(precess_angle_low)
        numpass_at_precess_low_list.append(numpass_at_precess_low)
    print()
    df['precess_angle_low'] = precess_angle_low_list
    df['numpass_at_precess_low'] = numpass_at_precess_low_list
    return df




def SingleField_Preprocess(processed1data, save_pth):
    fielddf_dict = dict(cell_id=[], field_id=[], num_spikes=[], border=[], aver_rate=[], peak_rate=[],
                        rate_angle=[], rate_R=[], rate_R_pval=[], field_area=[], field_bound=[],
                        precess_df=[], precess_angle=[], precess_R=[], precess_R_pval=[],
                        numpass_at_precess=[])

    NeuronDF = processed1data['NeuronDF']
    BehDF = processed1data['BehDF']
    occ_dict = processed1data['occupancy']
    wave = dict(tax=BehDF.t.to_numpy(), phase=BehDF.I_phase.to_numpy(), theta=np.ones(BehDF.shape[0]))
    vthresh = 3
    sthresh = 80
    NShuffles = 200
    num_neurons = NeuronDF.shape[0]
    aedges = np.linspace(-np.pi, np.pi, 36)
    sp_binwidth = 5
    aedges_precess = np.linspace(-np.pi, np.pi, 6)
    kappa_precess = 1
    precess_filter = PrecessionFilter()
    smoothkernel = None
    tunner = IndataProcessor(BehDF, vthresh=vthresh, sthresh=sthresh, minpasstime=0.4, smooth=smoothkernel)
    interpolater_angle = interp1d(tunner.t, tunner.angle)
    interpolater_x = interp1d(tunner.t, tunner.x)
    interpolater_y = interp1d(tunner.t, tunner.y)
    trange = (tunner.t.max(), tunner.t.min())
    dt = tunner.t[1] - tunner.t[0]
    precesser = PrecessionProcesser(wave=wave)
    precesser.set_trange(trange)
    xx, yy = occ_dict['xx'], occ_dict['yy']
    x_ax, y_ax = xx[0, :], yy[:, 0]
    for neuron_i in range(num_neurons):

        field_df = NeuronDF.loc[neuron_i, 'fields']
        tsp = NeuronDF.loc[neuron_i, 'tsp']
        pf = NeuronDF.loc[neuron_i, 'pf']
        num_fields = field_df.shape[0]

        for nf in range(num_fields):
            print('%d/%d neuron, %d/%d field' % (neuron_i, num_neurons, nf, num_fields))
            field_id, mask, xyval = field_df.loc[nf, ['field_id', 'mask', 'xyval']]

            output = single_field_processing_wrapper(tunner, precesser, precess_filter,
                                                     pf, mask, x_ax, y_ax, tsp, trange, NShuffles,
                                                     interpolater_x, interpolater_y, interpolater_angle,
                                                     dt, aedges, sp_binwidth,
                                                     aedges_precess, kappa_precess)
            if output is None:
                continue

            normal_info, direct_info, precess_info = output
            num_spikes, field_area, border, aver_rate, peak_rate, _ = normal_info
            rate_angle, rate_R, rate_R_pval = direct_info
            fitted_precessdf, precess_angle, precess_R, precess_R_pval, numpass_at_precess = precess_info


            fielddf_dict['cell_id'].append(neuron_i)
            fielddf_dict['field_id'].append(field_id)
            fielddf_dict['num_spikes'].append(num_spikes)
            fielddf_dict['field_area'].append(field_area)
            fielddf_dict['field_bound'].append(xyval)
            fielddf_dict['border'].append(border)
            fielddf_dict['aver_rate'].append(aver_rate)
            fielddf_dict['peak_rate'].append(peak_rate)
            fielddf_dict['rate_angle'].append(rate_angle)
            fielddf_dict['rate_R'].append(rate_R)
            fielddf_dict['rate_R_pval'].append(rate_R_pval)
            fielddf_dict['precess_df'].append(fitted_precessdf)
            fielddf_dict['precess_angle'].append(precess_angle)
            fielddf_dict['precess_R'].append(precess_R)
            fielddf_dict['precess_R_pval'].append(precess_R_pval)
            fielddf_dict['numpass_at_precess'].append(numpass_at_precess)

    fielddf_raw = pd.DataFrame(fielddf_dict)
    # Compute low-spike precession
    all_pdf = pd.concat(fielddf_raw['precess_df'].to_list(), axis=0, ignore_index=True)
    pass_nspikes = all_pdf[all_pdf['precess_exist']]['pass_nspikes'].to_numpy()
    lqr = np.quantile(pass_nspikes, 0.25)
    print('LQR of pass spikes = %0.2f'%(lqr))
    fielddf_raw = compute_lowspikeprecession(fielddf_raw, lqr)
    # Save data
    fielddf_raw.to_pickle(save_pth)
    return None


def PairField_Preprocess(processed1data, save_pth=None):
    pairdata_dict = dict(field_id1=[], field_id2=[], overlap=[],
                         border1=[], border2=[], field_area1=[], field_area2=[], xyval1=[], xyval2=[],
                         aver_rate1=[], aver_rate2=[], aver_rate_pair=[],
                         peak_rate1=[], peak_rate2=[],
                         fieldcoor1=[], fieldcoor2=[], com1=[], com2=[],

                         rate_angle1=[], rate_angle2=[], rate_anglep=[],
                         rate_R1=[], rate_R2=[], rate_Rp=[],

                         num_spikes1=[], num_spikes2=[], num_spikes_pair=[],
                         phaselag_AB=[], phaselag_BA=[], corr_info_AB=[], corr_info_BA=[],
                         thetaT_AB=[], thetaT_BA=[],
                         rate_AB=[], rate_BA=[], corate=[], pair_rate=[],
                         rate_R_pvalp=[],
                         precess_df1=[], precess_angle1=[], precess_R1=[],
                         precess_df2=[], precess_angle2=[], precess_R2=[],
                         numpass_at_precess1=[], numpass_at_precess2=[],
                         precess_dfp=[])
    NeuronDF = processed1data['NeuronDF']
    BehDF = processed1data['BehDF']
    occ_dict = processed1data['occupancy']
    theta_freq = processed1data['theta_freq']
    PairDF = processed1data['PairDF']
    wave = dict(tax=BehDF.t.to_numpy(), phase=BehDF.I_phase.to_numpy(), theta=np.ones(BehDF.shape[0]))

    vthresh = 3
    sthresh = 80
    NShuffles = 200
    num_neurons = NeuronDF.shape[0]
    aedges = np.linspace(-np.pi, np.pi, 36)
    abind = aedges[1] - aedges[0]
    sp_binwidth = 5
    aedges_precess = np.linspace(-np.pi, np.pi, 6)
    kappa_precess = 1
    precess_filter = PrecessionFilter()
    lqr_passnspikes = 11
    smoothkernel = np.concatenate([np.zeros(100), np.ones(500)])/500  # moving average of 500ms-window
    tunner = IndataProcessor(BehDF, vthresh=vthresh, sthresh=sthresh, minpasstime=0.4, smooth=smoothkernel)
    interpolater_angle = interp1d(tunner.t, tunner.angle)
    interpolater_x = interp1d(tunner.t, tunner.x)
    interpolater_y = interp1d(tunner.t, tunner.y)

    trange = (tunner.t.max(), tunner.t.min())
    dt = tunner.t[1] - tunner.t[0]
    precesser = PrecessionProcesser(wave=wave)
    precesser.set_trange(trange)
    xx, yy = occ_dict['xx'], occ_dict['yy']
    x_ax, y_ax = xx[0, :], yy[:, 0]

    num_pairs = PairDF.shape[0]
    for npair in range(num_pairs):
        print('%d/%d pair' % (npair, num_pairs))
        cell_id1, cell_id2, field_id1, field_id2 = PairDF.loc[npair, ['cell_id1', 'cell_id2', 'field_id1', 'field_id2']]

        # # Single field 1
        field_df1 = NeuronDF.loc[cell_id1, 'fields']
        field_df1 = field_df1[field_df1.field_id == field_id1].reset_index(drop=True)
        tsp1 = NeuronDF.loc[cell_id1, 'tsp']
        pf1 = NeuronDF.loc[cell_id1, 'pf']
        mask1, xyval1 = field_df1.loc[0, ['mask', 'xyval']]

        output1 = single_field_processing_wrapper(tunner, precesser, precess_filter,
                                                 pf1, mask1, x_ax, y_ax, tsp1, trange, None,
                                                 interpolater_x, interpolater_y, interpolater_angle,
                                                 dt, aedges, sp_binwidth,
                                                 aedges_precess, kappa_precess)
        if output1 is None:
            continue
        normal_info1, direct_info1, precess_info1 = output1
        num_spikes1, field_area1, border1, aver_rate1, peak_rate1, tok1 = normal_info1
        rate_angle1, rate_R1, rate_R_pval1 = direct_info1
        fitted_precessdf1, precess_angle1, precess_R1, precess_R_pval1, numpass_at_precess1 = precess_info1
        maskedmap1 = pf1['rate'] * mask1
        cooridx1 = np.unravel_index(maskedmap1.argmax(), maskedmap1.shape)
        fcoor1 = np.array([xx[cooridx1[0], cooridx1[1]], yy[cooridx1[0], cooridx1[1]]])
        XY = np.stack([xx.ravel(), yy.ravel()])
        com1 = np.sum(XY * (maskedmap1/np.sum(maskedmap1)).ravel().reshape(1, -1), axis=1)

        # # Single field 2
        field_df2 = NeuronDF.loc[cell_id2, 'fields']
        field_df2 = field_df2[field_df2.field_id == field_id2].reset_index(drop=True)
        tsp2 = NeuronDF.loc[cell_id2, 'tsp']
        pf2 = NeuronDF.loc[cell_id2, 'pf']
        mask2, xyval2 = field_df2.loc[0, ['mask', 'xyval']]
        output2 = single_field_processing_wrapper(tunner, precesser, precess_filter,
                                                  pf2, mask2, x_ax, y_ax, tsp2, trange, None,
                                                  interpolater_x, interpolater_y, interpolater_angle,
                                                  dt, aedges, sp_binwidth,
                                                  aedges_precess, kappa_precess, )
        if output2 is None:
            continue
        normal_info2, direct_info2, precess_info2 = output2
        num_spikes2, field_area2, border2, aver_rate2, peak_rate2, tok2 = normal_info2
        rate_angle2, rate_R2, rate_R_pval2 = direct_info2
        fitted_precessdf2, precess_angle2, precess_R2, precess_R_pval2, numpass_at_precess2 = precess_info2
        maskedmap2 = pf2['rate'] * mask2
        cooridx2 = np.unravel_index(maskedmap2.argmax(), maskedmap2.shape)
        fcoor2 = np.array([xx[cooridx2[0], cooridx2[1]], yy[cooridx2[0], cooridx2[1]]])
        XY = np.stack([xx.ravel(), yy.ravel()])
        com2 = np.sum(XY * (maskedmap2/np.sum(maskedmap2)).ravel().reshape(1, -1), axis=1)

        # # Pair
        # Find overlap
        _, ks_dist, _ = dist_overlap(pf1['rate'], pf2['rate'], mask1, mask2)

        # Construct pairedpasses
        mask_union = mask1 | mask2
        field_d_union = np.sqrt(mask_union.sum()/np.pi)*2
        tok_pair, _ = tunner.get_idin(mask_union, x_ax, y_ax)
        pairedpasses = tunner.construct_pairfield_passdf(tok_pair, tok1, tok2, tsp1, tsp2, interpolater_x,
                                                         interpolater_y, interpolater_angle)

        # Phase lags
        phase_finder = ThetaEstimator(0.005, 0.3, [5, 12])
        AB_tsp1_list, BA_tsp1_list = [], []
        AB_tsp2_list, BA_tsp2_list = [], []
        nspikes_AB_list, nspikes_BA_list = [], []
        duration_AB_list, duration_BA_list = [], []
        t_all = []
        passangles_all, x_all, y_all = [], [], []
        paired_tsp_list = []


        accepted_df = pairedpasses[(~pairedpasses['rejected'])].reset_index(drop=True)
        for npass in range(accepted_df.shape[0]):

            # Get data
            t, tsp1, tsp2 = accepted_df.loc[npass, ['t', 'tsp1', 'tsp2']]
            x, y, pass_angles, v, direction = accepted_df.loc[npass, ['x', 'y', 'angle', 'v', 'direction']]
            duration = t.max() - t.min()


            # Find paired spikes
            pairidx1, pairidx2 = find_pair_times(tsp1, tsp2, tdiff=1/theta_freq)
            paired_tsp1, paired_tsp2 = tsp1[pairidx1], tsp2[pairidx2]
            if (paired_tsp1.shape[0] < 1) and (paired_tsp2.shape[0] < 1):
                continue
            paired_tsp_eachpass = np.concatenate([paired_tsp1, paired_tsp2])
            paired_tsp_list.append(paired_tsp_eachpass)
            passangles_all.append(pass_angles)
            x_all.append(x)
            y_all.append(y)
            t_all.append(t)
            if direction == 'A->B':
                AB_tsp1_list.append(tsp1)
                AB_tsp2_list.append(tsp2)
                nspikes_AB_list.append(tsp1.shape[0] + tsp2.shape[0])
                duration_AB_list.append(duration)

            elif direction == 'B->A':
                BA_tsp1_list.append(tsp1)
                BA_tsp2_list.append(tsp2)
                nspikes_BA_list.append(tsp1.shape[0] + tsp2.shape[0])
                duration_BA_list.append(duration)

        # Phase lags
        thetaT_AB, phaselag_AB, corr_info_AB = phase_finder.find_theta_isi_hilbert(AB_tsp1_list, AB_tsp2_list)
        thetaT_BA, phaselag_BA, corr_info_BA = phase_finder.find_theta_isi_hilbert(BA_tsp1_list, BA_tsp2_list)

        # Pair precession
        neuro_keys_dict1 = dict(tsp='tsp1', spikev='spike1v', spikex='spike1x', spikey='spike1y',
                                spikeangle='spike1angle')
        neuro_keys_dict2 = dict(tsp='tsp2', spikev='spike2v', spikex='spike2x', spikey='spike2y',
                                spikeangle='spike2angle')



        accept_mask = (~pairedpasses['rejected']) & (pairedpasses['chunked']<2) & ((pairedpasses['direction']=='A->B')| (pairedpasses['direction']=='B->A'))

        pairedpasses['excluded_for_precess'] = ~accept_mask
        precess_dfp = precesser.get_single_precession(pairedpasses, neuro_keys_dict1, field_d_union, tag='1')
        precess_dfp = precesser.get_single_precession(precess_dfp, neuro_keys_dict2, field_d_union, tag='2')
        precess_dfp = precess_filter.filter_pair(precess_dfp)
        fitted_precess_dfp = precess_dfp[precess_dfp['fitted1'] & precess_dfp['fitted2']].reset_index(drop=True)

        # Paired spikes
        if (len(paired_tsp_list) == 0) or (len(passangles_all) == 0):
            continue
        hd_pair = np.concatenate(passangles_all)
        x_pair, y_pair = np.concatenate(x_all), np.concatenate(y_all)
        pos_pair = np.stack([x_pair, y_pair]).T
        paired_tsp = np.concatenate(paired_tsp_list)
        paired_tsp = paired_tsp[(paired_tsp <= trange[0]) & (paired_tsp >= trange[1])]
        if paired_tsp.shape[0] < 1:
            continue
        num_spikes_pair = paired_tsp.shape[0]
        hdsp_pair = interpolater_angle(paired_tsp)
        xsp_pair = interpolater_x(paired_tsp)
        ysp_pair = interpolater_y(paired_tsp)
        possp_pair = np.stack([xsp_pair, ysp_pair]).T
        aver_rate_pair = num_spikes_pair / (x_pair.shape[0] * dt)

        # Pair Directionality
        occbinsp, _ = np.histogram(hd_pair, bins=aedges)
        spbinsp, _ = np.histogram(hdsp_pair, bins=aedges)
        mlmer_pair = DirectionerMLM(pos_pair, hd_pair, dt, sp_binwidth, abind)
        rate_anglep, rate_Rp, normprobp_mlm = mlmer_pair.get_directionality(possp_pair, hdsp_pair)
        normprobp_mlm[np.isnan(normprobp_mlm)] = 0

        # Time shift shuffling
        if np.isnan(rate_Rp):
            rate_R_pvalp = np.nan
        else:
            rate_R_pvalp = timeshift_shuffle_exp_wrapper(paired_tsp_list, t_all, rate_Rp,
                                                         NShuffles, mlmer_pair,
                                                         interpolater_x, interpolater_y,
                                                         interpolater_angle, trange)

        # Rates
        with np.errstate(divide='ignore', invalid='ignore'):  # None means no sample
            rate_AB = np.sum(nspikes_AB_list) / np.sum(duration_AB_list)
            rate_BA = np.sum(nspikes_BA_list) / np.sum(duration_BA_list)
            corate = np.sum(nspikes_AB_list + nspikes_BA_list) / np.sum(duration_AB_list + duration_BA_list)
            pair_rate = num_spikes_pair / np.sum(duration_AB_list + duration_BA_list)


        pairdata_dict['field_id1'].append(field_id1)
        pairdata_dict['field_id2'].append(field_id2)
        pairdata_dict['overlap'].append(ks_dist)
        pairdata_dict['border1'].append(border1)
        pairdata_dict['border2'].append(border2)
        pairdata_dict['field_area1'].append(mask1.sum())
        pairdata_dict['field_area2'].append(mask2.sum())
        pairdata_dict['xyval1'].append(xyval1)
        pairdata_dict['xyval2'].append(xyval2)

        pairdata_dict['aver_rate1'].append(aver_rate1)
        pairdata_dict['aver_rate2'].append(aver_rate2)
        pairdata_dict['aver_rate_pair'].append(aver_rate_pair)
        pairdata_dict['peak_rate1'].append(peak_rate1)
        pairdata_dict['peak_rate2'].append(peak_rate2)
        pairdata_dict['fieldcoor1'].append(fcoor1)
        pairdata_dict['fieldcoor2'].append(fcoor2)
        pairdata_dict['com1'].append(com1)
        pairdata_dict['com2'].append(com2)

        pairdata_dict['rate_angle1'].append(rate_angle1)
        pairdata_dict['rate_angle2'].append(rate_angle2)
        pairdata_dict['rate_anglep'].append(rate_anglep)
        pairdata_dict['rate_R1'].append(rate_R1)
        pairdata_dict['rate_R2'].append(rate_R2)
        pairdata_dict['rate_Rp'].append(rate_Rp)

        pairdata_dict['num_spikes1'].append(num_spikes1)
        pairdata_dict['num_spikes2'].append(num_spikes2)
        pairdata_dict['num_spikes_pair'].append(num_spikes_pair)

        pairdata_dict['phaselag_AB'].append(phaselag_AB)
        pairdata_dict['phaselag_BA'].append(phaselag_BA)
        pairdata_dict['corr_info_AB'].append(corr_info_AB)
        pairdata_dict['corr_info_BA'].append(corr_info_BA)
        pairdata_dict['thetaT_AB'].append(thetaT_AB)
        pairdata_dict['thetaT_BA'].append(thetaT_BA)

        pairdata_dict['rate_AB'].append(rate_AB)
        pairdata_dict['rate_BA'].append(rate_BA)
        pairdata_dict['corate'].append(corate)
        pairdata_dict['pair_rate'].append(pair_rate)
        pairdata_dict['rate_R_pvalp'].append(rate_R_pvalp)

        pairdata_dict['precess_df1'].append(fitted_precessdf1)
        pairdata_dict['precess_angle1'].append(precess_angle1)
        pairdata_dict['precess_R1'].append(precess_R1)
        pairdata_dict['numpass_at_precess1'].append(numpass_at_precess1)
        pairdata_dict['precess_df2'].append(fitted_precessdf2)
        pairdata_dict['precess_angle2'].append(precess_angle2)
        pairdata_dict['precess_R2'].append(precess_R2)
        pairdata_dict['numpass_at_precess2'].append(numpass_at_precess2)
        pairdata_dict['precess_dfp'].append(fitted_precess_dfp)


    pairdata = pd.DataFrame(pairdata_dict)
    pairdata = append_extrinsicity(pairdata)
    pairdata.to_pickle(save_pth)
    return pairdata




if __name__ == '__main__':

    processed1 = load_pickle('data/processed1_Idirect-10.pickle')
    save_pth = 'data/Single_processed2_Idirect-10_novthresh.pickle'
    SingleField_Preprocess(processed1data=processed1, save_pth=save_pth)

    # processed1 = load_pickle('data/processed1_Idirect-10.pickle')
    # save_pth = 'data/Pair_processed2_Idirect-10.pickle'
    # PairField_Preprocess(processed1data=processed1, save_pth=save_pth)
