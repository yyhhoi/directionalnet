from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from pycircstat import vtest, watson_williams, rayleigh
from pycircstat.descriptive import cdiff
from pycircstat.descriptive import mean as circmean
from scipy.stats import ranksums, chi2_contingency, spearmanr, chisquare, ttest_1samp, linregress, binom_test

from Preprocessing_step2 import compute_lowspikeprecession
from library.linear_circular_r import rcc
from library.utils import load_pickle
from library.stattests import p2str, stat_record, wwtable2text, my_ww_2samp, my_kruskal_2samp, my_chi_2way, \
    my_fisher_2way, my_kruskal_3samp, my_chi_1way, my_ttest_1samp
from library.comput_utils import midedges, circular_density_1d, repeat_arr, unfold_binning_2d, \
    linear_circular_gauss_density, circ_ktest, ranksums, shiftcyc_full2half, get_numpass_at_angle

from library.visualization import plot_marginal_slices, plot_correlogram, customlegend

from library.script_wrappers import DirectionalityStatsByThresh, permutation_test_average_slopeoffset, \
    permutation_test_arithmetic_average_slopeoffset, compute_precessangle
from library.shared_vars import total_figw, fontsize, ticksize, legendsize, titlesize, dpi
import warnings
warnings.filterwarnings("ignore")


def omniplot_singlefields_Romani(simdf, ax):


    stat_fn = 'fig8_SIM_single_directionality.txt'
    stat_record(stat_fn, True)

    linew = 0.75

    spike_threshs = np.arange(0, 701, 25)
    stats_getter = DirectionalityStatsByThresh('num_spikes', 'rate_R_pval', 'rate_R')
    linecolor = {'all':'k', 'border':'k', 'nonborder':'k'}
    linestyle = {'all':'solid', 'border':'dotted', 'nonborder':'dashed'}

    # Plot all
    all_dict = stats_getter.gen_directionality_stats_by_thresh(simdf, spike_threshs)
    ax[0].plot(spike_threshs, all_dict['medianR'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)
    ax[1].plot(spike_threshs, all_dict['sigfrac_shift'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)


    # Plot border
    simdf_b = simdf[simdf['border']].reset_index(drop=True)
    border_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_b, spike_threshs)
    ax[0].plot(spike_threshs, border_dict['medianR'], c=linecolor['border'], linestyle=linestyle['border'], label='B', linewidth=linew)
    ax[1].plot(spike_threshs, border_dict['sigfrac_shift'], c=linecolor['border'], linestyle=linestyle['border'], label='B', linewidth=linew)

    # Plot non-border
    simdf_nb = simdf[~simdf['border']].reset_index(drop=True)
    nonborder_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_nb, spike_threshs)
    ax[0].plot(spike_threshs, nonborder_dict['medianR'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='N-B', linewidth=linew)
    ax[1].plot(spike_threshs, nonborder_dict['sigfrac_shift'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='N-B', linewidth=linew)

    # Plot Fraction
    all_n = simdf.shape[0]
    border_nfrac = border_dict['n']/all_n
    nonborder_nfrac = nonborder_dict['n']/all_n
    ax[2].plot(spike_threshs, all_dict['datafrac'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)
    ax[2].plot(spike_threshs, border_nfrac, c=linecolor['border'], linestyle=linestyle['border'], label='B', linewidth=linew)
    ax[2].plot(spike_threshs, nonborder_nfrac, c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='N-B', linewidth=linew)


    # Binomial test for all fields
    signum_all, n_all = all_dict['shift_signum'][0], all_dict['n'][0]
    p_binom = binom_test(signum_all, n_all, p=0.05, alternative='greater')
    stat_txt = r'Binomial test, greater than p=0.05, %d/%d=%0.4f, $p=%s$'%(signum_all, n_all, signum_all/n_all, p2str(p_binom))
    stat_record(stat_fn, False, stat_txt)
    # ax[1].annotate('Sig. Frac. (All)\n%d/%d=%0.3f\np%s'%(signum_all, n_all, signum_all/n_all, p2str(p_binom)), xy=(0.1, 0.5), xycoords='axes fraction', fontsize=legendsize)

    # # Statistical test
    for idx, ntresh in enumerate(spike_threshs):
        stat_record(stat_fn, False, '======= Threshold=%d ======'%(ntresh))
        # KW test for border median R
        try:
            rs_bord_pR, (border_n, nonborder_n), mdns, rs_txt = my_kruskal_2samp(border_dict['allR'][idx], nonborder_dict['allR'][idx], 'border', 'nonborder')
        except IndexError:
            continue

        stat_record(stat_fn, False, "SIM, Median R, border vs non-border: %s" % (rs_txt))

        # Chisquared test for border fractions
        contin = pd.DataFrame({'border': [border_dict['shift_signum'][idx],
                                          border_dict['shift_nonsignum'][idx]],
                               'nonborder': [nonborder_dict['shift_signum'][idx],
                                             nonborder_dict['shift_nonsignum'][idx]]}).to_numpy()
        chi_pborder, _, txt_chiborder = my_chi_2way(contin)
        _, _, fishtxt = my_fisher_2way(contin)
        stat_record(stat_fn, False, "SIM, Significant fraction, border vs non-border: %s, %s" % (txt_chiborder, fishtxt))




    # Plotting asthestic
    ax_ylabels = ['Median R', "Sig. Frac.", "Data Frac."]
    for axid in range(3):
        # ax[axid].set_xticks(np.arange(0, spike_threshs.max()+1, 100))
        # ax[axid].set_xticks(np.arange(0, spike_threshs.max()+1, 50), minor=True)

        ax[axid].set_ylabel(ax_ylabels[axid], fontsize=fontsize)
        ax[axid].tick_params(axis='both', which='major', labelsize=ticksize)
        ax[axid].spines['top'].set_visible(False)
        ax[axid].spines['right'].set_visible(False)

    ax[1].set_xlabel('Spike count threshold', fontsize=fontsize)

    customlegend(ax[0], fontsize=legendsize, loc='lower left', bbox_to_anchor=(0.5, 0.5))

    ax[0].set_yticks([0, 0.1, 0.2, 0.3])


    ax[1].set_yticks(np.arange(0, 1.2, 0.2))
    ax[1].set_yticks(np.arange(0, 1.2, 0.1), minor=True)

    ax[2].set_yticks([0, 0.5, 1])
    ax[2].set_yticklabels(['0', '', '1'])
    ax[2].set_yticks(np.arange(0, 1.1, 0.1), minor=True)


def plot_field_bestprecession_Romani(simdf, ax):



    print('Plot field best precession')
    stat_fn = 'fig8_SIM_field_precess.txt'
    stat_record(stat_fn, True)
    nap_thresh = 1

    # Stack data
    numpass_mask = simdf['numpass_at_precess'].to_numpy() >= nap_thresh
    numpass_low_mask = simdf['numpass_at_precess_low'].to_numpy() >= nap_thresh
    precess_adiff = []
    precess_nspikes = []
    all_adiff = []
    all_slopes = []

    # Density of Fraction, Spikes and Ratio
    df_this = simdf[numpass_mask & (~simdf['rate_angle'].isna())].reset_index(drop=True)
    for i in range(df_this.shape[0]):
        numpass_at_precess = df_this.loc[i, 'numpass_at_precess']
        if numpass_at_precess < nap_thresh:
            continue
        refangle = df_this.loc[i, 'rate_angle']
        allprecess_df = df_this.loc[i, 'precess_df']
        if allprecess_df.shape[0] <1:
            continue
        precess_df = allprecess_df[allprecess_df['precess_exist']]
        precess_counts = precess_df.shape[0]
        if precess_counts > 0:
            precess_adiff.append(cdiff(precess_df['mean_anglesp'].to_numpy(), refangle))
            precess_nspikes.append(precess_df['pass_nspikes'].to_numpy())
        if allprecess_df.shape[0] > 0:
            all_adiff.append(cdiff(allprecess_df['mean_anglesp'].to_numpy(), refangle))
            all_slopes.append(allprecess_df['rcc_m'].to_numpy())
    precess_adiff = np.abs(np.concatenate(precess_adiff))
    precess_nspikes = np.abs(np.concatenate(precess_nspikes))
    adiff_spikes_p = repeat_arr(precess_adiff, precess_nspikes.astype(int))
    adiff_bins = np.linspace(0, np.pi, 45)
    adm = midedges(adiff_bins)
    precess_bins, _ = np.histogram(precess_adiff, bins=adiff_bins)
    spike_bins_p, _ = np.histogram(adiff_spikes_p, bins=adiff_bins)
    spike_bins = spike_bins_p
    norm_bins = (precess_bins / spike_bins)
    norm_bins[np.isnan(norm_bins)] = 0
    norm_bins[np.isinf(norm_bins)] = 0
    precess_allcount = precess_bins.sum()
    rho, pval = spearmanr(adm, norm_bins)
    pm, pc, pr, ppval, _ = linregress(adm, norm_bins/ norm_bins.sum())
    xdum = np.linspace(adm.min(), adm.max(), 10)
    linew_ax0 = 0.6
    ax[0].step(adm, precess_bins / precess_bins.sum(), color='navy', linewidth=linew_ax0,
               label='Precession')
    ax[0].step(adm, spike_bins / spike_bins.sum(), color='orange', linewidth=linew_ax0, label='Spike')
    ax[0].step(adm, norm_bins / norm_bins.sum(), color='green', linewidth=linew_ax0, label='Ratio')
    ax[0].plot(xdum, xdum*pm+pc, color='green', linewidth=linew_ax0)
    ax[0].set_xticks([0, np.pi / 2, np.pi])
    ax[0].set_xticklabels(['0', '$\pi/2$', '$\pi$'])
    # ax[0].set_ylim([0.01, 0.06])
    ax[0].set_yticks([0, 0.1])
    ax[0].tick_params(labelsize=ticksize)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].set_xlabel(r'$|d(\theta_{pass}, \theta_{rate})|$' + ' (rad)', fontsize=fontsize)
    ax[0].set_ylabel('Relative count', fontsize=fontsize, labelpad=5)
    customlegend(ax[0], fontsize=legendsize, bbox_to_anchor=[0, 0.5], loc='lower left')
    ax[0].annotate('p=%s'%(p2str(pval)), xy=(0.4, 0.5), xycoords='axes fraction', fontsize=legendsize, color='green')
    stat_record(stat_fn, False, r"SIM Spearman's correlation: $r_{s(%d)}=%0.2f, p=%s$ " % (precess_allcount, rho, p2str(pval)))



    # Plot Rate angles vs Precess angles

    nprecessmask = simdf['precess_df'].apply(lambda x: x['precess_exist'].sum()) > 1
    numpass_mask = numpass_mask[nprecessmask]
    numpass_low_mask = numpass_low_mask[nprecessmask]
    simdf2 = simdf[nprecessmask].reset_index(drop=True)
    rateangles = simdf2['rate_angle'].to_numpy()
    precessangles = simdf2['precess_angle'].to_numpy()
    precessangles_low = simdf2['precess_angle_low'].to_numpy()

    ax[1].scatter(rateangles[numpass_mask], precessangles[numpass_mask], marker='.', c='gray', s=2)
    ax[1].plot([0, np.pi], [np.pi, 2 * np.pi], c='k')
    ax[1].plot([np.pi, 2 * np.pi], [0, np.pi], c='k')
    ax[1].set_xlabel(r'$\theta_{rate}$', fontsize=fontsize)
    ax[1].set_xticks([0, np.pi, 2 * np.pi])
    ax[1].set_xticklabels(['$0$', '$\pi$', '$2\pi$'], fontsize=fontsize)
    ax[1].set_yticks([0, np.pi, 2 * np.pi])
    ax[1].set_yticklabels(['$0$', '$\pi$', '$2\pi$'], fontsize=fontsize)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].set_ylabel(r'$\theta_{Precess}$', fontsize=fontsize)


    # Plot Histogram: d(precess, rate)
    mask = (~np.isnan(rateangles)) & (~np.isnan(precessangles) & numpass_mask)
    adiff = cdiff(precessangles[mask], rateangles[mask])
    bins, edges = np.histogram(adiff, bins=np.linspace(-np.pi, np.pi, 36))
    bins_norm = bins / np.sum(bins)
    l = bins_norm.max()
    ax[2].bar(midedges(edges), bins_norm, width=edges[1] - edges[0], zorder=0, color='gray')
    linewidth = 1
    mean_angle = shiftcyc_full2half(circmean(adiff))
    ax[2].annotate("", xy=(mean_angle, l), xytext=(0, 0), color='k',  zorder=3,  arrowprops=dict(arrowstyle="->"))
    ax[2].plot([0, 0], [0, l], c='k', linewidth=linewidth, zorder=3)
    ax[2].scatter(0, 0, s=16, c='gray')
    ax[2].spines['polar'].set_visible(False)
    ax[2].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax[2].set_yticks([0, l/2])
    ax[2].set_yticklabels([])
    ax[2].set_xticklabels([])
    v_pval, v_stat = vtest(adiff, mu=np.pi)
    ax[2].annotate('p=%s'%(p2str(v_pval)), xy=(0.25, 0.95), xycoords='axes fraction', fontsize=legendsize)
    ax[2].annotate(r'$\theta_{rate}$', xy=(0.95, 0.525), xycoords='axes fraction', fontsize=fontsize + 1)
    stat_record(stat_fn, False, r'SIM, d(precess, rate), $V_{(%d)}=%0.2f, p=%s$' % (bins.sum(), v_stat, p2str(v_pval)))
    ax[2].set_ylabel('All\npasses', fontsize=fontsize)

    # Plot Histogram: d(precess_low, rate)
    mask_low = (~np.isnan(rateangles)) & (~np.isnan(precessangles_low) & numpass_low_mask)
    adiff = cdiff(precessangles_low[mask_low], rateangles[mask_low])
    bins, edges = np.histogram(adiff, bins=np.linspace(-np.pi, np.pi, 36))
    bins_norm = bins / np.sum(bins)
    l = bins_norm.max()
    ax[3].bar(midedges(edges), bins_norm, width=edges[1] - edges[0], color='gray', zorder=0)
    mean_angle = shiftcyc_full2half(circmean(adiff))
    ax[3].annotate("", xy=(mean_angle, l), xytext=(0, 0), color='k',  zorder=3,  arrowprops=dict(arrowstyle="->"))
    ax[3].plot([0, 0], [0, l], c='k', linewidth=linewidth, zorder=3)
    ax[3].scatter(0, 0, s=16, c='gray')
    ax[3].spines['polar'].set_visible(False)
    ax[3].set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax[3].set_yticks([0, l/2])
    ax[3].set_yticklabels([])
    ax[3].set_xticklabels([])
    v_pval, v_stat = vtest(adiff, mu=np.pi)
    ax[3].annotate('p=%s'%(p2str(v_pval)), xy=(0.25, 0.95), xycoords='axes fraction', fontsize=legendsize)
    ax[3].annotate(r'$\theta_{rate}$', xy=(0.95, 0.525), xycoords='axes fraction', fontsize=fontsize + 1)
    stat_record(stat_fn, False, r'SIM, d(precess_low, rate), $V_{(%d)}=%0.2f, p=%s$' % (bins.sum(), v_stat, p2str(v_pval)))
    ax[3].set_ylabel('Low-spike\npasses', fontsize=fontsize)



def plot_both_slope_offset_Romani(simdf, ax):

    def norm_div(target_hist, divider_hist):
        target_hist_norm = target_hist / divider_hist.reshape(-1, 1)
        target_hist_norm[np.isnan(target_hist_norm)] = 0
        target_hist_norm[np.isinf(target_hist_norm)] = 0
        target_hist_norm = target_hist_norm / np.sum(target_hist_norm) * np.sum(target_hist)
        return target_hist_norm


    nap_thresh = 1
    selected_adiff = np.linspace(0, np.pi, 6)  # 20
    offset_bound = (0, 2 * np.pi)
    slope_bound = (-2 * np.pi, 0)
    adiff_edges = np.linspace(0, np.pi, 100)
    offset_edges = np.linspace(offset_bound[0], offset_bound[1], 100)
    slope_edges = np.linspace(slope_bound[0], slope_bound[1], 100)


    stat_fn = 'fig8_SIM_slopeoffset.txt'
    stat_record(stat_fn, True, 'Average Phase precession')

    # Construct pass df
    refangle_key = 'rate_angle'
    passdf_dict = {'anglediff':[], 'slope':[], 'onset':[], 'pass_nspikes':[]}
    spikedf_dict = {'anglediff':[], 'phasesp':[]}
    dftmp = simdf[(~simdf[refangle_key].isna()) & (simdf['numpass_at_precess'] >= nap_thresh)].reset_index()

    for i in range(dftmp.shape[0]):
        allprecess_df = dftmp.loc[i, 'precess_df']
        precessdf = allprecess_df[allprecess_df['precess_exist']].reset_index(drop=True)
        numprecess = precessdf.shape[0]
        if numprecess < 1:
            continue
        ref_angle = dftmp.loc[i, refangle_key]
        anglediff_tmp = cdiff(precessdf['mean_anglesp'].to_numpy(), ref_angle)
        phasesp_tmp = np.concatenate(precessdf['phasesp'].to_list())

        passdf_dict['anglediff'].extend(anglediff_tmp)
        passdf_dict['slope'].extend(precessdf['rcc_m'])
        passdf_dict['onset'].extend(precessdf['rcc_c'])
        passdf_dict['pass_nspikes'].extend(precessdf['pass_nspikes'])

        spikedf_dict['anglediff'].extend(repeat_arr(anglediff_tmp, precessdf['pass_nspikes'].to_numpy().astype(int)))
        spikedf_dict['phasesp'].extend(phasesp_tmp)
    passdf = pd.DataFrame(passdf_dict)
    passdf['slope_piunit'] = passdf['slope'] * 2 * np.pi
    spikedf = pd.DataFrame(spikedf_dict)

    absadiff_pass = np.abs(passdf['anglediff'].to_numpy())
    offset = passdf['onset'].to_numpy()
    slope = passdf['slope_piunit'].to_numpy()

    absadiff_spike = np.abs(spikedf['anglediff'].to_numpy())
    phase_spike = spikedf['phasesp'].to_numpy()

    # 1D spike hisotgram
    spikes_bins, spikes_edges = np.histogram(absadiff_spike, bins=adiff_edges)

    # 2D slope/offset histogram
    offset_bins, offset_xedges, offset_yedges = np.histogram2d(absadiff_pass, offset,
                                                               bins=(adiff_edges, offset_edges))
    slope_bins, slope_xedges, slope_yedges = np.histogram2d(absadiff_pass, slope,
                                                            bins=(adiff_edges, slope_edges))
    offset_xedm, offset_yedm = midedges(offset_xedges), midedges(offset_yedges)
    slope_xedm, slope_yedm = midedges(slope_xedges), midedges(slope_yedges)
    offset_normbins = norm_div(offset_bins, spikes_bins)
    slope_normbins = norm_div(slope_bins, spikes_bins)

    # Unbinning
    offset_adiff, offset_norm = unfold_binning_2d(offset_normbins, offset_xedm, offset_yedm)
    slope_adiff, slope_norm = unfold_binning_2d(slope_normbins, slope_xedm, slope_yedm)


    # Linear-circular regression
    regress = rcc(offset_adiff, offset_norm)
    offset_m, offset_c, offset_rho, offset_p = regress['aopt'], regress['phi0'], regress['rho'], regress['p']
    regress = rcc(slope_adiff, slope_norm)
    slope_m, slope_c, slope_rho, slope_p = regress['aopt'], regress['phi0'], regress['rho'], regress['p']
    slope_c = slope_c - 2 * np.pi
    stat_record(stat_fn, False, 'LC_Regression Onset-adiff $r_{(%d)}=%0.3f, p=%s$' % (
        offset_bins.sum(), offset_rho, p2str(offset_p)))
    stat_record(stat_fn, False,
                'LC_Regression Slope-adiff $r_{(%d)}=%0.3f, p=%s$' % (slope_bins.sum(), slope_rho, p2str(slope_p)))



    # # Plot average precession curves
    low_mask = absadiff_pass < (np.pi / 6)
    high_mask = absadiff_pass > (np.pi - np.pi / 6)
    slopes_high_all, offsets_high_all = slope[high_mask], offset[high_mask]
    slopes_low_all, offsets_low_all = slope[low_mask], offset[low_mask]

    pval_slope, _, slope_descrips, slopetxt = my_kruskal_2samp(slopes_low_all, slopes_high_all, 'low-$|d|$', 'high-$|d|$')
    (mdn_slopel, lqr_slopel, hqr_slopel), (mdn_slopeh, lqr_slopeh, hqr_slopeh) = slope_descrips

    pval_offset, _, offset_descrips, offsettxt = my_ww_2samp(offsets_low_all, offsets_high_all, 'low-$|d|$', 'high-$|d|$')
    (cmean_offsetl, sem_offsetl), (cmean_offseth, sem_offseth) = offset_descrips

    xdum = np.linspace(0, 1, 10)
    high_agg_ydum = mdn_slopeh * xdum + cmean_offseth
    low_agg_ydum = mdn_slopel * xdum + cmean_offsetl
    slopel_valstr = r'low-$|d|=%0.2f$'%(mdn_slopel)
    slopeh_valstr = r'high-$|d|=%0.2f$'%(mdn_slopeh)
    offsetl_valstr = r'low-$|d|=%0.2f$'%(cmean_offsetl)
    offseth_valstr = r'high-$|d|=%0.2f$'%(cmean_offseth)
    stat_record(stat_fn, False, '===== Average precession curves ====' )
    stat_record(stat_fn, False, 'Slope, %s, %s, %s'%(slopel_valstr, slopeh_valstr, slopetxt))
    stat_record(stat_fn, False, 'Onset, %s, %s, %s'%(offsetl_valstr, offseth_valstr, offsettxt))

    ax[0].plot(xdum, high_agg_ydum, c='lime', label='$|d|>5\pi/6$')
    ax[0].plot(xdum, low_agg_ydum, c='darkblue', label='$|d|<\pi/6$')

    ax[0].annotate('$p_s$'+'=%s'% (p2str(pval_slope)), xy=(0.015, 0.2 + 0.03), xycoords='axes fraction', fontsize=legendsize)
    ax[0].annotate('$p_o$'+'=%s'% (p2str(pval_offset)), xy=(0.015, 0.035 + 0.03), xycoords='axes fraction', fontsize=legendsize)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].set_xticks([0, 1])
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(-np.pi-1, np.pi + 0.3)
    ax[0].set_yticks([-np.pi, 0, np.pi])
    ax[0].set_yticklabels(['$-\pi$', '0', '$\pi$'])
    ax[0].tick_params(labelsize=ticksize)
    ax[0].set_xlabel('Position')
    customlegend(ax[0], fontsize=legendsize, loc='lower left', handlelength=0.5, bbox_to_anchor=(0.1, 0.7))
    ax[0].set_ylabel('Phase (rad)', fontsize=fontsize)

    # # Spike phases
    low_mask_sp = absadiff_spike < (np.pi / 6)
    high_mask_sp = absadiff_spike > (np.pi - np.pi / 6)
    phase_spike = shiftcyc_full2half(phase_spike)
    phasesph = phase_spike[high_mask_sp]
    phasespl = phase_spike[low_mask_sp]
    fstat, k_pval = circ_ktest(phasesph, phasespl)
    p_ww, _, _, p_wwtxt = my_ww_2samp(phasesph, phasespl, r'$high-|d|$', r'$low-|d|$')
    mean_phasesph = shiftcyc_full2half(circmean(phasesph))
    mean_phasespl = shiftcyc_full2half(circmean(phasespl))
    nh, _, _ = ax[1].hist(phasesph, bins=36, density=True, histtype='step', color='lime')
    nl, _, _ = ax[1].hist(phasespl, bins=36, density=True, histtype='step', color='darkblue')
    ml = max(nh.max(), nl.max())
    ax[1].axvline(mean_phasesph, ymin=0.9, ymax=1, color='lime', linewidth=0.75)
    ax[1].axvline(mean_phasespl, ymin=0.9, ymax=1, color='darkblue', linewidth=0.75)
    ax[1].annotate(r'$p$=%s'%(p2str(p_ww)), xy=(0.3, 0.1), xycoords='axes fraction', fontsize=legendsize)
    ax[1].set_xlim(-np.pi, np.pi)
    ax[1].set_xticks([-np.pi, 0, np.pi])
    ax[1].set_xticklabels(['$-\pi$', '0', '$\pi$'])
    ax[1].set_yticks([])
    ax[1].tick_params(labelsize=ticksize)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    ax[1].set_xlabel('Phase (rad)', fontsize=fontsize)
    ax[1].set_ylabel('Relative\nfrequency', fontsize=fontsize)

    stat_record(stat_fn, False, 'SIM, difference of mean spike phases, %s' % (p_wwtxt))
    stat_record(stat_fn, False,
                r'SIM, difference of concentration Bartlett\'s test $F_{(%d, %d)}=%0.2f, p=%s$' % \
                (phasesph.shape[0], phasespl.shape[0], fstat, p2str(k_pval)))



def singlefield_analysis_Romani(single_simdf, save_dir):
    fig = plt.figure(figsize=(total_figw, total_figw/1.25))

    ax_baseh = 1/3
    ax_basew = 1/3

    figxshift = 0.032

    x_btw_squeeze = 0.025

    # Directionality
    # 0, 1, 2, 3
    xsqueeze, ysqueeze = 0.15, 0.15

    yshift = 0.05
    ax_direct = [fig.add_axes([0+xsqueeze/2+figxshift+x_btw_squeeze, 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                 fig.add_axes([ax_basew+xsqueeze/2+figxshift, 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                 fig.add_axes([ax_basew*2+xsqueeze/2+figxshift-x_btw_squeeze, 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze])
                 ]

    # Field's best precession
    # 0, 2
    # 1, 3
    xsqueeze, ysqueeze = 0.15, 0.15
    yshiftall = 0.05
    yshift23 = -0.05
    xshift12, xshift23 = 0, 0
    y_squeeze23 = 0.025
    ax_bestprecess = [fig.add_axes([0+xsqueeze/2+xshift12+figxshift+x_btw_squeeze, 1-ax_baseh*2+ysqueeze/2+yshiftall, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                      fig.add_axes([0+xsqueeze/2+xshift12+figxshift+x_btw_squeeze, 1-ax_baseh*3+ysqueeze/2+yshiftall, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                      fig.add_axes([ax_basew+xsqueeze/2+xshift23+figxshift, 1-ax_baseh*2+ysqueeze/2+yshiftall-y_squeeze23+yshift23, ax_basew-xsqueeze, ax_baseh-ysqueeze], polar=True),
                      fig.add_axes([ax_basew+xsqueeze/2+xshift23+figxshift, 1-ax_baseh*3+ysqueeze/2+yshiftall+y_squeeze23+yshift23, ax_basew-xsqueeze, ax_baseh-ysqueeze], polar=True)]

    # Marginals, aver curve, spike phases
    # 0,
    # 1,
    xsqueeze, ysqueeze = 0.15, 0.15
    xshift01 = 0
    yshiftall = 0.05
    ax_mar = [fig.add_axes([ax_basew*2+xsqueeze/2 + xshift01+figxshift-x_btw_squeeze, 1-ax_baseh*2+ysqueeze/2+yshiftall, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
              fig.add_axes([ax_basew*2+xsqueeze/2 + xshift01+figxshift-x_btw_squeeze, 1-ax_baseh*3+ysqueeze/2+yshiftall, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
              ]


    omniplot_singlefields_Romani(simdf=single_simdf, ax=ax_direct)
    plot_field_bestprecession_Romani(simdf=single_simdf, ax=ax_bestprecess)
    plot_both_slope_offset_Romani(simdf=single_simdf, ax=ax_mar)
    fig.savefig(join(save_dir, 'SIM_single.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'SIM_single.eps'), dpi=dpi)


def omniplot_pairfields_Romani(simdf, ax):
    stat_fn = 'fig9_SIM_pair_directionality.txt'
    stat_record(stat_fn, True)

    linew = 0.75


    spike_threshs = np.arange(0, 520, 20)
    stats_getter = DirectionalityStatsByThresh('num_spikes_pair', 'rate_R_pvalp', 'rate_Rp')
    linecolor = {'all':'k', 'border':'k', 'nonborder':'k'}
    linestyle = {'all':'solid', 'border':'dotted', 'nonborder':'dashed'}

    # Plot all
    all_dict = stats_getter.gen_directionality_stats_by_thresh(simdf, spike_threshs)
    ax[0].plot(spike_threshs, all_dict['medianR'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)
    ax[1].plot(spike_threshs, all_dict['sigfrac_shift'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)


    # Plot border
    simdf_b = simdf[simdf['border']].reset_index(drop=True)
    border_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_b, spike_threshs)
    ax[0].plot(spike_threshs, border_dict['medianR'], c=linecolor['border'], linestyle=linestyle['border'], label='Border', linewidth=linew)
    ax[1].plot(spike_threshs, border_dict['sigfrac_shift'], c=linecolor['border'], linestyle=linestyle['border'], label='Border', linewidth=linew)

    # Plot non-border
    simdf_nb = simdf[~simdf['border']].reset_index(drop=True)
    nonborder_dict = stats_getter.gen_directionality_stats_by_thresh(simdf_nb, spike_threshs)
    ax[0].plot(spike_threshs, nonborder_dict['medianR'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border', linewidth=linew)
    ax[1].plot(spike_threshs, nonborder_dict['sigfrac_shift'], c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border', linewidth=linew)

    # Plot Fraction
    border_nfrac = border_dict['n']/all_dict['n']
    ax[2].plot(spike_threshs, all_dict['datafrac'], c=linecolor['all'], linestyle=linestyle['all'], label='All', linewidth=linew)
    ax[2].plot(spike_threshs, border_nfrac, c=linecolor['border'], linestyle=linestyle['border'], label='Border', linewidth=linew)
    ax[2].plot(spike_threshs, 1-border_nfrac, c=linecolor['nonborder'], linestyle=linestyle['nonborder'], label='Non-border', linewidth=linew)


    # Binomial test for all fields
    signum_all, n_all = all_dict['shift_signum'][0], all_dict['n'][0]
    p_binom = binom_test(signum_all, n_all, p=0.05, alternative='greater')
    stat_txt = 'Binomial test, greater than p=0.05, %d/%d=%0.4f, p=%s'%(signum_all, n_all, signum_all/n_all, p2str(p_binom))
    stat_record(stat_fn, False, stat_txt)
    # ax[1].annotate('Sig. Frac. (All)\n%d/%d=%0.3f\np%s'%(signum_all, n_all, signum_all/n_all, p2str(p_binom)), xy=(0.1, 0.5), xycoords='axes fraction', fontsize=legendsize)

    # # Statistical test
    for idx, ntresh in enumerate(spike_threshs):
        stat_record(stat_fn, False, '======= Threshold=%d ======'%(ntresh))
        # KW test for border median R
        rs_bord_pR, (border_n, nonborder_n), mdns, rs_txt = my_kruskal_2samp(border_dict['allR'][idx], nonborder_dict['allR'][idx], 'border', 'nonborder')
        stat_record(stat_fn, False, "SIM, Median R, border vs non-border: %s" % (rs_txt))

        # Chisquared test for border fractions
        contin = pd.DataFrame({'border': [border_dict['shift_signum'][idx],
                                          border_dict['shift_nonsignum'][idx]],
                               'nonborder': [nonborder_dict['shift_signum'][idx],
                                             nonborder_dict['shift_nonsignum'][idx]]}).to_numpy()

        chi_pborder, _, txt_chiborder = my_chi_2way(contin)
        _, _, fishtxt = my_fisher_2way(contin)
        stat_record(stat_fn, False, "SIM, Significant fraction, border vs non-border: %s, %s" % (txt_chiborder, fishtxt))





    # Plotting asthestic
    ax_ylabels = ['Median R', "Sig. Frac.", "Data Frac."]
    for axid in range(3):
        ax[axid].set_xticks([0, 200, 400])
        ax[axid].set_xticks(np.arange(0, 501, 100), minor=True)
        ax[axid].set_ylabel(ax_ylabels[axid], fontsize=fontsize)
        ax[axid].tick_params(axis='both', which='major', labelsize=ticksize)
        ax[axid].spines['top'].set_visible(False)
        ax[axid].spines['right'].set_visible(False)

    ax[1].set_xlabel('Spike-pair count threshold', fontsize=fontsize)

    customlegend(ax[0], fontsize=legendsize, loc='lower left', bbox_to_anchor=(0.2, 0.5))

    ax[0].set_yticks([0, 0.2, 0.4, 0.6])
    ax[0].set_yticks(np.arange(0, 0.7, 0.1), minor=True)
    ax[1].set_yticks([0, 0.1, 0.2])
    ax[1].set_yticks(np.arange(0, 0.25, 0.05), minor=True)
    ax[2].set_yticks([0, 0.5, 1])
    ax[2].set_yticklabels(['0', '', '1'])
    ax[2].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
    ax[3].axis('off')



def plot_pair_correlation_Romani(simdf, ax):
    stat_fn = 'fig9_SIM_paircorr.txt'
    stat_record(stat_fn, True)

    linew = 0.75
    markersize = 1

    # A->B
    ax[0], x, y, regress = plot_correlogram(ax=ax[0], df=simdf, tag='', direct='A->B', color='gray', alpha=1,
                                            markersize=markersize, linew=linew)

    nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
    stat_record(stat_fn, False, 'SIM A->B, y = %0.2fx + %0.2f, $r_{(%d)}=%0.2f, p=%s$' % \
                (regress['aopt'] * 2 * np.pi, regress['phi0'], nsamples, regress['rho'], p2str(regress['p'])))

    # B->A
    ax[1], x, y, regress = plot_correlogram(ax=ax[1], df=simdf, tag='', direct='B->A', color='gray', alpha=1,
                                            markersize=markersize, linew=linew)

    nsamples = np.sum((~np.isnan(x)) & (~np.isnan(y)))
    stat_record(stat_fn, False, 'SIM B->A, y = %0.2fx + %0.2f, $r_{(%d)}=%0.2f, p=%s$' % \
                (regress['aopt'] * 2 * np.pi, regress['phi0'], nsamples, regress['rho'], p2str(regress['p'])))


    ax[0].set_ylabel('Phase shift (rad)', fontsize=fontsize)
    ax[0].get_shared_y_axes().join(ax[0], ax[1])
    ax[1].set_yticklabels(['']*4)

    for ax_each in [ax[0], ax[1]]:
        ax_each.spines['top'].set_visible(False)
        ax_each.spines['right'].set_visible(False)

    ax[0].set_title(r'$A\rightarrow B$', fontsize=fontsize)
    ax[1].set_title(r'$B\rightarrow A$', fontsize=fontsize)


def plot_exintrinsic_Romani(simdf, ax):
    stat_fn = 'fig9_SIM_exintrinsic.txt'
    stat_record(stat_fn, True)

    ms = 0.2
    # Filtering
    smallerdf = simdf[(~simdf['overlap_ratio'].isna())]


    corr_overlap = smallerdf['overlap_plus'].to_numpy()
    corr_overlap_flip = smallerdf['overlap_minus'].to_numpy()
    corr_overlap_ratio = smallerdf['overlap_ratio'].to_numpy()

    # 1-sample chisquare test
    n_ex = np.sum(corr_overlap_ratio > 0)
    n_in = np.sum(corr_overlap_ratio <= 0)
    n_total = n_ex + n_in


    pchi, _, pchitxt = my_chi_1way([n_ex, n_in])
    stat_record(stat_fn, False, r'SIM, %d/%d=%0.2f, %s' % \
                (n_ex, n_in, n_ex/n_in, pchitxt))
    # 1-sample t test
    mean_ratio = np.mean(corr_overlap_ratio)
    p_1d1samp, _, p_1d1samptxt = my_ttest_1samp(corr_overlap_ratio, 0)
    stat_record(stat_fn, False, 'SIM, mean=%0.4f, %s' % \
                (mean_ratio, p_1d1samptxt))
    # Plot scatter 2d
    ax[0].scatter(corr_overlap_flip, corr_overlap, marker='.', s=ms, c='gray')
    ax[0].plot([0.3, 1], [0.3, 1], c='k', linewidth=0.75)
    ax[0].annotate('%0.2f'%(n_ex/n_in), xy=(0.05, 0.17), xycoords='axes fraction', size=legendsize, color='r')
    ax[0].annotate('p=%s'%(p2str(pchi)), xy=(0.05, 0.025), xycoords='axes fraction', size=legendsize)
    ax[0].set_xlabel('Extrinsicity', fontsize=fontsize)
    ax[0].set_xticks([0, 1])
    ax[0].set_yticks([0, 1])
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)
    ax[0].tick_params(axis='both', which='major', labelsize=ticksize)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].set_ylabel('Intrinsicity', fontsize=fontsize)

    # Plot 1d histogram
    edges = np.linspace(-1, 1, 75)
    width = edges[1]-edges[0]
    (bins, _, _) = ax[1].hist(corr_overlap_ratio, bins=edges, color='gray',
                              density=True, histtype='stepfilled')
    ax[1].plot([mean_ratio, mean_ratio], [0, bins.max()], c='k')
    ax[1].annotate('$\mu$'+ '=%0.3f\np=%s'%(mean_ratio, p2str(p_1d1samp)), xy=(0.2, 0.8), xycoords='axes fraction', fontsize=legendsize)
    ax[1].set_xticks([-0.5, 0, 0.5])
    ax[1].set_yticks([0, 0.1/width] )
    ax[1].set_yticklabels(['0', '0.1'])
    ax[1].set_ylim(0, 6.5)
    ax[1].set_xlabel('Extrinsicity - Intrinsicity', fontsize=fontsize)
    ax[1].tick_params(axis='both', which='major', labelsize=ticksize)
    ax[1].set_xlim(-0.5, 0.5)
    ax[1].set_ylabel('Normalized counts', fontsize=fontsize)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

def pairfield_analysis_Romani(pair_simdf, save_dir):
    fig = plt.figure(figsize=(total_figw, total_figw/1.75))

    ax_baseh = 1/2
    ax_basew = 1/4

    # Directionality
    # 0, 1, 2, 3
    xsqueeze, ysqueeze = 0.1, 0.23
    base_xshift, yshift = 0.15, 0.05
    xshift = [base_xshift, base_xshift + 0.02, base_xshift + 0.02*2, base_xshift + 0.02*3]
    ax_direct = [fig.add_axes([0+xsqueeze/2+xshift[0], 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                 fig.add_axes([ax_basew+xsqueeze/2+xshift[1], 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                 fig.add_axes([ax_basew*2+xsqueeze/2+xshift[2], 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                 fig.add_axes([ax_basew*3+xsqueeze/2+xshift[3], 1-ax_baseh+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze])]

    # Pair Correlation
    # 0, 1
    xsqueeze, ysqueeze = 0.1, 0.23
    xbtw_squeeze = 0.05
    xshift = 0.032 - 0.05/2
    yshift = 0.025
    ax_paircorr = [fig.add_axes([0+xsqueeze/2+xbtw_squeeze/2+xshift, 1-ax_baseh*2+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
                   fig.add_axes([ax_basew+xsqueeze/2-xbtw_squeeze/2+xshift, 1-ax_baseh*2+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze])]
    fig.text(0.185, 0.025, 'Field Overlap', fontsize=fontsize)

    # Ex-intrinsicity
    # 0, 1
    xsqueeze, ysqueeze = 0.1, 0.23
    xshift = -0.025
    yshift = 0.025
    ax_exin = [fig.add_axes([ax_basew*2+xsqueeze/2+xshift, 1-ax_baseh*2+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze]),
               fig.add_axes([ax_basew*3+xsqueeze/2+xshift, 1-ax_baseh*2+ysqueeze/2+yshift, ax_basew-xsqueeze, ax_baseh-ysqueeze])]


    omniplot_pairfields_Romani(simdf=pair_simdf, ax=ax_direct)
    pair_simdf = pair_simdf[~pair_simdf['border']].reset_index(drop=True)
    plot_pair_correlation_Romani(simdf=pair_simdf, ax=ax_paircorr)
    plot_exintrinsic_Romani(simdf=pair_simdf, ax=ax_exin)
    fig.savefig(join(save_dir, 'SIM_pair_nonborder.png'), dpi=dpi)
    fig.savefig(join(save_dir, 'SIM_pair_nonborder.eps'), dpi=dpi)


def reconstruct_singledf():
    rewrite_pth = 'data/Single_processed2_Idirect-5_refrac5.pickle'
    fielddf_raw = load_pickle(rewrite_pth)
    all_pdf = pd.concat(fielddf_raw['precess_df'].to_list(), axis=0, ignore_index=True)
    pass_nspikes = all_pdf[all_pdf['precess_exist']]['pass_nspikes'].to_numpy()
    lqr = np.quantile(pass_nspikes, 0.25)
    print('LQR of pass spikes = %0.2f'%(lqr))
    fielddf_raw = compute_lowspikeprecession(fielddf_raw, lqr)
    fielddf_raw.to_pickle(rewrite_pth)
    return None


def main():

    # tag_list = ['_expo2']
    # I_list = ['10']
    # for tag in tag_list:
    #     for Itag in I_list:
    #         print('%s%s'%(Itag, tag))
    #
    #         save_dir = 'plots/Idirect-%s%s'%(Itag, tag)
    #         os.makedirs(save_dir, exist_ok=True)
    #         single_simdf = load_pickle('data/Single_processed2_Idirect-%s%s.pickle'%(Itag, tag))
    #
    #         # processed1 = load_pickle('data/processed1_Idirect-%s.pickle'%(Itag))
    #         # NeuronDF = processed1['NeuronDF']
    #         # cellid = single_simdf['cell_id'].to_numpy()
    #         # single_simdf['rate_angle'] = NeuronDF.loc[cellid, 'neurona'].to_numpy()
    #
    #
    #         singlefield_analysis_Romani(single_simdf, save_dir=save_dir)
    save_dir = 'plots/Idirect-10'
    pair_simdf = load_pickle('data/Pair_processed2_Idirect-10.pickle')
    pair_simdf['border'] = (pair_simdf['border1'] & pair_simdf['border2'])
    pairfield_analysis_Romani(pair_simdf, save_dir=save_dir)

if __name__ == '__main__':
    main()
    # reconstruct_singledf()
