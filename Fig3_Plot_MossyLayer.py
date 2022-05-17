from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from pycircstat import cdiff, mean as cmean
from library.comput_utils import pair_diff, circgaufunc, get_tspdiff, calc_exin_samepath
from library.script_wrappers import best_worst_analysis
from library.shared_vars import total_figw
from library.utils import save_pickle, load_pickle
from library.visualization import customlegend, plot_popras, plot_phase_precession, plot_sca_onsetslope, \
    plot_marginal_phase
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p, directional_tuning_tile, simulate_SNN

# ====================================== Global params and paths ==================================
legendsize = 8
load_dir = 'sim_results/fig3'
save_dir = 'plots/fig3/'
os.makedirs(save_dir, exist_ok=True)
plt.rcParams.update({'font.size': legendsize})
# ====================================== Figure initialization ==================================
figw = 5.2
figh = 5.6
scheme_space_h = (0.8/figw)
ax_h = (1 - scheme_space_h ) / 4
ax_ymax = 1 - scheme_space_h
ax_w = 1/4
hgap = 0.075
wgap = 0.075
xshitf_colL = 0.045
xshitf_colR = 0.02
hgap_sca = 0.03
wgap_sca = 0.03

yshift_precess = 0.02
yshift_popstats = 0.03
yshift_exin = 0.02


fig = plt.figure(figsize=(figw, figh))

ax_ras = [
    fig.add_axes([ax_w * 0 + wgap/2 + xshitf_colL, ax_ymax - ax_h + hgap/2, ax_w * 2 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshitf_colR, ax_ymax - ax_h + hgap/2, ax_w * 2 - wgap, ax_h - hgap])
]

ax_precess = [
    fig.add_axes([ax_w * 0 + wgap/2 + xshitf_colL, ax_ymax - ax_h*2 + hgap/2 + yshift_precess, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 1 + wgap/2 + xshitf_colL, ax_ymax - ax_h*2 + hgap/2 + yshift_precess, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshitf_colR, ax_ymax - ax_h*2 + hgap/2 + yshift_precess, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2 + xshitf_colR, ax_ymax - ax_h*2 + hgap/2 + yshift_precess, ax_w * 1 - wgap, ax_h - hgap])
]

ax_popstats = [
    fig.add_axes([ax_w * 0 + wgap/2 + xshitf_colL, ax_ymax - ax_h*3 + hgap/2 + yshift_popstats, ax_w * 1 - wgap - wgap_sca, ax_h - hgap - hgap_sca]),
    fig.add_axes([ax_w * 1 + wgap/2 + xshitf_colL, ax_ymax - ax_h*3 + hgap/2 + yshift_popstats, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshitf_colR, ax_ymax - ax_h*3 + hgap/2 + yshift_popstats, ax_w * 1 - wgap - wgap_sca, ax_h - hgap - hgap_sca]),
    fig.add_axes([ax_w * 3 + wgap/2 + xshitf_colR, ax_ymax - ax_h*3 + hgap/2 + yshift_popstats, ax_w * 1 - wgap, ax_h - hgap]),
]

ax_exin = [
    fig.add_axes([ax_w * 0 + wgap/2 + xshitf_colL, ax_ymax - ax_h*4 + hgap/2 + yshift_exin, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 1 + wgap/2 + xshitf_colL, ax_ymax - ax_h*4 + hgap/2 + yshift_exin, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshitf_colR, ax_ymax - ax_h*4 + hgap/2 + yshift_exin, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2 + xshitf_colR, ax_ymax - ax_h*4 + hgap/2 + yshift_exin, ax_w * 1 - wgap, ax_h - hgap]),
]

ax_all = np.concatenate([ax_ras, ax_precess, ax_popstats, ax_exin])


for ax_each in ax_all:
    ax_each.tick_params(labelsize=legendsize)
    ax_each.spines['top'].set_visible(False)
    ax_each.spines['right'].set_visible(False)

# ======================================Analysis and plotting ==================================

all_nidx_dict = dict()
simdata0 = load_pickle(join(load_dir, 'fig3_MossyLayer_Mosdeg0.pkl'))
simdata180 = load_pickle(join(load_dir, 'fig3_MossyLayer_Mosdeg180.pkl'))
for mosi, mosdeg, simdata in ((0, 0, simdata0), (1, 180, simdata180)):

    base_axid = mosi*2
    # ======================== Get data =================
    BehDF = simdata['BehDF']
    SpikeDF = simdata['SpikeDF']
    NeuronDF = simdata['NeuronDF']
    ActivityData = simdata['ActivityData']
    MetaData = simdata['MetaData']
    config_dict = simdata['Config']

    theta_phase_plot = BehDF['theta_phase_plot']
    traj_x = BehDF['traj_x'].to_numpy()
    traj_y = BehDF['traj_y'].to_numpy()
    t = BehDF['t'].to_numpy()
    theta_phase = BehDF['theta_phase']

    nn_ca3 = MetaData['nn_ca3']
    w = MetaData['w']

    xxtun1d = NeuronDF['neuronx'].to_numpy()
    yytun1d = NeuronDF['neurony'].to_numpy()
    aatun1d = NeuronDF['neurona'].to_numpy()

    Isen = ActivityData['Isen']
    Isen_fac = ActivityData['Isen_fac']

    w_ca3ca3 = w[:nn_ca3, :nn_ca3]
    xxtun1d_ca3 = xxtun1d[:nn_ca3]
    yytun1d_ca3 = yytun1d[:nn_ca3]
    aatun1d_ca3 = aatun1d[:nn_ca3]
    nx_ca3, ny_ca3 = config_dict['nx_ca3'], config_dict['ny_ca3']
    xxtun2d_ca3 = xxtun1d_ca3.reshape(nx_ca3, nx_ca3)  # Assuming nx = ny
    yytun2d_ca3 = yytun1d_ca3.reshape(nx_ca3, nx_ca3)  # Assuming nx = ny
    aatun2d_ca3 = aatun1d_ca3.reshape(nx_ca3, nx_ca3)  # Assuming nx = ny

    Ipos_max_compen = config_dict['Ipos_max_compen']
    Iangle_diff = config_dict['Iangle_diff']
    Iangle_kappa = config_dict['Iangle_kappa']
    xmin, xmax, ymin, ymax = config_dict['xmin'], config_dict['xmax'], config_dict['ymin'], config_dict['ymax']
    traj_d = np.append(0, np.cumsum(np.sqrt(np.diff(traj_x)**2 + np.diff(traj_y)**2)))

    # # Population raster CA3
    # Indices along the trajectory
    all_nidx = np.zeros(traj_x.shape[0])
    for i in range(traj_x.shape[0]):
        run_x, run_y = traj_x[i], traj_y[i]
        nidx = np.argmin(np.square(run_x - xxtun1d_ca3) + np.square(run_y - yytun1d_ca3))
        all_nidx[i] = nidx
    all_nidx = all_nidx[np.sort(np.unique(all_nidx, return_index=True)[1])]
    all_nidx = all_nidx.astype(int)
    all_nidx_dict[mosdeg] = all_nidx


    all_egnidxs = all_nidx[[13, 19]]
    best_nidx, worst_nidx = all_egnidxs[0], all_egnidxs[1]
    direct_c = ['tomato', 'royalblue']

    # ======================== Plotting =================


    # Population Raster
    plot_popras(ax_ras[mosi], SpikeDF, t, all_nidx, all_egnidxs[0], all_egnidxs[1], direct_c[0], direct_c[1])
    ax_ras[mosi].set_ylim(10, 30)
    ax_ras[mosi].set_xlim(t.max()/2-600, t.max()/2+600)
    ax_ras[mosi].set_xticks([500, 1000, 1500])
    ax_ras[mosi].set_xticklabels(['500', '', '1500'])
    ax_ras[mosi].set_xticks(np.arange(400, 1601, 100), minor=True)
    ax_ras[mosi].set_xlabel('Time (ms)', fontsize=legendsize, labelpad=-6)


    # Phase precession
    for i, label in enumerate(['Best', 'Worst']):
        egnidx = all_egnidxs[i]
        tidxsp_eg = SpikeDF.loc[SpikeDF['neuronid'] == egnidx, 'tidxsp'].to_numpy()
        tsp_eg, phasesp_eg = t[tidxsp_eg], theta_phase[tidxsp_eg]
        dsp_eg = traj_d[tidxsp_eg]
        plot_phase_precession(ax_precess[base_axid+i], dsp_eg, phasesp_eg, s=4, c=direct_c[i], fontsize=legendsize,
                              plotmeanphase=True)
        ax_precess[base_axid+i].set_xlabel('Position', labelpad=-6, fontsize=legendsize)

    # # All onsets & Marginal phases
    precessdf, info_best, info_worst = best_worst_analysis(SpikeDF, 0, range(nn_ca3), t, theta_phase, traj_d, xxtun1d, aatun1d)
    phasesp_best, onset_best, slope_best, nidx_best = info_best
    phasesp_worst, onset_worst, slope_worst, nidx_worst = info_worst
    all_nidx_dict['best_mos%d' % mosdeg] = nidx_best
    all_nidx_dict['worst_mos%d' % mosdeg] = nidx_worst

    # # Slopes and onsets of best-worst neurons
    plot_sca_onsetslope(fig, ax_popstats[base_axid + 0], onset_best, slope_best, onset_worst, slope_worst,
                        onset_lim=(0.25*np.pi, 1.25*np.pi), slope_lim=(-np.pi, 0.5*np.pi), direct_c=direct_c)
    ax_popstats[base_axid + 0].set_xlabel('Onset (rad)', fontsize=legendsize, labelpad=0)
    ax_popstats[base_axid + 0].set_xticks(np.arange(0.5*np.pi, 1.1*np.pi, 0.5*np.pi))
    ax_popstats[base_axid + 0].set_xticks(np.arange(0.25*np.pi, 1.25*np.pi, 0.25*np.pi), minor=True)
    ax_popstats[base_axid + 0].set_xticklabels([r'$\pi/2$', '$\pi$'])
    ax_popstats[base_axid + 0].set_yticks(np.arange(-np.pi, 0.51*np.pi, 0.5*np.pi))
    ax_popstats[base_axid + 0].set_yticklabels(['$-\pi$', r'$\frac{-\pi}{2}$', '$0$', r'$\frac{\pi}{2}$'])
    ax_popstats[base_axid + 0].set_yticks(np.arange(-np.pi, 0.5*np.pi, 0.25*np.pi), minor=True)

    # # Marginal spike phases
    plot_marginal_phase(ax_popstats[base_axid + 1], phasesp_best, phasesp_worst, direct_c, legendsize)


for axeach in ax_ras:
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for i in theta_cutidx:
        axeach.axvline(t[i], c='gray', linewidth=0.25)

# # Aver correlation only for 0 and 180 mos deg
mosdirect_c = ['green', 'brown']  # 0, 180
SpikeDF0 = simdata0['SpikeDF']
SpikeDF180 = simdata180['SpikeDF']
edges = np.arange(-100, 100, 5)
target_xdiff, epsilon = 5, 0.1
all_tsp_diff_list0, all_tsp_diff_list180 = [], []
nidx_corr = np.concatenate([all_nidx_dict['best_mos0'], all_nidx_dict['worst_mos0']])
univals_y_best = np.unique(yytun1d_ca3[nidx_corr])
for unival_y_best in univals_y_best:
    nidx_corr_uniy = nidx_corr[np.abs(yytun1d_ca3[nidx_corr] - unival_y_best) < epsilon]
    all_sampled_xcoords = xxtun1d_ca3[nidx_corr_uniy]
    sorted_idx = all_sampled_xcoords.argsort()
    sorted_sampled_nidx = nidx_corr_uniy[sorted_idx]
    sorted_sampled_xcoords = all_sampled_xcoords[sorted_idx]
    for pi in range(sorted_sampled_nidx.shape[0]):
        for pj in range(pi, sorted_sampled_nidx.shape[0]):
            x1, x2 = sorted_sampled_xcoords[pi], sorted_sampled_xcoords[pj]
            xdiff = x1-x2
            if (np.abs(xdiff) > (target_xdiff+epsilon)) or (np.abs(xdiff) < (target_xdiff-epsilon)) :
                continue
            nidx1, nidx2 = sorted_sampled_nidx[pi], sorted_sampled_nidx[pj]
            tsp_diff0 = get_tspdiff(SpikeDF0, t, nidx1, nidx2)
            tsp_diff180 = get_tspdiff(SpikeDF180, t, nidx1, nidx2)
            all_tsp_diff_list0.append(tsp_diff0)
            all_tsp_diff_list180.append(tsp_diff180)
all_tsp_diff0 = np.concatenate(all_tsp_diff_list0)
all_tsp_diff180 = np.concatenate(all_tsp_diff_list180)
ax_exin[0].hist(all_tsp_diff0, bins=edges, histtype='step', linewidth=0.5, density=True, color=mosdirect_c[0])
ax_exin[0].hist(all_tsp_diff180, bins=edges, histtype='step', linewidth=0.5, density=True, color=mosdirect_c[1])
ax_exin[0].set_xlabel('Time lag (ms)', fontsize=legendsize, labelpad=0)
ax_exin[0].set_xticks((-100, -50, 0, 50, 100))
ax_exin[0].set_xticklabels(('-100', '', '0', '', '100'))
ax_exin[0].set_xticks(np.arange(-100, 101, 10), minor=True)
ax_exin[0].set_ylabel('Spike density', labelpad=0)
ax_exin[0].set_ylim(0, 0.025)
ax_exin[0].set_yticks([0, 0.01, 0.02])
ax_exin[0].set_yticks(np.arange(0, 0.025, 0.005), minor=True)
ax_exin[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ax_exin[0].yaxis.get_offset_text().set_visible(False)
ax_exin[0].annotate(r'$\times 10^{-1}$', xy=(0.01, 0.9), xycoords='axes fraction', fontsize=legendsize)
ax_exin[0].annotate(r'$\theta_{DG}=0^\circ$', xy=(0.4, 0.9), xycoords='axes fraction', fontsize=legendsize, color=mosdirect_c[0])
ax_exin[0].annotate(r'$\theta_{DG}=180^\circ$', xy=(0.4, 0.75), xycoords='axes fraction', fontsize=legendsize, color=mosdirect_c[1])

# # Ex-intrinsicity - All neurons for Sim, Dissim, Best, Worst


edges = np.arange(-100, 100, 5)
tspdiff_dict = dict(Best=[], Worst=[])
pair_idx_dict = {'Bestidx':[], 'Worstidx':[], 'Simidx':[], 'Dissimidx':[],
                 'Bestbins0':[], 'Worstbins0':[], 'Simbins0':[], 'Dissimbins0':[],
                 'Bestbins180':[], 'Worstbins180':[], 'Simbins180':[], 'Dissimbins180':[]}
nidx_best = all_nidx_dict['best_mos0']
nidx_worst = all_nidx_dict['worst_mos0']

all_sampled_nidx = np.concatenate([nidx_best, nidx_worst])
all_sampled_xcoords = xxtun1d[all_sampled_nidx]
sorted_idx = all_sampled_xcoords.argsort()
sorted_sampled_nidx = all_sampled_nidx[sorted_idx]
sorted_sampled_xcoords = all_sampled_xcoords[sorted_idx]

for i in range(sorted_sampled_nidx.shape[0]):
    for j in range(i, sorted_sampled_nidx.shape[0]):
        x1, x2 = sorted_sampled_xcoords[i], sorted_sampled_xcoords[j]
        if x1 == x2:
            continue
        nidx1, nidx2 = sorted_sampled_nidx[i], sorted_sampled_nidx[j]
        tsp_diff0 = get_tspdiff(SpikeDF0.sample(frac=0.6, random_state=i*j, ignore_index=True), t, nidx1, nidx2)
        tsp_diff180 = get_tspdiff(SpikeDF180.sample(frac=0.6, random_state=i*j, ignore_index=True), t, nidx1, nidx2)

        if (tsp_diff0.shape[0] < 2) or (tsp_diff180.shape[0]< 2):
            continue
        tspdiff_bins0, _ = np.histogram(tsp_diff0, bins=edges)
        tspdiff_bins180, _ = np.histogram(tsp_diff180, bins=edges)
        a1, a2 = aatun1d[nidx1], aatun1d[nidx2]
        absadiff = np.abs(cdiff(a1, a2))
        absadiff_a1pass = np.abs(cdiff(a1, 0))
        absadiff_a2pass = np.abs(cdiff(a2, 0))
        if absadiff < (np.pi/2):  # Similar
            pair_idx_dict['Simidx'].append((nidx1, nidx2))
            pair_idx_dict['Simbins0'].append(tspdiff_bins0)
            pair_idx_dict['Simbins180'].append(tspdiff_bins180)
        if absadiff > (np.pi - np.pi/2):  # dismilar
            pair_idx_dict['Dissimidx'].append((nidx1, nidx2))
            pair_idx_dict['Dissimbins0'].append(tspdiff_bins0)
            pair_idx_dict['Dissimbins180'].append(tspdiff_bins180)
        if (absadiff_a1pass < (np.pi/6)) and (absadiff_a2pass < (np.pi/6)):  # Both best
            pair_idx_dict['Bestidx'].append((nidx1, nidx2))
            pair_idx_dict['Bestbins0'].append(tspdiff_bins0)
            pair_idx_dict['Bestbins180'].append(tspdiff_bins180)
            # tspdiff_dict['Best'].append(tsp_diff)
        if (absadiff_a1pass > (np.pi - np.pi/6)) and (absadiff_a2pass > (np.pi - np.pi/6)):  # Both worst
            pair_idx_dict['Worstidx'].append((nidx1, nidx2))
            pair_idx_dict['Worstbins0'].append(tspdiff_bins0)
            pair_idx_dict['Worstbins180'].append(tspdiff_bins180)
            # tspdiff_dict['Worst'].append(tsp_diff)


exindict = dict()
nspdict = {'pairidx': [], 'Nsp': []}
N_list = []
Nsp_idx0, Nsp_idx180 = [], []
for pairtype in ['Sim', 'Dissim', 'Best', 'Worst']:
    exindict[pairtype] = {'ex': [], 'in': [], 'ex_bias': [], 'bins0': [], 'bins180': []}
    for k in range(len(pair_idx_dict[pairtype + 'bins0'])):
        pairidx = pair_idx_dict[pairtype + 'idx'][k]
        tspdiff_bins0 = pair_idx_dict[pairtype + 'bins0'][k]
        tspdiff_bins180 = pair_idx_dict[pairtype + 'bins180'][k]
        ex_val, in_val, ex_bias = calc_exin_samepath(tspdiff_bins0, tspdiff_bins180)
        if ex_bias == 0:
            continue
        exindict[pairtype]['ex'].append(ex_val)
        exindict[pairtype]['in'].append(in_val)
        exindict[pairtype]['ex_bias'].append(ex_bias)
        nspdict['pairidx'].extend([pairidx]*2)
        nspdict['Nsp'].extend([tspdiff_bins0.sum(), tspdiff_bins180.sum()])
    exindict[pairtype]['ex_n'] = (np.array(exindict[pairtype]['ex_bias']) > 0).sum()
    exindict[pairtype]['in_n'] = (np.array(exindict[pairtype]['ex_bias']) < 0).sum()
    exindict[pairtype]['exin_ratio'] = exindict[pairtype]['ex_n']/exindict[pairtype]['in_n']
    exindict[pairtype]['ex_bias_mu'] = np.mean(exindict[pairtype]['ex_bias'])




bestlabel = 'Best: %0.2f' %(exindict['Best']['ex_n']/exindict['Best']['in_n'])
worstlabel = 'Worst: %0.2f' %(exindict['Worst']['ex_n']/exindict['Worst']['in_n'])
ax_exin[1].scatter(exindict['Best']['ex'], exindict['Best']['in'], c=direct_c[0], marker='.', s=0.5, label=bestlabel)
ax_exin[1].scatter(exindict['Worst']['ex'], exindict['Worst']['in'], c=direct_c[1], marker='.', s=0.5, label=worstlabel)
ax_exin[1].annotate(bestlabel, xy=(0.02, 0.16), xycoords='axes fraction', fontsize=legendsize, color=direct_c[0])
ax_exin[1].annotate(worstlabel, xy=(0.02, 0.01), xycoords='axes fraction', fontsize=legendsize, color=direct_c[1])

ax_exin[1].set_xlabel('Ex', fontsize=legendsize, labelpad=0)
ax_exin[1].set_ylabel('In', fontsize=legendsize, labelpad=0)
ax_exin[1].plot([0.3, 1], [0.3, 1], linewidth=0.75, c='k')
ax_exin[1].set_xticks(np.arange(0, 1.1, 0.5))
ax_exin[1].set_xticklabels(['0', '', 1])
ax_exin[1].set_xticks(np.arange(0, 1, 0.1), minor=True)
ax_exin[1].set_xlim(0, 1)
ax_exin[1].set_yticks(np.arange(0, 1.1, 0.5))
ax_exin[1].set_yticklabels(['0', '', 1])
ax_exin[1].set_yticks(np.arange(0, 1, 0.1), minor=True)
ax_exin[1].set_ylim(0, 1)
sim_c, dissim_c = 'm', 'goldenrod'
simlabel = 'Sim: %0.2f' %(exindict['Sim']['ex_n']/exindict['Sim']['in_n'])
dissimlabel = 'Dissim: %0.2f' %(exindict['Dissim']['ex_n']/exindict['Dissim']['in_n'])
ax_exin[2].scatter(exindict['Sim']['ex'], exindict['Sim']['in'], c=sim_c, marker='.', s=0.5, label=simlabel)
ax_exin[2].scatter(exindict['Dissim']['ex'], exindict['Dissim']['in'], c=dissim_c, marker='.', s=0.5, label=dissimlabel)
ax_exin[2].annotate(simlabel, xy=(0.02, 0.16), xycoords='axes fraction', fontsize=legendsize, color=sim_c)
ax_exin[2].annotate(dissimlabel, xy=(0.02, 0.01), xycoords='axes fraction', fontsize=legendsize, color=dissim_c)
ax_exin[2].set_xlabel('Ex', fontsize=legendsize, labelpad=0)
ax_exin[2].set_ylabel('In', fontsize=legendsize, labelpad=0)
ax_exin[2].plot([0.3, 1], [0.3, 1], linewidth=0.75, c='k')
ax_exin[2].set_xticks(np.arange(0, 1.1, 0.5))
ax_exin[2].set_xticklabels(['0', '', 1])
ax_exin[2].set_xticks(np.arange(0, 1, 0.1), minor=True)
ax_exin[2].set_xlim(0, 1)
ax_exin[2].set_yticks(np.arange(0, 1.1, 0.5))
ax_exin[2].set_yticklabels(['0', '', 1])
ax_exin[2].set_yticks(np.arange(0, 1, 0.1), minor=True)
ax_exin[2].set_ylim(0, 1)

bias_edges = np.linspace(-1, 1, 50)
stattext_BW = 'Best-Worst = %0.3f' %( exindict['Best']['ex_bias_mu'] - exindict['Worst']['ex_bias_mu'])
stattext_DS = 'Dissim-Sim = %0.3f\n' %( exindict['Dissim']['ex_bias_mu'] - exindict['Sim']['ex_bias_mu'])
nbins1, _ = np.histogram(exindict['Best']['ex_bias'], bins=bias_edges)
nbins2, _ = np.histogram(exindict['Worst']['ex_bias'], bins=bias_edges)
nbins3, _ = np.histogram(exindict['Sim']['ex_bias'], bins=bias_edges)
nbins4, _ = np.histogram(exindict['Dissim']['ex_bias'], bins=bias_edges)
nbins1, nbins2, nbins3, nbins4 = np.cumsum(nbins1)/nbins1.sum(), np.cumsum(nbins2)/nbins2.sum(), np.cumsum(nbins3)/nbins3.sum(), np.cumsum(nbins4)/nbins4.sum()
ax_exin[3].plot(bias_edges[:-1], nbins1, linewidth=0.75, color=direct_c[0])
ax_exin[3].plot(bias_edges[:-1], nbins2, linewidth=0.75, color=direct_c[1])
ax_exin[3].plot(bias_edges[:-1], nbins3, linewidth=0.75, color=sim_c)
ax_exin[3].plot(bias_edges[:-1], nbins4, linewidth=0.75, color=dissim_c)
ax_exin[3].axvline(exindict['Best']['ex_bias_mu'], ymin=0.75, ymax=1, linewidth=0.5, color=direct_c[0])
ax_exin[3].axvline(exindict['Worst']['ex_bias_mu'], ymin=0.75, ymax=1, linewidth=0.5, color=direct_c[1])
ax_exin[3].axvline(exindict['Sim']['ex_bias_mu'], ymin=0.75, ymax=1, linewidth=0.5, color=sim_c)
ax_exin[3].axvline(exindict['Dissim']['ex_bias_mu'], ymin=0.75, ymax=1, linewidth=0.5, color=dissim_c)
ax_exin[3].set_xlabel(r'Ex$\minus$In', fontsize=legendsize, labelpad=0)
ax_exin[3].set_xticks(np.arange(-0.5, 0.6, 0.5))
ax_exin[3].set_xticklabels(['-0.5', '0', '0.5'])
ax_exin[3].set_xticks(np.arange(-0.5, 0.6, 0.1), minor=True)
ax_exin[3].set_xlim(-0.5, 0.5)
ax_exin[3].set_ylabel('CDF', fontsize=legendsize, labelpad=0)
ax_exin[3].set_yticks([0, 0.5, 1])
ax_exin[3].set_yticks(np.arange(0, 1, 0.1), minor=True)
ax_exin[3].set_yticklabels(['0', '', '1'])
ax_exin[3].set_ylim(0, np.max(np.concatenate([nbins1, nbins2, nbins3, nbins4]))*1.3)


# Remaining aesthetics
ax_ras[0].set_ylabel('Place cell index', fontsize=legendsize, labelpad=0)
ax_precess[0].set_ylabel('Slope (rad)', fontsize=legendsize, labelpad=0)
ax_precess[0].set_ylabel('Onset (rad)', fontsize=legendsize, labelpad=0)


for j in range(1, 4):
    ax_precess[j].set_yticklabels([])
for j in range(2, 4):
    ax_popstats[j].set_yticklabels([])
    ax_popstats[j].set_ylabel('')
ax_popstats[0].set_ylabel('Slope (rad)', fontsize=legendsize, labelpad=0)
ax_ras[1].set_yticklabels([])
ax_exin[2].set_yticklabels([])
ax_exin[2].set_ylabel('')

fig.savefig(join(save_dir, 'fig3.png'), dpi=300)
fig.savefig(join(save_dir, 'fig3.pdf'), dpi=300)

fig.savefig(join(save_dir, 'fig3.eps'), dpi=300)