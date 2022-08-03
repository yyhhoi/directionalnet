import time
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from pycircstat import cdiff, mean as cmean
from library.comput_utils import pair_diff, circgaufunc, get_tspdiff, calc_exin_samepath
from library.correlogram import ThetaEstimator
from library.script_wrappers import best_worst_analysis, exin_analysis
from library.shared_vars import total_figw
from library.utils import save_pickle, load_pickle
from library.visualization import customlegend, plot_popras, plot_phase_precession, plot_sca_onsetslope, \
    plot_marginal_phase, plot_exin_bestworst_simdissim
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p, directional_tuning_tile, simulate_SNN

# ====================================== Global params and paths ==================================

load_dir = 'sim_results/fig6'
save_dir = 'plots/fig6/'
os.makedirs(save_dir, exist_ok=True)
legendsize = 8
plt.rcParams.update({'font.size': legendsize,
                     "axes.titlesize": legendsize,
                     'axes.labelpad': 0,
                     'axes.titlepad': 0,
                     'xtick.major.pad': 0,
                     'ytick.major.pad': 0,

                     })
# ====================================== Figure initialization ==================================

fig, ax = plt.subplots(3, 1, figsize=(9, 9))


for ax_each in ax:
    ax_each.tick_params(labelsize=legendsize)
    ax_each.spines['top'].set_visible(False)
    ax_each.spines['right'].set_visible(False)

# ======================================Analysis and plotting ==================================
direct_c = ['tomato', 'royalblue']
all_nidx_dict = dict()

simdata_ctrl = load_pickle(join(load_dir, 'fig6_Ctrl.pkl'))
simdata_onlyin = load_pickle(join(load_dir, 'fig6_OnlyIn.pkl'))
simdata_full = load_pickle(join(load_dir, 'fig6_Full.pkl'))

ctrl_nsp = simdata_ctrl['SpikeDF'].shape[0]
onlyin_nsp = simdata_onlyin['SpikeDF'].shape[0]
full_nsp = simdata_full['SpikeDF'].shape[0]

nsp_min = min(ctrl_nsp, onlyin_nsp, full_nsp)
samp_frac_all = [nsp_min/ctrl_nsp, nsp_min/onlyin_nsp, nsp_min/full_nsp]
simdata_all = [simdata_ctrl, simdata_onlyin, simdata_full]
labels_all = ['Ctrl', 'OnlyIn', 'Full']


for tagi in range(3):

    simdata = simdata_all[tagi]
    label = labels_all[tagi]
    samp_frac = samp_frac_all[tagi]
    patternlabel = labels_all[tagi]


    # ======================== Get data =================
    BehDF = simdata['BehDF']
    SpikeDF = simdata['SpikeDF'].sample(frac=samp_frac, random_state=1).reset_index(drop=True)
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
    theta_f = config_dict['theta_f']
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

    all_egnidxs = all_nidx[[13, 19]]
    best_nidx, worst_nidx = all_egnidxs[0], all_egnidxs[1]

    # ======================== Plotting =================
    # Population Raster
    plot_popras(ax[tagi], SpikeDF, t, all_nidx, all_egnidxs[0], all_egnidxs[1], direct_c[0], direct_c[1])
    # ax[tagi].set_ylim(10, 30)
    # ax[tagi].set_xlim(t.max()/2-600, t.max()/2+600)
    # ax[tagi].set_xticks([500, 1000, 1500])
    # ax[tagi].set_xticklabels(['500', '', '1500'])
    # ax[tagi].set_xticks(np.arange(400, 1601, 100), minor=True)
    ax[tagi].set_xlabel('Time (ms)', fontsize=legendsize, labelpad=-6)
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for i in theta_cutidx:
        ax[tagi].axvline(t[i], c='gray', linewidth=0.25)
    ax[tagi].set_title(labels_all[tagi])
fig.savefig(join(save_dir, 'fig6.png'), dpi=300)
# fig.savefig(join(save_dir, 'fig3.pdf'), dpi=300)
# fig.savefig(join(save_dir, 'fig3.eps'), dpi=300)