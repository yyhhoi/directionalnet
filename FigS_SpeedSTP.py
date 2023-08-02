import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from library.script_wrappers import find_nidx_along_traj, best_worst_analysis
from library.shared_vars import sim_results_dir, plots_dir
from library.utils import load_pickle
from library.visualization import plot_popras, plot_phase_precession, plot_sca_onsetslope

legendsize = 8
plt.rcParams.update({'font.size': legendsize,
                     "axes.titlesize": legendsize,
                     'axes.labelpad': 0,
                     'axes.titlepad': 0,
                     'xtick.major.pad': 0,
                     'ytick.major.pad': 0,
                     'lines.linewidth': 1,
                     'figure.figsize': (6, 6),
                     'figure.dpi': 300,
                     'axes.spines.top': False,
                     'axes.spines.right': False,
                     })


tags_supp2 = [
    # A - original
    '_20cms_I2_6_Isd5_Wmos3000_MosProj4_ECtau500_STDtau500_theta10',
    # B - Change speed
    '_40cms_I2_6_Isd5_Wmos3000_MosProj4_ECtau500_STDtau500_theta10',
    '_100cms_I2_6_Isd5_Wmos3000_MosProj4_ECtau500_STDtau500_theta10',
    # C - Higher sd
    '_100cms_I2_6_Isd15_Wmos3000_MosProj12_ECtau500_STDtau500_theta10', # or with MosProj4
    # D - faster theta
    '_100cms_I2_6_Isd15_Wmos3000_MosProj12_ECtau500_STDtau500_theta12', # or with MosProj4
]
tags_supp1 = [
    # A - Change in STF time constants
    '_20cms_I2_6_Isd5_Wmos3000_MosProj4_ECtau250_STDtau500_theta10',
    '_20cms_I2_6_Isd5_Wmos3000_MosProj4_ECtau100_STDtau500_theta10',
    '_20cms_I2_6_Isd5_Wmos3000_MosProj4_ECtau2000_STDtau500_theta10',
    '_20cms_I6_6_Isd5_Wmos3000_MosProj4_ECtau100_STDtau500_theta10',
    # B - Lower STD time constants
    '_20cms_I2_6_Isd5_Wmos3000_MosProj4_ECtau500_STDtau100_theta10',
    '_20cms_I2_6_Isd5_Wmos3000_MosProj4_ECtau500_STDtau2000_theta10',
    # '_20cms_I2_6_Isd5_Wmos3000_MosProj4_ECtau150_STDtau150_theta10',
 ]
numtags_supp2 = len(tags_supp2)
numtags_supp1 = len(tags_supp1)
fig_supp2_ras, ax_supp2_ras = plt.subplots(numtags_supp2, 1, figsize=(4, 7))
fig_supp2_pp, ax_supp2_pp = plt.subplots(numtags_supp2, 1, figsize=(2, 7), sharex=True)
fig_supp1_ras, ax_supp1_ras = plt.subplots(numtags_supp1, 1, figsize=(4, 8.5), sharex=True)
fig_supp1_pp, ax_supp1_pp = plt.subplots(numtags_supp1, 1, figsize=(2, 8.5), sharex=True)




save_dir = join(plots_dir, 'fig3')
os.makedirs(save_dir, exist_ok=True)


def plot_figS_ras(simdata, ax, fig2, ax2, tag):
    direct_c = ['tomato', 'royalblue']
    all_nidx_dict = dict()
    # ======================== Get data =================
    BehDF = simdata['BehDF']
    SpikeDF = simdata['SpikeDF']
    NeuronDF = simdata['NeuronDF']
    MetaData = simdata['MetaData']
    config_dict = simdata['Config']

    theta_phase_plot = BehDF['theta_phase_plot']
    traj_x = BehDF['traj_x'].to_numpy()
    traj_y = BehDF['traj_y'].to_numpy()
    t = BehDF['t'].to_numpy()
    theta_phase = BehDF['theta_phase']

    nn_ca3 = MetaData['nn_ca3']

    xxtun1d = NeuronDF['neuronx'].to_numpy()
    yytun1d = NeuronDF['neurony'].to_numpy()
    aatun1d = NeuronDF['neurona'].to_numpy()

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
    all_nidx, all_nidx_all = find_nidx_along_traj(traj_x, traj_y, xxtun1d_ca3, yytun1d_ca3)
    all_nidx_dict[0] = all_nidx

    mid_idn = int(all_nidx.shape[0] / 2)
    all_egnidxs = all_nidx[[mid_idn - 1, mid_idn + 1]]
    best_nidx, worst_nidx = all_egnidxs[0], all_egnidxs[1]

    # ======================== Plotting =================
    # Population Raster
    plot_popras(ax, SpikeDF, t, all_nidx, all_egnidxs[0], all_egnidxs[1], 'gray', 'gray')
    traj_idx = [np.where(all_nidx == int(nidx))[0][0] for nidx in all_nidx_all]
    trajidx_uni, trajidx_uniidx = np.unique(traj_idx, return_index=True)
    ax.plot(t[trajidx_uniidx], trajidx_uni - 0.5, lw=0.5, color='k')
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for i in range(len(theta_cutidx) - 1):
        if i % 2 == 0:
            cutidx1, cutidx2 = theta_cutidx[i], theta_cutidx[i + 1]
            ax.axvspan(t[cutidx1], t[cutidx2], color='gray', alpha=0.1)
    ax.axvspan(t[theta_cutidx[-1]], t[-1], color='gray', alpha=0.1)
    ax.set_yticks([0, 40, 80])
    ax.set_yticks(np.arange(0, 81, 10), minor=True)

    # Tag-specific modification
    if tag == '_20cms_I2_6_Isd5_Wmos3000_MosProj4_ECtau500_STDtau500_theta10':
        ax.set_xlim(1600, 2400)
        ax.set_xticks([1600, 2000, 2400])
        ax.set_xticks(np.arange(1600, 2401, 100), minor=True)
        ax.set_ylim(20, 60)
        ax.set_yticks([20, 40, 60])
        ax.set_yticks(np.arange(20, 61, 10), minor=True)
    elif tag == '_40cms_I2_6_Isd5_Wmos3000_MosProj4_ECtau500_STDtau500_theta10':
        ax.set_xlim(600, 1400)
        ax.set_xticks([600, 1000, 1400])
        ax.set_xticks(np.arange(600, 1401, 100), minor=True)
        ax.set_ylim(20, 60)
        ax.set_yticks([20, 40, 60])
        ax.set_yticks(np.arange(20, 61, 10), minor=True)
    else:
        ax.set_xlim(0, 800)
        ax.set_xticks([0, 400, 800])
        ax.set_xticks(np.arange(0, 801, 100), minor=True)

    try:

        # # All onsets & Marginal phases
        precessdf, info_best, info_worst = best_worst_analysis(SpikeDF, 0, range(nn_ca3), t, theta_phase, traj_d, xxtun1d, aatun1d, abs_xlim=40)
        all_slopes = precessdf['slope'].to_numpy()
        median_slope = np.median(all_slopes)
        IQR_slope = np.quantile(all_slopes, 0.75) - np.quantile(all_slopes, 0.25)
        phasesp_best, onset_best, slope_best, nidx_best = info_best
        phasesp_worst, onset_worst, slope_worst, nidx_worst = info_worst
        # # Slopes and onsets of best-worst neurons
        plot_sca_onsetslope(fig2, ax2, onset_best, slope_best, onset_worst, slope_worst,
                            onset_lim=(0.25*np.pi, 1.25*np.pi), slope_lim=(-np.pi, 0.6*np.pi),
                            direct_c=direct_c, mar_axfrac=0.9) # Check if the bounds are correct
        # ax2.set_xlabel('Onset (rad)', fontsize=legendsize, labelpad=0)
        ax2.set_xticks(np.arange(0.5*np.pi, 1.1*np.pi, 0.5*np.pi))
        ax2.set_xticks(np.arange(0.25*np.pi, 1.25*np.pi, 0.25*np.pi), minor=True)
        ax2.set_xticklabels([r'$\pi/2$', '$\pi$'])
        ax2.set_yticks(np.arange(-np.pi, 0.6*np.pi, 0.5*np.pi))
        ax2.set_yticklabels(['$-\pi$', r'$\frac{-\pi}{2}$', '$0$', r'$\frac{\pi}{2}$'])
        ax2.set_yticks(np.arange(-np.pi, 0.6*np.pi, 0.25*np.pi), minor=True)

    except Exception as e:
        print(e)
        ax2.axis('off')


    return




for tagi, analysis_tag in enumerate(tags_supp2):
    print('Plotting ', tagi, analysis_tag)
    load_dir = join(sim_results_dir, 'fig3%s'%analysis_tag)
    simdata = load_pickle(join(load_dir, 'fig3_MossyLayer_Mosdeg0.pkl'))
    plot_figS_ras(simdata, ax_supp2_ras[tagi], fig_supp2_pp, ax_supp2_pp[tagi], analysis_tag)

for tagi, analysis_tag in enumerate(tags_supp1):
    print('Plotting ', tagi, analysis_tag)
    load_dir = join(sim_results_dir, 'fig3%s'%analysis_tag)
    simdata = load_pickle(join(load_dir, 'fig3_MossyLayer_Mosdeg0.pkl'))
    plot_figS_ras(simdata, ax_supp1_ras[tagi], fig_supp1_pp, ax_supp1_pp[tagi], analysis_tag)

    ax_supp1_ras[tagi].set_xlim(1600, 2400)
    ax_supp1_ras[tagi].set_xticks([1600, 2000, 2400])
    ax_supp1_ras[tagi].set_xticks(np.arange(1600, 2401, 100), minor=True)
    ax_supp1_ras[tagi].set_ylim(20, 60)
    ax_supp1_ras[tagi].set_yticks([20, 40, 60])
    ax_supp1_ras[tagi].set_yticks(np.arange(20, 61, 10), minor=True)


fig_supp2_ras.savefig(join(save_dir, 'fig_supp2_ras.png'), dpi=300)
fig_supp2_ras.savefig(join(save_dir, 'fig_supp2_ras.svg'), dpi=300)
fig_supp1_ras.savefig(join(save_dir, 'fig_supp1_ras.png'), dpi=300)
fig_supp1_ras.savefig(join(save_dir, 'fig_supp1_ras.svg'), dpi=300)
fig_supp2_pp.savefig(join(save_dir, 'fig_supp2_pp.png'), dpi=300)
fig_supp2_pp.savefig(join(save_dir, 'fig_supp2_pp.svg'), dpi=300)
fig_supp1_pp.savefig(join(save_dir, 'fig_supp1_pp.png'), dpi=300)
fig_supp1_pp.savefig(join(save_dir, 'fig_supp1_pp.svg'), dpi=300)