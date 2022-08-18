from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pycircstat import cdiff, mean as cmean
from library.comput_utils import pair_diff, circgaufunc, get_tspdiff, calc_exin_samepath, circular_density_1d
from library.correlogram import ThetaEstimator
from library.script_wrappers import best_worst_analysis, exin_analysis
from library.shared_vars import total_figw
from library.utils import save_pickle, load_pickle
from library.visualization import customlegend, plot_sca_onsetslope, plot_exin_bestworst_simdissim
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p, directional_tuning_tile, simulate_SNN

# ====================================== Global params and paths ==================================
legendsize = 8
load_dir = 'sim_results/fig5'
save_dir = 'plots/fig5/'
os.makedirs(save_dir, exist_ok=True)
plt.rcParams.update({'font.size': legendsize,
                     "axes.titlesize": legendsize,
                     'axes.labelpad': 0,
                     'axes.titlepad': 0,
                     'xtick.major.pad': 0,
                     'ytick.major.pad': 0,
                     'lines.linewidth': 1,
                     'figure.figsize': (5.2, 5.5),
                     'figure.dpi': 300,
                     'axes.spines.top': False,
                     'axes.spines.right': False,
                     })

# ====================================== Figure initialization ==================================

ax_h = 1/6
ax_w = 1/4
hgap = 0.070
hgap_exin = 0.05
wgap = 0.12

xshift_r0134c1 = -0.02
xshift_r0134c2 = -0.035
xshift_r0134c3 = -0.12
xshift_r0134c4 = -0.205
xshift_cdf = -0.06
yshift_r14 = 0.03
yshift_r25 = 0.025
yshift_r7 = 0.02
xgap_exin = 0.03


fig = plt.figure(figsize=(5.2, 5.5))

ax_45 = [
    fig.add_axes([ax_w * 0 + wgap/2, 1 - ax_h + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 1 + wgap/2 + xshift_r0134c1, 1 - ax_h + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshift_r0134c2, 1 - ax_h + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2 + xshift_r0134c3, 1 - ax_h + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 4 + wgap/2 + xshift_r0134c4, 1 - ax_h + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
]
ax_45[0].axis('off')  # Schematics

ax_135 = [
    fig.add_axes([ax_w * 0 + wgap/2, 1 - ax_h * 2 + hgap/2 + yshift_r14, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 1 + wgap/2 + xshift_r0134c1, 1 - ax_h * 2 + hgap/2 + yshift_r14, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshift_r0134c2, 1 - ax_h * 2 + hgap/2 + yshift_r14, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2 + xshift_r0134c3, 1 - ax_h * 2 + hgap/2 + yshift_r14, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 4 + wgap/2 + xshift_r0134c4, 1 - ax_h * 2 + hgap/2 + yshift_r14, ax_w * 1 - wgap, ax_h - hgap]),

]
ax_135[0].axis('off')  # Schematics

ax_exin1 = [
    fig.add_axes([ax_w * 0 + wgap/2+xgap_exin, 1 - ax_h * 3 + hgap_exin/2 + yshift_r25, ax_w * 1 - wgap, ax_h - hgap_exin]),
    fig.add_axes([ax_w * 1 + wgap/2-xgap_exin, 1 - ax_h * 3 + hgap_exin/2 + yshift_r25, ax_w * 1 - wgap, ax_h - hgap_exin]),
    fig.add_axes([ax_w * 2 + wgap/2+xshift_cdf, 1 - ax_h * 3 + hgap_exin/2 + yshift_r25, ax_w * 1 - wgap, ax_h - hgap_exin]),
    fig.add_axes([ax_w * 3 + wgap/2, 1 - ax_h * 3 + hgap_exin/2 + yshift_r25, ax_w * 1 - wgap, ax_h - hgap_exin]),
    fig.add_axes([ax_w * 4 + wgap/2, 1 - ax_h * 3 + hgap_exin/2 + yshift_r25, ax_w * 1 - wgap, ax_h - hgap_exin]),
]
ax_exin1[4].axis('off')  # Not used

ax_90 = [
    fig.add_axes([ax_w * 0 + wgap/2, 1 - ax_h * 4 + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 1 + wgap/2 + xshift_r0134c1, 1 - ax_h * 4 + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshift_r0134c2, 1 - ax_h * 4 + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2 + xshift_r0134c3, 1 - ax_h * 4 + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 4 + wgap/2 + xshift_r0134c4, 1 - ax_h * 4 + hgap/2, ax_w * 1 - wgap, ax_h - hgap])
]
ax_90[0].axis('off')  # Schematics

ax_270 = [
    fig.add_axes([ax_w * 0 + wgap/2, 1 - ax_h * 5 + hgap/2 + yshift_r14, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 1 + wgap/2 + xshift_r0134c1, 1 - ax_h * 5 + hgap/2 + yshift_r14, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshift_r0134c2, 1 - ax_h * 5 + hgap/2 + yshift_r14, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2 + xshift_r0134c3, 1 - ax_h * 5 + hgap/2 + yshift_r14, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 4 + wgap/2 + xshift_r0134c4, 1 - ax_h * 5 + hgap/2 + yshift_r14, ax_w * 1 - wgap, ax_h - hgap]),
]
ax_270[0].axis('off')  # Schematics

ax_exin2 = [
    fig.add_axes([ax_w * 0 + wgap/2+xgap_exin, 1 - ax_h * 6 + hgap_exin/2 + yshift_r25, ax_w * 1 - wgap, ax_h - hgap_exin]),
    fig.add_axes([ax_w * 1 + wgap/2-xgap_exin, 1 - ax_h * 6 + hgap_exin/2 + yshift_r25, ax_w * 1 - wgap, ax_h - hgap_exin]),
    fig.add_axes([ax_w * 2 + wgap/2+xshift_cdf, 1 - ax_h * 6 + hgap_exin/2 + yshift_r25, ax_w * 1 - wgap, ax_h - hgap_exin]),
    fig.add_axes([ax_w * 3 + wgap/2, 1 - ax_h * 6 + hgap_exin/2 + yshift_r25, ax_w * 1 - wgap, ax_h - hgap_exin]),
    fig.add_axes([ax_w * 4 + wgap/2, 1 - ax_h * 6 + hgap_exin/2 + yshift_r25, ax_w * 1 - wgap, ax_h - hgap_exin]),
]
ax_exin2[4].axis('off') # Not used
ax = np.array([ax_45, ax_135, ax_exin1, ax_90, ax_270, ax_exin2])

# ======================================Analysis and plotting ==================================
mosdeg_pairs = [(45, 225), (90, 270)]
direct_c = ['tomato', 'royalblue']
for mospairid, (mosdeg1, mosdeg2) in enumerate(mosdeg_pairs):
    print('Mos pair %s' % str(mosdeg_pairs[mospairid]))
    simdata1 = load_pickle(join(load_dir, 'fig5_MossyLayer_Mosdeg%d.pkl' % (mosdeg1)))
    simdata2 = load_pickle(join(load_dir, 'fig5_MossyLayer_Mosdeg%d.pkl' % (mosdeg2)))

    base_axid = mospairid*3

    for mosi, mosdeg, simdata in ((0, mosdeg1, simdata1), (1, mosdeg2, simdata2)):

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
        theta_phase = BehDF['theta_phase'].to_numpy()

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
        # # All onsets & Marginal phases
        precessdf, info_best, info_worst = best_worst_analysis(SpikeDF, 0, range(nn_ca3), t, theta_phase, traj_d, xxtun1d, aatun1d)
        phasesp_best, onset_best, slope_best, nidx_best = info_best
        phasesp_worst, onset_worst, slope_worst, nidx_worst = info_worst

        # # Slopes and onsets of best-worst neurons
        plot_sca_onsetslope(fig, ax[base_axid + mosi, 1], onset_best, slope_best, onset_worst, slope_worst,
                            onset_lim=(0.25*np.pi, 1.25*np.pi), slope_lim=(-np.pi, 0.5*np.pi), direct_c=direct_c)
        ax[base_axid + mosi, 1].set_xlabel('Onset (rad)', fontsize=legendsize, labelpad=0)
        ax[base_axid + mosi, 1].set_xticks(np.arange(0.5*np.pi, 1.1*np.pi, 0.5*np.pi))
        ax[base_axid + mosi, 1].set_xticks(np.arange(0.25*np.pi, 1.25*np.pi, 0.25*np.pi), minor=True)
        ax[base_axid + mosi, 1].set_xticklabels([r'$\pi/2$', '$\pi$'])
        ax[base_axid + mosi, 1].set_ylabel('Slope (rad)', fontsize=legendsize, labelpad=0)
        ax[base_axid + mosi, 1].set_yticks(np.arange(-np.pi, 0.51*np.pi, 0.5*np.pi))
        ax[base_axid + mosi, 1].set_yticklabels(['$-\pi$', r'$\frac{-\pi}{2}$', '$0$', r'$\frac{\pi}{2}$'])
        ax[base_axid + mosi, 1].set_yticks(np.arange(-np.pi, 0.5*np.pi, 0.25*np.pi), minor=True)

        # # Space resolved slopes and onsets
        precessdf2 = precessdf[(precessdf['slope'] > -np.pi) & ( precessdf['slope'] < np.pi)].reset_index(drop=True)
        precess_nidx = precessdf2['nidx'].to_numpy()
        precess_x, precess_y = xxtun1d_ca3[precess_nidx], yytun1d_ca3[precess_nidx]
        precess_onsets, precess_slopes = precessdf2['onset'].to_numpy(), precessdf2['slope'].to_numpy()
        precess_phasesp = precessdf2['phasesp'].apply(cmean).to_numpy()
        mappable1 = ax[base_axid+mosi, 2].scatter(precess_x, precess_y, c=precess_slopes, cmap='jet', marker='.', s=2, vmin=-np.pi, vmax=np.pi)
        mappable2 = ax[base_axid+mosi, 3].scatter(precess_x, precess_y, c=precess_onsets, cmap='jet', marker='.', s=2, vmin=0.5*np.pi, vmax=1.25*np.pi)
        mappable3 = ax[base_axid+mosi, 4].scatter(precess_x, precess_y, c=precess_phasesp, cmap='jet', marker='.', s=2, vmin=0.375*np.pi, vmax=np.pi)
        if (mosi == 0):

            cbaxes1 = inset_axes(ax[base_axid+mosi, 2], width="100%", height="100%", loc='lower center', bbox_transform=ax[base_axid+mosi, 2].transAxes, bbox_to_anchor=[0.15, 0.95, 0.7, 0.1])
            cbar1 = plt.colorbar(mappable1, cax=cbaxes1, orientation='horizontal')
            cbar1.set_ticks(np.arange(-np.pi, np.pi+0.01, 0.5*np.pi))
            cbar1.set_ticklabels(['$-\pi$', '', '$0$', '', '$\pi$'])
            cbar1.set_ticks(np.arange(-np.pi, np.pi, 0.25*np.pi), minor=True)
            cbar1.ax.set_title('Slope (rad)')

            cbaxes2 = inset_axes(ax[base_axid+mosi, 3], width="100%", height="100%", loc='lower center', bbox_transform=ax[base_axid+mosi, 3].transAxes, bbox_to_anchor=[0.15, 0.95, 0.7, 0.1])
            cbar2 = plt.colorbar(mappable2, cax=cbaxes2, orientation='horizontal')
            cbar2.set_ticks(np.arange(0.5*np.pi, np.pi + 0.1, 0.5*np.pi))
            cbar2.set_ticks(np.arange(0.5*np.pi, 1.25*np.pi+0.1, 0.25*np.pi), minor=True)
            cbar2.set_ticklabels([r'$\pi/2$', '$\pi$'])
            cbar2.ax.set_title('Onset (rad)')

            cbaxes3 = inset_axes(ax[base_axid+mosi, 4], width="100%", height="100%", loc='lower center', bbox_transform=ax[base_axid+mosi, 4].transAxes, bbox_to_anchor=[0.15, 0.95, 0.7, 0.1])
            cbar3 = plt.colorbar(mappable3, cax=cbaxes3, orientation='horizontal')
            cbar3.set_ticks(np.arange(0.5*np.pi, np.pi+0.1, 0.5*np.pi))
            cbar3.set_ticks(np.arange(0.5*np.pi, np.pi, 0.25*np.pi), minor=True)
            cbar3.set_ticklabels(['$\pi/2$', '$\pi$'])
            cbar3.ax.set_title('Phase (rad)')


        for jtmp in range(2, 5):
            ax[base_axid+mosi, jtmp].set_xlabel('x (cm)', labelpad=0)
            ax[base_axid+mosi, jtmp].set_xticks([-20, -10, 0, 10, 20])
            ax[base_axid+mosi, jtmp].set_xticklabels(['-20', '', '0', '', '20'])
            ax[base_axid+mosi, jtmp].set_xticks(np.arange(-20, 20, 5), minor=True)
            ax[base_axid+mosi, jtmp].set_ylabel('y (cm)', labelpad=-8)
            ax[base_axid+mosi, jtmp].set_yticks([-20, -10, 0, 10, 20])
            ax[base_axid+mosi, jtmp].set_yticklabels(['-20', '', '0', '', '20'])
            ax[base_axid+mosi, jtmp].set_yticks(np.arange(-20, 20, 5), minor=True)
        del simdata


    # # Ex-intrinsicity - All neurons for Sim, Dissim, Best, Worst
    SpikeDF1 = simdata1['SpikeDF']
    SpikeDF2 = simdata2['SpikeDF']
    sim_c, dissim_c = 'm', 'goldenrod'
    selected_nidxs = np.concatenate([nidx_best, nidx_worst])
    exindf, exindict = exin_analysis(SpikeDF1, SpikeDF2, t, selected_nidxs, xxtun1d, yytun1d, aatun1d, sortx=True, sorty=True, sampfrac=0.5)
    plot_exin_bestworst_simdissim(ax[base_axid + 2, 0:3], exindf, exindict, direct_c, sim_c, dissim_c)

    # # Direction-resolved intrinsicty
    xbound = (-20, 20)
    ybound = (-5, 5)
    mask = (xxtun1d_ca3 >= xbound[0]) & (xxtun1d_ca3 <= xbound[1]) & (yytun1d_ca3 >= ybound[0]) & (yytun1d_ca3 <= ybound[1] )
    id_tmp = np.where(mask)[0]
    space_nidx_list = []
    for id_tmp_each in id_tmp:
        numsp = SpikeDF1[SpikeDF1['neuronid'] == id_tmp_each].shape[0]
        if numsp < 5:
            continue
        else:
            space_nidx_list.append(id_tmp_each)
    selected_nidxs2 = np.array(space_nidx_list)
    exindf2, _ = exin_analysis(SpikeDF1, SpikeDF2, t, selected_nidxs2, xxtun1d, yytun1d, aatun1d, sortx=True, sorty=True, sampfrac=0.5)

    allpairidx = np.stack(exindf2['pairidx'].to_list())
    allexbias = exindf2['ex_bias'].to_numpy()
    pairidentity = allpairidx[:, 0] * 10000 + allpairidx[:, 1]
    _, arridtmp = np.unique(pairidentity, return_index=True)
    unique_pairids = allpairidx[arridtmp, :]
    unique_exbias = allexbias[arridtmp]

    ex_pairids = unique_pairids[ unique_exbias > 0 , :]
    in_pairids = unique_pairids[ unique_exbias < 0 , :]

    xy1_ex = np.stack([xxtun1d[ex_pairids[:, 0]], yytun1d[ex_pairids[:, 0]] ]).T
    xy2_ex = np.stack([xxtun1d[ex_pairids[:, 1]], yytun1d[ex_pairids[:, 1]] ]).T
    xy1_in = np.stack([xxtun1d[in_pairids[:, 0]], yytun1d[in_pairids[:, 0]] ]).T
    xy2_in = np.stack([xxtun1d[in_pairids[:, 1]], yytun1d[in_pairids[:, 1]] ]).T

    diff_ex = xy2_ex - xy1_ex
    theta_ex = np.angle(diff_ex[:, 0] + diff_ex[:, 1] * 1j)
    diff_in = xy2_in - xy1_in
    theta_in = np.angle(diff_in[:, 0] + diff_in[:, 1] * 1j)
    kappa = 40
    edge_ex, den_ex = circular_density_1d(theta_ex, kappa, 100, (-np.pi/2-0.5, np.pi/2+0.5))
    edge_in, den_in = circular_density_1d(theta_in, kappa, 100, (-np.pi/2-0.5, np.pi/2+0.5))

    ax[base_axid + 2, 3].plot(edge_ex, den_ex, linewidth=0.5, color='maroon', label='Ex')
    ax[base_axid + 2, 3].plot(edge_in, den_in, linewidth=0.5, color='navy', label='In')
    ax[base_axid + 2, 3].set_xlabel('Pair orientation (rad)', labelpad=0)
    ax[base_axid + 2, 3].set_xticks(np.arange(-np.pi/2, np.pi/2+0.1, np.pi/4))
    ax[base_axid + 2, 3].set_xticklabels(['$-\pi/2$', '', '0', '', '$\pi/2$'])
    ax[base_axid + 2, 3].set_xticks(np.arange(-np.pi/2, np.pi/2+0.1, np.pi/8), minor=True)
    ax[base_axid + 2, 3].set_xlim(-np.pi/2-0.5, np.pi/2+0.5)
    ax[base_axid + 2, 3].set_ylabel('Density of\npairs', labelpad=8, va='center')
    ax[base_axid + 2, 3].set_yticks([0, 0.01])
    ax[base_axid + 2, 3].set_yticks(np.arange(0, 0.016, 0.5), minor=True)
    ax[base_axid + 2, 3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax[base_axid + 2, 3].yaxis.get_offset_text().set_visible(False)
    ax[base_axid + 2, 3].annotate(r'$\times 10^{-1}$', xy=(0.01, 0.9), xycoords='axes fraction')
    customlegend(ax[base_axid + 2, 3], loc='center',
                 bbox_transform=ax[base_axid + 2, 3].transAxes, bbox_to_anchor=[0.5, 0.2])


    ax[base_axid, 1].set_xticklabels([])
    ax[base_axid, 1].set_xlabel('')

    ax[base_axid, 2].set_xlabel('')
    ax[base_axid, 2].set_xticklabels([])
    ax[base_axid, 3].set_xlabel('')
    ax[base_axid, 3].set_xticklabels([])
    ax[base_axid, 3].set_ylabel('')
    ax[base_axid, 3].set_yticklabels([])
    ax[base_axid, 4].set_xlabel('')
    ax[base_axid, 4].set_xticklabels([])
    ax[base_axid, 4].set_ylabel('')
    ax[base_axid, 4].set_yticklabels([])

    ax[base_axid + 1, 3].set_ylabel('')
    ax[base_axid + 1, 3].set_yticklabels([])
    ax[base_axid + 1, 4].set_ylabel('')
    ax[base_axid + 1, 4].set_yticklabels([])

    ax[base_axid + 2, 1].set_yticklabels([])
    ax[base_axid + 2, 1].set_ylabel('')


fig.savefig(join(save_dir, 'fig5.png'), dpi=300)
fig.savefig(join(save_dir, 'fig5.tiff'), dpi=300)
fig.savefig(join(save_dir, 'fig5.eps'))
fig.savefig(join(save_dir, 'fig5.pdf'))
fig.savefig(join(save_dir, 'fig5.svg'))