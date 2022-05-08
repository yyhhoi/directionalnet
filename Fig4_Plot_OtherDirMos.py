from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pycircstat import cdiff, mean as cmean
from library.comput_utils import pair_diff, circgaufunc, get_tspdiff, calc_exin_samepath
from library.shared_vars import total_figw
from library.utils import save_pickle, load_pickle
from library.visualization import customlegend
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p, directional_tuning_tile, simulate_SNN

legendsize = 7
load_dir = 'sim_results/fig4'
save_dir = 'plots/fig4/'
os.makedirs(save_dir, exist_ok=True)
fig, ax = plt.subplots(6, 5, figsize=(total_figw, total_figw*1.2))
mosdeg_pairs = [(45, 225), (90, 270)]

for mospairid, (mosdeg1, mosdeg2) in enumerate(mosdeg_pairs):
    print('Mos pair %s' % str(mosdeg_pairs[mospairid]))
    simdata1 = load_pickle(join(load_dir, 'fig4_MossyLayer_Mosdeg%d.pkl' % (mosdeg1)))
    simdata2 = load_pickle(join(load_dir, 'fig4_MossyLayer_Mosdeg%d.pkl' % (mosdeg2)))

    all_nidx_dict = dict()
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
        direct_c = ['r', 'b']

        # ======================== Plotting =================

        gs = ax[base_axid + mosi, 0].get_gridspec()  # Population raster
        for axeach in ax[base_axid + mosi, 0:2].ravel():
            axeach.remove()
        ax_popras = fig.add_subplot(gs[base_axid + mosi, 0:2])

        # Population Raster
        tsp_ras_ca3_list = []
        dxtun = xxtun1d[1] - xxtun1d[0]
        tt, traj_xx = np.meshgrid(t, xxtun1d[all_nidx])
        for counti, neuronid in enumerate(all_nidx):
            tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
            tsp_neuron = t[tidxsp_neuron]
            neuronx = xxtun1d[neuronid]
            if neuronid == all_egnidxs[0]:
                ras_c = direct_c[0]
            elif neuronid == all_egnidxs[1]:
                ras_c = direct_c[1]
            else:
                ras_c = 'gray'
            ax_popras.eventplot(tsp_neuron, lineoffsets=counti, linelengths=1, linewidths=0.5, color=ras_c)
            if tidxsp_neuron.shape[0] < 1:
                continue
        ax_popras.set_ylim(10, 30)
        ax_popras.set_xlim(t.max()/2-600, t.max()/2+600)
        ax_popras.tick_params(labelsize=legendsize)
        ax_popras.annotate('%d'%(mosdeg), xy=(0.05, 0.9), xycoords='axes fraction', fontsize=legendsize)

        # # Phase precession
        # for itmp, label in enumerate(['Best', 'Worst']):
        #     egnidx = all_egnidxs[itmp]
        #     tidxsp_eg = SpikeDF.loc[SpikeDF['neuronid'] == egnidx, 'tidxsp'].to_numpy()
        #     tsp_eg, phasesp_eg = t[tidxsp_eg], theta_phase[tidxsp_eg]
        #     dsp_eg = traj_d[tidxsp_eg]
        #     mean_phasesp = cmean(phasesp_eg)
        #     dspmin, dsprange = dsp_eg.min(), dsp_eg.max() - dsp_eg.min()
        #     dsp_norm_eg = (dsp_eg-dspmin)/dsprange
        #     regress = rcc(dsp_norm_eg, phasesp_eg, abound=(-1., 1.))
        #     rcc_c, rcc_slope_rad = regress['phi0'], regress['aopt'] * 2 * np.pi
        #     xdum = np.linspace(dsp_norm_eg.min(), dsp_norm_eg.max(), 100)
        #     ydum = xdum * rcc_slope_rad + rcc_c
        #     rate = tsp_eg.shape[0]/((tsp_eg.max() - tsp_eg.min())/1000)
        #     precess_txt = 'y= %0.2fx + %0.2f'%(rcc_slope_rad, rcc_c)
        #     ax[base_axid + mosi, 1 + itmp].scatter(dsp_norm_eg, phasesp_eg, marker='|', s=4, c=direct_c[itmp], label=label)
        #     ax[base_axid + mosi, 1 + itmp].axhline(mean_phasesp, xmin=0, xmax=0.3, linewidth=1, c='gray')
        #     ax[base_axid + mosi, 1 + itmp].plot(xdum, ydum, linewidth=0.75, c='gray')
        #     ax[base_axid + mosi, 1 + itmp].annotate(precess_txt, xy=(0.02, 0.8), xycoords='axes fraction', fontsize=legendsize)
        #     ax[base_axid + mosi, 1 + itmp].annotate('%0.2fHz' % (rate), xy=(0.02, 0.04), xycoords='axes fraction', fontsize=legendsize)
        #     # ax[base_axid + mosi, 1 + itmp].set_xlabel('Position', fontsize=legendsize)
        #     ax[base_axid + mosi, 1 + itmp].set_ylim(0, 2*np.pi)
        #     ax[base_axid + mosi, 1 + itmp].set_yticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])
        #     ax[base_axid + mosi, 1 + itmp].set_yticklabels(['0', '', '$\pi$', '', '$2\pi$'])
        #     # ax[base_axid + mosi, 1 + itmp].set_ylabel('Spike phase (rad)', fontsize=legendsize)

        # # All onsets & Marginal phases
        precessdf_dict = dict(nidx=[], adiff=[], phasesp=[], onset=[], slope=[])
        for neuronid in range(nn_ca3):
            neuronxtmp = xxtun1d[neuronid]
            if np.abs(neuronxtmp) > 20:
                continue

            tidxsp_tmp = SpikeDF.loc[SpikeDF['neuronid']==neuronid, 'tidxsp'].to_numpy()
            if tidxsp_tmp.shape[0] < 5:
                continue
            atun_this = aatun1d[neuronid]
            adiff = np.abs(cdiff(atun_this, np.deg2rad(0)))
            tsp_eg, phasesp_eg = t[tidxsp_tmp], theta_phase[tidxsp_tmp]
            dsp_eg = traj_d[tidxsp_tmp]
            dspmin, dsprange = dsp_eg.min(), dsp_eg.max() - dsp_eg.min()
            dsp_norm_eg = (dsp_eg-dspmin)/dsprange
            regress = rcc(dsp_norm_eg, phasesp_eg, abound=(-1., 1.))
            rcc_c, rcc_slope_rad = regress['phi0'], regress['aopt'] * 2 * np.pi
            if (rcc_slope_rad > (1.9 * np.pi)) or (rcc_slope_rad < (-1.9 * np.pi)):
                continue
            precessdf_dict['nidx'].append(neuronid)
            precessdf_dict['adiff'].append(adiff)
            precessdf_dict['phasesp'].append(phasesp_eg)
            precessdf_dict['onset'].append(rcc_c)
            precessdf_dict['slope'].append(rcc_slope_rad)
        precessdf = pd.DataFrame(precessdf_dict)

        bestprecessdf = precessdf[precessdf['adiff'] <= (np.pi/6)].reset_index(drop=True)
        worstprecessdf = precessdf[precessdf['adiff'] >= (np.pi - np.pi/6)].reset_index(drop=True)

        phasesp_best, phasesp_worst = np.concatenate(bestprecessdf['phasesp'].to_numpy()), np.concatenate(worstprecessdf['phasesp'].to_numpy())
        phasesp_bestmu, phasesp_worstmu = cmean(phasesp_best), cmean(phasesp_worst)
        onset_best, onset_worst = np.array(bestprecessdf['onset'].to_numpy()), np.array(worstprecessdf['onset'].to_numpy())
        onset_bestmu, onset_worstmu = cmean(onset_best), cmean(onset_worst)
        slope_best, slope_worst = np.array(bestprecessdf['slope'].to_numpy()), np.array(worstprecessdf['slope'].to_numpy())
        slope_bestmu, slope_worstmu = np.median(slope_best), np.median(slope_worst)
        nidx_best, nidx_worst = np.array(bestprecessdf['nidx'].to_numpy()), np.array(worstprecessdf['nidx'].to_numpy())
        all_nidx_dict['best_mos%d' % mosdeg] = nidx_best
        all_nidx_dict['worst_mos%d' % mosdeg] = nidx_worst
        phasebins = np.linspace(0, 2*np.pi, 30)

        # # # x,y- coordinates of best-worst sampled neurons
        # ax[base_axid+1, 0].plot(traj_x, traj_y, linewidth=0.75, c='gray')
        # ax[base_axid+1, 0].scatter(xxtun1d_ca3[nidx_best], yytun1d_ca3[nidx_best], c=direct_c[0], s=1, marker='o', label='Best')
        # ax[base_axid+1, 0].scatter(xxtun1d_ca3[nidx_worst], yytun1d_ca3[nidx_worst], c=direct_c[1], s=1, marker='o', label='Worst')
        # ax[base_axid+1, 0].set_xlim(-20, 20)
        # ax[base_axid+1, 0].set_ylim(-20, 20)
        # ax[base_axid+1, 0].set_xlabel('Neuron x (cm)', fontsize=legendsize)
        # ax[base_axid+1, 0].set_ylabel('Neuron y (cm)', fontsize=legendsize)
        # customlegend(ax[base_axid+1, 0], fontsize=legendsize)

        # # trajectory resolved slopes and onsets
        precessdf2 = precessdf[(precessdf['slope'] > -np.pi) & ( precessdf['slope'] < np.pi)].reset_index(drop=True)
        precess_nidx = precessdf2['nidx'].to_numpy()
        precess_x, precess_y = xxtun1d_ca3[precess_nidx], yytun1d_ca3[precess_nidx]
        precess_onsets, precess_slopes = precessdf2['onset'].to_numpy(), precessdf2['slope'].to_numpy()
        mappable1 = ax[base_axid+mosi, 2].scatter(precess_x, precess_y, c=precess_slopes, cmap='jet', marker='.', s=2, vmin=-2.5, vmax=2.5)
        ax[base_axid+mosi, 2].set_xlim(-21, 21)
        ax[base_axid+mosi, 2].set_ylim(-21, 21)
        cbaxes1 = inset_axes(ax[base_axid+mosi, 2], width="100%", height="100%", loc='lower center', bbox_transform=ax[base_axid+mosi, 2].transAxes, bbox_to_anchor=[0.3, 0.15, 0.5, 0.05])
        cbar1 = plt.colorbar(mappable1, cax=cbaxes1, orientation='horizontal')
        cbar1.ax.tick_params(labelsize=legendsize)
        mappable2 = ax[base_axid+mosi, 3].scatter(precess_x, precess_y, c=precess_onsets, cmap='jet', marker='.', s=2)
        ax[base_axid+mosi, 3].set_xlim(-21, 21)
        ax[base_axid+mosi, 3].set_ylim(-21, 21)
        cbaxes2 = inset_axes(ax[base_axid+mosi, 3], width="100%", height="100%", loc='lower center', bbox_transform=ax[base_axid+mosi, 3].transAxes, bbox_to_anchor=[0.3, 0.15, 0.5, 0.05])
        cbar2 = plt.colorbar(mappable2, cax=cbaxes2, orientation='horizontal')
        cbar2.ax.tick_params(labelsize=legendsize)

        # # Slopes and onsets of best-worst neurons
        ax[base_axid+mosi, 4].scatter(onset_best, slope_best, marker='.', c=direct_c[0], s=8, alpha=0.5)
        ax[base_axid+mosi, 4].scatter(onset_worst, slope_worst, marker='.', c=direct_c[1], s=8, alpha=0.5)
        ax[base_axid+mosi, 4].set_xlabel('Onset (rad)', fontsize=legendsize)
        ax[base_axid+mosi, 4].set_ylabel('Slope (rad)', fontsize=legendsize)
        ax[base_axid+mosi, 4].spines['top'].set_visible(False)
        ax[base_axid+mosi, 4].spines['right'].set_visible(False)
        ax[base_axid+mosi, 4].set_xlim(0, 2*np.pi)
        ax[base_axid+mosi, 4].set_ylim(-np.pi, np.pi)

        # # Marginal onset and slope
        axposori = ax[base_axid+mosi, 4].get_position()
        onsetbins = np.linspace(0, 2*np.pi, 60)
        ax_maronset = fig.add_axes([axposori.x0, axposori.y0+0.1, axposori.width, axposori.height * 0.2])
        binsonsetbest, _, _ = ax_maronset.hist(onset_best, bins=onsetbins, density=True, histtype='step', color=direct_c[0], linewidth=0.75)
        binsonsetworst, _, _ = ax_maronset.hist(onset_worst, bins=onsetbins, density=True, histtype='step', color=direct_c[1], linewidth=0.75)
        ax_maronset.axvline(onset_bestmu, ymin=0.55, ymax=0.9, linewidth=0.75, color=direct_c[0])
        ax_maronset.axvline(onset_worstmu, ymin=0.55, ymax=0.9, linewidth=0.75, color=direct_c[1])
        ax_maronset.set_xlim(0, 2*np.pi)
        ax_maronset.set_ylim(0, np.max([binsonsetbest.max(), binsonsetworst.max()])*2)
        ax_maronset.axis('off')

        slopebins = np.linspace(-np.pi, np.pi, 40)
        ax_marslope = fig.add_axes([axposori.x0+0.12, axposori.y0, axposori.width * 0.2, axposori.height])
        binsslopebest, _, _ = ax_marslope.hist(slope_best, bins=slopebins, density=True, histtype='step', color=direct_c[0], linewidth=0.75, orientation='horizontal')
        binsslopeworst, _, _ = ax_marslope.hist(slope_worst, bins=slopebins, density=True, histtype='step', color=direct_c[1], linewidth=0.75, orientation='horizontal')
        ax_marslope.axhline(slope_bestmu, xmin=0.55, xmax=0.9, linewidth=0.75, color=direct_c[0])
        ax_marslope.axhline(slope_worstmu, xmin=0.55, xmax=0.9, linewidth=0.75, color=direct_c[1])
        ax_marslope.set_xlim(0, np.max([binsslopebest.max(), binsslopeworst.max()])*2)
        ax_marslope.set_ylim(-np.pi, np.pi)
        ax_marslope.axis('off')
        #
        # # # Marginal spike phases
        # binsphasebest, _, _ = ax[base_axid+1, 2].hist(phasesp_best, bins=phasebins, density=True, histtype='step', color=direct_c[0], label='Best=%0.2f (n=%d)'%(phasesp_bestmu, phasesp_best.shape[0]), linewidth=0.75)
        # binsphaseworst, _, _ = ax[base_axid+1, 2].hist(phasesp_worst, bins=phasebins, density=True, histtype='step', color=direct_c[1], label='Worst=%0.2f (n=%d)'%(phasesp_worstmu, phasesp_worst.shape[0]), linewidth=0.75)
        # ax[base_axid+1, 2].axvline(phasesp_bestmu, ymin=0.55, ymax=0.65, linewidth=0.75, color=direct_c[0])
        # ax[base_axid+1, 2].axvline(phasesp_worstmu, ymin=0.55, ymax=0.65, linewidth=0.75, color=direct_c[1])
        # ax[base_axid+1, 2].annotate('diff=\n$%0.2f$'%(phasesp_worstmu-phasesp_bestmu), xy=(0.8, 0.5), xycoords='axes fraction', fontsize=legendsize)
        # ax[base_axid+1, 2].set_ylim(0, np.max([binsphasebest.max(), binsphaseworst.max()])*2)
        # ax[base_axid+1, 2].set_xlabel('Marginal spike phase (rad)', fontsize=legendsize)
        # customlegend(ax[base_axid+1, 2], fontsize=legendsize-2)
        #
        #
        # # # Correlations
        # # Find pairs with same xdiff (1, 2, 3, 4). xdiff is computed within every yline
        # edges = np.arange(-100, 100, 5)
        # target_xdiff, epsilon = 4, 0.1
        # all_tsp_diff_list = []
        # for corri, nidx_corr in enumerate([nidx_best, nidx_worst]):
        #     univals_y_best = np.unique(yytun1d_ca3[nidx_corr])
        #     for unival_y_best in univals_y_best:
        #         nidx_corr_uniy = nidx_corr[np.abs(yytun1d_ca3[nidx_corr] - unival_y_best) < epsilon]
        #         all_sampled_xcoords = xxtun1d_ca3[nidx_corr_uniy]
        #         sorted_idx = all_sampled_xcoords.argsort()
        #         sorted_sampled_nidx = nidx_corr_uniy[sorted_idx]
        #         sorted_sampled_xcoords = all_sampled_xcoords[sorted_idx]
        #         for pi in range(sorted_sampled_nidx.shape[0]):
        #             for pj in range(pi, sorted_sampled_nidx.shape[0]):
        #                 x1, x2 = sorted_sampled_xcoords[pi], sorted_sampled_xcoords[pj]
        #                 xdiff = x1-x2
        #                 if (np.abs(xdiff) > (target_xdiff+epsilon)) or (np.abs(xdiff) < (target_xdiff-epsilon)) :
        #                     continue
        #                 nidx1, nidx2 = sorted_sampled_nidx[pi], sorted_sampled_nidx[pj]
        #                 tsp_diff = get_tspdiff(SpikeDF, t, nidx1, nidx2)
        #                 all_tsp_diff_list.append(tsp_diff)
        #     all_tsp_diff = np.concatenate(all_tsp_diff_list)
        #     ax[base_axid+1, 3].hist(all_tsp_diff, bins=edges, histtype='step', linewidth=0.75, color=direct_c[corri])
        #
        #     ax[base_axid+1, 3].annotate('Aver correlation (xdiff=4cm)', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=legendsize-2)

        for i in range(2, 5):
            ax[base_axid + mosi, i].tick_params(labelsize=legendsize)
        theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
        for i in theta_cutidx:
            ax_popras.axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)


    # # Ex-intrinsicity - All neurons for Sim, Dissim, Best, Worst

    SpikeDF1 = simdata1['SpikeDF']
    SpikeDF2 = simdata2['SpikeDF']

    edges = np.linspace(-100, 100, 41)
    tspdiff_dict = dict(Best=[], Worst=[])
    pair_idx_dict = {'Bestidx':[], 'Worstidx':[], 'Simidx':[], 'Dissimidx':[],
                     'Bestbins1':[], 'Worstbins1':[], 'Simbins1':[], 'Dissimbins1':[],
                     'Bestbins2':[], 'Worstbins2':[], 'Simbins2':[], 'Dissimbins2':[]}
    nidx_best = all_nidx_dict['best_mos%d'%(mosdeg1)]
    nidx_worst = all_nidx_dict['worst_mos%d'%(mosdeg1)]

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
            tsp_diff1 = get_tspdiff(SpikeDF1.sample(frac=0.5, random_state=i*j, ignore_index=True), t, nidx1, nidx2)
            tsp_diff2 = get_tspdiff(SpikeDF2.sample(frac=0.5, random_state=i*j, ignore_index=True), t, nidx1, nidx2)
            if (tsp_diff1.shape[0] < 5) or (tsp_diff2.shape[0]< 5):
                continue
            tspdiff_bins1, _ = np.histogram(tsp_diff1, bins=edges)
            tspdiff_bins2, _ = np.histogram(tsp_diff2, bins=edges)
            a1, a2 = aatun1d[nidx1], aatun1d[nidx2]
            absadiff = np.abs(cdiff(a1, a2))
            absadiff_a1pass = np.abs(cdiff(a1, 0))
            absadiff_a2pass = np.abs(cdiff(a2, 0))
            if absadiff < (np.pi/2):  # Similar
                pair_idx_dict['Simidx'].append((nidx1, nidx2))
                pair_idx_dict['Simbins1'].append(tspdiff_bins1)
                pair_idx_dict['Simbins2'].append(tspdiff_bins2)
            if absadiff > (np.pi - np.pi/2):  # dismilar
                pair_idx_dict['Dissimidx'].append((nidx1, nidx2))
                pair_idx_dict['Dissimbins1'].append(tspdiff_bins1)
                pair_idx_dict['Dissimbins2'].append(tspdiff_bins2)
            if (absadiff_a1pass < (np.pi/6)) and (absadiff_a2pass < (np.pi/6)):  # Both best
                pair_idx_dict['Bestidx'].append((nidx1, nidx2))
                pair_idx_dict['Bestbins1'].append(tspdiff_bins1)
                pair_idx_dict['Bestbins2'].append(tspdiff_bins2)
            if (absadiff_a1pass > (np.pi - np.pi/6)) and (absadiff_a2pass > (np.pi - np.pi/6)):  # Both worst
                pair_idx_dict['Worstidx'].append((nidx1, nidx2))
                pair_idx_dict['Worstbins1'].append(tspdiff_bins1)
                pair_idx_dict['Worstbins2'].append(tspdiff_bins2)


    def plot_zero_bias(t, nidx1, nidx2, SpikeDF1, SpikeDF2, edges, bins1, bins2, save_dir, exin, tag):
        tidxsp0_n1 = SpikeDF1.loc[SpikeDF1['neuronid']==nidx1, 'tidxsp']
        tidxsp0_n2 = SpikeDF1.loc[SpikeDF1['neuronid']==nidx2, 'tidxsp']
        tidxsp180_n1 = SpikeDF2.loc[SpikeDF2['neuronid']==nidx1, 'tidxsp']
        tidxsp180_n2 = SpikeDF2.loc[SpikeDF2['neuronid']==nidx2, 'tidxsp']
        tsp0_n1, tsp0_n2, tsp180_n1, tsp180_n2 = t[tidxsp0_n1], t[tidxsp0_n2], t[tidxsp180_n1], t[tidxsp180_n2]
        width = edges[1] - edges[0]
        fig_tmp, ax_tmp = plt.subplots(2, 1, figsize=(8, 10))
        ax_tmp[0].bar(edges, bins1, width=width, color='b', alpha=0.5)
        ax_tmp[0].bar(edges, bins2, width=width, color='r', alpha=0.5)
        ax_tmp[0].set_title('Ex=%0.2f, In=%0.2f'%(exin[0], exin[1]))
        ax_tmp[1].eventplot(tsp0_n1, lineoffsets=0, linelengths=0.4, linewidths=0.75)
        ax_tmp[1].eventplot(tsp0_n2, lineoffsets=1, linelengths=0.4, linewidths=0.75)
        ax_tmp[1].eventplot(tsp180_n1, lineoffsets=4, linelengths=0.4, linewidths=0.75)
        ax_tmp[1].eventplot(tsp180_n2, lineoffsets=5, linelengths=0.4, linewidths=0.75)

        fig_tmp.savefig(join(save_dir, '%s.png'%(tag)), dpi=150)
        plt.close(fig_tmp)

    save_dirtmp = join(save_dir, 'exam2')
    os.makedirs(save_dirtmp, exist_ok=True)
    midedges = (edges[:-1] + edges[1:])/2
    exindict = dict()
    nspdict = {'pairidx': [], 'Nsp': []}
    N_list = []
    Nsp_idx0, Nsp_idx180 = [], []
    for pairtype in ['Sim', 'Dissim', 'Best', 'Worst']:
        exindict[pairtype] = {'ex': [], 'in': [], 'ex_bias': [], 'bins1': [], 'bins2': []}
        for k in range(len(pair_idx_dict[pairtype + 'bins1'])):
            pairidx = pair_idx_dict[pairtype + 'idx'][k]
            tspdiff_bins1 = pair_idx_dict[pairtype + 'bins1'][k]
            tspdiff_bins2 = pair_idx_dict[pairtype + 'bins2'][k]
            ex_val, in_val, ex_bias = calc_exin_samepath(tspdiff_bins1, tspdiff_bins2)
            exindict[pairtype]['ex'].append(ex_val)
            exindict[pairtype]['in'].append(in_val)
            exindict[pairtype]['ex_bias'].append(ex_bias)
            nspdict['pairidx'].extend([pairidx]*2)
            nspdict['Nsp'].extend([tspdiff_bins1.sum(), tspdiff_bins2.sum()])
            # if ex_bias == 0:
            #     plot_zero_bias(t, pairidx[0], pairidx[1], SpikeDF1, SpikeDF2, midedges, tspdiff_bins1, tspdiff_bins2, save_dirtmp,
            #                    exin=(ex_val, in_val), tag='%d_%d'%(pairidx[0], pairidx[1]))
        exindict[pairtype]['ex_n'] = (np.array(exindict[pairtype]['ex_bias']) > 0).sum()
        exindict[pairtype]['in_n'] = (np.array(exindict[pairtype]['ex_bias']) < 0).sum()
        exindict[pairtype]['exin_ratio'] = exindict[pairtype]['ex_n']/exindict[pairtype]['in_n']
        exindict[pairtype]['ex_bias_mu'] = np.mean(exindict[pairtype]['ex_bias'])

    bestlabel = 'Best %d / %d = %0.2f' %(exindict['Best']['ex_n'], exindict['Best']['in_n'], exindict['Best']['ex_n']/exindict['Best']['in_n'])
    worstlabel = 'Worst %d / %d = %0.2f' %(exindict['Worst']['ex_n'], exindict['Worst']['in_n'], exindict['Worst']['ex_n']/exindict['Worst']['in_n'])
    ax[base_axid + 2, 0].scatter(exindict['Best']['ex'], exindict['Best']['in'], c=direct_c[0], marker='.', s=8, alpha=0.7, label=bestlabel)
    ax[base_axid + 2, 0].scatter(exindict['Worst']['ex'], exindict['Worst']['in'], c=direct_c[1], marker='.', s=8, alpha=0.7, label=worstlabel)
    ax[base_axid + 2, 0].set_xlabel('Ex', fontsize=legendsize)
    ax[base_axid + 2, 0].set_ylabel('In', fontsize=legendsize)
    ax[base_axid + 2, 0].plot([0, 1], [0, 1], linewidth=0.75, c='k')
    ax[base_axid + 2, 0].set_xlim(0, 1)
    ax[base_axid + 2, 0].set_ylim(0, 1)
    customlegend(ax[base_axid + 2, 0], fontsize=legendsize-2)
    sim_c, dissim_c = 'm', 'gold'
    simlabel = 'Sim %d / %d = %0.2f' %(exindict['Sim']['ex_n'], exindict['Sim']['in_n'], exindict['Sim']['ex_n']/exindict['Sim']['in_n'])
    dissimlabel = 'Dissim %d / %d = %0.2f' %(exindict['Dissim']['ex_n'], exindict['Dissim']['in_n'], exindict['Dissim']['ex_n']/exindict['Dissim']['in_n'])
    ax[base_axid + 2, 1].scatter(exindict['Sim']['ex'], exindict['Sim']['in'], c=sim_c, marker='.', s=8, alpha=0.7, label=simlabel)
    ax[base_axid + 2, 1].scatter(exindict['Dissim']['ex'], exindict['Dissim']['in'], c=dissim_c, marker='.', s=8, alpha=0.7, label=dissimlabel)
    ax[base_axid + 2, 1].set_xlabel('Ex', fontsize=legendsize)
    ax[base_axid + 2, 1].set_ylabel('In', fontsize=legendsize)
    ax[base_axid + 2, 1].plot([0, 1], [0, 1], linewidth=0.75, c='k')
    ax[base_axid + 2, 1].set_xlim(0, 1)
    ax[base_axid + 2, 1].set_ylim(0, 1)
    customlegend(ax[base_axid + 2, 1], fontsize=legendsize-2)

    bias_edges = np.linspace(-1, 1, 50)
    stattext_BW = 'Best-Worst = %0.3f' %( exindict['Best']['ex_bias_mu'] - exindict['Worst']['ex_bias_mu'])
    stattext_DS = 'Dissim-Sim = %0.3f\n' %( exindict['Dissim']['ex_bias_mu'] - exindict['Sim']['ex_bias_mu'])
    nbins1, _, _ = ax[base_axid + 2, 2].hist(exindict['Best']['ex_bias'], bins=bias_edges, color=direct_c[0], linewidth=0.75, histtype='step', density=True)
    ax[base_axid + 2, 2].axvline(exindict['Best']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=direct_c[0])
    nbins2, _, _ = ax[base_axid + 2, 2].hist(exindict['Worst']['ex_bias'], bins=bias_edges, color=direct_c[1], linewidth=0.75, histtype='step', density=True)
    ax[base_axid + 2, 2].axvline(exindict['Worst']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=direct_c[1])
    nbins3, _, _ = ax[base_axid + 2, 3].hist(exindict['Sim']['ex_bias'], bins=bias_edges, color=sim_c, linewidth=0.75, histtype='step', density=True)
    ax[base_axid + 2, 3].axvline(exindict['Sim']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=sim_c)
    nbins4, _, _ = ax[base_axid + 2, 3].hist(exindict['Dissim']['ex_bias'], bins=bias_edges, color=dissim_c, linewidth=0.75, histtype='step', density=True)
    ax[base_axid + 2, 3].axvline(exindict['Dissim']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=dissim_c)

    ax[base_axid + 2, 2].set_xlabel('Ex minus In', fontsize=legendsize)
    ax[base_axid + 2, 2].set_ylim(0, np.max(np.concatenate([nbins1, nbins2, nbins3, nbins4]))*2)
    ax[base_axid + 2, 2].annotate(stattext_BW, xy=(0.2, 0.7), fontsize=legendsize-2, xycoords='axes fraction')
    ax[base_axid + 2, 3].set_xlabel('Ex minus In', fontsize=legendsize)
    ax[base_axid + 2, 3].set_ylim(0, np.max(np.concatenate([nbins1, nbins2, nbins3, nbins4]))*2)
    ax[base_axid + 2, 3].annotate(stattext_DS, xy=(0.2, 0.7), fontsize=legendsize-2, xycoords='axes fraction')

    for i in range(4):
        ax[base_axid + 2, i].tick_params(labelsize=legendsize)

    ax[base_axid + 2, 4].axis('off')
fig.savefig(join(save_dir, 'fig4_downsamp_WCA3CA3-2000.png'), dpi=300)

