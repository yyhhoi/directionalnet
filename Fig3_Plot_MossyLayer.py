from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from pycircstat import cdiff, mean as cmean
from library.comput_utils import pair_diff, circgaufunc, get_tspdiff, calc_exin_samepath
from library.shared_vars import total_figw
from library.utils import save_pickle, load_pickle
from library.visualization import customlegend
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p, directional_tuning_tile, simulate_SNN

legendsize = 7
load_dir = 'sim_results/fig3'
save_dir = 'plots/fig3/'
os.makedirs(save_dir, exist_ok=True)
all_nidx_dict = dict()

simdata0 = load_pickle(join(load_dir, 'fig3_MossyLayer_Mosdeg0.pkl'))
simdata180 = load_pickle(join(load_dir, 'fig3_MossyLayer_Mosdeg180.pkl'))


fig, ax = plt.subplots(5, 4, figsize=(total_figw, total_figw), constrained_layout=True)
for mosi, mosdeg, simdata in ((0, 0, simdata0), (1, 180, simdata180)):

    base_axid = mosi*2
    gs = ax[base_axid, 0].get_gridspec()  # Population raster
    for axeach in ax[base_axid, 0:2].ravel():
        axeach.remove()
    ax_popras = fig.add_subplot(gs[base_axid, 0:2])

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

    # Phase precession
    for i, label in enumerate(['Best', 'Worst']):
        egnidx = all_egnidxs[i]
        tidxsp_eg = SpikeDF.loc[SpikeDF['neuronid'] == egnidx, 'tidxsp'].to_numpy()
        tsp_eg, phasesp_eg = t[tidxsp_eg], theta_phase[tidxsp_eg]
        dsp_eg = traj_d[tidxsp_eg]
        mean_phasesp = cmean(phasesp_eg)
        dspmin, dsprange = dsp_eg.min(), dsp_eg.max() - dsp_eg.min()
        dsp_norm_eg = (dsp_eg-dspmin)/dsprange
        regress = rcc(dsp_norm_eg, phasesp_eg, abound=(-1., 1.))
        rcc_c, rcc_slope_rad = regress['phi0'], regress['aopt'] * 2 * np.pi
        xdum = np.linspace(dsp_norm_eg.min(), dsp_norm_eg.max(), 100)
        ydum = xdum * rcc_slope_rad + rcc_c
        rate = tsp_eg.shape[0]/((tsp_eg.max() - tsp_eg.min())/1000)
        precess_txt = 'y= %0.2fx + %0.2f'%(rcc_slope_rad, rcc_c)
        ax[base_axid, 2+i].scatter(dsp_norm_eg, phasesp_eg, marker='|', s=4, c=direct_c[i], label=label)
        ax[base_axid, 2+i].axhline(mean_phasesp, xmin=0, xmax=0.3, linewidth=1, c='gray')
        ax[base_axid, 2+i].plot(xdum, ydum, linewidth=0.75, c='gray')
        ax[base_axid, 2+i].annotate(precess_txt, xy=(0.02, 0.8), xycoords='axes fraction', fontsize=legendsize)
        ax[base_axid, 2+i].annotate('%0.2fHz' % (rate), xy=(0.02, 0.04), xycoords='axes fraction', fontsize=legendsize)
        # ax[base_axid, 2+i].set_xlabel('Position', fontsize=legendsize)
        ax[base_axid, 2+i].set_ylim(0, 2*np.pi)
        ax[base_axid, 2+i].set_yticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])
        ax[base_axid, 2+i].set_yticklabels(['0', '', '$\pi$', '', '$2\pi$'])
        # ax[base_axid, 2+i].set_ylabel('Spike phase (rad)', fontsize=legendsize)

    # # All onsets & Marginal phases
    phasesp_best_list, phasesp_worst_list = [], []
    onset_best_list, onset_worst_list = [], []
    slope_best_list, slope_worst_list = [], []
    nidx_best_list, nidx_worst_list = [], []
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
        if adiff < (np.pi/6):  # best
            phasesp_best_list.append(phasesp_eg)
            onset_best_list.append(rcc_c)
            nidx_best_list.append(neuronid)
            slope_best_list.append(rcc_slope_rad)
        elif adiff > (np.pi - np.pi/6):  # worst
            phasesp_worst_list.append(phasesp_eg)
            onset_worst_list.append(rcc_c)
            nidx_worst_list.append(neuronid)
            slope_worst_list.append(rcc_slope_rad)

        else:
            continue

    phasesp_best, phasesp_worst = np.concatenate(phasesp_best_list), np.concatenate(phasesp_worst_list)
    phasesp_bestmu, phasesp_worstmu = cmean(phasesp_best), cmean(phasesp_worst)
    onset_best, onset_worst = np.array(onset_best_list), np.array(onset_worst_list)
    onset_bestmu, onset_worstmu = cmean(onset_best), cmean(onset_worst)
    slope_best, slope_worst = np.array(slope_best_list), np.array(slope_worst_list)
    slope_bestmu, slope_worstmu = np.median(slope_best), np.median(slope_worst)
    nidx_best, nidx_worst = np.array(nidx_best_list), np.array(nidx_worst_list)
    all_nidx_dict['best_mos%d' % mosdeg] = nidx_best
    all_nidx_dict['worst_mos%d' % mosdeg] = nidx_worst
    phasebins = np.linspace(0, 2*np.pi, 30)

    # # x,y- coordinates of best-worst sampled neurons
    ax[base_axid+1, 0].plot(traj_x, traj_y, linewidth=0.75, c='gray')
    ax[base_axid+1, 0].scatter(xxtun1d_ca3[nidx_best], yytun1d_ca3[nidx_best], c=direct_c[0], s=1, marker='o', label='Best')
    ax[base_axid+1, 0].scatter(xxtun1d_ca3[nidx_worst], yytun1d_ca3[nidx_worst], c=direct_c[1], s=1, marker='o', label='Worst')
    ax[base_axid+1, 0].set_xlim(-20, 20)
    ax[base_axid+1, 0].set_ylim(-20, 20)
    ax[base_axid+1, 0].set_xlabel('Neuron x (cm)', fontsize=legendsize)
    ax[base_axid+1, 0].set_ylabel('Neuron y (cm)', fontsize=legendsize)
    customlegend(ax[base_axid+1, 0], fontsize=legendsize)


    # # Slopes and onsets of best-worst neurons
    ax[base_axid+1, 1].scatter(onset_best, slope_best, marker='.', c=direct_c[0], s=8, alpha=0.5)
    ax[base_axid+1, 1].scatter(onset_worst, slope_worst, marker='.', c=direct_c[1], s=8, alpha=0.5)
    ax[base_axid+1, 1].set_xlabel('Onset (rad)', fontsize=legendsize)
    ax[base_axid+1, 1].set_ylabel('Slope (rad)', fontsize=legendsize)
    ax[base_axid+1, 1].spines['top'].set_visible(False)
    ax[base_axid+1, 1].spines['right'].set_visible(False)
    ax[base_axid+1, 1].set_xlim(0, 2*np.pi)
    ax[base_axid+1, 1].set_ylim(-np.pi, np.pi)
    

    # # Marginal onset and slope
    axposori = ax[base_axid+1, 1].get_position()
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

    # # Marginal spike phases
    binsphasebest, _, _ = ax[base_axid+1, 2].hist(phasesp_best, bins=phasebins, density=True, histtype='step', color=direct_c[0], label='Best=%0.2f (n=%d)'%(phasesp_bestmu, phasesp_best.shape[0]), linewidth=0.75)
    binsphaseworst, _, _ = ax[base_axid+1, 2].hist(phasesp_worst, bins=phasebins, density=True, histtype='step', color=direct_c[1], label='Worst=%0.2f (n=%d)'%(phasesp_worstmu, phasesp_worst.shape[0]), linewidth=0.75)
    ax[base_axid+1, 2].axvline(phasesp_bestmu, ymin=0.55, ymax=0.65, linewidth=0.75, color=direct_c[0])
    ax[base_axid+1, 2].axvline(phasesp_worstmu, ymin=0.55, ymax=0.65, linewidth=0.75, color=direct_c[1])
    ax[base_axid+1, 2].annotate('diff=\n$%0.2f$'%(phasesp_worstmu-phasesp_bestmu), xy=(0.8, 0.5), xycoords='axes fraction', fontsize=legendsize)
    ax[base_axid+1, 2].set_ylim(0, np.max([binsphasebest.max(), binsphaseworst.max()])*2)
    ax[base_axid+1, 2].set_xlabel('Marginal spike phase (rad)', fontsize=legendsize)
    customlegend(ax[base_axid+1, 2], fontsize=legendsize-2)


    # # Correlations
    # Find pairs with same xdiff (1, 2, 3, 4). xdiff is computed within every yline
    edges = np.arange(-100, 100, 5)
    target_xdiff, epsilon = 4, 0.1
    all_tsp_diff_list = []
    for corri, nidx_corr in enumerate([nidx_best, nidx_worst]):
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
                    tsp_diff = get_tspdiff(SpikeDF, t, nidx1, nidx2)
                    all_tsp_diff_list.append(tsp_diff)
        all_tsp_diff = np.concatenate(all_tsp_diff_list)
        ax[base_axid+1, 3].hist(all_tsp_diff, bins=edges, histtype='step', linewidth=0.75, color=direct_c[corri])

        ax[base_axid+1, 3].annotate('Aver correlation (xdiff=4cm)', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=legendsize-2)

    for i in range(4):
        ax[base_axid+1, i].tick_params(labelsize=legendsize)
    ax[base_axid+0, 2].tick_params(labelsize=legendsize)
    ax[base_axid+0, 3].tick_params(labelsize=legendsize)
    ax_popras.tick_params(labelsize=legendsize)
    for axeach in [ax_popras]:
        theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
        for i in theta_cutidx:
            axeach.axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)


# # Ex-intrinsicity - All neurons for Sim, Dissim, Best, Worst

SpikeDF0 = simdata0['SpikeDF']
SpikeDF180 = simdata180['SpikeDF']

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
        tsp_diff0 = get_tspdiff(SpikeDF0.sample(frac=0.5, random_state=i*j, ignore_index=True), t, nidx1, nidx2)
        tsp_diff180 = get_tspdiff(SpikeDF180.sample(frac=0.5, random_state=i*j, ignore_index=True), t, nidx1, nidx2)

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
        exindict[pairtype]['ex'].append(ex_val)
        exindict[pairtype]['in'].append(in_val)
        exindict[pairtype]['ex_bias'].append(ex_bias)
        nspdict['pairidx'].extend([pairidx]*2)
        nspdict['Nsp'].extend([tspdiff_bins0.sum(), tspdiff_bins180.sum()])
    exindict[pairtype]['ex_n'] = (np.array(exindict[pairtype]['ex_bias']) > 0).sum()
    exindict[pairtype]['in_n'] = (np.array(exindict[pairtype]['ex_bias']) <= 0).sum()
    exindict[pairtype]['exin_ratio'] = exindict[pairtype]['ex_n']/exindict[pairtype]['in_n']
    exindict[pairtype]['ex_bias_mu'] = np.mean(exindict[pairtype]['ex_bias'])

nspdf = pd.DataFrame(nspdict)

fig_nsp, ax_nsp = plt.subplots(2, 1, figsize=(5, 10))
ax_nsp[0].hist(nspdf['Nsp'], bins=50, color='k')
ax_nsp[0].axvline(nspdf['Nsp'].median())

pairidx = nspdf[nspdf['Nsp']==nspdf['Nsp'].max()].iloc[0]['pairidx']
tidxsp0_n1 = SpikeDF0.loc[SpikeDF0['neuronid']==pairidx[0], 'tidxsp']
tidxsp0_n2 = SpikeDF0.loc[SpikeDF0['neuronid']==pairidx[1], 'tidxsp']
tidxsp180_n1 = SpikeDF180.loc[SpikeDF180['neuronid']==pairidx[0], 'tidxsp']
tidxsp180_n2 = SpikeDF180.loc[SpikeDF180['neuronid']==pairidx[1], 'tidxsp']
tsp0_n1, tsp0_n2, tsp180_n1, tsp180_n2 = t[tidxsp0_n1], t[tidxsp0_n2], t[tidxsp180_n1], t[tidxsp180_n2]
ax_nsp[1].eventplot(tsp0_n1, lineoffsets=0, linelengths=0.4, linewidths=0.75)
ax_nsp[1].eventplot(tsp0_n2, lineoffsets=1, linelengths=0.4, linewidths=0.75)
ax_nsp[1].eventplot(tsp180_n1, lineoffsets=4, linelengths=0.4, linewidths=0.75)
ax_nsp[1].eventplot(tsp180_n2, lineoffsets=5, linelengths=0.4, linewidths=0.75)

fig_nsp.savefig(join(save_dir, 'pairsp_count.png'), dpi=200)



bestlabel = 'Best %d / %d = %0.2f' %(exindict['Best']['ex_n'], exindict['Best']['in_n'], exindict['Best']['ex_n']/exindict['Best']['in_n'])
worstlabel = 'Worst %d / %d = %0.2f' %(exindict['Worst']['ex_n'], exindict['Worst']['in_n'], exindict['Worst']['ex_n']/exindict['Worst']['in_n'])
ax[4, 0].scatter(exindict['Best']['ex'], exindict['Best']['in'], c=direct_c[0], marker='.', s=8, alpha=0.7, label=bestlabel)
ax[4, 0].scatter(exindict['Worst']['ex'], exindict['Worst']['in'], c=direct_c[1], marker='.', s=8, alpha=0.7, label=worstlabel)
ax[4, 0].set_xlabel('Ex', fontsize=legendsize)
ax[4, 0].set_ylabel('In', fontsize=legendsize)
ax[4, 0].plot([0, 1], [0, 1], linewidth=0.75, c='k')
ax[4, 0].set_xlim(0, 1)
ax[4, 0].set_ylim(0, 1)
customlegend(ax[4, 0], fontsize=legendsize-2)
sim_c, dissim_c = 'm', 'gold'
simlabel = 'Sim %d / %d = %0.2f' %(exindict['Sim']['ex_n'], exindict['Sim']['in_n'], exindict['Sim']['ex_n']/exindict['Sim']['in_n'])
dissimlabel = 'Dissim %d / %d = %0.2f' %(exindict['Dissim']['ex_n'], exindict['Dissim']['in_n'], exindict['Dissim']['ex_n']/exindict['Dissim']['in_n'])
ax[4, 1].scatter(exindict['Sim']['ex'], exindict['Sim']['in'], c=sim_c, marker='.', s=8, alpha=0.7, label=simlabel)
ax[4, 1].scatter(exindict['Dissim']['ex'], exindict['Dissim']['in'], c=dissim_c, marker='.', s=8, alpha=0.7, label=dissimlabel)
ax[4, 1].set_xlabel('Ex', fontsize=legendsize)
ax[4, 1].set_ylabel('In', fontsize=legendsize)
ax[4, 1].plot([0, 1], [0, 1], linewidth=0.75, c='k')
ax[4, 1].set_xlim(0, 1)
ax[4, 1].set_ylim(0, 1)
customlegend(ax[4, 1], fontsize=legendsize-2)


bias_edges = np.linspace(-1, 1, 50)
stattext_BW = 'Best-Worst = %0.3f' %( exindict['Best']['ex_bias_mu'] - exindict['Worst']['ex_bias_mu'])
stattext_DS = 'Dissim-Sim = %0.3f\n' %( exindict['Dissim']['ex_bias_mu'] - exindict['Sim']['ex_bias_mu'])
nbins1, _, _ = ax[4, 2].hist(exindict['Best']['ex_bias'], bins=bias_edges, color=direct_c[0], linewidth=0.75, histtype='step', density=True)
ax[4, 2].axvline(exindict['Best']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=direct_c[0])
nbins2, _, _ = ax[4, 2].hist(exindict['Worst']['ex_bias'], bins=bias_edges, color=direct_c[1], linewidth=0.75, histtype='step', density=True)
ax[4, 2].axvline(exindict['Worst']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=direct_c[1])
nbins3, _, _ = ax[4, 3].hist(exindict['Sim']['ex_bias'], bins=bias_edges, color=sim_c, linewidth=0.75, histtype='step', density=True)
ax[4, 3].axvline(exindict['Sim']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=sim_c)
nbins4, _, _ = ax[4, 3].hist(exindict['Dissim']['ex_bias'], bins=bias_edges, color=dissim_c, linewidth=0.75, histtype='step', density=True)
ax[4, 3].axvline(exindict['Dissim']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=dissim_c)

ax[4, 2].set_xlabel('Ex minus In', fontsize=legendsize)
ax[4, 2].set_ylim(0, np.max(np.concatenate([nbins1, nbins2, nbins3, nbins4]))*2)
ax[4, 2].annotate(stattext_BW, xy=(0.2, 0.7), fontsize=legendsize-2, xycoords='axes fraction')
ax[4, 3].set_xlabel('Ex minus In', fontsize=legendsize)
ax[4, 3].set_ylim(0, np.max(np.concatenate([nbins1, nbins2, nbins3, nbins4]))*2)
ax[4, 3].annotate(stattext_DS, xy=(0.2, 0.7), fontsize=legendsize-2, xycoords='axes fraction')

for i in range(4):
    ax[4, i].tick_params(labelsize=legendsize)

fig.savefig(join(save_dir, 'fig3_downsamp.png'), dpi=300)

