from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from pycircstat import cdiff, mean as cmean
from library.comput_utils import pair_diff, circgaufunc, get_tspdiff
from library.shared_vars import total_figw
from library.utils import save_pickle, load_pickle
from library.visualization import customlegend
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p, directional_tuning_tile, simulate_SNN

legendsize = 7
load_dir = 'sim_results/fig2'
save_dir = 'plots/fig2/'
os.makedirs(save_dir, exist_ok=True)
all_nidx_dict = dict()
simdata = load_pickle(join(load_dir, 'fig2_DirectionalExtrinsic_0.pkl'))
simdata180 = load_pickle(join(load_dir, 'fig2_DirectionalExtrinsic_180.pkl'))

fig, ax = plt.subplots(4, 4, figsize=(total_figw, total_figw))
gs = ax[1, 0].get_gridspec()  # Population raster
for axeach in ax[1:3, 0:3].ravel():
    axeach.remove()
ax_popras = fig.add_subplot(gs[1:3, 0:3])



# ======================== Get data =================
BehDF = simdata['BehDF']
SpikeDF = simdata['SpikeDF']
SpikeDF180 = simdata180['SpikeDF']
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
all_nidx_dict[0] = all_nidx


all_egnidxs = all_nidx[[13, 19]]
best_nidx, worst_nidx = all_egnidxs[0], all_egnidxs[1]
direct_c = ['r', 'b']

# ======================== Plotting =================

# # Directional and positional tuning
xlim = (-6, 5)
ylim = (-5, 6)
egnidx = np.argmin(np.square(0 - xxtun1d_ca3) + np.square(0 - yytun1d_ca3) + np.square(0 - aatun1d_ca3))
w2d_preeg = w_ca3ca3[:, egnidx].reshape(nx_ca3, nx_ca3)
arrow_alpha = (w2d_preeg-w2d_preeg.min())/w2d_preeg.max()
arrow_alpha[arrow_alpha<0.1] = 0
ax[0, 0].scatter(xxtun2d_ca3, yytun2d_ca3, c=arrow_alpha, cmap='Blues', s=20)
ax[0, 0].quiver(xxtun2d_ca3, yytun2d_ca3, np.cos(aatun2d_ca3),
                np.sin(aatun2d_ca3), scale=10, alpha=arrow_alpha*0.95+0.05, headwidth=7)
ax[0, 0].set_xlim(*xlim)
ax[0, 0].set_ylim(*ylim)
ax[0, 0].set_xlabel('x (cm)', fontsize=legendsize)
ax[0, 0].set_ylabel('y (cm)', fontsize=legendsize)
ax[0, 0].set_xticks(np.arange(*xlim, 2))
ax[0, 0].set_yticks(np.arange(*ylim, 2))
ax[0, 0].tick_params(labelsize=legendsize)
ax[0, 0].set_title('$W_{i, j}$ from (0, 0) to (x, y)', fontsize=legendsize)

# # Directional tuning curve
deg_space = np.linspace(-np.pi, np.pi, 100)
I_output = Ipos_max_compen + circgaufunc(deg_space, 0, Iangle_kappa, Iangle_diff)
ax[0, 1].plot(deg_space, I_output)
ax[0, 1].set_ylim(0, 18)
ax[0, 1].set_title('Directional tuning', fontsize=legendsize)
ax[0, 1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax[0, 1].set_xticklabels(['$-\pi$', '', '0', '', '$\pi$'])

# # Best and worst sensory input
for i, label in enumerate(['Best', 'Worst']):
    egnidx = all_egnidxs[i]
    rel_dist = traj_x-xxtun1d_ca3[egnidx]
    ax[0, 2+i].plot(rel_dist, Isen[:, egnidx], linewidth=0.75)
    ax[0, 2+i].plot(rel_dist, Isen_fac[:, egnidx], linewidth=0.75)
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for j in theta_cutidx:
        ax[0, 2+i].axvline(rel_dist[j], c='gray', linewidth=0.75, alpha=0.5)
    ax[0, 2+i].set_xticks(np.arange(-8, 9, 4))
    ax[0, 2+i].set_xticks(np.arange(-9, 9, 1), minor=True)
    ax[0, 2+i].set_xlim(-10, 10)
    ax[0, 2+i].set_ylim(0, 30)
ax[0, 2].set_title('Positional input (best)', fontsize=legendsize-2)
ax[0, 3].set_title('Positional input (worst)', fontsize=legendsize-2)

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
ax_popras.tick_params(labelsize=legendsize)

# ax_popras.set_ylim(10, 30)
# ax_popras.set_xlim(t.max()/2-600, t.max()/2+600)

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
    ax[1+i, 3].scatter(dsp_norm_eg, phasesp_eg, marker='|', s=4, c=direct_c[i], label=label)
    ax[1+i, 3].axhline(mean_phasesp, xmin=0, xmax=0.3, linewidth=1, c='gray')
    ax[1+i, 3].plot(xdum, ydum, linewidth=0.75, c='gray')
    ax[1+i, 3].annotate(precess_txt, xy=(0.02, 0.8), xycoords='axes fraction', fontsize=legendsize)
    ax[1+i, 3].annotate('%0.2fHz' % (rate), xy=(0.02, 0.04), xycoords='axes fraction', fontsize=legendsize)
    # ax[1+i, 3].set_xlabel('Position', fontsize=legendsize)
    ax[1+i, 3].set_ylim(0, 2*np.pi)
    ax[1+i, 3].set_yticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])
    ax[1+i, 3].set_yticklabels(['0', '', '$\pi$', '', '$2\pi$'])
    # ax[1+i, 3].set_ylabel('Spike phase (rad)', fontsize=legendsize)
    customlegend(ax[1+i, 3], fontsize=legendsize)

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
    adiff = np.abs(cdiff(atun_this, 0))
    tsp_eg, phasesp_eg = t[tidxsp_tmp], theta_phase[tidxsp_tmp]
    dsp_eg = traj_d[tidxsp_tmp]
    dspmin, dsprange = dsp_eg.min(), dsp_eg.max() - dsp_eg.min()
    dsp_norm_eg = (dsp_eg-dspmin)/dsprange
    regress = rcc(dsp_norm_eg, phasesp_eg, abound=(-1., 1.))
    rcc_c, rcc_slope_rad = regress['phi0'], regress['aopt'] * 2 * np.pi
    if rcc_slope_rad > 0:
        continue

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

# # x,y- coordinates of best-worst sampled neurons
ax[3, 0].plot(traj_x, traj_y, linewidth=0.75, c='gray')
ax[3, 0].scatter(xxtun1d_ca3[nidx_best], yytun1d_ca3[nidx_best], c=direct_c[0], s=1, marker='o', label='Best')
ax[3, 0].scatter(xxtun1d_ca3[nidx_worst], yytun1d_ca3[nidx_worst], c=direct_c[1], s=1, marker='o', label='Worst')
ax[3, 0].set_xlim(-20, 20)
ax[3, 0].set_ylim(-20, 20)
ax[3, 0].set_xlabel('Neuron x (cm)', fontsize=legendsize)
ax[3, 0].set_ylabel('Neuron y (cm)', fontsize=legendsize)
customlegend(ax[3, 0], fontsize=legendsize)

# # Slopes and onsets of best-worst neurons
ax[3, 1].scatter(onset_best, slope_best, marker='.', c=direct_c[0], s=8, alpha=0.5)
ax[3, 1].scatter(onset_worst, slope_worst, marker='.', c=direct_c[1], s=8, alpha=0.5)
ax[3, 1].set_xlim(0, 2*np.pi)
ax[3, 1].set_ylim(-np.pi, 1)
ax[3, 1].set_xlabel('Onset (rad)', fontsize=legendsize)
ax[3, 1].set_ylabel('Slope (rad)', fontsize=legendsize)
ax[3, 1].spines['top'].set_visible(False)
ax[3, 1].spines['right'].set_visible(False)

# # Marginal onset and slope
axposori = ax[3, 1].get_position()
onsetbins = np.linspace(0, 2*np.pi, 60)
ax_maronset = fig.add_axes([axposori.x0, axposori.y0+0.14, axposori.width, axposori.height * 0.2])
binsonsetbest, _, _ = ax_maronset.hist(onset_best, bins=onsetbins, density=True, histtype='step', color=direct_c[0], linewidth=0.75)
binsonsetworst, _, _ = ax_maronset.hist(onset_worst, bins=onsetbins, density=True, histtype='step', color=direct_c[1], linewidth=0.75)
ax_maronset.axvline(onset_bestmu, ymin=0.55, ymax=0.9, linewidth=0.75, color=direct_c[0])
ax_maronset.axvline(onset_worstmu, ymin=0.55, ymax=0.9, linewidth=0.75, color=direct_c[1])
ax_maronset.set_xlim(0, 2*np.pi)
ax_maronset.set_ylim(0, np.max([binsonsetbest.max(), binsonsetworst.max()])*2)
ax_maronset.axis('off')

slopebins = np.linspace(-np.pi, 0, 40)
ax_marslope = fig.add_axes([axposori.x0+0.13, axposori.y0, axposori.width * 0.2, axposori.height])
binsslopebest, _, _ = ax_marslope.hist(slope_best, bins=slopebins, density=True, histtype='step', color=direct_c[0], linewidth=0.75, orientation='horizontal')
binsslopeworst, _, _ = ax_marslope.hist(slope_worst, bins=slopebins, density=True, histtype='step', color=direct_c[1], linewidth=0.75, orientation='horizontal')
ax_marslope.axhline(slope_bestmu, xmin=0.55, xmax=0.9, linewidth=0.75, color=direct_c[0])
ax_marslope.axhline(slope_worstmu, xmin=0.55, xmax=0.9, linewidth=0.75, color=direct_c[1])
ax_marslope.set_xlim(0, np.max([binsslopebest.max(), binsslopeworst.max()])*2)
ax_marslope.set_ylim(-np.pi, 0)
ax_marslope.axis('off')

# # Marginal spike phases
phasebins = np.linspace(0, 2*np.pi, 30)
binsphasebest, _, _ = ax[3, 2].hist(phasesp_best, bins=phasebins, density=True, histtype='step', color=direct_c[0], label='Best=%0.2f (n=%d)'%(phasesp_bestmu, phasesp_best.shape[0]), linewidth=0.75)
binsphaseworst, _, _ = ax[3, 2].hist(phasesp_worst, bins=phasebins, density=True, histtype='step', color=direct_c[1], label='Worst=%0.2f (n=%d)'%(phasesp_worstmu, phasesp_worst.shape[0]), linewidth=0.75)
ax[3, 2].axvline(phasesp_bestmu, ymin=0.55, ymax=0.65, linewidth=0.75, color=direct_c[0])
ax[3, 2].axvline(phasesp_worstmu, ymin=0.55, ymax=0.65, linewidth=0.75, color=direct_c[1])
ax[3, 2].annotate('diff=\n$%0.2f$'%(phasesp_worstmu-phasesp_bestmu), xy=(0.8, 0.5), xycoords='axes fraction', fontsize=legendsize)
ax[3, 2].set_ylim(0, np.max([binsphasebest.max(), binsphaseworst.max()])*2)
ax[3, 2].set_xlabel('Marginal spike phase (rad)', fontsize=legendsize)
customlegend(ax[3, 2], fontsize=legendsize-2)

# # # Correlation
edges = np.arange(-100, 100, 5)
target_xdiff, epsilon = 4, 0.1
all_tsp_diff_list = []
all_tsp_diff_180_list = []
nidx_corr = np.concatenate([nidx_best, nidx_worst])
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
            if (np.abs(xdiff) > (target_xdiff+epsilon)) or (np.abs(xdiff) < (target_xdiff-epsilon)):
                continue
            nidx1, nidx2 = sorted_sampled_nidx[pi], sorted_sampled_nidx[pj]
            tsp_diff = get_tspdiff(SpikeDF, t, nidx1, nidx2)
            tsp_diff180 = get_tspdiff(SpikeDF180, t, nidx1, nidx2)
            all_tsp_diff_list.append(tsp_diff)
            all_tsp_diff_180_list.append(tsp_diff180)
all_tsp_diff = np.concatenate(all_tsp_diff_list)
all_tsp_diff_180 = np.concatenate(all_tsp_diff_180_list)
ax[3, 3].hist(all_tsp_diff, bins=edges, histtype='step', linewidth=0.75)
ax[3, 3].hist(all_tsp_diff_180, bins=edges, histtype='step', linewidth=0.75)

for i in range(4):
    ax[0, i].tick_params(labelsize=legendsize)
    ax[3, i].tick_params(labelsize=legendsize)
ax[1, 3].tick_params(labelsize=legendsize)
ax[2, 3].tick_params(labelsize=legendsize)


for axeach in [ax_popras]:
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for i in theta_cutidx:
        axeach.axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)
fig.savefig(join(save_dir, 'fig2.png'), dpi=300)

