from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from pycircstat import cdiff, mean as cmean
from library.comput_utils import pair_diff, circgaufunc, get_tspdiff, rcc_wrapper
from library.script_wrappers import best_worst_analysis, find_nidx_along_traj
from library.shared_vars import total_figw
from library.utils import save_pickle, load_pickle
from library.visualization import customlegend, plot_phase_precession, plot_popras, plot_sca_onsetslope, \
    plot_marginal_phase
from library.linear_circular_r import rcc


# ====================================== Global params and paths ==================================
legendsize = 8
load_dir = 'sim_results/fig2'
save_dir = 'plots/fig2/'
os.makedirs(save_dir, exist_ok=True)
plt.rcParams.update({'font.size': legendsize,
                     "axes.titlesize": legendsize,
                     'axes.labelpad': 0,
                     'axes.titlepad': 0,
                     'xtick.major.pad': 0,
                     'ytick.major.pad': 0,
                     'lines.linewidth': 1,
                     'figure.figsize': (5.2, 4.5),
                     'figure.dpi': 300,
                     'axes.spines.top': False,
                     'axes.spines.right': False,
                     })
# ====================================== Figure initialization ==================================

ax_h = 1/3
ax_w = 1/4
hgap = 0.14
wgap = 0.08
xshift_WI1 = 0.03
xshift_WI2 = -0.04
xshift_WIeg2 = 0.02
xshift_WIeg3 = -0.04

xshift_WIeg2_2 = 0.09
xshift_WIeg3_2 = 0.03

xshift_ras = 0.06
xshift_popstats123 = 0.02
hgap_popstats2 = 0.02
wgap_popstats2 = 0.02
hgap_popstats = 0.02
yshift_popstats = 0.02

fig = plt.figure()

ax_WI = [
    fig.add_axes([ax_w * 0 + wgap/2 + xshift_WI1, 1 - ax_h + hgap/2, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 1 + wgap/2 + xshift_WIeg2, 1 - ax_h + hgap/2, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshift_WIeg3, 1 - ax_h + hgap/2, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2 + xshift_WI2, 1 - ax_h + hgap/2, ax_w - wgap, ax_h - hgap]),
]

ax_ras = fig.add_axes([ax_w * 0 + wgap/2 + xshift_ras, 1 - ax_h*2 + hgap/2, ax_w * 2 - wgap, ax_h - hgap])


ax_eg = [
    fig.add_axes([ax_w * 2 + wgap/2 + xshift_WIeg2_2, 1 - ax_h*2 + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2 + xshift_WIeg3_2, 1 - ax_h*2 + hgap/2, ax_w * 1 - wgap, ax_h - hgap])
]

ax_popstats = [
    fig.add_axes([ax_w * 0 + wgap/2 + xshift_popstats123, 1 - ax_h*3 + hgap/2 + hgap_popstats/2 + yshift_popstats, ax_w * 1 - wgap, ax_h - hgap - hgap_popstats]),
    fig.add_axes([ax_w * 1 + wgap/2 + xshift_popstats123, 1 - ax_h*3 + hgap/2 + hgap_popstats/2 + yshift_popstats, ax_w * 1 - wgap - wgap_popstats2, ax_h - hgap - hgap_popstats2 - hgap_popstats]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshift_popstats123, 1 - ax_h*3 + hgap/2 + hgap_popstats/2 + yshift_popstats, ax_w * 1 - wgap, ax_h - hgap - hgap_popstats]),
    fig.add_axes([ax_w * 3 + wgap/2, 1 - ax_h*3 + hgap/2 + hgap_popstats/2 + yshift_popstats, ax_w * 1 - wgap, ax_h - hgap - hgap_popstats]),
]


ax_list = [ax_WI, [ax_ras], ax_eg, ax_popstats]
ax_all = np.concatenate(ax_list)


# ======================================Analysis and plotting ==================================


# ======================== Get data =================
all_nidx_dict = dict()
simdata = load_pickle(join(load_dir, 'fig2_DirectionalExtrinsic_0.pkl'))
simdata180 = load_pickle(join(load_dir, 'fig2_DirectionalExtrinsic_180.pkl'))


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
all_nidx = find_nidx_along_traj(traj_x, traj_y, xxtun1d_ca3, yytun1d_ca3)
all_nidx_dict[0] = all_nidx

all_egnidxs = all_nidx[[13, 19]]
best_nidx, worst_nidx = all_egnidxs[0], all_egnidxs[1]
direct_c = ['tomato', 'royalblue']
direct_c2 = ['rosybrown', 'lightsteelblue']

# ======================== Plotting =================

# # Directional and positional tuning
xlim = (-6, 5)
ylim = (-5, 6)
egnidx = np.argmin(np.square(0 - xxtun1d_ca3) + np.square(0 - yytun1d_ca3) + np.square(0 - aatun1d_ca3))
plot_nidx = np.where((xxtun1d_ca3 < xlim[1]) & (xxtun1d_ca3 > xlim[0]) & (yytun1d_ca3 < ylim[1]) & (yytun1d_ca3 > ylim[0]))[0]
w2d_preeg = w_ca3ca3[plot_nidx, egnidx]
arrow_alpha = (w2d_preeg-w2d_preeg.min())/w2d_preeg.max()
# arrow_alpha[arrow_alpha<0.1] = 0
im = ax_WI[3].scatter(xxtun1d_ca3[plot_nidx], yytun1d_ca3[plot_nidx], c=arrow_alpha, cmap='Blues', s=20)
ax_WI[3].quiver(xxtun1d_ca3[plot_nidx], yytun1d_ca3[plot_nidx], np.cos(aatun1d_ca3[plot_nidx]),
                np.sin(aatun1d_ca3[plot_nidx]), scale=10, alpha=arrow_alpha, headwidth=7)
ax_WI[3].set_xlim(*xlim)
ax_WI[3].set_ylim(*ylim)
ax_WI[3].set_xlabel('x (cm)', fontsize=legendsize, labelpad=0)
ax_WI[3].set_ylabel('y (cm)', fontsize=legendsize, labelpad=-2)
ax_WI[3].set_xticks(np.arange(-5, 6, 5))
ax_WI[3].set_xticks(np.arange(-5, 5, 1), minor=True)
ax_WI[3].set_yticks(np.arange(-5, 6, 5))
ax_WI[3].set_yticks(np.arange(-5, 6, 1), minor=True)
ax_WI[3].tick_params(labelsize=legendsize)
norm = mpl.colors.Normalize(vmin=w2d_preeg.min(), vmax=w2d_preeg.max())
ax_WIpos = ax_WI[3].get_position()
ax_WIcb = fig.add_axes([ax_WIpos.x0 + ax_WIpos.width, ax_WIpos.y0, ax_WIpos.width*0.075, ax_WIpos.height])
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='Blues'),
             cax=ax_WIcb, orientation='vertical')
cb.set_ticks(np.arange(0, 3001, 1000))
cb.set_ticklabels(['0', '1', '2', '3'])  # add x1e3
cb.set_ticks(np.arange(0, 3501, 500), minor=True)
cb.ax.tick_params(labelsize=legendsize, pad=1)
cb.ax.set_ylabel('$W_{ij}$', fontsize=legendsize, labelpad=0.5)

# # Directional tuning curve
deg_space = np.linspace(-np.pi, np.pi, 100)
I_output = circgaufunc(deg_space, 0, Iangle_kappa, Iangle_diff)
ax_WI[0].plot(deg_space, I_output - I_output.min(), color='k', linewidth=0.75)
ax_WI[0].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax_WI[0].set_xticklabels(['$-\pi$', '', '0', '', '$\pi$'])
ax_WI[0].set_xlabel('Heading (rad)', fontsize=legendsize, labelpad=0)
ax_WI[0].set_yticks(np.arange(0, 11, 5))
ax_WI[0].set_yticks(np.arange(0, 10, 1), minor=True)
ax_WI[0].set_ylabel('$I_{HD}$ (mA)', fontsize=legendsize,  labelpad=-2)

# # Best and worst sensory input
for i, label in enumerate(['Best', 'Worst']):
    egnidx = all_egnidxs[i]
    rel_dist = traj_x-xxtun1d_ca3[egnidx]
    ax_WI[1+i].plot(rel_dist, Isen[:, egnidx], linewidth=0.5, color=direct_c2[i], linestyle='dashed')
    ax_WI[1+i].plot(rel_dist, Isen_fac[:, egnidx], linewidth=0.5, color=direct_c[i])
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for j in theta_cutidx:
        ax_WI[1+i].axvline(rel_dist[j], c='gray', linewidth=0.25)
    ax_WI[1+i].set_xticks(np.arange(-8, 9, 4))
    ax_WI[1+i].set_xticks(np.arange(-9, 9, 1), minor=True)
    ax_WI[1+i].set_xlim(-10, 10)
    ax_WI[1+i].set_yticks(np.arange(0, 31, 10))
    ax_WI[1+i].set_yticks(np.arange(0, 31, 5), minor=True)
    ax_WI[1+i].set_ylim(0, 30)

ax_WI[2].plot([-4, 4], [-5, -5], linewidth=0.5, color=direct_c2[0], linestyle='dashed', label=r'${\rm Best}$')
ax_WI[2].plot([-4, 4], [-5, -5], linewidth=0.5, color=direct_c[0], label=r'${\rm Best_{fac}}$')
ax_WI[2].plot([-4, 4], [-5, -5], linewidth=0.5, color=direct_c2[1], linestyle='dashed', label=r'${\rm Worst}$')
ax_WI[2].plot([-4, 4], [-5, -5], linewidth=0.5, color=direct_c[1], label=r'${\rm Worst_{fac}}$')
customlegend(ax_WI[2], fontsize=legendsize, loc='upper right', handlelength=1.5, handletextpad=0.1)
ax_WI[1].set_ylabel('$I_{Sen}$ (mA)', fontsize=legendsize, labelpad=0)
ax_WI[2].set_yticklabels([])
fig.text(0.375, 0.67, 'Relative position to\nplace cell center (cm)', fontsize=legendsize, va='center')

# Population Raster
plot_popras(ax_ras, SpikeDF, t, all_nidx, all_egnidxs[0], all_egnidxs[1], direct_c[0], direct_c[1])
ax_ras.set_ylim(5, 25)
ax_ras.set_xlim(t.max()/2-700, t.max()/2+500)
ax_ras.set_xlabel('Time (ms)', fontsize=legendsize, labelpad=0)
ax_ras.set_ylabel('Place cell index\n along the trajectory', fontsize=legendsize, labelpad=0)

# Phase precession
for i, label in enumerate(['Best', 'Worst']):
    egnidx = all_egnidxs[i]
    tidxsp_eg = SpikeDF.loc[SpikeDF['neuronid'] == egnidx, 'tidxsp'].to_numpy()
    tsp_eg, phasesp_eg = t[tidxsp_eg], theta_phase[tidxsp_eg]
    dsp_eg = traj_d[tidxsp_eg]
    plot_phase_precession(ax_eg[i], dsp_eg, phasesp_eg, s=4, c=direct_c[i], fontsize=legendsize,
                          plotmeanphase=True)
ax_eg[0].set_ylabel('Spike phase (rad)', fontsize=legendsize, labelpad=0)
ax_eg[1].set_yticklabels([])
fig.text(0.66, 0.35, 'Relative position in the field', fontsize=legendsize)


# # All onsets & Marginal phases
precessdf, info_best, info_worst = best_worst_analysis(SpikeDF, 0, range(nn_ca3), t, theta_phase, traj_d, xxtun1d, aatun1d, abs_xlim=20)
phasesp_best, onset_best, slope_best, nidx_best = info_best
phasesp_worst, onset_worst, slope_worst, nidx_worst = info_worst
fig2, ax2 = plt.subplots(1, 1, figsize=(2, 2))
ax2.scatter(precessdf['onset'], precessdf['slope'], c=precessdf['adiff'], cmap='jet', marker='.', s=1)
ax2.set_xlim(0.5*np.pi, 1.5*np.pi)
ax2.set_ylim(-np.pi, 0)
fig2.savefig(join(save_dir, 'fig2_allangles.png'), dpi=300)


# # x,y- coordinates of best-worst sampled neurons
ax_popstats[0].plot(traj_x, traj_y, linewidth=0.75, c='gray')
ax_popstats[0].scatter(xxtun1d_ca3[nidx_best], yytun1d_ca3[nidx_best], c=direct_c[0], s=1, marker='o')
ax_popstats[0].scatter(xxtun1d_ca3[nidx_worst], yytun1d_ca3[nidx_worst], c=direct_c[1], s=1, marker='o')
ax_popstats[0].set_xlim(-20, 20)
ax_popstats[0].set_ylim(-20, 20)
ax_popstats[0].set_xlabel('x (cm)', fontsize=legendsize, labelpad=0)
ax_popstats[0].set_ylabel('y (cm)', fontsize=legendsize, labelpad=-10)

# # Slopes and onsets of best-worst neurons
plot_sca_onsetslope(fig, ax_popstats[1], onset_best, slope_best, onset_worst, slope_worst,
                    onset_lim=(0.5*np.pi, 1.5*np.pi), slope_lim=(-np.pi, 0), direct_c=direct_c)

ax_popstats[1].set_xlim(0.5*np.pi, 1.5*np.pi)
ax_popstats[1].set_xticks([0.5*np.pi, np.pi, 1.5*np.pi])
ax_popstats[1].set_xticks(np.arange(0.5*np.pi, 1.5*np.pi, 0.25*np.pi), minor=True)
ax_popstats[1].set_xticklabels(['$0.5\pi$', '$\pi$', '$1.5\pi$'])
ax_popstats[1].set_xlabel('Onset (rad)', fontsize=legendsize, labelpad=0)
ax_popstats[1].set_ylim(-np.pi, 0)
ax_popstats[1].set_yticks([-np.pi, -np.pi/2, 0])
ax_popstats[1].set_yticks(np.arange(-np.pi, 0.1, np.pi/4), minor=True)
ax_popstats[1].set_yticklabels(['$-\pi$', r'$\frac{-\pi}{2}$', '0'])
ax_popstats[1].set_ylabel('Slope (rad)', fontsize=legendsize, labelpad=-2)


# # Marginal spike phases
plot_marginal_phase(ax_popstats[2], phasesp_best, phasesp_worst, direct_c, legendsize)

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
ax_popstats[3].hist(all_tsp_diff, bins=edges, histtype='step', linewidth=0.75, density=True)
ax_popstats[3].hist(all_tsp_diff_180, bins=edges, histtype='step', linewidth=0.75, density=True)
ax_popstats[3].set_xticks((-100, -50, 0, 50, 100))
ax_popstats[3].set_xticklabels(('-100', '', '0', '', '100'))
ax_popstats[3].set_xticks(np.arange(-100, 101, 10), minor=True)
ax_popstats[3].set_xlabel('Time lag (ms)', fontsize=legendsize, labelpad=0)
ax_popstats[3].set_ylabel('Spike density', fontsize=legendsize, labelpad=0)
ax_popstats[3].set_ylim(0, 0.0175)
ax_popstats[3].set_yticks([0, 0.01])
ax_popstats[3].set_yticks(np.arange(0, 0.02, 0.005), minor=True)
ax_popstats[3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ax_popstats[3].yaxis.get_offset_text().set_visible(False)
ax_popstats[3].annotate(r'$\times 10^{-1}$', xy=(0.01, 0.9), xycoords='axes fraction', fontsize=legendsize)

for axeach in [ax_ras]:
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for i in theta_cutidx:
        axeach.axvline(t[i], c='gray', linewidth=0.25)
fig.savefig(join(save_dir, 'fig2.png'), dpi=300)
fig.savefig(join(save_dir, 'fig2.pdf'), dpi=300)
fig.savefig(join(save_dir, 'fig2.svg'), dpi=300)
