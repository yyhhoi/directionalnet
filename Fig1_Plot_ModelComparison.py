from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from pycircstat import cdiff, mean as cmean
from library.comput_utils import pair_diff
from library.shared_vars import total_figw
from library.utils import save_pickle, load_pickle
from library.visualization import customlegend, plot_phase_precession, plot_popras
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p, directional_tuning_tile, simulate_SNN


# ====================================== Global params and paths ==================================
legendsize = 8
load_dir = 'sim_results/fig1'
save_dir = 'plots/fig1/'
os.makedirs(save_dir, exist_ok=True)
save_pth = join(save_dir, 'fig1')

# ====================================== Figure initialization ==================================
figw = 5.2
figh = 5.2
ax_h = 1/7
ax_w = 1/7
hgap = 0.04
ylift_precess = 0.015
ylift_12 = 0.04
ylift_3 = 0.02
wgap = 0.02
wgap_corr = 0.05
fig = plt.figure(figsize=(figw, figh))

ax_trajs = [
    fig.add_axes([ax_w * 1.5 + wgap/2, 1 - ax_h + hgap/2, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3.5 + wgap/2, 1 - ax_h + hgap/2, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 5.5 + wgap/2, 1 - ax_h + hgap/2, ax_w - wgap, ax_h - hgap]),
]

ygp_in = 0.02

ax_ras_in = [
    fig.add_axes([ax_w * 1 + wgap/2, 1 - ax_h*2 + hgap/2 - ygp_in + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2, 1 - ax_h*2 + hgap/2 - ygp_in + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2, 1 - ax_h*2 + hgap/2 - ygp_in + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 4 + wgap/2, 1 - ax_h*2 + hgap/2 - ygp_in + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 5 + wgap/2, 1 - ax_h*2 + hgap/2 - ygp_in + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 6 + wgap/2, 1 - ax_h*2 + hgap/2 - ygp_in + ylift_12, ax_w - wgap, ax_h - hgap]),
]

ax_precess_in = [
    fig.add_axes([ax_w * 1 + wgap/2, 1 - ax_h*3 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2, 1 - ax_h*3 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2, 1 - ax_h*3 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 4 + wgap/2, 1 - ax_h*3 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 5 + wgap/2, 1 - ax_h*3 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 6 + wgap/2, 1 - ax_h*3 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
]

ax_corr_in = [
    fig.add_axes([ax_w * 1 + wgap/2 + wgap_corr/2, 1 - ax_h*4 + hgap/2 + ygp_in + ylift_3, ax_w*2 - wgap - wgap_corr, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2 + wgap_corr/2, 1 - ax_h*4 + hgap/2 + ygp_in + ylift_3, ax_w*2 - wgap - wgap_corr, ax_h - hgap]),
    fig.add_axes([ax_w * 5 + wgap/2 + wgap_corr/2, 1 - ax_h*4 + hgap/2 + ygp_in + ylift_3, ax_w*2 - wgap - wgap_corr, ax_h - hgap]),
]


ygp_ex = 0.02
ax_ras_ex = [
    fig.add_axes([ax_w * 1 + wgap/2, 1 - ax_h*5 + hgap/2 - ygp_ex + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2, 1 - ax_h*5 + hgap/2 - ygp_ex + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2, 1 - ax_h*5 + hgap/2 - ygp_ex + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 4 + wgap/2, 1 - ax_h*5 + hgap/2 - ygp_ex + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 5 + wgap/2, 1 - ax_h*5 + hgap/2 - ygp_ex + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 6 + wgap/2, 1 - ax_h*5 + hgap/2 - ygp_ex + ylift_12, ax_w - wgap, ax_h - hgap]),
]

ax_precess_ex = [
    fig.add_axes([ax_w * 1 + wgap/2, 1 - ax_h*6 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2, 1 - ax_h*6 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2, 1 - ax_h*6 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 4 + wgap/2, 1 - ax_h*6 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 5 + wgap/2, 1 - ax_h*6 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 6 + wgap/2, 1 - ax_h*6 + hgap/2 + ylift_precess + ylift_12, ax_w - wgap, ax_h - hgap]),
]

ax_corr_ex = [
    fig.add_axes([ax_w * 1 + wgap/2 + wgap_corr/2, 1 - ax_h*7 + hgap/2 + ygp_ex + ylift_3, ax_w*2 - wgap - wgap_corr, ax_h - hgap]),
    fig.add_axes([ax_w * 3 + wgap/2 + wgap_corr/2, 1 - ax_h*7 + hgap/2 + ygp_ex + ylift_3, ax_w*2 - wgap - wgap_corr, ax_h - hgap]),
    fig.add_axes([ax_w * 5 + wgap/2 + wgap_corr/2, 1 - ax_h*7 + hgap/2 + ygp_ex + ylift_3, ax_w*2 - wgap - wgap_corr, ax_h - hgap]),
]
ax_list = [ax_trajs, ax_ras_in, ax_precess_in, ax_corr_in, ax_ras_ex, ax_precess_ex, ax_corr_ex]
ax_all = np.concatenate(ax_list)

for ax_each in ax_all:
    ax_each.tick_params(labelsize=legendsize)
    ax_each.spines['top'].set_visible(False)
    ax_each.spines['right'].set_visible(False)

# ======================================Analysis and plotting ==================================

BehDF_degpairs = [(0, 180), (45, 225), (90, 270)]
trajxshift = {0:0, 180:0, 90:-5, 270:5, 45:-5, 225:5}
trajyshift = {0:5, 180:-5, 90:0, 270:0, 45:5, 225:-5}
direct_c = ['darkorange', 'mediumspringgreen']
all_nidx_dict = dict()
for exin_id, exin_tag in enumerate(['intrinsic', 'extrinsic']):

    for Behdeg_pairid, Behdegpair in enumerate(BehDF_degpairs):


        for Behdeg_id, BehDF_deg in enumerate(Behdegpair):


            print(exin_tag, BehDF_deg)
            simdata = load_pickle(join(load_dir, 'fig1_%s_%d.pkl'%(exin_tag, BehDF_deg)))

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

            w_ca3ca3 = w[:nn_ca3, :nn_ca3]
            xxtun1d_ca3 = xxtun1d[:nn_ca3]
            yytun1d_ca3 = yytun1d[:nn_ca3]
            aatun1d_ca3 = aatun1d[:nn_ca3]
            nx_ca3, ny_ca3 = config_dict['nx_ca3'], config_dict['ny_ca3']
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
            all_nidx_dict[BehDF_deg] = all_nidx

            mid_nidx_id = int(all_nidx.shape[0]/2)
            all_egnidxs = [all_nidx[mid_nidx_id-1], all_nidx[mid_nidx_id], all_nidx[mid_nidx_id+1]]

            # ======================== Plotting =================
            # Trajectories
            rowid1, colid1 = 0, Behdeg_pairid
            if exin_id == 0:
                ax_list[rowid1][colid1].arrow(traj_x[0] + trajxshift[BehDF_deg], traj_y[0] + trajyshift[BehDF_deg],
                                              dx=traj_x[-1] - traj_x[0], dy=traj_y[-1] - traj_y[0], width=0.01,
                                              head_width=4, color=direct_c[Behdeg_id])
                ax_list[rowid1][colid1].set_xlim(-40, 40)
                ax_list[rowid1][colid1].set_ylim(-40, 40)
                ax_list[rowid1][colid1].set_xticks([-40, 0, 40])
                ax_list[rowid1][colid1].set_yticks([-40, 0, 40])
                ax_list[rowid1][colid1].axis('off')



            # Raster plot - CA3 pyramidal
            rowid2, colid2 = 1+exin_id*3, 0 + Behdeg_pairid*2 + Behdeg_id
            plot_popras(ax_list[rowid2][colid2], SpikeDF, t, all_nidx, all_egnidxs[0], all_egnidxs[1], 'gray', direct_c[Behdeg_id])
            ax_list[rowid2][colid2].set_ylim(8, 32)
            ax_list[rowid2][colid2].set_xlim(t.max()/2-400, t.max()/2+400)
            ax_list[rowid2][colid2].set_xticks([])
            ax_list[rowid2][colid2].set_yticks([])
            ax_list[rowid2][colid2].spines['bottom'].set_visible(False)
            ax_list[rowid2][colid2].spines['left'].set_visible(False)
            theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
            for i in theta_cutidx:
                ax_list[rowid2][colid2].axvline(t[i], c='gray', linewidth=0.25)


            # Phase precession
            rowid3, colid3 = 2 + exin_id*3, 0 + Behdeg_pairid*2 + Behdeg_id
            egnidx = all_egnidxs[1]
            tidxsp_eg = SpikeDF.loc[SpikeDF['neuronid'] == egnidx, 'tidxsp'].to_numpy()
            tsp_eg, phasesp_eg = t[tidxsp_eg], theta_phase[tidxsp_eg]
            dsp_eg = traj_d[tidxsp_eg]
            plot_phase_precession(ax_list[rowid3][colid3], dsp_eg, phasesp_eg, s=4, c=direct_c[Behdeg_id],
                                  fontsize=legendsize, plotmeanphase=False, statxy=(0.27, 0.7))

            # Correlation
            rowid4, colid4 = 3 + exin_id*3, Behdeg_pairid
            id_dist = 5
            if (exin_tag=='intrinsic'):
                if (BehDF_deg == 0) or (BehDF_deg == 180):
                    id_dist = 8
                if (BehDF_deg == 45) or (BehDF_deg == 225):
                    id_dist = 7
            all_tsp_diff_list = []
            ref_all_nidx = all_nidx_dict[Behdegpair[0]]
            for i in range(ref_all_nidx.shape[0]-id_dist):
                corr_egnid1, corr_egnid2 = ref_all_nidx[i], ref_all_nidx[i + id_dist]
                tidxsp_eg1 = SpikeDF.loc[SpikeDF['neuronid'] == corr_egnid1, 'tidxsp'].to_numpy()
                tidxsp_eg2 = SpikeDF.loc[SpikeDF['neuronid'] == corr_egnid2, 'tidxsp'].to_numpy()
                tsp_eg1, tsp_eg2 = t[tidxsp_eg1], t[tidxsp_eg2]
                tsp_diff = pair_diff(tsp_eg1, tsp_eg2).flatten()
                tsp_diff = tsp_diff[np.abs(tsp_diff) < 100]
                all_tsp_diff_list.append(tsp_diff)
            all_tsp_diff = np.concatenate(all_tsp_diff_list)
            corr_bins = np.linspace(-100, 100, 41)
            ax_list[rowid4][colid4].hist(all_tsp_diff, bins=corr_bins, histtype='step', color=direct_c[Behdeg_id], linewidth=0.75, density=True)
            ax_list[rowid4][colid4].spines['right'].set_visible(False)
            ax_list[rowid4][colid4].spines['top'].set_visible(False)
            ax_list[rowid4][colid4].set_xticks((-100, -50, 0, 50, 100))
            ax_list[rowid4][colid4].set_xticklabels(('-100', '', '0', '', '100'))
            ax_list[rowid4][colid4].set_xticks(np.arange(-100, 101, 10), minor=True)
            ax_list[rowid4][colid4].set_xlabel('Time lag (ms)', fontsize=legendsize, labelpad=0)
            ax_list[rowid4][colid4].set_yticks([0, 0.02])
            ax_list[rowid4][colid4].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
            ax_list[rowid4][colid4].yaxis.get_offset_text().set_y(-0.4)
            ax_list[rowid4][colid4].yaxis.get_offset_text().set_fontsize(legendsize)


            del simdata

for i in range(6):
    if i > 0:
        ax_precess_in[i].set_yticklabels([])
        ax_precess_ex[i].set_yticklabels([])

ax_precess_in[0].set_ylabel('Phase (rad)', fontsize=legendsize, labelpad=0)
ax_precess_ex[0].set_ylabel('Phase (rad)', fontsize=legendsize, labelpad=0)

for j in range(3):
    ax_corr_in[j].set_ylim(0, 0.035)
    ax_corr_in[j].tick_params(axis="x", pad=1)
    ax_corr_ex[j].set_ylim(0, 0.025)
    if j > 0:
        ax_corr_in[j].set_yticklabels([])
        ax_corr_ex[j].set_yticklabels([])
ax_corr_in[0].set_ylabel('Spike density', fontsize=legendsize, labelpad=0)
ax_corr_ex[0].set_ylabel('Spike density', fontsize=legendsize, labelpad=0)



fig.savefig(save_pth + '.png', dpi=300)
fig.savefig(save_pth + '.eps')
fig.savefig(save_pth + '.pdf')
plt.close(fig)

