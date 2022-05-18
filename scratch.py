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

legendsize = 8
load_dir = 'sim_results/fig4'
save_dir = 'plots/test/'
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


        # # Plot CA3-Mos and Mos-CA3 matrices
        egnidx_premos = np.argmin(np.square(0 - xxtun1d_mos) + np.square(0 - yytun1d_mos) + np.square(0 - aatun1d_mos))
        pre_mos_x, pre_mos_y = xxtun1d_mos[egnidx_premos], yytun1d_mos[egnidx_premos]
        egnidx_preca3 = np.argmin(np.square(pre_mos_x - xxtun1d_ca3) + np.square(pre_mos_y - yytun1d_ca3) + np.square(0 - aatun1d_ca3))
        pre_ca3_x, pre_ca3_y = xxtun1d_ca3[egnidx_preca3], yytun1d_ca3[egnidx_preca3]
        post_ca3mos_w = w_ca3mos[:, egnidx_preca3].reshape(nx_mos, ny_mos)
        post_mosca3_w = w_mosca3[:, egnidx_premos].reshape(nx_ca3, ny_ca3)

        # Plot CA3-Mos
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.scatter(xxtun1d_mos, yytun1d_mos, c=post_ca3mos_w, cmap='Blues')
        ax.scatter(pre_mos_x, pre_mos_y, c='r', marker='x')
        plt.colorbar(im, ax=ax_config)
        ax.set_xlabel('x (cm)', fontsize=legendsize+2)
        ax.set_ylabel('y (cm)', fontsize=legendsize+2)
        ax.set_title('CA3-Mos synaptic weights', fontsize=legendsize+2)
        ax.tick_params(labelsize=legendsize+2)
        fig.savefig(join(save_dir, 'ca3mos_w.png'))
        plt.close(fig)

        # Plot Mos-CA3
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.scatter(xxtun1d_ca3, yytun1d_ca3, c=post_mosca3_w, cmap='Blues')
        ax.scatter(pre_ca3_x, pre_ca3_y, c='r', marker='x')
        plt.colorbar(im, ax=ax_config)
        ax.set_xlabel('x (cm)', fontsize=legendsize+2)
        ax.set_ylabel('y (cm)', fontsize=legendsize+2)
        ax.set_title('Mos-CA3 synaptic weights', fontsize=legendsize+2)
        ax.tick_params(labelsize=legendsize+2)
        fig.savefig(join(save_dir, 'mosca3_w.png'))
        plt.close(fig)

    break
    break