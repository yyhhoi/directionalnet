from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from library.correlogram import ThetaEstimator
from library.utils import load_pickle
from library.linear_circular_r import rcc

# ====================================== Global params and paths ==================================
legendsize = 8
load_dir = 'sim_results/fig4'
save_dir = 'plots/fig4/'
plt.rcParams.update({'font.size': legendsize,
                     "axes.titlesize": legendsize,
                     'axes.labelpad': 0,
                     'axes.titlepad': 0,
                     'xtick.major.pad': 0,
                     'ytick.major.pad': 0,
                     'lines.linewidth': 1,
                     })
# ====================================== Figure initialization ==================================
figw = 3.5
figh = 2

ax_h = 1/2
ax_w = 1/3
hgap = 0.15
wgap = 0.1

xshift_c0 = 0.04
xshift_c1 = 0.08
xshift_c2 = 0.04

yshift_r1 = 0.05


fig = plt.figure(figsize=(figw, figh))

ax_sen = fig.add_axes([ax_w * 0 + wgap/2 + xshift_c0, 1 - ax_h * 1.5 + hgap/2, ax_w * 1 - wgap, ax_h - hgap])

ax_ctrl = np.array([
    fig.add_axes([ax_w * 1 + wgap/2 + xshift_c1, 1 - ax_h + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshift_c2, 1 - ax_h + hgap/2, ax_w * 1 - wgap, ax_h - hgap]),
])


ax_lesion = np.array([
    fig.add_axes([ax_w * 1 + wgap/2 + xshift_c1, 1 - ax_h * 2 + hgap/2 + yshift_r1, ax_w * 1 - wgap, ax_h - hgap]),
    fig.add_axes([ax_w * 2 + wgap/2 + xshift_c2, 1 - ax_h * 2 + hgap/2 + yshift_r1, ax_w * 1 - wgap, ax_h - hgap]),
])

ax_corr = np.stack([ax_ctrl, ax_lesion])

ax = np.append(ax_sen, np.concatenate([ax_ctrl, ax_lesion]))

for ax_each in ax.ravel():
    ax_each.spines['top'].set_visible(False)
    ax_each.spines['right'].set_visible(False)

# ======================================Analysis and plotting ==================================


mosdeg_pair = (0, 180)
dgcase_c = ['lime', 'm']
for dgid, dglabel in enumerate(['Ctrl', 'DGlesion']):
    for mosdeg_id, mosdeg in enumerate(mosdeg_pair):
        print('Mos %d %s'%(mosdeg, dglabel))
        simdata = load_pickle(join(load_dir, 'fig4_MossyLayer_Mosdeg%d_%s.pkl' % (mosdeg, dglabel)))

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
        theta_f = config_dict['theta_f']
        dt = config_dict['dt']
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

        # # Removing EC STF for Control case
        if mosdeg == 0:
            egnidx = all_egnidxs[0]
            rel_dist = traj_x-xxtun1d_ca3[egnidx]
            ax_sen.plot(rel_dist, Isen_fac[:, egnidx], linewidth=0.5, color=dgcase_c[dgid])
            theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
            for j in theta_cutidx:
                ax_sen.axvline(rel_dist[j], c='gray', linewidth=0.25)
            ax_sen.set_xticks(np.arange(-8, 9, 4))
            ax_sen.set_xticklabels(['-8', '', '0', '', '8'])
            ax_sen.set_xticks(np.arange(-9, 9, 1), minor=True)
            ax_sen.set_xlim(-10, 10)
            ax_sen.set_xlabel('$\Delta x$ (cm)')
            ax_sen.set_ylabel('$I_{sen} (mA)$')
            # ax_sen.set_yticks(np.arange(0, 31, 10))
            # ax_sen.set_yticks(np.arange(0, 31, 5), minor=True)
            # ax_sen.set_ylim(0, 30)


        # # Correlation lags for Control and DG Lesion
        xdiffmax = 20 if dglabel == 'Ctrl' else 20
        # xlimmax = 22
        TE = ThetaEstimator(bin_size=5e-3, window_width=200e-3, bandpass=(5, 12))
        thetaT = 1/theta_f
        allxdiff, allcorrlag = [], []
        peakrates = np.zeros(all_nidx.shape[0])
        poprate_list = []
        nsp_edges = np.arange(-dt/2, t.max()+dt, dt)
        ker = np.ones(int(100/dt)) / 0.1  # sum over 100ms, divided by 100ms gives Hz
        for i in range(all_nidx.shape[0]):
            tspidx_r = SpikeDF.loc[SpikeDF['neuronid'] == all_nidx[i], 'tidxsp'].to_numpy()
            tsp_r = t[tspidx_r]
            tsp_r_bins, _ = np.histogram(tsp_r, bins=nsp_edges)
            rates = np.convolve(tsp_r_bins, ker, mode='same')
            poprate_list.append(rates)
            peakrates[i] = rates.max()
            for j in range(i+1, all_nidx.shape[0]):

                nidx1, nidx2 = all_nidx[i], all_nidx[j]
                x1, x2 = xxtun1d_ca3[nidx1], xxtun1d_ca3[nidx2]

                if x1 > x2:
                    nidx1, nidx2 = all_nidx[j], all_nidx[i]
                    x1, x2 = xxtun1d_ca3[nidx1], xxtun1d_ca3[nidx2]
                xdiff = x2-x1
                tidxsp1 = SpikeDF.loc[SpikeDF['neuronid'] == nidx1, 'tidxsp'].to_numpy()
                tidxsp2 = SpikeDF.loc[SpikeDF['neuronid'] == nidx2, 'tidxsp'].to_numpy()
                tsp1 = t[tidxsp1] * (1e-3)
                tsp2 = t[tidxsp2] * (1e-3)
                estThetaT, corrlag, corrinfo = TE.find_theta_isi_hilbert(tsp1, tsp2, theta_window=thetaT*2, default_Ttheta=thetaT)
                if np.isnan(corrlag):
                    continue

                (isibins, isiedges, signal_filt, _, alphas, _) = corrinfo
                if np.sum(isibins) < 10:
                    continue
                isiedgesm = (isiedges[:-1] + isiedges[1:])/2
                allxdiff.append(xdiff)
                allcorrlag.append(corrlag)

        allxdiff = np.array(allxdiff)
        allcorrlag = np.array(allcorrlag)
        allxdiff_fit = allxdiff[np.abs(allxdiff) < xdiffmax]
        allcorrlag_fit = allcorrlag[np.abs(allxdiff) < xdiffmax]
        ctmp = allxdiff_fit.max()
        regress = rcc(allxdiff_fit/ctmp, allcorrlag_fit, abound=(-1, 1))
        xdum = np.linspace(0, 1, 100)
        ydum = regress['phi0'] + 2*np.pi*regress['aopt']*xdum
        R, rho, p = regress['R'], regress['rho'], regress['p']
        ax_corr[dgid, mosdeg_id].scatter(allxdiff, allcorrlag, marker='.', s=1, color=dgcase_c[dgid])
        ax_corr[dgid, mosdeg_id].plot(xdum*ctmp, ydum, color='k', linewidth=0.75)
        ax_corr[dgid, mosdeg_id].plot(xdum*ctmp, ydum+2*np.pi, color='k', linewidth=0.75)
        ax_corr[dgid, mosdeg_id].plot(xdum*ctmp, ydum-2*np.pi, color='k', linewidth=0.75)
        ax_corr[dgid, mosdeg_id].annotate('rho=%0.2f'%(rho), xy=(0.05, 0.05), xycoords='axes fraction', fontsize=legendsize)



        ax_corr[dgid, mosdeg_id].set_ylim(-np.pi, np.pi)
        ax_corr[dgid, mosdeg_id].set_yticks(np.arange(-np.pi, np.pi+0.1, np.pi/2))
        ax_corr[dgid, mosdeg_id].set_yticks(np.arange(-np.pi, np.pi+0.1, np.pi/4), minor=True)
        ax_corr[dgid, mosdeg_id].set_yticklabels(['']*5)
        del simdata


for i in range(2):
    ax_corr[i, 0].set_ylabel('Lag (rad)')
    ax_corr[i, 0].set_yticklabels(['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])
    ax_corr[1, i].set_xlabel('Dist. diff (cm)')

fig.savefig(join(save_dir, 'fig4.png'), dpi=300)
fig.savefig(join(save_dir, 'fig4.eps'), dpi=300)
fig.savefig(join(save_dir, 'fig4.pdf'), dpi=300)