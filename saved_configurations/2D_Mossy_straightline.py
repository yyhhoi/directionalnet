# EC-STF, with CH
# 2D environment. Experiment for box-input, mossy in "Same" and "Opposite" direction, and
# independent inhibition loop in mossy layer
# Results: Both-worst direction is more intrinsic than Both-best
# Changing I3-6 to I4-8 will make the above intrinsic trend more apparent

from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from scipy.stats import vonmises, pearsonr
from pycircstat import cdiff, mean as cmean
from library.comput_utils import cal_hd_np, get_nidx_np, pair_diff
from library.visualization import customlegend
from library.linear_circular_r import rcc

mpl.rcParams['figure.dpi'] = 150
legendsize = 7


def gaufunc2d(x, mux, y, muy, sd, outmax):
    return outmax * np.exp(-(np.square(x - mux) + np.square(y - muy))/(2*sd**2) )

def circgaufunc(x, loc, kappa, outmax):
    return outmax * np.exp(kappa * (np.cos(x - loc) - 1))

def boxfunc2d(x, mux, y, muy, sd, outmax):

    dist = np.sqrt( np.square(x-mux) + np.square(y-muy))
    out = np.ones(dist.shape) * outmax
    out[dist > sd] = 0
    return out

# Constant
izhi_d = 2
Ipos_max = 3
Iangle_diff = 6
U_stdx = 0.35  # 0.25
# mos_xshift = 0
# mos_yshift = 0
wmax_ca3ca3 = 140  # 250 for gau, 140 for box
wmax_mosmos = 20  # 20
wmax_ca3mos = 500  # 500
wmax_mosca3 = 500  # 500
# wmax_Mosin = 50
# wmax_inMos = 20

wsd_global = 2
save_dir = 'plots/BoxEC_newGraph_Mos_I%d-%d_WCA3-%d_Wmos-%d-%d-%d'%(Ipos_max, Iangle_diff, wmax_ca3ca3, wmax_mosmos, wmax_ca3mos, wmax_mosca3)

os.makedirs(save_dir, exist_ok=True)
# Parameter Scan
mos_shifts = ((4, 0, "Same"), (-4, 0, 'Opp'))
wmax_biMosIns = [0, 60, 140]

for wmax_biMosIn in wmax_biMosIns:
    wmax_inMos = wmax_biMosIn
    wmax_Mosin = wmax_biMosIn
    mos_corr_dict = dict()
    for mos_xshift, mos_yshift, mos_label in mos_shifts:
        save_pth = join(save_dir, 'BoxEC_MosX%dY%d_WMosIn-%d_WinMos-%d.png'%(mos_xshift, mos_yshift, wmax_Mosin, wmax_inMos))
        print(save_pth)
        # if os.path.exists(save_pth):
        #     print('Exists. Skipped')
        #     continue

        # Environment & agent
        dt = 0.1 # 0.1ms
        running_speed = 20  # cm/s
        arena_xmin, arena_xmax = -40, 40  # in cm
        arena_ymin, arena_ymax = -40, 40  # in cm
        t = np.arange(0, 2e3, dt)
        traj_x = np.linspace(-20, 20, t.shape[0])
        traj_y = np.zeros(traj_x.shape[0])
        traj_a = cal_hd_np(traj_x, traj_y)


        # Izhikevich's model parameters
        izhi_a, izhi_b, izhi_c = 0.02, 0.2, -50  # CH
        # izhi_d = 2

        V_ex, V_in = 60, -80
        V_thresh = 30
        spdelay = int(2/dt)

        # Theta inhibition
        theta_amp = 7
        theta_f = 10
        theta_T = 1/theta_f * 1e3
        theta_phase = np.mod(t, theta_T)/theta_T * 2*np.pi
        theta_phase_plot = np.mod(theta_phase + 2*np.pi, 2*np.pi)
        Itheta = (1 + np.cos(theta_phase))/2 * theta_amp

        # Positional drive
        EC_phase = np.deg2rad(290)
        # Ipos_max = 6  # CH: 15,  IB: 20
        # Iangle_diff = 6  # CH: 25, IB: 25
        Iangle_kappa = 1
        Ipos_sd = 5
        ECstf_rest, ECstf_target = 0, 2
        tau_ECstf = 0.5e3
        U_ECstf = 0.001  # 0.001
        Ipos_max_compen = Ipos_max + (np.cos(EC_phase) + 1)/2 * theta_amp

        # Sensory tuning
        xmin, xmax, nx_ca3, nx_mos = -40, 40, 80, 30
        ymin, ymax, ny_ca3, ny_mos = -40, 40, 80, 30
        nn_inca3, nn_inmos = 250, 250
        nn_in = nn_inca3 + nn_inmos
        xtun_ca3 = np.linspace(xmin, xmax, nx_ca3)
        ytun_ca3 = np.linspace(ymin, ymax, ny_ca3)
        atun_seeds = np.array([0, np.pi/2, np.pi, np.pi*3/2])
        xxtun2d_ca3, yytun2d_ca3 = np.meshgrid(xtun_ca3, ytun_ca3)
        aatun2d_ca3 = np.zeros(xxtun2d_ca3.shape)
        seed_i = 0
        for i in np.arange(0, nx_ca3, 2):
            for j in np.arange(0, ny_ca3, 2):
                np.random.seed(seed_i)
                rand_shift = np.random.uniform(0, 2*np.pi)
                perm_atun_seeds = atun_seeds + rand_shift
                aatun2d_ca3[i, j:j+2] = perm_atun_seeds[0:2]
                aatun2d_ca3[i+1, j:j+2] = perm_atun_seeds[2:4]
                seed_i += 1
        aatun2d_ca3 = np.mod(aatun2d_ca3, 2*np.pi) - np.pi  # range = (-pi, pi]
        xxtun1d_ca3, yytun1d_ca3, aatun1d_ca3 = xxtun2d_ca3.flatten(), yytun2d_ca3.flatten(), aatun2d_ca3.flatten()

        xtun_mos = np.linspace(xmin, xmax, nx_mos)
        ytun_mos = np.linspace(ymin, ymax, ny_mos)
        xxtun2d_mos, yytun2d_mos = np.meshgrid(xtun_mos, ytun_mos)
        xxtun1d_mos, yytun1d_mos = xxtun2d_mos.flatten(), yytun2d_mos.flatten()
        aatun1d_mos = np.zeros(xxtun1d_mos.shape)  # Dummy tunning

        xxtun1d_in, yytun1d_in, aatun1d_in = np.zeros(nn_in), np.zeros(nn_in), np.zeros(nn_in)  # Inhibitory neurons no tuning

        nn_ca3, nn_mos = xxtun1d_ca3.shape[0], xxtun1d_mos.shape[0]
        xxtun1d = np.concatenate([xxtun1d_ca3, xxtun1d_mos, xxtun1d_in])
        yytun1d = np.concatenate([yytun1d_ca3, yytun1d_mos, yytun1d_in])
        aatun1d = np.concatenate([aatun1d_ca3, aatun1d_mos, aatun1d_in])
        endidx_ca3, endidx_mos, endidx_in = nn_ca3, nn_ca3 + nn_mos, nn_ca3 + nn_mos + nn_in
        endidx_inca3 = endidx_mos + nn_inca3
        nn = xxtun1d.shape[0]


        # # Weights
        # mos_xshift = 4  # 4
        # mos_yshift = 0
        # wmax_ca3ca3 = 3
        wsd_ca3ca3 = wsd_global
        # wmax_ca3mos = 0
        wsd_ca3mos = wsd_global
        # wmax_mosca3 = 0
        wsd_mosca3 = wsd_global
        # wmax_mosmos = 0
        wsd_mosmos = wsd_global
        wmax_inin = 0
        wmax_CA3in, wmax_inCA3 = 30, 20
        # wmax_Mosin, wmax_inMos = 30, 20
        wprob_InCA3, wprob_InMos = 0.8, 0.8
        w_ca3ca3 = gaufunc2d(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_ca3.reshape(nn_ca3, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_ca3.reshape(nn_ca3, 1), wsd_ca3ca3, wmax_ca3ca3)
        w_ca3mos = gaufunc2d(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_mos.reshape(nn_mos, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_mos.reshape(nn_mos, 1), wsd_ca3mos, wmax_ca3mos)
        w_mosca3 = gaufunc2d(xxtun1d_mos.reshape(1, nn_mos), xxtun1d_ca3.reshape(nn_ca3, 1) - mos_xshift, yytun1d_mos.reshape(1, nn_mos), yytun1d_ca3.reshape(nn_ca3, 1) - mos_yshift, wsd_mosca3, wmax_mosca3)
        w_mosmos = gaufunc2d(xxtun1d_mos.reshape(1, nn_mos), xxtun1d_mos.reshape(nn_mos, 1), yytun1d_mos.reshape(1, nn_mos), yytun1d_mos.reshape(nn_mos, 1), wsd_mosmos, wmax_mosmos)

        np.random.seed(0)
        w_CA3In = (np.random.uniform(0, 1, size=(nn_inca3, nn_ca3)) < wprob_InCA3) * wmax_CA3in
        np.random.seed(1)
        w_InCA3 = (np.random.uniform(0, 1, size=(nn_ca3, nn_inca3)) < wprob_InCA3) * wmax_inCA3
        np.random.seed(2)
        w_MosIn = (np.random.uniform(0, 1, size=(nn_inmos, nn_mos)) < wprob_InMos) * wmax_Mosin
        np.random.seed(3)
        w_InMos = (np.random.uniform(0, 1, size=(nn_mos, nn_inmos)) < wprob_InMos) * wmax_inMos
        w_inin = np.ones((nn_in, nn_in)) * wmax_inin

        w = np.zeros((nn, nn))
        w[0:nn_ca3, 0:nn_ca3] = w_ca3ca3
        w[nn_ca3:endidx_mos, 0:nn_ca3] = w_ca3mos
        w[0:nn_ca3, nn_ca3:endidx_mos] = w_mosca3
        w[nn_ca3:endidx_mos, nn_ca3:endidx_mos] = w_mosmos
        w[endidx_mos:endidx_inca3, 0:endidx_ca3] = w_CA3In
        w[0:endidx_ca3, endidx_mos:endidx_inca3] = w_InCA3
        w[endidx_inca3:endidx_in, endidx_ca3:endidx_mos] = w_MosIn
        w[endidx_ca3:endidx_mos, endidx_inca3:endidx_in] = w_InMos
        w[endidx_mos:endidx_in, endidx_mos:endidx_in] = w_inin

        # # Plot Configuration
        # fig_config, ax_config = plt.subplots(1, 2, figsize=(12, 5))
        # im = ax_config[0].scatter(xxtun1d_ca3, yytun1d_ca3, c=aatun1d_ca3, cmap='hsv', vmax=np.pi, vmin=-np.pi, s=1)
        # ax_config[0].plot(traj_x, traj_y, c='k', linewidth=0.75)
        # im_w = ax_config[1].imshow(w)
        # plt.colorbar(im, ax=ax_config[0])
        # plt.colorbar(im_w, ax=ax_config[1])
        # fig_config.savefig(join(save_dir, 'tuning.png'), dpi=150)
        # # plt.show()


        # Synapse parameters
        tau_gex = 10
        tau_gin = 10
        # U_stdx = 0.450  # CH 0.450
        tau_stdx = 0.5e3  # 1

        # Initialization
        v = np.ones(nn) * izhi_c
        u = np.zeros(nn)
        Isyn = np.zeros(nn)
        gex = np.zeros(nn)
        gin = np.zeros(nn)
        stdx = np.ones(nn)
        ECstfx = np.ones(nn) * ECstf_rest
        fidx_buffer = []
        SpikeDF_dict = dict(neuronid=[], tidxsp=[])
        v_pop = np.zeros((t.shape[0], nn))
        Isen_pop = np.zeros((t.shape[0], nn))
        Isen_fac_pop = np.zeros((t.shape[0], nn))
        Isyn_pop = np.zeros((t.shape[0], nn))
        Itotal_pop = np.zeros((t.shape[0], nn))
        syneff_pop = np.zeros((t.shape[0], nn))
        ECstfx_pop = np.zeros((t.shape[0], nn))

        # # Simulation runtime
        numt = t.shape[0]
        t1 = time.time()

        for i in range(numt):
            print('\rSimulation %d/%d'%(i, numt), flush=True, end='')
            # Behavioural
            run_x, run_y, run_a = traj_x[i], traj_y[i], traj_a[i]

            # Sensory input
            Iangle = circgaufunc(run_a, aatun1d, Iangle_kappa, Iangle_diff)
            ECtheta = (np.cos(theta_phase[i] + EC_phase) + 1)/2
            Isen = boxfunc2d(run_x, xxtun1d, run_y, yytun1d, Ipos_sd, Ipos_max_compen+Iangle) * ECtheta
            Isen[nn_ca3:] = 0
            ECstfx += ((ECstf_rest-ECstfx)/tau_ECstf + (ECstf_target - ECstfx) * U_ECstf * Isen) * dt
            Isen_fac = np.square(ECstfx) * Isen

            # Total Input
            Itotal = Isyn + Isen_fac - Itheta[i]

            # Izhikevich
            v += (0.04*v**2 + 5*v + 140 - u + Itotal) * dt
            u += izhi_a * (izhi_b * v - u) * dt
            fidx = np.where(v > V_thresh)[0]
            v[fidx] = izhi_c
            u[fidx] = u[fidx] + izhi_d
            fidx_buffer.append(fidx)

            # STD & STF
            d_stdx_dt = (1 - stdx)/tau_stdx
            d_stdx_dt[fidx] = d_stdx_dt[fidx] - U_stdx * stdx[fidx]
            d_stdx_dt[nn_ca3:] = 0
            stdx += d_stdx_dt * dt
            syneff = stdx


            if i > spdelay:  # 2ms delay
                delayed_fidx = fidx_buffer.pop(0)
                delayed_fidx_ex = delayed_fidx[delayed_fidx < endidx_mos]
                delayed_fidx_in = delayed_fidx[delayed_fidx >= endidx_mos]

                # Synaptic input (Excitatory)
                spike_sum = np.sum(syneff[delayed_fidx_ex].reshape(1, -1) * w[:, delayed_fidx_ex], axis=1) / endidx_mos
                gex += (-gex/tau_gex + spike_sum) * dt
                Isyn_ex = gex * (V_ex - v)

                # Synaptic input (Inhibitory)
                spike_sum = np.sum(w[:, delayed_fidx_in], axis=1) / nn_in
                gin += (-gin/tau_gin + spike_sum) * dt
                Isyn_in = gin * (V_in - v)
                Isyn = Isyn_ex + Isyn_in

            # Store data
            SpikeDF_dict['neuronid'].extend(list(fidx))
            SpikeDF_dict['tidxsp'].extend([i] * len(fidx))

            v_pop[i, :] = v
            Isen_pop[i, :] = Isen
            Isen_fac_pop[i, :] = Isen_fac
            Isyn_pop[i, :] = Isyn
            Itotal_pop[i, :] = Itotal
            syneff_pop[i, :] = syneff
            ECstfx_pop[i, :] = ECstfx

        print('\nSimulation time = %0.2fs'%(time.time()-t1))

        # # Storage
        SpikeDF = pd.DataFrame(SpikeDF_dict)
        SpikeDF['neuronx'] = SpikeDF['neuronid'].apply(lambda x : xxtun1d[x])

        NeuronDF = pd.DataFrame(dict(neuronid=np.arange(nn), neuronx=xxtun1d, neurony=yytun1d, neurona=aatun1d,
                                     neurontype=["CA3"]*nn_ca3 + ['Mos']*nn_mos + ['In']*nn_in))

        BehDF = pd.DataFrame(dict(t=t, x=traj_x, y=traj_y, a=traj_a, Itheta=Itheta, theta_phase=theta_phase,
                                  theta_phase_plot=theta_phase_plot))

        ActivityData = dict(v=v_pop, Isen=Isen_pop, Isyn=Isyn_pop, Isen_fac=Isen_fac_pop,
                            Itotal=Itotal_pop, syneff=syneff_pop, ECstf=ECstfx_pop)

        MetaData = dict(nn=nn, nn_ca3=nn_ca3, nn_mos=nn_mos, nn_in=nn_in, w=w, EC_phase=EC_phase)

        # # Plot Analysis
        fig, ax = plt.subplots(3, 5, figsize=(14, 6), facecolor='white', constrained_layout=True)
        gs = ax[0, 0].get_gridspec()
        for axeach in ax[:, 0:3].ravel():
            axeach.remove()
        axbig = fig.add_subplot(gs[:, 0:3])


        all_nidx = np.zeros(traj_x.shape[0])
        for i in range(traj_x.shape[0]):
            run_x, run_y = traj_x[i], traj_y[i]
            nidx = np.argmin(np.square(run_x - xxtun1d_ca3) + np.square(run_y - yytun1d_ca3))
            all_nidx[i] = nidx
        all_nidx = np.unique(all_nidx).astype(int)
        egnidxs = all_nidx[[13, 19]]
        eg_cs = ['r', 'darkgreen']
        bestpair_nidxs = all_nidx[[13, 13+8]]
        worstpair_nidxs = all_nidx[[19, 19+8]]
        bestpair_c, worstpair_c = 'm', 'gold'


        # Population raster
        dxtun = xxtun1d[1] - xxtun1d[0]
        tt, traj_xx = np.meshgrid(t, xxtun1d[all_nidx])
        mappable = axbig.pcolormesh(tt, traj_xx, syneff_pop[:, all_nidx].T, shading='auto', vmin=0, vmax=2, cmap='seismic')
        for neuronid in all_nidx:
            tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
            tsp_neuron = t[tidxsp_neuron]
            neuronx = xxtun1d[neuronid]
            if neuronid == egnidxs[0]:
                ras_c = eg_cs[0]
            elif neuronid == egnidxs[1]:
                ras_c = eg_cs[1]
            elif neuronid == bestpair_nidxs[1]:
                ras_c = bestpair_c
            elif neuronid == worstpair_nidxs[1]:
                ras_c = worstpair_c
            else:
                ras_c = 'lime'
            axbig.eventplot(tsp_neuron, lineoffsets=neuronx, linelengths=dxtun, linewidths=0.75, color=ras_c)

        # Population raster - Inhibitory
        nn_in_startidx = nn_ca3+nn_mos
        neuron_in_xmin, neuron_in_xmax = traj_x.max(), traj_x.max()*1.1
        dx_in = (neuron_in_xmax - neuron_in_xmin)/(nn_in)
        for neuronid in np.arange(nn_in_startidx, nn_in_startidx+nn_in, 1):
            tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
            tsp_neuron = t[tidxsp_neuron]
            plot_onset = neuron_in_xmin+(neuronid-nn_in_startidx) * dx_in
            axbig.eventplot(tsp_neuron, lineoffsets=plot_onset, linelengths=dx_in, linewidths=0.75, color='m')
        axbig.plot(t, traj_x, c='k', linewidth=0.75)
        theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
        for i in theta_cutidx:
            axbig.axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)

        axbig.annotate('Mossy vec (%d, %d)\nGray lines = Theta phase 0\nEC phase shift = %d deg'%(mos_xshift, mos_yshift, 360-np.rad2deg(EC_phase)),
                       xy=(0.02, 0.75), xycoords='axes fraction', fontsize=12)
        axbig.tick_params(labelsize=legendsize)
        cbar = fig.colorbar(mappable, ax=axbig, ticks=[0, 1, 2], shrink=0.5)
        cbar.set_label('Syn Efficacy', rotation=90, fontsize=legendsize)

        for axid, egnidx in enumerate(egnidxs):

            # Phase precession
            tidxsp_eg = SpikeDF.loc[SpikeDF['neuronid']==egnidx, 'tidxsp'].to_numpy()
            tsp_eg, phasesp_eg = t[tidxsp_eg], theta_phase[tidxsp_eg]
            xsp_eg = traj_x[tidxsp_eg]
            mean_phasesp = cmean(phasesp_eg)
            xspmin, xsprange = xsp_eg.min(), xsp_eg.max() - xsp_eg.min()
            xsp_norm_eg = (xsp_eg-xspmin)/xsprange
            regress = rcc(xsp_norm_eg, phasesp_eg, abound=(-1., 1.))
            rcc_c, rcc_slope_rad = regress['phi0'], regress['aopt'] * 2 * np.pi
            xdum = np.linspace(xsp_norm_eg.min(), xsp_norm_eg.max(), 100)
            ydum = xdum * rcc_slope_rad + rcc_c
            ax[0, 3+axid].scatter(xsp_norm_eg, phasesp_eg, marker='|', s=4, color=eg_cs[axid])
            ax[0, 3+axid].axhline(mean_phasesp, xmin=0, xmax=0.3, linewidth=1)
            ax[0, 3+axid].plot(xdum, ydum, linewidth=0.75, color=eg_cs[axid])
            # ax[0, 3+axid].annotate('y= %0.2fx + %0.2f'%(rcc_slope_rad, rcc_c), xy=(0.25, 0.85), xycoords='axes fraction', fontsize=legendsize+2)
            ax[0, 3+axid].set_title('y= %0.2fx + %0.2f, atun=%0.2f'%(rcc_slope_rad, rcc_c, aatun1d[egnidx]), fontsize=legendsize+1)
            ax[0, 3+axid].set_ylim(0, 2*np.pi)
            ax[0, 3+axid].set_yticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])
            ax[0, 3+axid].set_yticklabels(['0', '$\pi/2$', '$\pi$', '$1.5\pi$', '$2\pi$'])

            # Correlograms
            if axid == 0:
                besttidxsp_eg1 = SpikeDF.loc[SpikeDF['neuronid']==bestpair_nidxs[0], 'tidxsp'].to_numpy()
                besttsp_eg1 = t[besttidxsp_eg1]
                besttidxsp_eg2 = SpikeDF.loc[SpikeDF['neuronid']==bestpair_nidxs[1], 'tidxsp'].to_numpy()
                besttsp_eg2 = t[besttidxsp_eg2]

                worsttidxsp_eg1 = SpikeDF.loc[SpikeDF['neuronid']==worstpair_nidxs[0], 'tidxsp'].to_numpy()
                worsttsp_eg1 = t[worsttidxsp_eg1]
                worsttidxsp_eg2 = SpikeDF.loc[SpikeDF['neuronid']==worstpair_nidxs[1], 'tidxsp'].to_numpy()
                worsttsp_eg2 = t[worsttidxsp_eg2]

                edges = np.arange(-100, 100, 5)
                for tsp_eg1, tsp_eg2, egc in ((besttsp_eg1, besttsp_eg2, bestpair_c), (worsttsp_eg1, worsttsp_eg2, worstpair_c)):
                    tsp_diff = pair_diff(tsp_eg1, tsp_eg2).flatten()
                    tsp_diff = tsp_diff[np.abs(tsp_diff) < 100]
                    legend_txt = 'Best' if egc=='m' else 'Worst'
                    bins, _, _ = ax[1, 3+axid].hist(tsp_diff, bins=edges, histtype='step', color=egc, label=legend_txt)
                    ax[1, 3+axid].set_title('Spike correlation', fontsize=legendsize)
                    mos_corr_dict['%s%s'%(mos_label, legend_txt)] = bins
                customlegend(ax[1, 3+axid], fontsize=legendsize)

            else:
                # ax[1, 3+axid].axis('off')
                for egnidxtmp, egaxidx, egtimes in zip(egnidxs, [0, 1], [(5000, 6000), (8000, 9000)]):
                    tidxstart, tidxend = egtimes  # in index = 0.1ms
                    tslice, Isynslice = t[tidxstart:tidxend], Isyn_pop[tidxstart:tidxend, egnidxtmp]

                    ax[1, 3+axid].plot(tslice-tslice.min(), Isynslice, color=eg_cs[egaxidx], linewidth=0.5)
                    mask_tmp = (SpikeDF['neuronid']==egnidxtmp) & (SpikeDF['tidxsp']>tidxstart) & (SpikeDF['tidxsp']<=tidxend)
                    tidxsp_eg_tmp = SpikeDF.loc[mask_tmp, 'tidxsp'].to_numpy()
                    tsp_offset = t[tidxsp_eg_tmp]-tslice.min()
                    aver_freq = tsp_offset.shape[0] / (tsp_offset.max() - tsp_offset.min()) * (1e3)

                    ax[1, 3+axid].eventplot(tsp_offset, lineoffsets=Isynslice.max()*1.1, linelengths=1, linewidths=0.5, color=eg_cs[egaxidx], label='freq=%0.2fHz'%(aver_freq))
                    ax[1, 3+axid].set_xticks(np.arange(0, 101, 10))
                    ax[1, 3+axid].set_xticks(np.arange(0, 101, 5), minor=True)
                    ax[1, 3+axid].set_ylim(0, Isynslice.max()*1.3)
                    customlegend(ax[1, 3+axid], fontsize=legendsize)






                # nidx_mostmp = np.argmin(np.square(-25 - xxtun1d_mos) + np.square(25 - yytun1d_mos))
                # nidx_mos = endidx_ca3 + nidx_mostmp
                # w_mosplot = w[:endidx_ca3, nidx_mos].reshape(*xxtun2d_ca3.shape)
                # ax[1, 3+axid].pcolormesh(xxtun2d_ca3, yytun2d_ca3, w_mosplot, shading='auto', cmap='Greens')
                # ax[1, 3+axid].scatter(xxtun1d[nidx_mos], yytun1d[nidx_mos], marker='x', color='red', label='Presynaptic Mos example')
                #
                # ax[1, 3+axid].arrow(traj_x[0], traj_y[0], traj_x.max() - traj_x.min(), traj_y.max() - traj_y.min(), color='k', label='Trajectory', head_width=2)
                # ax[1, 3+axid].arrow(0, 10, mos_xshift, mos_yshift, color='green', label='Mossy direction', head_width=2)
                # ax[1, 3+axid].set_xlim(xmin, xmax)
                # ax[1, 3+axid].set_ylim(ymin, ymax)
                # customlegend(ax[1, 3+axid], fontsize=legendsize)
                # ax[1, 3+axid].set_title('Mos postsyn weight heatmap\n(same direction at all (x,y) locations)', fontsize=legendsize)



            # Currents and STD
            ax[2, 3+axid].plot(t, Isyn_pop[:, egnidx], label='Isyn', linewidth=0.75, color='orange')
            ax[2, 3+axid].plot(t, Isen_pop[:, egnidx], label='Isen', linewidth=0.75, color='blue')
            ax[2, 3+axid].plot(t, Isen_fac_pop[:, egnidx], label='Isen_fac', linewidth=0.75, color='cyan')
            ax[2, 3+axid].set_ylim(-1, 40)
            customlegend(ax[2, 3+axid], fontsize=legendsize, loc='upper left')
            axsyneff = ax[2, 3+axid].twinx()
            axsyneff.plot(t, syneff_pop[:, egnidx], label='snyeff', color='r', linewidth=0.75)
            axsyneff.plot(t, ECstfx_pop[:, egnidx], label='ECSTF', color='green', linewidth=0.75)
            axsyneff.set_ylim(-0.1, 2.1)
            axsyneff.eventplot(tsp_eg, lineoffsets = 1.8, linelength=0.1, linewidths=0.5, color='r')
            customlegend(axsyneff, fontsize=legendsize)
            theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
            for i in theta_cutidx:
                ax[2, 3+axid].axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)

            axsyneff.tick_params(labelsize=legendsize)

        for ax_each in ax.ravel():
            ax_each.tick_params(labelsize=legendsize)


        if mos_label == 'Opp':
            samebest_bins, oppbest_bins = mos_corr_dict['SameBest'], mos_corr_dict['OppBest']
            sameworst_bins, oppworst_bins = mos_corr_dict['SameWorst'], mos_corr_dict['OppWorst']

            best_ex_tmp, _ = pearsonr(samebest_bins, oppbest_bins)
            worst_ex_tmp, _ = pearsonr(sameworst_bins, oppworst_bins)
            best_in_tmp, _ = pearsonr(samebest_bins, oppbest_bins[::-1])
            worst_in_tmp, _ = pearsonr(sameworst_bins, oppworst_bins[::-1])
            best_ex = (best_ex_tmp + 1)/2
            worst_ex = (worst_ex_tmp + 1)/2
            best_in = (best_in_tmp + 1)/2
            worst_in = (worst_in_tmp + 1)/2
            ax[1, 3].annotate('Best Ex(%0.2f)-In(%0.2f) = %0.2f\nWorst Ex(%0.2f)-In(%0.2f) = %0.2f'%(best_ex, best_in, best_ex - best_in, worst_ex, worst_in, worst_ex - worst_in),
                              xy=(0.4, 0.7), xycoords='axes fraction', fontsize=legendsize-1)
        fig.savefig(save_pth, dpi=150)
        plt.close()


