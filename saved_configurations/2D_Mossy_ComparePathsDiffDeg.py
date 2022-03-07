# Current addition: Interference of multiple mossy paths
# Parameters not yet tuned
from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import vonmises, pearsonr
from pycircstat import cdiff, mean as cmean
from library.comput_utils import cal_hd_np, get_nidx_np, pair_diff
from library.visualization import customlegend
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p

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

projl_MosCA3 = 4
side_projl_MosCA3 = projl_MosCA3/np.sqrt(2)

mos_startx0 = np.arange(-20, 21, 2)  # 0 degree
mos_starty0 = np.zeros(mos_startx0.shape[0])
mos_endx0, mos_endy0 = mos_startx0 + projl_MosCA3, mos_starty0

mos_startx45 = np.arange(-20, 21, 2)  # 45 degree
mos_starty45 = np.arange(-20, 21, 2)
mos_endx45, mos_endy45 = mos_startx45 + side_projl_MosCA3, mos_starty45 + side_projl_MosCA3

mos_starty90 = np.arange(-20, 21, 2)
mos_startx90 = np.zeros(mos_starty90.shape[0])  # 90 degree
mos_endx90, mos_endy90 = mos_startx90, mos_starty90 + projl_MosCA3

mos_startx135 = np.arange(20, -21, -2)  # 135 degree
mos_starty135 = np.arange(-20, 21, 2)
mos_endx135, mos_endy135 = mos_startx135 - side_projl_MosCA3, mos_starty135 + side_projl_MosCA3

mos_startx180 = np.arange(20, -21, -2)  # 180 degree
mos_starty180 = np.zeros(mos_startx180.shape[0])
mos_endx180, mos_endy180 = mos_startx180 - projl_MosCA3, mos_starty180


# mos_configs = (
#     (mos_startx0, mos_starty0, mos_endx0, mos_endy0, 0, 'x'),
#     (mos_startx45, mos_starty45, mos_endx45, mos_endy45, 45, 'y'),
#     (mos_startx90, mos_starty90, mos_endx90, mos_endy90, 90, 'y'),
#     (mos_startx135, mos_starty135, mos_endx135, mos_endy135, 135, 'x'),
#     (mos_startx180, mos_starty180, mos_endx180, mos_endy180, 180, 'x'),
# )

mos_configs = (
    (mos_startx0, mos_starty0, mos_endx0, mos_endy0, 0, 'x'),
    (mos_startx180, mos_starty180, mos_endx180, mos_endy180, 180, 'x'),
)

# Constant
Ipos_max = 2
Iangle_diff = 4
Ipos_sd = 5
izhi_a_ex = 0.02
izhi_b_ex = 0.2
izhi_c_ex = -55
izhi_d_ex = 4
# izhi_a_in = 0.1  # Fast spiking
# izhi_b_in = 0.2
izhi_a_in = 0.02  # LTS
izhi_b_in = 0.25
izhi_c_in = -65
izhi_d_in = 2
U_stdx = 0.7
# wmax_ca3ca3 = 0  # 120
wmax_mosmos = 100  # 100
wmax_ca3mos = 1500  # 600
wmax_mosca3 = 1500  # 600
wmax_Mosin = 100
wmax_inMos = 6
wmax_CA3in = 100
wmax_inCA3 = 6
wmax_inin = 0  # 0

# Paths
# save_dir = 'plots/IB_CheckI'
save_dir = 'plots/IBLTS_I%d-%d_Ustdx-%0.2f_WINCA3-%d-%d-%d_Wmos-%d-%d-%d_WINMOS-%d-%d'%(Ipos_max, Iangle_diff, U_stdx, wmax_CA3in, wmax_inCA3, wmax_inin, wmax_mosmos, wmax_ca3mos, wmax_mosca3, wmax_Mosin, wmax_inMos)
os.makedirs(save_dir, exist_ok=True)
# Parameter Scan
# Ipos_maxs = np.arange(0.5, 5.5, 0.5)
# for Ipos_max in Ipos_maxs:
#     Iangle_diff = Ipos_max * 2
for wmax_ca3ca3 in [400, 500, 600, 680]:

    for mos_startx2, mos_starty2, mos_endx2, mos_endy2, mosproj_deg2, sortby2 in mos_configs:
        mos_startx1, mos_starty1, mos_endx1, mos_endy1, mosproj_deg1, sortby1 = mos_configs[0]

        # pthtxt = 'I%0.2f-%0.2f.png'%(Ipos_max, Iangle_diff)
        pthtxt = 'deg-%d_WCA3CA3_%d.png'%(mosproj_deg2, wmax_ca3ca3)
        save_pth = join(save_dir, pthtxt)
        print(save_pth)
        if os.path.exists(save_pth):
            print('Exists. Skipped')
            continue

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
        # izhi_a, izhi_b = 0.02, 0.2  # CH
        # izhi_d, izhi_c = 2, -50
        V_ex, V_in = 0, -80
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
        # Ipos_max = 3
        # Iangle_diff = 6
        Iangle_kappa = 1
        # Ipos_sd = 5
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
        posvec_mos = np.stack([xxtun1d_mos, yytun1d_mos]).T
        posvec_CA3 = np.stack([xxtun1d_ca3, yytun1d_ca3]).T

        # # Weights
        wsd_global = 2
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


        # wmax_Mosin, wmax_inMos = 30, 20
        wprob_InCA3, wprob_InMos, wprob_InIn = 0.8, 0.8, 0.8
        w_ca3ca3 = gaufunc2d(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_ca3.reshape(nn_ca3, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_ca3.reshape(nn_ca3, 1), wsd_ca3ca3, wmax_ca3ca3)
        w_ca3mos = gaufunc2d(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_mos.reshape(nn_mos, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_mos.reshape(nn_mos, 1), wsd_ca3mos, wmax_ca3mos)
        w_mosmos = gaufunc2d(xxtun1d_mos.reshape(1, nn_mos), xxtun1d_mos.reshape(nn_mos, 1), yytun1d_mos.reshape(1, nn_mos), yytun1d_mos.reshape(nn_mos, 1), wsd_mosmos, wmax_mosmos)

        # Mos to CA3
        mos_startpos = np.stack([np.concatenate([mos_startx1, mos_startx2]), np.concatenate([mos_starty1, mos_starty2])]).T
        mos_endpos = np.stack([np.concatenate([mos_endx1, mos_endx2]), np.concatenate([mos_endy1, mos_endy2])]).T
        w_mosca3 = np.zeros((xxtun1d_ca3.shape[0], xxtun1d_mos.shape[0], mos_startpos.shape[0]))
        mos_act = np.zeros((posvec_mos.shape[0], mos_startpos.shape[0]))
        for i in range(mos_startpos.shape[0]):
            print('\rConstructing Weight matrix %d/%d'%(i, mos_startpos.shape[0]), flush=True, end='')
            w_mosca3[:, :, i], mos_act[:, i] = createMosProjMat_p2p(mos_startpos[i, :], mos_endpos[i, :], posvec_mos, posvec_CA3, wmax_mosca3, wsd_mosca3)
        print()
        w_mosca3 = np.max(w_mosca3, axis=2)
        mos_act = np.max(mos_act, axis=1).reshape(*xxtun2d_mos.shape) * wmax_mosca3

        # Inhibitory weights
        np.random.seed(0)
        w_CA3In = np.random.uniform(0, 1, size=(nn_inca3, nn_ca3)) * wmax_CA3in
        np.random.seed(1)
        w_InCA3 = np.random.uniform(0, 1, size=(nn_ca3, nn_inca3)) * wmax_inCA3
        np.random.seed(2)
        w_MosIn = np.random.uniform(0, 1, size=(nn_inmos, nn_mos)) * wmax_Mosin
        np.random.seed(3)
        w_InMos = np.random.uniform(0, 1, size=(nn_mos, nn_inmos)) * wmax_inMos
        np.random.seed(4)
        w_inin = np.random.uniform(0, 1, size=(nn_in, nn_in)) * wmax_inin


        # Assembling weights
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

        # # Plot Weight matrix
        # fig_config, ax_config = plt.subplots(1, 2, figsize=(12, 5))
        # im = ax_config[0].scatter(xxtun1d_ca3, yytun1d_ca3, c=aatun1d_ca3, cmap='hsv', vmax=np.pi, vmin=-np.pi, s=1)
        # ax_config[0].plot(traj_x, traj_y, c='k', linewidth=0.75)
        # im_w = ax_config[1].imshow(w)
        # plt.colorbar(im, ax=ax_config[0])
        # plt.colorbar(im_w, ax=ax_config[1])
        # fig_config.savefig(join(save_dir, 'tuning.png'), dpi=150)
        # # plt.show()

        # Synapse parameters
        tau_gex = 12
        tau_gin = 10
        # U_stdx = 0.350  # CH 0.350
        tau_stdx = 0.5e3  # 1

        # Initialization
        izhi_a = np.ones(nn) * izhi_a_ex
        izhi_b = np.ones(nn) * izhi_b_ex
        izhi_c = np.ones(nn) * izhi_c_ex
        izhi_d = np.ones(nn) * izhi_d_ex
        izhi_a[endidx_mos:endidx_in] = izhi_a_in
        izhi_b[endidx_mos:endidx_in] = izhi_b_in
        izhi_c[endidx_mos:endidx_in] = izhi_c_in
        izhi_d[endidx_mos:endidx_in] = izhi_d_in

        v = np.ones(nn) * izhi_c
        u = np.zeros(nn)
        Isyn = np.zeros(nn)
        Isyn_in = np.zeros(nn)
        Isyn_ex = np.zeros(nn)
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
        IsynIN_pop = np.zeros((t.shape[0], nn))
        IsynEX_pop = np.zeros((t.shape[0], nn))
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
            # Itheta_arr = np.ones(nn) * Itheta[i]
            # Itheta_arr[endidx_mos:] = 0
            # Itotal = Isyn + Isen_fac - Itheta_arr
            Itotal = Isyn + Isen_fac - Itheta[i]

            # Izhikevich
            v += (0.04*v**2 + 5*v + 140 - u + Itotal) * dt
            u += izhi_a * (izhi_b * v - u) * dt
            fidx = np.where(v > V_thresh)[0]
            v[fidx] = izhi_c[fidx]
            u[fidx] = u[fidx] + izhi_d[fidx]
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
            IsynIN_pop[i, :] = Isyn_in
            IsynEX_pop[i, :] = Isyn_ex
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
        fig, ax = plt.subplots(4, 5, figsize=(16, 10), facecolor='white', constrained_layout=True)

        gs = ax[0, 0].get_gridspec()
        for axeach in ax[0:2, 0:3].ravel():
            axeach.remove()
        axbig = fig.add_subplot(gs[0:2, 0:3])

        gs2 = ax[2, 0].get_gridspec()
        for axeach in ax[2:4, 0:3].ravel():
            axeach.remove()
        axbig2 = fig.add_subplot(gs2[2:4, 0:3])


        # # 1. Population raster CA3
        # Indices along the trajectory
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

        # Raster plot - CA3 pyramidal
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

        # Raster plot - CA3 inhibitory population
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

        axbig.annotate('Gray lines = Theta phase 0\nEC phase shift = %d deg'%(360-np.rad2deg(EC_phase)),
                       xy=(0.02, 0.80), xycoords='axes fraction', fontsize=12)
        axbig.tick_params(labelsize=legendsize)
        cbar = fig.colorbar(mappable, ax=axbig, ticks=[0, 1, 2], shrink=0.5)
        cbar.set_label('Syn Efficacy', rotation=90, fontsize=legendsize)

        # # 2. Raster - Second Mos paths
        # Indices along mossy projections
        mos_projtraj_x = np.linspace(mos_startx2[0], mos_startx2[-1], 100)
        mos_projtraj_y = np.linspace(mos_starty2[0], mos_starty2[-1], 100)
        all_nidx_proj = np.zeros(mos_projtraj_x.shape[0])
        for i in range(mos_projtraj_x.shape[0]):
            run_x, run_y = mos_projtraj_x[i], mos_projtraj_y[i]
            nidx = np.argmin(np.square(run_x - xxtun1d_ca3) + np.square(run_y - yytun1d_ca3))
            all_nidx_proj[i] = nidx
        all_nidx_proj = np.unique(all_nidx_proj).astype(int)

        # Raster plot - CA3 pyramical cells along the second mossy path
        if sortby2 == 'y':
            dwtun = ytun_ca3[1] - ytun_ca3[0]
            tt, traj_ww = np.meshgrid(t, yytun1d_ca3[all_nidx_proj])
        elif sortby2 == 'x':
            dwtun = xtun_ca3[1] - xtun_ca3[0]
            tt, traj_ww = np.meshgrid(t, xxtun1d_ca3[all_nidx_proj])
        mappable2 = axbig2.pcolormesh(tt, traj_ww, syneff_pop[:, all_nidx_proj].T, shading='auto', vmin=0, vmax=2, cmap='seismic')
        for neuronid in all_nidx_proj:
            tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
            tsp_neuron = t[tidxsp_neuron]
            if sortby2 == 'y':
                neuronw = yytun1d[neuronid]
            elif sortby2 == 'x':
                neuronw = xxtun1d[neuronid]
            axbig2.eventplot(tsp_neuron, lineoffsets=neuronw, linelengths=dwtun, linewidths=0.75, color='lime')
        axbig2.tick_params(labelsize=legendsize)
        cbar2 = fig.colorbar(mappable2, ax=axbig2, ticks=[0, 1, 2], shrink=0.5)
        cbar2.set_label('Syn Efficacy', rotation=90, fontsize=legendsize)
        for i in theta_cutidx:
            axbig2.axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)

        # # 3. 4. Phase precession for two example cells
        for axid, egnidx in enumerate(egnidxs):
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
            ax[axid, 3].scatter(xsp_norm_eg, phasesp_eg, marker='|', s=4, color=eg_cs[axid])
            ax[axid, 3].axhline(mean_phasesp, xmin=0, xmax=0.3, linewidth=1)
            ax[axid, 3].plot(xdum, ydum, linewidth=0.75, color=eg_cs[axid])
            ax[axid, 3].set_title('y= %0.2fx + %0.2f, atun=%0.2f'%(rcc_slope_rad, rcc_c, aatun1d[egnidx]), fontsize=legendsize+1)
            ax[axid, 3].set_ylim(0, 2*np.pi)
            ax[axid, 3].set_yticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])
            ax[axid, 3].set_yticklabels(['0', '$\pi/2$', '$\pi$', '$1.5\pi$', '$2\pi$'])

            ax_rate = ax[axid, 3].twinx().twiny()
            binsize = 20  # 20ms
            tmin, tmax = tsp_eg.min(), tsp_eg.max()
            count_bins, edges = np.histogram(tsp_eg, bins=np.arange(tmin, tmax+binsize, binsize))
            rate_bins = (count_bins / binsize * 1000)
            ax_rate.bar(edges[:-1], rate_bins, width=np.diff(edges), align='edge', facecolor='gray', alpha=0.5, zorder=0.1)
            ax_rate.set_ylim(0, 600)
            ax_rate.tick_params(labelsize=legendsize)

        # # 5. Currents, STD and STF for neuron example 1
        tidxsp_eg = SpikeDF.loc[SpikeDF['neuronid']==egnidxs[0], 'tidxsp'].to_numpy()
        tsp_eg, phasesp_eg = t[tidxsp_eg], theta_phase[tidxsp_eg]
        ax[2, 3].plot(t, Isyn_pop[:, egnidxs[0]], label='Isyn', linewidth=0.75, color='orange')
        ax[2, 3].plot(t, Isen_pop[:, egnidxs[0]], label='Isen', linewidth=0.75, color='blue')
        ax[2, 3].plot(t, Isen_fac_pop[:, egnidxs[0]], label='Isen_fac', linewidth=0.75, color='cyan')
        ax[2, 3].set_ylim(-1, 40)
        customlegend(ax[2, 3], fontsize=legendsize, loc='upper left')
        axsyneff = ax[2, 3].twinx()
        axsyneff.plot(t, syneff_pop[:, egnidxs[0]], label='snyeff', color='r', linewidth=0.75)
        axsyneff.plot(t, ECstfx_pop[:, egnidxs[0]], label='ECSTF', color='green', linewidth=0.75)
        axsyneff.set_ylim(-0.1, 2.1)
        axsyneff.eventplot(tsp_eg, lineoffsets = 1.8, linelength=0.1, linewidths=0.5, color='r')
        customlegend(axsyneff, fontsize=legendsize)
        theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
        for i in theta_cutidx:
            ax[2, 3].axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)
        axsyneff.tick_params(labelsize=legendsize)

        # # 6. Input synaptic currents into neuron example 1
        egaxidx = 0
        egidx_in = egnidxs[0]
        inzoom_egtimes = (5000, 6000)
        tidxstart, tidxend = inzoom_egtimes  # in index = 0.1ms
        tslice, IsynINslice = t[tidxstart:tidxend], IsynIN_pop[tidxstart:tidxend, egidx_in]
        Isynslice = Isyn_pop[tidxstart:tidxend, egidx_in]
        IsynEXslice = IsynEX_pop[tidxstart:tidxend, egidx_in]
        tslice_offset = tslice-tslice.min()
        ax[3, 3].plot(tslice_offset, Isynslice, color='gray', linewidth=0.5, alpha=0.7)
        ax[3, 3].plot(tslice_offset, IsynEXslice, color='b', linewidth=0.5, alpha=0.7)
        ax[3, 3].plot(tslice_offset, IsynINslice, color=eg_cs[egaxidx], linewidth=0.5, alpha=0.7)
        ax[3, 3].set_xticks(np.arange(0, 101, 10))
        ax[3, 3].set_xticks(np.arange(0, 101, 5), minor=True)

        # # 7. Raster plot for inhibitory neurons - Zoomed in
        tsp_rasIn_list = []
        for neuronid in np.arange(nn_in_startidx, nn_in_startidx+nn_in, 1):
            tidxsp_neuron = SpikeDF[(SpikeDF['neuronid'] == neuronid) & (SpikeDF['tidxsp'] <= inzoom_egtimes[1]) & (SpikeDF['tidxsp'] > inzoom_egtimes[0])]['tidxsp']
            if tidxsp_neuron.shape[0] < 1:
                continue
            tsp_neuron = t[tidxsp_neuron]
            tsp_rasIn_list.append(tsp_neuron)
            ax[0, 4].eventplot(tsp_neuron, lineoffsets=neuronid-nn_in_startidx, linelengths=2, linewidths=2, color='m')
        ax[0, 4].set_xlim(inzoom_egtimes[0]/10, inzoom_egtimes[1]/10)
        if len(tsp_rasIn_list) > 0:
            tsp_rasIn = np.concatenate(tsp_rasIn_list)
            binsize = 2
            count_bins, edges = np.histogram(tsp_rasIn, bins=np.arange(inzoom_egtimes[0]/10, inzoom_egtimes[1]/10+binsize, binsize))
            rate_bins = count_bins / len(tsp_rasIn_list) / binsize * 1000
            ax_inrate = ax[0, 4].twinx()
            ax_inrate.bar(edges[:-1], rate_bins, width=np.diff(edges), align='edge', facecolor='gray', alpha=0.5, zorder=0.1)
            ax_inrate.tick_params(labelsize=legendsize)
            ax_inrate.set_ylim(0, 600)

        # # 8. ISI in frequency
        isi_dict = dict(ca3=[], mos=[], inca3=[], inmos=[])
        isi_c = dict(ca3='r', mos='g', inca3='purple', inmos='teal')
        for neuronid in np.arange(nn):
            tidxsp_tmp = SpikeDF[(SpikeDF['neuronid'] == neuronid)]['tidxsp']
            if tidxsp_tmp.shape[0] < 2:
                continue
            tsp_tmp = t[tidxsp_tmp]
            if neuronid < endidx_ca3:  # < and >=
                neuron_type = 'ca3'
            elif (neuronid >= endidx_ca3) and (neuronid < endidx_mos):
                neuron_type = 'mos'
            elif (neuronid >= endidx_mos) and (neuronid < endidx_inca3):
                neuron_type = 'inca3'
            elif (neuronid >= endidx_inca3) and (neuronid < endidx_in):
                neuron_type = 'inmos'
            else:
                raise RuntimeError
            isi_dict[neuron_type].append(np.diff(tsp_tmp))
        isi_edges = np.arange(15, 655, 10)
        for neuron_type in isi_dict.keys():
            isi_list = isi_dict[neuron_type]
            if len(isi_list) < 1:
                continue
            isi_eachtype = 1000/np.concatenate(isi_list)
            ax[1, 4].hist(isi_eachtype, bins=isi_edges, density=True, histtype='step', color=isi_c[neuron_type],
                        label=neuron_type, linewidth=0.75)
        customlegend(ax[1, 4], fontsize=legendsize)
        ax[1, 4].tick_params(axis='both', labelsize=legendsize)


        # # 9. Correlograms
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
            bins, _, _ = ax[2, 4].hist(tsp_diff, bins=edges, histtype='step', color=egc, label=legend_txt)
            ax[2, 4].set_title('Spike correlation', fontsize=legendsize)
        customlegend(ax[2, 4], fontsize=legendsize)


        # # Save
        for ax_each in ax.ravel():
            ax_each.tick_params(labelsize=legendsize)
        for ax_each in [axbig2]:
            ax_each.tick_params(labelsize=legendsize)
        fig.savefig(save_pth, dpi=150)
        plt.close()


