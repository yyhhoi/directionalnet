# Different STD across different post-synaptic neurons

from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from scipy.stats import vonmises, pearsonr, circmean
from pycircstat import cdiff, mean as cmean
from library.comput_utils import cal_hd_np, pair_diff, gaufunc2d, gaufunc2d_angles, circgaufunc, boxfunc2d
from library.visualization import customlegend
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p, directional_tuning_tile

mpl.rcParams['figure.dpi'] = 150
legendsize = 7


def get_tspdiff(SpikeDF, t, nidx1, nidx2):
    tidxsp1 = SpikeDF.loc[SpikeDF['neuronid'] == nidx1, 'tidxsp'].to_numpy()
    tidxsp2 = SpikeDF.loc[SpikeDF['neuronid'] == nidx2, 'tidxsp'].to_numpy()
    tsp1 = t[tidxsp1]
    tsp2 = t[tidxsp2]
    tsp_diff = pair_diff(tsp1, tsp2).flatten()
    tsp_diff = tsp_diff[np.abs(tsp_diff) < 100]
    return tsp_diff

def calc_exin_samepath(samebins, oppbins):
    # This function can only be used when the pass remains the same while the mossy projection is opposite
    ex_val_tmp, _ = pearsonr(samebins, oppbins)
    in_val_tmp, _ = pearsonr(samebins, np.flip(oppbins))
    ex_val, in_val = (ex_val_tmp + 1)/2, (in_val_tmp + 1)/2
    ex_bias = ex_val - in_val
    return ex_val, in_val, ex_bias

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
    # (mos_startx0, mos_starty0, mos_endx0, mos_endy0, 999, 'x'),
)

# Constant

wmax_ca3ca3 = 0  # 120
wmax_ca3ca3_adiff = 3000
w_ca3ca3_akappa = 2
wmax_ca3mosca3_adiff = 6000
w_ca3mosca3_akappa = 2
wmax_mosmos = 0  # 100
wmax_ca3mos = 0  # 1500
wmax_mosca3 = 0  # 2000
wmax_Mosin = 350  # 350
wmax_inMos = 35  # 35
wmax_CA3in = 50  # 50
wmax_inCA3 = 5  # 5
wmax_inin = 0  # 0
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
w_ca3ca3 = gaufunc2d_angles(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_ca3.reshape(nn_ca3, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_ca3.reshape(nn_ca3, 1), aatun1d_ca3.reshape(1, nn_ca3), aatun1d_ca3.reshape(nn_ca3, 1), wsd_ca3ca3, wmax_ca3ca3, wmax_ca3ca3_adiff, w_ca3ca3_akappa)
w_ca3mos = gaufunc2d_angles(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_mos.reshape(nn_mos, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_mos.reshape(nn_mos, 1), aatun1d_ca3.reshape(1, nn_ca3), aatun1d_mos.reshape(nn_mos, 1), wsd_ca3mos, wmax_ca3mos, wmax_ca3mosca3_adiff, w_ca3mosca3_akappa)
w_mosmos = gaufunc2d(xxtun1d_mos.reshape(1, nn_mos), xxtun1d_mos.reshape(nn_mos, 1), yytun1d_mos.reshape(1, nn_mos), yytun1d_mos.reshape(nn_mos, 1), wsd_mosmos, wmax_mosmos)

# Mos to CA3
act_adiff_max = wmax_ca3mosca3_adiff * np.exp(w_ca3mosca3_akappa * (np.cos(aatun1d_mos.reshape(1, -1) - aatun1d_ca3.reshape(-1, 1)) - 1))
mos_startpos = np.stack([mos_startx1, mos_starty1]).T
mos_endpos = np.stack([mos_endx1, mos_endy1]).T
w_mosca3 = np.zeros((xxtun1d_ca3.shape[0], xxtun1d_mos.shape[0], mos_startpos.shape[0]))
if mosproj_deg1 != 999:

    mos_act = np.zeros((posvec_mos.shape[0], mos_startpos.shape[0]))
    for i in range(mos_startpos.shape[0]):
        print('\rConstructing Weight matrix %d/%d'%(i, mos_startpos.shape[0]), flush=True, end='')
        w_mosca3[:, :, i], mos_act[:, i] = createMosProjMat_p2p(mos_startpos[i, :], mos_endpos[i, :], posvec_mos, posvec_CA3, wsd_mosca3)
    print()
    w_mosca3 = np.max(w_mosca3, axis=2)
    w_mosca3 = (wmax_mosca3 + act_adiff_max) * w_mosca3
else:
    w_mosca3 = w_mosca3[:, :, 0]
    print('No Mos.')

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

# Paths
save_dir = 'plots/test'

os.makedirs(save_dir, exist_ok=True)

mos_corr_dict = dict()
pair_idx_dict = {'Bestidx':[], 'Worstidx':[], 'Simidx':[], 'Dissimidx':[],
                 'Bestbins0':[], 'Worstbins0':[], 'Simbins0':[], 'Dissimbins0':[],
                 'Bestbins180':[], 'Worstbins180':[], 'Simbins180':[], 'Dissimbins180':[]}
for mos_startx1, mos_starty1, mos_endx1, mos_endy1, mosproj_deg1, sortby1 in mos_configs:

    pthtxt = 'deg-%d.png'%(mosproj_deg1)
    save_pth = join(save_dir, pthtxt)
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
    izhi_a_ex = 0.035
    izhi_b_ex = 0.2
    izhi_c_ex = -60
    izhi_d_ex = 8
    izhi_a_in = 0.02  # LTS
    izhi_b_in = 0.25
    izhi_c_in = -65
    izhi_d_in = 2
    V_ex, V_in = 0, -80
    V_thresh = 30
    spdelay = int(2/dt)  # 2ms
    I_noiseSD = 5

    # Theta inhibition
    theta_amp = 7
    theta_f = 10
    theta_T = 1/theta_f * 1e3
    theta_phase = np.mod(t, theta_T)/theta_T * 2*np.pi
    theta_phase_plot = np.mod(theta_phase + 2*np.pi, 2*np.pi)
    Itheta = (1 + np.cos(theta_phase))/2 * theta_amp

    # Positional drive
    EC_phase = np.deg2rad(290)
    Ipos_max = 2
    Iangle_diff = 10
    Ipos_sd = 5
    Iangle_kappa = 1
    ECstf_rest, ECstf_target = 0, 2
    tau_ECstf = 0.5e3
    U_ECstf = 0.001  # 0.001
    Ipos_max_compen = Ipos_max + (np.cos(EC_phase) + 1)/2 * theta_amp

    # Sensory tuning
    xmin, xmax, nx_ca3, nx_mos = -40, 40, 80, 40
    ymin, ymax, ny_ca3, ny_mos = -40, 40, 80, 40
    nn_inca3, nn_inmos = 250, 250
    nn_in = nn_inca3 + nn_inmos
    xtun_ca3 = np.linspace(xmin, xmax, nx_ca3)
    ytun_ca3 = np.linspace(ymin, ymax, ny_ca3)
    xxtun2d_ca3, yytun2d_ca3 = np.meshgrid(xtun_ca3, ytun_ca3)
    atun_seeds = np.array([0, np.pi/2, np.pi, np.pi*3/2])
    aatun2d_ca3 = directional_tuning_tile(*xxtun2d_ca3.shape, atun_seeds=atun_seeds, start_seed=0)
    xxtun1d_ca3, yytun1d_ca3, aatun1d_ca3 = xxtun2d_ca3.flatten(), yytun2d_ca3.flatten(), aatun2d_ca3.flatten()

    xtun_mos = np.linspace(xmin, xmax, nx_mos)
    ytun_mos = np.linspace(ymin, ymax, ny_mos)
    xxtun2d_mos, yytun2d_mos = np.meshgrid(xtun_mos, ytun_mos)
    xxtun1d_mos, yytun1d_mos = xxtun2d_mos.flatten(), yytun2d_mos.flatten()
    aatun2d_mos = directional_tuning_tile(*xxtun2d_mos.shape, atun_seeds=atun_seeds, start_seed=0)
    aatun1d_mos = aatun2d_mos.flatten()
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



    # Synapse parameters
    tau_gex = 12
    tau_gin = 10
    U_stdx_CA3 = 0.7
    U_stdx_mos = 0.7
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
    stdx2ca3 = np.ones(nn)
    stdx2mos = np.ones(nn)
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
        np.random.seed(i)
        Itotal = Isyn + Isen_fac + np.random.normal(0, I_noiseSD, size=nn) - Itheta[i]

        # Izhikevich
        v += (0.04*v**2 + 5*v + 140 - u + Itotal) * dt
        u += izhi_a * (izhi_b * v - u) * dt
        fidx = np.where(v > V_thresh)[0]
        v[fidx] = izhi_c[fidx]
        u[fidx] = u[fidx] + izhi_d[fidx]
        fidx_buffer.append(fidx)

        # STD
        d_stdx2ca3_dt = (1 - stdx2ca3)/tau_stdx
        d_stdx2ca3_dt[fidx] = d_stdx2ca3_dt[fidx] - U_stdx_CA3 * stdx2ca3[fidx]
        d_stdx2ca3_dt[nn_ca3:] = 0
        stdx2ca3 += d_stdx2ca3_dt * dt
        d_stdx2mos_dt = (1 - stdx2mos)/tau_stdx
        d_stdx2mos_dt[fidx] = d_stdx2mos_dt[fidx] - U_stdx_mos * stdx2mos[fidx]
        d_stdx2mos_dt[nn_ca3:] = 0
        stdx2mos += d_stdx2mos_dt * dt

        if i > spdelay:  # 2ms delay
            delayed_fidx = fidx_buffer.pop(0)
            delayed_fidx_ex = delayed_fidx[delayed_fidx < endidx_mos]
            delayed_fidx_in = delayed_fidx[delayed_fidx >= endidx_mos]

            # Synaptic input (Excitatory)
            spike2ca3_sum = np.sum(stdx2ca3[delayed_fidx_ex].reshape(1, -1) * w[:endidx_ca3, delayed_fidx_ex], axis=1) / endidx_mos
            spike2mos_sum = np.sum(stdx2mos[delayed_fidx_ex].reshape(1, -1) * w[endidx_ca3:endidx_mos, delayed_fidx_ex], axis=1) / endidx_mos
            spike2in_sum = np.sum(stdx2ca3[delayed_fidx_ex].reshape(1, -1) * w[endidx_mos:, delayed_fidx_ex], axis=1) / endidx_mos
            gex += (-gex/tau_gex + np.concatenate([spike2ca3_sum, spike2mos_sum, spike2in_sum])) * dt
            # spike_sum = np.sum(stdx2ca3[delayed_fidx_ex].reshape(1, -1) * w[:, delayed_fidx_ex], axis=1) / endidx_mos
            # gex += (-gex/tau_gex + spike_sum) * dt
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
        syneff_pop[i, :] = stdx2ca3
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
    fig, ax = plt.subplots(4, 6, figsize=(20, 10), facecolor='white', constrained_layout=True)

    gs = ax[0, 0].get_gridspec()
    for axeach in ax[0:2, 0:3].ravel():
        axeach.remove()
    axbig = fig.add_subplot(gs[0:2, 0:3])

    gs2 = ax[2, 0].get_gridspec()
    for axeach in ax[2:4, 0:3].ravel():
        axeach.remove()
    axbig2 = fig.add_subplot(gs2[2:4, 0:3])


    # # Population raster CA3
    # Indices along the trajectory
    all_nidx = np.zeros(traj_x.shape[0])
    for i in range(traj_x.shape[0]):
        run_x, run_y = traj_x[i], traj_y[i]
        nidx = np.argmin(np.square(run_x - xxtun1d_ca3) + np.square(run_y - yytun1d_ca3))
        all_nidx[i] = nidx
    all_nidx = np.unique(all_nidx).astype(int)
    egnidxs = all_nidx[[13, 19]]
    best_c, worst_c = 'r', 'blue'
    eg_cs = [best_c, worst_c]
    bestpair_nidxs = all_nidx[[13, 13+8]]
    worstpair_nidxs = all_nidx[[19, 19+8]]


    # Raster plot - CA3 pyramidal
    tsp_ras_ca3_list = []
    dxtun = xxtun1d[1] - xxtun1d[0]
    tt, traj_xx = np.meshgrid(t, xxtun1d[all_nidx])
    mappable = axbig.pcolormesh(tt, traj_xx, syneff_pop[:, all_nidx].T, shading='auto', vmin=0, vmax=1, cmap='gray')
    for neuronid in all_nidx:
        tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
        tsp_neuron = t[tidxsp_neuron]
        neuronx = xxtun1d[neuronid]
        if neuronid == egnidxs[0]:
            ras_c = eg_cs[0]
        elif neuronid == egnidxs[1]:
            ras_c = eg_cs[1]
        else:
            ras_c = 'lime'
        axbig.eventplot(tsp_neuron, lineoffsets=neuronx, linelengths=dxtun, linewidths=0.75, color=ras_c)
        if tidxsp_neuron.shape[0] < 1:
            continue
        tsp_ras_ca3_list.append(tsp_neuron)
    if len(tsp_ras_ca3_list) > 0:
        binsize = 5
        tsp_rasCA3 = np.concatenate(tsp_ras_ca3_list)
        count_bins, edges = np.histogram(tsp_rasCA3, bins=np.arange(t.min(), t.max(), binsize))
        rate_bins = count_bins / len(tsp_ras_ca3_list) / binsize * 1000
        ax_ca3rate = axbig.twinx()
        ax_ca3rate.bar(edges[:-1], rate_bins, width=np.diff(edges), align='edge', facecolor='gray', alpha=0.5, zorder=0.1)
        ax_ca3rate.tick_params(labelsize=legendsize)
        ax_ca3rate.set_ylim(0, 600)

    # Raster plot - CA3 inhibitory population
    nn_in_startidx = nn_ca3+nn_mos
    neuron_in_xmin, neuron_in_xmax = traj_x.max(), traj_x.max()*1.1
    dx_in = (neuron_in_xmax - neuron_in_xmin)/(nn_in)
    for neuronid in np.arange(nn_in_startidx, nn_in_startidx+nn_in, 1):
        tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
        tsp_neuron = t[tidxsp_neuron]
        plot_onset = neuron_in_xmin+(neuronid-nn_in_startidx) * dx_in
        if neuronid > endidx_inca3:
            ras_c = 'b'
        else:
            ras_c = 'm'
        axbig.eventplot(tsp_neuron, lineoffsets=plot_onset, linelengths=dx_in, linewidths=0.75, color=ras_c)
    axbig.plot(t, traj_x, c='k', linewidth=0.75)
    axbig.tick_params(labelsize=legendsize)
    cbar = fig.colorbar(mappable, ax=axbig, ticks=[0, 1, 2], shrink=0.5)
    cbar.set_label('Syn Efficacy', rotation=90, fontsize=legendsize)

    # # Raster plot for mossy neurons
    all_nidx_mos = np.zeros(traj_x.shape[0])
    for i in range(traj_x.shape[0]):
        nidx_tmp = np.argmin(np.square(traj_x[i] - xxtun1d_mos) + np.square(traj_y[i] - yytun1d_mos))
        all_nidx_mos[i] = nidx_tmp + endidx_ca3
    all_nidx_mos = np.unique(all_nidx_mos).astype(int)
    tsp_ras_list = []
    dxtun = xxtun1d[1] - xxtun1d[0]
    tt, traj_xx = np.meshgrid(t, xxtun1d[all_nidx_mos])
    mappable = axbig2.pcolormesh(tt, traj_xx, syneff_pop[:, all_nidx_mos].T, shading='auto', vmin=0, vmax=2, cmap='seismic')
    for neuronid in all_nidx_mos:
        tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
        tsp_neuron = t[tidxsp_neuron]
        neuronx = xxtun1d[neuronid]
        axbig2.eventplot(tsp_neuron, lineoffsets=neuronx, linelengths=dxtun, linewidths=0.75, color='lime')
        if tidxsp_neuron.shape[0] < 1:
            continue
        tsp_ras_list.append(tsp_neuron)
    axbig2.plot(t, traj_x, c='k', linewidth=0.75)
    if len(tsp_ras_list) > 0:
        binsize = 5
        tsp_rasMos = np.concatenate(tsp_ras_list)
        count_bins, edges = np.histogram(tsp_rasMos, bins=np.arange(t.min(), t.max(), binsize))
        rate_bins = count_bins / len(tsp_ras_list) / binsize * 1000
        ax_mosrate = axbig2.twinx()
        ax_mosrate.bar(edges[:-1], rate_bins, width=np.diff(edges), align='edge', facecolor='gray', alpha=0.5, zorder=0.1)
        ax_mosrate.tick_params(labelsize=legendsize)
        ax_mosrate.set_ylim(0, 600)

    # # 2, 3. Phase precession for two example cells
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
        ax[0, 3+axid].scatter(xsp_norm_eg, phasesp_eg, marker='|', s=4, color=eg_cs[axid])
        ax[0, 3+axid].axhline(mean_phasesp, xmin=0, xmax=0.3, linewidth=1)
        ax[0, 3+axid].plot(xdum, ydum, linewidth=0.75, color=eg_cs[axid])
        ax[0, 3+axid].set_title('y= %0.2fx + %0.2f, atun=%0.2f'%(rcc_slope_rad, rcc_c, aatun1d[egnidx]), fontsize=legendsize+1)
        ax[0, 3+axid].set_xlabel('Position', fontsize=legendsize)
        ax[0, 3+axid].set_ylim(0, 2*np.pi)
        ax[0, 3+axid].set_yticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])
        ax[0, 3+axid].set_yticklabels(['0', '$\pi/2$', '$\pi$', '$1.5\pi$', '$2\pi$'])
        ax[0, 3+axid].set_ylabel('Spike phase (rad)', fontsize=legendsize)

        ax_rate = ax[0, 3+axid].twinx().twiny()
        binsize = 20  # 20ms
        tmin, tmax = tsp_eg.min(), tsp_eg.max()
        count_bins, edges = np.histogram(tsp_eg, bins=np.arange(tmin, tmax+binsize, binsize))
        rate_bins = (count_bins / binsize * 1000)
        ax_rate.bar(edges[:-1], rate_bins, width=np.diff(edges), align='edge', facecolor='gray', alpha=0.5, zorder=0.1)
        ax_rate.set_ylim(0, 600)
        ax_rate.tick_params(axis='both', labelsize=legendsize)


    # # 3. Raster plot for inhibitory neurons - Zoomed in
    tsp_rasInCA3_list = []
    tsp_rasInMos_list = []
    inzoom_egtimes = (5000, 6000)
    for neuronid in np.arange(endidx_mos, endidx_in, 1):
        tidxsp_neuron = SpikeDF[(SpikeDF['neuronid'] == neuronid) & (SpikeDF['tidxsp'] <= inzoom_egtimes[1]) & (SpikeDF['tidxsp'] > inzoom_egtimes[0])]['tidxsp']
        tsp_neuron = t[tidxsp_neuron]

        if neuronid < endidx_inca3:
            ras_c = 'm'
            tsp_rasInCA3_list.append(tsp_neuron)
        else:
            ras_c = 'b'
            tsp_rasInMos_list.append(tsp_neuron)
        ax[1, 3].eventplot(tsp_neuron, lineoffsets=neuronid-endidx_mos, linelengths=2, linewidths=2, color=ras_c)

    for tsp_rasIn_list, hist_c in zip((tsp_rasInCA3_list, tsp_rasInMos_list), ('m', 'b')):
        if len(tsp_rasIn_list) > 0:
            tsp_rasIn = np.concatenate(tsp_rasIn_list)
            binsize = 2
            count_bins, edges = np.histogram(tsp_rasIn, bins=np.arange(inzoom_egtimes[0]/10, inzoom_egtimes[1]/10+binsize, binsize))
            rate_bins = count_bins / len(tsp_rasIn_list) / binsize * 1000
            ax_inrate = ax[1, 3].twinx()
            ax_inrate.bar(edges[:-1], rate_bins, width=np.diff(edges), align='edge', facecolor=hist_c, alpha=0.3, zorder=0.1)
            ax_inrate.tick_params(labelsize=legendsize)
            ax_inrate.set_ylim(0, 600)
    ax[1, 3].set_xlim(inzoom_egtimes[0]/10, inzoom_egtimes[1]/10)

    # # 4. ISI in frequency
    isi_dict = dict(ca3=[], mos=[], inca3=[], inmos=[])
    isi_c = dict(ca3='r', mos='g', inca3='purple', inmos='cyan')
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
    isi_edges = np.linspace(np.log10(1), np.log10(500), 75)

    for neuron_type in isi_dict.keys():
        isi_list = isi_dict[neuron_type]
        if len(isi_list) < 1:
            continue
        isi_eachtype = 1000/np.concatenate(isi_list)
        logISI_eachtype = np.log10(isi_eachtype)

        ax[1, 4].hist(logISI_eachtype, bins=isi_edges, histtype='step', color=isi_c[neuron_type],
                      label=neuron_type, linewidth=0.75, log=True)
    ax[1, 4].axvspan(np.log10(4), np.log10(15), color='gray', alpha=0.15)
    ax[1, 4].axvspan(np.log10(50), np.log10(150), color='gray', alpha=0.15)
    isi_xticks = np.array([4, 10, 15, 50, 150, 300])
    ax[1, 4].set_xticks(np.log10(isi_xticks))
    ax[1, 4].set_xticklabels(isi_xticks)
    ax[1, 4].set_xticks(np.log10(np.append(1, np.arange(10, 510, 10))), minor=True)
    customlegend(ax[1, 4], fontsize=legendsize+2)
    ax[1, 4].tick_params(axis='both', labelsize=legendsize)

    # # All onsets & Marginal phases
    phasesp_best_list, phasesp_worst_list = [], []
    onset_best_list, onset_worst_list = [], []
    slope_best_list, slope_worst_list = [], []
    phasenidx_best_list, phasenidx_worst_list = [], []
    for neuronid in range(endidx_ca3):
        neuronxtmp = xxtun1d[neuronid]
        if np.abs(neuronxtmp) > 20:
            continue

        tidxsp_tmp = SpikeDF.loc[SpikeDF['neuronid']==neuronid, 'tidxsp'].to_numpy()
        if tidxsp_tmp.shape[0] < 5:
            continue
        atun_this = aatun1d[neuronid]
        adiff = np.abs(cdiff(atun_this, 0))
        tsp_eg, phasesp_eg = t[tidxsp_tmp], theta_phase[tidxsp_tmp]
        xsp_eg = traj_x[tidxsp_tmp]
        xspmin, xsprange = xsp_eg.min(), xsp_eg.max() - xsp_eg.min()
        xsp_norm_eg = (xsp_eg-xspmin)/xsprange
        regress = rcc(xsp_norm_eg, phasesp_eg, abound=(-1., 1.))
        rcc_c, rcc_slope_rad = regress['phi0'], regress['aopt'] * 2 * np.pi
        if rcc_slope_rad > 0:
            continue

        if adiff < (np.pi/6):  # best
            phasesp_best_list.append(phasesp_eg)
            onset_best_list.append(rcc_c)
            phasenidx_best_list.append(neuronid)
            slope_best_list.append(rcc_slope_rad)
        elif adiff > (np.pi - np.pi/6):  # worst
            phasesp_worst_list.append(phasesp_eg)
            onset_worst_list.append(rcc_c)
            phasenidx_worst_list.append(neuronid)
            slope_worst_list.append(rcc_slope_rad)

        else:
            continue
    phasesp_best, phasesp_worst = np.concatenate(phasesp_best_list), np.concatenate(phasesp_worst_list)
    phasesp_bestmu, phasesp_worstmu = cmean(phasesp_best), cmean(phasesp_worst)
    onset_best, onset_worst = np.array(onset_best_list), np.array(onset_worst_list)
    onset_bestmu, onset_worstmu = cmean(onset_best), cmean(onset_worst)
    slope_best, slope_worst = np.array(slope_best_list), np.array(slope_worst_list)
    phasenidx_best, phasenidx_worst = np.array(phasenidx_best_list), np.array(phasenidx_worst_list)
    phasebins = np.linspace(0, 2*np.pi, 30)


    # # x,y- coordinates of best-worst sampled neurons
    ax[2, 3].plot(traj_x, traj_y, linewidth=0.75, c='gray')
    ax[2, 3].scatter(xxtun1d_ca3[phasenidx_best], yytun1d_ca3[phasenidx_best], c=best_c, s=1, marker='o', label='Best')
    ax[2, 3].scatter(xxtun1d_ca3[phasenidx_worst], yytun1d_ca3[phasenidx_worst], c=worst_c, s=1, marker='o', label='Worst')
    ax[2, 3].set_xlim(-20, 20)
    ax[2, 3].set_ylim(-20, 20)
    ax[2, 3].set_xlabel('Neuron x (cm)', fontsize=legendsize)
    ax[2, 3].set_ylabel('Neuron y (cm)', fontsize=legendsize)
    customlegend(ax[2, 3], fontsize=legendsize+2)


    # # Slopes and onsets of best-worst neurons
    ax[2, 4].scatter(onset_best, slope_best, marker='.', c=best_c, s=8, alpha=0.5)
    ax[2, 4].scatter(onset_worst, slope_worst, marker='.', c=worst_c, s=8, alpha=0.5)
    ax[2, 4].set_xlim(0, 2*np.pi)
    ax[2, 4].set_ylim(-np.pi, 0)
    ax[2, 4].set_xlabel('Onset (rad)', fontsize=legendsize)
    ax[2, 4].set_ylabel('Slope (rad)', fontsize=legendsize)

    # # Marginal spike phases
    binsphasebest, _, _ = ax[3, 3].hist(phasesp_best, bins=phasebins, density=True, histtype='step', color=best_c, label='Best=%0.2f (n=%d)'%(phasesp_bestmu, phasesp_best.shape[0]), linewidth=0.75)
    binsphaseworst, _, _ = ax[3, 3].hist(phasesp_worst, bins=phasebins, density=True, histtype='step', color=worst_c, label='Worst=%0.2f (n=%d)'%(phasesp_worstmu, phasesp_worst.shape[0]), linewidth=0.75)
    ax[3, 3].axvline(phasesp_bestmu, ymin=0.55, ymax=0.65, linewidth=0.75, color=best_c)
    ax[3, 3].axvline(phasesp_worstmu, ymin=0.55, ymax=0.65, linewidth=0.75, color=worst_c)
    ax[3, 3].annotate(r'$%0.2f$'%(phasesp_worstmu-phasesp_bestmu), xy=(0.8, 0.5), xycoords='axes fraction', fontsize=legendsize+4)
    ax[3, 3].set_ylim(0, np.max([binsphasebest.max(), binsphaseworst.max()])*2)
    ax[3, 3].set_xlabel('Marginal spike phase (rad)', fontsize=legendsize)
    customlegend(ax[3, 3], fontsize=legendsize+2)

    # # Marginal Onsets
    binsonsetbest, _, _ = ax[3, 4].hist(onset_best, bins=phasebins, density=True, histtype='step', color=best_c, label='Best=%0.2f (n=%d)'%(onset_bestmu, onset_best.shape[0]), linewidth=0.75)
    binsonsetworst, _, _ = ax[3, 4].hist(onset_worst, bins=phasebins, density=True, histtype='step', color=worst_c, label='Worst=%0.2f (n=%d)'%(onset_worstmu, onset_worst.shape[0]), linewidth=0.75)
    ax[3, 4].axvline(onset_bestmu, ymin=0.55, ymax=0.65, linewidth=0.75, color=best_c)
    ax[3, 4].axvline(onset_worstmu, ymin=0.55, ymax=0.65, linewidth=0.75, color=worst_c)
    ax[3, 4].annotate(r'$%0.2f$'%(onset_worstmu-onset_bestmu), xy=(0.8, 0.5), xycoords='axes fraction', fontsize=legendsize+4)
    ax[3, 4].set_ylim(0, np.max([binsonsetbest.max(), binsonsetworst.max()])*2)
    ax[3, 4].set_xlabel('Population onset (rad)', fontsize=legendsize)
    customlegend(ax[3, 4], fontsize=legendsize+2)


    # # # Correlograms - Single Examples
    # edges = np.arange(-100, 100, 5)
    # best_c, worst_c = 'r', 'g'
    # bestlabel, worstlabel = 'Best', 'Worst'
    # best_tspdiff = get_tspdiff(SpikeDF, t, bestpair_nidxs[0], bestpair_nidxs[1])
    # worst_tspdiff = get_tspdiff(SpikeDF, t, worstpair_nidxs[0], worstpair_nidxs[1])
    # bestbins, _, _ = ax[0, 5].hist(best_tspdiff, bins=edges, histtype='step', color=best_c, label=bestlabel)
    # worstbins, _, _ = ax[0, 5].hist(worst_tspdiff, bins=edges, histtype='step', color=worst_c, label=worstlabel)
    # ax[0, 5].set_title('Spike correlation', fontsize=legendsize)
    # mos_corr_dict['%d%s' % (mosproj_deg1, bestlabel)] = bestbins
    # mos_corr_dict['%d%s' % (mosproj_deg1, worstlabel)] = worstbins
    # customlegend(ax[0, 5], fontsize=legendsize)
    # if mosproj_deg1 == 180:
    #     samebest_bins, oppbest_bins = mos_corr_dict['0Best'], mos_corr_dict['180Best']
    #     sameworst_bins, oppworst_bins = mos_corr_dict['0Worst'], mos_corr_dict['180Worst']
    #     best_ex, best_in, best_exbias = calc_exin_samepath(samebest_bins, oppbest_bins)
    #     worst_ex, worst_in, worst_exbias = calc_exin_samepath(sameworst_bins, oppworst_bins)
    #     ax[0, 5].annotate('Best\nEx %0.2f In %0.2f \n= %0.2f\nWorst\nEx %0.2f In %0.2f \n= %0.2f'%(best_ex, best_in, best_exbias, worst_ex, worst_in, worst_exbias),
    #                       xy=(0.65, 0.50), xycoords='axes fraction', fontsize=legendsize+1)

    # # Ex-intrinsicity - All neurons for Sim, Dissim, Best, Worst
    edges = np.arange(-100, 100, 5)
    tspdiff_dict = dict(Best=[], Worst=[])
    if mosproj_deg1 == 0:
        all_sampled_nidx = np.concatenate([phasenidx_best, phasenidx_worst])
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
                tsp_diff = get_tspdiff(SpikeDF, t, nidx1, nidx2)
                if tsp_diff.shape[0] < 10:
                    continue
                tspdiff_bins, _ = np.histogram(tsp_diff, bins=edges)
                a1, a2 = aatun1d[nidx1], aatun1d[nidx2]
                absadiff = np.abs(cdiff(a1, a2))
                absadiff_a1pass = np.abs(cdiff(a1, 0))
                absadiff_a2pass = np.abs(cdiff(a2, 0))
                if absadiff < (np.pi/2):  # Similar
                    pair_idx_dict['Simidx'].append((nidx1, nidx2))
                    pair_idx_dict['Simbins0'].append(tspdiff_bins)
                if absadiff > (np.pi - np.pi/2):  # dismilar
                    pair_idx_dict['Dissimidx'].append((nidx1, nidx2))
                    pair_idx_dict['Dissimbins0'].append(tspdiff_bins)
                if (absadiff_a1pass < (np.pi/6)) & (absadiff_a2pass < (np.pi/6)):  # Both best
                    pair_idx_dict['Bestidx'].append((nidx1, nidx2))
                    pair_idx_dict['Bestbins0'].append(tspdiff_bins)
                    tspdiff_dict['Best'].append(tsp_diff)
                if (absadiff_a1pass > (np.pi - np.pi/6)) & (absadiff_a2pass > (np.pi - np.pi/6)):  # Both worst
                    pair_idx_dict['Worstidx'].append((nidx1, nidx2))
                    pair_idx_dict['Worstbins0'].append(tspdiff_bins)
                    tspdiff_dict['Worst'].append(tsp_diff)

        all_adiffsim = []
        all_adiffdissim = []
        for k in range(len(pair_idx_dict['Simidx'])):
            nidx1, nidx2 = pair_idx_dict['Simidx'][k]
            a1, a2 = aatun1d[nidx1], aatun1d[nidx2]
            all_adiffsim.append(np.abs(cdiff(a1, a2)))
        for k in range(len(pair_idx_dict['Dissimidx'])):
            nidx1, nidx2 = pair_idx_dict['Dissimidx'][k]
            a1, a2 = aatun1d[nidx1], aatun1d[nidx2]
            all_adiffdissim.append(np.abs(cdiff(a1, a2)))


    elif (mosproj_deg1 == 180) and (len(pair_idx_dict['Simidx']) > 0):
        exindict = dict()
        for pairtype in ['Sim', 'Dissim', 'Best', 'Worst']:
            exindict[pairtype] = {'ex': [], 'in': [], 'ex_bias': []}
            for k in range(len(pair_idx_dict[pairtype + 'idx'])):
                nidx1, nidx2 = pair_idx_dict[pairtype + 'idx'][k]
                tsp_diff = get_tspdiff(SpikeDF, t, nidx1, nidx2)
                if tsp_diff.shape[0] < 10:
                    continue
                if (pairtype == 'Best') or (pairtype == 'Worst'):
                    tspdiff_dict[pairtype].append(tsp_diff)
                tspdiff_180bins, _ = np.histogram(tsp_diff, bins=edges)
                tspdiff_0bins = pair_idx_dict[pairtype+'bins0'][k]
                ex_val, in_val, ex_bias = calc_exin_samepath(tspdiff_0bins, tspdiff_180bins)
                exindict[pairtype]['ex'].append(ex_val)
                exindict[pairtype]['in'].append(in_val)
                exindict[pairtype]['ex_bias'].append(ex_bias)
            exindict[pairtype]['ex_n'] = (np.array(exindict[pairtype]['ex_bias']) > 0).sum()
            exindict[pairtype]['in_n'] = (np.array(exindict[pairtype]['ex_bias']) <= 0).sum()
            exindict[pairtype]['exin_ratio'] = exindict[pairtype]['ex_n']/exindict[pairtype]['in_n']
            exindict[pairtype]['ex_bias_mu'] = np.mean(exindict[pairtype]['ex_bias'])


        sim_c, dissim_c = 'm', 'gold'
        simlabel = 'Sim %d / %d = %0.2f' %(exindict['Sim']['ex_n'], exindict['Sim']['in_n'], exindict['Sim']['ex_n']/exindict['Sim']['in_n'])
        dissimlabel = 'Dissim %d / %d = %0.2f' %(exindict['Dissim']['ex_n'], exindict['Dissim']['in_n'], exindict['Dissim']['ex_n']/exindict['Dissim']['in_n'])
        ax[1, 5].scatter(exindict['Sim']['ex'], exindict['Sim']['in'], c=sim_c, marker='.', s=8, alpha=0.7, label=simlabel)
        ax[1, 5].scatter(exindict['Dissim']['ex'], exindict['Dissim']['in'], c=dissim_c, marker='.', s=8, alpha=0.7, label=dissimlabel)
        ax[1, 5].plot([0, 1], [0, 1], linewidth=0.75, c='k')
        ax[1, 5].set_xlim(0, 1)
        ax[1, 5].set_ylim(0, 1)
        customlegend(ax[1, 5], fontsize=legendsize+2)
        bestlabel = 'Best %d / %d = %0.2f' %(exindict['Best']['ex_n'], exindict['Best']['in_n'], exindict['Best']['ex_n']/exindict['Best']['in_n'])
        worstlabel = 'Worst %d / %d = %0.2f' %(exindict['Worst']['ex_n'], exindict['Worst']['in_n'], exindict['Worst']['ex_n']/exindict['Worst']['in_n'])
        ax[2, 5].scatter(exindict['Best']['ex'], exindict['Best']['in'], c=best_c, marker='.', s=8, alpha=0.7, label=bestlabel)
        ax[2, 5].scatter(exindict['Worst']['ex'], exindict['Worst']['in'], c=worst_c, marker='.', s=8, alpha=0.7, label=worstlabel)
        ax[2, 5].plot([0, 1], [0, 1], linewidth=0.75, c='k')
        ax[2, 5].set_xlim(0, 1)
        ax[2, 5].set_ylim(0, 1)
        customlegend(ax[2, 5], fontsize=legendsize+2)

        bias_edges = np.linspace(-1, 1, 50)
        stattext = 'Sim= %0.3f, Dissim= %0.3f\n' % (exindict['Sim']['ex_bias_mu'], exindict['Dissim']['ex_bias_mu'])
        stattext += 'Dissim-Sim = %0.3f\n' %( exindict['Dissim']['ex_bias_mu'] - exindict['Sim']['ex_bias_mu'])
        stattext += 'Best= %0.3f, Worst= %0.3f\n' % (exindict['Best']['ex_bias_mu'], exindict['Worst']['ex_bias_mu'])
        stattext += 'Best-Worst = %0.3f' %( exindict['Best']['ex_bias_mu'] - exindict['Worst']['ex_bias_mu'])
        nbins1, _, _ = ax[3, 5].hist(exindict['Sim']['ex_bias'], bins=bias_edges, color=sim_c, linewidth=0.75, histtype='step', density=True)
        ax[3, 5].axvline(exindict['Sim']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=sim_c)
        nbins2, _, _ = ax[3, 5].hist(exindict['Dissim']['ex_bias'], bins=bias_edges, color=dissim_c, linewidth=0.75, histtype='step', density=True)
        ax[3, 5].axvline(exindict['Dissim']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=dissim_c)
        nbins3, _, _ = ax[3, 5].hist(exindict['Best']['ex_bias'], bins=bias_edges, color=best_c, linewidth=0.75, histtype='step', density=True)
        ax[3, 5].axvline(exindict['Best']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=best_c)
        nbins4, _, _ = ax[3, 5].hist(exindict['Worst']['ex_bias'], bins=bias_edges, color=worst_c, linewidth=0.75, histtype='step', density=True)
        ax[3, 5].axvline(exindict['Worst']['ex_bias_mu'], ymin=0.55, ymax=0.65, linewidth=0.75, color=worst_c)
        ax[3, 5].set_ylim(0, np.max(np.concatenate([nbins1, nbins2, nbins3, nbins4]))*2)
        ax[3, 5].annotate(stattext, xy=(0.4, 0.7), fontsize=legendsize+1, xycoords='axes fraction')
    else:
        pass

    if mosproj_deg1 != 999:
        edges = np.arange(-100, 100, 5)
        best_tspdifftheta = np.concatenate(tspdiff_dict['Best'])
        worst_tspdifftheta = np.concatenate(tspdiff_dict['Worst'])
        bestbins, _, _ = ax[0, 5].hist(best_tspdifftheta, bins=edges, histtype='step', color=best_c, label='Best', density=True)
        worstbins, _, _ = ax[0, 5].hist(worst_tspdifftheta, bins=edges, histtype='step', color=worst_c, label='Worst', density=True)



    # # All plot setting / Save
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for ax_each in [axbig, axbig2]:
        for i in theta_cutidx:
            ax_each.axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)
    for ax_each in ax.ravel():
        ax_each.tick_params(labelsize=legendsize)
    for ax_each in [axbig2]:
        ax_each.tick_params(labelsize=legendsize)
    fig.savefig(save_pth, dpi=150)
    plt.close()

    # # ========================= Legacy code =======================================================
    # # # x. Currents, STD and STF for neuron example 1
    # tidxsp_eg = SpikeDF.loc[SpikeDF['neuronid']==egnidxs[0], 'tidxsp'].to_numpy()
    # tsp_eg, phasesp_eg = t[tidxsp_eg], theta_phase[tidxsp_eg]
    # ax[2, 3].plot(t, Isyn_pop[:, egnidxs[0]], label='Isyn', linewidth=0.75, color='orange')
    # ax[2, 3].plot(t, Isen_pop[:, egnidxs[0]], label='Isen', linewidth=0.75, color='blue')
    # ax[2, 3].plot(t, Isen_fac_pop[:, egnidxs[0]], label='Isen_fac', linewidth=0.75, color='cyan')
    # ax[2, 3].set_ylim(-1, 40)
    # customlegend(ax[2, 3], fontsize=legendsize, loc='upper left')
    # axsyneff = ax[2, 3].twinx()
    # axsyneff.plot(t, syneff_pop[:, egnidxs[0]], label='snyeff', color='r', linewidth=0.75)
    # axsyneff.plot(t, ECstfx_pop[:, egnidxs[0]], label='ECSTF', color='green', linewidth=0.75)
    # axsyneff.set_ylim(-0.1, 2.1)
    # axsyneff.eventplot(tsp_eg, lineoffsets = 1.8, linelength=0.1, linewidths=0.5, color='r')
    # customlegend(axsyneff, fontsize=legendsize)
    # axsyneff.tick_params(labelsize=legendsize)

    # # # x. Input synaptic currents into neuron example 1
    # egaxidx = 0
    # egidx_in = egnidxs[0]
    # tidxstart, tidxend = inzoom_egtimes  # in index = 0.1ms
    # tslice, IsynINslice = t[tidxstart:tidxend], IsynIN_pop[tidxstart:tidxend, egidx_in]
    # Isynslice = Isyn_pop[tidxstart:tidxend, egidx_in]
    # IsynEXslice = IsynEX_pop[tidxstart:tidxend, egidx_in]
    # tslice_offset = tslice-tslice.min()
    # ax[0, 4].plot(tslice_offset, Isynslice, color='gray', linewidth=0.5, alpha=0.7)
    # ax[0, 4].plot(tslice_offset, IsynEXslice, color='b', linewidth=0.5, alpha=0.7)
    # ax[0, 4].plot(tslice_offset, IsynINslice, color=eg_cs[egaxidx], linewidth=0.5, alpha=0.7)
    # ax[0, 4].set_xticks(np.arange(0, 101, 10))
    # ax[0, 4].set_xticks(np.arange(0, 101, 5), minor=True)