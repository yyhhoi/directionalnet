# For Plotting

from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from scipy.stats import vonmises, pearsonr, circmean
from pycircstat import cdiff, mean as cmean
from library.comput_utils import cal_hd_np, get_nidx_np, pair_diff
from library.visualization import customlegend
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p
mpl.rcParams['figure.dpi'] = 150
legendsize = 7


def gaufunc2d(x, mux, y, muy, sd, outmax):
    return outmax * np.exp(-(np.square(x - mux) + np.square(y - muy))/(2*sd**2))

def gaufunc2d_angles(x, mux, y, muy, a, mua, sd, outmax, w_adiff, w_akappa):
    outmax_angled = outmax + w_adiff * np.exp(w_akappa * (np.cos(a - mua) - 1))
    return outmax_angled * np.exp(-(np.square(x - mux) + np.square(y - muy))/(2*sd**2))

def circgaufunc(x, loc, kappa, outmax):
    return outmax * np.exp(kappa * (np.cos(x - loc) - 1))

def boxfunc2d(x, mux, y, muy, sd, outmax):

    dist = np.sqrt( np.square(x-mux) + np.square(y-muy))
    out = np.ones(dist.shape) * outmax
    out[dist > sd] = 0
    return out

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
    # (mos_startx180, mos_starty180, mos_endx180, mos_endy180, 180, 'x'),
    # (mos_startx0, mos_starty0, mos_endx0, mos_endy0, 999, 'x'),
)

# Constant
Ipos_max = 2
Iangle_diff = 10
Ipos_sd = 5
I_noiseSD = 5
izhi_a_ex = 0.035
izhi_b_ex = 0.2
izhi_c_ex = -60
izhi_d_ex = 8
izhi_a_in = 0.02  # LTS
izhi_b_in = 0.25
izhi_c_in = -65
izhi_d_in = 2
U_stdx_CA3 = 0.7
U_stdx_mos = 0.7
wmax_ca3ca3 = 0  # 120
wmax_adiff = 3500

w_akappa = 2
wmax_mosmos = 0  # 100
wmax_ca3mos = 1500  # 1500
wmax_mosca3 = 2000  # 2000
wmax_Mosin = 350  # 500
wmax_inMos = 35  # 300
wmax_CA3in = 50  # 57
wmax_inCA3 = 5  # 6
wmax_inin = 0  # 0

# Paths
save_dir = 'plots/presentation/'
os.makedirs(save_dir, exist_ok=True)
# Parameter Scan
# Ipos_maxs = np.arange(0.5, 5.5, 0.5)
# for Ipos_max in Ipos_maxs:
#     Iangle_diff = Ipos_max * 2
mos_corr_dict = dict()
pair_idx_dict = {'Bestidx':[], 'Worstidx':[], 'Simidx':[], 'Dissimidx':[],
                 'Bestbins0':[], 'Worstbins0':[], 'Simbins0':[], 'Dissimbins0':[],
                 'Bestbins180':[], 'Worstbins180':[], 'Simbins180':[], 'Dissimbins180':[]}
for mos_startx1, mos_starty1, mos_endx1, mos_endy1, mosproj_deg1, sortby1 in mos_configs:


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
    xmin, xmax, nx_ca3, nx_mos = -40, 40, 80, 80
    ymin, ymax, ny_ca3, ny_mos = -40, 40, 80, 80
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


    # Plot Positional and Directional tuning
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xxtun1d_ca3, yytun1d_ca3, c='k', marker='.', s =1)
    ax.set_xlabel('x (cm)', fontsize=legendsize+2)
    ax.set_ylabel('y (cm)', fontsize=legendsize+2)
    ax.set_title('Positional and Directional Tuning of CA3 place cells', fontsize=legendsize+2)
    ax.tick_params(labelsize=legendsize+2)
    fig.savefig(join(save_dir, 'Directional Tuning.png'))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xxtun1d_ca3, yytun1d_ca3, c='gray', alpha=0.7, marker='o')
    ax.quiver(xxtun1d_ca3, yytun1d_ca3, np.cos(aatun1d_ca3), np.sin(aatun1d_ca3), scale=20)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.tick_params(labelsize=legendsize+2)
    fig.savefig(join(save_dir, 'Directional Tuning zoomed.png'))
    plt.close(fig)

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
    # w_ca3ca3 = gaufunc2d(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_ca3.reshape(nn_ca3, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_ca3.reshape(nn_ca3, 1), wsd_ca3ca3, wmax_ca3ca3)
    w_ca3ca3 = gaufunc2d_angles(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_ca3.reshape(nn_ca3, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_ca3.reshape(nn_ca3, 1), aatun1d_ca3.reshape(1, nn_ca3), aatun1d_ca3.reshape(nn_ca3, 1), wsd_ca3ca3, wmax_ca3ca3, wmax_adiff, w_akappa)
    w_ca3mos = gaufunc2d(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_mos.reshape(nn_mos, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_mos.reshape(nn_mos, 1), wsd_ca3mos, wmax_ca3mos)
    w_mosmos = gaufunc2d(xxtun1d_mos.reshape(1, nn_mos), xxtun1d_mos.reshape(nn_mos, 1), yytun1d_mos.reshape(1, nn_mos), yytun1d_mos.reshape(nn_mos, 1), wsd_mosmos, wmax_mosmos)

    # Mos to CA3
    mos_startpos = np.stack([mos_startx1, mos_starty1]).T
    mos_endpos = np.stack([mos_endx1, mos_endy1]).T
    w_mosca3 = np.zeros((xxtun1d_ca3.shape[0], xxtun1d_mos.shape[0], mos_startpos.shape[0]))
    if mosproj_deg1 != 999:

        mos_act = np.zeros((posvec_mos.shape[0], mos_startpos.shape[0]))
        for i in range(mos_startpos.shape[0]):
            print('\rConstructing Weight matrix %d/%d'%(i, mos_startpos.shape[0]), flush=True, end='')
            w_mosca3[:, :, i], mos_act[:, i] = createMosProjMat_p2p(mos_startpos[i, :], mos_endpos[i, :], posvec_mos, posvec_CA3, wmax_mosca3, wsd_mosca3)
        print()
        w_mosca3 = np.max(w_mosca3, axis=2)
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

    # Plot Weight matrix
    egnidx = np.argmin(np.square(0 - xxtun1d_ca3) + np.square(0 - yytun1d_ca3) + np.square(0 - aatun1d_ca3))
    w2d_preeg = w_ca3ca3[:, egnidx].reshape(nx_ca3, nx_ca3)
    fig_config, ax_config = plt.subplots(figsize=(6, 4))
    im = ax_config.scatter(xxtun2d_ca3, yytun2d_ca3, c=w2d_preeg, cmap='Blues')
    plt.colorbar(im, ax=ax_config)
    ax_config.quiver(xxtun1d_ca3, yytun1d_ca3, np.cos(aatun1d_ca3), np.sin(aatun1d_ca3), scale=30, alpha=w2d_preeg.flatten()/w2d_preeg.max())
    ax_config.set_xlim(-10, 10)
    ax_config.set_ylim(-10, 10)
    ax_config.set_xlabel('x (cm)', fontsize=legendsize+2)
    ax_config.set_ylabel('y (cm)', fontsize=legendsize+2)
    ax_config.tick_params(labelsize=legendsize+2)
    ax_config.set_title('Synaptic weight $W_{i, j}$ \nfrom place cell at (0, 0) to (x, y)', fontsize=legendsize+2)
    fig_config.savefig(join(save_dir, 'CA3-CA3_Weights.png'), dpi=150)
    plt.close(fig_config)


    # Plot Mos positions
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xxtun1d_mos, yytun1d_mos, c='gray')
    ax.set_xlabel('x (cm)', fontsize=legendsize+2)
    ax.set_ylabel('y (cm)', fontsize=legendsize+2)
    ax.set_title('Positions of mossy cells', fontsize=legendsize+2)
    ax.tick_params(labelsize=legendsize+2)
    fig.savefig(join(save_dir, 'Mossy cells.png'))
    plt.close(fig)

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



# # Synapse parameters
    # tau_gex = 12
    # tau_gin = 10
    # # U_stdx = 0.350
    # tau_stdx = 0.5e3  # 1
    #
    # # Initialization
    # izhi_a = np.ones(nn) * izhi_a_ex
    # izhi_b = np.ones(nn) * izhi_b_ex
    # izhi_c = np.ones(nn) * izhi_c_ex
    # izhi_d = np.ones(nn) * izhi_d_ex
    # izhi_a[endidx_mos:endidx_in] = izhi_a_in
    # izhi_b[endidx_mos:endidx_in] = izhi_b_in
    # izhi_c[endidx_mos:endidx_in] = izhi_c_in
    # izhi_d[endidx_mos:endidx_in] = izhi_d_in
    #
    # v = np.ones(nn) * izhi_c
    # u = np.zeros(nn)
    # Isyn = np.zeros(nn)
    # Isyn_in = np.zeros(nn)
    # Isyn_ex = np.zeros(nn)
    # gex = np.zeros(nn)
    # gin = np.zeros(nn)
    # stdx2ca3 = np.ones(nn)
    # stdx2mos = np.ones(nn)
    # ECstfx = np.ones(nn) * ECstf_rest
    # fidx_buffer = []
    # SpikeDF_dict = dict(neuronid=[], tidxsp=[])
    # v_pop = np.zeros((t.shape[0], nn))
    # Isen_pop = np.zeros((t.shape[0], nn))
    # Isen_fac_pop = np.zeros((t.shape[0], nn))
    # Isyn_pop = np.zeros((t.shape[0], nn))
    # IsynIN_pop = np.zeros((t.shape[0], nn))
    # IsynEX_pop = np.zeros((t.shape[0], nn))
    # Itotal_pop = np.zeros((t.shape[0], nn))
    # syneff_pop = np.zeros((t.shape[0], nn))
    # ECstfx_pop = np.zeros((t.shape[0], nn))
    #
    # # # Simulation runtime
    # numt = t.shape[0]
    # t1 = time.time()
    #
    # for i in range(numt):
    #     print('\rSimulation %d/%d'%(i, numt), flush=True, end='')
    #     # Behavioural
    #     run_x, run_y, run_a = traj_x[i], traj_y[i], traj_a[i]
    #
    #     # Sensory input
    #     Iangle = circgaufunc(run_a, aatun1d, Iangle_kappa, Iangle_diff)
    #     ECtheta = (np.cos(theta_phase[i] + EC_phase) + 1)/2
    #     Isen = boxfunc2d(run_x, xxtun1d, run_y, yytun1d, Ipos_sd, Ipos_max_compen+Iangle) * ECtheta
    #     Isen[nn_ca3:] = 0
    #     ECstfx += ((ECstf_rest-ECstfx)/tau_ECstf + (ECstf_target - ECstfx) * U_ECstf * Isen) * dt
    #     Isen_fac = np.square(ECstfx) * Isen
    #
    #
    #     # Total Input
    #     np.random.seed(i)
    #     Itotal = Isyn + Isen_fac - Itheta[i] + np.random.normal(0, I_noiseSD, size=nn)
    #
    #     # Izhikevich
    #     v += (0.04*v**2 + 5*v + 140 - u + Itotal) * dt
    #     u += izhi_a * (izhi_b * v - u) * dt
    #     fidx = np.where(v > V_thresh)[0]
    #     v[fidx] = izhi_c[fidx]
    #     u[fidx] = u[fidx] + izhi_d[fidx]
    #     fidx_buffer.append(fidx)
    #
    #     # STD
    #     d_stdx2ca3_dt = (1 - stdx2ca3)/tau_stdx
    #     d_stdx2ca3_dt[fidx] = d_stdx2ca3_dt[fidx] - U_stdx_CA3 * stdx2ca3[fidx]
    #     d_stdx2ca3_dt[nn_ca3:] = 0
    #     stdx2ca3 += d_stdx2ca3_dt * dt
    #     d_stdx2mos_dt = (1 - stdx2mos)/tau_stdx
    #     d_stdx2mos_dt[fidx] = d_stdx2mos_dt[fidx] - U_stdx_mos * stdx2mos[fidx]
    #     d_stdx2mos_dt[nn_ca3:] = 0
    #     stdx2mos += d_stdx2mos_dt * dt
    #
    #     if i > spdelay:  # 2ms delay
    #         delayed_fidx = fidx_buffer.pop(0)
    #         delayed_fidx_ex = delayed_fidx[delayed_fidx < endidx_mos]
    #         delayed_fidx_in = delayed_fidx[delayed_fidx >= endidx_mos]
    #
    #         # Synaptic input (Excitatory)
    #         spike2ca3_sum = np.sum(stdx2ca3[delayed_fidx_ex].reshape(1, -1) * w[:endidx_ca3, delayed_fidx_ex], axis=1) / endidx_mos
    #         spike2mos_sum = np.sum(stdx2mos[delayed_fidx_ex].reshape(1, -1) * w[endidx_ca3:endidx_mos, delayed_fidx_ex], axis=1) / endidx_mos
    #         spike2in_sum = np.sum(stdx2ca3[delayed_fidx_ex].reshape(1, -1) * w[endidx_mos:, delayed_fidx_ex], axis=1) / endidx_mos
    #         gex += (-gex/tau_gex + np.concatenate([spike2ca3_sum, spike2mos_sum, spike2in_sum])) * dt
    #         # spike_sum = np.sum(stdx2ca3[delayed_fidx_ex].reshape(1, -1) * w[:, delayed_fidx_ex], axis=1) / endidx_mos
    #         # gex += (-gex/tau_gex + spike_sum) * dt
    #         Isyn_ex = gex * (V_ex - v)
    #
    #         # Synaptic input (Inhibitory)
    #         spike_sum = np.sum(w[:, delayed_fidx_in], axis=1) / nn_in
    #         gin += (-gin/tau_gin + spike_sum) * dt
    #         Isyn_in = gin * (V_in - v)
    #         Isyn = Isyn_ex + Isyn_in
    #
    #     # Store data
    #     SpikeDF_dict['neuronid'].extend(list(fidx))
    #     SpikeDF_dict['tidxsp'].extend([i] * len(fidx))
    #
    #     v_pop[i, :] = v
    #     Isen_pop[i, :] = Isen
    #     Isen_fac_pop[i, :] = Isen_fac
    #     Isyn_pop[i, :] = Isyn
    #     IsynIN_pop[i, :] = Isyn_in
    #     IsynEX_pop[i, :] = Isyn_ex
    #     Itotal_pop[i, :] = Itotal
    #     syneff_pop[i, :] = stdx2ca3
    #     ECstfx_pop[i, :] = ECstfx
    #
    # print('\nSimulation time = %0.2fs'%(time.time()-t1))
    #
    # # # Storage
    # SpikeDF = pd.DataFrame(SpikeDF_dict)
    # SpikeDF['neuronx'] = SpikeDF['neuronid'].apply(lambda x : xxtun1d[x])
    #
    # NeuronDF = pd.DataFrame(dict(neuronid=np.arange(nn), neuronx=xxtun1d, neurony=yytun1d, neurona=aatun1d,
    #                              neurontype=["CA3"]*nn_ca3 + ['Mos']*nn_mos + ['In']*nn_in))
    #
    # BehDF = pd.DataFrame(dict(t=t, x=traj_x, y=traj_y, a=traj_a, Itheta=Itheta, theta_phase=theta_phase,
    #                           theta_phase_plot=theta_phase_plot))
    #
    # ActivityData = dict(v=v_pop, Isen=Isen_pop, Isyn=Isyn_pop, Isen_fac=Isen_fac_pop,
    #                     Itotal=Itotal_pop, syneff=syneff_pop, ECstf=ECstfx_pop)
    #
    # MetaData = dict(nn=nn, nn_ca3=nn_ca3, nn_mos=nn_mos, nn_in=nn_in, w=w, EC_phase=EC_phase)
    #
    # # # Plot cells along the trajectory
    # all_nidx = np.zeros(traj_x.shape[0])
    # for i in range(traj_x.shape[0]):
    #     run_x, run_y = traj_x[i], traj_y[i]
    #     nidx = np.argmin(np.square(run_x - xxtun1d_ca3) + np.square(run_y - yytun1d_ca3))
    #     all_nidx[i] = nidx
    # all_nidx = np.unique(all_nidx).astype(int)
    # egnidxs = all_nidx[[13, 19]]
    # best_c, worst_c = 'r', 'green'
    # eg_cs = [best_c, worst_c]
    # bestpair_nidxs = all_nidx[[13, 13+8]]
    # worstpair_nidxs = all_nidx[[19, 19+8]]
    # bestpair_c, worstpair_c = 'm', 'green'
    #
    #
    #
    # # # Plot Sensory input
    # egnidx = np.argmin(np.square(0 - xxtun1d_ca3) + np.square(0 - yytun1d_ca3) + np.square(0 - aatun1d_ca3))
    # fig, ax = plt.subplots(figsize=(6, 3.75))
    # ax.plot(traj_x, Isen_pop[:, egnidx], label='$I_{sen}$', linewidth=0.75, color='blue')
    # ax.plot(traj_x, Isen_fac_pop[:, egnidx], label=r"$I_{sen\_fac}$", linewidth=0.75, color='cyan')
    # ax.set_xlabel('x (cm)')
    # ax.set_ylabel('Current')
    # ax.set_ylim(0, 45)
    # ax.legend(fontsize=legendsize+4, loc='upper left')
    # ax.tick_params(labelsize=legendsize+2)
    # ax2 = ax.twiny().twinx()
    # ax2.plot(t, ECstfx_pop[:, egnidx], label='$x_{STF}$', linewidth=0.75, color='green')
    # ax2.set_xlabel('time (ms)')
    # ax2.xaxis.set_label_position('top')
    # ax2.set_ylim(0, 2)
    # ax2.set_ylabel('Synaptic efficacy')
    # ax2.tick_params(labelsize=legendsize+2)
    # ax2.legend(fontsize=legendsize+4, loc='upper right')
    # theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    # for i in theta_cutidx:
    #     ax2.axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)
    # fig.savefig(join(save_dir, 'Sensory input 0.png'))
    # fig.tight_layout()
    # plt.close()
    #
    # # # Plot cells along the trajectory
    # fig, ax = plt.subplots(figsize=(6, 1.5), constrained_layout=True)
    # ax.plot(traj_x, traj_y, c='k', linewidth=1)
    # ax.scatter(xxtun1d_ca3[all_nidx], yytun1d_ca3[all_nidx], marker='o', c='gray', alpha=0.7)
    # ax.scatter(xxtun1d_ca3[egnidxs[0]], yytun1d_ca3[egnidxs[0]], marker='o', c=eg_cs[0], alpha=0.7)
    # ax.scatter(xxtun1d_ca3[egnidxs[1]], yytun1d_ca3[egnidxs[1]], marker='o', c=eg_cs[1], alpha=0.7)
    # ax.quiver(xxtun1d_ca3[all_nidx], yytun1d_ca3[all_nidx], np.cos(aatun1d_ca3[all_nidx]), np.sin(aatun1d_ca3[all_nidx]), scale=20, headwidth=5, width=0.0025)
    # ax.set_xlabel('x (cm)', fontsize=legendsize+2)
    # ax.set_ylabel('y (cm)', fontsize=legendsize+2)
    # ax.set_xlim(-22, 22)
    # ax.set_ylim(-5, 5)
    # ax.tick_params(labelsize=legendsize+2)
    # fig.savefig(join(save_dir, 'population_along_traj.png'))
    # plt.close(fig)
