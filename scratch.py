# The script is used as a "scratch" - testing small function/script segments.
# The code is discarded after use every time




# EC-STF, with CH
# Current addition: 2D

from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from scipy.stats import vonmises, pearsonr
from pycircstat import cdiff, mean as cmean
from library.comput_utils import cal_hd_np, get_nidx_xy_np, pair_diff
from library.visualization import customlegend
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p

def gaufunc2d(x, mux, y, muy, sd, outmax):
    return outmax * np.exp(-(np.square(x - mux) + np.square(y - muy))/(2*sd**2) )

def circgaufunc(x, loc, kappa, outmax):
    return outmax * np.exp(kappa * (np.cos(x - loc) - 1))

def boxfunc2d(x, mux, y, muy, sd, outmax):

    dist = np.sqrt( np.square(x-mux) + np.square(y-muy))
    out = np.ones(dist.shape) * outmax
    out[dist > sd] = 0
    return out

save_dir = 'plots/scratch'
os.makedirs(save_dir, exist_ok=True)

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
wsd_global = 2


# Parameter Scan
mos_shifts = ((4, 0, "Same"), (-4, 0, 'Opp'))

wmax_inMos = 140
wmax_Mosin = 140
mos_corr_dict = dict()


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



posvec_mos = np.stack([xxtun1d_mos, yytun1d_mos]).T
posvec_CA3 = np.stack([xxtun1d_ca3, yytun1d_ca3]).T
actmax = 10
actsd = 4

# Mossy layer
proj_len = 4
proj_len_sqrt = 4 / np.sqrt(2)
mos_startx1 = np.arange(-20, 21, 2)  # 0 degree
mos_starty1 = np.zeros(mos_startx1.shape[0])
mos_endx1, mos_endy1 = mos_startx1 + proj_len, mos_starty1

# mos_startx2 = np.arange(-20, 21, 2)  # 45 degree
# mos_starty2 = np.arange(-20, 21, 2)
# mos_endx2, mos_endy2 = mos_startx2 + proj_len_sqrt, mos_starty2 + proj_len_sqrt

mos_startx2 = np.zeros(mos_startx1.shape[0])  # 90 degree
mos_starty2 = np.arange(-20, 21, 2)
mos_endx2, mos_endy2 = mos_startx2, mos_starty2 + proj_len

# mos_startx4 = np.arange(20, -21, -2)  # 135 degree
# mos_starty4 = np.arange(-20, 21, 2)
# mos_endx4, mos_endy4 = mos_startx4 + proj_len_sqrt, mos_starty4 + proj_len_sqrt


mos_startpos = np.stack([ np.concatenate([mos_startx1, mos_startx2]), np.concatenate([mos_starty1, mos_starty2]) ]).T
mos_endpos = np.stack([ np.concatenate([mos_endx1, mos_endx2]), np.concatenate([mos_endy1, mos_endy2]) ]).T
syn_MosCA3 = np.zeros((xxtun1d_ca3.shape[0], xxtun1d_mos.shape[0], mos_startpos.shape[0]))
mos_act = np.zeros((posvec_mos.shape[0], mos_startpos.shape[0]))
for i in range(mos_startpos.shape[0]):
    print(i)
    syn_MosCA3[:, :, i], mos_act[:, i] = createMosProjMat_p2p(mos_startpos[i, :], mos_endpos[i, :], posvec_mos, posvec_CA3, actmax, actsd)
syn_MosCA3 = np.max(syn_MosCA3, axis=2)
mos_act = np.max(mos_act, axis=1).reshape(*xxtun2d_mos.shape) * actmax


chkpts = (
    (0, 0), (-20, 0), (-10, -4), (-4, 4), (4, -4), (10, 4), (0, 10), (4, -15)
)

mos_chkpts = np.zeros((len(chkpts), 2))

fig, ax = plt.subplots(3, 3, figsize=(16, 12))
ax = ax.ravel()

im0 = ax[0].pcolormesh(xxtun2d_mos, yytun2d_mos, mos_act, shading='auto', vmax=actmax, vmin=0)
plt.colorbar(im0, ax=ax[0])

for axid, (chkpt_x, chkpt_y) in enumerate(chkpts):

    pre_nidxMos = get_nidx_xy_np(chkpt_x, chkpt_y, xxtun1d_mos, yytun1d_mos)

    mos_chkx, mos_chky = xxtun1d_mos[pre_nidxMos], yytun1d_mos[pre_nidxMos]
    mos_chkpts[axid, :] = np.array([mos_chkx, mos_chky])
    post_w_MosCA3 = syn_MosCA3[:, pre_nidxMos].reshape(*xxtun2d_ca3.shape)

    im = ax[axid+1].pcolormesh(xxtun2d_ca3, yytun2d_ca3, post_w_MosCA3, shading='auto', vmax=actmax, vmin=0)
    plt.colorbar(im, ax=ax[axid+1])
    ax[axid+1].scatter(mos_chkx, mos_chky, c='k', marker='o')


for ax_each in ax:
    for i in range(mos_chkpts.shape[0]):
        ax_each.scatter(*mos_chkpts[i, :], marker='x')


# save_pth = join(save_dir, 'Mossy Weights.png')
save_pth = join('plots/', 'Mossy Weights Interference of two paths.png')
fig.savefig(save_pth)

plt.close(fig)

