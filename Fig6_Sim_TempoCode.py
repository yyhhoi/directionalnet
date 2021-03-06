# Mossy layer at 0 deg

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
from library.shared_vars import total_figw
from library.utils import save_pickle, load_pickle
from library.visualization import customlegend
from library.linear_circular_r import rcc
from library.simulation import createMosProjMat_p2p, directional_tuning_tile, simulate_SNN


# # ================================= Network Parameters ==========================================
config_dict = dict()
config_dict['dt'] = 0.1
# # Izhikevich's model
config_dict['izhi_c1'] = 0.04
config_dict['izhi_c2'] = 5
config_dict['izhi_c3'] = 140
config_dict['izhi_a_ex'] = 0.035
config_dict['izhi_b_ex'] = 0.2
config_dict['izhi_c_ex'] = -60
config_dict['izhi_d_ex'] = 8
config_dict['izhi_a_in'] = 0.02
config_dict['izhi_b_in'] = 0.25
config_dict['izhi_c_in'] = -65
config_dict['izhi_d_in'] = 2
config_dict['V_ex'] = 0
config_dict['V_in'] = -80
config_dict['V_thresh'] = 30
config_dict['spdelay'] = int(2/config_dict['dt'])
config_dict['noise_rate'] = 0

# # Theta inhibition
config_dict['theta_amp'] = 7
config_dict['theta_f'] = 10

# Positional drive
config_dict['EC_phase_deg'] = 290
config_dict['Ipos_max'] = 2
config_dict['Iangle_diff'] = 6
config_dict['Ipos_sd'] = 5
config_dict['Iangle_kappa'] = 1
config_dict['ECstf_rest'] = 0
config_dict['ECstf_target'] = 2
config_dict['tau_ECstf'] = 0.5e3
config_dict['U_ECstf'] = 0.001

# Sensory tuning
config_dict['xmin'] = -40
config_dict['xmax'] = 40
config_dict['nx_ca3'] = 80
config_dict['nx_mos'] = 40
config_dict['ymin'] = -40
config_dict['ymax'] = 40
config_dict['ny_ca3'] = 80
config_dict['ny_mos'] = 40
config_dict['nn_inca3'] = 250
config_dict['nn_inmos'] = 250

# Synapse parameters
config_dict['tau_gex'] = 12
config_dict['tau_gin'] = 10
config_dict['U_stdx_CA3'] = 0.7  # 0.7
config_dict['U_stdx_mos'] = 0  # 0.7
config_dict['tau_stdx'] = 0.5e3

# # Weights
# CA3-CA3
config_dict['wmax_ca3ca3'] = 0
config_dict['wmax_ca3ca3_adiff'] = 1500
config_dict['w_ca3ca3_akappa'] = 1
config_dict['asym_flag'] = False

# CA3-Mos and Mos-CA3
config_dict['wmax_ca3mos'] = 0
config_dict['wmax_mosca3'] = 0
config_dict['wmax_ca3mosca3_adiff'] = 3000  # 3000
config_dict['mos_exist'] = True
config_dict['w_ca3mosca3_akappa'] = 1

# CA3-In and In-CA3
config_dict['wmax_CA3in'] = 50
config_dict['wmax_inCA3'] = 5

# Mos-In and In-Mos
config_dict['wmax_Mosin'] = 350
config_dict['wmax_inMos'] = 35

# Mos-Mos, In-In
config_dict['wmax_mosmos'] = 0
config_dict['wmax_inin'] = 0

# Mossy layer projection trajectory
projl_MosCA3 = 4  # 4
mos_startx0 = np.arange(-20, 21, 2)  # 0 degree
mos_starty0 = np.zeros(mos_startx0.shape[0]) + 20
mos_endx0, mos_endy0 = mos_startx0 + projl_MosCA3, mos_starty0
config_dict['mos_startpos'] = np.stack([mos_startx0, mos_starty0]).T
config_dict['mos_endpos'] = np.stack([mos_endx0, mos_endy0]).T
config_dict['wsd_global'] = 2

# # ================================= Traj ==========================================

dt = config_dict['dt']


traj_r = 10
traj_t_each = 1e3  # 1s, 1000ms
traj_angles = np.arange(24)/24 * (2*np.pi)
xlist, ylist, tlist, alist = [], [], [], []
typelist = []
traj_t_start, traj_t_end = 0, traj_t_each
for traji, traj_angle in enumerate(traj_angles):
    traj_vec = np.array([np.cos(traj_angle), np.sin(traj_angle)])
    start_pt = -traj_r * traj_vec
    end_pt = traj_r * traj_vec
    traj_t = np.arange(traj_t_start, traj_t_end, dt)
    traj_x = np.linspace(start_pt[0], end_pt[0], traj_t.shape[0])
    traj_y = np.linspace(start_pt[1], end_pt[1], traj_t.shape[0])
    traj_a = cal_hd_np(traj_x, traj_y)
    traj_t_start += traj_t_each
    traj_t_end += traj_t_each
    xlist.append(traj_x)
    ylist.append(traj_y)
    tlist.append(traj_t)
    alist.append(traj_a)
    typelist.extend([traji] * traj_t.shape[0])
traj_x_all = np.concatenate(xlist)
traj_y_all = np.concatenate(ylist)
traj_a_all = np.concatenate(alist)
traj_t_all = np.concatenate(tlist)
BehDF0 = pd.DataFrame(dict(t=traj_t_all, traj_x=traj_x_all, traj_y=traj_y_all, traj_a =traj_a_all,
                           traj_type=typelist))

# # ============================ Parameter notes =====================================
# Below is for WITHOUT directional tuning
config_dict['wmax_ca3ca3'] = 500
config_dict['wmax_ca3mos'] = 500
config_dict['wmax_mosca3'] = 500



# Below is for WITH directional tuning
# config_dict['wmax_ca3ca3_adiff'] = 1500  # 1500
# config_dict['wmax_ca3mosca3_adiff'] = 4500  # 3000


# # ============================ Simulation =====================================
save_dir = join('sim_results', 'fig6_NoAngle')
os.makedirs(save_dir, exist_ok=True)

# Along the DG pathway
BehDF0['traj_y'] = BehDF0['traj_y'] + 20
save_pth = join(save_dir, 'fig6_in.pkl')
print(save_pth)
simdata = simulate_SNN(BehDF0, config_dict, store_Activity=False, store_w=False)
save_pickle(save_pth, simdata)
del simdata

# Outside of DG pathway
BehDF0['traj_y'] = BehDF0['traj_y'] - 20 - 20
save_pth = join(save_dir, 'fig6_ex.pkl')
print(save_pth)
simdata = simulate_SNN(BehDF0, config_dict, store_Activity=False, store_w=False)
save_pickle(save_pth, simdata)
del simdata
# ===================== Sanity check ============================

# # Plot the mos conenctions
# fig, ax = plt.subplots(figsize=(5, 5))
# for mos_start in mos_startlist:
#
#     ax.plot(mos_start[:, 0], mos_start[:, 1])
#     ax.plot(mos_start[-1, 0], mos_start[-1, 1], marker='o')
#
# ax.set_xlim(-20, 20)
# ax.set_ylim(-20, 20)
# fig.savefig(join(save_dir, 'MosConfig.png'))
