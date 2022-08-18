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
config_dict['Iangle_compen'] = 0
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

# Mossy layer projection trajectory, 0 deg
projl_MosCA3 = 4
mos_startx0 = np.arange(-20, 21, 2)  # 0 degree
mos_starty0 = np.zeros(mos_startx0.shape[0])
mos_endx0, mos_endy0 = mos_startx0 + projl_MosCA3, mos_starty0

# Mossy layer 180 deg
mos_startx180 = np.arange(20, -21, -2)  # 180 degree
mos_starty180 = np.zeros(mos_startx180.shape[0])
mos_endx180, mos_endy180 = mos_startx180 - projl_MosCA3, mos_starty180

# SD
config_dict['wsd_global'] = 2

# # ================================= Traj ==========================================

dt = config_dict['dt']

# 0 deg
traj_r = 20
t = np.arange(0, 2e3, dt)
traj_x = np.linspace(-traj_r, traj_r, t.shape[0])
traj_y = np.zeros(traj_x.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF0 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y=traj_y, traj_a =traj_a))



save_dir = join('sim_results', 'fig3')

os.makedirs(save_dir, exist_ok=True)

config_dict['mos_startpos'] = np.stack([mos_startx0, mos_starty0]).T
config_dict['mos_endpos'] = np.stack([mos_endx0, mos_endy0]).T
save_pth = join(save_dir, 'fig3_MossyLayer_Mosdeg0.pkl')
print(save_pth)
simdata = simulate_SNN(BehDF0, config_dict)
save_pickle(save_pth, simdata)
del simdata

config_dict['mos_startpos'] = np.stack([mos_startx180, mos_starty180]).T
config_dict['mos_endpos'] = np.stack([mos_endx180, mos_endy180]).T
save_pth = join(save_dir, 'fig3_MossyLayer_Mosdeg180.pkl')
print(save_pth)
simdata = simulate_SNN(BehDF0, config_dict, store_Activity=False, store_w=False)
save_pickle(save_pth, simdata)
del simdata
