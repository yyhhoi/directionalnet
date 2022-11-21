# Mossy layer at 45, 90, 135 degrees

from os.path import join
import numpy as np
import os
import pandas as pd
from library.comput_utils import cal_hd_np
from library.utils import save_pickle
from library.simulation import simulate_SNN


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
config_dict['Iangle_compen'] = 3
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
config_dict['U_stdx_mos'] = 0
config_dict['tau_stdx'] = 0.5e3

# # Weights
# CA3-CA3
config_dict['wmax_ca3ca3'] = 0
config_dict['wmax_ca3ca3_adiff'] = 3000
config_dict['w_ca3ca3_akappa'] = 1
config_dict['asym_flag'] = False

# CA3-Mos and Mos-CA3
config_dict['wmax_ca3mos'] = 0
config_dict['wmax_mosca3'] = 0
config_dict['wmax_ca3mosca3_adiff'] = 3000
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

# SD
config_dict['wsd_global'] = 2

# # ================================= Traj ==========================================

dt = config_dict['dt']

# Behaviour 0 deg
traj_r = 20
t = np.arange(0, 2e3, dt)
traj_x = np.linspace(-traj_r, traj_r, t.shape[0])
traj_y = np.zeros(traj_x.shape[0])
traj_a = cal_hd_np(traj_x, traj_y) * np.nan
BehDF0 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y=traj_y, traj_a =traj_a))

# (45, 225), (90, 270), (135, 315)
projl_MosCA3 = 5
mosdegdict = dict(startx={}, starty={}, endx={}, endy={})

mosdegdict['startx'][45] = np.arange(-20, 21, 2)  # 45 degree
mosdegdict['starty'][45] = np.arange(-20, 21, 2)
mosdegdict['endx'][45], mosdegdict['endy'][45] = mosdegdict['startx'][45] + projl_MosCA3, mosdegdict['starty'][45] + projl_MosCA3

mosdegdict['startx'][225] = np.arange(20, -21, -2)  # 225 degree
mosdegdict['starty'][225] = np.arange(20, -21, -2)
mosdegdict['endx'][225], mosdegdict['endy'][225] = mosdegdict['startx'][225] - projl_MosCA3, mosdegdict['starty'][225] - projl_MosCA3

mosdegdict['starty'][90] = np.arange(-20, 21, 2)  # 90 degree
mosdegdict['startx'][90] = np.zeros(mosdegdict['starty'][90].shape[0])
mosdegdict['endx'][90], mosdegdict['endy'][90] = mosdegdict['startx'][90], mosdegdict['starty'][90] + projl_MosCA3

mosdegdict['starty'][270] = np.arange(20, -21, -2)  # 270 degree
mosdegdict['startx'][270] = np.zeros(mosdegdict['starty'][270].shape[0])
mosdegdict['endx'][270], mosdegdict['endy'][270] = mosdegdict['startx'][270], mosdegdict['starty'][270] - projl_MosCA3

mosdegdict['startx'][0] = np.arange(-20, 21, 2)  # 0 degree
mosdegdict['starty'][0] = np.zeros(mosdegdict['startx'][0].shape[0])
mosdegdict['endx'][0], mosdegdict['endy'][0] = mosdegdict['startx'][0] + projl_MosCA3, mosdegdict['starty'][0]

mosdegdict['startx'][180] = np.arange(20, -21, -2)  # 180 degree
mosdegdict['starty'][180] = np.zeros(mosdegdict['startx'][180].shape[0])
mosdegdict['endx'][180], mosdegdict['endy'][180] = mosdegdict['startx'][180] - projl_MosCA3, mosdegdict['starty'][180]

config_dict['noise_rate'] = 0

save_dir = join('sim_results', 'Cosyne')
os.makedirs(save_dir, exist_ok=True)
for mosdeg in (45, 90, 225):
    save_pth = join(save_dir, 'Cosyne_Mosdeg%d.pkl'%(mosdeg))
    print(save_pth)
    config_dict['mos_startpos'] = np.stack([mosdegdict['startx'][mosdeg], mosdegdict['starty'][mosdeg]]).T
    config_dict['mos_endpos'] = np.stack([mosdegdict['endx'][mosdeg], mosdegdict['endy'][mosdeg]]).T
    simdata = simulate_SNN(BehDF0, config_dict, store_Activity=False, store_w=False)
    save_pickle(save_pth, simdata)
    del simdata
