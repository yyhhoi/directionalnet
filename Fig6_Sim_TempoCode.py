# Mossy layer at 0 deg
from os.path import join
import numpy as np
import os
import pandas as pd
from library.comput_utils import cal_hd_np
from library.params import ParamsHolder
from library.shared_vars import sim_results_dir
from library.utils import save_pickle
from library.simulation import simulate_SNN


# # ================================= Network Parameters ==========================================
ph = ParamsHolder()
config_dict = ph.fig6()
dt = config_dict['dt']

# Mossy layer projection trajectory
projl_MosCA3 = 4  # 4
mos_startx0 = np.arange(-20, 21, 2)  # 0 degree
mos_starty0 = np.zeros(mos_startx0.shape[0]) + 20
mos_endx0, mos_endy0 = mos_startx0 + projl_MosCA3, mos_starty0
config_dict['mos_startpos'] = np.stack([mos_startx0, mos_starty0]).T
config_dict['mos_endpos'] = np.stack([mos_endx0, mos_endy0]).T


# # ================================= Traj ==========================================

traj_r = 10
traj_t_each = 1e3  # 1s, 1000ms
max_num_trajtypes = 24
traj_types = np.arange(max_num_trajtypes)
traj_angles = traj_types/max_num_trajtypes * (2*np.pi)
xlist, ylist, tlist, alist, typelist = [], [], [], [], []

traj_t_start, traj_t_end = 0, traj_t_each
for traji in range(-1, max_num_trajtypes):
    traj_type = traji
    traj_angle = traj_angles[traji]
    if traj_type == -1:  # Training set
        start_pt = np.array([0, 0])
        end_pt = np.array([0, 0])
    else:
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
    typelist.extend([traj_type] * traj_t.shape[0])
traj_x_all = np.concatenate(xlist)
traj_y_all = np.concatenate(ylist)
traj_a_all = np.concatenate(alist)
traj_t_all = np.concatenate(tlist)
BehDF_ori = pd.DataFrame(dict(t=traj_t_all, traj_x=traj_x_all, traj_y=traj_y_all, traj_a =traj_a_all,
                              traj_type=typelist))

BehDF_in = BehDF_ori.copy()
BehDF_ex = BehDF_ori.copy()

# Offset trajectory vertically to y=+20 (intrinsic center) and y=-20 (Extrinsic center)
BehDF_in['traj_y'] = BehDF_ori['traj_y'] + 20
BehDF_ex['traj_y'] = BehDF_ori['traj_y'] - 20

# # ============================ Parameter notes =====================================

# Uncomment below if you do not want EC directionality in Training
# config_dict['Ipos_max'] = 2
# config_dict['Iangle_diff'] = 6
# config_dict['Iangle_compen'] = 2
BehDF_ex.loc[BehDF_ex['traj_type'] == -1, 'traj_a'] = np.nan
BehDF_in.loc[BehDF_in['traj_type'] == -1, 'traj_a'] = np.nan

# # ============================ Simulation =====================================
save_dir = join(sim_results_dir, 'fig6')
os.makedirs(save_dir, exist_ok=True)

# Along the DG pathway
save_pth = join(save_dir, 'fig6_in.pkl')
print(save_pth)
simdata = simulate_SNN(BehDF_in, config_dict, store_Activity=False, store_w=False)
save_pickle(save_pth, simdata)
del simdata

# Outside of DG pathway
save_pth = join(save_dir, 'fig6_ex.pkl')
print(save_pth)
simdata = simulate_SNN(BehDF_ex, config_dict, store_Activity=False, store_w=False)
save_pickle(save_pth, simdata)
del simdata

