# Demonstrate the difference in 2D environment between intrinsic (asymmetric recurrence) and extrinsic (STD) model
from os.path import join
import numpy as np
import os
import pandas as pd
from library.comput_utils import cal_hd_np
from library.shared_vars import sim_results_dir
from library.utils import save_pickle, load_pickle
from library.simulation import simulate_SNN
from library.params import ParamsHolder

# # ================================= Network Parameters ==========================================
ph = ParamsHolder()
config_dict_1in = ph.fig1_in()
config_dict_1ex = ph.fig1_ex()
dt = config_dict_1in['dt']



# 0 deg
traj_r = 20
t = np.arange(0, 2e3, dt)
traj_x = np.linspace(-traj_r, traj_r, t.shape[0])
traj_y = np.zeros(traj_x.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF0 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y=traj_y, traj_a =traj_a))

# 45 deg
t = np.arange(0, 2e3*np.sqrt(2), dt)
traj_x = np.linspace(-traj_r, traj_r, t.shape[0])
traj_y = np.linspace(-traj_r, traj_r, t.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF45 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y =traj_y, traj_a =traj_a))

# 90 deg
t = np.arange(0, 2e3, dt)
traj_y = np.linspace(-traj_r, traj_r, t.shape[0])
traj_x = np.zeros(traj_y.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF90 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y =traj_y, traj_a =traj_a))

# 135
t = np.arange(0, 2e3*np.sqrt(2), dt)
traj_x = np.linspace(traj_r, -traj_r, t.shape[0])
traj_y = np.linspace(-traj_r, traj_r, t.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF135 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y =traj_y, traj_a =traj_a))

# 180
t = np.arange(0, 2e3, dt)
traj_x = np.linspace(traj_r, -traj_r, t.shape[0])
traj_y = np.zeros(traj_x.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF180 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y=traj_y, traj_a =traj_a))

# 225
t = np.arange(0, 2e3*np.sqrt(2), dt)
traj_x = np.linspace(traj_r, -traj_r, t.shape[0])
traj_y = np.linspace(traj_r, -traj_r, t.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF225 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y=traj_y, traj_a =traj_a))

# 270
t = np.arange(0, 2e3, dt)
traj_y = np.linspace(traj_r, -traj_r, t.shape[0])
traj_x = np.zeros(traj_y.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF270 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y=traj_y, traj_a =traj_a))

# 315
t = np.arange(0, 2e3*np.sqrt(2), dt)
traj_x = np.linspace(-traj_r, traj_r, t.shape[0])
traj_y = np.linspace(traj_r, -traj_r, t.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF315 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y=traj_y, traj_a =traj_a))


BehDF_degs = [0, 180, 90, 270, 45, 225]
BehDFs = [BehDF0, BehDF180, BehDF90, BehDF270, BehDF45, BehDF225]

save_dir = join(sim_results_dir, 'fig1')
os.makedirs(save_dir, exist_ok=True)

# Intrinsic model
for BehDF_deg, BehDF in zip(BehDF_degs, BehDFs):
    save_pth = join(save_dir, 'fig1_intrinsic_%d.pkl'%(BehDF_deg))
    print(save_pth)
    simdata = simulate_SNN(BehDF, config_dict_1in, store_Activity=False, store_w=False)
    save_pickle(save_pth, simdata)
    del simdata

# Extrinsic
for BehDF_deg, BehDF in zip(BehDF_degs, BehDFs):
    save_pth = join(save_dir, 'fig1_extrinsic_%d.pkl'%(BehDF_deg))
    print(save_pth)
    simdata = simulate_SNN(BehDF, config_dict_1ex, store_Activity=False, store_w=False)
    save_pickle(save_pth, simdata)
    del simdata
