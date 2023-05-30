# Directional sensory input and synaptic weights
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
config_dict = ph.fig2()
dt = config_dict['dt']

# 0 deg
traj_r = 20
t = np.arange(0, 2e3, dt)
traj_x = np.linspace(-traj_r, traj_r, t.shape[0])
traj_y = np.zeros(traj_x.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF0 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y=traj_y, traj_a =traj_a))


# 180
t = np.arange(0, 2e3, dt)
traj_x = np.linspace(traj_r, -traj_r, t.shape[0])
traj_y = np.zeros(traj_x.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF180 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y=traj_y, traj_a =traj_a))



BehDF_degs = [0, 180]
BehDFs = [BehDF0, BehDF180]
save_dir = join(sim_results_dir, 'fig2')
os.makedirs(save_dir, exist_ok=True)
for BehDF_deg, BehDF in zip(BehDF_degs, BehDFs):
    save_pth = join(save_dir, 'fig2_DirectionalExtrinsic_%d.pkl'%(BehDF_deg))
    print(save_pth)
    simdata = simulate_SNN(BehDF, config_dict)
    save_pickle(save_pth, simdata)
    del simdata
