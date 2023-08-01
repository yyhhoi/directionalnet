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


analysis_tag = '_Speed_100cms_I2_4_Isd20_Wmos3000_MosProj12_ECtau1000'
print('Analysis Tag = ', analysis_tag)

# # ================================= Network Parameters ==========================================
ph = ParamsHolder()
config_dict = ph.fig3_Speed()

dt = config_dict['dt']

# Mossy layer projection trajectory, 0 deg
projl_MosCA3 = 12
mos_startx0 = np.arange(-40, 41, 2)  # 0 degree
mos_starty0 = np.zeros(mos_startx0.shape[0])
mos_endx0, mos_endy0 = mos_startx0 + projl_MosCA3, mos_starty0

# Mossy layer 180 deg
mos_startx180 = np.arange(40, -41, -2)  # 180 degree
mos_starty180 = np.zeros(mos_startx180.shape[0])
mos_endx180, mos_endy180 = mos_startx180 - projl_MosCA3, mos_starty180

# # ================================= Traj ==========================================


# 0 deg
speed = 100  # cm/s
traj_r = 40
t = np.arange(0, (traj_r*2/speed)*1000, dt)
traj_x = np.linspace(-traj_r, traj_r, t.shape[0])
traj_y = np.zeros(traj_x.shape[0])
traj_a = cal_hd_np(traj_x, traj_y)
BehDF0 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y=traj_y, traj_a =traj_a))



save_dir = join(sim_results_dir, 'fig3%s'%(analysis_tag))

os.makedirs(save_dir, exist_ok=True)

config_dict['mos_startpos'] = np.stack([mos_startx0, mos_starty0]).T
config_dict['mos_endpos'] = np.stack([mos_endx0, mos_endy0]).T
save_pth = join(save_dir, 'fig3_MossyLayer_Mosdeg0.pkl')
print(save_pth)
simdata = simulate_SNN(BehDF0, config_dict, store_Activity=False, store_w=False)
save_pickle(save_pth, simdata)
del simdata

config_dict['mos_startpos'] = np.stack([mos_startx180, mos_starty180]).T
config_dict['mos_endpos'] = np.stack([mos_endx180, mos_endy180]).T
save_pth = join(save_dir, 'fig3_MossyLayer_Mosdeg180.pkl')
print(save_pth)
simdata = simulate_SNN(BehDF0, config_dict, store_Activity=False, store_w=False)
save_pickle(save_pth, simdata)
del simdata
