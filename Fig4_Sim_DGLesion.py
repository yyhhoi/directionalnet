# Mossy layer at 45, 90, 135 degrees

from os.path import join
import numpy as np
import os
import pandas as pd
from library.comput_utils import cal_hd_np
from library.params import ParamsHolder
from library.shared_vars import sim_results_dir
from library.utils import save_pickle
from library.simulation import simulate_SNN

for analysis_tag in ['', '_NoRecurrence']:
    print('Analysis Tag = ', analysis_tag)

    # # ================================= Network Parameters ==========================================

    ph = ParamsHolder()

    if analysis_tag == '_NoRecurrence':
        config_dict_Ctrl = ph.fig4_Ctrl_NoRecurrence()
        config_dict_DGlesion = ph.fig4_DGlesion_NoRecurrence()
    else:
        config_dict_Ctrl = ph.fig4_Ctrl()
        config_dict_DGlesion = ph.fig4_DGlesion()


    dt = config_dict_Ctrl['dt']

    # # ================================= Traj ==========================================

    # Behaviour 0 deg
    traj_r = 20
    t = np.arange(0, 2e3, dt)
    traj_x = np.linspace(-traj_r, traj_r, t.shape[0])
    traj_y = np.zeros(traj_x.shape[0])
    traj_a = cal_hd_np(traj_x, traj_y)
    BehDF0 = pd.DataFrame(dict(t=t, traj_x=traj_x, traj_y=traj_y, traj_a =traj_a))

    # (45, 225), (90, 270), (135, 315)
    projl_MosCA3 = 4
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


    save_dir = join(sim_results_dir, 'fig4%s'%(analysis_tag))
    os.makedirs(save_dir, exist_ok=True)
    for dglabel, config_dict in zip(['Ctrl', 'DGlesion'], [config_dict_Ctrl, config_dict_DGlesion]):

        for mosdeg in (0, 180):
            save_pth = join(save_dir, 'fig4_MossyLayer_Mosdeg%d_%s.pkl'%(mosdeg, dglabel))
            print(save_pth)
            config_dict['mos_startpos'] = np.stack([mosdegdict['startx'][mosdeg], mosdegdict['starty'][mosdeg]]).T
            config_dict['mos_endpos'] = np.stack([mosdegdict['endx'][mosdeg], mosdegdict['endy'][mosdeg]]).T
            simdata = simulate_SNN(BehDF0, config_dict, store_Activity=True, store_w=False)
            save_pickle(save_pth, simdata)
            del simdata
