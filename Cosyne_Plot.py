from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pycircstat import mean as cmean, resultant_vector_length
from library.comput_utils import circular_density_1d
from library.script_wrappers import best_worst_analysis, exin_analysis
from library.utils import load_pickle
from library.visualization import customlegend, plot_sca_onsetslope, plot_exin_bestworst_simdissim

# ====================================== Global params and paths ==================================
legendsize = 8
load_dir = 'sim_results/Cosyne'
save_dir = 'plots/Cosyne/'
os.makedirs(save_dir, exist_ok=True)
plt.rcParams.update({'font.size': legendsize,
                     "axes.titlesize": legendsize,
                     'axes.labelpad': 0,
                     'axes.titlepad': 0,
                     'xtick.major.pad': 0,
                     'ytick.major.pad': 0,
                     'lines.linewidth': 1,
                     'figure.figsize': (5.2, 5.5),
                     'figure.dpi': 300,
                     'axes.spines.top': False,
                     'axes.spines.right': False,
                     })


# ======================================Analysis and plotting ==================================


mosdegs = [45, 90, 225]
direct_c = ['tomato', 'royalblue']


for mosi, mosdeg in enumerate(mosdegs):
    simdata = load_pickle(join(load_dir, 'Cosyne_Mosdeg%d.pkl' % (mosdeg)))

    # ======================== Get data =================
    BehDF = simdata['BehDF']
    SpikeDF = simdata['SpikeDF']
    NeuronDF = simdata['NeuronDF']
    ActivityData = simdata['ActivityData']
    MetaData = simdata['MetaData']
    config_dict = simdata['Config']

    theta_phase_plot = BehDF['theta_phase_plot']
    traj_x = BehDF['traj_x'].to_numpy()
    traj_y = BehDF['traj_y'].to_numpy()
    t = BehDF['t'].to_numpy()
    theta_phase = BehDF['theta_phase'].to_numpy()

    nn_ca3 = MetaData['nn_ca3']

    xxtun1d = NeuronDF['neuronx'].to_numpy()
    yytun1d = NeuronDF['neurony'].to_numpy()
    aatun1d = NeuronDF['neurona'].to_numpy()

    xxtun1d_ca3 = xxtun1d[:nn_ca3]
    yytun1d_ca3 = yytun1d[:nn_ca3]
    aatun1d_ca3 = aatun1d[:nn_ca3]
    nx_ca3, ny_ca3 = config_dict['nx_ca3'], config_dict['ny_ca3']
    xxtun2d_ca3 = xxtun1d_ca3.reshape(nx_ca3, nx_ca3)  # Assuming nx = ny
    yytun2d_ca3 = yytun1d_ca3.reshape(nx_ca3, nx_ca3)  # Assuming nx = ny
    aatun2d_ca3 = aatun1d_ca3.reshape(nx_ca3, nx_ca3)  # Assuming nx = ny
    nidx_2d_ca3 = np.arange(nn_ca3).reshape(nx_ca3, nx_ca3)


    Ipos_max_compen = config_dict['Ipos_max_compen']
    Iangle_diff = config_dict['Iangle_diff']
    Iangle_kappa = config_dict['Iangle_kappa']
    xmin, xmax, ymin, ymax = config_dict['xmin'], config_dict['xmax'], config_dict['ymin'], config_dict['ymax']
    traj_d = np.append(0, np.cumsum(np.sqrt(np.diff(traj_x)**2 + np.diff(traj_y)**2)))



    # ======================== (Cosyne) Instantenous phases at tidx=10000 ===================================

    fig_instaphase, ax_instaphase = plt.subplots(figsize=(6, 4))

    spdf_the = SpikeDF[(SpikeDF['tidxsp'] >= 9000) & (SpikeDF['tidxsp'] < 10000) & (SpikeDF['neuronid'] < nn_ca3)]

    uni_neuronids = spdf_the['neuronid'].unique().astype(int)
    uni_x = xxtun1d[uni_neuronids]
    uni_y = yytun1d[uni_neuronids]
    uni_phase = np.zeros(uni_neuronids.shape[0])
    for uniidx, uni_nid in enumerate(uni_neuronids):
        tidxsp_unithe = spdf_the[spdf_the['neuronid'] == uni_nid]['tidxsp']
        uni_phase[uniidx] = cmean(theta_phase[tidxsp_unithe])
    im = ax_instaphase.scatter(uni_x, uni_y, c=uni_phase, cmap='hsv', s=80, marker='.')
    plt.colorbar(im, ax=ax_instaphase)
    ax_instaphase.scatter(traj_x[9500], traj_y[9500], marker='x', s=40, c='k')
    ax_instaphase.set_xlim()
    # vec_idxs = []
    # vec_cmean = []
    # vec_R = []
    #
    # for uniidx, uni_nid in enumerate(uni_neuronids):
    #
    #     rowids, colids = np.where(nidx_2d_ca3 == uni_nid)
    #     rowid, colid = rowids[0], colids[0]
    #
    #     nid1 = nidx_2d_ca3[rowid, colid + 1]
    #     nid2 = nidx_2d_ca3[rowid, colid - 1]
    #     nid3 = nidx_2d_ca3[rowid + 1, colid]
    #     nid4 = nidx_2d_ca3[rowid - 1, colid]
    #     nid5 = nidx_2d_ca3[rowid + 1, colid + 1]
    #     nid6 = nidx_2d_ca3[rowid - 1, colid + 1]
    #     nid7 = nidx_2d_ca3[rowid + 1, colid - 1]
    #     nid8 = nidx_2d_ca3[rowid - 1, colid - 1]
    #
    #     neigh_idxs = [nid1, nid2, nid3, nid4, nid5, nid6, nid7, nid8]
    #
    #     Checkneighs = [True if neigh_idx in uni_neuronids else False for neigh_idx in neigh_idxs]
    #
    #
    #     if np.sum(Checkneighs) < 8:
    #         continue
    #     else:
    #         phase_this = uni_phase[uniidx]
    #         neigh_uniidx = [np.where(uni_neuronids == neigh_idx)[0][0]  for neigh_idx in neigh_idxs]
    #         neigh_phases = uni_phase[neigh_uniidx]
    #         neigh_xs = uni_x[neigh_uniidx]
    #         neigh_ys = uni_y[neigh_uniidx]
    #         neigh_vecs = (neigh_xs + 1j*neigh_ys) - (uni_x[uniidx] + 1j*uni_y[uniidx])
    #         neigh_as = np.angle(neigh_vecs)
    #         cos_sim = (np.cos(phase_this - neigh_phases) + 1)/2
    #         dist = 1 - cos_sim
    #         # dist = (phase_this - neigh_phases)/np.pi
    #         mean_neigh_angle = cmean(neigh_as, w=dist)
    #         mean_neigh_R = resultant_vector_length(neigh_as, w=dist)
    #
    #         vec_idxs.append(uniidx)
    #         vec_cmean.append(mean_neigh_angle)
    #         vec_R.append(mean_neigh_R)



    # vec_dx = np.cos(vec_cmean) * np.array(vec_R)
    # vec_dy = np.sin(vec_cmean) * np.array(vec_R)
    # ax_instaphase.quiver(uni_x[vec_idxs], uni_y[vec_idxs], vec_dx, vec_dy, color='k')

    fig_instaphase.savefig(join(save_dir, 'Insta_phase_%d.png'%(mosdeg)), dpi=200)



