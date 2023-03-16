from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pycircstat import mean as cmean, cdiff
from library.comput_utils import circular_density_1d, rcc_wrapper
from library.script_wrappers import best_worst_analysis, exin_analysis
from library.utils import load_pickle
from library.visualization import customlegend, plot_sca_onsetslope, plot_exin_bestworst_simdissim

# ====================================== Global params and paths ==================================
legendsize = 8
load_dir = 'sim_results/FigNWGtalk'
save_dir = 'plots/FigNWGtalk/'
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

# ====================================== Figure initialization ==================================

fig, ax = plt.subplots(1, 3, figsize=(12, 3))

# ======================================Analysis and plotting ==================================
mosdegs = np.arange(0, 360, 45)

onsets = np.zeros(mosdegs.shape)
slopes = np.zeros(mosdegs.shape)
marphases = np.zeros(mosdegs.shape)

for mosi, mosdeg in enumerate(mosdegs):
    print('Mos deg %d' % mosdeg)
    # ======================== Get data =================
    simdata = load_pickle(join(load_dir, 'FigNWGtalk_MossyLayer_Mosdeg%d.pkl' % (mosdeg)))
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

    Ipos_max_compen = config_dict['Ipos_max_compen']
    Iangle_diff = config_dict['Iangle_diff']
    Iangle_kappa = config_dict['Iangle_kappa']
    xmin, xmax, ymin, ymax = config_dict['xmin'], config_dict['xmax'], config_dict['ymin'], config_dict['ymax']
    traj_d = np.append(0, np.cumsum(np.sqrt(np.diff(traj_x)**2 + np.diff(traj_y)**2)))


    nid_min = np.argmin(xxtun1d_ca3 ** 2 + yytun1d_ca3 ** 2)

    tidxsp_tmp = SpikeDF.loc[SpikeDF['neuronid'] == nid_min, 'tidxsp'].to_numpy()
    tsp_eg, phasesp_eg = t[tidxsp_tmp], theta_phase[tidxsp_tmp]
    dsp_eg = traj_d[tidxsp_tmp]
    (dsp_norm, phasesp), (onset, slope), _ = rcc_wrapper(dsp_eg, phasesp_eg)

    onsets[mosi] = onset
    slopes[mosi] = slope * np.pi
    marphases[mosi] = cmean(phasesp)


ax[0].plot(mosdegs, onsets, marker='x')
ax[1].plot(mosdegs, slopes, marker='x')
ax[2].plot(mosdegs, marphases, marker='x')

for axeach in ax:
    axeach.axvline(np.rad2deg(aatun1d_ca3[nid_min]), c='r')
fig.savefig(join(save_dir, 'FigNWG.png'), dpi=300)
fig.savefig(join(save_dir, 'FigNWG.pdf'))
fig.savefig(join(save_dir, 'FigNWG.svg'))