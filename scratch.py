import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import pandas as pd
from library.Tempotron import Tempotron
from library.comput_utils import acc_metrics
from library.script_wrappers import find_nidx_along_traj, datagen_jitter
from library.utils import save_pickle, load_pickle
from library.visualization import plot_popras


# Plotting and path parameters
legendsize = 8
plt.rcParams.update({'font.size': legendsize,
                     "axes.titlesize": legendsize,
                     'axes.labelpad': 0,
                     'axes.titlepad': 0,
                     'xtick.major.pad': 0,
                     'ytick.major.pad': 0,
                     'lines.linewidth': 1,
                     'figure.figsize': (5.2, 5.5)
                     })

project_tag = 'TrainNoJit_Jit100_3ms'
simdata_dir = 'sim_results/fig6_TrainStand_ExInRun_Icompen2'
data_dir = join(simdata_dir, project_tag)
plot_dir = 'plots/fig6'
os.makedirs(plot_dir, exist_ok=True)

# Load and organize data
for exintag in ['in', 'ex']:
    dataset = load_pickle(join(data_dir, 'data_train_test_%s_%s.pickle'%(project_tag, exintag)))

    new_dataset = dict(
        X_train_ori=dataset['X_train_ori'],
        Y_train_ori=dataset['Y_train_ori'],
        X_test_ori=dataset['X_test_ori'],
        Y_test_ori=dataset['Y_test_ori'],
        trajtype_train_ori=dataset['trajtype_train_ori'],
        trajtype_test_ori=dataset['trajtype_test_ori'],
        theta_bounds=dataset['theta_bounds'],
        all_nidx=dataset['all_nidx']
    )
    save_pickle(join(data_dir, 'data_train_test_ori_%s_%s.pickle'%(project_tag, exintag)), new_dataset)
