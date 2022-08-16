# # Training tempotrons for the simulation
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from library.Tempotron import Tempotron
from library.comput_utils import midedges
from library.script_wrappers import directional_acc_metrics
from library.utils import save_pickle, load_pickle

# ====================================== Global params and paths ==================================
legendsize = 8
plt.rcParams.update({'font.size': legendsize,
                     "axes.titlesize": legendsize,
                     'axes.labelpad': 0,
                     'axes.titlepad': 0,
                     'xtick.major.pad': 0,
                     'ytick.major.pad': 0,

                     })
project_tag = 'Jit100_2ms_gau'
sim_tag = 'fig6_TrainStand_Icompen2a4'
data_dir = 'sim_results/%s/%s'% (sim_tag, project_tag)
save_dir = 'sim_results/%s/%s'% (sim_tag, project_tag)
os.makedirs(save_dir, exist_ok=True)

# ====================================== Train & Test ==================================

fig_enditer, ax_enditer = plt.subplots(3, 1, figsize=(12, 12), dpi=200, facecolor='w', sharex=True, sharey=True,
                                       constrained_layout=True)


exintags = ['ex', 'in']
for exinid, exintag in enumerate(exintags):
    print(exintag)
    dataset = load_pickle(join(data_dir, 'data_train_test_%s_%s.pickle'%(project_tag, exintag)))
    X_test_ori = dataset['X_test_ori']
    Y_test_ori = dataset['Y_test_ori']
    trajtype_test_ori = dataset['trajtype_test_ori']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']
    trajtype_test = dataset['trajtype_test']
    # if exintag == 'ex':
    #     dataset = load_pickle(join(data_dir, 'data_train_test_%s_%s.pickle' % (project_tag, 'in')))
    X_train_ori = dataset['X_train_ori']
    Y_train_ori = dataset['Y_train_ori']
    trajtype_train_ori = dataset['trajtype_train_ori']
    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    trajtype_train = dataset['trajtype_train']

    train_M = X_train.shape[0]
    test_M = X_test.shape[0]
    N = len(X_train[0])
    num_trajtypes = trajtype_test_ori.max()+1
    trajtype_ax = np.arange(num_trajtypes)
    a_ax = trajtype_ax/num_trajtypes*2*np.pi
    deg_ax = np.around(np.rad2deg(a_ax), 0).astype(int)
    deg_ax_half = np.array([deg_ax[i] for i in range(0, num_trajtypes, 2)])

    print('Training set = %d\n Testing set = %d\n Number of neurons %d'%(train_M, test_M, N))


    # Training and testing
    num_iter = 500
    Vthresh = 2
    tau = 5
    tau_s = tau/4
    w_seed = 0
    lr = 0.01
    temN_tax = np.arange(0, 100, 1)
    temN = Tempotron(N=N, lr=lr, Vthresh=Vthresh, tau=tau, tau_s=tau_s, w_seed=w_seed)
    # temN.w = np.load(join(data_dir, 'w_%s_%s.npy'%(project_tag, exintag)))

    ACC_train_list, TPR_train_list, TNR_train_list = [], [], []
    ACC_test_list, TPR_test_list, TNR_test_list = [], [], []
    ACCse_train_list, TPRse_train_list, TNRse_train_list = [], [], []
    ACCse_test_list, TPRse_test_list, TNRse_test_list = [], [], []
    w_train_list = []


    for Y_pred_train, wTMP in temN.train(X_train, Y_train, temN_tax, num_iter=num_iter, progress=True):
        val_train, se_train, firepercent_train = directional_acc_metrics(Y_train, Y_pred_train, trajtype_train, num_trajtypes=num_trajtypes)
        ACC_train_list.append(val_train[0])
        ACCse_train_list.append(se_train[0])
        TPR_train_list.append(val_train[1])
        TPRse_train_list.append(se_train[1])
        TNR_train_list.append(val_train[2])
        TNRse_train_list.append(se_train[2])

    np.save(join(save_dir, 'w_%s_%s.npy'%(project_tag, exintag)), temN.w)

    ACC_train = np.stack(ACC_train_list)
    ACCse_train = np.stack(ACCse_train_list)
    TPR_train = np.stack(TPR_train_list)
    TPRse_train = np.stack(TPRse_train_list)
    TNR_train = np.stack(TNR_train_list)
    TNRse_train = np.stack(TNRse_train_list)


    # # Plot accuracy at the end iteration
    Y_pred_test, _, _ = temN.predict(X_test, temN_tax)
    val_test, se_test, firepercent_test = directional_acc_metrics(Y_test, Y_pred_test, trajtype_test, num_trajtypes=num_trajtypes)
    ACC_test_plot, ACCse_test_plot = val_test[0], se_test[0]
    TPR_test_plot, TPRse_test_plot = val_test[1], se_test[1]
    TNR_test_plot, TNRse_test_plot = val_test[2], se_test[2]
    ax_enditer[0].errorbar(x=deg_ax, y=ACC_test_plot, yerr=ACCse_test_plot, label=exintag)
    ax_enditer[0].set_ylabel('ACC')
    ax_enditer[1].errorbar(x=deg_ax, y=TPR_test_plot, yerr=TPRse_test_plot, label=exintag)
    ax_enditer[1].set_ylabel('TPR')
    ax_enditer[2].errorbar(x=deg_ax, y=TNR_test_plot, yerr=TNRse_test_plot, label=exintag)
    ax_enditer[2].set_ylabel('TNR')
    for ax_i in [0, 1, 2]:
        ax_enditer[ax_i].set_xlabel('Trajectory angle (deg)')
        ax_enditer[ax_i].set_xticks(deg_ax_half)
        ax_enditer[ax_i].set_xticklabels(deg_ax_half.astype(str))
        ax_enditer[ax_i].legend()
        ax_enditer[ax_i].set_yticks(np.arange(0, 1.1, 0.2))
        ax_enditer[ax_i].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
        ax_enditer[ax_i].grid()

    save_pickle(join(save_dir, 'ACC_%s.pickle'%(exintag)), dict(
        ACC_test=ACC_test_plot, ACCse_test=ACCse_test_plot,
        TPR_test=TPR_test_plot, TPRse_test=TPRse_test_plot,
        TNR_test=TNR_test_plot, TNRse_test=TNRse_test_plot,
        firepercent_test=firepercent_test,
    ))

fig_enditer.savefig(join(save_dir, 'ACC_%s.png'%(project_tag)))











