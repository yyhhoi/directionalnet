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
project_tag = 'TrainNoJit_Jit200_5ms'
data_dir = 'sim_results/fig6_TrainStand_ExInRun_Icompen2/' + project_tag
save_dir = 'sim_results/fig6_TrainStand_ExInRun_Icompen2/' + project_tag
os.makedirs(save_dir, exist_ok=True)

pred_perIter = False
# ====================================== Train & Test ==================================

if pred_perIter:
    fig_metrics, ax_metrics = plt.subplots(24, 2, figsize=(12, 30), dpi=200, facecolor='w', sharex='col', sharey=True,
                                           constrained_layout=True)
fig_enditer, ax_enditer = plt.subplots(2, 1, figsize=(12, 12), dpi=200, facecolor='w', sharex=True, sharey=True,
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
    # temN.w = np.load(join(save_dir, 'w_%s_%s.npy'%(project_tag, exintag)))

    ACC_train_list, TPR_train_list, TNR_train_list = [], [], []
    ACC_test_list, TPR_test_list, TNR_test_list = [], [], []
    ACCse_train_list, TPRse_train_list, TNRse_train_list = [], [], []
    ACCse_test_list, TPRse_test_list, TNRse_test_list = [], [], []
    w_train_list = []


    for Y_pred_train, wTMP in temN.train(X_train, Y_train, temN_tax, num_iter=num_iter, progress=True):
        val_train, se_train = directional_acc_metrics(Y_train, Y_pred_train, trajtype_train, num_trajtypes=num_trajtypes)
        ACC_train_list.append(val_train[0])
        ACCse_train_list.append(se_train[0])
        TPR_train_list.append(val_train[1])
        TPRse_train_list.append(se_train[1])
        TNR_train_list.append(val_train[2])
        TNRse_train_list.append(se_train[2])

        # Test result per iteration
        if pred_perIter:
            Y_pred_test, _, _, = temN.predict(X_test, temN_tax)
            val_test, se_test = directional_acc_metrics(Y_test, Y_pred_test, trajtype_test, num_trajtypes=num_trajtypes)
            ACC_test_list.append(val_test[0])
            ACCse_test_list.append(se_test[0])
            TPR_test_list.append(val_test[1])
            TPRse_test_list.append(se_test[1])
            TNR_test_list.append(val_test[2])
            TNRse_test_list.append(se_test[2])
            w_train_list.append(wTMP)
    np.save(join(save_dir, 'w_%s_%s.npy'%(project_tag, exintag)), temN.w)

    ACC_train = np.stack(ACC_train_list)
    ACCse_train = np.stack(ACCse_train_list)
    TPR_train = np.stack(TPR_train_list)
    TPRse_train = np.stack(TPRse_train_list)
    TNR_train = np.stack(TNR_train_list)
    TNRse_train = np.stack(TNRse_train_list)

    if pred_perIter:
        ACC_test = np.stack(ACC_test_list)
        ACCse_test = np.stack(ACCse_test_list)
        TPR_test = np.stack(TPR_test_list)
        TPRse_test = np.stack(TPRse_test_list)
        TNR_test = np.stack(TNR_test_list)
        TNRse_test = np.stack(TNRse_test_list)
        all_w_train = np.stack(w_train_list)
        # #  Plot accuracy per iteration
        iter_ax = np.arange(ACCse_train.shape[0]) + 1

        for deg_i in range(-1, ax_metrics.shape[0]):  # -1 is the training set

            if deg_i == -1:
                ax_metrics[0, exinid].errorbar(x=iter_ax, y=TPR_train[:, deg_i], yerr=TPRse_train[:, deg_i], label='Train TPR', color='r')
                ax_metrics[0, exinid].errorbar(x=iter_ax, y=TNR_train[:, deg_i], yerr=TNRse_train[:, deg_i], label='Train TNR', color='b')

            else:
                trajdeg = deg_ax[deg_i]
                ax_metrics[deg_i, exinid].errorbar(x=iter_ax, y=TPR_test[:, deg_i], yerr=TPRse_test[:, deg_i], label='Test TPR', color='lightcoral')
                ax_metrics[deg_i, exinid].errorbar(x=iter_ax, y=TNR_test[:, deg_i], yerr=TNRse_test[:, deg_i], label='Test TNR', color='skyblue')
                ax_metrics[deg_i, exinid].set_ylabel('Test %d deg'%(deg_ax[deg_i]))
                ax_metrics[deg_i, exinid].set_yticks(np.arange(0, 1.1, 0.2))
                ax_metrics[deg_i, exinid].set_yticks(np.arange(0, 1.1, 0.2), minor=True)
                ax_metrics[deg_i, exinid].grid()
        ax_metrics[0, exinid].legend()


    # # Plot accuracy at the end iteration
    if pred_perIter:
        TPR_test_plot, TPRse_test_plot = TPR_test[-1, :], TPRse_test[-1, :]
        TNR_test_plot, TNRse_test_plot = TNR_test[-1, :], TNRse_test[-1, :]
    else:
        Y_pred_test, _, _, = temN.predict(X_test, temN_tax)
        val_test, se_test = directional_acc_metrics(Y_test, Y_pred_test, trajtype_test, num_trajtypes=num_trajtypes)
        TPR_test_plot, TPRse_test_plot = val_test[1], se_test[1]
        TNR_test_plot, TNRse_test_plot = val_test[2], se_test[2]

    ax_enditer[0].errorbar(x=deg_ax, y=TPR_test_plot, yerr=TPRse_test_plot, label=exintag)
    ax_enditer[0].set_ylabel('TPR')
    ax_enditer[1].errorbar(x=deg_ax, y=TNR_test_plot, yerr=TNRse_test_plot, label=exintag)
    ax_enditer[1].set_ylabel('TNR')
    for ax_i in [0, 1]:
        ax_enditer[ax_i].set_xlabel('Trajectory angle (deg)')
        ax_enditer[ax_i].set_xticks(deg_ax_half)
        ax_enditer[ax_i].set_xticklabels(deg_ax_half.astype(str))
        ax_enditer[ax_i].legend()
        ax_enditer[ax_i].set_yticks(np.arange(0, 1.1, 0.2))
        ax_enditer[ax_i].set_yticks(np.arange(0, 1.1, 0.1), minor=True)
        ax_enditer[ax_i].grid()


if pred_perIter:
    ax_metrics[0, 0].set_title('Extrinsic')
    ax_metrics[0, 1].set_title('Intrinsic')
    fig_metrics.savefig(join(save_dir, 'Metrics_%s.png'%(project_tag)))
fig_enditer.savefig(join(save_dir, 'ACC_%s.png'%(project_tag)))











