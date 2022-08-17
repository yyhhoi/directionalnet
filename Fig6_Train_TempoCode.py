# # Training tempotrons for the simulation
import os
from os.path import join
import numpy as np
import pandas as pd
from library.Tempotron import Tempotron
from library.script_wrappers import datagen_jitter
from library.utils import save_pickle, load_pickle
import sys

# ====================================== Global params and paths ==================================
jitter_times = int(sys.argv[1])
jitter_ms = float(sys.argv[2])
project_tag = 'Jit%d_%dms_gau'%(jitter_times, jitter_ms)
sim_tag = 'fig6_TrainStand_Icompen2a6'
data_dir = 'sim_results/%s' % sim_tag
save_dir = 'sim_results/%s/%s' % (sim_tag, project_tag)
os.makedirs(save_dir, exist_ok=True)
legendsize = 8
# ======================== Construct training and testing set =================
exintags = ['ex', 'in']

for exintag in exintags:
    print(exintag)
    if exintag == 'in':
        center_x, center_y = 0, 20
    elif exintag == 'ex':
        center_x, center_y = 0, -20
    else:
        raise ValueError

    simdata = load_pickle(join(data_dir, 'fig6_%s.pkl' % exintag))

    BehDF = simdata['BehDF']
    SpikeDF = simdata['SpikeDF']
    NeuronDF = simdata['NeuronDF']
    MetaData = simdata['MetaData']
    config_dict = simdata['Config']

    theta_phase_plot = BehDF['theta_phase_plot']
    traj_x = BehDF['traj_x'].to_numpy()
    traj_y = BehDF['traj_y'].to_numpy()
    traj_a = BehDF['traj_a'].to_numpy()
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
    theta_f = config_dict['theta_f']  # in Hz
    theta_T = 1 / theta_f * 1e3  # in ms
    dt = config_dict['dt']
    traj_d = np.append(0, np.cumsum(np.sqrt(np.diff(traj_x) ** 2 + np.diff(traj_y) ** 2)))

    # Find all the neurons in the input space
    all_nidx = np.where((np.abs(xxtun1d_ca3 - center_x) < 10) & (np.abs(yytun1d_ca3 - center_y) < 10))[0]
    all_nidx = np.sort(all_nidx)

    # Trim down SpikeDF
    SpikeDF['tsp'] = SpikeDF['tidxsp'].apply(lambda x: t[x])
    spdftmplist = []
    for nidx in all_nidx:
        spdftmplist.append(SpikeDF[SpikeDF['neuronid'] == nidx])
    SpikeDF_subset = pd.concat(spdftmplist, ignore_index=True)

    # Loop for theta cycles (patterns)
    data_M = []
    label_M = []
    trajtype = []
    theta_bounds = []
    tmax = t.max()
    overlap_r = 2
    cycle_i = 0
    while (cycle_i * theta_T) < tmax:
        print('\rCurrent cycle %d' % cycle_i, end='', flush=True)

        # Create input data - spikes
        theta_tstart, theta_tend = cycle_i * theta_T, (cycle_i + 1) * theta_T
        spdf_M = SpikeDF_subset[(SpikeDF_subset['tsp'] > theta_tstart) & (SpikeDF_subset['tsp'] <= theta_tend)]
        if spdf_M.shape[0] < 1:
            cycle_i += 1
            continue
        data_MN = []
        for nidx in all_nidx:
            spdf_MN = spdf_M[spdf_M['neuronid'] == nidx]
            tsp = spdf_MN['tsp'].to_numpy() - theta_tstart
            data_MN.append(tsp)
        t_intheta = (t > theta_tstart) & (t <= theta_tend)
        data_M.append(data_MN)

        # Create Labels
        traj_r = np.sqrt((traj_x[t_intheta] - center_x) ** 2 + (traj_y[t_intheta] - center_y) ** 2)
        r05 = np.median(traj_r)
        if r05 < overlap_r:
            label = True
        else:
            label = False
        label_M.append(label)

        # Traj type
        behdf_M = BehDF[(BehDF['t'] > theta_tstart) & (BehDF['t'] <= theta_tend)]
        traj_type = int(behdf_M['traj_type'].median())
        trajtype.append(traj_type)
        theta_bounds.append(np.array([theta_tstart, theta_tend]))

        cycle_i += 1
    print()
    theta_bounds = np.stack(theta_bounds)
    data_M = np.array(data_M, dtype=object)
    trajtype = np.array(trajtype)
    labels = np.array(label_M)

    # # Separate train/test set
    train_idx = np.where(trajtype == -1)[0].astype(int)
    test_idx = np.setdiff1d(np.arange(trajtype.shape[0]), train_idx)
    X_train_ori = data_M[train_idx]
    X_test_ori = data_M[test_idx]
    Y_train_ori = labels[train_idx]
    Y_test_ori = labels[test_idx]
    trajtype_train_ori = trajtype[train_idx]
    trajtype_test_ori = trajtype[test_idx]

    # # Jittering
    X_train, Y_train, trajtype_train, Marr_train, jitbatch_train = datagen_jitter(X_train_ori, Y_train_ori,
                                                                                  trajtype_train_ori, jitter_times,
                                                                                  jitter_ms, 0)
    X_test, Y_test, trajtype_test, Marr_test, jitbatch_test = datagen_jitter(X_test_ori, Y_test_ori,
                                                                             trajtype_test_ori, jitter_times, jitter_ms,
                                                                             0)

    print('Training data = %d' % (X_train_ori.shape[0]))
    print('True = %d \nFalse = %d' % (Y_train_ori.sum(), Y_train_ori.shape[0] - Y_train_ori.sum()))

    print('Testing data = %d' % (X_test_ori.shape[0]))
    print('True = %d \nFalse = %d' % (Y_test_ori.sum(), Y_test_ori.shape[0] - Y_test_ori.sum()))

    print('Training data after jittering = %d' % (X_train.shape[0]))
    print('True = %d \nFalse = %d' % (Y_train.sum(), Y_train.shape[0] - Y_train.sum()))

    print('Testing data after jittering  = %d' % (X_test.shape[0]))
    print('True = %d \nFalse = %d' % (Y_test.sum(), Y_test.shape[0] - Y_test.sum()))

    del simdata

    # ======================== Training and testing =====================================
    N = len(X_train[0])
    num_iter = 500
    Vthresh = 2
    tau = 5
    tau_s = tau / 4
    w_seed = 0
    lr = 0.01
    temN_tax = np.arange(0, 100, 1)
    temN = Tempotron(N=N, lr=lr, Vthresh=Vthresh, tau=tau, tau_s=tau_s, w_seed=w_seed)

    for Y_pred_train, wTMP in temN.train(X_train, Y_train, temN_tax, num_iter=num_iter, progress=True):
        pass
    print()

    Y_pred_test, _, _ = temN.predict(X_test, temN_tax)
    Y_pred_train_ori = temN.predict(X_train_ori, temN_tax)
    Y_pred_test_ori = temN.predict(X_test_ori, temN_tax)

    savedata_jit = dict(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test,
                        Y_pred_train=Y_pred_train, Y_pred_test=Y_pred_test,
                        trajtype_train=trajtype_train, trajtype_test=trajtype_test, all_nidx=all_nidx,
                        theta_bounds=theta_bounds)
    savedata_ori = dict(
        X_train_ori=X_train_ori,
        Y_train_ori=Y_train_ori,
        Y_pred_train_ori=Y_pred_train_ori,
        X_test_ori=X_test_ori,
        Y_test_ori=Y_test_ori,
        Y_pred_test_ori=Y_pred_test_ori,
        trajtype_train_ori=trajtype_train_ori,
        trajtype_test_ori=trajtype_test_ori,
        theta_bounds=theta_bounds,
        all_nidx=all_nidx,
    )

    TrainResult = pd.DataFrame(dict(
        Y_test=Y_test,
        Y_pred_test=Y_pred_test,
        trajtype_test=trajtype_test,
        jitbatch_test=jitbatch_test,
        Marr_test=Marr_test,
    ))

    np.save(join(save_dir, 'w_%s_%s.npy' % (project_tag, exintag)), temN.w)
    save_pickle(join(save_dir, 'data_train_test_%s_%s.pickle' % (project_tag, exintag)), savedata_jit)
    save_pickle(join(save_dir, 'data_train_test_ori_%s_%s.pickle' % (project_tag, exintag)), savedata_ori)
    save_pickle(join(save_dir, 'TrainResult_%s_%s.pickle'% (project_tag, exintag)), TrainResult)
