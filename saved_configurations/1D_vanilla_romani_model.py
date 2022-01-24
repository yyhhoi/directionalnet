# 1D track, with correct time constants ~0.5s for STD and STF
# about 0.5 phase diff, -2 rad slope

from os.path import join

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from pycircstat import cdiff, mean as cmean

from library.comput_utils import cal_hd_np, get_nidx_1d_np
from library.visualization import customlegend
from library.linear_circular_r import rcc
import time
mpl.rcParams['figure.dpi'] = 150
legendsize = 7

def gaufunc(x, mu, sd, outmax):
    return outmax * np.exp(-np.square(x - mu)/(sd**2))

def squarefunc(x, mu, radius, amp):
    senamp = amp * (np.abs(x - mu) < radius)
    return senamp

def parafunc(x, mu, radius, slope, amp):
    pass

def plot_sequences(SpikeDF, BehDF, ActivityData, NeuronDF, MetaData, egnidx, popnidxs, fig, ax, axbig, tag):

    # Get data
    traj_x = BehDF['x']
    t = BehDF['t']
    theta_phase = BehDF['theta_phase']
    theta_phase_plot = BehDF['theta_phase_plot']
    Isyn_pop = ActivityData['Isyn']
    Itotal_pop = ActivityData['Itotal']
    Isen_pop = ActivityData['Isen']
    Isen_fac_pop = ActivityData['Isen_fac']
    syneff_pop = ActivityData['syneff']
    ECstf_pop = ActivityData['ECstf']
    xxtun1d, aatun1d = NeuronDF['neuronx'].to_numpy(), NeuronDF['neurona'].to_numpy()
    nn, nn_ca3 = MetaData['nn'], MetaData['nn_ca3']
    EC_phase = MetaData['EC_phase']

    # Phase precession
    tidxsp_eg = SpikeDF.loc[SpikeDF['neuronid']==egnidx, 'tidxsp']
    tsp_eg, phasesp_eg = t[tidxsp_eg], theta_phase[tidxsp_eg]
    xsp_eg = traj_x[tidxsp_eg]
    mean_phasesp = cmean(phasesp_eg)
    xspmin, xsprange = xsp_eg.min(), xsp_eg.max() - xsp_eg.min()
    xsp_norm_eg = (xsp_eg-xspmin)/xsprange
    regress = rcc(xsp_norm_eg, phasesp_eg, abound=(-1., 1.))
    rcc_c, rcc_m = regress['phi0'], regress['aopt']
    rcc_slope_rad = 2*np.pi*rcc_m
    xdum = np.linspace(xsp_norm_eg.min(), xsp_norm_eg.max(), 100)
    ydum = xdum * rcc_m * 2 * np.pi + rcc_c
    ax[0].scatter(xsp_norm_eg, phasesp_eg, marker='|', s=4)
    ax[0].axhline(mean_phasesp, xmin=0, xmax=0.3, linewidth=1)
    ax[0].plot(xdum, ydum, c='k', linewidth=0.75)
    ax[0].annotate('y= %0.2fx + %0.2f'%(rcc_slope_rad, rcc_c), xy=(0.3, 0.9), xycoords='axes fraction', fontsize=legendsize+4)
    ax[0].set_ylim(0, 2*np.pi)
    # ax[0].set_title('Phase range = %0.2f - %0.2f (%0.2f) '%(phasesp_eg.min(), phasesp_eg[0], phasesp_eg[0]-phasesp_eg.min()), fontsize=legendsize)
    ax[0].set_yticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])
    ax[0].set_yticklabels(['0', '$\pi/2$', '$\pi$', '$1.5\pi$', '$2\pi$'])

    # current
    ax[1].plot(t, Itotal_pop[:, egnidx], label='Itotal', linewidth=0.75)
    ax[1].plot(t, Isyn_pop[:, egnidx], label='Isyn', linewidth=0.75)
    ax[1].set_ylim(-25, 20)
    customlegend(ax[1], fontsize=legendsize)
    axtsp10 = ax[1].twinx()
    axtsp10.eventplot(tsp_eg, lineoffsets = 1, linelength=0.05, linewidths=0.5, color='r')
    axtsp10.set_ylim(0, 1.2)
    axtsp10.axis('off')

    # STD
    ax[2].plot(t, Isyn_pop[:, egnidx], label='Isyn', linewidth=0.75, color='orange')
    ax[2].plot(t, Isen_pop[:, egnidx], label='Isen', linewidth=0.75, color='blue')
    ax[2].plot(t, Isen_fac_pop[:, egnidx], label='Isen_fac', linewidth=0.75, color='cyan')
    ax[2].set_ylim(-5, 40)
    customlegend(ax[2], fontsize=legendsize, loc='upper left')
    axsyneff = ax[2].twinx()
    axsyneff.plot(t, syneff_pop[:, egnidx], label='snyeff', color='r', linewidth=0.75)
    axsyneff.plot(t, ECstf_pop[:, egnidx], label='ECSTF', color='green', linewidth=0.75)
    axsyneff.set_ylim(-0.1, 2.1)
    customlegend(axsyneff, fontsize=legendsize)
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for i in theta_cutidx:
        ax[1].axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)
        ax[2].axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)
    for ax_each in ax.ravel():
        ax_each.tick_params(labelsize=legendsize)
    axsyneff.tick_params(labelsize=legendsize)

    # Population raster
    dxtun = xxtun1d[1] - xxtun1d[0]
    tt, traj_xx = np.meshgrid(t, xxtun1d[popnidxs])
    mappable = axbig.pcolormesh(tt, traj_xx, syneff_pop[:, popnidxs].T, shading='auto', vmin=0, vmax=2, cmap='seismic')
    for neuronid in popnidxs:
        tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
        tsp_neuron = t[tidxsp_neuron]
        neuronx = xxtun1d[neuronid]
        if neuronid == egnidx:
            ras_c = 'r'
        else:
            ras_c = 'green'
        axbig.eventplot(tsp_neuron, lineoffsets=neuronx, linelengths=dxtun, linewidths=0.75, color=ras_c)
    # Population raster - Inhibitory
    neuron_in_xmin, neuron_in_xmax = traj_x.max(), traj_x.max()*1.1
    dx_in = (neuron_in_xmax - neuron_in_xmin)/(nn-nn_ca3)
    for neuronid in np.arange(nn_ca3, nn, 1):
        tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
        tsp_neuron = t[tidxsp_neuron]
        plot_onset = neuron_in_xmin+(neuronid-nn_ca3) * dx_in

        axbig.eventplot(tsp_neuron, lineoffsets=plot_onset, linelengths=dx_in, linewidths=0.75, color='m')

    axbig.plot(t, traj_x, c='k', linewidth=0.75)
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for i in theta_cutidx:
        axbig.axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)

    axbig.annotate('%s direction\nGray lines = Theta phase 0\nEC phase shift = %d deg'%(tag, np.rad2deg(EC_phase)),
                   xy=(0.02, 0.75), xycoords='axes fraction', fontsize=12)
    axbig.tick_params(labelsize=legendsize)

    cbar = fig.colorbar(mappable, ax=axbig)
    cbar.set_label('Syn Efficacy', rotation=90)

    stat_dict = dict(mean_phasesp=mean_phasesp, slope=rcc_slope_rad, onset=rcc_c)
    return stat_dict

def params_search(save_pth):

    # Environment & agent
    dt = 0.1  # 0.1ms
    t = np.arange(0, 2e3, dt)
    traj_x = t * 5 * 1e-3
    traj_y = np.zeros(traj_x.shape[0]) * 0.0
    traj_a = cal_hd_np(traj_x, traj_y)

    # Izhikevich's model parameters
    izhi_a, izhi_b, izhi_c, izhi_d = 0.02, 0.2, -50, 2  # CH
    V_ca3, V_in = 60, -80
    V_thresh = 30
    tau_deadx = 10
    deadtime = 0  # in ms
    deadx_thresh = np.exp(-deadtime/tau_deadx) + 1e-4

    # Theta inhibition
    theta_amp = 7
    theta_f = 10
    theta_T = 1/theta_f * 1e3
    theta_phase = np.mod(t, theta_T)/theta_T * 2*np.pi
    theta_phase_plot = np.mod(theta_phase + 2*np.pi, 2*np.pi)
    Itheta = (1 + np.cos(theta_phase))/2 * theta_amp

    # Positional drive
    EC_phase = np.deg2rad(290)
    Ipos_max = 10  # CH: 10
    Iangle_diff = 20  # CH: 20
    Ipos_sd = 1
    ECstf_rest, ECstf_target = 0, 2  # 0, 2
    tau_ECstf = 0.5e3
    U_ECstf = 0.001  # 0.001
    Ipos_max_compen = Ipos_max + (np.cos(EC_phase) + 1)/2 * theta_amp

    # Sensory tuning
    xmin, xmax, nx = traj_x.min(), traj_x.max(), 100
    xtun_ca3 = np.linspace(traj_x.min(), traj_x.max(), nx)
    atun_ca3 = np.deg2rad(np.array([0, 180]))
    xxtun2d_ca3, aatun2d_ca3 = np.meshgrid(xtun_ca3, atun_ca3)
    xxtun1d_ca3, aatun1d_ca3 = xxtun2d_ca3.flatten(), aatun2d_ca3.flatten()
    nn_ca3 = xxtun1d_ca3.shape[0]
    nn_in = 100
    xxtun1d_in, aatun1d_in = np.zeros(nn_in), np.zeros(nn_in)
    xxtun1d = np.concatenate([xxtun1d_ca3, xxtun1d_in])
    aatun1d = np.concatenate([aatun1d_ca3, aatun1d_in])
    nn = xxtun1d.shape[0]

    # # Weights
    wmax_ca3ca3 = 3 # 5
    wsd_ca3ca3 = 2
    w_ca3ca3 = gaufunc(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_ca3.reshape(nn_ca3, 1), wsd_ca3ca3, wmax_ca3ca3)
    wmax_ca3in = 2
    w_ca3in = np.ones((nn_in, nn_ca3)) * wmax_ca3in
    wmax_inca3 = 10
    w_inca3 = np.ones((nn_ca3, nn_in)) * wmax_inca3
    wmax_inin = 0
    w_inin = np.ones((nn_in, nn_in)) * wmax_inin
    w = np.zeros((nn, nn))
    w[:nn_ca3, :nn_ca3] = w_ca3ca3
    w[:nn_ca3, nn_ca3:] = w_inca3
    w[nn_ca3:, :nn_ca3] = w_ca3in
    w[nn_ca3:, nn_ca3:] = w_inin
    # w[:100, 100:200] = 0
    # w[100:200, :100] = 0

    # Synapses
    tau_gex = 10
    tau_gin = 10
    U_stdx = 0.450  # 0.375
    U_stfx = 0
    tau_stdx = 0.5e3  # 1
    tau_stfx = 0.2e3
    spdelay = 40  # in index unit, 2ms

    # Initialization
    v = np.ones(nn) * izhi_c
    u = np.zeros(nn)
    Isyn = np.zeros(nn)
    gex = np.zeros(nn)
    gin = np.zeros(nn)
    stdx = np.ones(nn)
    stfx = np.ones(nn)
    ECstfx = np.ones(nn) * ECstf_rest
    deadx = np.zeros(nn)
    spdelayx = np.zeros(nn)
    spdelayDiffx = np.zeros(nn)
    SpikeDF_dict = dict(neuronid=[], tidxsp=[])

    v_pop = np.zeros((t.shape[0], nn))
    Isen_pop = np.zeros((t.shape[0], nn))
    Isen_fac_pop = np.zeros((t.shape[0], nn))
    Isyn_pop = np.zeros((t.shape[0], nn))
    Itotal_pop = np.zeros((t.shape[0], nn))
    syneff_pop = np.zeros((t.shape[0], nn))
    ECstfx_pop = np.zeros((t.shape[0], nn))

    # # Simulation runtime
    t1 = time.time()
    for i in range(t.shape[0]):
        # Behavioural
        run_x = traj_x[i]
        run_a = traj_a[i]

        # Sensory input
        angle_max = np.exp(-np.square(cdiff(run_a, aatun1d)) / 0.1) * Iangle_diff
        posangle_max = Ipos_max_compen + angle_max
        # Isen = squarefunc(run_x, xxtun1d, Ipos_sd, posangle_max) * (np.cos(theta_phase[i] + EC_phase) + 1)/2
        Isen = gaufunc(run_x, xxtun1d, Ipos_sd, posangle_max) * (np.cos(theta_phase[i] + EC_phase) + 1)/2
        Isen[nn_ca3:] = 0
        ECstfx += ((ECstf_rest-ECstfx)/tau_ECstf + (ECstf_target - ECstfx) * U_ECstf * Isen) * dt
        Isen_fac = np.square(ECstfx) * Isen

        # Total Input
        Itotal = Isyn + Isen_fac - Itheta[i]

        # Izhikevich
        deadmask = deadx < deadx_thresh
        v[deadmask] += (0.04*v[deadmask]**2 + 5*v[deadmask] + 140 - u[deadmask] + Itotal[deadmask]) * dt
        u[deadmask] += izhi_a * (izhi_b * v[deadmask] - u[deadmask]) * dt
        fidx = np.where(v > V_thresh)[0]
        deadx = deadx + (-deadx/tau_deadx) * dt
        deadx[fidx] = deadx[fidx] + 1
        v[fidx] = izhi_c
        u[fidx] = u[fidx] + izhi_d

        # STD & STF
        d_stdx_dt = (1 - stdx)/tau_stdx
        d_stdx_dt[fidx] = d_stdx_dt[fidx] - U_stdx * stdx[fidx]
        d_stdx_dt[nn_ca3:] = 0
        stdx += d_stdx_dt * dt
        d_stfx_dt = (1 - stfx)/tau_stfx
        d_stfx_dt[fidx] = d_stfx_dt[fidx] + U_stfx * (2 - stfx[fidx])
        d_stfx_dt[nn_ca3:] = 0
        stfx += d_stfx_dt * dt
        syneff = stdx * stfx

        # Spike delay counter
        spdelayx[fidx] = spdelayx[fidx] + spdelay
        diff = -np.sign(spdelayx)
        spdelayx += diff
        spdelayDiffx += -diff
        delayed_fidx = np.where(spdelayDiffx >= spdelay)[0]
        spdelayDiffx[delayed_fidx] = 0
        fidx_ca3 = delayed_fidx[delayed_fidx < nn_ca3]
        fidx_in = delayed_fidx[delayed_fidx >= nn_ca3]

        # Synaptic input (Excitatory)
        spike_sum = np.sum(syneff[fidx_ca3].reshape(1, -1) * w[:, fidx_ca3], axis=1) / nn_ca3
        gex += (-gex/tau_gex + spike_sum) * dt
        Isyn_ca3 = gex * (V_ca3 - v)

        # Synaptic input (Inhibitory)
        spike_sum = np.sum(w[:, fidx_in], axis=1) / nn_in
        gin += (-gin/tau_gin + spike_sum) * dt
        Isyn_in = gin * (V_in - v)
        Isyn = Isyn_ca3 + Isyn_in


        # Store data
        SpikeDF_dict['neuronid'].extend(list(fidx))
        SpikeDF_dict['tidxsp'].extend([i] * len(fidx))

        v_pop[i, :] = v
        Isen_pop[i, :] = Isen
        Isen_fac_pop[i, :] = Isen_fac
        Isyn_pop[i, :] = Isyn
        Itotal_pop[i, :] = Itotal
        syneff_pop[i, :] = syneff
        ECstfx_pop[i, :] = ECstfx

    print('Simulation time = %0.2fs'%(time.time()-t1))

    SpikeDF = pd.DataFrame(SpikeDF_dict)
    SpikeDF['neuronx'] = SpikeDF['neuronid'].apply(lambda x : xxtun1d[x])

    NeuronDF = pd.DataFrame(dict(neuronid=np.arange(nn), neuronx=xxtun1d, neurona=aatun1d))

    BehDF = pd.DataFrame(dict(t=t, x=traj_x, y=traj_y, a=traj_a, Itheta=Itheta, theta_phase=theta_phase,
                              theta_phase_plot=theta_phase_plot))

    ActivityData = dict(v=v_pop, Isen=Isen_pop, Isyn=Isyn_pop, Isen_fac=Isen_fac_pop,
                        Itotal=Itotal_pop, syneff=syneff_pop, ECstf=ECstfx_pop)

    MetaData = dict(nn=nn, nn_ca3=nn_ca3, nn_in=nn_in, w=w, EC_phase=EC_phase)

    best_nidx = get_nidx_1d_np(x=5, a=0, xtun=xxtun1d_ca3, atun=aatun1d_ca3)  # best angle
    best_popnidxs = np.arange(100)
    worst_nidx = get_nidx_1d_np(x=5, a=np.pi, xtun=xxtun1d_ca3, atun=aatun1d_ca3)  # worst angle
    worst_popnidxs = np.arange(100, nn_ca3)

    fig, ax = plt.subplots(6, 5, figsize=(14, 12), facecolor='white', constrained_layout=True)
    gs = ax[0, 1].get_gridspec()
    for axeach in ax[0:3, 1:].ravel():
        axeach.remove()
    axbig_best = fig.add_subplot(gs[0:3, 1:])

    gs2 = ax[3, 1].get_gridspec()
    for axeach in ax[3:, 1:].ravel():
        axeach.remove()
    axbig_worst = fig.add_subplot(gs2[3:, 1:])



    data_best = plot_sequences(SpikeDF, BehDF, ActivityData, NeuronDF, MetaData,
                               best_nidx, best_popnidxs, fig, ax[0:3, 0], axbig_best, tag='Best')
    data_worst = plot_sequences(SpikeDF, BehDF, ActivityData, NeuronDF, MetaData,
                                worst_nidx, worst_popnidxs, fig, ax[3:, 0], axbig_worst, tag='Worst')

    mean_phasesp_best, onset_best = data_best['mean_phasesp'], data_best['onset']
    mean_phasesp_worst, onset_worst = data_worst['mean_phasesp'], data_worst['onset']
    axbig_worst.annotate('Mean phase diff = %0.2f rad\nOnset diff = %0.2f' %(mean_phasesp_worst - mean_phasesp_best,
                                                                             onset_worst - onset_best),
                         xy=(0.02, 0.65), xycoords='axes fraction', fontsize=legendsize+4)

    fig.savefig(save_pth, dpi=150)


save_dir = 'plots/Change_ECtau_STDtau'
os.makedirs(save_dir, exist_ok=True)
save_pth = join(save_dir, 'I10+20_TauSTD0f5_USTD0450.png')
params_search(save_pth)



