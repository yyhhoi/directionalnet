# EC-STF, with IB
# Current addition: 1D mossy. Note: Mos-CA3 connection is assymetrical gaussian

from os.path import join
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from pycircstat import cdiff, mean as cmean
from library.comput_utils import cal_hd_np, get_nidx_1d_np, pair_diff
from library.visualization import customlegend
from library.linear_circular_r import rcc

mpl.rcParams['figure.dpi'] = 150
legendsize = 7

def gaufunc(x, mu, sd, outmax):
    return outmax * np.exp(-np.square(x - mu)/(sd**2))

def uniformfunc(x, mu, sd, outmax):
    out = np.exp(-np.square(x - mu)/(sd**2))
    out[out < 0.2] = 0
    out[out >= 0.2] = outmax
    return out

def asymgaufunc(x, mu, sd, outmax):
    diff = x - mu
    out = outmax * np.exp(-np.square(diff)/(sd**2))
    out[diff > 0] = 0
    return out


def plot_precession_and_currents(ax, egnidxs, eg_cs, t, traj_x, theta_phase, theta_phase_plot, SpikeDF, Itotal_pop, Isyn_pop,
                                 Isen_pop, Isen_fac_pop, syneff_pop, ECstf_pop, tag):
    # Phase precession
    tidxsp_eg = SpikeDF.loc[SpikeDF['neuronid']==egnidxs[0], 'tidxsp'].to_numpy()
    tsp_eg, phasesp_eg = t[tidxsp_eg], theta_phase[tidxsp_eg]
    xsp_eg = traj_x[tidxsp_eg]
    mean_phasesp = cmean(phasesp_eg)
    xspmin, xsprange = xsp_eg.min(), xsp_eg.max() - xsp_eg.min()
    xsp_norm_eg = (xsp_eg-xspmin)/xsprange
    if tag[-3:] == 'Opp':
        xsp_norm_eg = -xsp_norm_eg + 1
    regress = rcc(xsp_norm_eg, phasesp_eg, abound=(-1., 1.))
    rcc_c, rcc_slope_rad = regress['phi0'], regress['aopt'] * 2 * np.pi
    xdum = np.linspace(xsp_norm_eg.min(), xsp_norm_eg.max(), 100)
    ydum = xdum * rcc_slope_rad + rcc_c
    ax[0].scatter(xsp_norm_eg, phasesp_eg, marker='|', s=4, color=eg_cs[0])
    ax[0].axhline(mean_phasesp, xmin=0, xmax=0.3, linewidth=1)
    ax[0].plot(xdum, ydum, linewidth=0.75, color=eg_cs[0])
    # ax[0].annotate('y= %0.2fx + %0.2f'%(rcc_slope_rad, rcc_c), xy=(0.25, 0.85), xycoords='axes fraction', fontsize=legendsize+2)
    ax[0].set_title('%s, y= %0.2fx + %0.2f'%(tag, rcc_slope_rad, rcc_c), fontsize=legendsize+1)
    ax[0].set_ylim(0, 2*np.pi)
    ax[0].set_yticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])
    ax[0].set_yticklabels(['0', '$\pi/2$', '$\pi$', '$1.5\pi$', '$2\pi$'])
    # ax[0].set_title('%s'%(tag), fontsize=legendsize+1)

    # Current
    tidxsp_eg1 = SpikeDF.loc[SpikeDF['neuronid']==egnidxs[0], 'tidxsp'].to_numpy()
    tsp_eg1 = t[tidxsp_eg1]
    bins = np.linspace(-100, 100, 50)
    for egi, egnidx in enumerate(egnidxs):
        if egi == 0:
            continue
        tidxsp_eg2 = SpikeDF.loc[SpikeDF['neuronid']==egnidx, 'tidxsp'].to_numpy()
        tsp_eg2 = t[tidxsp_eg2]
        tsp_diff = pair_diff(tsp_eg1, tsp_eg2).flatten()
        tsp_diff = tsp_diff[np.abs(tsp_diff) < 100]
        ax[1].hist(tsp_diff, bins=bins, histtype='step', color=eg_cs[egi])

    ax[2].axis('off')

    # Currents and STD
    ax[2].plot(t, Isyn_pop[:, egnidxs[0]], label='Isyn', linewidth=0.75, color='orange')
    ax[2].plot(t, Isen_pop[:, egnidxs[0]], label='Isen', linewidth=0.75, color='blue')
    ax[2].plot(t, Isen_fac_pop[:, egnidxs[0]], label='Isen_fac', linewidth=0.75, color='cyan')
    ax[2].set_ylim(-1, 40)
    customlegend(ax[2], fontsize=legendsize, loc='upper left')
    axsyneff = ax[2].twinx()
    axsyneff.plot(t, syneff_pop[:, egnidxs[0]], label='snyeff', color='r', linewidth=0.75)
    axsyneff.plot(t, ECstf_pop[:, egnidxs[0]], label='ECSTF', color='green', linewidth=0.75)
    axsyneff.set_ylim(-0.1, 2.1)
    axsyneff.eventplot(tsp_eg, lineoffsets = 1.8, linelength=0.1, linewidths=0.5, color='r')
    customlegend(axsyneff, fontsize=legendsize)
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for i in theta_cutidx:
        ax[2].axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)
    for ax_each in ax.ravel():
        ax_each.tick_params(labelsize=legendsize)
    axsyneff.tick_params(labelsize=legendsize)
    return mean_phasesp, rcc_slope_rad, rcc_c

def plot_sequences(SpikeDF, BehDF, ActivityData, NeuronDF, MetaData, egnidxs, eg_cs, popnidxs, fig, ax, axbig, tag):

    # Get data
    traj_x = BehDF['x'].to_numpy()
    t = BehDF['t'].to_numpy()
    theta_phase = BehDF['theta_phase'].to_numpy()
    theta_phase_plot = BehDF['theta_phase_plot'].to_numpy()
    Isyn_pop = ActivityData['Isyn']
    Itotal_pop = ActivityData['Itotal']
    Isen_pop = ActivityData['Isen']
    Isen_fac_pop = ActivityData['Isen_fac']
    syneff_pop = ActivityData['syneff']
    ECstf_pop = ActivityData['ECstf']
    xxtun1d, aatun1d = NeuronDF['neuronx'].to_numpy(), NeuronDF['neurona'].to_numpy()
    nn, nn_ca3, nn_mos, nn_in = MetaData['nn'], MetaData['nn_ca3'], MetaData['nn_mos'], MetaData['nn_in']
    EC_phase = MetaData['EC_phase']
    thalf_sepidx = MetaData['thalf_sepidx']

    # Population raster
    dxtun = xxtun1d[1] - xxtun1d[0]
    tt, traj_xx = np.meshgrid(t, xxtun1d[popnidxs])
    mappable = axbig.pcolormesh(tt, traj_xx, syneff_pop[:, popnidxs].T, shading='auto', vmin=0, vmax=2, cmap='seismic')
    for neuronid in popnidxs:
        tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
        tsp_neuron = t[tidxsp_neuron]
        neuronx = xxtun1d[neuronid]
        if neuronid == egnidxs[0]:
            ras_c = eg_cs[0]
        elif neuronid == egnidxs[1]:
            ras_c = eg_cs[1]
        elif neuronid == egnidxs[2]:
            ras_c = eg_cs[2]
        else:
            ras_c = 'lime'
        axbig.eventplot(tsp_neuron, lineoffsets=neuronx, linelengths=dxtun, linewidths=0.75, color=ras_c)

    # Population raster - Inhibitory
    nn_in_startidx = nn_ca3+nn_mos
    neuron_in_xmin, neuron_in_xmax = traj_x.max(), traj_x.max()*1.1
    dx_in = (neuron_in_xmax - neuron_in_xmin)/(nn_in)
    for neuronid in np.arange(nn_in_startidx, nn_in_startidx+nn_in, 1):
        tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
        tsp_neuron = t[tidxsp_neuron]
        plot_onset = neuron_in_xmin+(neuronid-nn_in_startidx) * dx_in
        axbig.eventplot(tsp_neuron, lineoffsets=plot_onset, linelengths=dx_in, linewidths=0.75, color='m')
    axbig.plot(t, traj_x, c='k', linewidth=0.75)
    theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
    for i in theta_cutidx:
        axbig.axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)

    axbig.annotate('%s direction\nGray lines = Theta phase 0\nEC phase shift = %d deg'%(tag, 360-np.rad2deg(EC_phase)),
                   xy=(0.02, 0.75), xycoords='axes fraction', fontsize=12)
    axbig.tick_params(labelsize=legendsize)
    cbar = fig.colorbar(mappable, ax=axbig, ticks=[0, 1, 2], shrink=0.5)
    cbar.set_label('Syn Efficacy', rotation=90, fontsize=legendsize)


    # Plot precession and currents, separately for along/against mossy trajectory
    before_idxs = np.arange(thalf_sepidx)
    after_idxs = np.arange(thalf_sepidx, t.shape[0])
    before_SpikeDF = SpikeDF.loc[SpikeDF['tidxsp'] < thalf_sepidx].reset_index(drop=True)
    after_SpikeDF = SpikeDF.loc[SpikeDF['tidxsp'] >= thalf_sepidx].reset_index(drop=True)
    after_SpikeDF['tidxsp'] = after_SpikeDF['tidxsp'] - thalf_sepidx
    plot_tag_same = '%s Same'%tag
    info_same = plot_precession_and_currents(ax[:, 0], egnidxs, eg_cs, t[before_idxs], traj_x[before_idxs], theta_phase[before_idxs],
                                             theta_phase_plot[before_idxs], before_SpikeDF, Itotal_pop[before_idxs],
                                             Isyn_pop[before_idxs], Isen_pop[before_idxs], Isen_fac_pop[before_idxs],
                                             syneff_pop[before_idxs], ECstf_pop[before_idxs], tag=plot_tag_same)
    if tag == 'Best':
        tag2 = 'Worst'
    elif tag == 'Worst':
        tag2 = 'Best'
    plot_tag_opp = '%s Opp'%tag2
    info_opp = plot_precession_and_currents(ax[:, 1], egnidxs, eg_cs, t[after_idxs], traj_x[after_idxs], theta_phase[after_idxs],
                                            theta_phase_plot[after_idxs], after_SpikeDF, Itotal_pop[after_idxs],
                                            Isyn_pop[after_idxs], Isen_pop[after_idxs], Isen_fac_pop[after_idxs],
                                            syneff_pop[after_idxs], ECstf_pop[after_idxs], tag=plot_tag_opp)
    same_stat_dict = dict(mean_phasesp=info_same[0], slope=info_same[1], onset=info_same[2])
    opp_stat_dict = dict(mean_phasesp=info_opp[0], slope=info_opp[1], onset=info_opp[2])
    stat_dict = {plot_tag_same: same_stat_dict, plot_tag_opp: opp_stat_dict}
    return stat_dict

def params_search(Ipos_max, Iangle_diff, mos_shift, wmax_ca3mos, wmax_mosca3, wmax_mosmos, save_pth):
    # Environment & agent
    dt = 0.1  # 0.1ms
    tmin, tmax, thalf = 0, 4e3, 2e3  # 4s
    trajx_min, trajx_max = 0, 10  # from 0cm to 5cm, and then return to 0cm
    t_right, t_left = np.arange(tmin, thalf, dt), np.arange(thalf, tmax, dt)
    t = np.concatenate([t_right, t_left])
    traj_x_rightwards = np.linspace(trajx_min, trajx_max, t_right.shape[0])
    traj_x_leftwards = np.linspace(trajx_max, trajx_min, t_left.shape[0])
    traj_x = np.concatenate([traj_x_rightwards, traj_x_leftwards])
    traj_y = np.zeros(traj_x.shape[0]) * 0.0
    traj_a = cal_hd_np(traj_x, traj_y)

    # Izhikevich's model parameters
    # izhi_a, izhi_b, izhi_c, izhi_d = 0.02, 0.2, -50, 2  # CH
    izhi_a, izhi_b, izhi_c, izhi_d = 0.02, 0.2, -55, 4  # Intrinsic bursting
    V_ex, V_in = 60, -80
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
    # Ipos_max = 20  # CH: 15,  IB: 20
    # Iangle_diff = 25  # CH: 25, IB: 25
    Ipos_sd = 1
    ECstf_rest, ECstf_target = 0, 2
    tau_ECstf = 0.5e3
    U_ECstf = 0.001
    Ipos_max_compen = Ipos_max + (np.cos(EC_phase) + 1)/2 * theta_amp

    # Sensory tuning
    xmin, xmax, nx = traj_x.min(), traj_x.max(), 100
    atun_ca3 = np.deg2rad(np.array([0, 180]))
    xtun_ca3 = np.linspace(xmin, xmax, nx)
    xxtun2d_ca3, aatun2d_ca3 = np.meshgrid(xtun_ca3, atun_ca3)
    xxtun1d_ca3, aatun1d_ca3 = xxtun2d_ca3.flatten(), aatun2d_ca3.flatten()
    nn_ca3 = xxtun1d_ca3.shape[0]
    xxtun1d_mos = np.linspace(xmin, xmax, nx)
    nn_mos = xxtun1d_mos.shape[0]
    aatun1d_mos = np.zeros(nn_mos)  # Dummy variables
    nn_in = 50
    xxtun1d_in, aatun1d_in = np.zeros(nn_in), np.zeros(nn_in)  # Dummy Variables
    xxtun1d = np.concatenate([xxtun1d_ca3, xxtun1d_mos, xxtun1d_in])
    aatun1d = np.concatenate([aatun1d_ca3, aatun1d_mos, aatun1d_in])
    nn = xxtun1d.shape[0]
    endidx_ca3, endidx_mos, endidx_in = nn_ca3, nn_ca3 + nn_mos, nn_ca3 + nn_mos + nn_in

    # # Weights
    # mos_shift = 1
    wmax_ca3ca3, wsd_ca3ca3 = 5.5, 2  # CH: 4, IB:5.5
    # wmax_ca3mos, wmax_mosca3, wmax_mosmos = 5, 5, 5
    wsd_ca3mos, wsd_mosca3, wsd_mosmos= 1, 1, 1
    wmax_allin, wmax_inall, wmax_inin = 3, 2.5, 0  # CH: 3, 7.5, IB: 3, 7.5
    w_ca3ca3 = gaufunc(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_ca3.reshape(nn_ca3, 1), wsd_ca3ca3, wmax_ca3ca3)
    w_ca3mos = gaufunc(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_mos.reshape(nn_mos, 1), wsd_ca3mos, wmax_ca3mos)
    w_mosca3 = asymgaufunc(xxtun1d_mos.reshape(1, nn_mos), xxtun1d_ca3.reshape(nn_ca3, 1) - mos_shift, wsd_mosca3, wmax_mosca3)
    w_mosmos = gaufunc(xxtun1d_mos.reshape(1, nn_mos), xxtun1d_mos.reshape(nn_mos, 1), wsd_mosmos, wmax_mosmos)
    w_allin = np.ones((nn_in, nn_ca3 + nn_mos)) * wmax_allin
    w_inall = np.ones((nn_ca3 + nn_mos, nn_in)) * wmax_inall
    w_inin = np.ones((nn_in, nn_in)) * wmax_inin
    w = np.zeros((nn, nn))
    w[0:nn_ca3, 0:nn_ca3] = w_ca3ca3
    w[nn_ca3:endidx_mos, 0:nn_ca3] = w_ca3mos
    w[0:nn_ca3, nn_ca3:endidx_mos] = w_mosca3
    w[nn_ca3:endidx_mos, nn_ca3:endidx_mos] = w_mosmos
    w[endidx_mos:endidx_in, 0:endidx_mos] = w_allin
    w[0:endidx_mos, endidx_mos:endidx_in] = w_inall
    w[endidx_mos:endidx_in, endidx_mos:endidx_in] = w_inin

    # Synapses
    tau_gex = 10
    tau_gin = 10
    U_stdx = 0.450  # CH 0.450
    U_stfx = 0
    tau_stdx = 0.5e3  # 1
    tau_stfx = 0.2e3
    spdelay = 20  # in index unit, 2ms

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
        run_x, run_a = traj_x[i], traj_a[i]

        # Sensory input
        angle_max = np.exp(-np.square(cdiff(run_a, aatun1d)) / 0.1) * Iangle_diff
        posangle_max = Ipos_max_compen + angle_max
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
        fidx_ex = delayed_fidx[delayed_fidx < endidx_mos]
        fidx_in = delayed_fidx[delayed_fidx >= endidx_mos]

        # Synaptic input (Excitatory)
        spike_sum = np.sum(syneff[fidx_ex].reshape(1, -1) * w[:, fidx_ex], axis=1) / endidx_mos
        gex += (-gex/tau_gex + spike_sum) * dt
        Isyn_ex = gex * (V_ex - v)

        # Synaptic input (Inhibitory)
        spike_sum = np.sum(w[:, fidx_in], axis=1) / nn_in
        gin += (-gin/tau_gin + spike_sum) * dt
        Isyn_in = gin * (V_in - v)
        Isyn = Isyn_ex + Isyn_in


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

    NeuronDF = pd.DataFrame(dict(neuronid=np.arange(nn), neuronx=xxtun1d, neurona=aatun1d,
                                 neurontype=["CA3"]*nn_ca3 + ['Mos']*nn_mos + ['In']*nn_in))

    BehDF = pd.DataFrame(dict(t=t, x=traj_x, y=traj_y, a=traj_a, Itheta=Itheta, theta_phase=theta_phase,
                              theta_phase_plot=theta_phase_plot))

    ActivityData = dict(v=v_pop, Isen=Isen_pop, Isyn=Isyn_pop, Isen_fac=Isen_fac_pop,
                        Itotal=Itotal_pop, syneff=syneff_pop, ECstf=ECstfx_pop)

    MetaData = dict(nn=nn, nn_ca3=nn_ca3, nn_mos=nn_mos, nn_in=nn_in, w=w, EC_phase=EC_phase, thalf_sepidx=t_right.shape[0])

    dist1, dist2 = 0.75, 1.5

    right_egnidx1 = get_nidx_1d_np(x=trajx_max/2, a=0, xtun=xxtun1d_ca3, atun=aatun1d_ca3)  # best angle
    right_egnidx2 = get_nidx_1d_np(x=trajx_max/2+dist1, a=0, xtun=xxtun1d_ca3, atun=aatun1d_ca3)
    right_egnidx3 = get_nidx_1d_np(x=trajx_max/2+dist2, a=0, xtun=xxtun1d_ca3, atun=aatun1d_ca3)
    right_egnidxs = [right_egnidx1, right_egnidx2, right_egnidx3]
    right_popnidxs = np.arange(100)

    left_egnidx1 = get_nidx_1d_np(x=trajx_max/2, a=np.pi, xtun=xxtun1d_ca3, atun=aatun1d_ca3)  # best angle
    left_egnidx2 = get_nidx_1d_np(x=trajx_max/2+dist1, a=np.pi, xtun=xxtun1d_ca3, atun=aatun1d_ca3)
    left_egnidx3 = get_nidx_1d_np(x=trajx_max/2+dist2, a=np.pi, xtun=xxtun1d_ca3, atun=aatun1d_ca3)
    left_egnidxs = [left_egnidx1, left_egnidx2, left_egnidx3]
    left_popnidxs = np.arange(100, nn_ca3)

    eg_cs = ['r', 'darkgreen', 'm']

    fig, ax = plt.subplots(6, 7, figsize=(16, 10), facecolor='white', constrained_layout=True)
    gs = ax[0, 0].get_gridspec()
    for axeach in ax[0:3, 0:5].ravel():
        axeach.remove()
    axbig_right = fig.add_subplot(gs[0:3, 0:5])

    gs2 = ax[3, 0].get_gridspec()
    for axeach in ax[3:, 0:5].ravel():
        axeach.remove()
    axbig_left = fig.add_subplot(gs2[3:, 0:5])


    data_right = plot_sequences(SpikeDF, BehDF, ActivityData, NeuronDF, MetaData,
                                right_egnidxs, eg_cs, right_popnidxs, fig, ax[0:3, 5:7], axbig_right, tag='Best')
    data_left = plot_sequences(SpikeDF, BehDF, ActivityData, NeuronDF, MetaData,
                               left_egnidxs, eg_cs, left_popnidxs, fig, ax[3:, 5:7], axbig_left, tag='Worst')

    stat_dict_all = {**data_right, **data_left}


    mean_phasesp_best, onset_best = stat_dict_all['Best Same']['mean_phasesp'], stat_dict_all['Best Same']['onset']
    mean_phasesp_worst, onset_worst = stat_dict_all['Worst Same']['mean_phasesp'], stat_dict_all['Worst Same']['onset']
    axbig_left.annotate('Mean phase diff = %0.2f rad\nOnset diff = %0.2f' %(mean_phasesp_worst - mean_phasesp_best,
                                                                            onset_worst - onset_best),
                        xy=(0.02, 0.65), xycoords='axes fraction', fontsize=legendsize+4)

    fig.savefig(save_pth, dpi=150)
    return


Ipos_max = 10
Iangle_diff = 15
mos_shift = 1
# wmax_ca3mos = 15
# wmax_mosca3 = 15
wmax_mosmos = 0

for wmos in [10, 15, 20]:
    wmax_ca3mos = wmos
    wmax_mosca3 = wmos
    params_text = 'I%d+%d_MosShift%0.2f_Wmos%0.2f-%0.2f-%0.2f'%(Ipos_max, Iangle_diff, mos_shift, wmax_ca3mos, wmax_mosca3, wmax_mosmos)
    save_dir = 'plots/Mossy_Asym'
    os.makedirs(save_dir, exist_ok=True)
    save_pth = join(save_dir, 'Result_%s_WInAll2.5.png'%(params_text))
    print('start processing %s'%(save_pth))
    params_search(Ipos_max, Iangle_diff, mos_shift, wmax_ca3mos, wmax_mosca3, wmax_mosmos, save_pth)


