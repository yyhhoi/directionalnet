import time

import numpy as np
import pandas as pd

from library.comput_utils import circgaufunc, boxfunc2d, gaufunc2d_angles, gaufunc2d, cal_hd_np, pair_diff


def createMosProjMat_p2p(startpos, endpos, posvec_mos, posvec_CA3, act_sd):
    """
    Create weight matrix with shape (N, M), from M mossy neurons at the starting xy- position to
    N CA3 neurons at the ending xy- position.

    Parameters
    ----------
    startpos: ndarray
        Shape (2, ). Starting xy- position of the mossy projection.
    endpos: ndarray
        Shape (2, ). Targeted ending position of the mossy projection.
    posvec_mos: ndarray
        Shape (M, 2). xy- positional tuning of M mossy neuron's.
    posvec_CA3: ndarray
        Shape (N, 2). xy- positional tuning of N CA3 neuron's.
    act_max: float
        Peak value of the gaussian-shape activation strengths of mossy projection.
    act_sd: float
        SD of the gaussian-shape activation strengths of mossy projection.

    Returns
    -------
    w_mosCA3: ndarray
        Shape (N, M). Weight matrix.
    mos_act: ndarray
        Shape (M, ). Post-synaptic activation strength.

    """
    mos_act = np.exp(-np.sum(np.square(posvec_mos - startpos.reshape(1, 2)), axis=1) / (2 * (act_sd ** 2)))

    diff_vec = endpos - startpos
    ca3_offset = posvec_CA3 - diff_vec.reshape(1, 2)
    CA3_end_xdiff = ca3_offset[:, 0].reshape(-1, 1) - posvec_mos[:, 0].reshape(1, -1)  # (N, M)
    CA3_end_ydiff = ca3_offset[:, 1].reshape(-1, 1) - posvec_mos[:, 1].reshape(1, -1)  # (N, M)
    CA3_end_squarediff = (CA3_end_xdiff ** 2) + (CA3_end_ydiff ** 2)  # (N, M)
    CA3_expo = np.exp(-CA3_end_squarediff / (2 * (act_sd ** 2)))  # (N, M)
    w_mosCA3 = mos_act.reshape(1, -1) * CA3_expo
    return w_mosCA3, mos_act



def directional_tuning_tile(N, M, atun_seeds, start_seed=0):
    if (N%4 != 0) or (M%4 != 0):
        raise ValueError('Shape of directional tuning matrix should be the multiples of 4')
    aatun2d = np.zeros((N, M))
    seed_i = start_seed
    for i in np.arange(0, N, 2):
        for j in np.arange(0, M, 2):
            np.random.seed(seed_i)
            rand_shift = np.random.uniform(0, 2*np.pi)
            perm_atun_seeds = atun_seeds + rand_shift
            aatun2d[i, j:j+2] = perm_atun_seeds[0:2]
            aatun2d[i+1, j:j+2] = perm_atun_seeds[2:4]
            seed_i += 1
    aatun2d = np.mod(aatun2d, 2*np.pi) - np.pi  # range = (-pi, pi]
    return aatun2d


def simulate_SNN(BehDF, config_dict, store_Activity=True, store_w=True):

    # ============================================================================================
    # ======================================== Parameters ========================================
    # ============================================================================================
    # # Environment & agent
    dt = config_dict['dt']  # 0.1ms
    t = BehDF['t'].to_numpy()
    traj_x = BehDF['traj_x'].to_numpy()
    traj_y = BehDF['traj_y'].to_numpy()
    traj_a = BehDF['traj_a'].to_numpy()

    # # Izhikevich's model
    izhi_c1 = config_dict['izhi_c1']
    izhi_c2 = config_dict['izhi_c2']
    izhi_c3 = config_dict['izhi_c3']
    izhi_a_ex = config_dict['izhi_a_ex']
    izhi_b_ex = config_dict['izhi_b_ex']
    izhi_c_ex = config_dict['izhi_c_ex']
    izhi_d_ex = config_dict['izhi_d_ex']
    izhi_a_in = config_dict['izhi_a_in']
    izhi_b_in = config_dict['izhi_b_in']
    izhi_c_in = config_dict['izhi_c_in']
    izhi_d_in = config_dict['izhi_d_in']
    V_ex = config_dict['V_ex']
    V_in = config_dict['V_in']
    V_thresh = config_dict['V_thresh']
    spdelay = config_dict['spdelay']
    noise_rate = config_dict['noise_rate']

    # # Theta inhibition
    theta_amp = config_dict['theta_amp']
    theta_f = config_dict['theta_f']

    # Positional drive
    EC_phase = np.deg2rad(config_dict['EC_phase_deg'])
    Ipos_max = config_dict['Ipos_max']
    Iangle_diff = config_dict['Iangle_diff']
    Ipos_sd = config_dict['Ipos_sd']
    Iangle_kappa = config_dict['Iangle_kappa']
    ECstf_rest = config_dict['ECstf_rest']
    ECstf_target = config_dict['ECstf_target']
    tau_ECstf = config_dict['tau_ECstf']
    U_ECstf = config_dict['U_ECstf']

    # Sensory tuning
    xmin, xmax = config_dict['xmin'], config_dict['xmax']
    nx_ca3, nx_mos = config_dict['nx_ca3'], config_dict['nx_mos']
    ymin, ymax = config_dict['ymin'], config_dict['ymax']
    ny_ca3, ny_mos = config_dict['ny_ca3'], config_dict['ny_mos']
    nn_inca3, nn_inmos = config_dict['nn_inca3'], config_dict['nn_inmos']

    # Synapse parameters
    tau_gex = config_dict['tau_gex']
    tau_gin = config_dict['tau_gin']
    U_stdx_CA3 = config_dict['U_stdx_CA3']
    U_stdx_mos = config_dict['U_stdx_mos']
    tau_stdx = config_dict['tau_stdx']

    # # Weights
    # CA3-CA3
    wmax_ca3ca3 = config_dict['wmax_ca3ca3']
    wmax_ca3ca3_adiff = config_dict['wmax_ca3ca3_adiff']
    w_ca3ca3_akappa = config_dict['w_ca3ca3_akappa']
    asym_flag = config_dict['asym_flag']  # False: No asymmetry. True: rightward

    # CA3-Mos and Mos-CA3
    wmax_ca3mos = config_dict['wmax_ca3mos']  # 1500
    wmax_mosca3 = config_dict['wmax_mosca3']  # 2000
    mos_exist = config_dict['mos_exist']
    wmax_ca3mosca3_adiff = config_dict['wmax_ca3mosca3_adiff']  # 6000
    w_ca3mosca3_akappa = config_dict['w_ca3mosca3_akappa']  # 2

    # CA3-In and In-CA3
    wmax_CA3in = config_dict['wmax_CA3in']  # 50
    wmax_inCA3 = config_dict['wmax_inCA3']  # 5

    # Mos-In and In-Mos
    wmax_Mosin = config_dict['wmax_Mosin']  # 350
    wmax_inMos = config_dict['wmax_inMos']  # 35

    # Mos-Mos, In-In
    wmax_mosmos = config_dict['wmax_mosmos']  # 100
    wmax_inin = config_dict['wmax_inin']  # 0

    # Mossy layer projection trajectory
    mos_startpos = config_dict['mos_startpos']  # Shape (T, 2)
    mos_endpos = config_dict['mos_endpos']  # Shape (T, 2)

    # SD
    wsd_global = config_dict['wsd_global']

    # ============================================================================================
    # ======================================== Initialization ====================================
    # ============================================================================================

    numt = t.shape[0]

    # # Theta Inhibition
    theta_T = 1/theta_f * 1e3
    theta_phase = np.mod(t, theta_T)/theta_T * 2*np.pi
    theta_phase_plot = np.mod(theta_phase + 2*np.pi, 2*np.pi)
    Itheta = (1 + np.cos(theta_phase))/2 * theta_amp

    # # Positional drive
    Ipos_max_compen = Ipos_max + (np.cos(EC_phase) + 1)/2 * theta_amp
    config_dict['Ipos_max_compen'] = Ipos_max_compen


    # # Sensory tuning
    nn_in = nn_inca3 + nn_inmos
    xtun_ca3 = np.linspace(xmin, xmax, nx_ca3)
    ytun_ca3 = np.linspace(ymin, ymax, ny_ca3)
    xxtun2d_ca3, yytun2d_ca3 = np.meshgrid(xtun_ca3, ytun_ca3)
    atun_seeds = np.array([0, np.pi/2, np.pi, np.pi*3/2])
    aatun2d_ca3 = directional_tuning_tile(*xxtun2d_ca3.shape, atun_seeds=atun_seeds, start_seed=0)
    xxtun1d_ca3, yytun1d_ca3, aatun1d_ca3 = xxtun2d_ca3.flatten(), yytun2d_ca3.flatten(), aatun2d_ca3.flatten()
    xtun_mos = np.linspace(xmin, xmax, nx_mos)
    ytun_mos = np.linspace(ymin, ymax, ny_mos)
    xxtun2d_mos, yytun2d_mos = np.meshgrid(xtun_mos, ytun_mos)
    xxtun1d_mos, yytun1d_mos = xxtun2d_mos.flatten(), yytun2d_mos.flatten()
    aatun2d_mos = directional_tuning_tile(*xxtun2d_mos.shape, atun_seeds=atun_seeds, start_seed=0)
    aatun1d_mos = aatun2d_mos.flatten()
    xxtun1d_in, yytun1d_in, aatun1d_in = np.zeros(nn_in), np.zeros(nn_in), np.zeros(nn_in)  # Inhibitory neurons no tuning
    nn_ca3, nn_mos = xxtun1d_ca3.shape[0], xxtun1d_mos.shape[0]
    xxtun1d = np.concatenate([xxtun1d_ca3, xxtun1d_mos, xxtun1d_in])
    yytun1d = np.concatenate([yytun1d_ca3, yytun1d_mos, yytun1d_in])
    aatun1d = np.concatenate([aatun1d_ca3, aatun1d_mos, aatun1d_in])
    endidx_ca3, endidx_mos, endidx_in = nn_ca3, nn_ca3 + nn_mos, nn_ca3 + nn_mos + nn_in
    endidx_inca3 = endidx_mos + nn_inca3
    nn = xxtun1d.shape[0]

    # # Weight SD
    wsd_ca3ca3 = wsd_global
    wsd_ca3mos = wsd_global
    wsd_mosca3 = wsd_global
    wsd_mosmos = wsd_global

    # Constructing weights CA3-CA3
    w_ca3ca3 = gaufunc2d_angles(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_ca3.reshape(nn_ca3, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_ca3.reshape(nn_ca3, 1), aatun1d_ca3.reshape(1, nn_ca3), aatun1d_ca3.reshape(nn_ca3, 1), wsd_ca3ca3, wmax_ca3ca3, wmax_ca3ca3_adiff, w_ca3ca3_akappa)
    # w_ca3ca3noiseexp = gaufunc2d(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_ca3.reshape(nn_ca3, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_ca3.reshape(nn_ca3, 1), wsd_ca3ca3, 1)
    # np.random.seed(32)
    # w_ca3ca3noise = (np.random.uniform(0, 1, size=(nn_ca3, nn_ca3)) < 0.2) * wmax_ca3ca3_noise

    if asym_flag:
        w_ca3ca3[pair_diff(xxtun1d_ca3, xxtun1d_ca3) < 0] = 0

    # Constructing weights for Mos-CA3, CA3-Mos, Mos-Mos
    posvec_mos = np.stack([xxtun1d_mos, yytun1d_mos]).T
    posvec_CA3 = np.stack([xxtun1d_ca3, yytun1d_ca3]).T
    if mos_exist:
        act_adiff_max = wmax_ca3mosca3_adiff * np.exp(w_ca3mosca3_akappa * (np.cos(aatun1d_mos.reshape(1, -1) - aatun1d_ca3.reshape(-1, 1)) - 1))
        w_mosca3 = np.zeros((xxtun1d_ca3.shape[0], xxtun1d_mos.shape[0], mos_startpos.shape[0]))
        mos_act = np.zeros((posvec_mos.shape[0], mos_startpos.shape[0]))
        for i in range(mos_startpos.shape[0]):
            print('\rConstructing Weight matrix %d/%d'%(i, mos_startpos.shape[0]), flush=True, end='')
            w_mosca3[:, :, i], mos_act[:, i] = createMosProjMat_p2p(mos_startpos[i, :], mos_endpos[i, :], posvec_mos, posvec_CA3, wsd_mosca3)
        print()
        w_mosca3 = np.max(w_mosca3, axis=2)
        w_mosca3 = (wmax_mosca3 + act_adiff_max) * w_mosca3
        w_ca3mos = gaufunc2d_angles(xxtun1d_ca3.reshape(1, nn_ca3), xxtun1d_mos.reshape(nn_mos, 1), yytun1d_ca3.reshape(1, nn_ca3), yytun1d_mos.reshape(nn_mos, 1), aatun1d_ca3.reshape(1, nn_ca3), aatun1d_mos.reshape(nn_mos, 1), wsd_ca3mos, wmax_ca3mos, wmax_ca3mosca3_adiff, w_ca3mosca3_akappa)
        w_mosmos = gaufunc2d(xxtun1d_mos.reshape(1, nn_mos), xxtun1d_mos.reshape(nn_mos, 1), yytun1d_mos.reshape(1, nn_mos), yytun1d_mos.reshape(nn_mos, 1), wsd_mosmos, wmax_mosmos)
    else:
        w_mosca3 = np.zeros((xxtun1d_ca3.shape[0], xxtun1d_mos.shape[0]))
        w_ca3mos = np.zeros((xxtun1d_mos.shape[0], xxtun1d_ca3.shape[0]))
        w_mosmos = np.zeros((xxtun1d_mos.shape[0], xxtun1d_mos.shape[0]))
        print('No Mos.')

    # Inhibitory weights
    np.random.seed(0)
    w_CA3In = np.random.uniform(0, 1, size=(nn_inca3, nn_ca3)) * wmax_CA3in
    np.random.seed(1)
    w_InCA3 = np.random.uniform(0, 1, size=(nn_ca3, nn_inca3)) * wmax_inCA3
    np.random.seed(2)
    w_MosIn = np.random.uniform(0, 1, size=(nn_inmos, nn_mos)) * wmax_Mosin
    np.random.seed(3)
    w_InMos = np.random.uniform(0, 1, size=(nn_mos, nn_inmos)) * wmax_inMos
    np.random.seed(4)
    w_inin = np.random.uniform(0, 1, size=(nn_in, nn_in)) * wmax_inin

    # Assembling weights
    w = np.zeros((nn, nn))
    w[0:nn_ca3, 0:nn_ca3] = w_ca3ca3
    w[nn_ca3:endidx_mos, 0:nn_ca3] = w_ca3mos
    w[0:nn_ca3, nn_ca3:endidx_mos] = w_mosca3
    w[nn_ca3:endidx_mos, nn_ca3:endidx_mos] = w_mosmos
    w[endidx_mos:endidx_inca3, 0:endidx_ca3] = w_CA3In
    w[0:endidx_ca3, endidx_mos:endidx_inca3] = w_InCA3
    w[endidx_inca3:endidx_in, endidx_ca3:endidx_mos] = w_MosIn
    w[endidx_ca3:endidx_mos, endidx_inca3:endidx_in] = w_InMos
    w[endidx_mos:endidx_in, endidx_mos:endidx_in] = w_inin

    # Network states
    izhi_a = np.ones(nn) * izhi_a_ex
    izhi_b = np.ones(nn) * izhi_b_ex
    izhi_c = np.ones(nn) * izhi_c_ex
    izhi_d = np.ones(nn) * izhi_d_ex
    izhi_a[endidx_mos:endidx_in] = izhi_a_in
    izhi_b[endidx_mos:endidx_in] = izhi_b_in
    izhi_c[endidx_mos:endidx_in] = izhi_c_in
    izhi_d[endidx_mos:endidx_in] = izhi_d_in
    v = np.ones(nn) * izhi_c
    u = np.zeros(nn)
    Isyn = np.zeros(nn)
    Isyn_in = np.zeros(nn)
    Isyn_ex = np.zeros(nn)
    gex = np.zeros(nn)
    gin = np.zeros(nn)
    stdx2ca3 = np.ones(nn)
    stdx2mos = np.ones(nn)
    ECstfx = np.ones(nn) * ECstf_rest
    fidx_buffer = []
    SpikeDF_dict = dict(neuronid=[], tidxsp=[])
    if store_Activity:
        v_pop = np.zeros((numt, nn))
        Isen_pop = np.zeros((numt, nn))
        Isen_fac_pop = np.zeros((numt, nn))
        Isyn_pop = np.zeros((numt, nn))
        Itotal_pop = np.zeros((numt, nn))
        syneff_pop = np.zeros((numt, nn))
        ECstfx_pop = np.zeros((numt, nn))

    # Spike arrival noise
    np.random.seed(10)
    insta_r = noise_rate/1000 * dt
    sp_noise = np.zeros((numt, nn))
    sp_noise[:, :nn_ca3] = np.random.poisson(insta_r, size=(numt, nn_ca3))

    # ============================================================================================
    # ======================================== Simulation Runtime ================================
    # ============================================================================================
    t1 = time.time()

    for i in range(numt):
        if i % 100 == 0:
            print('\rSimulation %d/%d'%(i, numt), flush=True, end='')
        # Behavioural
        run_x, run_y, run_a = traj_x[i], traj_y[i], traj_a[i]

        # Sensory input
        Iangle = circgaufunc(run_a, aatun1d, Iangle_kappa, Iangle_diff)
        ECtheta = (np.cos(theta_phase[i] + EC_phase) + 1)/2
        Isen = boxfunc2d(run_x, xxtun1d, run_y, yytun1d, Ipos_sd, Ipos_max_compen+Iangle) * ECtheta
        Isen[nn_ca3:] = 0
        ECstfx += ((ECstf_rest-ECstfx)/tau_ECstf + (ECstf_target - ECstfx) * U_ECstf * Isen) * dt
        Isen_fac = np.square(ECstfx) * Isen


        # Total Input
        np.random.seed(i)
        Itotal = Isyn + Isen_fac - Itheta[i]

        # Izhikevich
        v += (izhi_c1*v**2 + izhi_c2*v + izhi_c3 - u + Itotal) * dt
        u += izhi_a * (izhi_b * v - u) * dt
        fidx = np.where(v > V_thresh)[0]
        v[fidx] = izhi_c[fidx]
        u[fidx] = u[fidx] + izhi_d[fidx]
        fidx_buffer.append(fidx)

        # STD
        d_stdx2ca3_dt = (1 - stdx2ca3)/tau_stdx
        d_stdx2ca3_dt[fidx] = d_stdx2ca3_dt[fidx] - U_stdx_CA3 * stdx2ca3[fidx]
        d_stdx2ca3_dt[nn_ca3:] = 0
        stdx2ca3 += d_stdx2ca3_dt * dt
        d_stdx2mos_dt = (1 - stdx2mos)/tau_stdx
        d_stdx2mos_dt[fidx] = d_stdx2mos_dt[fidx] - U_stdx_mos * stdx2mos[fidx]
        d_stdx2mos_dt[nn_ca3:] = 0
        stdx2mos += d_stdx2mos_dt * dt

        if i > spdelay:  # 2ms delay
            delayed_fidx = fidx_buffer.pop(0)
            delayed_fidx_ex = delayed_fidx[delayed_fidx < endidx_mos]
            delayed_fidx_in = delayed_fidx[delayed_fidx >= endidx_mos]

            # Synaptic noise
            noisethis = sp_noise[i, :]

            # Synaptic input (Excitatory)
            spike2ca3_sum = np.sum(stdx2ca3[delayed_fidx_ex].reshape(1, -1) * w[:endidx_ca3, delayed_fidx_ex], axis=1) / endidx_mos
            spike2mos_sum = np.sum(stdx2mos[delayed_fidx_ex].reshape(1, -1) * w[endidx_ca3:endidx_mos, delayed_fidx_ex], axis=1) / endidx_mos
            spike2in_sum = np.sum(stdx2ca3[delayed_fidx_ex].reshape(1, -1) * w[endidx_mos:, delayed_fidx_ex], axis=1) / endidx_mos
            gex += (-gex/tau_gex + np.concatenate([spike2ca3_sum, spike2mos_sum, spike2in_sum]) + noisethis) * dt
            Isyn_ex = gex * (V_ex - v)

            # Synaptic input (Inhibitory)
            spike_sum = np.sum(w[:, delayed_fidx_in], axis=1) / nn_in
            gin += (-gin/tau_gin + spike_sum) * dt
            Isyn_in = gin * (V_in - v)
            Isyn = Isyn_ex + Isyn_in

        # Store data
        SpikeDF_dict['neuronid'].extend(list(fidx))
        SpikeDF_dict['tidxsp'].extend([i] * len(fidx))

        if store_Activity:
            v_pop[i, :] = v
            Isen_pop[i, :] = Isen
            Isen_fac_pop[i, :] = Isen_fac
            Isyn_pop[i, :] = Isyn
            Itotal_pop[i, :] = Itotal
            syneff_pop[i, :] = stdx2ca3
            ECstfx_pop[i, :] = ECstfx

    print('\nSimulation time = %0.2fs'%(time.time()-t1))

    # # Storage
    BehDF['Itheta'] = Itheta
    BehDF['theta_phase'] = theta_phase
    BehDF['theta_phase_plot'] = theta_phase_plot

    SpikeDF = pd.DataFrame(SpikeDF_dict)
    SpikeDF['neuronx'] = SpikeDF['neuronid'].apply(lambda x : xxtun1d[x])

    NeuronDF = pd.DataFrame(dict(neuronid=np.arange(nn), neuronx=xxtun1d, neurony=yytun1d, neurona=aatun1d,
                                 neurontype=["CA3"]*nn_ca3 + ['Mos']*nn_mos + ['In']*nn_in))

    if store_Activity:
        ActivityData = dict(v=v_pop, Isen=Isen_pop, Isen_fac=Isen_fac_pop, Isyn=Isyn_pop,
                            Itotal=Itotal_pop, syneff=syneff_pop, ECstf=ECstfx_pop)
    else:
        ActivityData = None
    if ~store_w:
        w = None
    MetaData = dict(nn=nn, nn_ca3=nn_ca3, nn_mos=nn_mos, nn_in=nn_in, w=w, EC_phase=EC_phase)

    alldata =dict(
        BehDF=BehDF, SpikeDF=SpikeDF, NeuronDF=NeuronDF, ActivityData=ActivityData, MetaData=MetaData, Config=config_dict
    )
    return alldata