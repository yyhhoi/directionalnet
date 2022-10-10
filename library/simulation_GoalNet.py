import time
import pickle
import numpy as np
import pandas as pd



def gaufunc1d(x, mux, sd, outmax):
    return outmax * np.exp(-(np.square(x - mux))/(2*sd**2))

def circgaufunc1d(x, loc, outmax):
    return outmax * (np.cos(x - loc) + 1)/2

def boxfunc1d(x, mux, sd, outmax):

    dist = np.sqrt( np.square(x-mux))
    out = np.ones(dist.shape) * outmax
    out[dist > sd] = 0
    return out



def simulate_SNN(BehDF, config_dict, store_Activity=True, store_w=True):

    # ============================================================================================
    # ======================================== Parameters ========================================
    # ============================================================================================
    # # Environment & agent
    dt = config_dict['dt']  # 0.1ms
    t = BehDF['t'].to_numpy()
    traj_x = BehDF['traj_x'].to_numpy()
    traj_a = BehDF['traj_a'].to_numpy()

    # # Izhikevich's model
    izhi_a_ex = config_dict['izhi_a_ex']
    izhi_b_ex = config_dict['izhi_b_ex']
    izhi_c_ex = config_dict['izhi_c_ex']
    izhi_d_ex = config_dict['izhi_d_ex']
    V_ex = config_dict['V_ex']
    V_thresh = config_dict['V_thresh']
    spdelay = config_dict['spdelay']

    # # Theta inhibition
    theta_amp = config_dict['theta_amp']
    theta_f = config_dict['theta_f']

    # Positional drive
    EC_phase = np.deg2rad(config_dict['EC_phase_deg'])
    Ipos_max = config_dict['Ipos_max']
    Iangle_diff = config_dict['Iangle_diff']
    Ipos_sd = config_dict['Ipos_sd']

    # Sensory tuning
    xmin, xmax = config_dict['xmin'], config_dict['xmax']
    nn = config_dict['nn']

    # Synapse parameters
    tau_gex = config_dict['tau_gex']

    # Goal-directed drive
    goal_x = config_dict['goal_x']
    Imax_goal = config_dict['Imax_goal']
    Isd_goal = config_dict['Isd_goal']
    Imax_angle_goal = config_dict['Imax_angle_goal']

    # # Weights
    # CA3-CA3
    wmax = config_dict['wmax']
    wsd = config_dict['wsd']
    wmax_angle = config_dict['wmax_angle']
    wsd_angle = config_dict['wsd_angle']

    # ============================================================================================
    # ======================================== Initialization ====================================
    # ============================================================================================

    numt = t.shape[0]

    # # Theta Inhibition
    theta_T = 1/theta_f * 1e3
    theta_phase = np.mod(t, theta_T)/theta_T * 2*np.pi
    theta_phase_plot = np.mod(theta_phase + 2*np.pi, 2*np.pi)
    Itheta = (1 + np.cos(theta_phase))/2 * theta_amp


    # # Sensory tuning
    xtun = np.linspace(xmin, xmax, nn)
    np.random.seed(1)
    atun = np.array([0, np.pi] * int(nn / 2))
    atun_type = np.array([0, 1] * int(nn / 2)).astype(int)

    # # Directional weights
    w_dir = np.zeros((nn, nn))
    for atuntype_each in [0, 1]:
        atype_idx = np.where(atun_type == atuntype_each)[0]
        num_atype = atype_idx.shape[0]
        for i in range(num_atype):
            for j in range(i, num_atype):
                idx1, idx2 = atype_idx[i], atype_idx[j]
                xtun1, xtun2 = xtun[idx1], xtun[idx2]
                if atuntype_each == 0:  # rightward connection
                    w_dir[idx2, idx1] = gaufunc1d(xtun1, xtun2, wsd_angle, wmax_angle)
                if atuntype_each == 1:  # leftward connection
                    w_dir[idx1, idx2] = gaufunc1d(xtun1, xtun2, wsd_angle, wmax_angle)

    # # Positional weights
    w_pos = gaufunc1d(xtun.reshape(1, nn), xtun.reshape(nn, 1), wsd, wmax)

    # # Combined weights
    w = w_pos + w_dir

    # Network states
    izhi_a = np.ones(nn) * izhi_a_ex
    izhi_b = np.ones(nn) * izhi_b_ex
    izhi_c = np.ones(nn) * izhi_c_ex
    izhi_d = np.ones(nn) * izhi_d_ex
    v = np.ones(nn) * izhi_c
    u = np.zeros(nn)
    Isyn = np.zeros(nn)
    gex = np.zeros(nn)
    fidx_buffer = []
    SpikeDF_dict = dict(neuronid=[], tidxsp=[])
    if store_Activity:
        Isen_pop = np.zeros((numt, nn))
        Igoal_pop = np.zeros((numt, nn))
        Isyn_pop = np.zeros((numt, nn))
        Itotal_pop = np.zeros((numt, nn))

    # ============================================================================================
    # ======================================== Simulation Runtime ================================
    # ============================================================================================
    t1 = time.time()

    for i in range(numt):
        if i % 100 == 0:
            print('\rSimulation %d/%d'%(i, numt), flush=True, end='')
        # Behavioural
        run_x, run_a = traj_x[i], traj_a[i]

        # Sensory input: A box function modulated by oscillation phase shifted by 70 (290) deg
        ECtheta = (np.cos(theta_phase[i] + EC_phase) + 1) / 2
        Iangle = circgaufunc1d(run_a, atun, Iangle_diff)

        # Goal-directed input
        xdiff_goal = goal_x - run_x
        angle2goal = 0 if xdiff_goal > 0 else np.pi
        Iangle_goal = circgaufunc1d(angle2goal, atun, Imax_angle_goal)
        IanglePos_goal = gaufunc1d(run_x, xtun, Isd_goal, Iangle_goal)
        Ipos_goal = gaufunc1d(run_x, goal_x, Isd_goal, Imax_goal)
        # Ipos_goal = (xmax - np.abs(xdiff_goal))/xmax * Imax_goal
        Igoal = gaufunc1d(xtun, goal_x, Isd_goal, Ipos_goal + IanglePos_goal)


        Isen = boxfunc1d(run_x, xtun, Ipos_sd, Ipos_max + Iangle + 4.7) * ECtheta

        # Total Input
        Itotal = Isyn + Isen + Igoal - Itheta[i]

        # Izhikevich
        v += (0.04*v**2 + 5*v + 140 - u + Itotal) * dt
        u += izhi_a * (izhi_b * v - u) * dt
        fidx = np.where(v > V_thresh)[0]
        v[fidx] = izhi_c[fidx]
        u[fidx] = u[fidx] + izhi_d[fidx]
        fidx_buffer.append(fidx)


        if i > spdelay:  # 2ms delay
            delayed_fidx = fidx_buffer.pop(0)

            # Synaptic input (Excitatory)
            spike_sum = np.sum(w[:, delayed_fidx], axis=1) / nn
            gex += (-gex/tau_gex + spike_sum) * dt
            Isyn = gex * (V_ex - v)


        # Store data
        SpikeDF_dict['neuronid'].extend(list(fidx))
        SpikeDF_dict['tidxsp'].extend([i] * len(fidx))

        if store_Activity:
            Isen_pop[i, :] = Isen
            Isyn_pop[i, :] = Isyn
            Igoal_pop[i, :] = Igoal
            Itotal_pop[i, :] = Itotal

    print('\nSimulation time = %0.2fs'%(time.time()-t1))

    # # Storage
    BehDF['Itheta'] = Itheta
    BehDF['theta_phase'] = theta_phase
    BehDF['theta_phase_plot'] = theta_phase_plot

    SpikeDF = pd.DataFrame(SpikeDF_dict)
    SpikeDF['neuronx'] = SpikeDF['neuronid'].apply(lambda x : xtun[x])

    NeuronDF = pd.DataFrame(dict(neuronid=np.arange(nn), neuronx=xtun, neurona=atun,
                                 neurontype=["CA3"]*nn))

    if store_Activity:
        ActivityData = dict(Isen=Isen_pop, Igoal=Igoal_pop, Isyn=Isyn_pop,
                            Itotal=Itotal_pop)
    else:
        ActivityData = None
    if store_w is False:
        w = None
    MetaData = dict(nn=nn, w=w)

    alldata =dict(
        BehDF=BehDF, SpikeDF=SpikeDF, NeuronDF=NeuronDF, ActivityData=ActivityData, MetaData=MetaData, Config=config_dict
    )
    return alldata