import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
import torch
from scipy.interpolate import interp1d
from pycircstat.descriptive import cdiff, mean as circmean
from library.linear_circular_r import rcc

def cal_hd(x, y):
    hd = torch.angle(torch.diff(x) + 1j * np.diff(y))  # angle is converted to (-pi, pi)
    hd = torch.cat([hd, hd[[-1]]])
    return hd

def get_nidx(x, y, a, xxtun, yytun, aatun):
    nidx = torch.argmin((xxtun - x) ** 2 + (yytun - y) ** 2 + (aatun - a) ** 2)
    return nidx.item()

def fr_transfer(I):
    """for alpha = 1, In-place operation
    """
    I[I <= 4] = torch.log(1 + torch.exp(I[I <= 4]))
    return I


test_traj_dir = 'data/test_traj'
behdf = pd.read_pickle(join(test_traj_dir, 'behdf.pickle'))


# # Neuron's tunning
xmin, xmax, nx = -40, 40, 21  # 12.5 neuron per unit
ymin, ymax, ny = -40, 40, 21
xtun = torch.linspace(xmin, xmax, nx)
ytun = torch.linspace(ymin, ymax, ny)
atun = torch.tensor([0, np.pi/2, np.pi, -np.pi/2])
na = atun.shape[0]
xxtun2d, yytun2d, aatun2d = torch.meshgrid(xtun, ytun, atun)
xxtun1d, yytun1d, aatun1d = xxtun2d.flatten(), yytun2d.flatten(), aatun2d.flatten()
n_total = nx * ny * na


# # Simulation time
dt = 1e-3  # 1ms
total_t = behdf.t.max() # 8cm/s
t_s = torch.from_numpy(behdf['t'].to_numpy())

# # running trajectory
traj_x = torch.from_numpy(behdf['x'].to_numpy())
traj_y = torch.from_numpy(behdf['y'].to_numpy())
traj_hd = torch.from_numpy(behdf['angle'].to_numpy())
traj = torch.stack([traj_x, traj_y, traj_hd])


# # Positional & directional input
Ipos_max, Ipos_sd = 10, 0.6/np.pi*40  #in T-maze, moving task
Iangle_max, Iangle_sd = 10, 1.25


# # Excitatory synapse 
wmax_ex = 400  # 66.67
wmax_offset = 300  # 50
w_ex_pos_sd = 0.5/np.pi*40  # 0.3/pi*40
postun = torch.stack([xxtun1d, yytun1d]).T
postun_repeatrows = torch.stack([postun]*n_total)
postun_repeatcols = postun_repeatrows.permute([1, 0, 2])
syn_posdiff = postun_repeatcols - postun_repeatrows  # post - pre
syn_posdiff_expo = -torch.sqrt(torch.sum(torch.square(syn_posdiff), axis=2))/w_ex_pos_sd
w_ex = wmax_ex * torch.exp(syn_posdiff_expo) - wmax_offset


# # Base and theta current
I_base = 10
I_theta_amp, theta_freq = 12, 8

# # Soma and STP
tau_m = 10e-3
tau_R = 0.8
U = 0.6


# # Initialize
m = torch.zeros(n_total)
stpx = torch.ones(n_total)
I_syn = torch.zeros(n_total)
allm = torch.zeros((t_s.shape[0], n_total))

indata_dict = dict(tidx=[], t=[], x=[], y=[], angle=[], I_phase=[], I_theta=[])

# # Simulate
for tidx in range(t_s.shape[0]):
    print('\rSim %0.2f/%0.2f' % (tidx, t_s.shape[0]), flush=True, end='')

    run_state = traj[:, tidx]
    run_x, run_y, run_hd = run_state
    t_each_s = t_s[tidx]

    # Theta input
    I_phase = np.mod(2*np.pi*theta_freq * t_each_s, 2*np.pi)
    I_theta = torch.cos(I_phase) * I_theta_amp

    # Sensory input
    pos_dist = -torch.sqrt(torch.square(run_x-xxtun1d) + torch.square(run_y-yytun1d)) / Ipos_sd
    angle_dist = -torch.square(torch.from_numpy(cdiff(run_hd, aatun1d))) / 2 / (Iangle_sd**2)
    I_sen = Ipos_max * torch.exp(pos_dist) + Iangle_max * torch.exp(angle_dist + pos_dist)

    # Sum of inputs
    I_ext = I_base + I_theta + I_sen
    total_input = I_ext + I_syn

    # Soma update
    dmdt = (-m + fr_transfer(total_input))/tau_m
    m += dmdt * dt
    
    # STP update
    dstpxdt = (1-stpx)/tau_R - U * stpx * m
    stpx += dstpxdt * dt
    # stpx = torch.zeros(n_total)
    
    # Synaptic input
    I_syn = torch.sum(w_ex * m.reshape(1, -1) * stpx.reshape(1, -1), axis=1) / n_total


    # Store data
    indata_dict["tidx"].append(tidx)
    indata_dict["t"].append(t_each_s.item())
    indata_dict["x"].append(run_x.item())
    indata_dict["y"].append(run_y.item())
    indata_dict["angle"].append(run_hd.item())
    indata_dict["I_phase"].append(I_phase.item())
    indata_dict["I_theta"].append(I_theta.item())
    allm[tidx, :] = m.clone()
    if np.isnan(m).sum()>0:
        break

NeuronPos = pd.DataFrame(dict(neuronx=xxtun1d, neurony=yytun1d, neurona=aatun1d))
Indata = pd.DataFrame(indata_dict)
simdata = dict(NeuronPos=NeuronPos, Indata=Indata, theta_freq=theta_freq, w=w_ex, Activity=allm.numpy())

with open(join('data/simulated.pickle'), 'wb') as fh:
    pickle.dump(simdata, fh)