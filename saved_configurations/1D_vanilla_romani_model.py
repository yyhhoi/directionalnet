# 1D track spiking romani model with STD and optional STF

from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pandas as pd
from scipy.signal import find_peaks
from pycircstat.descriptive import cdiff, mean as circmean
from library.visualization import customlegend
from library.linear_circular_r import rcc
import time
def gaufunc(x, mu, sd, outmax):
    return outmax * np.exp(-np.square(x - mu)/(sd**2))

def senfunc(x, mu, radius, amp):
    senamp = amp * (np.abs(x - mu) < radius)
    return senamp

mpl.rcParams['figure.dpi'] = 150
legendsize = 7


# Environment & agent
dt = 0.1  # 0.1ms
t = np.arange(0, 2e3, dt)
traj_x = t * 5 * 1e-3

# Theta inhibition
theta_amp = 7
theta_f = 10
theta_T = 1/theta_f * 1e3
theta_phase = np.mod(t, theta_T)/theta_T * 2*np.pi
theta_phase_plot = np.mod(theta_phase + 2*np.pi, 2*np.pi)

# Positional drive
EC_phase = 290
Ipos_max = 12 + (np.cos(np.deg2rad(EC_phase)) + 1)/2 * theta_amp
Ipos_sd = 0.5

# izhikevich's model parameters
izhi_a, izhi_b, izhi_c, izhi_d = 0.02, 0.2, -65, 0
V_ex, V_in = 60, -80
V_thresh = 30
tau_deadx = 10

# Sensory tuning
nn_ex = 100
nn_in = 100
nn = nn_ex + nn_in
xtun_ex = np.linspace(traj_x.min(), traj_x.max(), nn_ex)
xtun_in = np.linspace(traj_x.min(), traj_x.max(), nn_in)
xtun = np.concatenate([xtun_ex, xtun_in])

# # Weights
exex_posmax = 4  # 4
exex_possd = 2
w_exex = gaufunc(xtun_ex.reshape(1, nn_ex), xtun_ex.reshape(nn_ex, 1), exex_possd, exex_posmax)
exin_max = 2
w_exin = np.ones((nn_in, nn_ex)) * exin_max
inex_max = 10
w_inex = np.ones((nn_ex, nn_in)) * inex_max
inin_max = 0
w_inin = np.ones((nn_in, nn_in)) * inin_max
w = np.zeros((nn, nn))
w[:nn_ex, :nn_ex] = w_exex
w[:nn_ex, nn_ex:] = w_inex
w[nn_ex:, :nn_ex] = w_exin
w[nn_ex:, nn_ex:] = w_inin

# Synapses
tau_gex = 10
tau_gin = 10
U_stdx = 0.4  # 1    # 0.4  # 1
U_stfx = 0    # 1.4  # 0    # 2.3
tau_stdx = 1e3
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
deadx = np.zeros(nn)
spdelayx = np.zeros(nn)
spdelayDiffx = np.zeros(nn)
eg_neuronid = int(nn_ex/2)
spdf_dict = dict(neuronid=[], tidxsp=[])
prob_dict = dict(t=[], Isen=[], Itheta=[], Isyn=[], Itotal=[], v=[], syneff=[])

Isen_pop = np.zeros((t.shape[0], nn))
Itheta_pop = np.zeros((t.shape[0], nn))
Isyn_pop = np.zeros((t.shape[0], nn))
Itotal_pop = np.zeros((t.shape[0], nn))
v_pop = np.zeros((t.shape[0], nn))
syneff_pop = np.zeros((t.shape[0], nn))
deadx_pop = np.zeros((t.shape[0], nn))


# # Simulation runtime
t1 = time.time()
for i in range(t.shape[0]):
    # Behavioural
    pos_x = traj_x[i]
    t_now = t[i]

    # Sensory input
    Isen = senfunc(pos_x, xtun, Ipos_sd, Ipos_max) * (np.cos(theta_phase[i] + np.deg2rad(EC_phase)) + 1)/2
    # Isen = gaufunc(xtun, pos_x, Ipos_sd, Ipos_max)
    Isen[nn_ex:] = 0

    # Theta Input
    Itheta = (1 + np.cos(theta_phase[i]))/2 * theta_amp

    # Total Input
    Itotal = Isyn + Isen - Itheta

    # Izhikevich
    deadmask = deadx<0.9
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
    d_stdx_dt[nn_ex:] = 0
    stdx += d_stdx_dt * dt
    d_stfx_dt = (1 - stfx)/tau_stfx
    d_stfx_dt[fidx] = d_stfx_dt[fidx] + U_stfx * (2 - stfx[fidx])
    d_stfx_dt[nn_ex:] = 0
    stfx += d_stfx_dt * dt
    syneff = stdx * stfx

    # Spike delay counter
    spdelayx[fidx] = spdelayx[fidx] + spdelay
    diff = -np.sign(spdelayx)
    spdelayx += diff
    spdelayDiffx += -diff
    delayed_fidx = np.where(spdelayDiffx >= spdelay)[0]
    spdelayDiffx[delayed_fidx] = 0
    fidx_ex = delayed_fidx[delayed_fidx < nn_ex]
    fidx_in = delayed_fidx[delayed_fidx >= nn_in]

    # Synaptic input (Excitatory)
    spike_sum = np.sum(syneff[fidx_ex].reshape(1, -1) * w[:, fidx_ex], axis=1) / nn_ex
    gex += (-gex/tau_gex + spike_sum) * dt
    Isyn_ex = gex * (V_ex - v)

    # Synaptic input (Inhibitory)
    spike_sum = np.sum(w[:, fidx_in], axis=1) / nn_in
    gin += (-gin/tau_gin + spike_sum) * dt
    Isyn_in = gin * (V_in - v)
    Isyn = Isyn_ex + Isyn_in


    # Store data
    spdf_dict['neuronid'].extend(list(fidx))
    spdf_dict['tidxsp'].extend([i] * len(fidx))

    prob_dict['t'].append(t_now)
    prob_dict['Isen'].append(Isen[eg_neuronid])
    prob_dict['Itheta'].append(Itheta)
    prob_dict['Isyn'].append(Isyn[eg_neuronid])
    prob_dict['Itotal'].append(Itotal[eg_neuronid])
    prob_dict['v'].append(v[eg_neuronid])
    prob_dict['syneff'].append(syneff[eg_neuronid])

    v_pop[i, :] = v
    Isen_pop[i, :] = Isen
    Itheta_pop[i, :] = Itheta
    Isyn_pop[i, :] = Isyn
    Itotal_pop[i, :] = Itotal
    syneff_pop[i, :] = syneff
    deadx_pop[i, :] = deadx

t2 = time.time()
spdf = pd.DataFrame(spdf_dict)
probdf = pd.DataFrame(prob_dict)
print('Simulation time = %0.2fs'%(t2-t1))
spdf['neuronx'] = spdf['neuronid'].apply(lambda x : xtun[x])


fig, ax = plt.subplots(3, 4, figsize=(10, 6), facecolor='white')
gs = ax[1, 0].get_gridspec()
for axeach in ax[1:, :].ravel():
    axeach.remove()
axbig = fig.add_subplot(gs[1:, :])

# Population raster
for neuronid in range(xtun.shape[0]):
    tidxsp_neuron = spdf[spdf['neuronid'] == neuronid]['tidxsp']
    tsp_neuron = t[tidxsp_neuron]
    neuronx = xtun[neuronid]
    if neuronid >= nn_ex:
        ax[0, 0].eventplot(tsp_neuron, lineoffsets=(neuronid-nn_ex)/xtun.max()+xtun.max(), linelengths=xtun[1]-xtun[0], linewidths=0.75, color='m')
    else:
        ax[0, 0].eventplot(tsp_neuron, lineoffsets=neuronx, linelengths=xtun[1]-xtun[0], linewidths=0.75, color='green')
ax[0, 0].plot(t, traj_x, c='k', linewidth=0.75)


# Phase precession
tidxsp_eg = spdf.loc[spdf['neuronid']==eg_neuronid, 'tidxsp']
tsp_eg, phasesp_eg = t[tidxsp_eg], theta_phase[tidxsp_eg]
xsp_eg = traj_x[tidxsp_eg]
xspmin, xsprange = xsp_eg.min(), xsp_eg.max() - xsp_eg.min()
xsp_norm_eg = (xsp_eg-xspmin)/xsprange
regress = rcc(xsp_norm_eg, phasesp_eg)
rcc_c, rcc_m = regress['phi0'], regress['aopt']
xdum = np.linspace(xsp_norm_eg.min(), xsp_norm_eg.max(), 100)
ydum = xdum * rcc_m * 2 * np.pi + rcc_c
ax[0, 1].scatter(xsp_norm_eg, phasesp_eg, marker='|', s=1)
ax[0, 1].plot(xdum, ydum, c='k', linewidth=0.75)
ax[0, 1].annotate('y= %0.2fx + %0.2f'%(2*np.pi*rcc_m, rcc_c), xy=(0.5, 0.8), xycoords='axes fraction', fontsize=legendsize)
ax[0, 1].set_ylim(0, 2*np.pi)
ax[0, 1].set_title('Phase range = %0.2f - %0.2f (%0.2f) '%(phasesp_eg.min(), phasesp_eg[0], phasesp_eg[0]-phasesp_eg.min()), fontsize=legendsize)
ax[0, 1].set_yticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])
ax[0, 1].set_yticklabels(['0', '$\pi/2$', '$\pi$', '$1.5\pi$', '$2\pi$'])



# current
ax[0, 2].plot(probdf['t'], probdf['Itotal'], label='Itotal', linewidth=0.75)
ax[0, 2].plot(probdf['t'], probdf['Isyn'], label='Isyn', linewidth=0.75)
ax[0, 2].set_ylim(-25, 20)
customlegend(ax[0, 2], fontsize=legendsize)
axtsp10 = ax[0, 2].twinx()
axtsp10.eventplot(tsp_eg, lineoffsets = 1, linelength=0.05, linewidths=0.5, color='r')
axtsp10.set_ylim(0, 1.2)
axtsp10.axis('off')

# STD
ax[0, 3].plot(probdf['t'], probdf['Isyn'], label='Isyn', linewidth=0.75, color='orange')
axsyneff = ax[0, 3].twinx()
axsyneff.plot(probdf['t'], probdf['syneff'], label='STD', color='r', linewidth=0.75)
axsyneff.set_ylim(-0.1, 2.1)
ax[0, 3].set_ylim(-20, 15)
customlegend(axsyneff, fontsize=legendsize)
theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]

for i in theta_cutidx:
    ax[0, 0].axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)
    ax[0, 2].axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)
for ax_each in ax.ravel():
    ax_each.tick_params(labelsize=legendsize)
axsyneff.tick_params(labelsize=legendsize)

# Population raster
tt, traj_xx = np.meshgrid(t, xtun[:nn_ex])
mappable = axbig.pcolormesh(tt, traj_xx, syneff_pop[:, :nn_ex].T, shading='auto', vmin=0, vmax=2, cmap='seismic')
for neuronid in range(xtun.shape[0]):
    tidxsp_neuron = spdf[spdf['neuronid'] == neuronid]['tidxsp']
    tsp_neuron = t[tidxsp_neuron]
    neuronx = xtun[neuronid]
    if neuronid < nn_ex:
        if neuronid == eg_neuronid:
            ras_c = 'r'
        else:
            ras_c = 'green'
        axbig.eventplot(tsp_neuron, lineoffsets=neuronx, linelengths=xtun[1]-xtun[0], linewidths=0.75, color=ras_c)

axbig.plot(t, traj_x, c='k', linewidth=0.75)
theta_cutidx = np.where(np.diff(theta_phase_plot) < -6)[0]
for i in theta_cutidx:
    axbig.axvline(t[i], c='gray', linewidth=0.75, alpha=0.5)

axbig.annotate('Gray lines = Theta phase 0\nEC phase shift = %d deg'%(EC_phase),
               xy=(0.02, 0.9), xycoords='axes fraction', fontsize=12)
plt.colorbar(mappable, ax=axbig)
fig.tight_layout()
# plt.show()

save_dir = 'plots/ECphases_STD10_STF23_Compen'
os.makedirs(save_dir, exist_ok=True)
fig.savefig(join(save_dir, 'ECphase%d.png'%(EC_phase)), dpi=150)


