import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from library.Tempotron import Tempotron
from library.shared_vars import sim_results_dir, plots_dir
from library.utils import load_pickle
from library.visualization import plot_tempotron_traces, customlegend
# ====================================== Global params and paths ==================================
legendsize = 8
plt.rcParams.update({'font.size': legendsize,
                     "axes.titlesize": legendsize,
                     'axes.labelpad': 0,
                     'axes.titlepad': 0,
                     'xtick.major.pad': 0,
                     'ytick.major.pad': 0,
                     'lines.linewidth': 1,
                     'figure.figsize': (5.2, 6.1),
                     'figure.dpi': 300,
                     'axes.spines.top': False,
                     'axes.spines.right': False,

                     })

jitter_times = 100
jitter_ms = 0.25
project_tag = 'Jit%d_%0.1fms'%(jitter_times, jitter_ms)
sim_tag = 'fig6'
simdata_dir = join(sim_results_dir, sim_tag)
data_dir = join(simdata_dir, project_tag)
plot_dir = join(plots_dir, sim_tag)
os.makedirs(plot_dir, exist_ok=True)

# ====================================== Plot setting ==================================
# Top 1st column: Training scheme
ax_h1 = 0.590/5
ax_w1 = 0.75/5
hgap1 = 0.02
wgap1 = 0.02
xshift1 = 0.07
yshift1 = -0.015

trainSeq_h = 0.075
trainSeq_w = 1/5
trainSeq_xshift = 0.055
inseq_yshift = -0.035
exseq_yshift = -0.02

# 2nd & 3rd column: Intrinsic raster & Tempotron illustration
axtem_w = 1.3/5  # The next half 1.375 is for drawing
axtem_x = ax_w1
tem_xshift = xshift1 + ax_w1 - 0.06
tem_yshift = -0.03
tem_hgap = 0.02
tem_wgap = 0.1
trace_xshift = -0.05


# 3rd col, testing scheme and ACC
ax_h31 = 0.5/5
ax_w31 = 0.5/5
xshift31 = -0.015
yshift31 = -0.028
hgap31 = 0.02
wgap31 = 0

ax_h32 = 0.7/5
ax_w32 = 0.7/5
xshift32 = -0.035
yshift32 = -0.065
hgap32 = 0.02
wgap32 = 0

# Population Raster plots
ax_h_ras = 1.5/5
ax_w_ras = 1/12  # 0.5/10.5 for 1d weights
hgap_ras = 0.06
ras_xshift = 0.075
ras_yshift = -0.08
ras_in_yshift = 0 + ras_yshift
ras_ex_yshift = 0.025 + ras_yshift

# Child plots in the Population raster plots
trace_h = 0.04
trace_hshift = 0.03
w1d_w = 0.5/12

fig = plt.figure()
ax_trainScheme = fig.add_axes([ax_w1 * 0 + wgap1/2 + xshift1, 1 - ax_h1 * 1 + hgap1/2 + yshift1, ax_w1 - wgap1, ax_h1 - hgap1])
ax_trainSeqIn = fig.add_axes([ax_w1 * 0 + wgap1/2 + trainSeq_xshift, 1 - ax_h1 - trainSeq_h + hgap1/2 + yshift1 + inseq_yshift, trainSeq_w - wgap1, trainSeq_h - hgap1])
ax_trainSeqEx = fig.add_axes([ax_w1 * 0 + wgap1/2 + trainSeq_xshift, 1 - ax_h1 - trainSeq_h*2 + hgap1/2 + yshift1 + exseq_yshift, trainSeq_w - wgap1, trainSeq_h - hgap1])

ax_tem   = fig.add_axes([axtem_x + tem_wgap/2 + tem_xshift, 1 - ax_h1 * 2 + tem_hgap/2 + tem_yshift, axtem_w - tem_wgap, ax_h1*2 - tem_hgap])
ax_trace = fig.add_axes([axtem_x + axtem_w + tem_wgap/2 + tem_xshift + trace_xshift, 1 - ax_h1 * 2 + tem_hgap/2 + tem_yshift, axtem_w - tem_wgap, ax_h1*0.5])

ax_x3 = axtem_x + axtem_w + tem_wgap/2 + tem_xshift + trace_xshift + axtem_w
ax_testScheme = fig.add_axes([ax_x3 + wgap31/2 + xshift31, 1 - ax_h31 * 1 + hgap31/2 + yshift31, ax_w31 - wgap31, ax_h31 - hgap31])
ax_acc = fig.add_axes([ax_x3 + wgap32/2 + xshift32, 1 - ax_h31 * 2 + hgap32/2 + yshift32, ax_w32 - wgap32, ax_h32 - hgap32])

ax_inRas180 = [fig.add_axes([ax_w_ras * i + ras_xshift, 1 - ax_h1*2 - ax_h_ras * 1 + trace_h + hgap_ras/2 + ras_in_yshift, ax_w_ras, ax_h_ras - hgap_ras - trace_h]) for i in range(10)]
ax_exRas180 = [fig.add_axes([ax_w_ras * i + ras_xshift, 1 - ax_h1*2 - ax_h_ras * 2 + trace_h + hgap_ras/2 + ras_ex_yshift, ax_w_ras, ax_h_ras - hgap_ras - trace_h]) for i in range(10)]

ax_inRasW1d = fig.add_axes([ax_w_ras * 10 + ras_xshift, 1 - ax_h1*2 - ax_h_ras * 1 + trace_h + hgap_ras/2 + ras_in_yshift, w1d_w, ax_h_ras - hgap_ras - trace_h])
ax_exRasW1d = fig.add_axes([ax_w_ras * 10 + ras_xshift, 1 - ax_h1*2 - ax_h_ras * 2 + trace_h + hgap_ras/2 + ras_ex_yshift, w1d_w, ax_h_ras - hgap_ras - trace_h])

ax_inRasTrace = [fig.add_axes([ax_w_ras * i + ras_xshift, 1 - ax_h1*2 - ax_h_ras * 1 + trace_hshift + ras_in_yshift, ax_w_ras, trace_h]) for i in range(10)]
ax_exRasTrace = [fig.add_axes([ax_w_ras * i + ras_xshift, 1 - ax_h1*2 - ax_h_ras * 2 + trace_hshift + ras_ex_yshift, ax_w_ras, trace_h]) for i in range(10)]


ax_cbar = fig.add_axes([0.88, 0.68, 0.07, 0.005])


ax_exinRas = [ax_inRas180, ax_exRas180]
ax_exinRasTrace = [ax_inRasTrace, ax_exRasTrace]
ax_exinRasW1d = [ax_inRasW1d, ax_exRasW1d]
ax_exinSeq = [ax_trainSeqIn, ax_trainSeqEx]


fig_exinW2d, ax_exinW2D = plt.subplots(2, 1, figsize=(1.5, 3))
# ax_cbar = fig_exinW2d.add_axes([0.1, 0.9, 0.8, 0.02])
fig_Wcompare, ax_Wcompare = plt.subplots(figsize=(5,5))

# ====================================== Determine weight color scale ==================================
all_temNw = np.concatenate([np.load(join(data_dir, 'w_%s_ex.npy'%(project_tag))), np.load(join(data_dir, 'w_%s_in.npy'%(project_tag)))])
wmin, wmax = all_temNw.min(), all_temNw.max()
abswmax = max(np.abs(wmin), np.abs(wmax))
norm = mpl.colors.Normalize(vmin=-abswmax, vmax=abswmax)
val2cmap = cm.ScalarMappable(norm=norm, cmap=cm.jet)

# ====================================== Plot ==================================

exin_c = {'ex':'tomato', 'in':'royalblue'}

for exin_i, exintag in enumerate(['in', 'ex']):

    # # Load and organize data
    simdata = load_pickle(join(simdata_dir, 'fig6_%s.pkl'%(exintag)))
    dataset = load_pickle(join(data_dir, 'data_train_test_ori_%s_%s.pickle'%(project_tag, exintag)))
    temNw =  np.load(join(data_dir, 'w_%s_%s.npy'%(project_tag, exintag)))
    TrainResult = load_pickle(join(data_dir, 'TrainResult_%s_%s.pickle'% (project_tag, exintag)))

    X_test_ori = dataset['X_test_ori']
    Y_test_ori = dataset['Y_test_ori']
    trajtype_test_ori = dataset['trajtype_test_ori']
    X_train_ori = dataset['X_train_ori']
    Y_train_ori = dataset['Y_train_ori']
    trajtype_train_ori = dataset['trajtype_train_ori']
    theta_bounds = dataset['theta_bounds']
    all_nidx = dataset['all_nidx']

    N = len(X_train_ori[0])
    num_trajtypes = trajtype_test_ori.max()+1
    trajtype_ax = np.arange(num_trajtypes)
    a_ax = trajtype_ax/num_trajtypes*2*np.pi
    deg_ax = np.around(np.rad2deg(a_ax), 0).astype(int)

    NeuronDF = simdata['NeuronDF']
    xxtun1d = NeuronDF['neuronx'].to_numpy()
    yytun1d = NeuronDF['neurony'].to_numpy()

    # # Set up tempotron and run predictions
    Vthresh = 2
    temN_tax = np.arange(0, 100, 1)
    temN = Tempotron(N=N, lr=0.01, Vthresh=Vthresh, tau=5, tau_s=5/4, w_seed=0)
    temN.w = temNw
    Y_train_ori_pred, kout_train_ori, tspout_train_ori = temN.predict(X_train_ori, temN_tax, shunt=True)
    Y_test_ori_pred, kout_test_ori, tspout_test_ori = temN.predict(X_test_ori, temN_tax, shunt=True)

    # # Plot training sequence (Ex+In) examples
    yoffset = 20 if exintag=='in' else -20
    yneurons = yytun1d[all_nidx]
    idx_down = np.min(np.where(yneurons > yoffset)[0])
    idx_up = idx_down + 20
    tsp_yfixed = X_train_ori[:, idx_down:idx_up]
    for yi in range(20):
        tsp_train = np.concatenate([tsp_yfixed[mi, yi] + (mi * 100) for mi in range(tsp_yfixed.shape[0])])
        ax_exinSeq[exin_i].scatter(tsp_train, [yi]*tsp_train.shape[0], s=1, lw=0.5, c=exin_c[exintag], marker='|')
    for thetai in range(0, 10):
        ax_exinSeq[exin_i].axvline(thetai * 100, c='k', lw=0.1)
    ax_exinSeq[exin_i].set_xlim(200, 600)
    ax_exinSeq[exin_i].set_ylim(0, 20)
    ax_exinSeq[exin_i].axis('off')

    # # Plot Across-Y sequence example
    if exintag == 'in':
        chosen_cycle = 5
        tsp_ally = np.concatenate(X_train_ori[chosen_cycle, :])
        nidx_ally = np.concatenate([ np.ones(X_train_ori[chosen_cycle, yi].shape[0]) * yi for yi in range(N)])
        ax_tem.scatter(tsp_ally, nidx_ally, s=1, lw=1, c=exin_c[exintag], marker='|')
        ysepN_ax = np.arange(0, 400, 20).astype(int)
        ysep_ax = ysepN_ax -0.5
        for ysep in ysep_ax:
            ax_tem.axhline(ysep, color='gray', lw=0.5)

        ax_tem.set_xlabel('Time (ms)')
        ax_tem.set_xticks(np.arange(0, 61, 20))
        ax_tem.set_xticks(np.arange(0, 61, 10), minor=True)
        ax_tem.set_xlim(0, 60)
        ax_tem.set_ylabel('y (cm)')
        ax_tem.set_yticks(ysepN_ax + 10)
        ax_tem.set_yticklabels(np.around(yytun1d[all_nidx[ysepN_ax+10]], 0).astype(int).astype(str))
        ax_tem.set_ylim(120, 260)

        ax_trace.plot(temN_tax, kout_train_ori[chosen_cycle], lw=1, color='k')
        ax_trace.axhline(Vthresh, color='gray', lw=0.5)
        ax_trace.eventplot([tspout_train_ori[chosen_cycle][0]], lineoffsets=3.2, linelengths=1, color='r', linewidths=1, zorder=3)
        ax_trace.spines.left.set_visible(False)
        ax_trace.set_xlabel('Time (ms)')
        ax_trace.set_xticks(np.arange(0, 61, 20))
        ax_trace.set_xticks(np.arange(0, 61, 10), minor=True)
        ax_trace.set_xlim(0, 60)
        ax_trace.set_yticks([])

    # # Plot Accuracies
    # The prediction is correct if the trajectory has at least one theta cycle that fires
    # Each jittered realization of spikes is a sample for accuracy calculation
    acc = np.zeros(num_trajtypes)
    for trajtype, dftmp in TrainResult.groupby('trajtype_test'):
        jitnum = int(dftmp.jitbatch_test.max() + 1)
        jitPred = np.zeros(jitnum).astype(bool)
        for jiti, dftmp2 in dftmp.groupby('jitbatch_test'):
            jitPred[jiti] = dftmp2.Y_pred_test.sum() > 0.5
        acc[trajtype] = jitPred.mean()
    acclabel = 'With-loop' if exintag=='in' else 'No-loop'
    ax_acc.plot(deg_ax, acc, color=exin_c[exintag], label=acclabel )

    # # Plot 2D Weights
    im = ax_exinW2D[exin_i].scatter(xxtun1d[all_nidx], yytun1d[all_nidx], c=temNw, cmap=cm.jet, vmin=-abswmax, vmax=abswmax, s=0.5, lw=0.5)
    ax_exinW2D[exin_i].set_xticks([])
    ax_exinW2D[exin_i].set_yticks([])
    for spinekey in ['bottom', 'top', 'right', 'left']:
        ax_exinW2D[exin_i].spines[spinekey].set_visible(True)
        ax_exinW2D[exin_i].spines[spinekey].set_color(exin_c[exintag])
        ax_exinW2D[exin_i].spines[spinekey].set_linewidth(2)
    border = 0.5
    ax_exinW2D[exin_i].set_xlim(-10-border, 10+border)
    if exintag == 'ex':
        ax_exinW2D[exin_i].set_ylim(-30-border, -10+border)
    else:
        ax_exinW2D[exin_i].set_ylim(10 - border, 30 + border)
    if exintag == 'in':
        cbar = fig.colorbar(im, cax=ax_cbar, orientation='horizontal')
        cbar.ax.set_xticks([-0.06, 0, 0.06])
        cbar.ax.set_xticklabels(['', '0', ''])
        cbar.ax.set_xlabel('$w$', labelpad=-22)

    comparenidx = np.where(np.abs(temNw) > 0.015)[0]
    compareshift = -20 if exintag=='in' else 20
    ax_Wcompare.scatter(yytun1d[all_nidx[comparenidx]]+compareshift, temNw[comparenidx], c=exin_c[exintag], marker='.', s=4, alpha=0.5)
    ax_Wcompare.axvline(np.sum((yytun1d[all_nidx[comparenidx]]+compareshift) * temNw[comparenidx]/np.sum(temNw[comparenidx])), color=exin_c[exintag])

    # # Plot 180 deg test traces for ex & in
    traj_deg = 180
    chosen_trajtype = trajtype_ax[deg_ax==traj_deg].squeeze()
    mask = trajtype_test_ori == chosen_trajtype
    X_test_chosen = X_test_ori[mask]
    Y_test_chosen = Y_test_ori[mask]
    Y_test_ori_pred_chosen = Y_test_ori_pred[mask]
    kout_test_chosen = kout_test_ori[mask]
    tspout_test_chosen = tspout_test_ori[mask]
    plot_tempotron_traces(ax_exinRas[exin_i], ax_exinRasTrace[exin_i], ax_exinRasW1d[exin_i], N,
                          X=X_test_chosen, temN_tax=temN_tax, temNw=temNw, Vthresh=Vthresh,
                          all_nidx=all_nidx, yytun1d=yytun1d, kout_all=kout_test_chosen, tspout_all=tspout_test_chosen,
                          val2cmap=val2cmap, exintag=exintag, exin_c=exin_c)
ax_exRasTrace[5].set_xlabel('Time (ms)')

# # Plot training trajectories outside the loop
xxtun1dca3, yytun1dca3 = xxtun1d[NeuronDF['neurontype']=='CA3'], yytun1d[NeuronDF['neurontype']=='CA3']
mos_actx = np.arange(-20, 21, 2)  # 0 degree
mos_acty = np.zeros(mos_actx.shape[0]) + 20
zzgauca3 = np.zeros((xxtun1dca3.shape[0], mos_actx.shape[0]))
for mosi in range(mos_actx.shape[0]):
    actx, acty = mos_actx[mosi], mos_acty[mosi]
    zzgauca3[:, mosi] = np.exp(- ((xxtun1dca3-actx)**2 + (yytun1dca3-acty)**2)  / (2 * (2 ** 2)))
zzgauca3 = np.max(zzgauca3, axis=1)
ax_trainScheme.arrow(-20, 20, 40, 0, width=0.01, head_width=2, color=exin_c['in'])
ax_trainScheme.scatter((0, 0), (20, -20), c='k', s=8, lw=0.5, marker='x')
ax_trainScheme.plot((-10, 10, 10, -10, -10), (30, 30, 10, 10, 30), linewidth=0.5, color=exin_c['in'])   # In
ax_trainScheme.plot((-10, 10, 10, -10, -10), (-10, -10, -30, -30, -10), linewidth=0.5, color=exin_c['ex'])   # Ex
ax_trainScheme.set_xlabel('x (cm)')
ax_trainScheme.set_ylabel('y (cm)')
ax_trainScheme.set_xlim(-40, 40)
ax_trainScheme.set_ylim(-40, 40)
ax_trainScheme.set_xticks(np.arange(-40, 41, 20))
ax_trainScheme.set_xticklabels(['', '-20', '0', '20', ''])
ax_trainScheme.set_xticks(np.arange(-40, 41, 5), minor=True)
ax_trainScheme.set_yticks(np.arange(-40, 41, 20))
ax_trainScheme.set_yticklabels(['', '-20', '0', '20', ''])
ax_trainScheme.set_yticks(np.arange(-40, 41, 5), minor=True)

# # Plot testing illustration (for drawing later)
ax_testScheme.scatter(xxtun1d[all_nidx], yytun1d[all_nidx], c='0.9', s=0.5, lw=0.5)
circ_t = np.linspace(0, 2*np.pi, 20)
# ax_testScheme.plot(2 * np.cos(circ_t), 2 * np.sin(circ_t)-20, c='k', lw=0.5)
ax_testScheme.axis('off')

# # Asthestics for accuracy
customlegend(ax_acc, loc='center')
ax_acc.set_xticks([0, 180, 360])
ax_acc.set_xticks([90, 270], minor=True)
ax_acc.set_xticklabels(['0', '', '360'])
ax_acc.set_yticks([0, 0.5, 1])
ax_acc.set_yticklabels(['0', '', '1'])
ax_acc.set_yticks(np.arange(0, 1, 0.1), minor=True)
ax_acc.set_ylim(-0.05, 1.1)
ax_acc.set_ylabel('ACC', labelpad=-5)
ax_acc.set_xlabel(r'$\varphi (\circ)$', labelpad=-5)

fig.savefig(join(plot_dir, 'fig6.png'), dpi=300)
fig.savefig(join(plot_dir, 'fig6.pdf'))
fig.savefig(join(plot_dir, 'fig6.svg'))

fig_exinW2d.savefig(join(plot_dir, 'fig6_W2D.png'), dpi=150)

fig_Wcompare.savefig(join(plot_dir, 'fig6_Wcompare.png'), dpi=150)