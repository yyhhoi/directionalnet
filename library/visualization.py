import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
from pycircstat import mean as cmean

from library.comput_utils import rcc_wrapper
from library.linear_circular_r import rcc

def customlegend(ax, handlelength=1.2, linewidth=1.2, handletextpad=0.1, **kwargs):
    leg = ax.legend(handlelength=handlelength, labelspacing=0.1, handletextpad=handletextpad,
                    borderpad=0.1, frameon=False, **kwargs)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(linewidth)
    return ax

def plot_popras(ax, SpikeDF, t, allnidx, bestnidx, worstnidx, best_c, worst_c, ras_c='gray'):
    for counti, neuronid in enumerate(allnidx):
        tidxsp_neuron = SpikeDF[SpikeDF['neuronid'] == neuronid]['tidxsp']
        tsp_neuron = t[tidxsp_neuron]
        if neuronid == bestnidx:
            ras_c = best_c
        elif neuronid == worstnidx:
            ras_c = worst_c
        else:
            ras_c = 'gray'
        ax.eventplot(tsp_neuron, lineoffsets=counti, linelengths=1, linewidths=0.5, color=ras_c)
        if tidxsp_neuron.shape[0] < 1:
            continue


def plot_phase_precession(ax, dsp, phasesp, s, c, label='', fontsize=8, plotmeanphase=False, statxy=(0.3, 0.75)):
    (dsp_norm, phasesp), (onset, slope), (xdum, ydum) = rcc_wrapper(dsp, phasesp)

    ax.scatter(dsp_norm, phasesp, marker='.', s=s, c=c, label=label)
    if plotmeanphase:
        mean_phasesp = cmean(phasesp)
        ax.axhline(mean_phasesp, xmin=0, xmax=0.3, linewidth=1, c='gray')
    ax.plot(xdum, ydum, linewidth=0.75, c='gray')
    precess_txt = 'y=%0.2fx\n    +%0.2f'%(slope, onset)
    ax.annotate(precess_txt, xy=statxy, xycoords='axes fraction', fontsize=fontsize)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '', '1'], fontsize=fontsize)
    ax.set_xticks(np.arange(0, 1, 0.1), minor=True)
    ax.set_ylim(0, 2*np.pi)
    ax.set_yticks([0, np.pi/2, np.pi, np.pi*3/2, 2*np.pi])
    ax.set_yticklabels(['0', '', '$\pi$', '', '$2\pi$'])
    ax.set_yticks(np.arange(0, 2*np.pi, np.pi/8), minor=True)
    return None


def plot_sca_onsetslope(fig, ax, onset_best, slope_best, onset_worst, slope_worst, onset_lim, slope_lim, direct_c):
    onset_bestmu, onset_worstmu = cmean(onset_best), cmean(onset_worst)
    slope_bestmu, slope_worstmu = np.median(slope_best), np.median(slope_worst)

    # # Slopes and onsets of best-worst neurons
    ax.scatter(onset_best, slope_best, marker='.', c=direct_c[0], s=4)
    ax.scatter(onset_worst, slope_worst, marker='.', c=direct_c[1], s=4)
    ax.set_xlim(onset_lim[0], onset_lim[1])
    ax.set_ylim(slope_lim[0], slope_lim[1])


    # # Marginal onset and slope
    axposori = ax.get_position()
    onsetbins = np.linspace(onset_lim[0], onset_lim[1], 30)
    ax_maronset = fig.add_axes([axposori.x0, axposori.y0+axposori.height, axposori.width, axposori.height * 0.3])
    binsonsetbest, _, _ = ax_maronset.hist(onset_best, bins=onsetbins, density=True, histtype='step', color=direct_c[0], linewidth=0.75)
    binsonsetworst, _, _ = ax_maronset.hist(onset_worst, bins=onsetbins, density=True, histtype='step', color=direct_c[1], linewidth=0.75)
    ax_maronset.axvline(onset_bestmu, ymin=0.75, ymax=0.95, linewidth=0.75, color=direct_c[0])
    ax_maronset.axvline(onset_worstmu, ymin=0.75, ymax=0.95, linewidth=0.75, color=direct_c[1])
    ax_maronset.set_xlim(onset_lim[0], onset_lim[1])
    ax_maronset.set_ylim(0, np.max([binsonsetbest.max(), binsonsetworst.max()])*1.5)
    ax_maronset.axis('off')

    slopebins = np.linspace(slope_lim[0], slope_lim[1], 30)
    ax_marslope = fig.add_axes([axposori.x0+axposori.width, axposori.y0, axposori.width * 0.3, axposori.height])
    binsslopebest, _, _ = ax_marslope.hist(slope_best, bins=slopebins, density=True, histtype='step', color=direct_c[0], linewidth=0.75, orientation='horizontal')
    binsslopeworst, _, _ = ax_marslope.hist(slope_worst, bins=slopebins, density=True, histtype='step', color=direct_c[1], linewidth=0.75, orientation='horizontal')
    ax_marslope.axhline(slope_bestmu, xmin=0.75, xmax=0.95, linewidth=0.75, color=direct_c[0])
    ax_marslope.axhline(slope_worstmu, xmin=0.75, xmax=0.95, linewidth=0.75, color=direct_c[1])
    ax_marslope.set_xlim(0, np.max([binsslopebest.max(), binsslopeworst.max()])*1.5)
    ax_marslope.set_ylim(slope_lim[0], slope_lim[1])
    ax_marslope.axis('off')


def plot_marginal_phase(ax, phasesp_best, phasesp_worst, direct_c, fontsize):
    phasesp_bestmu, phasesp_worstmu = cmean(phasesp_best), cmean(phasesp_worst)
    phasebins = np.linspace(0, 2*np.pi, 30)
    binsphasebest, _, _ = ax.hist(phasesp_best, bins=phasebins, density=True, histtype='step', color=direct_c[0], linewidth=0.75)
    binsphaseworst, _, _ = ax.hist(phasesp_worst, bins=phasebins, density=True, histtype='step', color=direct_c[1], linewidth=0.75)
    ax.axvline(phasesp_bestmu, ymin=0.75, ymax=0.9, linewidth=0.75, color=direct_c[0])
    ax.axvline(phasesp_worstmu, ymin=0.75, ymax=0.9, linewidth=0.75, color=direct_c[1])
    ax.set_xlabel('Spike phase (rad)', fontsize=fontsize, labelpad=0)
    ax.set_xticks(np.arange(0, np.pi*2+0.01, np.pi/2))
    ax.set_xticklabels(['0', '', '$\pi$', '', '$2\pi$'])
    ax.set_xticks(np.arange(0, np.pi*2, np.pi/4), minor=True)
    ax.set_yticks([0, 0.5])
    ax.set_ylim(0, np.max([binsphasebest.max(), binsphaseworst.max()])*1.5)
    ax.set_ylabel('Spike density', fontsize=fontsize, labelpad=0)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    # ax.yaxis.get_offset_text().set_x(0.02)
    # ax.yaxis.get_offset_text().set_fontsize(fontsize)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.annotate(r'$\times 10^{-1}$', xy=(0.01, 0.9), xycoords='axes fraction', fontsize=fontsize)


def plot_exin_bestworst_simdissim(ax, exindf, exindict, direct_c, sim_c, dissim_c):

    Best_exindf = exindf[exindf['Best']].reset_index(drop=True)
    Worst_exindf = exindf[exindf['Worst']].reset_index(drop=True)
    Sim_exindf = exindf[exindf['Sim']].reset_index(drop=True)
    Dissim_exindf = exindf[exindf['Dissim']].reset_index(drop=True)

    bestlabel = 'Best: %0.2f' %(exindict['Best']['ex_n']/exindict['Best']['in_n'])
    worstlabel = 'Worst: %0.2f' %(exindict['Worst']['ex_n']/exindict['Worst']['in_n'])
    ax[0].scatter(Best_exindf['ex'], Best_exindf['in'], c=direct_c[0], marker='.', s=0.5, label=bestlabel)
    ax[0].scatter(Worst_exindf['ex'], Worst_exindf['in'], c=direct_c[1], marker='.', s=0.5, label=worstlabel)
    ax[0].annotate(bestlabel, xy=(0.02, 0.16), xycoords='axes fraction', color=direct_c[0])
    ax[0].annotate(worstlabel, xy=(0.02, 0.01), xycoords='axes fraction', color=direct_c[1])

    ax[0].set_xlabel('Ex', labelpad=0)
    ax[0].set_ylabel('In', labelpad=0)
    ax[0].plot([0.3, 1], [0.3, 1], linewidth=0.75, c='k')
    ax[0].set_xticks(np.arange(0, 1.1, 0.5))
    ax[0].set_xticklabels(['0', '', 1])
    ax[0].set_xticks(np.arange(0, 1, 0.1), minor=True)
    ax[0].set_xlim(0, 1)
    ax[0].set_yticks(np.arange(0, 1.1, 0.5))
    ax[0].set_yticklabels(['0', '', 1])
    ax[0].set_yticks(np.arange(0, 1, 0.1), minor=True)
    ax[0].set_ylim(0, 1)

    simlabel = 'Sim: %0.2f' %(exindict['Sim']['ex_n']/exindict['Sim']['in_n'])
    dissimlabel = 'Dissim: %0.2f' %(exindict['Dissim']['ex_n']/exindict['Dissim']['in_n'])
    ax[1].scatter(Sim_exindf['ex'], Sim_exindf['in'], c=sim_c, marker='.', s=0.5, label=simlabel)
    ax[1].scatter(Dissim_exindf['ex'], Dissim_exindf['in'], c=dissim_c, marker='.', s=0.5, label=dissimlabel)
    ax[1].annotate(simlabel, xy=(0.02, 0.16), xycoords='axes fraction', color=sim_c)
    ax[1].annotate(dissimlabel, xy=(0.02, 0.01), xycoords='axes fraction', color=dissim_c)
    ax[1].set_xlabel('Ex', labelpad=0)
    ax[1].set_ylabel('In', labelpad=0)
    ax[1].plot([0.3, 1], [0.3, 1], linewidth=0.75, c='k')
    ax[1].set_xticks(np.arange(0, 1.1, 0.5))
    ax[1].set_xticklabels(['0', '', 1])
    ax[1].set_xticks(np.arange(0, 1, 0.1), minor=True)
    ax[1].set_xlim(0, 1)
    ax[1].set_yticks(np.arange(0, 1.1, 0.5))
    ax[1].set_yticklabels(['0', '', 1])
    ax[1].set_yticks(np.arange(0, 1, 0.1), minor=True)
    ax[1].set_ylim(0, 1)

    bias_edges = np.linspace(-1, 1, 50)
    nbins1, _ = np.histogram(Best_exindf['ex_bias'], bins=bias_edges)
    nbins2, _ = np.histogram(Worst_exindf['ex_bias'], bins=bias_edges)
    nbins3, _ = np.histogram(Sim_exindf['ex_bias'], bins=bias_edges)
    nbins4, _ = np.histogram(Dissim_exindf['ex_bias'], bins=bias_edges)
    nbins1, nbins2, nbins3, nbins4 = np.cumsum(nbins1)/nbins1.sum(), np.cumsum(nbins2)/nbins2.sum(), np.cumsum(nbins3)/nbins3.sum(), np.cumsum(nbins4)/nbins4.sum()
    ax[2].plot(bias_edges[:-1], nbins1, linewidth=0.75, color=direct_c[0])
    ax[2].plot(bias_edges[:-1], nbins2, linewidth=0.75, color=direct_c[1])
    ax[2].plot(bias_edges[:-1], nbins3, linewidth=0.75, color=sim_c)
    ax[2].plot(bias_edges[:-1], nbins4, linewidth=0.75, color=dissim_c)
    ax[2].axvline(exindict['Best']['ex_bias_mu'], ymin=0.75, ymax=1, linewidth=0.5, color=direct_c[0])
    ax[2].axvline(exindict['Worst']['ex_bias_mu'], ymin=0.75, ymax=1, linewidth=0.5, color=direct_c[1])
    ax[2].axvline(exindict['Sim']['ex_bias_mu'], ymin=0.75, ymax=1, linewidth=0.5, color=sim_c)
    ax[2].axvline(exindict['Dissim']['ex_bias_mu'], ymin=0.75, ymax=1, linewidth=0.5, color=dissim_c)
    ax[2].set_xlabel(r'Ex$\minus$In', labelpad=0)
    ax[2].set_xticks(np.arange(-0.5, 0.6, 0.5))
    ax[2].set_xticklabels(['-0.5', '0', '0.5'])
    ax[2].set_xticks(np.arange(-0.5, 0.6, 0.1), minor=True)
    ax[2].set_xlim(-0.5, 0.5)
    ax[2].set_ylabel('CDF of pairs', labelpad=0)
    ax[2].set_yticks([0, 0.5, 1])
    ax[2].set_yticks(np.arange(0, 1, 0.1), minor=True)
    ax[2].set_yticklabels(['0', '', '1'])
    ax[2].set_ylim(0, np.max(np.concatenate([nbins1, nbins2, nbins3, nbins4]))*1.3)


def plot_tempotron_traces(axRas, axTrace, axW1d, N, X, temN_tax, temNw, Vthresh, all_nidx, yytun1d, kout_all, tspout_all, val2cmap, exintag):

    M = len(axRas)
    w_yax = np.arange(N)
    y_nidx = yytun1d[all_nidx[w_yax]]

    if exintag == 'in':

        ylim = (np.max(np.where(y_nidx <14.7)[0]), np.min(np.where(y_nidx > 25.8)[0]))
    else:
        ylim = (np.max(np.where(y_nidx <-24.8)[0]), np.min(np.where(y_nidx > -14.7)[0]))

    for Mi in range(M):

        for Ni in range(N):
            tsp = X[Mi, Ni]
            axRas[Mi].scatter(tsp, [Ni]*tsp.shape[0], marker='|', s=1.5, linewidths=1, color=val2cmap.to_rgba(temNw[Ni]))

        ysep_NiList = np.arange(0, 400, 20).astype(int)
        ysep_ax = np.arange(0, 400, 20) - 0.5
        for ysep in ysep_ax:
            axRas[Mi].axhline(ysep, color='gray', lw=0.5)
        axRas[Mi].set_yticks(ysep_NiList + 10)
        axRas[Mi].set_xlim(0, 100)
        axRas[Mi].set_xticks([])
        axRas[Mi].set_yticklabels(np.around(yytun1d[all_nidx[ysep_NiList + 10]], 0).astype(int).astype(str))
        axRas[Mi].set_ylim(ylim[0], ylim[1])
        if Mi > 0:
            axRas[Mi].set_yticks([])
        if Mi == 0:
            axRas[Mi].set_ylabel('y (cm)')

        # Plot voltage trace
        axTrace[Mi].plot(temN_tax, kout_all[Mi], color='gray')
        if tspout_all[Mi].shape[0] > 0:
            axTrace[Mi].eventplot([tspout_all[Mi][0]], lineoffsets=2.5, linelengths=0.5, linewidths=1, color='r')
        axTrace[Mi].set_xlim(0, 100)
        axTrace[Mi].set_xticks([0, 100])
        if exintag == 'ex':
            axTrace[Mi].set_xticklabels([100*Mi, 100*(Mi+1)])
        else:
            axTrace[Mi].set_xticklabels([])
        axTrace[Mi].set_ylim(0.1, 3)
        axTrace[Mi].axhline(Vthresh, color='k', linewidth=0.1)
        axTrace[Mi].set_yticks([])
        axTrace[Mi].spines.left.set_visible(False)
        if Mi > 0:
            axTrace[Mi].set_yticks([])
        if exintag=='in':
            axTrace[Mi].set_xticks([])

        # Plot flattened weights
        axW1d.barh(w_yax, temNw, color=val2cmap.to_rgba(temNw))
        axW1d.axvline(0, color='gray', lw=0.1)
        axW1d.set_yticks(np.around(np.arange(N), 2))
        axW1d.axis('off')
        axW1d.set_ylim(ylim[0], ylim[1])
