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
