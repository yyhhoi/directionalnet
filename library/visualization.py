import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np
import pandas as pd
from pycircstat import cdiff
from pycircstat import mean as circmean
from library.linear_circular_r import rcc
from library.utils import load_pickle
from library.stattests import p2str
from library.comput_utils import dist_overlap, correlate, smooth_xy, \
    normsig, linear_circular_gauss_density, circular_density_1d, linear_density_1d, fisherexact, midedges, ranksums
from library.correlogram import Crosser, ThetaEstimator
from pycircstat.tests import watson_williams, rayleigh

from scipy import interpolate
from scipy.stats import pearsonr, chi2_contingency, chi2, ttest_ind, pearsonr, spearmanr, chisquare, ttest_1samp, \
    fisher_exact
from library.shared_vars import fontsize, ticksize, legendsize, titlesize, ca_c, dpi, total_figw


def plot_correlogram(ax, df, tag, direct='A->B', overlap_key='overlap', color='gray', density=False,
                     regress=True, x_dist=True, y_dist=True, ab_key='phaselag_AB', ba_key='phaselag_BA', alpha=0.2,
                     markersize=8, linew=1):
    """

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
    df : Dataframe
    tag : str
    direct : str
        Either 'A->B', 'B->A' or 'combined'
    overlap_key : str
        Key of the overlap metric
    color : str
        Color of the scatter
    regress : bool
    x_dist : bool
    y_dist : bool
    ab_key : str
    ba_key : str
    alpha : float

    Returns
    -------

    """
    overlap = df[overlap_key].to_numpy()
    phaselag_AB = df[ab_key].to_numpy()
    phaselag_BA = df[ba_key].to_numpy()

    # Define x, y
    if direct == 'A->B':
        x = overlap
        y = phaselag_AB
    elif direct == 'B->A':
        x = overlap
        y = phaselag_BA
    elif direct == 'combined':
        x = np.concatenate([overlap, overlap])
        y = np.concatenate([phaselag_AB, -phaselag_BA])

    nan_mask = np.isnan(x) | np.isnan(y)
    x_nonan, y_nonan = x[~nan_mask], y[~nan_mask]

    # Scatter or density
    if density:
        xx, yy, zz = linear_circular_gauss_density(x_nonan, y_nonan, cir_kappa=3 * np.pi, lin_std=0.05, xbins=200,
                                                   ybins=800, ybound=(-np.pi, 2 * np.pi))
        ax.pcolormesh(xx, yy, zz)
    else:
        ax.scatter(x, y, c=color, alpha=alpha, s=markersize, marker='.')
        ax.scatter(x, y + (2 * np.pi), c=color, alpha=alpha, s=markersize, marker='.')

    # Plot marginal mean
    mean_y = circmean(y_nonan)
    ax.plot([0, 0.2], [mean_y, mean_y], c='k', linewidth=linew)

    # Regression
    regress_d = None
    if regress:
        regress_d = rcc(x_nonan, y_nonan, abound=[-2, 2])
        m, c, r, pval = regress_d['aopt'], regress_d['phi0'], regress_d['rho'], regress_d['p']
        r_regress = np.linspace(0, 1, 20)
        lag_regress = 2 * np.pi * r_regress * m
        for tmpid, intercept in enumerate([2 * mul_idx * np.pi + c for mul_idx in [-2, -1, 0, 1, 2]]):
            if tmpid == 0:
                ax.plot(r_regress, lag_regress + intercept, c='k', linewidth=linew, label='rho=%0.2f (p%s)'%(r, p2str(pval)))
            else:
                ax.plot(r_regress, lag_regress + intercept, c='k', linewidth=linew)

    # X-axis
    ax.plot([0, 1], [0, 0], c='k', alpha=1, linewidth=linew)

    # (y-axis) Interpolation and Smoothened histrogram
    if y_dist:
        p, _ = rayleigh(y)
        yax, yden = circular_density_1d(y_nonan, 10*np.pi, 60, (-np.pi, 3*np.pi))

        ax.plot(yden/yden.max()*0.2, yax, c='k', linewidth=linew)

    # (x-axis) Interpolation and Smoothened histrogram
    if x_dist:
        xax, xden = linear_density_1d(x_nonan, std=0.05, bins=100, bound=(0, 1))
        ax.plot(xax, xden/xden.max()*(np.pi/2) - np.pi, c='k', linewidth=linew)

    # Set title

    # ax.set_title("%s %s (n=%d)" % (tag, direct, num_pairs), fontsize=fontsize)
    ax.set_ylim(-np.pi, 2 * np.pi)
    ax.set_yticks([-np.pi, 0, np.pi, np.pi * 2])
    _ = ax.set_yticklabels(['$-\pi$',  '0', '$\pi$', '$2\pi$'])
    return ax, x, y, regress_d


def plot_dcf_phi_histograms(df, ax, color, phidict, dict_tag, title_tag, ca):
    phidict['phi0_%s' % (dict_tag)] = np.abs(df['phi0'])
    bins, edges = np.histogram(np.abs(df['phi0']), bins=50, range=(0, np.pi))
    ax[0].plot(edges[0:-1], np.cumsum(bins) / np.sum(bins), label='%s (n=%d)' % (ca, df['phi0'].shape[0]),
               color=color)
    ax[0].set_title('$|\phi_0|$ %s' % (title_tag), fontsize='xx-large')
    ax[0].legend()

    phidict['phi1_%s' % (dict_tag)] = df['phi1']
    bins, edges = np.histogram(df['phi1'], bins=50, range=(0, np.pi))
    ax[1].plot(edges[0:-1], np.cumsum(bins) / np.sum(bins), label='%s (n=%d)' % (ca, df['phi1'].shape[0]),
               color=color)
    ax[1].set_title('$\phi_1$ %s' % (title_tag), fontsize='xx-large')
    ax[1].legend()
    return ax, phidict

def color_wheel(cmap_key='hsv', figsize=(4, 4)):
    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')
    # ax._direction = np.pi  ## This is a nasty hack - using the hidden field to
    ## multiply the values such that 1 become 2*pi
    ## this field is supposed to take values 1 or -1 only!!

    norm = mpl.colors.Normalize(-np.pi, np.pi)

    # Plot the colorbar onto the polar axis
    # note - use orientation horizontal so that the gradient goes around
    # the wheel rather than centre out
    quant_steps = 2056
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cm.get_cmap(cmap_key, quant_steps),
                                   norm=norm,
                                   orientation='horizontal')

    # aesthetics - get rid of border and axis labels
    cb.outline.set_visible(False)
    ax.set_axis_off()
    return fig, ax


def directionality_polar_plot(ax, x, y, mean_x, linewidth=0.75):
    ax.plot(x, y, c='k', linewidth=linewidth)
    ax.plot([x[-1], x[0]], [y[-1], y[0]], c='k', linewidth=linewidth)
    ax.plot([mean_x, mean_x], [0, y.max()], c='gray', linewidth=linewidth)
    ax.axis('off')
    ax.grid(False)
    return ax


def gau2d(xx, yy, mux, muy, sd):
    const = 1 / (2 * np.pi * np.power(sd, 2))
    exponent = (-1 / (2 * sd)) * (np.power((xx - mux), 2) + np.power((yy - muy), 2))
    return const * np.exp(exponent)


def smooth_heatmap(x, y, sd, xbound=[-np.pi, np.pi], ybound=[-np.pi, np.pi], bins1d=200):
    x_axis = np.linspace(xbound[0], xbound[1], bins1d)
    y_axis = np.linspace(ybound[0], ybound[1], bins1d)

    xx, yy = np.meshgrid(x_axis, y_axis)

    zz = np.zeros(xx.shape)
    assert x.shape[0] == y.shape[0]

    for idx in range(x.shape[0]):
        x_each, y_each = x[idx], y[idx]
        zz += gau2d(xx, yy, x_each, y_each, sd)
    return x_axis, y_axis, xx, yy, zz


def plot_marginal_slices(ax, xx, yy, zz, selected_x, slice_yrange, slicegap):
    """
    Marginal slices at different x values, as a function of y-axis
    Parameters
    ----------
    ax : axes object
    xx : ndarray
    yy : ndarray
    zz : ndarray
    selected_x : tuple
    slice_yrange : tuple
    slicegap : float

    Returns
    -------

    """


    # Slices
    yax = yy[:, 0]  # y
    xax = xx[0, :]  # x
    color_list = []
    for idx, xval in enumerate(selected_x):
        selected_xid = np.argmin(np.abs(xval - xax))
        yrange_ids = [np.argmin(np.abs(yax - y_bound_val)) for y_bound_val in slice_yrange]

        slice_cut = zz[yrange_ids[0]:yrange_ids[1], selected_xid]
        slice_norm = slice_cut / np.sum(slice_cut)
        slice_shifted = (slice_norm - slice_norm.min()) + idx * slicegap
        yax_cut = yax[yrange_ids[0]:yrange_ids[1]]

        slice_color = cm.brg(idx / selected_x.shape[0])
        color_list.append(slice_color)
        ax.plot(yax_cut, slice_shifted, c=slice_color)

        y_cog = circmean(yax_cut, w=slice_norm)
        slice_point = slice_shifted[np.argmin(np.abs(cdiff(yax_cut, y_cog)))]
        marker, m_c = '.', 'k'
        ax.scatter(y_cog, slice_point, marker=marker, c=[m_c], zorder=3.1)
        ax.scatter(y_cog-2*np.pi, slice_point, marker=marker, c=[m_c], zorder=3.1)
        ax.scatter(y_cog+2*np.pi, slice_point, marker=marker, c=[m_c], zorder=3.1)


    return ax, color_list


class SqueezedNorm(mpl.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self.vmin = vmin # minimum value
        self.mid  = mid  # middle value
        self.vmax = vmax # maximum value
        self.s1=s1; self.s2=s2
        f = lambda x, zero, vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero, vmin, vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                                    f(x,zero,vmin,s2)*(x<zero)+0.5
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
        return np.ma.masked_array(r)



def customlegend(ax, handlelength=1.2, linewidth=1.2, **kwargs):
    leg = ax.legend(handlelength=handlelength, labelspacing=0.1, handletextpad=0.1,
              borderpad=0.1, frameon=False, **kwargs)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(linewidth)
    return ax




def plot_prefer_nonprefer(ax, prefer_pdf, nonprefer_pdf):
    '''

    Parameters
    ----------
    ax: ndarray
        Numpy array of axes objects with shape (6, ).
        ax[0] - Fraction of negative slope
        ax[1] - Prefer's precession
        ax[2] - Nonprefer's precession
        ax[3] - Difference in onsets for Prefer and Nonprefer
        ax[4] - Difference in slopes for Prefer and Nonprefer
        ax[5] - Difference in phases for Prefer and Nonprefer

    prefer_pdf: dataframe
        Pandas dataframe containing prefered passes (including non-precessing)
    nonprefer_pdf: dataframe
        Pandas dataframe containing nonprefered passes (including non-precessing)

    Returns
    -------
    '''
    pre_c = dict(prefer='r', nonprefer='b')
    prefer_allslope = prefer_pdf['slope'].to_numpy()
    nonprefer_allslope = nonprefer_pdf['slope'].to_numpy()
    prefer_slope = prefer_pdf[prefer_pdf['precess_exist']]['slope'].to_numpy() * np.pi
    nonprefer_slope = nonprefer_pdf[nonprefer_pdf['precess_exist']]['slope'].to_numpy() * np.pi
    prefer_onset = prefer_pdf[prefer_pdf['precess_exist']]['onset'].to_numpy()
    nonprefer_onset = nonprefer_pdf[nonprefer_pdf['precess_exist']]['onset'].to_numpy()
    prefer_dsp = np.concatenate(prefer_pdf[prefer_pdf['precess_exist']]['dsp'].to_list())
    nonprefer_dsp = np.concatenate(nonprefer_pdf[nonprefer_pdf['precess_exist']]['dsp'].to_list())
    prefer_phasesp = np.concatenate(prefer_pdf[prefer_pdf['precess_exist']]['phasesp'].to_list())
    nonprefer_phasesp = np.concatenate(nonprefer_pdf[nonprefer_pdf['precess_exist']]['phasesp'].to_list())


    # Precession fraction
    prefer_neg_n, prefer_pos_n = (prefer_allslope<0).sum(), (prefer_allslope>0).sum()
    nonprefer_neg_n, nonprefer_pos_n = (nonprefer_allslope<0).sum(), (nonprefer_allslope>0).sum()
    table_arr = np.array([[prefer_neg_n, prefer_pos_n], [nonprefer_neg_n, nonprefer_pos_n] ])
    table_df = pd.DataFrame(table_arr, columns=['-ve', '+ve'], index=['Prefer', 'Nonprefer'])
    p_frac = fisherexact(table_arr)
    prefer_frac = prefer_neg_n/(prefer_pos_n+prefer_neg_n)
    nonprefer_frac = nonprefer_neg_n/(nonprefer_pos_n+nonprefer_neg_n)
    bar_w, bar_x, bar_y = 0.5, np.array([0, 1]), np.array([prefer_frac, nonprefer_frac])
    ax[0].bar(bar_x, bar_y, width=bar_w)
    ax[0].errorbar(x=bar_x.mean(), y=0.8, xerr=1, c='k', capsize=2.5)
    ax[0].text(x=0, y=0.85, s='p%s'%(p2str(p_frac)), fontsize=legendsize)
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(['Prefer', 'Nonprefer'])
    ax[0].tick_params(labelsize=fontsize)

    # Prefer precession
    xdum = np.linspace(0, 1, 10)
    mean_phasesp = circmean(prefer_phasesp)
    regress = rcc(prefer_dsp, prefer_phasesp)
    ydum = regress['phi0'] + xdum * 2*np.pi * regress['aopt']
    ax[1].scatter(prefer_dsp, prefer_phasesp, marker='.', c='gray', s=1)
    ax[1].scatter(prefer_dsp, prefer_phasesp + 2*np.pi, marker='.', c='gray', s=1)
    ax[1].plot(xdum, ydum, c='k')
    ax[1].plot(xdum, ydum+2*np.pi, c='k')
    ax[1].plot(xdum, ydum+2*np.pi, c='k')
    ax[1].axhline(mean_phasesp, xmin=0, xmax=0.1, color='r')
    ax[1].set_ylim(-np.pi, 3*np.pi)
    ax[1].set_xlim(0, 1)

    # Non-Prefer precession
    xdum = np.linspace(0, 1, 10)
    mean_phasesp = circmean(nonprefer_phasesp)
    regress = rcc(nonprefer_dsp, nonprefer_phasesp)
    ydum = regress['phi0'] + xdum * 2*np.pi * regress['aopt']
    ax[2].scatter(nonprefer_dsp, nonprefer_phasesp, marker='.', c='gray', s=1)
    ax[2].scatter(nonprefer_dsp, nonprefer_phasesp + 2*np.pi, marker='.', c='gray', s=1)
    ax[2].plot(xdum, ydum, c='k')
    ax[2].plot(xdum, ydum+2*np.pi, c='k')
    ax[2].plot(xdum, ydum+2*np.pi, c='k')
    ax[2].axhline(mean_phasesp, xmin=0, xmax=0.1, color='r')
    ax[2].set_ylim(-np.pi, 3*np.pi)
    ax[2].set_xlim(0, 1)

    # Onset vs Slope
    p_onset, _ = watson_williams(prefer_onset, nonprefer_onset)
    _, p_slope = ranksums(prefer_slope, nonprefer_slope)
    ax[3].scatter(prefer_slope, prefer_onset, marker='o', c=pre_c['prefer'], label='prefer')
    ax[3].scatter(nonprefer_slope, nonprefer_onset, marker='x', c=pre_c['nonprefer'], label='nonprefer')
    ax[3].set_xlabel('Precession slope (rad)')
    ax[3].set_ylabel('Precession onset (rad)')
    customlegend(ax[3], fontsize=legendsize)

    # # Onset
    # p_onset, _ = watson_williams(prefer_onset, nonprefer_onset)
    # onset_bins = np.linspace(0, 2*np.pi, 20)
    # prefer_onsetbins, _ = np.histogram(prefer_onset, bins=onset_bins)
    # nonprefer_onsetbins, _ = np.histogram(nonprefer_onset, bins=onset_bins)
    # ax[3].step(onset_bins[:-1], prefer_onsetbins/prefer_onsetbins.sum(), where='pre', c=pre_c['prefer'], label='prefer')
    # ax[3].step(onset_bins[:-1], nonprefer_onsetbins/nonprefer_onsetbins.sum(), where='pre', c=pre_c['nonprefer'], label='nonprefer')
    # ax[3].annotate('Watson-William\np%s'%(p2str(p_onset)), xy=(0.6, 0.6), xycoords='axes fraction')
    # ax[3].set_xlabel('Precession onset (rad)')
    # customlegend(ax[3], fontsize=legendsize)
    #
    # # Slope
    # _, p_slope = ranksums(prefer_slope, nonprefer_slope)
    # slope_bins = np.linspace(-2*np.pi, 0, 20)
    # prefer_slopebins, _ = np.histogram(prefer_slope, bins=slope_bins)
    # nonprefer_slopebins, _ = np.histogram(nonprefer_slope, bins=slope_bins)
    # ax[4].step(slope_bins[:-1], prefer_slopebins/prefer_slopebins.sum(), where='pre', c=pre_c['prefer'], label='prefer')
    # ax[4].step(slope_bins[:-1], nonprefer_slopebins/nonprefer_slopebins.sum(), where='pre', c=pre_c['nonprefer'], label='nonprefer')
    # ax[4].annotate('Ranksums\np%s'%(p2str(p_slope)), xy=(0.6, 0.6), xycoords='axes fraction')
    # ax[4].set_xlabel('Precession slope (rad)')
    # customlegend(ax[4], fontsize=legendsize)

    # Spike phase
    prefer_phasesp_mean, nonprefer_phasesp_mean = circmean(prefer_phasesp), circmean(nonprefer_phasesp)
    p_phasesp, _ = watson_williams(prefer_phasesp, nonprefer_phasesp)
    phasesp_bins = np.linspace(-np.pi, np.pi, 20)
    prefer_phasespbins, _ = np.histogram(prefer_phasesp, bins=phasesp_bins)
    nonprefer_phasespbins, _ = np.histogram(nonprefer_phasesp, bins=phasesp_bins)
    norm_preferphasebins = prefer_phasespbins/prefer_phasespbins.sum()
    norm_nonpreferphasebins = nonprefer_phasespbins/nonprefer_phasespbins.sum()
    maxl = max(norm_preferphasebins.max(), norm_nonpreferphasebins.max())
    ax[4].step(midedges(phasesp_bins), norm_preferphasebins, color='r', label='prefer')
    ax[4].step(midedges(phasesp_bins), norm_nonpreferphasebins, color='b', label='nonprefer')
    ax[4].scatter(prefer_phasesp_mean, maxl*1.1, color='r', marker='|', s=64)
    ax[4].scatter(nonprefer_phasesp_mean, maxl*1.1, color='b', marker='|', s=64)
    ax[4].annotate('Watson-William\np%s'%(p2str(p_phasesp)), xy=(0.6, 0.2), xycoords='axes fraction')
    ax[4].set_xlabel('Precession spike phase')
    customlegend(ax[4], fontsize=legendsize)

    return ax