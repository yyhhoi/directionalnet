import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, hilbert
from library.utils import mat2direction


class Crosser:
    def __init__(self, bin_size=0.002, window_width=0.3, bandpass=(5, 12)):
        self.bin_size = bin_size
        self.window_width = window_width
        self.bandpass = bandpass
        self.isiedges = self.sym_edges(bin_size, window_width)

        # Initialize
        self.nyqf = (1 / self.bin_size) / 2
        self.freq_kernel = self._set_freq_kernel()



    @staticmethod
    def find_direction(infield1, infield2, get_intermask=False):
        f1_begin, f1_end = infield1[0], infield1[-1]
        f2_begin, f2_end = infield2[0], infield2[-1]
        loc = [f1_begin, f1_end, f2_begin, f2_end]
        direction = mat2direction(loc)
        if get_intermask:
            intersect_mask = (infield1 + infield2) == 2
            return loc, direction, intersect_mask
        else:
            return loc, direction

    @staticmethod
    def sym_edges(bsize, winwidth):

        e1 = np.arange(-winwidth/2, 0-bsize+(bsize/2), bsize)
        e2 = np.arange(bsize, winwidth/2+bsize, bsize)
        if np.around(e1[-1], 10) != np.around(-e2[0], 10):
            raise ValueError('Bin edges need to be symmetrical. Window size needs to be divisible by bin size.')
        edges = np.concatenate([e1, e2])
        return edges


    def filter_intersect(self, infield1, infield2):
        intersect_mask = (infield1 + infield2) == 2
        self.intersect_mask = intersect_mask

    def _set_freq_kernel(self):

        if hasattr(self.bandpass, '__iter__'):
            return [self.bandpass[0] / self.nyqf, self.bandpass[1] / self.nyqf]
        else:
            return self.bandpass / self.nyqf

    def _filter(self, signal):
        if hasattr(self.bandpass, '__iter__'):
            b, a = butter(4, self.freq_kernel, btype='bandpass')
        else:
            b, a = butter(4, self.freq_kernel)
        signal_filt = filtfilt(b, a, signal)
        return signal_filt

    def _hist(self, tsp1, tsp2):
        total_tsp = np.concatenate([tsp1, tsp2])
        t_min, t_max = np.min(total_tsp), np.max(total_tsp)
        num_steps = int((t_max - t_min) / self.bin_size)
        bins_space = np.linspace(t_min, t_max, num_steps)
        N1, edges1 = np.histogram(tsp1, bins=bins_space)
        N2, edges2 = np.histogram(tsp2, bins=bins_space)
        return (N1, edges1), (N2, edges2)

    def _find_nearest_peak(self, t, x):

        mask_in_window = (np.abs(t) < self.window_width / 2)
        t_inwindow = t[mask_in_window]
        x_inwindow = x[mask_in_window]

        pk_idns = find_peaks(x_inwindow)[0]

        if pk_idns.shape[0] > 1:
            t_peaks = t_inwindow[pk_idns]
            x_peaks = x_inwindow[pk_idns]
            mask_minpeak = (np.abs(t_peaks) == np.min(np.abs(t_peaks)))
            lag_selected = t_peaks[mask_minpeak]
            x_selected = x_peaks[mask_minpeak]

        elif pk_idns.shape[0] == 1:
            lag_selected = t_inwindow[pk_idns]
            x_selected = x_inwindow[pk_idns]
        else:
            max_mask = (x_inwindow == np.max(x_inwindow))
            lag_selected = t_inwindow[max_mask]
            x_selected = x_inwindow[max_mask]

        return lag_selected[0], x_selected


class ThetaEstimator(Crosser):

    def find_theta_isi_hilbert(self, tsp1, tsp2, theta_window=0.3, default_Ttheta=1/9):
        '''

        Parameters
        ----------
        tsp1 : ndarray
        tsp2 : ndarray
        theta_window : float
        default_Ttheta : float

        Returns
        -------

        '''

        # Build inter-spike interval histogram
        isi = tsp1.reshape(tsp1.shape[0], 1) - tsp2.reshape(1, tsp2.shape[0])
        isi = isi.flatten()
        isi = isi[np.abs(isi) < theta_window/2]
        if isi.shape[0] < 1:  # If no spike within the isi window is found
            return default_Ttheta, np.nan, np.nan

        isiedges = self.isiedges
        isibins, _ = np.histogram(isi, bins=self.isiedges)
        isiedges_m = (isiedges[1:] + isiedges[:-1]) / 2

        # Filter and Hilbert transform
        signal_filt = self._filter(isibins)
        z = hilbert(signal_filt)
        alphas = np.angle(z)

        if (np.max(signal_filt) < 0.0001):  # If no signal is found
            return default_Ttheta, np.nan, np.nan
        
        # Find phase at 0 lag
        zero_idx = np.where( np.abs(isiedges_m) == np.min(np.abs(isiedges_m)) )
        phase_at_zero = alphas[zero_idx[0][0]]

        # Find Theta period (time difference between 2 troughs)
        dalphas = alphas[1:] - alphas[0:-1]
        alpha_idxes = np.where(dalphas < -np.pi)
        Tperiod_list = [default_Ttheta]
        if alpha_idxes[0].shape[0] > 1:
            for aind in range(alpha_idxes[0].shape[0] - 1):
                Tperiod = isiedges_m[alpha_idxes[0][aind + 1]] - isiedges_m[alpha_idxes[0][aind]]
                Tperiod_list.append(Tperiod)

        return np.mean(Tperiod_list), phase_at_zero, (isibins, isiedges, signal_filt, z, alphas, isi)
