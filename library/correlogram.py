import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, hilbert
# from .comput_utils import correlate, custum_interpolate
# from .utils import mat2direction, check_iter

from library.utils import mat2direction


class Crosser:
    def __init__(self, bin_size=0.002, window_width=0.3, bandpass=(5, 12), method='huxter'):
        self.bin_size = bin_size
        self.window_width = window_width
        self.bandpass = bandpass
        self.method = method

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
            b, a = butter(1, self.freq_kernel, btype='bandpass')
        else:
            b, a = butter(1, self.freq_kernel)
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

    def find_theta(self, pairedpasses, default_Ttheta=1 / 9):
        '''
        Args:
            pairedpasses (pandas.DataFrame):
        '''
        
        tsp1_list, tsp2_list = [], []
        num_passes = pairedpasses.shape[0]
        for pass_nth in range(num_passes):
            tsp1 = pairedpasses.loc[pass_nth, 'tsp1']
            tsp2 = pairedpasses.loc[pass_nth, 'tsp2']
            tsp1_list.append(tsp1)
            tsp2_list.append(tsp2)
        mean_Ttheta, phase_at_zero, info_tuple = self.find_theta_isi_hilbert(tsp1_list, tsp2_list, theta_window=0.3, default_Ttheta=default_Ttheta)
        return mean_Ttheta, phase_at_zero, info_tuple

    def find_theta_isi_hilbert(self, tsp1_list, tsp2_list, theta_window=0.3, default_Ttheta=1/9):
        if len(tsp1_list) == 0 or len(tsp2_list) == 0:
            return default_Ttheta, np.nan, np.nan

        tsp1_concat = np.concatenate(tsp1_list)
        tsp2_concat = np.concatenate(tsp2_list)

        # Build inter-spike interval histogram
        tsp1_expanded = np.matmul(tsp1_concat.reshape(-1, 1), np.ones((1, tsp2_concat.shape[0])))
        tsp2_expanded = np.matmul(np.ones((tsp1_concat.shape[0], 1)), tsp2_concat.reshape(1, -1))
        isi = (tsp1_expanded - tsp2_expanded).flatten()
        isi = isi[np.abs(isi) < theta_window/2]
        isibins, isiedges = np.histogram(isi,
                                     bins=np.arange(-self.window_width / 2 - (self.bin_size/2), self.window_width / 2 + (self.bin_size/2) + self.bin_size, self.bin_size))


        isiedges_m = (isiedges[1:] + isiedges[:-1]) / 2

        # Filter and Hilbert transform
        signal_filt = self._filter(isibins)
        z = hilbert(signal_filt)
        alphas = np.angle(z)

        if (np.max(signal_filt) < 0.0001) or (isi.shape[0]==0):  # If no signal is found
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
