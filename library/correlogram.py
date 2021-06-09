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

    # def cross(self, tsp1, tsp2, intersect_mask=None):
    #
    #     if intersect_mask:
    #         tsp1, tsp2 = tsp1[intersect_mask], tsp2[intersect_mask]
    #
    #     (N1, edges1), (N2, edges2) = self._hist(tsp1, tsp2)
    #
    #     r, lags = correlate(N1, N2)
    #
    #     lags_t = lags * self.bin_size
    #
    #     r_filt = self._filter(r)
    #
    #     if self.method == 'huxter':
    #         lags_inter, r_inter = custum_interpolate(lags_t, r_filt)
    #         timelag, r_at_timelag = self._find_nearest_peak(lags_inter, r_inter)
    #         return timelag, r_at_timelag


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
        # isi = isi[np.abs(isi) < theta_window/2]
        # isibins, isiedges = np.histogram(isi,
        #                              bins=np.arange(-self.window_width / 2 - (self.bin_size/2), self.window_width / 2 + (self.bin_size/2) + self.bin_size, self.bin_size))
        isi = isi[np.abs(isi) < theta_window]
        isibins, isiedges = np.histogram(isi,
                                         bins=np.arange(-self.window_width - (self.bin_size/2), self.window_width + (self.bin_size/2) + self.bin_size, self.bin_size))


        isiedges_m = (isiedges[1:] + isiedges[:-1]) / 2

        # Filter and Hilbert transform
        signal_filt = self._filter(isibins)
        z = hilbert(signal_filt)
        alphas = np.angle(z)

        # Cut the intervals to -0.15s to 0.15s
        winmask = np.abs(isiedges_m)< (theta_window/2-0.0001)
        isiedges = isiedges[np.abs(isiedges) < (theta_window/2-0.0001)]
        isiedges_m = isiedges_m[winmask]
        isibins = isibins[winmask]
        signal_filt, z, alphas = signal_filt[winmask], z[winmask], alphas[winmask]
        isi = isi[(isi < np.max(isiedges_m)) & (isi > np.min(isiedges_m))]

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


class Bootstrapper:
    def __init__(self, field_df, bs_size=None, bs_rinterval=0.1, bs_roffset=0, ks_metric='ks_dist_masked', seed_num=None, verbose=False):
        self.field_df = field_df.copy()
        self.bs_size = bs_size
        self.bs_rinterval = bs_rinterval
        self.bs_roffset = bs_roffset
        self.ks_metric = ks_metric
        self.seed_num = seed_num if seed_num else np.random.randint(9e5)
        self.verbose = verbose
        self.inter_idx = int(1/self.bs_rinterval)
        self.r_intervals = [(x/self.inter_idx, x/self.inter_idx+self.bs_rinterval) for x in range(self.inter_idx)] + \
                           [(x/self.inter_idx + self.bs_roffset, x/self.inter_idx+self.bs_rinterval + bs_roffset) for x in range(self.inter_idx)]
    
    def _msg(self, *arg, **kwarg):
        if self.verbose:
            print(*arg, **kwarg)
        else:
            pass

    def _get_bs_size(self):
        
        self._msg('Adaptively determine the bootstrapping size from the data')
        num_samples_list = []
        for ca_idx, ca in enumerate(['CA{}'.format(x+1) for x in range(3)]): 
            for rinter in self.r_intervals:
                mask = (self.field_df[self.ks_metric] < rinter[1]) & (self.field_df[self.ks_metric] > rinter[0]) & (self.field_df.ca == ca) & \
                       (np.isnan(self.field_df.phaselag_AB) == False) & (np.isnan(self.field_df.phaselag_BA) == False)
                
                num_samples = self.field_df[mask].shape[0]
                num_samples_list.append(num_samples)
        selected_bs_size = np.max(num_samples_list)
        self._msg('All num_samples = \n%s' % str(np.around(num_samples_list, 3)))
        self._msg('Selected bootstrapping size = %d' % selected_bs_size)
        return selected_bs_size
    
    def bootstrap(self):
        if self.bs_size is None:
            self.bs_size = self._get_bs_size()
            
        # Random seeds
        for ca_idx, ca in enumerate(['CA{}'.format(x+1) for x in range(3)]): 
            self._msg('Bootstraping %s' % ca)
            for rinter in self.r_intervals:
                
                
                zero_phaser = ThetaEstimator(0.005, 0.3, [5, 12])  # Parameters must be the same as the pre-processing step

                mask = (self.field_df[self.ks_metric] < rinter[1]) & (self.field_df[self.ks_metric] > rinter[0]) & (self.field_df.ca == ca) & \
                       (np.isnan(self.field_df.phaselag_AB) == False) & (np.isnan(self.field_df.phaselag_BA) == False)
                df_rwithin = self.field_df[mask]
                num_samples = df_rwithin.shape[0]

                if (num_samples == 0) or (num_samples > self.bs_size):
                    self._msg('R=[%0.3f, %0.3f] %s N_samples = %d '% (rinter[0], rinter[1], 'skipped', num_samples))
                    continue
                new_to_add = self.bs_size - num_samples
                current_iter = 0
                self._msg('R=[%0.3f, %0.3f], N_samples = %d , %d to add'% (rinter[0], rinter[1], num_samples, new_to_add))

                while (current_iter < new_to_add):
                    df_sampled = df_rwithin.sample(frac=1, random_state=self.seed_num)
                    added_field_df_dict = {key:[] for key in list(self.field_df.columns)}
                    for i in range(df_sampled.shape[0]):
                        if current_iter > new_to_add:
                            break
                        AB1, AB2, BA1, BA2 = df_sampled.iloc[i][['AB_tsp1', 'AB_tsp2', 'BA_tsp1', 'BA_tsp2']]  
                        AB1np, AB2np, BA1np, BA2np = np.array(AB1), np.array(AB2), np.array(BA1), np.array(BA2)
                        np.random.seed(self.seed_num)
                        permAB = np.random.choice(len(AB1np), size=len(AB1np), replace=True)
                        np.random.seed(self.seed_num)
                        permBA = np.random.choice(len(BA1np), size=len(BA1np), replace=True)

                        _, phaselag_AB, corr_info_AB = zero_phaser.find_theta_isi_hilbert(AB1np[permAB], AB2np[permAB])
                        
                        _, phaselag_AB_flip, corr_info_AB_flip = zero_phaser.find_theta_isi_hilbert(AB2np[permAB], AB1np[permAB])
                        _, phaselag_BA, corr_info_BA = zero_phaser.find_theta_isi_hilbert(BA1np[permBA], BA2np[permBA])
                        _, phaselag_BA_flip, corr_info_BA_flip = zero_phaser.find_theta_isi_hilbert(BA2np[permBA], BA1np[permBA])
                        
                        if np.isnan(phaselag_AB) or np.isnan(phaselag_BA):
                            self.seed_num += 1
                            continue
                        
                        
                        for key in added_field_df_dict.keys():
                            added_field_df_dict[key].append(df_sampled[key].iloc[i])


                        added_field_df_dict['phaselag_AB'][-1] = phaselag_AB
                        added_field_df_dict['phaselag_BA'][-1] = phaselag_BA
                        added_field_df_dict['phaselag_AB_flip'][-1] = phaselag_AB_flip
                        added_field_df_dict['phaselag_BA_flip'][-1] = phaselag_BA_flip
                        added_field_df_dict['AB_tsp1'][-1] = AB1
                        added_field_df_dict['AB_tsp2'][-1] = AB2
                        added_field_df_dict['BA_tsp1'][-1] = BA1
                        added_field_df_dict['BA_tsp2'][-1] = BA2
                        added_field_df_dict['corr_info_AB'][-1] = corr_info_AB
                        added_field_df_dict['corr_info_BA'][-1] = corr_info_BA


                        current_iter += 1
                        self.seed_num += 1


                    added_field_df = pd.DataFrame(added_field_df_dict)
                    self.field_df = pd.concat([self.field_df, added_field_df], axis=0, ignore_index=True, sort=False)
                    
        _ = self._get_bs_size()
        return self.field_df