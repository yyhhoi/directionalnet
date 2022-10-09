import numpy as np
import pandas as pd

from library.comput_utils import acc_metrics


class PSPkernel:
    def __init__(self, tau, tau_s):
        """Alpha function for post-syanptic potential.

        Args:
            tau (float): Decay time constant of the alpha function kernel.
            tau_s (float): Rising time constant of the alpha function kernel. In the paper, tau_s = tau/4
        """
        self.tau = tau
        self.tau_s = tau_s
        self.tmax = tau*tau_s * np.log(tau/tau_s) / (tau - tau_s)
        self.norm = np.exp(-self.tmax/self.tau) - np.exp(-self.tmax/self.tau_s)
    def getk(self, t, t_i):
        """Evaluate the alpha function based on the time difference (t - t_i). For t < t_i, it returns zeros.

        Args:
            t (ndarray or float): An array of time axis or a time point.
            t_i (ndarray or float): An array of spike times or a spike time.

        Returns:
            Kernel values (ndarray or float).
        """
        out = ( np.exp(-(t-t_i)/self.tau) - np.exp(-(t-t_i)/self.tau_s) ) / self.norm
        out[t < t_i] = 0
        return out



class Tempotron:
    def __init__(self, N, lr=0.1, Vthresh=1.5, tau=4.0, tau_s=1.0, w_seed=0):
        """Tempotron with a pre-defined number of synaptic sites.

        Args:
            N (int): A pre-set number of synapses of the tempotron.
            lr (float, optional): Learning rate. Defaults to 0.1.
            Vthresh (float, optional): Threshold of membrane potential for producing a spike. Defaults to 1.5.
            tau (float, optional): Decay time constant of the PSP kernel. Defaults to 4.0.
            tau_s (float, optional): Rising time constant of the PSP kernel. Defaults to 1.0.
            w_seed (int, optional): Random state of the synaptic weights initialization. Defaults to 0.
        """
        self.N = N
        self.lr = lr
        self.Vthresh = Vthresh
        self.kern = PSPkernel(tau=tau, tau_s=tau_s)
        np.random.seed(w_seed)
        self.w = np.random.uniform(-0.01, 0.01, size=N)
        # self.w = np.random.uniform(-1, 1, size=N)


    def train(self, X, Y, t, num_iter, progress=True):
        """Given M number of input pattern X from N synapses, train the tempotron to predict Y.

        Args:
            X (list): List of M samples of patterns. Each contains N lists of spike time ndarrays.
            Y (list): List of M boolean labels. True means there should a an output spike.
            t (ndarray): Time axis used to evaluate the kernel values. Should encompass all spike times.
            num_iter (int): Number of iterations for the training.
            progress (bool): True if number of iterations should be printed.

        Raises:
            IndexError: If sample number of inputs does not match the output.
            IndexError: if number of synapses does not match the instance initialization.
        """

        MX = len(X)  # Number of input pattern samples
        NX = len(X[0])  # Number of synapses
        MY = len(Y)  # Number of labels


        if MX != MY:
            raise IndexError('Numbers of input patterns and labels must be the same.')
        if NX != self.N:
            raise IndexError('Number of synapses must be the same as the initialized.')

        Y = np.array(Y)
        flat_tsp_list = []
        M_list = []
        N_list = []
        for mi in range(MX):
            for ni in range(NX):
                tsp_list = X[mi][ni]
                tsp_len = len(tsp_list)
                flat_tsp_list.extend(tsp_list)
                M_list.extend([mi] * tsp_len)
                N_list.extend([ni] * tsp_len)
        alltspallm = np.array(flat_tsp_list)
        vtraces = self.kern.getk(t.reshape(-1, 1), alltspallm.reshape(1, -1))
        spdf = pd.DataFrame(dict(tsp=flat_tsp_list, M=M_list, N=N_list))

        num_correct = 0
        for iter_i in range(num_iter):
            if num_correct >= MX:
                break
            all_spikeFlags = np.zeros(MX)
            num_correct = 0
            for mi in range(MX):
                label = Y[mi]  # True or False
                M_mask = spdf['M'] == mi
                nidices = spdf.loc[M_mask, 'N']
                alltsp = spdf.loc[M_mask, 'tsp']
                allkout = vtraces[:, M_mask] * self.w[nidices].reshape(1, -1)
                kout = allkout.sum(axis=1)
                thresh_idxs = np.where(kout > self.Vthresh)[0]
                if thresh_idxs.shape[0] < 1:  # No Spike
                    all_spikeFlags[mi] = False
                    if label:  # (+) pattern. Error trial. Update weights
                        tmax = t[kout.argmax()]
                        self.weight_update(alltsp, nidices, tmax, plus=True)
                        # mask = alltsp < tmax
                        # nidices_inmax = nidices[mask]
                        # dw_alltsp = self.kern.getk(tmax, alltsp[mask]) * self.lr
                        # for synid in np.unique(nidices_inmax):
                        #     self.w[synid] = self.w[synid] + dw_alltsp[nidices_inmax == synid].sum()

                    else:  # (-) pattern. Correct trial. Do nothing
                        num_correct += 1

                else:  # Have spike
                    all_spikeFlags[mi] = True
                    if label:  # (+) pattern. Correct trial. Do nothing
                        num_correct += 1

                    else:  # (-) pattern. Error trial. Update weight

                        # Shunting inhibition after the first spike, it's required to find the real tmax
                        # by re-calculating the post-synaptic membrane potential up until the first spike.
                        t_thresh = t[thresh_idxs[0]]
                        mask_thresh = alltsp < t_thresh
                        nidices_inthresh = nidices[mask_thresh]
                        kout_recalctmp = vtraces[:, mask_thresh[mask_thresh].index] * self.w[nidices_inthresh].reshape(1, -1)
                        kout_recalc = kout_recalctmp.sum(axis=1)
                        tmax = t[kout_recalc.argmax()]
                        self.weight_update(alltsp, nidices, tmax, plus=False)
                        # mask = alltsp < tmax
                        # nidices_inmax = nidices[mask]
                        # dw_alltsp = self.kern.getk(tmax, alltsp[mask]) * self.lr
                        # for synid in np.unique(nidices_inmax):
                        #     self.w[synid] = self.w[synid] - dw_alltsp[nidices_inmax == synid].sum()
            if progress:
                print('\r Iter %d/%d: Correct trials %d/%d = %0.3f'%(iter_i+1, num_iter, num_correct, MX, num_correct/MX), end='', flush=True)

            yield all_spikeFlags, self.w.copy()

    def weight_update(self, alltsp, nidices, tmax, plus):
        mask = alltsp < tmax
        nidices_inmax = nidices[mask]
        dw_alltsp = self.kern.getk(tmax, alltsp[mask]) * self.lr

        synid_all, spcounts_all = np.unique(nidices_inmax, return_counts=True)
        for i in range(synid_all.shape[0]):
            synid = synid_all[i]
            if plus:
                self.w[synid] = self.w[synid] + dw_alltsp[nidices_inmax == synid].sum()
            else:
                self.w[synid] = self.w[synid] - dw_alltsp[nidices_inmax == synid].sum()
    def predict(self, X, t, shunt=False):
        """Given input pattern X, predict whether there is output spike

        Args:
            X (list): List of M samples of patterns. Each contains N lists of spike time ndarrays.
            t (ndarray): Time axis used to evaluate the kernel values. Should encompass all spike times.

        Raises:
            IndexError: If sample number of inputs does not match the output.

        Returns:
            Y, kout_list, tspout_list: lists of labels, membrane potential values and output spike times.
        """

        MX = len(X)  # Number of input pattern samples
        NX = len(X[0])  # Number of synapses

        if NX != self.N:
            raise IndexError('Number of synapses must be the same as the initialized.')

        flat_tsp_list = []
        M_list = []
        N_list = []
        for mi in range(MX):
            for ni in range(NX):
                tsp_list = X[mi][ni]
                tsp_len = len(tsp_list)
                flat_tsp_list.extend(tsp_list)
                M_list.extend([mi] * tsp_len)
                N_list.extend([ni] * tsp_len)
        alltspallm = np.array(flat_tsp_list)
        vtraces = self.kern.getk(t.reshape(-1, 1), alltspallm.reshape(1, -1))
        spdf = pd.DataFrame(dict(tsp=flat_tsp_list, M=M_list, N=N_list))

        Y_pred = []
        kout_list = []
        tspout_list = []

        for mi in range(MX):
            M_mask = spdf['M'] == mi
            nidices = spdf.loc[M_mask, 'N']
            alltsp = spdf.loc[M_mask, 'tsp']
            allkout = vtraces[:, M_mask] * self.w[nidices].reshape(1, -1)
            kout = allkout.sum(axis=1)
            thresh_idxs = np.where(kout > self.Vthresh)[0]

            if thresh_idxs.shape[0] > 0:
                Y_this = True
                if shunt:
                    t_thresh = t[thresh_idxs[0]]
                    mask_thresh = alltsp < t_thresh
                    nidices_inthresh = nidices[mask_thresh]
                    kout_recalctmp = vtraces[:, mask_thresh[mask_thresh].index] * self.w[nidices_inthresh].reshape(1, -1)
                    kout = kout_recalctmp.sum(axis=1)

            else:
                Y_this = False
            Y_pred.append(Y_this)
            kout_list.append(kout)
            tspout_list.append(t[thresh_idxs])

        Y_pred = np.array(Y_pred)
        kout_all = np.stack(kout_list)
        tspout_all = np.array(tspout_list, dtype=object)
        return Y_pred, kout_all, tspout_all

