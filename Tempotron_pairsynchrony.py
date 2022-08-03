import numpy as np
import matplotlib.pyplot as plt
from library.Tempotron import Tempotron
import time

def demo():
    # # Input patterns (X) and labels (Y)
    M = 700
    N = 500
    T = 500
    num_iter = 2000
    lr = 0.005
    nsp = 3
    assert N % 2 == 0
    assert M % 2 == 0



    minus_gpid1 = np.random.permutation(int(N/2))
    minus_gpid2 = minus_gpid1 + int(N/2)


    np.random.seed(0)
    Uplus = np.random.uniform(0, T, size=(int(M/2), int(N/2), nsp))  # (+) patterns
    np.random.seed(1)
    plus_gpid = np.random.permutation(N)
    plus_gpid1 = plus_gpid[:int(N/2)]
    plus_gpid2 = plus_gpid[int(N/2):]
    Uplus_all = np.zeros((int(M/2), N, nsp))
    Uplus_all[:, plus_gpid1, :] = Uplus
    Uplus_all[:, plus_gpid2, :] = Uplus


    np.random.seed(2)
    Uminus = np.random.uniform(0, T, size=(int(M/2), int(N/2), nsp))  # (+) patterns
    np.random.seed(3)
    minus_gpid = np.random.permutation(N)
    minus_gpid1 = minus_gpid[:int(N/2)]
    minus_gpid2 = minus_gpid[int(N/2):]
    Uminus_all = np.zeros((int(M/2), N, nsp))
    Uminus_all[:, minus_gpid1, :] = Uminus
    Uminus_all[:, minus_gpid2, :] = Uminus

    X = np.vstack([Uplus_all, Uminus_all])
    Y = np.array([True] * int(M/2) +  [False] * int(M /2))

    # # Initialize Tempotron class


    Vthresh = 1
    tau = 15
    tau_s = tau/4
    w_seed = 0
    temN = Tempotron(N=N, lr=lr, Vthresh=Vthresh, tau=tau, tau_s=tau_s, w_seed=w_seed)

    # Initilizae time for calculating the kernel value. It should be in the same range as the spike times
    dt = 0.1
    t = np.arange(0, 100, dt)

    # Predict before training
    Y_pred, koutlist_pred, tspout_list = temN.predict(X, t)

    Y_pred = np.array(Y_pred)
    acc = np.mean(Y_pred == Y)
    print('Acc before training = %0.3f' % acc)

    # Training
    t1 = time.time()
    temN.train(X, Y, t, num_iter=num_iter)
    print('Time used = %0.2fs'%(time.time() - t1))

    w = temN.w
    fig, ax = plt.subplots()

    ax.scatter(w[plus_gpid1], w[plus_gpid2], c='b', alpha=0.5, marker='.', label='(+)')
    ax.scatter(w[minus_gpid1], w[minus_gpid2], c='r', alpha=0.5, marker='.', label='(-)')

    fig.savefig('w.png', dpi=150)


if __name__ == '__main__':
    demo()

