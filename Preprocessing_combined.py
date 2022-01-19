import numpy as np

from Preprocessing_step1 import PreprocessStep1
from Preprocessing_step2 import SingleField_Preprocess, PairField_Preprocess
from library.utils import load_pickle




# tags = ['Idirect-10', 'Idirect-5', 'Idirect-2']
tags = ['Idirect-10']

for tag in tags:
    simdata = load_pickle('data/simulated_%s.pickle'%(tag))
    # simdata['Activity'] = simdata['ExpoActivity']
    m = simdata['Activity'].copy()
    simdata['Activity'] = np.exp((simdata['Activity'] - 5)/4)
    simdata['Activity'][m < (1e-3)] = 0

    save_pth = 'data/processed1_%s_expo2.pickle'%(tag)
    simdata = PreprocessStep1(simdata, save_pth)
    # simdata = load_pickle(save_pth)

    save_pth = 'data/Single_processed2_%s_expo2.pickle'%(tag)
    SingleField_Preprocess(processed1data=simdata, save_pth=save_pth)

    # save_pth = 'data/Pair_processed2_%s.pickle'%(tag)
    # PairField_Preprocess(processed1data=simdata, save_pth=save_pth)






