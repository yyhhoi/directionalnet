import numpy as np

def createMosProjMat_p2p(startpos, endpos, posvec_mos, posvec_CA3, act_max, act_sd):
    """
    Create weight matrix with shape (N, M), from M mossy neurons at the starting xy- position to
    N CA3 neurons at the ending xy- position.

    Parameters
    ----------
    startpos: ndarray
        Shape (2, ). Starting xy- position of the mossy projection.
    endpos: ndarray
        Shape (2, ). Targeted ending position of the mossy projection.
    posvec_mos: ndarray
        Shape (M, 2). xy- positional tuning of M mossy neuron's.
    posvec_CA3: ndarray
        Shape (N, 2). xy- positional tuning of N CA3 neuron's.
    act_max: float
        Peak value of the gaussian-shape activation strengths of mossy projection.
    act_sd: float
        SD of the gaussian-shape activation strengths of mossy projection.

    Returns
    -------
    w_mosCA3: ndarray
        Shape (N, M). Weight matrix.
    mos_act: ndarray
        Shape (M, ). Post-synaptic activation strength.

    """
    diff_vec = endpos - startpos

    mos_start_diff = posvec_mos - startpos.reshape(1, 2)
    mos_act = np.exp(-np.sum(np.square(mos_start_diff), axis=1) / (2 * (act_sd ** 2)))

    # mos_offset = posvec_mos + diff_vec.reshape(1, 2)
    ca3_offset = posvec_CA3 - diff_vec.reshape(1, 2)
    CA3_end_xdiff = ca3_offset[:, 0].reshape(-1, 1) - posvec_mos[:, 0].reshape(1, -1)  # (N, M)
    CA3_end_ydiff = ca3_offset[:, 1].reshape(-1, 1) - posvec_mos[:, 1].reshape(1, -1)  # (N, M)
    CA3_end_squarediff = (CA3_end_xdiff ** 2) + (CA3_end_ydiff ** 2)  # (N, M)
    CA3_expo = np.exp(-CA3_end_squarediff / (2 * (act_sd ** 2)))  # (N, M)
    w_mosCA3 = act_max * mos_act.reshape(1, -1) * CA3_expo
    return w_mosCA3, mos_act


def createMosProjMat3D_p2p(startpos, endpos, posvec_mos, posvec_CA3, act_max, act_sd, act_adiff_max):
    """
    Create weight matrix with shape (N, M), from M mossy neurons at the starting xy- position to
    N CA3 neurons at the ending xy- position.

    Parameters
    ----------
    startpos: ndarray
        Shape (2, ). Starting xy- position of the mossy projection.
    endpos: ndarray
        Shape (2, ). Targeted ending position of the mossy projection.
    posvec_mos: ndarray
        Shape (M, 2). xy- positional tuning of M mossy neuron's.
    posvec_CA3: ndarray
        Shape (N, 2). xy- positional tuning of N CA3 neuron's.
    act_max: float
        Peak value of the gaussian-shape activation strengths of mossy projection.
    act_sd: float
        SD of the gaussian-shape activation strengths of mossy projection.
    act_adiff_max: ndarray
        Shape (M, N). Precalculated directional dependent weight's maximum.

    Returns
    -------
    w_mosCA3: ndarray
        Shape (N, M). Weight matrix.
    mos_act: ndarray
        Shape (M, ). Post-synaptic activation strength.

    """
    mos_act = np.exp(-np.sum(np.square(posvec_mos - startpos.reshape(1, 2)), axis=1) / (2 * (act_sd ** 2)))

    diff_vec = endpos - startpos
    ca3_offset = posvec_CA3 - diff_vec.reshape(1, 2)
    CA3_end_xdiff = ca3_offset[:, 0].reshape(-1, 1) - posvec_mos[:, 0].reshape(1, -1)  # (N, M)
    CA3_end_ydiff = ca3_offset[:, 1].reshape(-1, 1) - posvec_mos[:, 1].reshape(1, -1)  # (N, M)
    CA3_end_squarediff = (CA3_end_xdiff ** 2) + (CA3_end_ydiff ** 2)  # (N, M)
    CA3_expo = np.exp(-CA3_end_squarediff / (2 * (act_sd ** 2)))  # (N, M)
    w_mosCA3 = (act_max + act_adiff_max) * mos_act.reshape(1, -1) * CA3_expo
    return w_mosCA3, mos_act