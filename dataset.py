import numpy as np

from utils import from_numpy_to_var
from scripts.data_configs import n_trials, traj_length

def get_torch_images_from_numpy(npy_list, conditional, normalize=True, one_image=False):
    """
    :param npy_list: a list of (image, attrs) pairs
    :param normalize: if True then the output is between 0 and 1
    :return: Torch Variable as input to model
    """
    if one_image:
        o = from_numpy_to_var(np.transpose(npy_list, (0, 3, 1, 2)))
    else:
        o = from_numpy_to_var(np.transpose(np.stack(npy_list[:, 0]), (0, 3, 1, 2)))
    if normalize:
        o /= 255
    if one_image:
        return o
    if conditional:
        return o[:, :3].contiguous(), o[:, 3:].contiguous()
    return o, None


def get_negative_examples(data, idx, batch_size, N, conditional):
    idx = np.tile(idx, N)
    idx_neg = np.random.choice(n_trials, batch_size * N)
    t_neg = np.random.choice(traj_length, batch_size * N)
    o_npy = data[(idx // n_trials) * n_trials + idx_neg, t_neg]
    return get_torch_images_from_numpy(o_npy, conditional)[0]


def get_torch_actions(npy_list):
    act = np.array([npy_list[i, 1]['action'].reshape(-1) for i in range(npy_list.shape[0])])
    return from_numpy_to_var(act)


def get_idx_t(batch_size, k_steps, n_trajs, data):
    idx = np.random.choice(n_trajs, size=batch_size)
    # Safe but slower --- doesn't assume the same length across trajectories
    t = np.array([np.random.randint(len(data[i]) - k_steps) for i in idx])
    return idx, t
