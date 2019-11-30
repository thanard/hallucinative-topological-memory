import numpy as np
import torch
from env.block_env import BlockMoveEnvRandomized, geom_size
from dataset import get_torch_images_from_numpy

test_locs = []
config = {
    "visible": False,
    "init_width": 128,
    "init_height": 128,
    "go_fast": True
}


def get_env(name):
    if name == "block":
        env = BlockMoveEnvRandomized()
    else:
        raise NotImplementedError
    return env


def get_n_test_locs(name):
    if name == "block":
        return 20


def generate_samples(name, env, n_samples=100):
    if name == "block":
        # Set first loc
        loc = [[.05, .42], [0., 0.]]
        start_state = np.zeros((3, 3))
        start_state[1:, :2] = loc
        start_state[1:, 2] = .22
        env.reset(start_state)
        all_images = []
        for i in range(n_samples):
            env.reset(keep_obs=True)
            img = env.get_current_img(config)
            img = get_torch_images_from_numpy(img[None, :], False, one_image=True)
            all_images.append(img)
        return torch.cat(all_images)
    return None


def set_valid_loc(name, env):
    if name == "block":
        # Goal img
        goal_state = env.reset()
        goal_img = env.get_current_img(config)
        goal_img = get_torch_images_from_numpy(goal_img[None, :], False, one_image=True)
        # Start img
        start_state = env.reset(keep_obs=True)
        start_img = env.get_current_img(config)
        start_img = get_torch_images_from_numpy(start_img[None, :], False, one_image=True)
        # Context img
        context = env.get_object(config=config)
        context_img = get_torch_images_from_numpy(context[None, :], False, one_image=True)
        return start_img, goal_img, start_state, goal_state, context_img
    else:
        raise NotImplementedError


def set_test_loc(i, name, env, config, g_npy=None, joints=False, f=''):
    """
    :param i: the index of test location
    :param name: name of env
    :param env: current env
    :return: start, goal, context observations (torch) and set the env to start (size
    1 x C x W x H)
    """
    global test_locs
    if name == "block":
        if len(test_locs) == 0:
            test_locs = [[[[.05, .42], [0., 0.]], [[-.02, -.42], [0., 0.]]],
                         [[[-0.535, .1609], [0., 0.]], [[0.3609, 0.0458], [0., 0.]]],
                         [[[.31, .02], [0., 0.]], [[-.2, -.05], [0., 0.]]],
                         [[[-.02, -.42], [0., 0.]], [[.05, .42], [0., 0.]]],
                         [[[.4, .51], [0., 0.]], [[-.52, -.58], [0., 0.]]],
                         [[[-.52, -.58], [0., 0.]], [[.4, .51], [0., 0.]]],
                         [[[-.55, .52], [0., 0.]], [[.49, -.56], [0., 0.]]],
                         [[[.49, -.56], [0., 0.]], [[-.55, .52], [0., 0.]]],
                         [[[.21, -.39], [0., 0.]], [[.49, .5], [0., 0.]]],
                         [[[-.23, -.07], [0., 0.]], [[.21, .01], [0., 0.]]],
                         [[[.53, -.01], [0., 0.]], [[-.45, .03], [0., 0.]]],
                         [[[-.03, -.61], [0., 0.]], [[.17, .58], [0., 0.]]],
                         [[[-.46, .59], [0., 0.]], [[-.32, -.51], [0., 0.]]],
                         [[[.485, -.61], [0., 0.]], [[-.39, -.31], [0., 0.]]],
                         [[[-.02, -.42], [0., 0.]], [[-.18, .37], [0., 0.]]],
                         [[[-.585, .59], [0., 0.]], [[.57, -.01], [0., 0.]]],
                         [[[.572, -.45], [0., 0.]], [[-.566, -.09], [0., 0.]]],
                         [[[.55, .3], [0., 0.]], [[-.55, -.3], [0., 0.]]],
                         [[[-.333, -.42], [0., 0.]], [[.18, .35], [0., 0.]]],
                         [[[.524, .555], [0., 0.]], [[-.56, .01], [0., 0.]]]]
        start_state = np.zeros((3, 3))
        start_state[1:, :2] = test_locs[i][0]
        start_state[1:, 2] = .22
        goal_state = np.zeros((3, 3))
        goal_state[1:, :2] = test_locs[i][1]
        goal_state[1:, 2] = .22
        start_state[1:, 1] -= 0.2
        goal_state[1:, 1] -= 0.2
        # Goal img
        goal_state = env.reset(goal_state)
        goal_img = env.get_current_img(config)
        goal_img = get_torch_images_from_numpy(goal_img[None, :], False, one_image=True)

        # Start img
        start_state = env.reset(start_state)
        start_img = env.get_current_img(config)
        start_img = get_torch_images_from_numpy(start_img[None, :], False, one_image=True)

        # Context img
        context = env.get_object(config=config)
        context_img = get_torch_images_from_numpy(context[None, :], False, one_image=True)
        return start_img, goal_img, start_state, goal_state, context_img
    else:
        raise NotImplementedError




