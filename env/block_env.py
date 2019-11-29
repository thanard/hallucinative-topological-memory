from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
import scipy.misc
import time
import argparse

# From xml file & same order
geom_size = 2 * np.array([[.8, .8, .1],
                          [.12, .12, .12],
                          [.06, .3, .12]])
idx = {"table": 0,
       "obj1": 1,
       "obs1": 2}

class BlockMoveEnvRandomized(MujocoEnv, Serializable):
    global FILE

    def __init__(self, *args, **kwargs):

        FILE = 'env/block_dynamic_obst.xml'
        kwargs['file_path'] = FILE
        super(BlockMoveEnvRandomized, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        return self.model.data.geom_xpos.copy()

    def get_current_img(self, config=None):
        img = self.render(mode='rgb_array', config=config)
        img = scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
        return img

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 2.75
        self.viewer.cam.elevation = -60

    def detect_contact(self, pos):
        """Detect if there is any contact.

        :param pos: configurations of all objects
        :return: True if any two objects overlap, False otherwise.
        """
        for i in range(len(geom_size)):
            for j in range(i+1, len(geom_size)):
                if np.all(np.abs(pos[j] - pos[i]) + 1e-6 < np.abs(geom_size[j] + geom_size[i])/2):
                    return True
        return False

    def clip_action_avoiding_contacts(self, pos, act):
        """Action clipping to avoid contacts/collisions.

        :param pos: configurations of all objects
        :param act: displacement applied to Block (obj1).
        :return: a clipped action that avoids collision into Wall (obs1).
        """
        min_distance = np.abs(geom_size[idx["obj1"]] + geom_size[idx["obs1"]])/2
        curr_distance = pos[idx["obj1"]][:2] + act - pos[idx["obs1"]][:2]
        change_act = np.zeros(2) # say x - k
        # Find x
        for i in range(2):
            k = act[i]
            d = min_distance[i]
            c = curr_distance[i] - k
            # d >= 0
            # |x + c| >= d
            if np.abs(k + c) < d:
                # -d < k + c < d
                if k >= 0:
                    # Pushing back along the dimension taken
                    # k >= x >= 0
                    # We know that
                    # x + c <= k + c < d
                    # Thus, x + c <= -d
                    # max possible x is -c-d
                    d = min_distance[i]
                    if c <= - d + 1e-6:
                        change_act[i] = (-c-d) - k
                    else:
                        change_act[i] = np.inf
                else:
                    # Pushing back along the dimension taken
                    # k <= x <= 0
                    # We know that
                    # x + c >= k + c > -d
                    # Thus, x + c >= d
                    # max possible x is d-c
                    if -c <= -d + 1e-6:
                        change_act[i] = (d-c) - k
                    else:
                        change_act[i] = np.inf
        min_idx = np.argmin(np.abs(change_act))
        action = act.copy()
        action[min_idx] += change_act[min_idx]
        return action

    def clip_action_boundary(self, pos, act):
        """Action clipping to going over table boundary.

        :param pos: configurations of all objects
        :param act: displacement applied to Block (obj1).
        :return: a clipped action that avoids going over Table's boundary.
        """
        table_size = geom_size[idx["table"]]
        obj1_size = geom_size[idx["obj1"]]
        loc_obs = (table_size - obj1_size)[:2]
        action = \
            np.clip(act,
                    -loc_obs/2 - pos[idx["obj1"]][:2],
                    loc_obs/2 - pos[idx["obj1"]][:2]
                    )
        return action

    def get_action_adjustment(self, pos, action):
        """Apply all action adjustments.

        :param pos: configurations of all objects
        :param act: displacement applied to Block (obj1).
        :return: The final action that gaurantees physical feasibility.
        """
        clip_at = self.get_max_action()
        action = np.clip(action, -clip_at, clip_at)
        action = self.clip_action_boundary(pos, action)
        action = self.clip_action_avoiding_contacts(pos, action)
        return action

    def get_action_dim(self):
        return 2

    def get_max_action(self):
        return 0.05

    def step_only(self, action):
        obs = self.get_current_obs()
        action = self.get_action_adjustment(obs, action)
        a = np.zeros_like(self.get_current_obs())
        a[idx["obj1"], :2] = action
        obs += a
        return self.reset(init_state=obs), action

    def get_object(self, to_remove=("obj1", ), config=None):
        state = self.get_current_obs()
        keep_states = {}
        # Removing objects
        for r in to_remove:
            keep_states[idx[r]] = state[idx[r], :2].copy()
            state[idx[r], :2] = 100
        # Reset & rendering objects
        full_state = np.zeros((1, 6 * 4))
        full_state[0, :6] = (state - self.model.body_pos)[1:, :].reshape(-1)
        super(BlockMoveEnvRandomized, self).reset(init_state=full_state)
        for i in range(10):
            img = self.render(mode='rgb_array', config=config)
        img = scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
        # Change the states back
        for r in to_remove:
            state[idx[r], :2] = keep_states[idx[r]]
        full_state = np.zeros((1, 6 * 4))
        full_state[0, :6] = (state - self.model.body_pos)[1:, :].reshape(-1)
        super(BlockMoveEnvRandomized, self).reset(init_state=full_state)
        return img

    def reset(self, init_state=None, keep_obs=False):
        """
        :param init_state: geom_xpos positions for the first and the second block.
        :return: set geom_xpos
        """
        table_size = geom_size[idx["table"]]
        table_z = table_size[2]
        obj1_size = geom_size[idx["obj1"]]
        obs1_size = geom_size[idx["obs1"]]
        loc_obj = (table_size - obj1_size)[:2]
        loc_obs = (table_size - obs1_size)[:2]

        while init_state is None or self.detect_contact(init_state):
            init_state = np.zeros_like(geom_size)
            init_state[idx["obj1"], :2] = (np.random.rand(2) - .5) * loc_obj
            init_state[idx["obj1"], 2] = (table_z + obj1_size[2])/2
            # Avoiding too close to boundary and cannot push up when making contact
            init_state[idx["obs1"], :2] = (np.random.rand(2) - .5) * (loc_obs - 2*obj1_size[:2])
            init_state[idx["obs1"], 2] = (table_z + obs1_size[2])/2
            if keep_obs:
                state = self.get_current_obs()
                init_state[idx["obs1"]] = state[idx["obs1"]].copy()

        # Cap init state to be in range.
        init_state[idx["obj1"], :2] = np.clip(init_state[idx["obj1"], :2], -loc_obj/2, loc_obj/2)
        init_state[idx["obs1"], :2] = np.clip(init_state[idx["obs1"], :2], -loc_obs/2, loc_obs/2)

        init_pos = (init_state - self.model.body_pos)[1:, :].reshape(-1)
        full_state = np.zeros((1, 6 * 4))
        full_state[0, :6] = init_pos

        super(BlockMoveEnvRandomized, self).reset(init_state=full_state)
        return self.get_current_obs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Move some blocks')
    parser.add_argument('--length', type=int, default=100,
                        help='max # steps after which you terminate a single simulation')
    parser.add_argument('--n_trials', type=int, default=150, help='number of simulations')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_contexts', type=int, default=1)
    parser.add_argument('-test', action="store_true")
    parser.add_argument('-visible', action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    config = {
        "visible": args.visible,
        "init_width": 128,
        "init_height": 128,
        "go_fast": not args.visible
    }
    start = time.time()
    all_data = []
    n_samples = 0
    n_trials = args.n_trials
    n_contexts = args.n_contexts
    n_save_loops = 1 if args.test else 2

    env = BlockMoveEnvRandomized()
    for i in range(n_contexts):
        print("##### Context %d/%d ######" % (i, n_contexts))
        env.reset()
        env.render(config=config)
        if i == 0:
            env.viewer_setup()
            for i in range(100):
                env.render(config=config)
        if n_contexts > 1:
            context = None
            while context is None or context.sum() == 0:
                context = env.get_object(config=config)

        if args.test and n_contexts > 1:
            all_data.append(context)
            continue

        for j in range(n_trials):
            # Reset only the object positions
            env.reset(keep_obs=True)
            img = env.render(mode='rgb_array', config=config)
            img = scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
            data = []
            while True:
                if env.detect_contact(env.get_current_obs()):
                    import ipdb
                    ipdb.set_trace()
                action = np.random.normal(
                    loc=0,
                    scale=0.05,
                    size=env.get_action_dim())
                s_next, action = env.step_only(action)
                img = env.render(mode='rgb_array', config=config)
                img = scipy.misc.imresize(img, (64, 64, 3), interp='nearest')
                label = {'state': s_next, 'action': action}
                if n_contexts > 1:
                    img = np.concatenate([img, context], axis=2)
                    from PIL import Image
                    new_im = Image.fromarray(context)
                    new_im.save("context.png")
                data.append((img, label))
                if len(data) > args.length:
                    break
            n_samples += len(data)
            all_data.append(data)

    total = time.time() - start
    print("number of trajectories: %d" % (len(all_data)))
    print("total sample size: %d" % (n_samples))
    if args.test:
        fname = 'data/test_context_%d_sharper' % n_contexts
    else:
        fname = 'data/randact_traj_length_%d_n_trials_%d_n_contexts_%d_sharper.npy' % (args.length, n_trials, n_contexts)
    np.save(fname, all_data)
    print("took {} seconds".format(total))
