import numpy as np
import os
import torch

from models import get_score
from env.eval_configs import get_env, get_n_test_locs, set_test_loc, set_valid_loc, generate_samples
from planning import build_memory_graph, localization, find_shortest_path, find_next_way_point, find_threshold
from torchvision.utils import save_image
from utils import write_number_on_images, set_requires_grad, from_numpy_to_var
from dataset import get_torch_images_from_numpy


def get_hp(new_hp):
    default_hp = {
        "init_width": 128,
        "init_height": 128,
        "n_test_locs": 100,
        "test_mode": "eval", # valid or eval
        "T": 501,
        "action_noise": 0.01,
        "block_size": 0.24, # for reach-goal threshold
        ### Planning configs ###
        "replanning_every": 200,
        "n_planning_samples": 300,
        "use_edge_weight": True,
        "use_vae_samples": True,
        "threshold_edge": 0, # TODO: currently not used
        "threshold_shortcut": 60, # TODO: currently not used
        "run_inverse_model": True,
        "score_type": "exp-neg"
    }
    if type(new_hp) == dict:
        default_hp.update(new_hp)
    return default_hp


def run_inverse_model(env, config, test_data, n_test_locs, hp, actor):
    total_dist = total_success = 0
    for i in range(n_test_locs):
        curr_obs, goal_obs, start_state, goal_state, context = test_data[i]
        env.reset(start_state)
        env.render(config=config)
        for t in range(hp["T"]):
            action = actor(curr_obs, goal_obs, context).cpu().numpy()
            _s_next, _a_taken = env.step_only(action + np.random.randn(2) * hp["action_noise"])
            # print(_s_next, _action)
            next_img = env.get_current_img(config)
            curr_obs = get_torch_images_from_numpy(next_img[None, :], False, one_image=True)
            curr_dist = np.linalg.norm(env.get_current_obs() - goal_state)
            is_success = (np.abs(env.get_current_obs() - goal_state) < hp["block_size"]).all()
            if is_success:
                break
        print("Final Distance for Task %d: %f; Success %s" % (i, curr_dist, is_success))
        total_dist += curr_dist
        total_success += is_success
    print("Summary: total dist %f; success %d out of %d" % (total_dist, total_success, n_test_locs))


def padding(*list):
    for x in list:
        x.insert(0, 0)
        x.insert(0, 0)
        x.append(0)

def run_planning_and_inverse_model(env, envname, hp, n_test_locs, test_data, config, test_mode, model, c_model, actor, savepath):
    # Finding threshold
    # TODO: auto collect this & make sure all c_type's work
    replanning = hp["replanning_every"]
    edge_weight = hp["use_edge_weight"]
    using_vae_samples = hp["use_vae_samples"]
    threshold_edge = hp["threshold_edge"]
    threshold_wp = hp["threshold_shortcut"]
    # threshold = 10.5
    n_planning_samples = hp["n_planning_samples"]
    score_type = hp["score_type"]
    # Precompute the graph
    o_samples_npy = graph = node_goal = None
    total_dist = total_success = 0
    for i in range(n_test_locs):
        curr_obs, goal_obs, start_state, goal_state, context = test_data[i]
        env.reset(start_state)
        env.render(config=config)

        # Generate data, Build Graph, Localize goal
        if i == 0 or test_mode == "eval":
            #TODO: do batching to get a larger graph
            if using_vae_samples:
                o_samples_npy = model.inference(context,
                                                n_samples=n_planning_samples,
                                                layer_cond=False).cpu().detach().numpy()
            else:
                o_samples_npy = generate_samples(envname, env, config, n_planning_samples).cpu().detach().numpy()
            # threshold = find_threshold(o_samples_npy, context, c_model, min_thres=0, max_thres=50, n_iters=10)
            graph = build_memory_graph(o_samples_npy, context, c_model, score_type, threshold_edge, edge_weight)
            save_image(from_numpy_to_var(o_samples_npy),
                       os.path.join(savepath, 'plan_samples.png'),
                       nrow=10)
        # Precompute goal node and recon o
        node_goal = localization(goal_obs, o_samples_npy, context, c_model)
        o_goal_pred, _, _, _ = model(goal_obs, context, determ=True)
        # Actor run
        for t in range(hp["T"]):
            o_curr_pred, _, _, _ = model(curr_obs, context, determ=True)
            # Localize and do planning
            if t % replanning == 0:
                node_cur = localization(o_curr_pred, o_samples_npy, context, c_model)
                shortest_path, edge_weights, edge_raw = find_shortest_path(graph, node_cur, node_goal)
                # padding(edge_weights, edge_raw)

            # Next goal img
            j = t % replanning
            next_way_point = find_next_way_point(
                o_curr_pred,
                o_samples_npy[shortest_path][min(j//10, len(shortest_path)-1):],
                o_goal_pred,
                context,
                threshold_wp,
                c_model)
            next_o_goal = next_way_point[None, :]
            # Visualize plans
            if t % 50 == 0:
                img_seq = torch.cat([curr_obs, from_numpy_to_var(o_samples_npy[shortest_path]), goal_obs])
                all_score = get_score(
                    c_model,
                    o_curr_pred.repeat(len(img_seq), 1, 1, 1),
                    img_seq,
                    context.repeat(len(img_seq), 1, 1, 1),
                    type="all"
                )
                img_seq_np = img_seq.detach().cpu().numpy()
                write_number_on_images(img_seq_np,
                                       ["%d, %.3f" % (i, j) for i, j in zip(all_score["raw"], all_score[score_type])],
                                       position="top-left")
                write_number_on_images(img_seq_np,
                                       ["", ""] + ["%d, %.3f" % (i, j) for i, j in zip(edge_raw, edge_weights)] + [""],
                                       position="bottom-left")
                save_image(torch.Tensor(img_seq_np),
                           os.path.join(savepath, 'plan_task_%d_step_%d.png' % (i, t)), nrow=len(shortest_path)+2)
            # Get action
            action = actor(o_curr_pred, next_o_goal, context).cpu().numpy()
            _, _ = env.step_only(action + np.random.randn(2)*hp["action_noise"])
            next_img = env.get_current_img(config)
            curr_obs = get_torch_images_from_numpy(next_img[None, :], False, one_image=True)
        # print("Final State: %s; Goal State: %s" % (env.get_current_obs(), goal_state))
            import matplotlib.pyplot as plt
            if not os.path.exists(os.path.join(savepath, '%d' % i)):
                os.makedirs(os.path.join(savepath, '%d' % i))
            plt.imshow(next_img)
            plt.savefig(os.path.join(savepath, '%d/%03d.png' % (i, t)))
            plt.close()
            curr_dist = np.linalg.norm(env.get_current_obs() - goal_state)
            is_success = (np.abs(env.get_current_obs() - goal_state) < hp["block_size"]).all()
            if is_success:
                break
        print("Final Distance for Task %d: %f; Success %s" % (i, curr_dist, is_success))
        total_dist += curr_dist
        total_success += is_success
    print("Summary: total dist %f; success %d out of %d" % (total_dist, total_success, n_test_locs))
    return total_dist, total_success, n_test_locs


def execute(all_models, envname, savepath, eval_hp=None):
    hp = get_hp(eval_hp)
    ### Inverse Model Evaluation ###
    # Set all models to eval
    for m in all_models:
        set_requires_grad(m, False)
        m.eval()

    # Get models
    model, c_model, actor = all_models

    # Set Env the viewer config
    config = {
            "visible": False,
            "init_width": hp["init_width"],
            "init_height": hp["init_height"],
            "go_fast": True
    }
    env = get_env(envname)
    env.reset()
    env.render(config=config)
    env.viewer_setup()
    for i in range(100):
        env.render(config=config)
    test_mode = hp["test_mode"]
    n_test_locs = get_n_test_locs(envname) if test_mode == "valid" else hp["n_test_locs"]

    # Generate test data
    test_data = []
    for i in range(n_test_locs):
        if test_mode == "valid":
            test_data.append(set_test_loc(i, envname, env, config))
        else:
            test_data.append(set_valid_loc(envname, env, config))

    ### Planning ###
    if hp["run_inverse_model"]:
        run_inverse_model(env, config, test_data, n_test_locs, hp, actor)

    total_dist, total_success, n_test_locs = run_planning_and_inverse_model(env, envname, hp, n_test_locs, test_data, config, test_mode, model, c_model, actor, savepath)
    return total_dist, total_success, n_test_locs



