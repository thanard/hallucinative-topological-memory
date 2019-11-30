import networkx as nx
import torch
import numpy as np

from utils import from_numpy_to_var
from models import get_score
from networkx.algorithms.components import is_connected

MIN_SCORE = 0
MAX_SCORE = 1


def build_memory_graph(o_samples_npy, context, c_model, score_type, edge_thresh=0, edge_weight=False):
    """
    :param o_samples_npy: in numpy
    :param c_model:
    :param edge_thresh:
    :param edge_weight:
    :return:
    """
    graph = nx.DiGraph()
    datalen = len(o_samples_npy)

    # add nodes representing each observation in the trajectory and
    # add edges if temporally close together: if |i-j| = 1
    for i in range(datalen):
        graph.add_node(i)

    # add edges to o_i, o_j if R(o_i, o_j) > s_thresh, and
    # i,j are separated by at least deltaT
    bs = min(datalen, 500)
    assert datalen % bs == 0
    batch_len = int(datalen / bs)
    all_pair_scores = []
    raw_scores = []
    with torch.no_grad():
        for i, oi in enumerate(o_samples_npy):
            cscores = []
            rscores = []
            o = from_numpy_to_var(tile(oi, bs))
            for batch in range(batch_len):
                o_next_batch = from_numpy_to_var(o_samples_npy[batch * bs:(batch + 1) * bs])
                scores = get_score(c_model, o, o_next_batch, context.repeat(bs, 1, 1, 1), type="all")
                # Weights
                ys = scores[score_type]
                cscores.append(ys)
                # Raw
                ys = scores["raw"]
                rscores.append(ys)
            cscores = np.concatenate(cscores)
            rscores = np.concatenate(rscores)
            all_pair_scores.append(cscores)
            raw_scores.append(rscores)
    all_pair_scores = np.array(all_pair_scores)
    raw_scores = np.array(raw_scores)
    assert all_pair_scores.shape[0] == all_pair_scores.shape[1] == datalen
    assert raw_scores.shape[0] == raw_scores.shape[1] == datalen

    # Normalizing scores
    global MIN_SCORE, MAX_SCORE
    MIN_SCORE = all_pair_scores.min()
    # all_pair_scores -= MIN_SCORE
    MAX_SCORE = all_pair_scores.max()
    # all_pair_scores /= MAX_SCORE
    # all_pair_scores *= 100
    # all_pair_scores -= 50
    for i in range(datalen):
        for j in range(datalen):
            # if not edge_weight:
            #     if all_pair_scores[i, j] >= edge_thresh:
            #         graph.add_edge(i, j, raw=raw_scores[i, j])
            # else:
            #     graph.add_edge(i, j, weight=all_pair_scores[i, j], raw=raw_scores[i, j])
            if i != j:
                graph.add_edge(i, j, weight=np.exp(raw_scores[i] - raw_scores[i, j]).sum(), raw=raw_scores[i, j])
    print("W edge: min & max", MIN_SCORE, MAX_SCORE)
    print("Raw score: min & max", raw_scores.min(), raw_scores.max())
    return graph


def tile(o, length):
    """
    o: should be a -1 x 3 x 64 x 64 array where -1 = len(traj)
    """
    o = np.tile([np.expand_dims(o, axis=0)], (length, 1, 1, 1)).squeeze()
    return o


def localization(o_cur_pred, o_samples_npy, context, c_model):
    """
    :param o_cur:
    :param o_samples_npy: in numpy
    :param c_model:
    :return:
    """
    # TODO: do batching like build graph
    # Finds the closest node v in the graph G of our exploration sequence to some observation o
    # will first look in local neighborhood of last node, unless we are at the start (start=True)
    datalen = len(o_samples_npy)
    with torch.no_grad():
        rscores = get_score(c_model, o_cur_pred.repeat(datalen, 1, 1, 1),
                            from_numpy_to_var(o_samples_npy),
                            context.repeat(datalen, 1, 1, 1), type="all")["raw"]
    i = np.argmax(rscores)
    return i


def find_shortest_path(graph, i_idx, g_idx):
    shortest_path = nx.dijkstra_path(graph, i_idx, g_idx)
    n = len(shortest_path)
    edges = [graph[shortest_path[i]][shortest_path[i + 1]]['weight'] for i in range(n - 1)]
    rscores = [graph[shortest_path[i]][shortest_path[i + 1]]['raw'] for i in range(n - 1)]
    return shortest_path, edges, rscores


def find_next_way_point(o_cur_pred, path, o_goal_pred, context, shortcut_thresh, c_model):
    path_to_goal = torch.cat([from_numpy_to_var(path), o_goal_pred])
    return path_to_goal[1]


def find_threshold(o_samples_npy, context, c_model, min_thres=0, max_thres=50, n_iters=10):
    graph = build_memory_graph(o_samples_npy, context, c_model, min_thres)
    assert is_connected(graph)
    graph = build_memory_graph(o_samples_npy, context, c_model, max_thres)
    assert not is_connected(graph)
    for count in range(n_iters):
        threshold = (min_thres + max_thres) / 2
        graph = build_memory_graph(o_samples_npy, context, c_model, threshold)
        if is_connected(graph):
            min_thres = threshold
        else:
            max_thres = threshold
    print(min_thres, max_thres)
    return min_thres
