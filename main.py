import os
import argparse
import json
import torch
import numpy as np
import torch.optim as optim

from models import VAE, CPC, Classifier, Actor
from utils import set_requires_grad
from trainer import train

parser = argparse.ArgumentParser()
parser.add_argument("--savepath",
                    help="output path for local run.",
                    type=str,
                    default='out/')
parser.add_argument("--loadpath_v",
                    help="load vae parameters from path.",
                    type=str,
                    default='')
parser.add_argument("--loadpath_c",
                    help="load cpc parameters from path.",
                    type=str,
                    default='')
parser.add_argument("--loadpath_a",
                    help="load actor parameters from path.",
                    type=str,
                    default='')
parser.add_argument("--data_dir", type=str,
                    default="data/randact_traj_length_20_n_trials_50_n_contexts_150.npy")
parser.add_argument("--test_dir", type=str,
                    default="data/test_context_20.npy")
parser.add_argument("--prefix", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n", type=int, default=1)

# Training Hyperparameters
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--e_arch", type=str, default="cnn-3-32-64-128-256-512")
parser.add_argument("--d_arch", type=str, default="cnn-512-256-128-64-32-3")
parser.add_argument("--c_arch", type=str, default="cnn-3-32-64-128-256")
parser.add_argument("--c_type", type=str, default="cpc",
                    help="Choose different types of energy model from: "
                         "cpc-w-cond-rank1, "
                         "cpc-w-cond-full, "
                         "cpc-w-ff, "
                         "cpc.")
parser.add_argument("--z_type", type=str, default="continuous",
                    help="continuous/binary/onehot")
parser.add_argument("--z_dim", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--N", type=int, default=50,
                    help="The number of (-) samples per (+) sample")
parser.add_argument("--vae_w", type=float, default=0.001)
parser.add_argument("--vae_b", type=int, default=10,
                    help="Beta weight term in vae loss")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--k", type=int, default=1,
                    help="The number of steps apart")
parser.add_argument("--pretrain", type=int, default=0,
                    help="The number of epochs to pretrain the vae")
parser.add_argument("-conditional", action="store_true")
parser.add_argument("-freeze_enc", action="store_true",
                    help="Use the same weights as the v enc in c enc and a enc.")
parser.add_argument("-use_o_neg", action="store_true",
                    help="Use real negative samples rather than random sampling from the VAE.")

# Planning & Inverse Model
parser.add_argument("--mode", type=str, default="",
                    help="Choose what models will be trained. "
                         "Choices are v (vae), v-c (vae + cpc/sptm), c (cpc/sptm), a (act), e (eval)")
parser.add_argument("--env", type=str, default="block",
                    help="block")
args = parser.parse_args()

if args.prefix is None:
    str_list = [args.mode,
                args.c_type,
                "z-dim", "%d" % args.z_dim,
                "batch-size", "%d" % args.batch_size,
                "N", "%d" % args.N,
                "vae-w", "%.2f" % args.vae_w,
                "vae-b", "%d" % args.vae_b,
                "k", "%d" % args.k,
                "pretrain", "%d" % args.pretrain,
                ]
    if len(args.loadpath_v + args.loadpath_c + args.loadpath_a) > 0:
        str_list.append("load")
    if len(args.loadpath_v) > 0:
        str_list.append("v")
    if len(args.loadpath_c) > 0:
        str_list.append("c")
    if len(args.loadpath_a) > 0:
        str_list.append("a")
    if args.freeze_enc:
        str_list.append("freeze-enc")
    if args.use_o_neg:
        str_list.append("real-o-neg")
    args.prefix = "-".join(str_list)
    print("Experiment name : ", args.prefix)

kwargs = vars(args)
kwargs['eval_hp'] = {
    'run_inverse_model': False,
    'test_mode': "valid",
    'score_type': "exp-neg",  # exp-neg, sig-neg
    'init_width': 64,
    'init_height': 64
}
kwargs['savepath'] = os.path.join(args.savepath, args.prefix)
savepath = kwargs['savepath']

# Save kwargs
with open('%s/params.json' % savepath, 'w') as fp:
    json.dump(kwargs, fp, indent=4, sort_keys=True)

# Set seed
seed = kwargs["seed"]
torch.manual_seed(seed)
np.random.seed(seed)

### Create VAE, C, actor models ###
e_arch = kwargs["e_arch"]
d_arch = kwargs["d_arch"]
c_arch = kwargs["c_arch"]
c_type = kwargs["c_type"]
z_dim = kwargs["z_dim"]
conditional = kwargs["conditional"]
freeze_enc = kwargs["freeze_enc"]
# VAE model
model = VAE(e_arch, d_arch, z_dim, conditional)
# C model
if c_type[:3] == "cpc":
    log_every = 100
    if c_type == "cpc-sptm":
        log_every = 100
    c_model = CPC(c_type, c_arch, e_arch, z_dim, model.encoder, conditional, freeze_enc)
elif c_type[:4] == "sptm":
    log_every = 100
    c_model = Classifier(c_type, c_arch, e_arch, z_dim, model.encoder, conditional, freeze_enc)
else:
    raise NotImplementedError
# Env & Actor
a_dim = 2
actor = Actor(e_arch, z_dim, a_dim, model.encoder, conditional, freeze_enc)

### Load models ###
loadpath = {"v": kwargs["loadpath_v"],
            "a": kwargs["loadpath_a"],
            "c": kwargs["loadpath_c"]
            }
get_models = {'v': model, 'c': c_model, 'a': actor}
for name in ['v', 'a', 'c']:
    if len(loadpath[name]) > 0:
        get_models[name].load_state_dict(torch.load(loadpath[name]))

### Define training and test models ###
mode = kwargs["mode"]
all_models = [model, c_model, actor]
if torch.cuda.is_available():
    for m in all_models:
        m.cuda()
if mode == "v":
    training_models = [model]
elif mode == "v-c":
    training_models = [model, c_model]
elif mode == "c":
    training_models = [c_model]
elif mode == "c-a":
    training_models = [c_model, actor]
elif mode == "a":
    training_models = [actor]
elif mode == "e":
    training_models = []
else:
    raise NotImplementedError

### Define Parameters ###
training_params = []
for m in all_models:
    if m not in training_models:
        set_requires_grad(m, False)
        m.eval()
    else:
        training_params += list(m.parameters())
solver = None
if len(training_params) > 0:
    solver = optim.Adam(training_params, lr=1e-3)

### Visual Planning & Acting ###
if mode == "e":
    from eval import execute

    execute(all_models, kwargs["env"], savepath, eval_hp=kwargs["eval_hp"])
else:
    train(all_models, training_models, solver, training_params, log_every, **kwargs)
