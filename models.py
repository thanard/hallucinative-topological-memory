import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import stochastic_binary_layer, from_numpy_to_var, to_var, detach_var_to_numpy


def get_score(cmodel, x, x_next, o_cond=None, type="exp-neg"):
    score = detach_var_to_numpy(cmodel.forward(x, x_next, o_cond))
    values = {"exp-neg": np.exp(-score),
              # "neg": -score,
              "sig-neg": 1. / (1. + np.exp(score)),  # eqv to 1-sigmoid
              "sig-neg-10": 1. / (1. + np.exp(10 * score)),
              # Don't use below this for planning shortest path.
              "raw": score,
              "sigmoid": 1. / (1. + np.exp(-score))}
    if type == "exp-neg" or type == "raw" or type == "sig-neg" or type == "sig-neg-10":
        return values[type]
    elif type == "all":
        return values
    else:
        raise NotImplementedError


def log_sum_exp(arr):
    max_arr = torch.max(arr, dim=1, keepdim=True)[0]
    return max_arr + torch.log(torch.sum(torch.exp(arr - max_arr), dim=1))


def get_c_loss(model,
               c_model,
               c_type,
               o,
               o_next,
               context,
               N,
               o_neg=None
               ):
    batch_size = o.size(0)
    if c_type[:3] == "cpc":
        # Positive
        positive_log_density = c_model.log_density(o, o_next, context)

        # Negative
        negative_c = context.repeat(N, 1, 1, 1)
        if o_neg is None:
            negative_o_pred = model.inference(negative_c, n_samples=1, layer_cond=False)
        else:
            negative_o_pred = model(o_neg, negative_c)[0]
        negative_log_density = c_model.log_density(o.repeat(N, 1, 1, 1), negative_o_pred, negative_c).view(N,
                                                                                                           batch_size).t()

        # Loss
        density_ratio = torch.cat([from_numpy_to_var(np.zeros((batch_size, 1))),
                                   negative_log_density - positive_log_density[:, None]],
                                  dim=1)
        if c_type == "cpc-sptm":
            density_ratio = torch.cat([density_ratio, -positive_log_density[:, None], negative_log_density], dim=1)
        c_loss = torch.mean(log_sum_exp(density_ratio))
    elif c_type[:4] == "sptm":
        # Positive
        positive_y_pred = c_model(o, o_next, context)
        # Negative
        if o_neg is None:
            negative_o_pred = model.inference(context, n_samples=1, layer_cond=False)
        else:
            negative_o_pred = model(o_neg, context)[0]
        negative_y_pred = c_model(o, negative_o_pred, context)
        ys = torch.cat([positive_y_pred, negative_y_pred])
        labels = torch.cat([torch.ones(batch_size),
                            torch.zeros(batch_size)])
        if torch.cuda.is_available():
            labels = labels.cuda()
        c_loss = c_model.loss(ys, labels)
    else:
        raise NotImplementedError
    assert not torch.isnan(c_loss)
    return c_loss


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, *size):
        super(UnFlatten, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(-1, *self.size)


class Actor(nn.Module):
    def __init__(self,
                 e_arch,
                 z_dim,
                 a_dim,
                 shared_encoder,
                 conditional,
                 freeze_enc,
                 img_size=(64, 64)):
        super(Actor, self).__init__()
        self.l2loss = nn.MSELoss()
        if not freeze_enc:
            self.encoder = type(shared_encoder)(e_arch,
                                                z_dim,
                                                conditional,
                                                3,
                                                img_size)
            self.encoder.load_state_dict(shared_encoder.state_dict())
        else:
            self.encoder = shared_encoder
            self.encoder.eval()

        self.f = nn.Sequential(
            nn.Linear(z_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(.2, inplace=True),

            nn.Linear(512, a_dim),
            nn.Tanh()
        )

    def encode(self, x, o_cond=None):
        return self.encoder(x, o_cond)[0]

    def forward(self, x, x_next, o_cond=None):
        z = self.encode(x, o_cond)
        z_next = self.encode(x_next, o_cond)
        f_out = self.f(torch.cat([z, z_next], dim=1))
        return f_out.squeeze() / 100

    def loss(self, act_true, x, x_next, o_cond):
        return self.l2loss(act_true, self.forward(x, x_next, o_cond)) * 1000


class Classifier(nn.Module):
    def __init__(self,
                 c_type,
                 c_arch,
                 e_arch,
                 z_dim,
                 shared_encoder,
                 conditional,
                 freeze_enc,
                 img_size=(64, 64)):
        super(Classifier, self).__init__()
        if not freeze_enc:
            self.encoder = type(shared_encoder)(e_arch,
                                                z_dim,
                                                conditional,
                                                3,
                                                img_size)
            self.encoder.load_state_dict(shared_encoder.state_dict())
        else:
            self.encoder = shared_encoder
            self.encoder.eval()
        self.c_type = c_type
        self.bceloss = nn.BCELoss()
        if c_type == "sptm":
            self.f = nn.Sequential(
                nn.Linear(2 * z_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                # nn.Sigmoid()
            )
        elif c_type == "sptm-w":
            self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        elif c_type == "sptm-large":
            self.f = nn.Sequential(
                nn.Linear(z_dim * 2, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(.2, inplace=True),

                # nn.Linear(64, 128),
                # nn.BatchNorm1d(128),
                # nn.LeakyReLU(.2, inplace=True),

                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(.2, inplace=True),

                nn.Linear(256, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(.2, inplace=True),

                nn.Linear(512, 1),
                # nn.Sigmoid()
            )
        else:
            raise NotImplementedError

    def encode(self, x, o_cond=None):
        return self.encoder(x, o_cond)[0]

    def forward(self, x, x_next, o_cond=None):
        """
        :param x: img
        :param x_next: next img
        :param o_cond: conditioned context
        :return: score
        """
        z = self.encode(x, o_cond)
        z_next = self.encode(x_next, o_cond)
        if self.c_type == "sptm" or self.c_type == "sptm-large":
            f_out = self.f(torch.cat([z, z_next], dim=1))
        elif self.c_type == "sptm-w":
            w = self.W
            z = z.unsqueeze(2)
            z_next = z_next.unsqueeze(2)
            w = w.repeat(z.size(0), 1, 1)
            f_out = torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z) / 1000
        else:
            raise NotImplementedError
        return f_out.squeeze()

    def loss(self, ys, labels):
        return self.bceloss(F.sigmoid(ys), labels)


class CPC(nn.Module):
    def __init__(self,
                 c_type,
                 c_arch,
                 e_arch,
                 z_dim,
                 shared_encoder,
                 conditional,
                 freeze_enc,
                 n_conditional_layers=3,
                 img_size=(64, 64)):
        super(CPC, self).__init__()
        if not freeze_enc:
            self.encoder = type(shared_encoder)(e_arch,
                                                z_dim,
                                                conditional,
                                                n_conditional_layers,
                                                img_size)
            self.encoder.load_state_dict(shared_encoder.state_dict())
        else:
            self.encoder = shared_encoder
            self.encoder.eval()
        self.c_type = c_type
        if c_type == "cpc-w-cond-rank1":
            self.W = Encoder(c_arch, z_dim, False, 0, img_size)
        elif c_type == "cpc-w-cond-full":
            self.W = Encoder(c_arch, z_dim * z_dim, False, 0, img_size)
        elif c_type == "cpc-w-ff":
            self.W = Encoder(c_arch, z_dim, False, 0, img_size)
            self.f = nn.Sequential(
                nn.Linear(3 * z_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        elif c_type == "cpc" or c_type == "cpc-sptm":
            self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        else:
            raise NotImplementedError
        # TODO: try rank 1 approximation or other architectures
        self.z_dim = z_dim

    def encode(self, x, o_cond=None):
        return self.encoder(x, o_cond)

    def log_density(self, x, x_next, o_cond=None):
        # Same as density
        assert x_next.size(0) == x.size(0)
        z, _ = self.encode(x, o_cond)
        z_next, _ = self.encode(x_next, o_cond)
        z = z.unsqueeze(2)  # bs x z_dim x 1
        z_next = z_next.unsqueeze(2)
        if self.c_type == "cpc-w-cond-rank1":
            assert o_cond is not None
            w1, w2 = self.W(o_cond)
            f1 = torch.bmm(z_next.permute(0, 2, 1), w1[:, :, None])
            f2 = torch.bmm(w2[:, None, :], z)
            f_out = (f1 * f2).squeeze()
        elif self.c_type == "cpc-w-cond-full":
            w = self.W(o_cond)[0].view(z.size(0), self.z_dim, self.z_dim)
            f_out = torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z)
            f_out = f_out.squeeze()
        elif self.c_type == "cpc" or self.c_type == "cpc-sptm":
            w = self.W
            w = w.repeat(z.size(0), 1, 1)
            f_out = torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z)
            f_out = f_out.squeeze()
        elif self.c_type == "cpc-w-ff":
            inp_list = [z[:, :, 0], z_next[:, :, 0]]
            if o_cond is not None:
                context_emb, _ = self.W(o_cond)
                inp_list.append(context_emb)
            f_out = self.f(torch.cat(inp_list, dim=1)).squeeze()
        else:
            raise NotImplementedError
        return f_out / 1000

    def forward(self, *input):
        return self.log_density(*input)


def loss_function(recon_x, x, mu, logvar,
                  mu_cond=None, logvar_cond=None, beta=1):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 64 * 64), x.contiguous().view(-1, 64 * 64), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if mu_cond is not None and logvar_cond is not None:
        KLD = -0.5 * torch.sum(1 - (mu - mu_cond).pow(2) / logvar_cond.exp() -
                               (logvar.exp() / logvar_cond.exp()) + logvar - logvar_cond)
    return (BCE + beta * KLD) / x.size(0)


def variational_lower_bound(recon_x, x, mu, logvar):
    input = recon_x.view(-1, 3 * 64 * 64)
    target = x.view(-1, 3 * 64 * 64)
    BCE = target * torch.log(input) + (1 - target) * torch.log(1 - input)
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return BCE.sum(1), KLD.sum(1)


class Filling(nn.Module):
    """
    Output num_labels x img_size x img_size
    """

    def __init__(self, num_labels, img_size):
        self.num_labels = num_labels
        self.img_size = img_size
        super(Filling, self).__init__()

    def forward(self, c):
        return c.repeat(1, self.img_size[0] * self.img_size[1]).view(-1, self.num_labels, *self.img_size)


class Encoder(nn.Module):

    def __init__(self, etype,
                 latent_size,
                 conditional,
                 n_conditional_layers,
                 img_size):
        layers = etype.split("-")
        self.arch_type = layers[0]
        layer_sizes = [int(x) for x in layers[1:]]
        super(Encoder, self).__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += n_conditional_layers
        self.preprocess_x = nn.Sequential()
        if self.arch_type == "mlp":
            """
            MLP
                Example: mlp-784-256
            """
            self.preprocess_x.add_module(name="Flatten", module=Flatten())
            self.main = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                self.main.add_module(name="L%i" % i, module=nn.Linear(in_size, out_size))
                self.main.add_module(name="A%i" % i, module=nn.ReLU())

            self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
            self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)
        elif self.arch_type == "cnn":
            """
            CNN
                Example: cnn-3-32-64, img_size 32x32
                3x32x32 --> 32x16x16 ---> 64x8x8
            """
            self.main = nn.Sequential(nn.BatchNorm2d(layer_sizes[0]))
            self.z_dim = img_size[0] // 2 ** (len(layer_sizes) - 1)  # Assume square img downsample x times.
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                self.main.add_module(name="Conv2d%i" % i,
                                     module=nn.Conv2d(in_size, out_size, 4, 2, 1))
                self.main.add_module(name="BatchNorm%i" % i,
                                     module=nn.BatchNorm2d(out_size))
                self.main.add_module(name="LeakyReLU%i" % i,
                                     module=nn.LeakyReLU(0.1, inplace=True))
            self.main.add_module(name="Flatten", module=Flatten())
            self.linear_means = nn.Linear(layer_sizes[-1] * self.z_dim ** 2, latent_size)
            self.linear_log_var = nn.Linear(layer_sizes[-1] * self.z_dim ** 2, latent_size)
        else:
            raise NotImplementedError
        print("\nEncoder: ", self.main, "\n", self.linear_means)

    def forward(self, x, o_cond=None):
        if o_cond is not None:
            assert self.conditional
            x = torch.cat((x, o_cond), dim=1)
        x = self.preprocess_x(x)
        x = self.main(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self,
                 dtype,
                 latent_size,
                 conditional,
                 n_conditional_layers,
                 img_size):
        layers = dtype.split("-")
        self.arch_type = layers[0]
        layer_sizes = [int(x) for x in layers[1:]]
        super(Decoder, self).__init__()

        self.n_conditional_layers = n_conditional_layers
        self.conditional = conditional
        input_size = latent_size

        if self.arch_type == "mlp":
            """
            MLP
                Example: mlp-784-256
            """
            self.main = nn.Sequential()
            self.z_develop = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
                self.main.add_module(name="L%i" % (i), module=nn.Linear(in_size, out_size))
                if i + 1 < len(layer_sizes):
                    self.main.add_module(name="A%i" % (i), module=nn.ReLU())
                else:
                    self.main.add_module(name="sigmoid", module=nn.Sigmoid())
        elif self.arch_type == "cnn":
            """
            CNN
                Example: cnn-64-32-3, img_size 32x32
                64x8x8 --> 32x16x16 ---> 3x32x32
            """
            self.main = nn.Sequential()
            self.z_dim = img_size[0] // 2 ** (len(layer_sizes) - 1)  # Assume square img downsample x times.
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                if i != 0:
                    self.main.add_module(name="BatchNorm%i" % i,
                                         module=nn.BatchNorm2d(in_size))
                    self.main.add_module(name="ReLU%i" % i,
                                         module=nn.ReLU(inplace=True))
                self.main.add_module(name="ConvTranspose2d%i" % i,
                                     module=nn.ConvTranspose2d(in_size, out_size, 4, 2, 1))
            self.z_develop = nn.Sequential(
                nn.Linear(input_size, layer_sizes[0] * self.z_dim ** 2),
                UnFlatten(layer_sizes[0], self.z_dim, self.z_dim)
            )
            self.W = nn.Parameter(torch.ones(1, n_conditional_layers, *img_size))
        else:
            raise NotImplementedError
        print("\nDecoder: ", self.z_develop, "\n", self.main)

    def forward(self, z, o_cond=None):
        """
        :param z: batch_size x latent size
        :param o_cond: batch size x C x H x W
        :return: img: batch size x C x H x W
        """
        if self.z_develop:
            z = self.z_develop(z)
        x = self.main(z)
        # TODO: use o_cond to make the cvae more correct.
        x = F.sigmoid(x)
        return x


class VAE(nn.Module):

    def __init__(self,
                 etype,
                 dtype,
                 latent_size,
                 conditional=False,
                 n_conditional_layers=3,
                 img_size=(64, 64)):

        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.W = nn.Parameter(torch.rand(latent_size, latent_size))
        self.encoder = Encoder(etype, latent_size, conditional, n_conditional_layers, img_size)
        self.decoder = Decoder(dtype, latent_size, conditional, n_conditional_layers, img_size)
        self.prior = Encoder(etype, latent_size, False, 0, img_size)

    def encode(self, x, o_cond=None):
        kwargs = {}
        batch_size = x.size(0)
        means, log_var = self.encoder(x, o_cond)
        if o_cond is not None:
            means_cond, log_var_cond = self.prior(o_cond)
            kwargs['means_cond'] = means_cond
            kwargs['log_var_cond'] = log_var_cond
        std = torch.exp(0.5 * log_var)
        eps = to_var(torch.randn([batch_size, self.latent_size]))
        z = eps * std + means
        return z, means, log_var, kwargs

    def decode(self, z, o_cond):
        return self.decode(z, o_cond)

    def forward(self, x, o_cond=None, determ=False):
        # Sample z from q_phi(z|x, c), then compute the mean of p_theta(x|z, c)
        # Return generated x, mu_x(z),
        z, means, log_var, kwargs = self.encode(x, o_cond)
        if determ:
            recon_x = self.decoder(means, o_cond)
        else:
            recon_x = self.decoder(z, o_cond)
        return recon_x, means, log_var, kwargs

    def inference(self, o_cond=None, n_samples=1, layer_cond=True):
        """
        :param o_cond: if we want to condition on real images  n x C x H x W
        :param n_samples: the number of z samples per o_cond
        :return: sample from learned P_theta
        """
        batch_size = n_samples if o_cond is None else n_samples * o_cond.size(0)
        z = to_var(torch.randn([1, n_samples, self.latent_size])).repeat(batch_size // n_samples, 1, 1) \
            .permute(1, 0, 2).reshape(batch_size, -1)
        o_cond_rep = None
        if o_cond is not None:
            o_cond_rep = o_cond.repeat(n_samples, 1, 1, 1)
            mu_cond, logvar_cond = self.prior(o_cond_rep)
            std_cond = torch.exp(0.5 * logvar_cond)
            eps = to_var(torch.randn([batch_size, self.latent_size]))
            z = eps * std_cond + mu_cond
        # Now both z and o_cond has size 0 = batch_size
        recon_x = self.decoder(z, o_cond_rep)
        if o_cond is not None and layer_cond:
            recon_x = torch.cat([o_cond, recon_x])
        return recon_x

    def log_density(self, x_next, z, o_cond=None, img_input=False):
        # Same as density
        assert x_next.size(0) == z.size(0)
        if img_input:
            _, z, _, _ = self.encode(z, o_cond)
        _, z_next, _, _ = self.encode(x_next, o_cond)
        z_next = z_next.unsqueeze(2)
        z = z.unsqueeze(2)
        w = self.W.repeat(z.size(0), 1, 1)
        f_out = torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z)
        f_out = f_out.squeeze()
        return f_out
