import torch
import os.path
import numpy as np
from utils import write_number_on_images
from tensorboard_logger import log_value
from torchvision.utils import save_image


def log_info(c_loss, vae_loss, a_loss, model, conditional, cond_info, it, n_batch, epoch):
    print('### Iter %d/%d' % (it, n_batch))
    print("C loss: ", c_loss.cpu().data)
    print("VAE loss: ", vae_loss.cpu().data)
    print("Actor loss: ", a_loss.cpu().data)
    dec_w = model.decoder.W.cpu().detach().numpy()
    print("Decoder weight on O_cond: %.2f +- %.2f (%.2f, %.2f)" % (dec_w.mean(), dec_w.std(), dec_w.min(), dec_w.max()))
    if conditional:
        mu_cond = cond_info["means_cond"].cpu().detach().numpy()
        logvar_cond = cond_info["log_var_cond"].cpu().detach().numpy()
        print("Means of O_cond: %.2f +- %.2f (%.2f, %.2f)" % (
            mu_cond.mean(), mu_cond.std(), mu_cond.min(), mu_cond.max()))
        print("Log vars of O_cond: %.2f +- %.2f (%.2f, %.2f)" % (
            logvar_cond.mean(), logvar_cond.std(), logvar_cond.min(), logvar_cond.max()))
    log_value('c_loss', c_loss, it + n_batch * epoch)
    log_value('vae_loss', vae_loss, it + n_batch * epoch)
    log_value('a_loss', a_loss, it + n_batch * epoch)


def log_images(o, o_pred, o_neg, c, test_context, model, c_model, n_contexts, n_samples_per_c, savepath, epoch):
    assert o.size(0) == n_contexts
    ### Reconstruction
    comparison = torch.cat([o,
                            o_pred,
                            c])
    save_image(comparison.data.cpu(),
               os.path.join(savepath, 'reconstruction_' + str(epoch) + '.png'), nrow=n_contexts)

    ### Sampling
    sample_o = model.inference(c,
                               n_samples=n_samples_per_c,
                               layer_cond=False)
    sorted_mask_o = score_images(sample_o, o_pred, c, c_model, n_samples_per_c, n_contexts)
    # save image
    save_image(torch.Tensor(sorted_mask_o), os.path.join(savepath, 'sample_' + str(epoch) + '.png'), nrow=n_contexts)

    ### Sample from data
    o_neg_pred = model(o_neg.reshape(-1, *o_neg.size()[2:]), c.repeat(n_samples_per_c, 1, 1, 1))[0]
    sorted_mask_o = score_images(o_neg_pred, o_pred, c, c_model, n_samples_per_c, n_contexts)
    # save image
    save_image(torch.Tensor(sorted_mask_o), os.path.join(savepath, 'sample_data_' + str(epoch) + '.png'),
               nrow=n_contexts)

    ### Test contexts
    sample_o = model.inference(test_context[:n_contexts],
                               n_samples=n_samples_per_c)
    save_image(sample_o, os.path.join(savepath, 'sample_test_' + str(epoch) + '.png'), nrow=n_contexts)


def score_images(sample_o, o_pred, c, c_model, n_samples_per_c, n_contexts):
    c_score = c_model(o_pred.repeat(n_samples_per_c, 1, 1, 1),
                      sample_o,
                      c.repeat(n_samples_per_c, 1, 1, 1))
    mask_o = sample_o.detach().cpu().numpy()
    c_score = c_score.detach().cpu().numpy()
    write_number_on_images(mask_o, c_score)
    # sort the scores from max to min
    mask_o = mask_o.reshape(-1, n_contexts, *mask_o.shape[1:])
    inds = c_score.reshape(-1, n_contexts).argsort(0)
    sorted_mask_o = mask_o[inds[::-1].reshape(-1),
                           np.tile(np.arange(n_contexts)[None],
                                   (n_samples_per_c, 1)).reshape(-1)]
    sorted_mask_o = np.concatenate([o_pred.detach().cpu().numpy(), sorted_mask_o.reshape(-1, *mask_o.shape[2:])],
                                   axis=0)
    return sorted_mask_o
