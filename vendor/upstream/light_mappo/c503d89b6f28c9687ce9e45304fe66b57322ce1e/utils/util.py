import math

import numpy as np
import torch


def check(input):
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    return input


def get_gard_norm(it):
    sum_grad = 0.0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


def mse_loss(e):
    return e ** 2 / 2


def get_shape_from_obs_space(obs_space):
    cls_name = obs_space.__class__.__name__
    if cls_name == "Box":
        return obs_space.shape
    if cls_name == "list":
        return obs_space
    raise NotImplementedError


def get_shape_from_act_space(act_space):
    cls_name = act_space.__class__.__name__
    if cls_name == "Discrete":
        return 1
    if cls_name == "MultiDiscrete":
        return act_space.shape
    if cls_name in ("Box", "MultiBinary"):
        return act_space.shape[0]
    # agar
    return act_space[0].shape[0] + 1


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    padded = np.zeros((H * W, h, w, c), dtype=img_nhwc.dtype)
    padded[:N] = img_nhwc
    return padded.reshape(H, W, h, w, c).transpose(0, 2, 1, 3, 4).reshape(H * h, W * w, c)