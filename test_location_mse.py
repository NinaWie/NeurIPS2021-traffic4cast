import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import psutil

from util.h5_util import load_h5_file
from baselines.baselines_configs import configs
from metrics.mse import mse
from competition.submission.submission import create_patches, stitch_patches
from util.monitoring import system_status

model_path = "trained_models/ckpt_upp_patch_d100.pt"
model_str = "up_patch"
radius = 50
# Test data must first be created by running python baselines/naive_shifted_stats.py
path_data_x = "../../../data/t4c2021/temp_test_data/ANTWERP_train_data_x.h5"
path_data_y = "../../../data/t4c2021/temp_test_data/ANTWERP_train_data_y.h5"

device = "cuda"


def load_model(path):
    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str].get("model_config", {})
    model = model_class(**model_config, img_len=2 * radius)
    loaded_dict = torch.load(path, map_location=torch.device("cpu"))
    print(loaded_dict["epoch"])
    model.load_state_dict(loaded_dict["model"])
    return model


strides, samples, mses_of_patches, mses_of_stitched, corr_mses_of_patches, corr_mses_of_stitched = [], [], [], [], [], []
model = load_model(model_path)
model = model.to(device)
model.eval()
stride = 30

out_patch_collection = np.zeros((3, 234, 6, 100, 100, 8))
for i in range(3):
    x_hour = load_h5_file(path_data_x, sl=slice(i, i + 1), to_torch=False)[0]
    y_hour = load_h5_file(path_data_x, sl=slice(i, i + 1),
                          to_torch=False)[0, [0, 1, 2, 5, 8, 11]]
    print("loaded data for sample ", i, x_hour.shape, y_hour.shape)
    # make multiple patches
    patch_collection, avg_arr, index_arr = create_patches(
        x_hour, radius=radius, stride=stride
    )

    # pretransform
    pre_transform = configs[model_str]["pre_transform"]
    inp_patch = pre_transform(
        patch_collection, from_numpy=True, batch_dim=True
    )

    # run - batch if it's too big
    internal_batch_size = 50
    n_samples = inp_patch.size()[0]
    img_len = inp_patch.size()[2]
    out = torch.zeros(n_samples, 48, img_len, img_len)
    e_b = 0
    for j in range(n_samples // internal_batch_size):
        s_b = j * internal_batch_size
        e_b = (j + 1) * internal_batch_size
        batch_patch = inp_patch[s_b:e_b].to(device)
        # print(system_status())
        out[s_b:e_b] = model(batch_patch).detach().cpu()
    if n_samples % internal_batch_size != 0:
        last_batch = inp_patch[e_b:].to(device)
        out[e_b:] = model(last_batch).detach().cpu()
    # out = model(inp_patch)

    # post transform
    post_transform = configs[model_str]["post_transform"]
    out_patch = post_transform(out, city="ANTWERP",
                               normalize=True).detach().numpy()

    gt_patches, _, _ = create_patches(y_hour, radius=radius, stride=stride)

    mse_of_patches = mse(out_patch, gt_patches)
    print("MSE patches", mse_of_patches)
    print(out_patch.shape)
    out_patch_collection[i] = out_patch

np.save("out_patches.npy", out_patch_collection)