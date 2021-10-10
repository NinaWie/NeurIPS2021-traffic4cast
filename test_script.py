import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import glob
import time

from util.check_data_raw_dir import check_raw_data_dir
from util.data_range import generate_date_range
from util.h5_util import load_h5_file
from baselines.baselines_unet_separate import load_torch_model_from_checkpoint
from baselines.baselines_configs import configs
from metrics.mse import mse
from competition.submission.submission import create_patches, stitch_patches

model_path = "../../../scratch/wnina/ckpt_backups/ckpt_patch_2/epoch_0649.pt"
model_str = "up_patch"
radius = 50
path_data_x = "../../../scratch/wnina/temp_test_data/ANTWERP_train_data_x.h5"
path_data_y = "../../../scratch/wnina/temp_test_data/ANTWERP_train_data_y.h5"


results = defaultdict(list)


def load_model(path):
    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str].get("model_config", {})
    model = model_class(**model_config, img_len=2 * radius)
    loaded_dict = torch.load(path, map_location=torch.device("cpu"))
    print(loaded_dict["epoch"])
    model.load_state_dict(loaded_dict["model"])
    return model


test_data_x = load_h5_file(path_data_x)
test_data_y = load_h5_file(path_data_y)
print(test_data_x.shape, test_data_y.shape)


model = load_model(model_path)

for i in range(100):
    for stride in [30, 50, 100]:
        print(i, "stride", stride)
        x_hour = test_data_x[i]
        y_hour = test_data_y[i]
        # make multiple patches
        patch_collection, avg_arr, index_arr = create_patches(x_hour, radius=radius, stride=stride)

        # pretransform
        pre_transform = configs[model_str]["pre_transform"]
        inp_patch = pre_transform(patch_collection, from_numpy=True, batch_dim=True)

        # run
        out = model(inp_patch)

        # post transform
        post_transform = configs[model_str]["post_transform"]
        out_patch = post_transform(out, city="ANTWERP", normalize=True).detach().numpy()

        gt_patches, _, _ = create_patches(y_hour, radius=radius, stride=stride)

        mse_of_patches = mse(out_patch, gt_patches)
        print("MSE patches", mse_of_patches)

        # stitch back together
        pred = stitch_patches(out_patch, avg_arr, index_arr)

        mse_of_stitched = mse(pred, y_hour)
        print("MSE stitched", mse_of_stitched)
        results["stride"].append(stride)
        results["sample"].append(i)
        results["mse_patches"].append(mse_of_patches)
        results["mse_stitched"].append(mse_of_stitched)


with open(os.path.join("results_test_script.json"), "w") as outfile:
    json.dump(results, outfile)
