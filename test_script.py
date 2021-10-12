import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from util.h5_util import load_h5_file
from baselines.baselines_configs import configs
from metrics.mse import mse
from competition.submission.submission import create_patches, stitch_patches

model_path = "trained_models/ckpt_1010_up_r50_epoch_0499.pt"
# "../../../scratch/wnina/ckpt_backups/ckpt_patch_2/epoch_0649.pt"
model_str = "up_patch"
radius = 50
path_data_x = "../../../data/t4c2021/temp_test_data/ANTWERP_train_data_x.h5"
path_data_y = "../../../data/t4c2021/temp_test_data/ANTWERP_train_data_y.h5"


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


strides, samples, mses_of_patches, mses_of_stitched, corr_mses_of_patches, corr_mses_of_stitched = [], [], [], [], [], []
model = load_model(model_path)

for i in range(100):
    for stride in [30, 50, 75, 100]:
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
        # cutoff = inp_patch.size()[0] # divide into two
        # inp_1 = inp_patch[:cutoff]
        # inp_2 = inp_patch[cutoff:]
        # out_1 = model(inp_1)
        # out_2 = model(inp_2)
        # out = torch.cat((out_1, out_2), dim=0)
        # print(inp_1.size(), inp_2.size(), out_1.size(), out.size())

        # post transform
        post_transform = configs[model_str]["post_transform"]
        out_patch = post_transform(out, city="ANTWERP", normalize=True).detach().numpy()

        gt_patches, _, _ = create_patches(y_hour, radius=radius, stride=stride)

        mse_of_patches = mse(out_patch, gt_patches)
        # print("MSE patches", mse_of_patches)

        # stitch back together
        pred = stitch_patches(out_patch, avg_arr, index_arr)

        mse_of_stitched = mse(pred, y_hour)
        # print("MSE stitched", mse_of_stitched)
        print("(MSEs bef correction", mse_of_patches, mse_of_stitched, ")")

        # correct for error
        use_inds = []
        for k, ind_lin in enumerate(index_arr):
            if np.all(ind_lin <= 400):
                use_inds.append(k)
        use_x_max, use_y_max = (int(np.max(index_arr[use_inds, 1])), int(np.max(index_arr[use_inds, 3])))
        corrected_mse_of_patches = mse(out_patch[use_inds], gt_patches[use_inds])
        corrected_mse_of_stitched = mse(pred[:, :use_x_max, :use_y_max], y_hour[:, :use_x_max, :use_y_max])
        print("MSEs patches vs stitched", corrected_mse_of_patches, corrected_mse_of_stitched)

        strides.append(stride)
        samples.append(i)
        mses_of_patches.append(mse_of_patches)
        mses_of_stitched.append(mse_of_stitched)
        corr_mses_of_patches.append(corrected_mse_of_patches)
        corr_mses_of_stitched.append(corrected_mse_of_stitched)

df = pd.DataFrame()
df["sample"] = samples
df["stride"] = strides
df["mse_patches"] = mses_of_patches
df["mse_stitched"] = mses_of_stitched
df["corr_mse_patches"] = corr_mses_of_patches
df["corr_mse_stitched"] = corr_mses_of_stitched
df.to_csv("results_test_script.csv")
