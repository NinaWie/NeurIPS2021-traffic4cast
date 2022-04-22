import torch
import os
import numpy as np
import time
import argparse

from util.h5_util import load_h5_file
from baselines.baselines_configs import configs
from metrics.mse import mse
from competition.submission.submission import create_patches, stitch_patches


def load_model(path):
    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str].get("model_config", {})
    model = model_class(**model_config, img_len=2 * radius)
    loaded_dict = torch.load(path, map_location=torch.device("cpu"))
    print(loaded_dict["epoch"])
    model.load_state_dict(loaded_dict["model"])
    return model


def get_std(out_patch, means, nr_samples_arr, index_arr):
    """
    out_patch: collection of patches
    nr_samples_arr: Number of samples available per cell
    index_arr: indices of start and end of each patch
    """
    xlen, ylen = avg_arr.shape

    std_prediction = np.zeros((6, xlen, ylen, 8))
    assert len(out_patch) == len(index_arr)

    for i in range(len(out_patch)):
        (start_x, end_x, start_y,
         end_y) = tuple(index_arr[i].astype(int).tolist())
        std_prediction[:, start_x:end_x, start_y:end_y] += (
            out_patch[i] - means[:, start_x:end_x, start_y:end_y]
        )**2

    expand_nr_samples_arr = np.tile(np.expand_dims(nr_samples_arr, 2), 8)

    avg_prediction = np.sqrt(std_prediction / expand_nr_samples_arr)
    return avg_prediction


def std_v1(out_patch, index_arr, out_shape):
    std_preds = np.zeros(out_shape)

    for p_x in range(495):
        for p_y in range(436):
            pixel = (p_x, p_y)
            # find the patches corresponding to this pixel and get the
            # middleness and pred per patch
            preds = []
            for j, inds in enumerate(index_arr):
                x_s, x_e, y_s, y_e = inds
                if x_s <= pixel[0] and x_e > pixel[0] and y_s <= pixel[
                    1] and y_e > pixel[1]:
                    rel_x, rel_y = int(pixel[0] - x_s), int(pixel[1] - y_s)
                    # what values were predicted for this pixel?
                    pred_pixel = out_patch[j, :, rel_x, rel_y, :]
                    # how much in the middle is a pixel?
                    preds.append(pred_pixel)
            std_preds[:, pixel[0], pixel[1], :] = np.std(preds, axis=0)
    return std_preds


parser = argparse.ArgumentParser()
parser.add_argument(
    '-m',
    "--model_path",
    type=str,
    default="trained_models/ckpt_upp_patch_d100.pt"
)
parser.add_argument('-r', "--radius", type=int, default=50)
parser.add_argument('-t', "--model_type", type=str, default="up_patch")
parser.add_argument('-d', '--data_path', type=str, required=True)
parser.add_argument('-o', '--out_path', type=str, default="output_std")
parser.add_argument('-s', '--stride', type=int, default=30)
parser.add_argument('-g', '--gpu', type=str, default="cuda")
args = parser.parse_args()

model_path = args.model_path
model_str = args.model_type
radius = args.radius
stride = args.stride
# Test data must first be created by running python baselines/naive_shifted_stats.py
path_data_x = args.data_path
# "data/temp_test_data/ANTWERP_train_data_x.h5"
# "../../../data/t4c2021/tests_specialprize/ISTANBUL/ISTANBUL_test_specialprize.h5"
# "../../../data/t4c2021/temp_test_data/ANTWERP_train_data_x.h5"
# path_data_y = "../../../data/t4c2021/temp_test_data/ANTWERP_train_data_y.h5"
device = args.gpu

os.makedirs(args.out_path, exist_ok=True)

# load model
model = load_model(model_path)
model = model.to(device)
model.eval()

# load data ones to get size
all_test_data = load_h5_file(path_data_x, to_torch=False)
print("Shape of test data", all_test_data.shape)
data_len = len(all_test_data)
all_test_data = None  # save space

samples, mse_bl_list, mse_weighted_list, mse_middle_list = [], [], [], []
out_std = np.zeros((data_len, 6, 495, 436, 8))
out_mean = np.zeros((data_len, 6, 495, 436, 8))

for i in range(data_len):
    x_hour = load_h5_file(path_data_x, sl=slice(i, i + 1), to_torch=False)[0]
    print("loaded data for sample ", i, x_hour.shape)
    tic = time.time()
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
    out_patch = post_transform(out, normalize=True).detach().numpy()

    # baseline: simply stitch with mean
    pred = stitch_patches(out_patch, avg_arr, index_arr)

    # stitch for std
    std_preds = get_std(out_patch, pred, avg_arr, index_arr)

    # save results
    out_mean[i] = pred
    out_std[i] = std_preds
    print(time.time() - tic)

np.save(os.path.join(args.out_path, "mean.npy"), out_mean)
np.save(os.path.join(args.out_path, "std.npy"), out_std)
