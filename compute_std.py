import torch
import os
import json
import numpy as np
import time
import argparse
import pandas as pd
from scipy.stats import pearsonr

from util.h5_util import load_h5_file
from baselines.baselines_configs import configs
from metrics.mse import mse
from competition.submission.submission import create_patches, stitch_patches
from competition.prepare_test_data.prepare_test_data import prepare_test


def load_model(path, static_map=False):
    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str].get("model_config", {})
    if static_map:
        model_config["in_channels"] += 9
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
        (start_x, end_x, start_y, end_y) = tuple(index_arr[i].astype(int).tolist())
        std_prediction[:, start_x:end_x, start_y:end_y] += (out_patch[i] - means[:, start_x:end_x, start_y:end_y]) ** 2

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
                if x_s <= pixel[0] and x_e > pixel[0] and y_s <= pixel[1] and y_e > pixel[1]:
                    rel_x, rel_y = int(pixel[0] - x_s), int(pixel[1] - y_s)
                    # what values were predicted for this pixel?
                    pred_pixel = out_patch[j, :, rel_x, rel_y, :]
                    # how much in the middle is a pixel?
                    preds.append(pred_pixel)
            std_preds[:, pixel[0], pixel[1], :] = np.std(preds, axis=0)
    return std_preds


def correlation(err_arr, std_arr):
    r, p = pearsonr(err_arr.flatten(), std_arr.flatten())
    return r


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path", type=str, default="trained_models/ckpt_upp_patch_d100.pt")
parser.add_argument("-r", "--radius", type=int, default=50)
parser.add_argument("-t", "--model_type", type=str, default="up_patch")
parser.add_argument("-d", "--data_path", type=str, default="data/raw")
# sample from the 2020 data from one city
parser.add_argument("-c", "--city", type=str, default="ANTWERP")
# based on the metainfo of another city
parser.add_argument("-a", "--metacity", type=str, default="BERLIN")
parser.add_argument("-o", "--out_path", type=str, default="output_std")
parser.add_argument("-s", "--stride", type=int, default=30)
parser.add_argument("-g", "--gpu", type=str, default="cuda")
parser.add_argument("--static_map", action="store_true", default=False, help="Use static map?")
args = parser.parse_args()

model_path = args.model_path
model_str = args.model_type
radius = args.radius
data_path = args.data_path
metacity = args.metacity
# # V1 To run with test data that is already created by running python baselines/naive_shifted_stats.py
# "data/temp_test_data"
# "../../../data/t4c2021/temp_test_data"
# path_data_x = os.path.join(args.data_path, "data_x.h5")
# path_data_y = os.path.join(args.data_path, "data_y.h5")

device = args.gpu

# set random seed to make this test dataset reproducible
np.random.seed(42)

os.makedirs(args.out_path, exist_ok=True)

# load model
model = load_model(model_path, static_map=args.static_map)
model = model.to(device)
model.eval()

# # V1 load data ones to get size
# all_test_data = load_h5_file(path_data_x, to_torch=False)
# print("Shape of test data", all_test_data.shape)
# data_len = len(all_test_data)
# all_test_data = None  # save space

samples, mse_bl_list, mse_weighted_list, mse_middle_list = [], [], [], []

# spatial output --> keep time and channels for analysis, but average over samples
out_std = np.zeros((6, 495, 436, 8))  # save avg std per cell
out_err = np.zeros((6, 495, 436, 8))  # save avg err per cell

# save all corelation results in a df
final_df = []

# load weekday info
with open(os.path.join(data_path, "weekday2dates_2020.json"), "r") as infile:
    weekday2date = json.load(infile)
# load additional data from some other city, e.g. berlin
metainfo = load_h5_file(os.path.join(data_path, metacity, f"{metacity}_test_additional_temporal.h5"))
data_len = len(metainfo)

# load static map:
if args.static_map:
    static_map = load_h5_file(os.path.join(data_path, args.city, f"{args.city}_static.h5"))

for i in range(data_len):
    # get sample based on the i-th sample of the metainfo file
    day = metainfo[i, 0]
    timepoint = int(metainfo[i, 1])
    possible_dates = weekday2date[str(day)]
    use_date = np.random.choice(possible_dates)
    print("weekday", day, "date", use_date, "time", timepoint)
    print("load file for one day ...", f"{data_path}/{args.city}/training/{use_date}_{args.city}_8ch.h5")
    two_hours = load_h5_file(f"{data_path}/{args.city}/training/{use_date}_{args.city}_8ch.h5", sl=slice(timepoint, timepoint + 24))
    x_hour, y_hour = prepare_test(two_hours)
    print(x_hour.shape, y_hour.shape)

    # # V1: use slice of one file
    # x_hour = load_h5_file(path_data_x, sl=slice(i, i + 1), to_torch=False)[0]
    # y_hour = load_h5_file(path_data_y, sl=slice(i, i + 1), to_torch=False)[0]
    print("loaded data for sample ", i, x_hour.shape, y_hour.shape)
    tic = time.time()
    # make multiple patches
    if args.static_map:
        patch_collection, avg_arr, index_arr, data_static = create_patches(x_hour, radius=radius, stride=args.stride, static_map=static_map)
    else:
        patch_collection, avg_arr, index_arr = create_patches(x_hour, radius=radius, stride=args.stride)

    # pretransform
    pre_transform = configs[model_str]["pre_transform"]
    inp_patch = pre_transform(patch_collection, from_numpy=True, batch_dim=True)

    if args.static_map:
        data_static = pre_transform(np.expand_dims(data_static, axis=-1), from_numpy=True, batch_dim=True)
        inp_patch = torch.cat((inp_patch, data_static), dim=1)

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

    time_predict_with_uncertainty = time.time() - tic
    print(time_predict_with_uncertainty)

    # compute error
    mse_err = (pred - y_hour) ** 2
    rmse_err = np.sqrt(mse_err)
    avg_mse = np.mean(mse_err)
    print("avg mse:", avg_mse)

    # calibration
    res_dict = {"sample": i, "city": args.city, "date": use_date, "time": timepoint, "weekday": day}
    res_dict["mse"] = avg_mse
    res_dict["time"] = time_predict_with_uncertainty
    res_dict["r_all_mse"] = correlation(mse_err, std_preds)
    res_dict["r_all_rmse"] = correlation(rmse_err, std_preds)
    res_dict["r_vol_rmse"] = correlation(rmse_err[:, :, :, [0, 2, 4, 6]], std_preds[:, :, :, [0, 2, 4, 6]])
    res_dict["r_speed_rmse"] = correlation(rmse_err[:, :, :, [1, 3, 5, 7]], std_preds[:, :, :, [1, 3, 5, 7]])
    print(res_dict)
    final_df.append(res_dict)

    # save results
    out_err += rmse_err
    out_std += std_preds

df = pd.DataFrame(final_df)
df.to_csv(os.path.join(args.out_path, "correlation_df.csv"), index=False)

# Save to files
np.save(os.path.join(args.out_path, f"{args.city}_err.npy"), out_err / data_len)
np.save(os.path.join(args.out_path, f"{args.city}_std.npy"), out_std / data_len)
