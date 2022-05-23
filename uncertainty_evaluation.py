from cmath import pi
import os
import json
import numpy as np
import time
import argparse
import pandas as pd
from scipy.stats import pearsonr

from util.h5_util import load_h5_file
from competition.prepare_test_data.prepare_test_data import prepare_test
from methods_uncertainty.patch_uncertainty import PatchUncertainty
from methods_uncertainty.unet_uncertainty import TrivialUnetUncertainty
from methods_uncertainty.attenuation_uncertainty import AttenuationUncertainty
from methods_uncertainty.tta_uncertainty import TTAUncertainty
from metrics.uq_metrics import *


def correlation(err_arr, std_arr):
    r, p = pearsonr(err_arr.flatten(), std_arr.flatten())
    return r


MIN_DATE_TEST_DATA = "2020-04-01"

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_path", type=str, required=True)
parser.add_argument("-r", "--radius", type=int, default=50)
parser.add_argument("-t", "--model_str", type=str, default="up_patch")
parser.add_argument("-d", "--data_path", type=str, default=os.path.join("data", "raw"))
# sample from the 2020 data from one city
parser.add_argument("-c", "--city", type=str, default="ANTWERP")
# based on the metainfo of another city
parser.add_argument("-a", "--metacity", type=str, default="BERLIN")
parser.add_argument("-o", "--out_path", type=str, default="output_std")
parser.add_argument("-s", "--stride", type=int, default=30)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--alex_folder_structure", action="store_true", default=False)
parser.add_argument("--static_map", action="store_true", default=False, help="Use static map?")
args = parser.parse_args()

data_path = args.data_path
metacity = args.metacity

# set random seed to make this test dataset reproducible
np.random.seed(42)

os.makedirs(args.out_path, exist_ok=True)

# load static map:
if args.static_map:
    static_map_arr = load_h5_file(os.path.join(data_path, args.city, f"{args.city}_static.h5"))
else:
    static_map_arr = None

# load model and initialize uncertainty estimation
uncertainty_estimator = TTAUncertainty(**vars(args))
# AttenuationUncertainty(**vars(args))
# TrivialUnetUncertainty(**vars(args))
# TTAUncertainty(**vars(args))
# PatchUncertainty(static_map_arr=static_map_arr, **vars(args))

samples, mse_bl_list, mse_weighted_list, mse_middle_list = [], [], [], []

# save all corelation results in a df
final_df = []

# load weekday info
with open(os.path.join(data_path, "weekday2dates_2020.json"), "r") as infile:
    weekday2date = json.load(infile)
# load additional data from some other city, e.g. berlin
metainfo = load_h5_file(os.path.join(data_path, metacity, f"{metacity}_test_additional_temporal.h5"))
data_len = len(metainfo)

# # Evaluate single timepoint
# possible_dates = np.array([a for elem in list(weekday2date.values()) for a in elem]) # .flatten()
# possible_dates = possible_dates[possible_dates > MIN_DATE_TEST_DATA]
# data_len = len(possible_dates)

# outputs of all samples
out_samples = np.zeros((data_len, 3, 495, 436, 8))  # save avg std per cell

for i in range(data_len):
    # get sample based on the i-th sample of the metainfo file
    day = metainfo[i, 0]
    timepoint = int(metainfo[i, 1])
    possible_dates = np.array(weekday2date[str(day)])
    # select sample from the second half of 2020 data
    possible_dates = possible_dates[possible_dates > MIN_DATE_TEST_DATA]
    use_date = np.random.choice(possible_dates)
    # # Evaluate single timepoint
    # day = 0
    # timepoint = 168
    # use_date = possible_dates[i]
    print("weekday", day, "date", use_date, "time", timepoint)

    # load sample
    folder_test_data = "test" if args.alex_folder_structure else "training"
    sample_path = os.path.join(data_path, args.city, folder_test_data, f"{use_date}_{args.city}_8ch.h5")
    # load and cut two hour time slot
    two_hours = load_h5_file(sample_path, sl=slice(timepoint, timepoint + 24))
    x_hour, y_hour = prepare_test(two_hours)
    print(x_hour.shape, y_hour.shape)

    print("loaded data for sample ", i, x_hour.shape, y_hour.shape)
    tic = time.time()

    # Run uncertainty-aware model
    pred, uncertainty_scores = uncertainty_estimator(x_hour)

    time_predict_with_uncertainty = time.time() - tic
    print(time_predict_with_uncertainty)

    # compute error
    mse_err = (pred - y_hour) ** 2
    rmse_err = np.sqrt(mse_err)
    avg_mse = np.mean(mse_err)
    print("avg mse:", avg_mse)

    # calibration results - collect in a dictionary
    res_dict = {"sample": i, "city": args.city, "date": use_date, "time": timepoint, "weekday": day}
    res_dict["mse"] = avg_mse
    res_dict["runtime"] = time_predict_with_uncertainty
    res_dict["r_all_mse"] = correlation(mse_err, uncertainty_scores)
    res_dict["r_all_rmse"] = correlation(rmse_err, uncertainty_scores)
    res_dict["r_vol_rmse"] = correlation(rmse_err[:, :, :, [0, 2, 4, 6]], uncertainty_scores[:, :, :, [0, 2, 4, 6]])
    res_dict["r_speed_rmse"] = correlation(rmse_err[:, :, :, [1, 3, 5, 7]], uncertainty_scores[:, :, :, [1, 3, 5, 7]])
    print(res_dict)
    final_df.append(res_dict)

    # save results
    out_samples[i, 0] = y_hour[0] # TODO: lost into future (6) dimension
    out_samples[i, 1] = pred[0]
    out_samples[i, 2] = uncertainty_scores[0]

assert np.all(out_samples >= 0)
assert np.all(out_samples[:, 2] > 0)

# 1) ence
ence_scores = ence(out_samples)
print(ence_scores.shape)
np.save(os.path.join(args.out_path, "ence_scores.npy"), ence_scores)
print("saved ence scores")
print(np.mean(ence_scores))

print("computing intervals on val set") # TODO
quantiles = get_quantile(out_samples[:data_len//2], alpha=0.2)
print(quantiles.shape)
intervals = get_pred_interval(out_samples[data_len//2:, 1:], quantiles)
print(intervals.shape)

# 2) coverage
cov = coverage(intervals, out_samples[data_len//2:, 0])
print("saved coverage")
print(np.mean(cov))
print(cov.shape)
np.save(os.path.join(args.out_path, "coverage.npy"), cov)

# 3) pi width
pi_width = mean_pi_width(intervals)
np.save(os.path.join(args.out_path, "pi_width.npy"), pi_width)
print("Saved pi width")
print(np.mean(pi_width))
print(pi_width.shape)

# 4) correlation
out_err = np.sqrt((out_samples[:, 0, ...] - out_samples[:, 1, ...])**2)
per_cell_calib = np.zeros((495, 436, 8))
for i in range(495):
    for j in range(436):
        for c in range(8):
            per_cell_calib[i, j, c] = correlation(out_err[:, i, j, c], out_samples[:, 2, i, j, c])

# Save the sample-wise calibration results
df = pd.DataFrame(final_df)
df.to_csv(os.path.join(args.out_path, "correlation_df.csv"), index=False)

# save main result:
np.save(os.path.join(args.out_path, "calibration.npy"), per_cell_calib)

# Save city-wise error and std
np.save(os.path.join(args.out_path, "mean_err.npy"), np.mean(out_err, axis=0))
np.save(os.path.join(args.out_path, "mean_unc.npy"), np.mean(out_samples[:, 2], axis=0))

