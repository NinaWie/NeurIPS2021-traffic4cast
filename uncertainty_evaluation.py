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
from methods_uncertainty.patch_tta_uncertainty import PatchTTAUncertainty
from metrics.uq_metrics import *

UQ_METHOD_DICT = {
    "tta": {"model": "base_unet.pt", "uq_class": TTAUncertainty, "model_str": "unet"},
    "attenuation": {"model": "bayes_unet.pt", "uq_class": AttenuationUncertainty, "model_str": "bayes_unet"},
    "patch": {"model": "patch.pt", "uq_class": PatchUncertainty, "model_str": "up_patch"},
    "staticpatch": {"model": "static_patch.pt", "uq_class": PatchUncertainty, "model_str": "up_patch"},
    "trivial": {"model": "base_unet.pt", "uq_class": TrivialUnetUncertainty, "model_str": "unet"},
    "patchtta": {"model": "patch.pt", "uq_class": PatchTTAUncertainty, "model_str": "up_patch"},
}

# constants
QUANTILE_CONST = 0.1
PREVENT_UNC_0 = 1e-4
TIME_EVAL = 5 # 5 means one hour prediction into the future
# everything before that date is validation, everything afterwards is test
MIN_DATE_TEST_DATA = "2020-04-02"

def correlation(err_arr, std_arr):
    r, p = pearsonr(err_arr.flatten(), std_arr.flatten())
    return r

parser = argparse.ArgumentParser()
# uncertainty quantification method
parser.add_argument("-u", "--uq_method", type=str, default="patch")
parser.add_argument("-d", "--data_path", type=str, default="../../../data/t4c2021")
parser.add_argument("-o", "--out_name", type=str, default="output_std")
# sample from the 2020 data from one city
parser.add_argument("-c", "--city", type=str, default="ANTWERP")
# based on the metainfo of another city
parser.add_argument("-a", "--metacity", type=str, default="BERLIN")
# patch arguments
parser.add_argument("-r", "--radius", type=int, default=50)
parser.add_argument("-s", "--stride", type=int, default=30)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--alex_folder_structure", action="store_true", default=False)
parser.add_argument("--calibrate", action="store_true", default=False)
args = parser.parse_args()

# set random seed to make this test dataset reproducible
np.random.seed(42)

# make out paths
data_path = args.data_path
OUT_PATH_RUN = os.path.join(args.data_path, args.out_name)
os.makedirs(OUT_PATH_RUN, exist_ok=True)
stride = args.stride if "patch" in args.uq_method else 0
OUT_PATH = os.path.join(OUT_PATH_RUN, f"{args.uq_method}_{args.city.lower()}_{stride}")
os.makedirs(OUT_PATH, exist_ok=True)

# check that quantiles were computed beforehand
if not args.calibrate:
    assert os.path.exists(os.path.join(OUT_PATH, "speed_quantiles.npy"))

# uq method
uq_config = UQ_METHOD_DICT[args.uq_method]

# load static map:
if "static" in args.uq_method:
    static_map_arr = load_h5_file(os.path.join(data_path, args.city, f"{args.city}_static.h5"))
else:
    static_map_arr = None
# input arguments for model
model_args = vars(args)
model_args["static_map_arr"] = static_map_arr
model_args["model_str"] = uq_config["model_str"]
model_args["model_path"] = os.path.join(args.data_path, "trained_models", uq_config["model"])

# model class:
UQ_class = uq_config["uq_class"]

# load model and initialize uncertainty estimation
uncertainty_estimator = UQ_class(**model_args)

samples, mse_bl_list, mse_weighted_list, mse_middle_list = [], [], [], []

# save all corelation results in a df
final_df = []

# load weekday info
with open(os.path.join(data_path, "weekday2dates_2020.json"), "r") as infile:
    weekday2date = json.load(infile)
# load additional data from some other city, e.g. berlin
metainfo = load_h5_file(os.path.join(data_path, args.metacity, f"{args.metacity}_test_additional_temporal.h5"))
data_len = len(metainfo)

# # Evaluate single timepoint
# possible_dates = np.array([a for elem in list(weekday2date.values()) for a in elem]) # .flatten()
# possible_dates = possible_dates[possible_dates > MIN_DATE_TEST_DATA]
# data_len = len(possible_dates)

# outputs of all samples
out_samples_speed = np.zeros((data_len, 3, 495, 436, 4))  # save avg std per cell
out_samples_vol = np.zeros((data_len, 3, 495, 436, 4))  # save avg std per cell

for i in range(data_len):
    # get sample based on the i-th sample of the metainfo file
    day = metainfo[i, 0]
    timepoint = int(metainfo[i, 1])
    possible_dates = np.array(weekday2date[str(day)])
    # select sample from the second half of 2020 data
    if args.calibrate:
        possible_dates = possible_dates[possible_dates < MIN_DATE_TEST_DATA]
    else:
        possible_dates = possible_dates[possible_dates >= MIN_DATE_TEST_DATA]
    use_date = np.random.choice(possible_dates)
    # # Evaluate single timepoint
    # day = 0
    # timepoint = 168
    # use_date = possible_dates[i]
    print("\nSample", i, "weekday", day, "date", use_date, "time", timepoint)

    # load sample
    folder_test_data = "test" if args.alex_folder_structure else "training"
    sample_path = os.path.join(data_path, args.city, folder_test_data, f"{use_date}_{args.city}_8ch.h5")
    # load and cut two hour time slot
    two_hours = load_h5_file(sample_path, sl=slice(timepoint, timepoint + 24))
    x_hour, y_hour = prepare_test(two_hours)

    tic = time.time()

    # Run uncertainty-aware model
    pred, uncertainty_scores = uncertainty_estimator(x_hour)

    time_predict_with_uncertainty = time.time() - tic
    print("time processing", time_predict_with_uncertainty)

    # clipt to range
    pred = np.clip(pred, 0, 255)
    uncertainty_scores = np.clip(uncertainty_scores, PREVENT_UNC_0, None)

    # compute error
    mse_err = (pred - y_hour) ** 2
    rmse_err = np.sqrt(mse_err)
    
    print("avg mse:", np.mean(mse_err))

    # speed measures per sample - collect in a dictionary    
    speed_unc = uncertainty_scores[:, :, :, [1, 3, 5, 7]]
    speed_err = rmse_err[:, :, :, [1, 3, 5, 7]]
    # init dict
    res_dict = {"sample": i, "city": args.city, "date": use_date, "time": timepoint, "weekday": day}
    res_dict["rmse_speed"] = np.mean(speed_err)
    res_dict["mse"] = np.mean(mse_err)
    res_dict["runtime"] = time_predict_with_uncertainty
    res_dict["r_speed_rmse"] = correlation(speed_err, speed_unc)
    # evaluate change of uncertainty over time
    for ts in range(6):
        res_dict["mean_unc_"+str(ts)] = np.mean(speed_unc[ts])
        res_dict["mean_rmse_"+str(ts)] = np.mean(speed_err[ts])
    print("sample calibration:", res_dict["r_speed_rmse"])
    final_df.append(res_dict)

    # save results speed
    out_samples_speed[i, 0] = (y_hour[:, :, :, [1, 3, 5, 7]])[TIME_EVAL]
    out_samples_speed[i, 1] = (pred[:, :, :, [1, 3, 5, 7]])[TIME_EVAL]
    out_samples_speed[i, 2] = speed_unc[TIME_EVAL]
    # save results vol
    out_samples_vol[i, 0] = (y_hour[:, :, :, [0, 2, 4, 6]])[TIME_EVAL]
    out_samples_vol[i, 1] = (pred[:, :, :, [0, 2, 4, 6]])[TIME_EVAL]
    out_samples_vol[i, 2] = (uncertainty_scores[:, :, :, [0, 2, 4, 6]])[TIME_EVAL]

assert np.all(out_samples_speed >= 0)
assert np.all(out_samples_speed[:, 2] > 0)

# make df
df = pd.DataFrame(final_df)

if args.calibrate:
    # speed quantiles
    quantiles_speed = get_quantile(out_samples_speed, alpha=QUANTILE_CONST)
    print("computed speed quantiles", quantiles_speed.shape)
    np.save(os.path.join(OUT_PATH, "speed_quantiles.npy"), quantiles_speed)
    # vol quantiles
    quantiles_vol = get_quantile(out_samples_vol, alpha=QUANTILE_CONST)
    np.save(os.path.join(OUT_PATH, "vol_quantiles.npy"), quantiles_vol)
    print("computed vol quantiles", quantiles_vol.shape)

    # Save the sample-wise calibration results
    df.to_csv(os.path.join(OUT_PATH, "df_speed_calibration.csv"), index=False)
    exit()

# compute all scores for speed and vol
for out_samples, mode_name in zip([out_samples_speed, out_samples_vol], ["speed", "vol"]):
    # make folder
    channel_out_path = os.path.join(OUT_PATH, mode_name)
    os.makedirs(channel_out_path, exist_ok=True)

    # general result logs - only for speed
    if mode_name == "speed":
        df.to_csv(os.path.join(channel_out_path, "correlation_df.csv"), index=False)

    # 1) ence
    ence_scores = ence(out_samples)
    print(ence_scores.shape)
    np.save(os.path.join(channel_out_path, "ence_scores.npy"), ence_scores)
    print("saved ence scores")
    print(np.mean(ence_scores))

    print("load quantiles..")
    quantiles = np.load(os.path.join(OUT_PATH, f"{mode_name}_quantiles.npy"))
    print(quantiles.shape)
    intervals = get_pred_interval(out_samples[:, 1:], quantiles)
    print(intervals.shape)
    print("computed intervals")

    # 2) coverage
    cov = coverage(intervals, out_samples[:, 0])
    print("saved coverage - mean", np.mean(cov), cov.shape)
    np.save(os.path.join(channel_out_path, "coverage.npy"), cov)

    # 3) pi width
    pi_width = mean_pi_width(intervals)
    np.save(os.path.join(channel_out_path, "pi_width.npy"), pi_width)
    print("Saved pi width - mean", np.mean(pi_width), pi_width.shape)

    # 4) correlation
    out_err = np.sqrt((out_samples[:, 0, ...] - out_samples[:, 1, ...])**2)
    per_cell_calib = corr(out_err, out_samples[:, 2])
    # save pearsonr:
    np.save(os.path.join(channel_out_path, "calibration.npy"), per_cell_calib)
    print("saved calibration", np.nanmean(per_cell_calib), per_cell_calib.shape)

    # Save MEAN gt, pred, unc and err
    mean_unc_and_pred = np.mean(out_samples, axis=0)
    mean_err = np.expand_dims(np.mean(out_err, axis=0), 0)
    unc_pred_err = np.concatenate((mean_unc_and_pred, mean_err), axis=0)
    np.save(os.path.join(channel_out_path, "mean_gt_pred_unc_err.npy"), unc_pred_err)
    print("Saved mean unc, pred and err - shape", unc_pred_err.shape)

    # Save STD gt, pred, unc and err
    std_unc_and_pred = np.std(out_samples, axis=0)
    std_err = np.expand_dims(np.std(out_err, axis=0), 0)
    unc_pred_err = np.concatenate((std_unc_and_pred, std_err), axis=0)
    np.save(os.path.join(channel_out_path, "std_gt_pred_unc_err.npy"), unc_pred_err)
    print("Saved std unc, pred and err - shape", unc_pred_err.shape)
