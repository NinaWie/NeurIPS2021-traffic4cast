import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import glob

from baselines.baselines_configs import configs
from util.h5_util import load_h5_file
from metrics.mse import mse


BASE_FOLDER = "data/raw"
STATS_FOLDER = "data/stats"
TIMEDIFF_FOLDER = "data/time_diff_bins"
speed_inds = [1, 3, 5, 7]
vol_inds = [0, 2, 4, 6]


city_timelag_dict = {
    "ANTWERP": ["BANGKOK"]
    # TODO
}


def weighted_avg(x):
    """Simply copied from t4c baselines"""
    weights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6]
    x = np.average(x, axis=1, weights=weights)
    x = np.expand_dims(x, axis=1)
    x = np.repeat(x, repeats=6, axis=1)
    # Convert the float values from the average operation back to uint8
    x = x.astype(np.uint8)
    # Set all speeds to 0 where there is no volume in the corresponding heading
    x[:, :, :, :, 1] = x[:, :, :, :, 1] * (x[:, :, :, :, 0] > 0)
    x[:, :, :, :, 3] = x[:, :, :, :, 3] * (x[:, :, :, :, 2] > 0)
    x[:, :, :, :, 5] = x[:, :, :, :, 5] * (x[:, :, :, :, 4] > 0)
    x[:, :, :, :, 7] = x[:, :, :, :, 7] * (x[:, :, :, :, 6] > 0)
    return x


def get_historic_avg(city, year, weekday, time_slots):
    """Load the mean of a timeslot for a specific city"""
    stats_data = load_h5_file(os.path.join(STATS_FOLDER, f"{city}_{year}_{weekday}.h5"))
    return stats_data[time_slots]


def load_timeshift(use_timeshift_from):
    # use_timeshift_from = ["BANGKOK"]
    timeshift_arr = np.zeros((len(use_timeshift_from), 2, 7, 288))
    for i, city in enumerate(use_timeshift_from):
        timeshift_arr[i] = np.load(os.path.join(TIMEDIFF_FOLDER, f"{city}_timeshift.npy"))
    timeshift_arr = np.mean(timeshift_arr, axis=0)
    print("loaded timeshifts from cities", use_timeshift_from, timeshift_arr.shape)
    return timeshift_arr


def eval_on_train_city(test_city, nr_test=5, nr_time_test=5):
    # need this to weekday mapping
    with open("data/date2weekday_2020.json", "r") as infile:
        date2weekday = json.load(infile)

    # use 2019-2020 shift from other cities
    timeshift_arr = load_timeshift(city_timelag_dict[test_city])

    # Use some 2020 files as test data
    all_test_files = sorted(glob.glob(f"{BASE_FOLDER}/{test_city}/training/*2020*8ch.h5", recursive=True))
    test_files = np.random.choice(all_test_files, nr_test)

    # save mses: historic wo 2020 shift, historic with 2020 shift, avg, avg + hist
    mses = []
    for t_file in test_files:
        print("-------- test file", t_file, "------------")
        fn = t_file.split(os.sep)[-1]
        date = fn.split(os.sep)[-1].split("_")[0]
        weekday = date2weekday[date]

        # load the file
        day_arr = load_h5_file(t_file)

        # use some random time points for this file
        rand_time_points = np.random.permutation(288 - 24)[:nr_time_test]

        for time in rand_time_points:
            x_times = np.arange(12) + time
            y_times = np.add([1, 2, 3, 6, 9, 12], 11 + time)
            #         print("test_times:", x_times, y_times)

            # this is usually not available
            gt_y = day_arr[y_times]

            # run avg classifier
            x = day_arr[x_times]
            #         print("x", x.shape, "gt y", gt_y.shape)
            out_avg = weighted_avg(np.expand_dims(x, 0))[0]
            #         print("out avg", out_avg.shape)

            # load historic data from 2019
            out_hist_bef = get_historic_avg(test_city, 2019, weekday, y_times)
            # incorporate timeshift
            day_shifts = timeshift_arr[:, int(weekday), y_times]
            #         print("MSE before", mse(out_hist, gt_y))

            out_hist = out_hist_bef.copy()
            for slot in range(6):
                out_hist[slot, :, :, vol_inds] = out_hist[slot, :, :, vol_inds] * day_shifts[0, slot]
                out_hist[slot, :, :, speed_inds] = out_hist[slot, :, :, speed_inds] * day_shifts[1, slot]

            hist_avg = np.mean(np.stack((out_hist, out_avg)), axis=0)
            #         print("MSE", mse(out_avg, gt_y), mse(out_hist, gt_y)
            mses.append([mse(out_avg, gt_y), mse(out_hist_bef, gt_y), mse(out_hist, gt_y), mse(hist_avg, gt_y)])
            # print(mses[-1])
    mse_res = np.array(mses)
    print(mse_res.shape, np.mean(mse_res, axis=0))


def submit_city(test_city):
    """Use an actual test city with no ground truth data available"""

    metainfo = load_h5_file(os.path.join(BASE_FOLDER, test_city, f"{test_city}_test_additional_temporal.h5"))
    day_arr = load_h5_file(os.path.join(BASE_FOLDER, test_city, f"{test_city}_test_temporal.h5"))

    # use 2019-2020 shift from other cities
    timeshift_arr = load_timeshift(city_timelag_dict[test_city])

    for x_idx in range(100):
        weekday = metainfo[x_idx, 0]
        time = metainfo[x_idx, 1]
        # Main input
        x = day_arr[x_idx]

        x_times = np.arange(12) + time
        y_times = np.add([1, 2, 3, 6, 9, 12], 11 + time)

        # run avg classifier
        out_avg = weighted_avg(np.expand_dims(x, 0))[0]

        # load historic data from 2019
        out_hist_bef = get_historic_avg(test_city, 2019, weekday, y_times)

        # incorporate timeshift
        day_shifts = timeshift_arr[:, int(weekday), y_times]
        out_hist = out_hist_bef.copy()
        for slot in range(6):
            out_hist[slot, :, :, vol_inds] = out_hist[slot, :, :, vol_inds] * day_shifts[0, slot]
            out_hist[slot, :, :, speed_inds] = out_hist[slot, :, :, speed_inds] * day_shifts[1, slot]

        # combine avg and stats predictions
        hist_avg = np.mean(np.stack((out_hist, out_avg)), axis=0)


if __name__ == "__main__":
    eval_on_train_city("ANTWERP")
