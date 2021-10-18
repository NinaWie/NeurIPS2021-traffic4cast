import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import glob

from util.h5_util import load_h5_file, write_data_to_h5
from metrics.mse import mse
import logging
from util.logging import t4c_apply_basic_logging_config
import psutil
import tempfile
import zipfile

BASE_FOLDER = "data/raw"
STATS_FOLDER = "data/stats"
TIMEDIFF_FOLDER = "data/time_diff_bins"
speed_inds = [1, 3, 5, 7]
vol_inds = [0, 2, 4, 6]


city_timelag_dict = {
    "ANTWERP": ["BANGKOK", "BARCELONA", "MOSCOW"],
    "ISTANBUL": ["BANGKOK", "MOSCOW"],
    "BERLIN": ["ANTWERP", "BARCELONA"],
    "CHICAGO": ["ANTWERP", "BANGKOK", "BARCELONA", "MOSCOW"],  # no info
    "MELBOURNE": ["ANTWERP", "BANGKOK", "BARCELONA", "MOSCOW"],  # no info
}


def load_all_historic(city, year):
    all_historic = []
    for weekday in range(7):
        all_historic.append(load_h5_file(os.path.join(STATS_FOLDER, f"{city}_{year}_{weekday}.h5")))
    return np.stack(all_historic)


class NaiveStatsTemporal(torch.nn.Module):
    def __init__(self):
        """Returns prediction consisting of a weighted average of the last
        hour."""
        super(NaiveStatsTemporal, self).__init__()
        # use 2019-2020 shift from other cities
        self.timeshift_arr = load_timeshift(["BANGKOK", "BARCELONA", "MOSCOW"])
        # ["ANTWERP", "BANGKOK", "BARCELONA", "MOSCOW"])
        self.do_shift_other_cities = True

    def forward(self, x: torch.Tensor, additional_data: torch.Tensor, city: str, *args, **kwargs):
        x = x.numpy()
        additional_data = additional_data.numpy()

        stats = load_all_historic(city, 2019)
        print("loaded stats for city", city, stats.shape)

        preds = np.zeros((len(x), 6, 495, 436, 8))

        for x_idx in range(len(x)):
            weekday = additional_data[x_idx, 0]
            time = additional_data[x_idx, 1]

            # Input: data for hour in 2020
            x_timeslot = x[x_idx].astype(float)

            # times
            x_times = np.arange(12) + time
            y_times = np.add([1, 2, 3, 6, 9, 12], 11 + time)

            # get historic data of y time slot
            out_hist_bef = stats[weekday, y_times]

            # # VERSION 1: compare data from x with stats to get temporal shift
            # get historic data of x time slot:
            # hist_x = stats[weekday, x_times].astype(float) + 0.01  # to avoid division by zero
            # # get shift from hour before (2020 / 2019)
            # shift_2020_2019 = x_timeslot / hist_x
            # # set all to 1 which were division by close to zero
            # shift_2020_2019[hist_x == 0.01] = 1
            # out_hist = out_hist_bef * np.mean(shift_2020_2019, axis=0)

            # # VERSION 2: Get tempotal shift from other cities
            # incorporate timeshift from other cityes
            if self.do_shift_other_cities:
                day_shifts = self.timeshift_arr[:, int(weekday), y_times]
                out_hist = out_hist_bef.copy()
                for slot in range(6):
                    out_hist[slot, :, :, vol_inds] = out_hist[slot, :, :, vol_inds] * day_shifts[0, slot]
                    out_hist[slot, :, :, speed_inds] = out_hist[slot, :, :, speed_inds] * day_shifts[1, slot]

            preds[x_idx] = out_hist

        # Convert the float values from the operation back to uint8
        x = preds.astype(np.uint8)
        # Set all speeds to 0 where there is no volume in the corresponding heading
        # x[:, :, :, :, 1] = x[:, :, :, :, 1] * (x[:, :, :, :, 0] > 0)
        # x[:, :, :, :, 3] = x[:, :, :, :, 3] * (x[:, :, :, :, 2] > 0)
        # x[:, :, :, :, 5] = x[:, :, :, :, 5] * (x[:, :, :, :, 4] > 0)
        # x[:, :, :, :, 7] = x[:, :, :, :, 7] * (x[:, :, :, :, 6] > 0)

        x = torch.from_numpy(x).float()
        return x


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

    # OPTION 1: AS IN GENERATED TEST DATA
    # with open(os.path.join("data", "weekday2dates_2020.json"), "r") as infile:
    #     weekday2date = json.load(infile)
    # # load additional data from some other city, e.g. berlin
    # metainfo = load_h5_file(os.path.join(BASE_FOLDER, "BERLIN", f"BERLIN_test_additional_temporal.h5"))

    # save mses: historic wo 2020 shift, historic with 2020 shift, avg, avg + hist
    mses = []
    naive_stats = NaiveStatsTemporal()
    # for weekday in range(7):
    for t_file in test_files:
        print("-------- test file", t_file, "------------")
        fn = t_file.split(os.sep)[-1]
        date = fn.split(os.sep)[-1].split("_")[0]
        weekday = date2weekday[date]

        # load the file
        day_arr = load_h5_file(t_file)

        # use some random time points for this file
        rand_time_points = np.random.permutation(288 - 24)[:nr_time_test]

        # OPTION 1: AS IN GENERATED TEST DATA
        # # use some random time points for this file
        # rand_time_points = metainfo[metainfo[:, 0] == weekday, 1]
        # possible_dates = weekday2date[str(weekday)]
        # use_date = np.random.choice(possible_dates)

        # print("load file for one day ...", f"{BASE_FOLDER}/{test_city}/training/{use_date}_{test_city}_8ch.h5")
        # day_arr = load_h5_file(f"{BASE_FOLDER}/{test_city}/training/{use_date}_{test_city}_8ch.h5")

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

            # COMPARE TO RUNNING naive stats
            metainfo_new = np.array([[weekday, time]])
            pred_y = naive_stats(torch.from_numpy(np.expand_dims(x, 0)), torch.from_numpy(metainfo_new), test_city)
            print(pred_y.numpy().shape)
            print(mse(out_hist, gt_y), mse(pred_y[0].numpy(), gt_y))
            # hist_avg = np.mean(np.stack((out_hist, out_avg)), axis=0)
            #         print("MSE", mse(out_avg, gt_y), mse(out_hist, gt_y)
            mses.append([mse(out_avg, gt_y), mse(out_hist_bef, gt_y), mse(out_hist, gt_y)])  # , mse(hist_avg, gt_y)])
            # print(mses[-1])
    mse_res = np.array(mses)
    print(mse_res.shape, np.mean(mse_res, axis=0))


# # doesn't work and I should rather use the original function in the package
# def package_submission(prediction, city, submission_output_dir=None):
#     from pathlib import Path

#     if submission_output_dir is None:
#         submission_output_dir = Path(".")
#     submission_output_dir.mkdir(exist_ok=True, parents=True)
#     submission = submission_output_dir / f"submission_test_{city}.zip"
#     logging.info(submission)

#     with tempfile.TemporaryDirectory() as temp_dir:
#         with zipfile.ZipFile(submission, "w") as z:
#             h5_compression_params = {"compression_level": None}
#             prediction = prediction.astype(np.uint8)
#             if logging.getLogger().isEnabledFor(logging.DEBUG):
#                 logging.debug(str(np.unique(prediction)))
#             temp_h5 = os.path.join(temp_dir, os.path.basename(competition_file))
#             arcname = os.path.join(*competition_file.split(os.sep)[-2:])
#             logging.info(f"  writing h5 file {temp_h5} (RAM {psutil.virtual_memory()[2]}%)")
#             write_data_to_h5(prediction, temp_h5, **h5_compression_params)
#             logging.info(f"  adding {temp_h5} as {arcname} (RAM {psutil.virtual_memory()[2]}%)")
#             z.write(temp_h5, arcname=arcname)


def submit_city(test_city, mode="hist"):
    """
    Use an actual test city with no ground truth data available
    mode: one of hist, avg, hist_avg, hist_noshift
    """

    metainfo = load_h5_file(os.path.join(BASE_FOLDER, test_city, f"{test_city}_test_additional_temporal.h5"))
    day_arr = load_h5_file(os.path.join(BASE_FOLDER, test_city, f"{test_city}_test_temporal.h5"))

    # use 2019-2020 shift from other cities
    timeshift_arr = load_timeshift(city_timelag_dict[test_city])

    # allocate prediction array
    prediction = np.zeros(shape=(100, 6, 495, 436, 8), dtype=np.uint8)

    for x_idx in range(100):
        weekday = metainfo[x_idx, 0]
        time = metainfo[x_idx, 1]
        # Main input
        x = day_arr[x_idx]

        x_times = np.arange(12) + time
        y_times = np.add([1, 2, 3, 6, 9, 12], 11 + time)

        # run avg classifier
        if "avg" in mode:
            out_avg = weighted_avg(np.expand_dims(x, 0))[0]

        # load historic data from 2019
        if "hist" in mode:
            out_hist_bef = get_historic_avg(test_city, 2019, weekday, y_times)

        # incorporate timeshift
        if mode in ["hist_avg", "hist"]:
            day_shifts = timeshift_arr[:, int(weekday), y_times]
            out_hist = out_hist_bef.copy()
            for slot in range(6):
                out_hist[slot, :, :, vol_inds] = out_hist[slot, :, :, vol_inds] * day_shifts[0, slot]
                out_hist[slot, :, :, speed_inds] = out_hist[slot, :, :, speed_inds] * day_shifts[1, slot]

        if mode == "hist_avg":
            # combine avg and stats predictions
            pred = np.mean(np.stack((out_hist, out_avg)), axis=0)
        elif mode == "hist":
            pred = out_hist
        elif mode == "hist_noshift":
            pred = out_hist_bef
        elif mode == "avg":
            pred - out_avg
        else:
            raise RuntimeError("Invalid mode argument")


def create_test_data(test_city, use_additional_from="BERLIN", out_path=None):
    # load weekday info
    with open(os.path.join("data", "weekday2dates_2020.json"), "r") as infile:
        weekday2date = json.load(infile)
    # load additional data from some other city, e.g. berlin
    metainfo = load_h5_file(os.path.join(BASE_FOLDER, use_additional_from, f"{use_additional_from}_test_additional_temporal.h5"))

    # init x and y array
    train_data_x = np.zeros((100, 12, 495, 436, 8), dtype=np.uint8)
    train_data_y = np.zeros((100, 6, 495, 436, 8), dtype=np.uint8)

    new_additional_data = np.zeros(metainfo.shape, dtype=np.uint8)
    ind_counter = 0
    # FAST VERSION: use several timeslots from a single day
    # for day in range(7):
    #     possible_dates = weekday2date[str(day)]
    #     use_date = np.random.choice(possible_dates)

    #     print("load file for one day ...", f"{BASE_FOLDER}/{test_city}/training/{use_date}_{test_city}_8ch.h5")
    #     day_arr = load_h5_file(f"{BASE_FOLDER}/{test_city}/training/{use_date}_{test_city}_8ch.h5")

    #     use_times = metainfo[metainfo[:, 0] == day, 1]
    #     print("day", day, "times", use_times)

    #     for i, time in enumerate(use_times):
    #         train_data_x[ind_counter] = day_arr[time : time + 12]
    #         y_inds = np.add([1, 2, 3, 6, 9, 12], 11 + time)
    #         train_data_y[ind_counter] = day_arr[y_inds]
    #         new_additional_data[ind_counter] = [day, time]
    #         ind_counter += 1
    # BETTER VERSION: use really random data accoridng to metadata
    for i in range(len(metainfo)):
        day = metainfo[i, 0]
        time = metainfo[i, 1]
        possible_dates = weekday2date[str(day)]
        use_date = np.random.choice(possible_dates)
        print(day, use_date, time)
        print("load file for one day ...", f"{BASE_FOLDER}/{test_city}/training/{use_date}_{test_city}_8ch.h5")
        day_arr = load_h5_file(f"{BASE_FOLDER}/{test_city}/training/{use_date}_{test_city}_8ch.h5")
        train_data_x[ind_counter] = day_arr[time : time + 12]
        y_inds = np.add([1, 2, 3, 6, 9, 12], 11 + time)
        print(time, y_inds)
        train_data_y[ind_counter] = day_arr[y_inds]
        ind_counter += 1
    new_additional_data = metainfo

    if out_path is not None:
        write_data_to_h5(train_data_x, os.path.join(out_path, f"{test_city}_train_data_x.h5"))
        write_data_to_h5(train_data_y, os.path.join(out_path, f"{test_city}_train_data_y.h5"))
        write_data_to_h5(new_additional_data, os.path.join(out_path, f"{test_city}_additional.h5"))
    else:
        return train_data_x, train_data_y, new_additional_data


if __name__ == "__main__":
    # arr = load_all_historic("BERLIN", 2019)
    test_city = "ANTWERP"
    # eval_on_train_city(test_city)

    out_path = os.path.join("data", "temp_test_data")
    # create data
    create_test_data(test_city, out_path=out_path, use_additional_from="ISTANBUL")

    # train_data_x = load_h5_file(os.path.join(out_path, f"{test_city}_train_data_x.h5"))
    # train_data_y = load_h5_file(os.path.join(out_path, f"{test_city}_train_data_y.h5"))
    # metainfo = load_h5_file(os.path.join(out_path, f"{test_city}_additional.h5"))
    # print("Loaded x and y", train_data_x.shape, train_data_y.shape, metainfo.shape)

    # naive_stats = NaiveStatsTemporal()
    # pred_y = naive_stats(torch.from_numpy(train_data_x), torch.from_numpy(metainfo), test_city)
    # print(pred_y.size())
    # print("MSE naive stats", mse(pred_y.numpy(), train_data_y))

    # from baselines.naive_weighted_average import NaiveWeightedAverage

    # naive_stats = NaiveWeightedAverage()
    # pred_y = naive_stats(torch.from_numpy(train_data_x))  # , torch.from_numpy(metainfo), test_city)
    # print(pred_y.size())
    # print("MSE avg", mse(pred_y.numpy(), train_data_y))
