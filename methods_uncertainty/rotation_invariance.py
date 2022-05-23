from cmath import pi
import os
import json
import numpy as np
import time
import argparse
import torch
import pandas as pd
from scipy.stats import wilcoxon

from util.h5_util import load_h5_file
from competition.prepare_test_data.prepare_test_data import prepare_test
from methods_uncertainty.tta_uncertainty import DataAugmentation
from metrics.uq_metrics import *
from methods_uncertainty.unet_uncertainty import UnetBasedUncertainty

class RotationInvariance(UnetBasedUncertainty):

    def __init__(self, model_path, device="cuda", model_str="bayes_unet", num_channels=8, **kwargs) -> None:
        super().__init__(model_path, device, model_str, **kwargs)

        self.augmentor = DataAugmentation()

    def __call__(self, x_hour):

        # pretransform
        inp_data = self.pre_transform(np.expand_dims(x_hour,0), from_numpy=True, batch_dim=True)

        # inp_data: shape (1, 96, 496. 436)
        augmented_inp_data = self.augmentor.transform(inp_data)
        # augmented_inp_data: shape (8, 96, 496. 436)
        inp = augmented_inp_data.to(self.device)
        res = []
        for i in range(inp.size()[0]):
            pred_part = self.model(torch.unsqueeze(inp[i], 0))
            res.append(pred_part.detach().cpu())
        pred = torch.stack(res, dim=0)
        pred_deaugmented = self.augmentor.detransform(pred.squeeze())

        # post transform
        out = self.post_transform(pred_deaugmented, normalize=True, batch_dim=True).detach().numpy()

        return out

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

rot_processor = RotationInvariance(**vars(args))

samples, mse_bl_list, mse_weighted_list, mse_middle_list = [], [], [], []

# save all corelation results in a df
final_df = []

# load weekday info
with open(os.path.join(data_path, "weekday2dates_2020.json"), "r") as infile:
    weekday2date = json.load(infile)
# load additional data from some other city, e.g. berlin
metainfo = load_h5_file(os.path.join(data_path, metacity, f"{metacity}_test_additional_temporal.h5"))
data_len = len(metainfo)


for i in range(data_len):
    # get sample based on the i-th sample of the metainfo file
    day = metainfo[i, 0]
    timepoint = int(metainfo[i, 1])
    possible_dates = np.array(weekday2date[str(day)])
    # select sample from the second half of 2020 data
    possible_dates = possible_dates[possible_dates > MIN_DATE_TEST_DATA]
    use_date = np.random.choice(possible_dates)
    # # Evaluate single timepoint
    print("weekday", day, "date", use_date, "time", timepoint)

    # load sample
    folder_test_data = "test" if args.alex_folder_structure else "training"
    sample_path = os.path.join(data_path, args.city, folder_test_data, f"{use_date}_{args.city}_8ch.h5")
    # load and cut two hour time slot
    two_hours = load_h5_file(sample_path, sl=slice(timepoint, timepoint + 24))
    x_hour, y_hour = prepare_test(two_hours)


    # Run uncertainty-aware model
    rotated_predictions = rot_processor(x_hour)
    print(rotated_predictions.shape, len(rot_processor.augmentor.transform_names))

    # calibration results - collect in a dictionary
    res_dict = {"sample": i, "city": args.city, "date": use_date, "time": timepoint, "weekday": day}

    # compute error for each augmentation
    for (rot_name, pred) in zip(rot_processor.augmentor.transform_names, rotated_predictions):
        mse_err = np.mean((pred - y_hour) ** 2)
        res_dict[rot_name] = mse_err
    final_df.append(res_dict)
    print(res_dict)


# Save the sample-wise calibration results
df = pd.DataFrame(final_df)
df.to_csv(os.path.join(args.out_path, "rot_invariance_df.csv"), index=False)

rot_variants = df.drop(["sample", "city", "date", "time", "weekday"], axis=1)
print(rot_variants.mean().sort_values())

print("Wilcoxon test")
for method in rot_variants.columns[1:]:
    print(method, wilcoxon(df["same"].values, df[method].values))