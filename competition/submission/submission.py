#  Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import argparse
import datetime
import glob
import logging
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Union

import numpy as np
import psutil
import torch
import torch_geometric

from baselines.baselines_configs import configs
from baselines.checkpointing import load_torch_model_from_checkpoint
from util.h5_util import load_h5_file
from util.h5_util import write_data_to_h5
from util.logging import t4c_apply_basic_logging_config

# cut up
def create_patches(one_hour, radius=50, stride=50):
    tlen, xlen, ylen, chlen = one_hour.shape
    # print(xlen, (xlen - 2*radius), stride, (xlen - 2*radius)//stride)
    nr_in_x = (xlen - 2 * radius) // stride + 2
    nr_in_y = (ylen - 2 * radius) // stride + 2

    patch_collection = np.zeros((nr_in_x * nr_in_y, tlen, radius * 2, radius * 2, chlen))  # TODO: 2 auf 8 etc
    avg_arr = np.zeros((xlen, ylen))
    index_arr = np.zeros((nr_in_x * nr_in_y, 4))

    counter = 0
    for x in range(nr_in_x):
        # special case: we include the end patch
        if x == nr_in_x - 1:
            start_x = xlen - 2 * radius
            end_x = xlen
        else:
            start_x = x * stride
            end_x = x * stride + 2 * radius

        for y in range(nr_in_y):
            # special case: we include the end patch
            if y == nr_in_y - 1:
                start_y = ylen - 2 * radius
                end_y = ylen
            else:
                start_y = y * stride
                end_y = y * stride + 2 * radius

            patch_collection[counter] = one_hour[:, start_x:end_x, start_y:end_y, :]
            # remember how often each value was updated
            avg_arr[start_x:end_x, start_y:end_y] += 1
            index_arr[counter] = [start_x, end_x, start_y, end_y]
            counter += 1

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    # plt.imshow(avg_arr)
    # plt.colorbar()
    # plt.show()
    # exit()
    return patch_collection, avg_arr, index_arr


# stitch back together
def stitch_patches(out_patch, avg_arr, index_arr):
    xlen, ylen = avg_arr.shape

    prediction = np.zeros((6, xlen, ylen, 8))
    assert len(out_patch) == len(index_arr)

    for i in range(len(out_patch)):
        (start_x, end_x, start_y, end_y) = tuple(index_arr[i].astype(int).tolist())
        prediction[:, start_x:end_x, start_y:end_y] += out_patch[i]

    expand_avg_arr = np.tile(np.expand_dims(avg_arr, 2), 8)

    avg_prediction = prediction / expand_avg_arr
    return avg_prediction


def package_submission(
    data_raw_path: str,
    competition: str,
    model_str: str,
    model: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    device: str,
    submission_output_dir: Path,
    batch_size=10,
    num_tests_per_file=100,
    h5_compression_params: dict = None,
    zero_out_speed: bool = False,
    radius: int = 50,
    stride: int = 50,
    **additional_transform_args,
) -> Path:
    tstamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M")

    if h5_compression_params is None:
        h5_compression_params = {}

    use_patches = "patch" in model_str
    if use_patches:
        print("CUT UP IN PATCHES", use_patches)
        assert batch_size == 1, "for patches we need batch size 1"

    if submission_output_dir is None:
        submission_output_dir = Path(".")
    submission_output_dir.mkdir(exist_ok=True, parents=True)
    submission = submission_output_dir / f"submission_{model_str}_{competition}_{tstamp}.zip"
    logging.info(submission)

    # competition_files = [os.path.join("data", "temp_test_data", f"ANTWERP_train_data_x.h5")]
    competition_files = glob.glob(f"{data_raw_path}/**/*test_{competition}.h5", recursive=True)

    assert len(competition_files) > 0

    model_dict = None
    if isinstance(model, dict):
        model_dict = model

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(submission, "w") as z:
            for competition_file in competition_files:
                logging.info(f"  running model on {competition_file} (RAM {psutil.virtual_memory()[2]}%)")
                city = re.search(r".*/([A-Z]+)_test_", competition_file).group(1)
                if model_dict is not None:
                    model = model_dict[city]
                model = model.to(device)
                model.eval()

                pre_transform: Callable[[np.ndarray], Union[torch.Tensor, torch_geometric.data.Data]] = configs[model_str].get("pre_transform", None)
                post_transform: Callable[[Union[torch.Tensor, torch_geometric.data.Data]], np.ndarray] = configs[model_str].get(
                    "post_transform", None
                )

                assert num_tests_per_file % batch_size == 0, f"num_tests_per_file={num_tests_per_file} must be a multiple of batch_size={batch_size}"

                num_batches = num_tests_per_file // batch_size
                prediction = np.zeros(shape=(num_tests_per_file, 6, 495, 436, 8), dtype=np.uint8)

                with torch.no_grad():
                    for i in range(num_batches):
                        batch_start = i * batch_size
                        batch_end: np.ndarray = batch_start + batch_size
                        test_data: np.ndarray = load_h5_file(competition_file, sl=slice(batch_start, batch_end), to_torch=False)
                        additional_data = load_h5_file(
                            competition_file.replace("test", "test_additional"), sl=slice(batch_start, batch_end), to_torch=False
                        )
                        # additional_data = load_h5_file(
                        #     competition_file.replace("_train_data_x", "_additional"), sl=slice(batch_start, batch_end), to_torch=False
                        # )
                        if use_patches:
                            assert test_data.shape[0] == 1, "batch size must be 1"
                            test_data, avg_arr, index_arr = create_patches(test_data[0], radius=radius, stride=stride)
                            # print("test data", test_data.shape)
                            additional_transform_args["batch_dim"] = True
                            additional_transform_args["from_numpy"] = True
                            # print("additional_transform_args", additional_transform_args)

                        if pre_transform is not None:
                            test_data: Union[torch.Tensor, torch_geometric.data.Data] = pre_transform(
                                test_data, city=city, **additional_transform_args
                            )
                        else:
                            test_data = torch.from_numpy(test_data)
                            test_data = test_data.to(dtype=torch.float)
                        test_data = test_data.to(device)
                        additional_data = torch.from_numpy(additional_data)
                        additional_data = additional_data.to(device)
                        batch_prediction = model(test_data, city=city, additional_data=additional_data)

                        if post_transform is not None:
                            batch_prediction = post_transform(batch_prediction, city=city, **additional_transform_args)
                        # else:
                        batch_prediction = batch_prediction.cpu().detach().numpy()
                        batch_prediction = np.clip(batch_prediction, 0, 255)

                        if use_patches:
                            # print("prev stitching", batch_prediction.shape)
                            batch_prediction = stitch_patches(batch_prediction, avg_arr, index_arr)
                            # print("after stitching", batch_prediction.shape)

                        batch_prediction = np.around(batch_prediction).astype(np.uint8)
                        if zero_out_speed:
                            # Set all speeds to 0 where there is no volume in the corresponding heading
                            batch_prediction[:, :, :, 1] = batch_prediction[:, :, :, 1] * (batch_prediction[:, :, :, 0] > 0)
                            batch_prediction[:, :, :, 3] = batch_prediction[:, :, :, 3] * (batch_prediction[:, :, :, 2] > 0)
                            batch_prediction[:, :, :, 5] = batch_prediction[:, :, :, 5] * (batch_prediction[:, :, :, 4] > 0)
                            batch_prediction[:, :, :, 7] = batch_prediction[:, :, :, 7] * (batch_prediction[:, :, :, 6] > 0)

                        # clipping is important as assigning float array to uint8 array has not the intended effect.... (see `test_submission.test_assign_reload_floats)
                        prediction[batch_start:batch_end] = batch_prediction

                        # from metrics.mse import mse
                        # gt = load_h5_file(os.path.join("data", "temp_test_data", f"ANTWERP_train_data_y.h5"))
                        # print(batch_prediction.shape, gt.shape)
                        # print("MSE", mse(batch_prediction, gt))

                unique_values = np.unique(prediction)
                logging.info(f"  {len(unique_values)} unique values in prediction in the range [{np.min(prediction)}, {np.max(prediction)}]")
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(str(np.unique(prediction)))
                temp_h5 = os.path.join(temp_dir, os.path.basename(competition_file))
                arcname = os.path.join(*competition_file.split(os.sep)[-2:])
                logging.info(f"  writing h5 file {temp_h5} (RAM {psutil.virtual_memory()[2]}%)")
                write_data_to_h5(prediction, temp_h5, **h5_compression_params)
                logging.info(f"  adding {temp_h5} as {arcname} (RAM {psutil.virtual_memory()[2]}%)")
                z.write(temp_h5, arcname=arcname)
            logging.info(z.namelist())
    submission_mb_size = os.path.getsize(submission) / (1024 * 1024)
    logging.info(f"Submission {submission} with {submission_mb_size:.2f}MB")
    logging.info(
        f"RAM {psutil.virtual_memory()[2]}%, disk usage {(shutil.disk_usage('.')[0] - shutil.disk_usage('.')[1]) / (1024 * 1024 * 1024):.2f}GB left"
    )
    return submission


def create_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for this program."""
    parser = argparse.ArgumentParser(description=("This programs creates a submission."))
    parser.add_argument("--checkpoint", type=str, help="Torch checkpoint file", required=True, default=None)
    parser.add_argument("--model_str", type=str, help="The `model_str` in the config", required=False, default="unet")
    parser.add_argument("--data_raw_path", type=str, help="Path of raw data", required=False, default="./data/raw")
    parser.add_argument("--batch_size", type=int, help="Batch size for evaluation", required=False, default=10)
    parser.add_argument("--device", type=str, help="Device", required=False, default="cpu")
    parser.add_argument(
        "--submission_output_dir",
        type=str,
        default=None,
        required=False,
        help="If given, submission is stored to this directory instead of current.",
    )
    return parser


def main(model_str: str, checkpoint: str, batch_size: int, device: str, data_raw_path: str, submission_output_dir: Optional[str] = None):
    t4c_apply_basic_logging_config()
    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str].get("model_config", {})
    model = model_class(**model_config)
    load_torch_model_from_checkpoint(checkpoint=checkpoint, model=model)
    competitions = ["temporal", "spatiotemporal"]
    for competition in competitions:
        package_submission(
            data_raw_path=data_raw_path,
            competition=competition,
            model=model,
            model_str=model_str,
            batch_size=batch_size,
            device=device,
            h5_compression_params={"compression_level": 6},
            submission_output_dir=Path(submission_output_dir if submission_output_dir is not None else "."),
        )


if __name__ == "__main__":
    parser = create_parser()
    try:
        params = parser.parse_args()
        main(**vars(params))
    except Exception as e:
        print(f"There was an error during execution, please review: {e}")
        parser.print_help()
        exit(1)
