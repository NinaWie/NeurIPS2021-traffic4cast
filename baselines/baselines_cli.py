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
import binascii
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
from collections import defaultdict
import json

import numpy as np
import torch
import torch.nn.functional as F  # noqa
import torch.optim as optim
import torch_geometric
import tqdm
from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import GradsHistHandler
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import create_supervised_evaluator
from ignite.engine import create_supervised_trainer
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import Checkpoint
from ignite.handlers import DiskSaver
from ignite.handlers import global_step_from_engine
from ignite.metrics import Loss
from ignite.metrics import RunningAverage
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from baselines.baselines_configs import configs
from baselines.checkpointing import load_torch_model_from_checkpoint
from baselines.checkpointing import save_torch_model_to_checkpoint
from competition.scorecomp import scorecomp
from competition.submission.submission import package_submission
from data.dataset.dataset import T4CDataset
from data.dataset.dataset_geometric import GraphTransformer
from data.dataset.dataset_geometric import T4CGeometricDataset
from util.logging import t4c_apply_basic_logging_config
from util.monitoring import system_status
from util.tar_util import untar_files


def run_model(
    train_model: torch.nn.Module,
    train_dataset: T4CDataset,
    val_dataset: T4CDataset,
    random_seed: int,
    train_fraction: float,
    val_fraction: float,
    batch_size: int,
    num_workers: int,
    epochs: int,
    dataloader_config: dict,
    optimizer_config: dict,
    device: str = None,
    geometric: bool = False,
    limit: Optional[int] = None,
    data_parallel=False,
    device_ids=None,
    checkpoint_name="",
    padding=(0, 0, 0, 0),
    **kwargs,
):  # noqa

    logging.info("dataset has size %s", len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, **dataloader_config,)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, **dataloader_config)

    # Optimizer
    if "lr" not in optimizer_config:
        optimizer_config["lr"] = 1e-4

    if device is None:
        logging.warning("device not set, torturing CPU.")
        device = "cpu"
        # TODO data parallelism and whitelist

    if torch.cuda.is_available() and data_parallel:
        # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
        if torch.cuda.device_count() > 1:
            # https://stackoverflow.com/questions/59249563/runtimeerror-module-must-have-its-parameters-and-buffers-on-device-cuda1-devi
            train_model = torch.nn.DataParallel(train_model, device_ids=device_ids)
            logging.info(f"Let's use {len(train_model.device_ids)} GPUs: {train_model.device_ids}!")
            device = f"cuda:{train_model.device_ids[0]}"

    optimizer = optim.Adam(train_model.parameters(), **optimizer_config)

    train_model = train_model.to(device)

    # Loss
    loss = F.mse_loss
    if True:  # geometric:
        train_pure_torch(device, epochs, optimizer, train_loader, val_loader, train_model, checkpoint_name=checkpoint_name, padding=padding)
    else:
        train_ignite(device, epochs, loss, optimizer, train_loader, val_loader, train_model, checkpoint_name=checkpoint_name)
    logging.info("End training of on %s for %s epochs", device, epochs)
    return train_model, device


def train_pure_torch(device, epochs, optimizer, train_loader, val_loader, train_model, checkpoint_name="", padding=(0, 0, 0, 0)):
    save_out_dir = "ckpt" + checkpoint_name
    results_dict = defaultdict(list)
    best_val_loss = np.inf
    for epoch in range(epochs):
        train_loss = _train_epoch_pure_torch(train_loader, device, train_model, optimizer)
        val_loss = _val_pure_torch(val_loader, device, train_model, padding=padding)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save if best result
            save_torch_model_to_checkpoint(model=train_model, epoch=epoch, out_dir=save_out_dir)
            with open(os.path.join(save_out_dir, "results.json"), "w") as outfile:
                json.dump(results_dict, outfile)
        log = "Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}"
        # save results
        results_dict["epoch"].append(epoch)
        results_dict["train loss"].append(train_loss)
        results_dict["val_loss"].append(val_loss)
        logging.info(log.format(epoch, train_loss, val_loss))
        if (epoch + 1) % 50 == 0:
            # Save regularly as backup
            save_torch_model_to_checkpoint(model=train_model, epoch=epoch, out_dir=save_out_dir, out_name=f"epoch_{epoch:04}")
            with open(os.path.join(save_out_dir, "results.json"), "w") as outfile:
                json.dump(results_dict, outfile)


def bayes_criterion(y_pred, y_true, num_channels=8, validate=False):
    """Loss attenuation"""

    # unstack y true time dimension
    bs, ts_ch, xsize, ysize = y_true.size()
    num_time_steps = int(ts_ch / num_channels)
    # (k, 12 * 8, 495, 436) -> (k, 12, 8, 495, 436)
    y_true_unstacked = torch.reshape(y_true, (bs, num_time_steps, num_channels, xsize, ysize))
    
    # unstack pred and distinguish the mu and sigma
    y_pred_unstacked = torch.reshape(y_pred, (bs, num_time_steps, num_channels, 2, xsize, ysize))

    # extract mu and sigma
    mu = y_pred_unstacked[:, :, :, 0] # first output neuron

    # validation: only use mu
    if validate:
        return torch.mean((mu - y_true_unstacked)**2)

    log_sig = y_pred_unstacked[:, :, :, 1] # second output neuron
    sig = torch.exp(log_sig) # undo the log
    
    return torch.mean(torch.log(sig**2) + ((y_true_unstacked - mu) / sig)**2)

def bayes_criterion_val(y_pred, y_true, num_channels=8):
    return bayes_criterion(y_pred, y_true, num_channels=8, validate=True)

def _train_epoch_pure_torch(loader, device, model, optimizer):
    loss_to_print = 0
    
    if hasattr(model, "bayes_loss") and model.bayes_loss:
        criterion = bayes_criterion
    else:
        criterion = torch.nn.MSELoss()

    nr_train_data = len(loader)
    for i, (input_data, ground_truth) in enumerate(loader):  # tqdm.tqdm(loader, desc="train")):
        # if isinstance(input_data, torch_geometric.data.Data):
        #     input_data = input_data.to(device)
        #     ground_truth = input_data.y
        # else:
        #     input_data, ground_truth = input_data
        input_data = input_data.to(device)
        ground_truth = ground_truth.to(device)

        model.train()
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, ground_truth)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_to_print += loss.item()
        if (i + 1) % 50 == 0:
            print(f"train {i+1}/{nr_train_data} ({loss.item()})")
    return loss_to_print


@torch.no_grad()
def _val_pure_torch(loader, device, model, padding=(0, 0, 0, 0)):
    # For validation, we exclude the padding area because it biases the MSE
    s_x, e_x, s_y, e_y = padding
    # indexing :0 does not work
    e_x = 1 if e_x == 0 else e_x
    e_y = 1 if e_y == 0 else e_y

    if hasattr(model, "bayes_loss") and model.bayes_loss:
        criterion = bayes_criterion_val
    else:
        criterion = torch.nn.MSELoss()

    running_loss = 0
    nr_val_data = len(loader)
    for i, (input_data, ground_truth) in enumerate(loader):  # tqdm.tqdm(loader, desc="val"):
        # if isinstance(input_data, torch_geometric.data.Data):
        #     input_data = input_data.to(device)
        #     ground_truth = input_data.y
        # else:
        #     input_data, ground_truth = input_data
        input_data = input_data.to(device)
        ground_truth = ground_truth.to(device)

        model.eval()
        output = model(input_data)

        loss = criterion(output[:, :, s_x:-e_x, s_y:-e_y] * 255, ground_truth[:, :, s_x:-e_x, s_y:-e_y] * 255)
        # print(f"eval {i+1}/{nr_val_data} Loss {loss.item()}")
        running_loss += loss.item()
    return running_loss / nr_val_data  # len(loader) if len(loader) > 0 else running_loss


def train_ignite(device, epochs, loss, optimizer, train_loader, val_loader, train_model, checkpoint_name=""):
    # Validator
    validation_evaluator = create_supervised_evaluator(train_model, metrics={"val_loss": Loss(loss)}, device=device)
    # Trainer
    trainer = create_supervised_trainer(train_model, optimizer, loss, device=device)
    train_evaluator = create_supervised_evaluator(train_model, metrics={"loss": Loss(loss)}, device=device)
    run_id = binascii.hexlify(os.urandom(15)).decode("utf-8")
    artifacts_path = os.path.join(os.path.curdir, f"artifacts/{run_id}")
    logs_path = os.path.join(artifacts_path, "tensorboard")
    checkpoints_dir = os.path.join(os.path.curdir, "checkpoints" + checkpoint_name)
    RunningAverage(output_transform=lambda x: x).attach(trainer, name="loss")
    pbar = ProgressBar(persist=True, bar_format="{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}{postfix} [{elapsed}<{remaining}]{rate_fmt}")
    pbar.attach(trainer, metric_names="all")

    @trainer.on(Events.EPOCH_STARTED)  # noqa
    def log_epoch_start(engine: Engine):
        logging.info(f"Started epoch {engine.state.epoch}")
        logging.info(system_status())

    @trainer.on(Events.EPOCH_COMPLETED)  # noqa
    def log_epoch_summary(engine: Engine):
        # Training
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        train_avg_loss = metrics["loss"]

        # Validation
        validation_evaluator.run(val_loader)
        metrics = validation_evaluator.state.metrics
        val_avg_loss = metrics["val_loss"]

        msg = f"Epoch summary for epoch {engine.state.epoch}: loss: {train_avg_loss:.4f}, val_loss: {val_avg_loss:.4f}\n"
        pbar.log_message(msg)
        logging.info(msg)
        logging.info(system_status())

    tb_logger = TensorboardLogger(log_dir=logs_path)
    tb_logger.attach(trainer, log_handler=GradsHistHandler(train_model), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach_output_handler(
        train_evaluator, event_name=Events.EPOCH_COMPLETED, tag="train", metric_names=["loss"], global_step_transform=global_step_from_engine(trainer)
    )
    tb_logger.attach_output_handler(
        validation_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="validation",
        metric_names=["val_loss"],
        global_step_transform=global_step_from_engine(trainer),
    )
    to_save = {"train_model": train_model, "optimizer": optimizer}
    checkpoint_handler = Checkpoint(to_save, DiskSaver(checkpoints_dir, create_dir=True, require_empty=False), n_saved=1)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
    # Run Training
    logging.info("Start training on %s for %s epochs", device, epochs)
    logging.info(f"tensorboard --logdir={artifacts_path}")
    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_str", type=str, help="One of configurations, e.g. 'unet'.", default="unet", required=False)
    parser.add_argument("--resume_checkpoint", type=str, help="torch pt file to be re-loaded.", default=None, required=False)
    parser.add_argument("--data_raw_path", type=str, help="Base dir of raw data", default="./data/raw")
    parser.add_argument("--submit", action="store_true", default=False, help="Create submission")
    parser.add_argument(
        "--data_compressed_path",
        type=str,
        help="If given, data is extracted from this location if no data at  data_raw_path  Standard layout: ./data/compressed",
        default=None,
    )
    parser.add_argument("-log", "--loglevel", default="info", help="Provide logging level. Example --loglevel debug, default=warning")
    parser.add_argument("--random_seed", type=int, default=123, required=False, help="Seed for shuffling the dataset.")
    parser.add_argument("--train_fraction", type=float, default=0.9, required=False, help="Fraction of the data set for training.")
    parser.add_argument("--val_fraction", type=float, default=0.1, required=False, help="Fraction of the data set for validation.")
    parser.add_argument("--batch_size", type=int, default=5, required=False, help="Batch Size for training and validation.")
    parser.add_argument("--num_workers", type=int, default=0, required=False, help="Number of workers for data loader.")
    parser.add_argument("--epochs", type=int, default=20, required=False, help="Number of epochs to train.")
    parser.add_argument("--file_filter", type=str, default=None, required=False, help='Filter files in the dataset. Defaults to "**/*8ch.h5"')
    parser.add_argument("--limit", type=int, default=None, required=False, help="Cap dataset size at this limit.")
    parser.add_argument("--val_limit", type=int, default=None, required=False, help="Cap dataset size at this limit.")
    parser.add_argument("--device", type=str, default=None, required=False, help="Force usage of device.")
    parser.add_argument("--stride", type=int, default=30, required=False, help="Stride for submission")
    parser.add_argument("--radius", type=int, default=50, required=False, help="Radius for patching")
    parser.add_argument("--train_city", type=str, default=None, required=False, help="Training data city")
    parser.add_argument("--static_map", action="store_true", default=False, help="Use static map?")
    parser.add_argument(
        "--device_ids", nargs="*", default=None, required=False, help="Whitelist of device ids. If not given, all device ids are taken."
    )
    parser.add_argument("--data_parallel", default=False, required=False, help="Use DataParallel.", action="store_true")
    parser.add_argument("--num_tests_per_file", default=100, type=int, required=False, help="Number of test slots per file")
    parser.add_argument("--checkpoint_name", default="", type=str, required=False, help="How to name the checkpoint")
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default=None,
        required=False,
        help='If given, submission is evaluated from ground truth zips "ground_truth_[spatio]tempmoral.zip" from this directory.',
    )
    parser.add_argument(
        "--submission_output_dir",
        type=str,
        default=None,
        required=False,
        help="If given, submission is stored to this directory instead of current.",
    )

    return parser


def main(args):
    parser = create_parser()
    args = parser.parse_args(args)

    t4c_apply_basic_logging_config()

    model_str = args.model_str
    resume_checkpoint = args.resume_checkpoint

    device = args.device

    logging.info("Start build dataset")
    # Data set
    dataset_config = configs[model_str].get("dataset_config", {})
    dataset_class = dataset_config.get("dataset", T4CDataset)
    dataset_config.pop("dataset", None)  # delete key if it exists
    dataset_config["use_static_map"] = args.static_map

    data_raw_path = args.data_raw_path
    file_filter = args.file_filter

    assert args.num_workers == 0, "Check seed problems for more workers before commenting this"

    geometric = configs[model_str].get("geometric", False)
    if data_raw_path is not None:
        logging.info("Check if files need to be untarred...")
        if args.data_compressed_path is not None:
            tar_files = list(Path(args.data_compressed_path).glob("**/*.tar"))
            logging.info("Going to untar %s tar balls to %s. ", len(tar_files), data_raw_path)
            untar_files(files=tar_files, destination=data_raw_path)
            logging.info("Done untar %s tar balls to %s.", len(tar_files), data_raw_path)

    assert args.stride <= 2 * args.radius, "stride must cover data"
    if args.epochs > 0:
        # Make one dataset for training and a separate one for validation
        train_dataset = dataset_class(root_dir=data_raw_path, auto_filter="train", **dataset_config, limit=args.limit, radius=args.radius,)
        val_dataset = dataset_class(
            root_dir=data_raw_path, auto_filter="test", **dataset_config, limit=args.val_limit, radius=args.radius, augment=False,
        )
        # if geometric:
        #     dataset = T4CGeometricDataset(root=str(Path(data_raw_path).parent), file_filter=file_filter, num_workers=args.num_workers, **dataset_config)
        # else:
        #     dataset = T4CDataset(root_dir=data_raw_path, file_filter=file_filter, **dataset_config)
        logging.info("Dataset has size %s", len(train_dataset))
        assert len(train_dataset) > 0

    # Model
    logging.info("Create train_model.")
    model_class = configs[model_str]["model_class"]
    model_config = configs[model_str].get("model_config", {})
    if args.static_map:
        model_config["in_channels"] += 9
    model = model_class(**model_config)
    if not model_str.startswith("naive"):
        dataloader_config = configs[model_str].get("dataloader_config", {})
        optimizer_config = configs[model_str].get("optimizer_config", {})
        if resume_checkpoint is not None:
            # CODE for using one model per city (make dictionary):
            # ckpt_list = [
            #     "ckpt_berlin_up_2_best.pt",
            #     "ckpt_istanbul_up_2_best.pt",
            #     "ckpt_melbourne_up/epoch_0199.pt",
            #     "ckpt_chicago_up/epoch_0199.pt",
            # ]
            # model = {}
            # for ckpt_city, city in zip(ckpt_list, ["BERLIN", "ISTANBUL", "MELBOURNE", "CHICAGO"]):
            #     logging.info("Reload checkpoint %s", ckpt_city)
            #     model_city = model_class(**model_config)
            #     load_torch_model_from_checkpoint(checkpoint=ckpt_city, model=model_city)
            #     model[city] = model_city
            logging.info("Reload checkpoint %s", resume_checkpoint)
            load_torch_model_from_checkpoint(checkpoint=resume_checkpoint, model=model)

        if args.epochs > 0:
            logging.info("Going to run train_model.")
            logging.info(system_status())
            padding = dataset_config["transform"].keywords["zeropad2d"]
            _, device = run_model(
                train_model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                dataloader_config=dataloader_config,
                optimizer_config=optimizer_config,
                geometric=geometric,
                padding=padding,
                **(vars(args)),
            )

    if args.submit:
        competitions = ["spatiotemporal"]

        for competition in competitions:
            additional_args = {"radius": args.radius, "stride": args.stride, "static_map": args.static_map}
            if geometric:
                processed_dir = str(Path(data_raw_path).parent)
                additional_args = {
                    "gt": GraphTransformer(processed_dir=processed_dir, raw_dir=data_raw_path, batch_size=1),
                    "processed_dir": processed_dir,
                }
            submission = package_submission(
                data_raw_path=data_raw_path,
                competition=competition,
                model=model,
                model_str=model_str,
                device=device,
                h5_compression_params={"compression_level": None},
                submission_output_dir=Path(args.submission_output_dir if args.submission_output_dir is not None else "."),
                # batch mode for submission
                batch_size=1,  #  if geometric else args.batch_size,
                num_tests_per_file=args.num_tests_per_file,
                submission_file_name=args.checkpoint_name,
                **additional_args,
            )
            ground_truth_dir = args.ground_truth_dir
            if ground_truth_dir is not None:
                ground_truth_dir = Path(ground_truth_dir)
                scorecomp.score_participant(
                    ground_truth_archive=str(ground_truth_dir / f"ground_truth_{competition}.zip"), input_archive=str(submission)
                )
            else:
                scorecomp.verify_submission(input_archive=submission, competition=competition)


if __name__ == "__main__":
    main(sys.argv[1:])
