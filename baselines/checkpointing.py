import datetime
import os

import torch


def load_torch_model_from_checkpoint(checkpoint: str, model: torch.nn.Module):
    map_location = None
    if not torch.cuda.is_available():
        map_location = "cpu"

    state_dict = torch.load(checkpoint, map_location=map_location)
    if "model" in state_dict.keys():
        model.load_state_dict(state_dict["model"])
    elif "train_model" in state_dict.keys():
        model.load_state_dict(state_dict["train_model"])
    else:
        model.load_state_dict(state_dict)


def save_torch_model_to_checkpoint(
    model: torch.nn.Module, out_dir: str, epoch: int, out_name: str = "best",
):
    tstamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    outpath = os.path.join(out_dir, out_name + ".pt")  # f"{epoch:04}_{tstamp}.pt")

    save_dict = {"epoch": epoch, "model": model.state_dict()}

    torch.save(save_dict, outpath)
