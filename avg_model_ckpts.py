import os
import torch
import numpy as np

from baselines.checkpointing import save_torch_model_to_checkpoint
from baselines.baselines_configs import configs

out_dir = "ckpt_averaging_patch2"
path_models = "ckpt_patch_2"
use_models = ["best.pt", "epoch_0349.pt", "epoch_0399.pt", "epoch_0449.pt", "epoch_0499.pt"]
# use_models = ["best.pt", "epoch_0549.pt", "epoch_0599.pt", "epoch_0649.pt", "epoch_0699.pt"]
model_str = "up_patch"
radius = 50

model_class = configs[model_str]["model_class"]
model_config = configs[model_str].get("model_config", {})
model = model_class(**model_config, img_len=2 * radius)


new_weight_dict = {}
for i, mod_name in enumerate(use_models):
    print("load weights of model", mod_name)
    loaded_dict = torch.load(os.path.join(path_models, mod_name), map_location=torch.device("cpu"))["model"]
    if i == 0:
        new_weight_dict = loaded_dict
    else:
        for key in loaded_dict.keys():
            new_weight_dict[key] = new_weight_dict[key] + loaded_dict[key]


for key in new_weight_dict.keys():
    new_weight_dict[key] = new_weight_dict[key] / len(use_models)

model.load_state_dict(new_weight_dict)

os.makedirs(out_dir)
save_torch_model_to_checkpoint(model=model, epoch=0, out_dir=out_dir)
