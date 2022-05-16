import torch
import numpy as np
from baselines.baselines_configs import configs


class UnetUncertainty:
    def __init__(self, model_path, device="cuda", model_str="unet", **kwargs) -> None:
        self.device = device
        self.model_str = model_str

        # load model
        self.model = self.load_model(model_path, use_static_map=(self.static_map is not None))
        self.model = self.model.to(device)
        self.model.eval()

    def load_model(self, path, use_static_map=False):
        model_class = configs[self.model_str]["model_class"]
        model_config = configs[self.model_str].get("model_config", {})
        model = model_class(**model_config)
        loaded_dict = torch.load(path, map_location=torch.device("cpu"))
        # print("loaded model from epoch", loaded_dict["epoch"])
        model.load_state_dict(loaded_dict["model"])
        return model

    def __call__(self, x_hour):

        # pretransform
        pre_transform = configs[self.model_str]["pre_transform"]
        inp_data = pre_transform(x_hour, from_numpy=True, batch_dim=True)

        pred = self.model(inp_data).detach().cpu()

        # post transform
        post_transform = configs[self.model_str]["post_transform"]
        out = post_transform(pred, normalize=True).detach().numpy()
        return out, out
