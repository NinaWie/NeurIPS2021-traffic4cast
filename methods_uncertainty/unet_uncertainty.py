import torch
import numpy as np
from baselines.baselines_configs import configs

class UnetBasedUncertainty:
    def __init__(self, model_path, device="cuda", model_str="unet", **kwargs) -> None:
        self.device = device
        self.model_str = model_str

        # load model
        self.model = self.load_model(model_path)
        self.model = self.model.to(device)
        self.model.eval()

        # pre and post transform:
        self.pre_transform = configs[self.model_str]["pre_transform"]
        self.post_transform = configs[self.model_str]["post_transform"]


    def load_model(self, path, use_static_map=False):
        model_class = configs[self.model_str]["model_class"]
        model_config = configs[self.model_str].get("model_config", {})
        model = model_class(**model_config)
        loaded_dict = torch.load(path, map_location=torch.device("cpu"))
        if self.model_str == "alex_unet":
            new_state_dict = {}
            for key, val in loaded_dict["model"].items():
                new_state_dict[key[7:]] = val
        else:
            new_state_dict = loaded_dict["model"]
        # print("loaded model from epoch", loaded_dict["epoch"])
        model.load_state_dict(new_state_dict)
        return model


class TrivialUnetUncertainty(UnetBasedUncertainty):

    def __call__(self, x_hour):

        # pretransform
        inp_data = self.pre_transform(np.expand_dims(x_hour,0), from_numpy=True, batch_dim=True)

        pred = self.model(inp_data.to(self.device)).detach().cpu()

        # post transform
        out = self.post_transform(pred).detach().numpy()
        out = np.clip(out, 0, 255)
        return out[0], out[0] + .001
