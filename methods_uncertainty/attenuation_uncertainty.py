import torch
import numpy as np
from baselines.baselines_configs import configs


class AttenuationUncertainty:
    def __init__(self, model_path, device="cuda", model_str="bayes_unet", num_channels=8, **kwargs) -> None:
        self.device = device
        self.model_str = model_str
        self.num_channels = num_channels

        # load model
        self.model = self.load_model(model_path)
        self.model = self.model.to(device)
        self.model.eval()

    def load_model(self, path):
        model_class = configs[self.model_str]["model_class"]
        model_config = configs[self.model_str].get("model_config", {})
        model = model_class(**model_config)
        loaded_dict = torch.load(path, map_location=torch.device("cpu"))
        # print("loaded model from epoch", loaded_dict["epoch"])
        model.load_state_dict(loaded_dict["model"])
        # load post transform
        self.post_transform = configs[self.model_str]["post_transform"]
        return model

    def __call__(self, x_hour):
        # pretransform
        pre_transform = configs[self.model_str]["pre_transform"]
        inp_data = pre_transform(np.expand_dims(x_hour, 0), from_numpy=True, batch_dim=True)

        pred = self.model(inp_data.to(self.device)).detach().cpu()

        # get the mean and variance - first reshape
        bs, ts_ch, xsize, ysize = pred.size()
        num_time_steps = int(ts_ch / self.num_channels)
        # (k, 12 * 8, 495, 436) -> (k, 6, 8, 2, 495, 436)
        pred_unstacked = torch.reshape(pred, (bs, num_time_steps // 2, self.num_channels, 2, xsize, ysize))
        # mu and sigma
        mu = pred_unstacked[:, :, :, 0]
        log_sig = pred_unstacked[:, :, :, 1]
        sig = torch.exp(log_sig)

        # post transform and return mu and sigma
        mu_out = self.post_transform(torch.squeeze(mu), normalize=True, stack_channels_on_time=False)
        sig_out = self.post_transform(torch.squeeze(sig), normalize=True, stack_channels_on_time=False)

        mu_out = torch.movedim(mu_out, 1, 3)
        sig_out = torch.movedim(sig_out, 1, 3)

        return mu_out.detach().numpy(), sig_out.detach().numpy()
