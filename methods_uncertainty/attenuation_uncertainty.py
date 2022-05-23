import torch
import numpy as np

from baselines.baselines_configs import configs
from methods_uncertainty.unet_uncertainty import UnetBasedUncertainty

class AttenuationUncertainty(UnetBasedUncertainty):
    def __init__(self, model_path, device="cuda", model_str="bayes_unet", num_channels=8, **kwargs) -> None:
        super().__init__(model_path, device, model_str, **kwargs)

        self.num_channels = num_channels

    def __call__(self, x_hour):
        # pretransform
        inp_data = self.pre_transform(np.expand_dims(x_hour, 0), from_numpy=True, batch_dim=True)

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

        mu_out = np.clip(torch.movedim(mu_out, 1, 3).detach().numpy(), 0, 255)
        sig_out = torch.movedim(sig_out, 1, 3).detach().numpy()

        return mu_out, sig_out
