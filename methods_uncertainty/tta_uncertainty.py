from tkinter import W
import torch
import numpy as np
import torch.nn as nn
from functools import partial
import torchvision.transforms as tf
import torchvision.transforms.functional as TF

from baselines.baselines_configs import configs
from methods_uncertainty.unet_uncertainty import UnetBasedUncertainty

class TTAUncertainty(UnetBasedUncertainty):

    def __init__(self, model_path, device="cuda", model_str="bayes_unet", num_channels=8, **kwargs) -> None:
        super().__init__(model_path, device, model_str, **kwargs)

        self.augmentor = DataAugmentation()

    def __call__(self, x_hour):

        # pretransform
        inp_data = self.pre_transform(np.expand_dims(x_hour,0), from_numpy=True, batch_dim=True)

        # TODO: apply augmentations here 
        # inp_data: shape (1, 96, 496. 436)
        augmented_inp_data = self.augmentor.transform(inp_data)
        # augmented_inp_data: shape (8, 96, 496. 436) (where 8 = num augmentations)

        pred = self.model(augmented_inp_data.to(self.device)).detach().cpu()

        pred_deaugmented = self.augmentor.detransform(pred)

        # post transform
        out = self.post_transform(pred_deaugmented, normalize=True, batch_dim=True).detach().numpy()

        # summarize augmentations
        mean_out = np.mean(out, axis=0)
        std_out = np.std(out, axis=0) # uncertainty = std over augmentation predictions

        return mean_out, std_out


class DataAugmentation:
    """
    https://github.com/alextimans/t4c2021-uncertainty-thesis/blob/main/uq/data_augmentation.py
    """
    def __init__(self):
        self.transformations = [
            TF.vflip,
            TF.hflip,
            partial(TF.rotate, angle=90, expand=True),
            # partial(TF.rotate, angle=180, expand=True),
            partial(TF.rotate, angle=270, expand=True),
            # tf.Compose([TF.vflip, partial(TF.rotate, angle=90, expand=True)]),
            # tf.Compose([TF.vflip, partial(TF.rotate, angle=-90, expand=True)])
            ]

        self.detransformations = [
            TF.vflip,
            TF.hflip,
            partial(TF.rotate, angle=-90, expand=True),
            # partial(TF.rotate, angle=-180, expand=True),
            partial(TF.rotate, angle=-270, expand=True),
            # tf.Compose([partial(TF.rotate, angle=-90, expand=True), TF.vflip]),
            # tf.Compose([partial(TF.rotate, angle=90, expand=True), TF.vflip])
            ]

        self.nr_augments = len(self.transformations)

        self.padder = None

    def transform(self, data: torch.Tensor) -> torch.Tensor:

        """
        Receives X = (1, 12 * Ch, H, W) and does k augmentations 
        returning X' = (1+k, 12 * Ch, H, W).
        """
        if self.padder is None:
            _, _, h, w = data.size()
            pad_quadratic = abs(h-w)
            if h > w:
                self.padder = nn.ZeroPad2d((0, pad_quadratic))
            else:
                self.padder = nn.ZeroPad2d((pad_quadratic, 0))

        data_padded = self.padder(data)
        X = data_padded.clone()
        for transform in self.transformations:
            X_aug = transform(data_padded)
            X = torch.cat((X, X_aug), dim=0)
        assert list(X.shape) == [1+self.nr_augments] + list(data_padded.shape[1:])

        return X

    def detransform(self, data: torch.Tensor) -> torch.Tensor:

        """
        Receives y_pred = (1+k, 6 * Ch, H, W), detransforms the 
        k augmentations and returns y_pred = (1+k, 6 * Ch, H, W).
        """

        for i, detransform in enumerate(self.detransformations):
            y_deaug = detransform(data[i+1, ...].unsqueeze(dim=0))
            # remove zero padding
            _, _, h, w = y_deaug.size()
            (h_new, w_new) = (h - self.padder.padding[0], w - self.padder.padding[1])
            y_deaug_cropped = y_deaug[:, :, :h_new, :w_new]
            # cat
            if i ==0:
                y = y_deaug_cropped
            else:
                y = torch.cat((y, y_deaug_cropped), dim=0)

        return y
