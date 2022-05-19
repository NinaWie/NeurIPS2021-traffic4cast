import torch
import numpy as np
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
            partial(TF.rotate, angle=180, expand=True),
            partial(TF.rotate, angle=270, expand=True),
            tf.Compose([TF.vflip, partial(TF.rotate, angle=90, expand=True)]),
            tf.Compose([TF.vflip, partial(TF.rotate, angle=-90, expand=True)])
            ]

        self.detransformations = [
            TF.vflip,
            TF.hflip,
            partial(TF.rotate, angle=-90, expand=True),
            partial(TF.rotate, angle=-180, expand=True),
            partial(TF.rotate, angle=-270, expand=True),
            tf.Compose([partial(TF.rotate, angle=-90, expand=True), TF.vflip]),
            tf.Compose([partial(TF.rotate, angle=90, expand=True), TF.vflip])
            ]

        self.nr_augments = len(self.transformations)

    def transform(self, data: torch.Tensor) -> torch.Tensor:

        """
        Receives X = (1, 12*8, 496, 448) and does k augmentations 
        returning X' = (1+k, 12*8, 496, 448)
        """

        X = data
        for transform in self.transformations:
            X_aug = transform(data)
            X = torch.cat((X, X_aug), dim=0)

        assert list(X.shape) == [self.nr_augments+1] + list(data.shape[1:])

        return X

    def detransform(self, data: torch.Tensor) -> torch.Tensor:

        """
        Receives y_pred = (1+k, 6*8, 496, 448), detransforms the 
        k augmentations and returns y_pred = (1+k, 6*8, 496, 448)
        """

        y = data[0, ...].unsqueeze(dim=0)
        for i, detransform in enumerate(self.detransformations):
            y_deaug = detransform(data[i+1, ...].unsqueeze(dim=0))
            y = torch.cat((y, y_deaug), dim=0)

        assert y.shape == data.shape

        return y
