import torch
import numpy as np
import torchvision.transforms as tf
import torchvision.transforms.functional as TF
from functools import partial

from methods_uncertainty.patch_uncertainty import PatchUncertainty
from methods_uncertainty.tta_uncertainty import DataAugmentation

class PatchTTAUncertainty(PatchUncertainty):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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


    def process_patches(self, inp_patch, avg_arr, index_arr):
        """Overwriting method to augment the input data"""

        # inp_patch: torch array size (nrpatches, ...)
        # avg_arr: np array size (nr_patches, ...)
        # tile index and avg arr
        avg_arr = avg_arr * (self.nr_augments + 1)
        index_arr = np.tile(index_arr, (self.nr_augments+1, 1))

        collect_outputs = self.predict_patches(inp_patch)

        for t_ind in range(self.nr_augments):
            # transform
            transform  = self.transformations[t_ind]
            augmented_inp_data = transform(inp_patch)
            # predict
            pred_augmented = self.predict_patches(augmented_inp_data)
            # detransform
            pred_deaugmented = self.detransformations[t_ind](pred_augmented)
            # cat
            collect_outputs = torch.cat((collect_outputs, pred_deaugmented), dim=0)

        return collect_outputs, avg_arr, index_arr

        