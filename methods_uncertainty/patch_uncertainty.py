import torch
import numpy as np
from baselines.baselines_configs import configs
from competition.submission.submission import create_patches, stitch_patches


def get_std(out_patch, means, nr_samples_arr, index_arr):
    """
    out_patch: collection of patches
    nr_samples_arr: Number of samples available per cell
    index_arr: indices of start and end of each patch
    """
    xlen, ylen = nr_samples_arr.shape

    std_prediction = np.zeros((6, xlen, ylen, 8))
    assert len(out_patch) == len(index_arr)

    for i in range(len(out_patch)):
        (start_x, end_x, start_y, end_y) = tuple(index_arr[i].astype(int).tolist())
        std_prediction[:, start_x:end_x, start_y:end_y] += (out_patch[i] - means[:, start_x:end_x, start_y:end_y]) ** 2

    expand_nr_samples_arr = np.tile(np.expand_dims(nr_samples_arr, 2), 8)

    avg_prediction = np.sqrt(std_prediction / expand_nr_samples_arr)
    return avg_prediction


def std_v1(out_patch, index_arr, out_shape):
    std_preds = np.zeros(out_shape)

    for p_x in range(495):
        for p_y in range(436):
            pixel = (p_x, p_y)
            # find the patches corresponding to this pixel and get the
            # middleness and pred per patch
            preds = []
            for j, inds in enumerate(index_arr):
                x_s, x_e, y_s, y_e = inds
                if x_s <= pixel[0] and x_e > pixel[0] and y_s <= pixel[1] and y_e > pixel[1]:
                    rel_x, rel_y = int(pixel[0] - x_s), int(pixel[1] - y_s)
                    # what values were predicted for this pixel?
                    pred_pixel = out_patch[j, :, rel_x, rel_y, :]
                    # how much in the middle is a pixel?
                    preds.append(pred_pixel)
            std_preds[:, pixel[0], pixel[1], :] = np.std(preds, axis=0)
    return std_preds


class PatchUncertainty:
    def __init__(self, model_path, static_map_arr=None, device="cuda", radius=50, stride=30, model_str="up_patch", **kwargs) -> None:
        self.device = device
        self.radius = radius
        self.model_str = model_str
        self.stride = stride

        # static map:
        self.static_map = static_map_arr

        # load model
        self.model = self.load_model(model_path, use_static_map=(self.static_map is not None))
        self.model = self.model.to(device)
        self.model.eval()

    def load_model(self, path, use_static_map=False):
        model_class = configs[self.model_str]["model_class"]
        model_config = configs[self.model_str].get("model_config", {})
        if use_static_map:
            model_config["in_channels"] += 9
        model = model_class(**model_config, img_len=2 * self.radius)
        loaded_dict = torch.load(path, map_location=torch.device("cpu"))
        # print("loaded model from epoch", loaded_dict["epoch"])
        model.load_state_dict(loaded_dict["model"])
        return model

    def __call__(self, x_hour):
        # divide into patches
        if self.static_map is not None:
            patch_collection, avg_arr, index_arr, data_static = create_patches(
                x_hour, radius=self.radius, stride=self.stride, static_map=self.static_map
            )
        else:
            patch_collection, avg_arr, index_arr = create_patches(x_hour, radius=self.radius, stride=self.stride)
        # print("Number of patches per cell", np.mean(avg_arr), np.median(avg_arr))

        # pretransform
        pre_transform = configs[self.model_str]["pre_transform"]
        inp_patch = pre_transform(patch_collection, from_numpy=True, batch_dim=True)

        if self.static_map is not None:
            data_static = pre_transform(np.expand_dims(data_static, axis=-1), from_numpy=True, batch_dim=True)
            inp_patch = torch.cat((inp_patch, data_static), dim=1)

        # run - batch if it's too big
        internal_batch_size = 50
        n_samples = inp_patch.size()[0]
        img_len = inp_patch.size()[2]
        out = torch.zeros(n_samples, 48, img_len, img_len)
        e_b = 0
        for j in range(n_samples // internal_batch_size):
            s_b = j * internal_batch_size
            e_b = (j + 1) * internal_batch_size
            batch_patch = inp_patch[s_b:e_b].to(self.device)
            # print(system_status())
            out[s_b:e_b] = self.model(batch_patch).detach().cpu()
        if n_samples % internal_batch_size != 0:
            last_batch = inp_patch[e_b:].to(self.device)
            out[e_b:] = self.model(last_batch).detach().cpu()
        # out = model(inp_patch)

        # post transform
        post_transform = configs[self.model_str]["post_transform"]
        out_patch = post_transform(out, normalize=True).detach().numpy()

        # Predictions: average over patches
        pred = stitch_patches(out_patch, avg_arr, index_arr)
        # Uncertainty: std over patches
        std_preds = get_std(out_patch, pred, avg_arr, index_arr)

        # return prediction and uncertainty scores (size (6, 436, 495, 8))
        # print(pred.shape, std_preds.shape)
        return pred, std_preds
