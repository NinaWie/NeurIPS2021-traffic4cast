#  Copyright 2021 Institute of Advanced Research in Artificial Intelligence (IARAI) GmbH.
#  IARAI licenses this file to You under the Apache License, Version 2.0
#  (the "License"); you may not use this file except in compliance with
#  the License. You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import time
import numpy as np
import torch
from torch.utils.data import Dataset

from competition.competition_constants import MAX_TEST_SLOT_INDEX
from competition.prepare_test_data.prepare_test_data import prepare_test
from util.h5_util import load_h5_file


class T4CDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        auto_filter: str = "train",
        file_filter: str = None,
        test_city="ANTWERP",
        limit: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_npy: bool = False,
        **kwargs,
    ):
        """torch dataset from training data.

        Parameters
        ----------
        root_dir
            data root folder, by convention should be `data/raw`, see `data/README.md`. All `**/training/*8ch.h5` will be added to the dataset.
        file_filter: str
            filter files under `root_dir`, defaults to `"**/training/*ch8.h5`
        limit
            truncate dataset size
        transform
            transform applied to both the input and label
        """
        self.root_dir = root_dir
        self.limit = limit
        self.files = []
        self.use_npy = use_npy
        self.transform = transform
        if file_filter is not None:
            self.file_filter = file_filter
        else:
            print("USING AUTO FILTER")
            if auto_filter == "train":
                self.file_filter = "**/training/*8ch.h5"
            elif auto_filter == "test":
                self.file_filter = f"**/training/*{test_city}*8ch.h5"
            print(self.file_filter)
        self._load_dataset()
        if auto_filter == "train" and (file_filter is None):
            self.files = [f for f in self.files if not (test_city in str(f))]

        print("auto filter: ", auto_filter, "examples:", np.random.choice(self.files, 20))

    def _load_dataset(self):
        self.files = list(Path(self.root_dir).rglob(self.file_filter))

    def _load_h5_file(self, fn, sl: Optional[slice]):
        if self.use_npy:
            return np.load(fn)
        else:
            return load_h5_file(fn, sl=sl)

    def __len__(self):
        size_240_slots_a_day = len(self.files) * MAX_TEST_SLOT_INDEX
        if self.limit is not None:
            return min(size_240_slots_a_day, self.limit)
        return size_240_slots_a_day

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if idx > self.__len__():
            raise IndexError("Index out of bounds")

        file_idx = int(np.random.rand() * len(self.files))
        start_hour = int(np.random.rand() * MAX_TEST_SLOT_INDEX)
        # For testing with __main__ below
        # print(self.files[file_idx], start_hour)
        # return None

        two_hours = self._load_h5_file(self.files[file_idx], sl=slice(start_hour, start_hour + 12 * 2 + 1))

        input_data, output_data = prepare_test(two_hours)

        input_data = self._to_torch(input_data)
        output_data = self._to_torch(output_data)

        if self.transform is not None:
            input_data = self.transform(input_data)
            output_data = self.transform(output_data)

        return input_data, output_data

    def _to_torch(self, data):
        data = torch.from_numpy(data)
        data = data.to(dtype=torch.float)
        return data


class PatchT4CDataset(T4CDataset):
    def __init__(
        self,
        root_dir: str,
        file_filter: str = None,
        limit: Optional[int] = 100,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_npy: bool = False,
        use_per_file=10,
        radius=50,
        auto_filter: str = "train",
        **kwargs,
    ):
        super().__init__(root_dir, file_filter=file_filter, auto_filter=auto_filter, limit=limit, transform=transform, use_npy=use_npy)

        self.n_load_files = limit  # The number of loaded files depends on whether we have train or test
        self.use_per_file = use_per_file  # use per file is fixed, we have to see how much it falsifies the val acc
        # We always run 2 epochs if we are at train time
        if auto_filter == "train":
            self.resample_every_x_epoch = 2
        else:
            self.resample_every_x_epoch = 1

        self.internal_counter = 0
        self.auto_filter = auto_filter
        self.radius = radius

        self.data_x, self.data_y = self._cache_data()

    def _cache_data(self):
        print("\n ---------- ", self.internal_counter, self.auto_filter, "MAKE NEW DATASET -------------")
        use_files = np.random.choice(self.files, size=self.n_load_files, replace=False)
        data_x = np.zeros((self.n_load_files * self.use_per_file, 12, 2 * self.radius, 2 * self.radius, 8))
        data_y = np.zeros((self.n_load_files * self.use_per_file, 6, 2 * self.radius, 2 * self.radius, 8))
        # print("allocated:", data_x.shape, data_y.shape)
        counter = 0
        for file in use_files:
            loaded_file = load_h5_file(file)
            random_times = (np.random.rand(self.use_per_file) * 264).astype(int)
            rand_x = (np.random.rand(self.use_per_file) * (495 - 2 * self.radius)).astype(int) + self.radius
            rand_y = (np.random.rand(self.use_per_file) * (436 - 2 * self.radius)).astype(int) + self.radius
            # print("loaded file ", file, loaded_file.shape, random_times)
            for i in range(self.use_per_file):
                start_hour = random_times[i]
                end_hour = start_hour + 24
                s_x, e_x = (rand_x[i] - self.radius, rand_x[i] + self.radius)  # start and end x of patch
                s_y, e_y = (rand_y[i] - self.radius, rand_y[i] + self.radius)  # start and end y of patch
                two_hours = loaded_file[start_hour:end_hour, s_x:e_x, s_y:e_y]
                # print("two hours", start_hour, s_x, s_y, two_hours.shape)
                # print("two hours", two_hours.shape)
                # self._load_h5_file(self.files[file_idx], sl=slice(start_hour, start_hour + 12 * 2 + 1))
                input_data, output_data = prepare_test(two_hours)

                # print("inp and outp", input_data.shape, output_data.shape)

                data_x[counter] = input_data
                data_y[counter] = output_data
                counter += 1

        data_x = self._to_torch(data_x)
        data_y = self._to_torch(data_y)

        if self.transform is not None:
            data_x = self.transform(data_x)
            data_y = self.transform(data_y)

        # print("inp and outp after transform", data_x.size(), data_y.size())
        return data_x, data_y

    def one_img_cache_data(self):
        """
        Test function for a single big file to create the patches
        """
        print("\n ---------- SPECIAL DATASET -------------")
        use_file = np.random.choice(self.files, 1, replace=False)[0]
        some_hour = int(np.random.rand() * 240)
        test_arr = load_h5_file(use_file)
        x_hour = test_arr[some_hour : some_hour + 12]
        test_out_gt_inds = np.add([1, 2, 3, 6, 9, 12], 11 + some_hour)
        y_hour = test_arr[test_out_gt_inds]

        data_x, _, _ = create_patches(x_hour)
        data_y, _, _ = create_patches(y_hour)
        print(data_x.shape)
        print(data_y.shape)

        data_x = self._to_torch(data_x)
        data_y = self._to_torch(data_y)

        if self.transform is not None:
            data_x = self.transform(data_x)
            data_y = self.transform(data_y)

        # print("inp and outp after transform", data_x.size(), data_y.size())
        return data_x, data_y

    def __len__(self):
        return self.n_load_files * self.use_per_file

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        self.internal_counter += 1
        # print(self.internal_counter, idx)
        if self.internal_counter % (len(self) * self.resample_every_x_epoch) == 0:
            self.data_x, self.data_y = self._cache_data()
        return self.data_x[idx], self.data_y[idx]


if __name__ == "__main__":
    dataset = PatchT4CDataset("data/raw", auto_filter="train", limit=3)
    from torch.utils.data import DataLoader

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    for epoch in range(10):
        print("NEW EPOCH")
        for d_x, d_y in train_loader:
            print(d_x.shape, d_y.shape)
