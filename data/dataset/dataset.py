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
        file_filter: str = None,
        limit: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_npy: bool = False,
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
        self.file_filter = file_filter
        self.use_npy = use_npy
        if self.file_filter is None:
            self.file_filter = "**/training/*8ch.h5"
            if self.use_npy:
                self.file_filter = "**/training_npy/*.npy"
        self.transform = transform
        self._load_dataset()

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

        file_idx = int(np.random.rand() * len(self.files)) # idx // MAX_TEST_SLOT_INDEX
        start_hour = int(np.random.rand() * MAX_TEST_SLOT_INDEX) # idx % MAX_TEST_SLOT_INDEX

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


class CachedT4CDataset(T4CDataset):
    def __init__(
        self,
        root_dir: str,
        file_filter: str = None,
        limit: Optional[int] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        use_npy: bool = False,
        load_files=10,
        use_per_file=5,
    ):
        super().__init__(root_dir, file_filter=file_filter, limit=limit, transform=transform, use_npy=use_npy)

        self.load_files = load_files
        self.use_per_file = use_per_file
        # cheat: just count accesses
        self.epoch_size_internal = 10
        self.internal_counter = 0
        self.resample_freq = self.load_files * self.use_per_file * 2  # do 2 epochs per cached data

        self.data_x, self.data_y = self._cache_data()

    def _cache_data(self):
        print("MAKE NEW DATASET")
        use_files = np.random.choice(self.files, size=self.load_files, replace=False)
        data_x = np.zeros((self.load_files * self.use_per_file, 12, 495, 436, 8))
        data_y = np.zeros((self.load_files * self.use_per_file, 6, 495, 436, 8))
        # print("allocated:", data_x.shape, data_y.shape)
        counter = 0
        for file in use_files:
            loaded_file = load_h5_file(file)
            random_times = (np.random.rand(self.use_per_file) * 240).astype(int)
            # print("loaded file ", file, loaded_file.shape, random_times)
            for i in range(self.use_per_file):
                start_hour = random_times[i]
                two_hours = loaded_file[start_hour : start_hour + 12 * 2 + 1]
                # print("two hours", two_hours.shape)
                # self._load_h5_file(self.files[file_idx], sl=slice(start_hour, start_hour + 12 * 2 + 1))
                input_data, output_data = prepare_test(two_hours)

                # print("inp and outp", input_data.shape, output_data.shape)

                data_x[counter] = input_data
                data_y[counter] = output_data
                counter += 1

        data_x = self._to_torch(data_x)
        data_y = self._to_torch(data_y)
        print("inp and outp after torch", data_x.size(), data_y.size())

        if self.transform is not None:
            data_x = self.transform(data_x)
            data_y = self.transform(data_y)
        return data_x, data_y

    def __len__(self):
        return self.load_files * self.use_per_file * self.epoch_size_internal

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        self.internal_counter += 1
        if self.internal_counter % self.resample_freq == 0:
            self.data = self._cache_data()
        return self.data_x[idx // self.epoch_size_internal], self.data_y[idx // self.epoch_size_internal]
