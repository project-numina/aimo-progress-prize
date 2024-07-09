# coding=utf-8
# Copyright 2024 The Numina Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union
from datasets import DatasetDict, concatenate_datasets, load_dataset

from ..configs import DataConfig

COLUMNS_TO_KEEP = ["messages", "chosen", "rejected", "prompt", "completion", "label", "score"]


def get_datasets(
    data_config: Union[DataConfig, dict],
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    if type(data_config) is DataConfig:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     'dataset1': 0.5
        #     'dataset2': 0.3
        #     'dataset3': 0.2
        # Or optionally, a revision can be specified:
        # datasets_mixer:
        #     'dataset1': 0.5
        #     'dataset2':
        #           'fraction': 0.3,
        #           'revision': 'main'
        #     'dataset3':
        #           'fraction': 0.2,
        #           'revision': 'other-branch'
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        # Or optionally, a revision can be specified:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset2": {"fraction": 0.3, "revision": "main"},
        #             "dataset3": {"fraction": 0.2, "revision": "other-branch"},
        #         }

        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    return raw_datasets


def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac_or_dict in dataset_mixer.items():
        revision = "main"
        if isinstance(frac_or_dict, dict):
            frac = frac_or_dict.get("fraction", 1.0)  # default to 1.0 if no fraction is specified
            revision = frac_or_dict.get("revision", "main")  # default to main if no revision is specified
        else:
            frac = frac_or_dict

        fracs.append(frac)
        for split in splits:
            if "train" in split:
                train_ds = load_dataset(
                    ds,
                    split=split,
                    revision=revision,
                )
                train_ds = train_ds.remove_columns(
                    [col for col in train_ds.column_names if col not in COLUMNS_TO_KEEP]
                )
                raw_train_datasets.append(train_ds)
            elif "test" in split:
                val_ds = load_dataset(
                    ds,
                    split=split,
                    revision=revision,
                )
                val_ds = val_ds.remove_columns([col for col in val_ds.column_names if col not in COLUMNS_TO_KEEP])
                raw_val_datasets.append(val_ds)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets
