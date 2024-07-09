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

from dataclasses import dataclass, field
from typing import List, Optional

import transformers


@dataclass
class SFTConfig(transformers.TrainingArguments):
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    benchmarks: List[str] = field(
        default_factory=lambda: [], metadata={"help": ("The benchmarks to run after training.")}
    )
    mask_user_turns: bool = field(
        default=False,
        metadata={"help": ("Whether to mask user turns.")},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    neftune_noise_alpha: Optional[float] = field(
        default=None, metadata={"help": ("If not `None`, this will activate NEFTune noise embeddings.")}
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    optim: Optional[str] = field(default="adamw_torch")
    overwrite_hub_revision: bool = field(default=False, metadata={"help": ("Whether to overwrite the Hub revision.")})
    packing: bool = field(
        default=True, metadata={"help": ("Whether to pack sequences of the dataset for faster training.")}
    )
    push_to_hub_revision: bool = field(default=False, metadata={"help": ("Whether to push to a Hub revision/branch.")})
    reward_loss_fn: Optional[str] = field(
        default="NegLogSigmoid",
        metadata={"help": ("Loss function for reward model.")},
    )
    quants: List[str] = field(
        default_factory=lambda: [], metadata={"help": ("Which quantization methods to apply on final model.")}
    )
    save_strategy: Optional[str] = field(default="steps")
    save_steps: Optional[int] = field(default=0.1)
    save_total_limit: Optional[int] = field(default=1)
    wandb_tags: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Tags to group and filter runs on Weights and Biases.")},
    )
    wandb_enabled: bool = field(
        default=True,
        metadata={"help": ("Whether to enable or disable WandB.")},
    )
    wandb_project: Optional[str] = field(
        default="h4",
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_entity: Optional[str] = field(
        default="huggingface",
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default="tr_00_some-descriptor",
        metadata={"help": ("Group multiple runs under this group name.")},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Set this to a globally unique string (per project) corresponding to a single run of your script."
            )
        },
    )
