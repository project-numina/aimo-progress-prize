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
from typing import Dict, List, Optional, Union


@dataclass
class DataConfig:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    add_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether to add special tokens from the dialogue template to the model's vocab"},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    dataset_mixer: Optional[Dict[str, Union[float, Dict[str, object]]]] = field(
        default=None,
        metadata={
            "help": (
                """
                    Datasets and either their proportions to be used for training,
                    or a dict of their proportions and the dataset revision to use.
                    e.g.
                    {
                        'HuggingFaceH4/testing_codealpaca_small': 0.5,
                        'HuggingFaceH4/testing_codealpaca_small': {
                            'fraction': 0.5,
                            'revision': '20-examples'
                        }
                    }

                    As yaml
                    dataset_mixer:
                        HuggingFaceH4/testing_codealpaca_small: 0.5
                        HuggingFaceH4/testing_codealpaca_small:
                            fraction: 0.5
                            revision: 20-examples
                """
            )
        },
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    data_filter_fn: Optional[str] = field(
        default=None, metadata={"help": "option to include non-default data filtering (e.g. for toxicity example)."}
    )
    data_tokenize_fn: Optional[str] = field(
        default=None,
        metadata={"help": "option to include non-default data tokenization (for datasets not in the format)."},
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Deprecated, use dataset_mixer the name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_split_name: Optional[str] = field(
        default="train", metadata={"help": "The dataset split to use (via the datasets library)."}
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the model prompt (usually an instruction)."
        },
    )
    filter_by_max_len: bool = field(
        default=False,
        metadata={"help": "Filter the training data so that each example is less than max_source_length."},
    )
    completion_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the completions."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    pre_tokenized: bool = field(
        default=False,
        metadata={"help": "If the training dataset is pre-tokenized."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    prompt_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the prompt template to use for conditioning the model. Deprecated in favour of `dialogue_template`"
        },
    )
    dialogue_template: Optional[str] = field(
        default="no_system",
        metadata={
            "help": "The name of the dialogue template to use for conditioning the model. See h4.training.dialogues for choices."
        },
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

        if self.dataset_name is not None:
            raise ValueError("Deprecated. Use `dataset_mixer` instead.")
