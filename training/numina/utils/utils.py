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

import dataclasses
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, NewType, Optional, Tuple, Union

from huggingface_hub import (
    create_branch,
    create_repo,
    list_repo_commits,
    list_repo_files,
    list_repo_refs,
    login,
    repo_exists,
    upload_folder,
)
from transformers import AutoTokenizer, HfArgumentParser, PreTrainedTokenizer
from transformers.trainer_utils import get_last_checkpoint

from ..configs import DataConfig, ModelConfig, SFTConfig

DEFAULT_CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation"],
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation']}"
        )
    return example


DataClassType = NewType("DataClassType", Any)


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a yaml file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (:obj:`str`): the path to the config file used
            other_args (:obj:`List[str]`, `optional`): a list of strings to parse as command line arguments.
                These will look like ['--arg=val', '--arg2=val2'].

        Returns:
            :obj:`List[dataclass]`: a list of dataclasses with the values from the yaml file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool or base_type == Optional[bool]:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        elif val in ["None", "none"]:
                            inputs[arg] = None
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        unparsed_args = set(other_args.keys()) - set(used_args.keys())

        if len(unparsed_args) > 0:
            raise ValueError(f"The following arguments were not parsed: {unparsed_args}")
        return outputs

    def parse(self, allow_extra_keys=False) -> Union[DataClassType, Tuple[DataClassType]]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]), allow_extra_keys=allow_extra_keys)
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


def hf_login():
    """Login to HuggingFace Hub if HF_TOKEN is defined in the environment"""
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is not None:
        login(token=hf_token)


def is_git_directory(path="."):
    return subprocess.call(["git", "-C", path, "status"], stderr=subprocess.STDOUT, stdout=open(os.devnull, "w")) == 0


def check_hub_revision_exists(training_args: SFTConfig):
    """Checks if a given Hub revision exists."""
    if repo_exists(training_args.hub_model_id):
        if training_args.push_to_hub_revision is True:
            # First check if the revision exists
            revisions = [rev.name for rev in list_repo_refs(training_args.hub_model_id).branches]
            # If the revision exists, we next check it has a README file
            if training_args.hub_model_revision in revisions:
                repo_files = list_repo_files(
                    repo_id=training_args.hub_model_id, revision=training_args.hub_model_revision
                )
                if "README.md" in repo_files and training_args.overwrite_hub_revision is False:
                    raise ValueError(
                        f"Revision {training_args.hub_model_revision} already exists. "
                        "Use --overwrite_hub_revision to overwrite it."
                    )


def get_tokenizer(model_args: ModelConfig, data_args: DataConfig, set_pad_token: bool = True) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Hack for Qwen-14b which doesn't have an EOS token defined in the tokenizer
    if "qwen-14b" in model_args.model_name_or_path.lower():
        tokenizer.eos_token_id = 151643  # <|endoftext|>

    if set_pad_token is True and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template
    elif tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer


def get_checkpoint(training_args: SFTConfig) -> Path | None:
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def init_wandb_training(training_args):
    """
    Helper function for setting up Weights & Biases logging tools.
    """
    os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group
    if training_args.wandb_run_id is not None:
        os.environ["WANDB_RUN_ID"] = training_args.wandb_run_id
    if training_args.wandb_tags is not None:
        os.environ["WANDB_TAGS"] = ",".join(tag for tag in training_args.wandb_tags)


logger = logging.getLogger(__name__)

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.setLevel(20)  # INFO by default


def push_to_hub_revision(training_args: SFTConfig, extra_ignore_patterns=[]) -> bool:
    """Pushes the model to branch on a Hub repo."""

    # Create a repo if it doesn't exist yet
    repo_url = create_repo(repo_id=training_args.hub_model_id, private=True, exist_ok=True)
    # Get initial commit to branch from
    initial_commit = list_repo_commits(training_args.hub_model_id)[-1]
    # Now create the branch we'll be pushing to
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info(f"Created target repo at {repo_url}")
    logger.info(f"Pushing to the Hub revision {training_args.hub_model_revision}...")
    ignore_patterns = ["checkpoint-*"]
    ignore_patterns.extend(extra_ignore_patterns)
    upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        commit_message=f"Add {training_args.hub_model_revision} checkpoint",
        ignore_patterns=ignore_patterns,
    )
    logger.info(f"Pushed to {repo_url} revision {training_args.hub_model_revision} successfully!")

    return True
