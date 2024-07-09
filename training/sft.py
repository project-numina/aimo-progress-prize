#!/usr/bin/env python
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

"""
Script to instruction fine-tune causal language models on a Hub dataset

Adapted from huggingface/transformers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
import logging
import math
import random
import shutil
import sys

import datasets
import torch
import transformers
import wandb
from accelerate import Accelerator
from numina.configs import DataConfig, ModelConfig, SFTConfig
from numina.utils import (
    H4ArgumentParser,
    apply_chat_template,
    check_hub_revision_exists,
    get_checkpoint,
    get_datasets,
    get_tokenizer,
    hf_login,
    init_wandb_training,
    push_to_hub_revision,
)
from transformers import set_seed
from trl import SFTTrainer

logger = logging.getLogger(__name__)


def main():
    accelerator = Accelerator()

    parser = H4ArgumentParser((ModelConfig, DataConfig, SFTConfig))
    model_config, data_config, sft_config = parser.parse()
    # Check if Hub revision exists
    check_hub_revision_exists(sft_config)
    # Set seed for reproducibility
    set_seed(sft_config.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = sft_config.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {sft_config.local_rank}, device: {sft_config.device}, n_gpu: {sft_config.n_gpu}"
        + f" distributed training: {bool(sft_config.local_rank != -1)}, 16-bits training: {sft_config.fp16}"
    )
    logger.info(f"Model parameters {model_config}")
    logger.info(f"Data parameters {data_config}")
    logger.info(f"Training/evaluation parameters {sft_config}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(sft_config)
    if last_checkpoint is not None and sft_config.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Setup WandB
    if sft_config.wandb_enabled:
        init_wandb_training(sft_config)

    # Login to HuggingFace Hub if needed
    hf_login()

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_config, splits=data_config.dataset_splits)

    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_config, data_config, set_pad_token=sft_config.packing)

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if sft_config.gradient_checkpointing else True,
    )

    model = model_config.model_name_or_path

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, "task": "sft"})

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    with sft_config.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        model_init_kwargs=model_kwargs,
        args=sft_config,
        train_dataset=raw_datasets["train"] if sft_config.do_train else None,
        eval_dataset=raw_datasets["test"] if sft_config.do_eval else None,
        dataset_text_field="text",
        max_seq_length=data_config.block_size,
        tokenizer=tokenizer,
        packing=sft_config.packing,
    )

    ###############
    # Training loop
    ###############
    if sft_config.do_train:
        logger.info("*** Train ***")
        checkpoint = None
        if sft_config.resume_from_checkpoint is not None:
            checkpoint = sft_config.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_config.max_train_samples if data_config.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        if wandb.run is not None:
            wandb.config.update(model_config, allow_val_change=True)
            wandb.config.update(data_config, allow_val_change=True)

    ##########
    # Evaluate
    ##########
    if sft_config.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_config.max_eval_samples if data_config.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(sft_config.output_dir)
    logger.info(f"Model saved to {sft_config.output_dir}")

    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_config.model_name_or_path,
            "dataset": list(data_config.dataset_mixer.keys()),
            "dataset_tags": list(data_config.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(sft_config.output_dir)

        if sft_config.push_to_hub_revision is True:
            logger.info("Pushing to hub...")
            is_model_on_hub = push_to_hub_revision(sft_config)
            # Delete local checkpoint if model on Hub
            if is_model_on_hub is True:
                shutil.rmtree(sft_config.output_dir)

    accelerator.wait_for_everyone()
    wandb.finish()


if __name__ == "__main__":
    main()
