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
Script to quantize a Transformers model with AutoGPTQ and push it to the Hub

Usage:

python scripts/deployment/quantize_model_gptq.py \
    --model_id <ORG>/deepseek-math-7b-sft \
    --revision aimo-tora
"""
import argparse
import shutil

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="<ORG>/deepseek-math-7b-sft",
        type=str,
        help="Name of repository on the Hub in '<ORG>/<NAME>' format.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="aimo_tora",
        help="Name of branch in repository to save experiments.",
    )
    parser.add_argument(
        "--trust_remote_code", type=str2bool, nargs="?", const=True, default=False, help="Trust remote code."
    )
    parser.add_argument(
        "--gptq_revision",
        type=str,
        default="gptq",
        help="Name of branch in repository to save experiments.",
    )
    parser.add_argument("--bits", type=int, default=8, help="Quantize model to bits.")
    parser.add_argument("--calibration_dataset", type=str, help="Dataset to use for calibration.")
    args = parser.parse_args()

    if args.bits not in [2, 3, 4, 8]:
        raise ValueError("bits should be 2, 3, 4 or 8")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    calibration_dataset = load_dataset(args.calibration_dataset, split="train").shuffle(seed=42).select(range(256))

    prompt = "{}"

    def apply_template(example, tokenizer, prompt: str):
        messages = [
            {"role": "user", "content": prompt.format(example["problem"], "{}")},
            {"role": "assistant", "content": example["messages"][1]["content"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        example["text"] = text
        return example

    calibration_dataset = calibration_dataset.map(apply_template, fn_kwargs={"tokenizer": tokenizer, "prompt": prompt})

    examples = [tokenizer(d["text"]) for d in calibration_dataset]

    quantize_config = BaseQuantizeConfig(
        bits=args.bits,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=True,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_id,
        quantize_config,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(examples, batch_size=4)

    # Save quantized model
    model_name = args.model_id.split("/")[-1]
    output_dir = f"{model_name}-{args.revision}-{args.gptq_revision}-{args.bits}bits"

    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Push to Hub
    api = HfApi()
    # Get initial commit to branch from
    initial_commit = api.list_repo_commits(args.model_id)[-1]
    api.create_branch(
        repo_id=args.model_id,
        branch=f"{args.revision}.{args.gptq_revision}-{args.bits}bits",
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    url = api.upload_folder(
        folder_path=output_dir,
        repo_id=args.model_id,
        revision=f"{args.revision}.{args.gptq_revision}-{args.bits}bits",
    )
    print(f"Model quantized to {args.bits} bits and pushed to {url.commit_url}")

    # Clean up
    shutil.rmtree(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
