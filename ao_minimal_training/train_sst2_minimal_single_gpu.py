#!/usr/bin/env python
"""Minimal single-GPU SST-2 Activation Oracle training.

This script lives outside the main repo but imports and reuses the repo's
dataset, activation-materialization, steering, batching, and loss functions.
It intentionally omits DDP, W&B, eval suites, and Hugging Face uploads.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers.optimization import get_linear_schedule_with_warmup

THIS_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = THIS_DIR.parent
REPO_DIR = WORKSPACE_ROOT / "activation_oracles"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import ClassificationDatasetConfig, ClassificationDatasetLoader
from nl_probes.sft import train_features_batch
from nl_probes.utils.activation_utils import get_hf_submodule, get_text_only_lora_targets
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import TrainingDataPoint, construct_batch, materialize_missing_steering_vectors


def parse_layer_percents(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def cached_path(loader: ClassificationDatasetLoader, split: str) -> Path:
    return Path(loader.dataset_config.dataset_folder) / loader.get_dataset_filename(split)  # type: ignore[arg-type]


def make_sst2_loader(
    *,
    model_name: str,
    dataset_folder: Path,
    layer_percents: list[int],
    batch_size: int,
    num_train: int,
    save_acts: bool,
    single_token: bool,
) -> ClassificationDatasetLoader:
    if single_token:
        params = ClassificationDatasetConfig(
            classification_dataset_name="sst2",
            max_window_size=1,
            min_window_size=1,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=2,
        )
    else:
        params = ClassificationDatasetConfig(
            classification_dataset_name="sst2",
            max_window_size=50,
            min_window_size=1,
            min_end_offset=-1,
            max_end_offset=-5,
            num_qa_per_sample=1,
        )

    cfg = DatasetLoaderConfig(
        custom_dataset_params=params,
        num_train=num_train,
        num_test=0,
        splits=["train"],
        model_name=model_name,
        layer_percents=layer_percents,
        save_acts=save_acts,
        batch_size=batch_size,
        dataset_folder=str(dataset_folder),
    )
    return ClassificationDatasetLoader(dataset_config=cfg, model_kwargs={})


def load_cached_training_data(loaders: list[ClassificationDatasetLoader]) -> list[TrainingDataPoint]:
    missing = [cached_path(loader, "train") for loader in loaders if not cached_path(loader, "train").exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Expected cached SST-2 training data, but these files were missing:\n"
            f"{formatted}\n\n"
            "Create the cache inside the main repo first, or pass --dataset-folder to the cache location. "
            "This minimal script deliberately does not regenerate datasets."
        )

    data: list[TrainingDataPoint] = []
    for loader in loaders:
        data.extend(loader.load_dataset("train"))
    return data


def attach_lora(model, cfg: SelfInterpTrainingConfig):
    if cfg.load_lora_path is not None:
        return PeftModel.from_pretrained(model, cfg.load_lora_path, is_trainable=True, autocast_adapter_dtype=True)

    target_modules = cfg.lora_target_modules
    vlm_targets = get_text_only_lora_targets(cfg.model_name)
    if vlm_targets and target_modules == "all-linear":
        target_modules = vlm_targets

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config, autocast_adapter_dtype=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--dataset-folder",
        default="sft_training_data",
        help=(
            "Dataset folder string used in the cached DatasetLoaderConfig. "
            "The repo default is 'sft_training_data'; changing it changes the cache filename hash."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR / "checkpoints_sst2_minimal_single_gpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-train", type=int, default=6000)
    parser.add_argument("--layer-percents", default="25,50,75")
    parser.add_argument("--hook-layer", type=int, default=1)
    parser.add_argument("--steering-coefficient", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--single-loader", action="store_true", help="Use only the single-token SST-2 loader.")
    parser.add_argument("--save-acts", action="store_true", help="Look for a cache built with save_acts=True.")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("This minimal trainer is single-GPU CUDA only; no CUDA device was found.")

    # Keep relative dataset folders compatible with caches created by running
    # the repo scripts from inside activation_oracles/.
    os.chdir(REPO_DIR)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    layer_percents = parse_layer_percents(args.layer_percents)

    set_seed(args.seed)

    loaders = [
        make_sst2_loader(
            model_name=args.model_name,
            dataset_folder=Path(args.dataset_folder),
            layer_percents=layer_percents,
            batch_size=args.batch_size,
            num_train=args.num_train,
            save_acts=args.save_acts,
            single_token=True,
        )
    ]
    if not args.single_loader:
        loaders.append(
            make_sst2_loader(
                model_name=args.model_name,
                dataset_folder=Path(args.dataset_folder),
                layer_percents=layer_percents,
                batch_size=args.batch_size,
                num_train=args.num_train,
                save_acts=args.save_acts,
                single_token=False,
            )
        )

    cfg = SelfInterpTrainingConfig(
        model_name=args.model_name,
        hook_onto_layer=args.hook_layer,
        layer_percents=layer_percents,
        train_batch_size=args.batch_size,
        lr=args.lr,
        steering_coefficient=args.steering_coefficient,
        save_dir=str(args.output_dir),
        seed=args.seed,
        gradient_checkpointing=False,
    ).finalize(dataset_loaders=loaders)

    print("Loading cached SST-2 training data...")
    training_data = load_cached_training_data(loaders)
    random.shuffle(training_data)
    if not training_data:
        raise ValueError("Loaded zero training datapoints.")

    print(f"Loaded {len(training_data)} datapoints.")
    print(f"Layers: {cfg.act_layers} from percents {cfg.layer_percents}")

    tokenizer = load_tokenizer(cfg.model_name)

    model_kwargs = {"device_map": {"": "cuda:0"}}
    if args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=dtype,
        )

    model = load_model(cfg.model_name, dtype, **model_kwargs)
    model.enable_input_require_grads()
    model = attach_lora(model, cfg)
    model.print_trainable_parameters()
    model.train()

    submodule = get_hf_submodule(model, cfg.hook_onto_layer, use_lora=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")
    total_steps = args.max_steps
    warmup_steps = max(0, int(total_steps * 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    optimizer.zero_grad()
    global_step = 0
    data_idx = 0
    pbar = tqdm(total=total_steps, desc="Training SST-2 AO")
    while global_step < total_steps:
        if data_idx + args.batch_size > len(training_data):
            random.shuffle(training_data)
            data_idx = 0

        batch_points = training_data[data_idx : data_idx + args.batch_size]
        data_idx += args.batch_size

        batch_points = materialize_missing_steering_vectors(batch_points, tokenizer, model)
        batch = construct_batch(batch_points, tokenizer, device)

        loss = train_features_batch(cfg, batch, model, submodule, device, dtype)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    pbar.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
