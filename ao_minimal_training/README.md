# Minimal Activation Oracle Training

This folder is intentionally separate from `activation_oracles/`. The scripts
import the repo package and reuse its training utilities, but do not modify the
repo itself.

## Files

- `train_sst2_minimal_single_gpu.py`: single-GPU LoRA training on cached SST-2
  Activation Oracle data.
- `overfit_single_sst2_sample.ipynb`: notebook that overfits one cached SST-2
  datapoint as a sanity check.

## Expected Cache

By default, the script changes into the repo directory and looks for cached
SST-2 files using the repo-standard dataset folder string:

```bash
sft_training_data
```

This matters because `dataset_folder` is included in the repo's cache filename
hash. If you created the cache with the repo defaults, leave
`--dataset-folder` unset.

The default uses `save_acts=False`, matching the repo's main classification
training path. That means cached datapoints store token ids and activation
positions; activation vectors are recomputed during training by
`materialize_missing_steering_vectors(...)`. If your cache contains precomputed
`steering_vectors`, the same call is a no-op for those examples.

## Run

Use the repo's Python environment. For a fresh checkout:

```bash
cd /Users/gloria/dev/ai_safety/evil_oracles/activation_oracles
uv sync
source .venv/bin/activate
```

Then, from any working directory:

```bash
python /Users/gloria/dev/ai_safety/evil_oracles/ao_minimal_training/train_sst2_minimal_single_gpu.py \
  --max-steps 200 \
  --batch-size 4 \
  --load-in-8bit
```

For a quick cache/import smoke test:

```bash
python -m py_compile /Users/gloria/dev/ai_safety/evil_oracles/ao_minimal_training/train_sst2_minimal_single_gpu.py
python /Users/gloria/dev/ai_safety/evil_oracles/ao_minimal_training/train_sst2_minimal_single_gpu.py --max-steps 1 --batch-size 1
```

If the model is not already in the Hugging Face cache, `from_pretrained` will
download it.
