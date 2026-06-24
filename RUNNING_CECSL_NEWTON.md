# Running WorldMem on CECSL and Newton

This repository is being inspected from a sandbox where the codebase is not being run. Treat the commands below as target-machine instructions for the CECSL PC or the Newton cluster.

## Machine Notes

- CECSL PC has the large local data area at `/data/ab575577/`. Keep datasets, model caches, W&B files, checkpoints, and large outputs there.
- CECSL PC has an A6000 Pro/newer NVIDIA architecture. Prefer a current CUDA PyTorch wheel first; fall back to the repo-pinned PyTorch only if it supports the installed driver and GPU.
- Newton does not need the `/data/ab575577/` layout. Use the normal Newton project/scratch locations and cluster module policy.
- The repo scripts currently assume the Minecraft data lives at `data/minecraft`, so on CECSL use a symlink to `/data/ab575577/...` or override `dataset.save_dir` in every run command.

## CECSL First Setup

Start from a clean shell on the CECSL PC:

```bash
cd /path/to/WorldMem

nvidia-smi
conda create -y -n worldmem python=3.10
conda activate worldmem
python -m pip install --upgrade pip setuptools wheel
conda install -y -c conda-forge ffmpeg=4.3.2
```

Set CECSL storage and cache locations before installing or downloading large files:

```bash
export WORLDMEM_ROOT=/data/ab575577/worldmem
export WORLDMEM_DATA=$WORLDMEM_ROOT/data
export HF_HOME=$WORLDMEM_ROOT/hf_cache
export WANDB_DIR=$WORLDMEM_ROOT/wandb
export WANDB_CACHE_DIR=$WORLDMEM_ROOT/wandb/cache
export TMPDIR=$WORLDMEM_ROOT/tmp

mkdir -p "$WORLDMEM_DATA" "$HF_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TMPDIR" "$WORLDMEM_ROOT/outputs"
mkdir -p data
ln -sfn "$WORLDMEM_DATA/minecraft" data/minecraft
```

For the CECSL A6000 Pro/newer-architecture path, install current PyTorch with CUDA 12.8 first, then install the rest of the repo requirements without letting `requirements.txt` downgrade PyTorch. Re-check the official PyTorch local install selector when the CECSL driver changes: <https://pytorch.org/get-started/locally/>.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
grep -vE '^(torch|torchvision)' requirements.txt > /tmp/worldmem-requirements-no-torch.txt
python -m pip install -r /tmp/worldmem-requirements-no-torch.txt
```

If that newer PyTorch path causes package compatibility problems and the GPU test below still works with the paper-era stack, use the closer-to-repo install instead:

```bash
python -m pip install -r requirements.txt
```

Verify CUDA before doing any long run:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x=torch.zeros(1, device='cuda'); print(x)"
```

If this fails with `no kernel image is available`, `invalid device function`, or a similar architecture error, the installed PyTorch wheel is too old for the CECSL GPU. Reinstall a newer CUDA PyTorch wheel and repeat the verification.

## Dataset Layout

The code expects:

```text
data/minecraft/
  training/
  validation/
  test/
```

On CECSL, because `data/minecraft` is symlinked, the real data should be:

```text
/data/ab575577/worldmem/data/minecraft/
  training/
  validation/
  test/
```

Download the released Minecraft dataset into the CECSL data directory. One practical route is the Hugging Face CLI:

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download zeqixiao/worldmem_minecraft_dataset \
  --repo-type dataset \
  --local-dir "$WORLDMEM_DATA/minecraft"
```

After download, confirm that each split contains `.mp4` files and matching `.npz` files. The loader looks for videos under each split and then opens the same path with a `.npz` suffix for actions and poses.

## Weights and Caches

Inference and evaluation scripts reference these Hugging Face checkpoint files:

```text
zeqixiao/worldmem_checkpoints/diffusion_only.ckpt
zeqixiao/worldmem_checkpoints/vae_only.ckpt
zeqixiao/worldmem_checkpoints/pose_prediction_model_only.ckpt
```

The code can download them through `huggingface_hub` when the machine has internet. Keep `HF_HOME=/data/ab575577/worldmem/hf_cache` on CECSL so the cache does not fill the home directory.

## W&B

`configurations/training.yaml` currently has:

```yaml
wandb:
  entity: xizaoqu
  project: worldmem
  mode: online
```

For your own online logging:

```bash
wandb login
```

Then set `wandb.entity=<your_wandb_entity>` on the command line or edit `configurations/training.yaml`.

For local/offline runs on CECSL:

```bash
export WANDB_MODE=offline
```

Also pass `wandb.mode=offline wandb.entity=local` to `python -m main` if you are not using your own W&B account. `main.py` requires a non-empty W&B entity even when the logger is offline/disabled.

## Smoke Tests

Run a tiny evaluation first. This checks imports, GPU, checkpoint download, dataset reading, and output writing:

```bash
python -m main +name=cecsl_eval_smoke \
  experiment.tasks=[test] \
  dataset.validation_multiplier=1 \
  +dataset.seed=42 \
  +diffusion_model_path=zeqixiao/worldmem_checkpoints/diffusion_only.ckpt \
  +vae_path=zeqixiao/worldmem_checkpoints/vae_only.ckpt \
  +customized_load=true \
  +seperate_load=true \
  dataset.n_frames=8 \
  dataset.save_dir=data/minecraft \
  +dataset.n_frames_valid=700 \
  algorithm.diffusion.sampling_timesteps=20 \
  +algorithm.memory_condition_length=8 \
  +algorithm.lpips_batch_size=16 \
  +algorithm.log_video=true \
  +algorithm.save_local=true \
  +dataset.customized_validation=true \
  +algorithm.n_tokens=8 \
  algorithm.context_frames=600 \
  experiment.test.batch_size=1 \
  experiment.test.limit_batch=1 \
  wandb.mode=offline \
  wandb.entity=local \
  +output_dir=$WORLDMEM_ROOT/outputs/manual/cecsl_eval_smoke
```

Then try inference:

```bash
sh infer.sh
```

If you want inference outputs under `/data/ab575577/`, run the equivalent `python -m main ... +output_dir=$WORLDMEM_ROOT/outputs/manual/infer_smoke` command instead of the shell script, or edit the script locally.

## Training

The paper README says training used 4 H100 GPUs and converged around 500K steps. On a single CECSL A6000 Pro, start with smaller smoke settings before committing to a long run:

```bash
python -m main +name=cecsl_train_smoke \
  +diffusion_model_path=your_diffusion_model_path \
  +vae_path=your_vae_path \
  +customized_load=true \
  +seperate_load=true \
  +zero_init_gate=true \
  dataset.n_frames=8 \
  dataset.save_dir=data/minecraft \
  +dataset.n_frames_valid=700 \
  +dataset.angle_range=110 \
  +dataset.pos_range=2 \
  +dataset.memory_condition_length=8 \
  +dataset.customized_validation=true \
  +dataset.add_timestamp_embedding=true \
  +dataset.wo_updown=true \
  +algorithm.n_tokens=8 \
  +algorithm.memory_condition_length=8 \
  algorithm.context_frames=600 \
  +algorithm.relative_embedding=true \
  +algorithm.log_video=true \
  +algorithm.add_timestamp_embedding=true \
  algorithm.metrics=[lpips,psnr] \
  experiment.training.batch_size=1 \
  experiment.validation.batch_size=1 \
  experiment.validation.limit_batch=1 \
  experiment.training.checkpointing.every_n_train_steps=2500 \
  experiment.training.max_steps=100 \
  wandb.mode=offline \
  wandb.entity=local \
  +output_dir=$WORLDMEM_ROOT/outputs/manual/cecsl_train_smoke
```

Once the smoke run is healthy, use the staged scripts:

```bash
sh train_stage_1.sh
sh train_stage_2.sh
sh train_stage_3.sh
```

Before real training, replace placeholder checkpoint paths in `train_stage_1.sh`, and replace the `resume=...` and `+output_dir=...` placeholders in stages 2 and 3. On CECSL, point long-run output directories into `/data/ab575577/worldmem/outputs/...`.

## Newton Notes

Newton is the cluster target. This repo has Slurm submission support in `utils/cluster_utils.py`, but this checkout does not currently include a `configurations/cluster/*.yaml` file. Without adding a cluster config, `python -m main ...` runs in the current shell/session.

For Newton, create the environment according to the cluster's module/conda policy, put data in the Newton-approved project or scratch location, and either:

- symlink `data/minecraft` to the Newton data location, or
- pass `dataset.save_dir=/path/on/newton/minecraft` in every command.

If you add a Hydra cluster config later, `main.py` will detect `cluster=...` and submit through Slurm instead of running locally.

## Common Issues

- `ModuleNotFoundError`: reactivate `conda activate worldmem`, then reinstall missing packages.
- `pyrealsense2` install failure: it is listed in `requirements.txt`, but the inspected train/infer/eval paths do not import it. If it blocks setup, install the remaining requirements and revisit only if data generation needs it.
- CUDA architecture errors: reinstall a newer PyTorch CUDA wheel, especially on the CECSL A6000 Pro/newer GPU.
- Home directory fills up: re-check `HF_HOME`, `WANDB_DIR`, `WANDB_CACHE_DIR`, `TMPDIR`, and `+output_dir`.
- Dataset has zero samples: check that `training`, `validation`, and `test` contain `.mp4` files, and that every video has a matching `.npz` action/pose file.
- W&B entity error: pass `wandb.entity=local` for offline tests or set your real W&B entity for online logging.
