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
conda create -y -n worldmem python=3.10 pip
conda activate worldmem
python -m pip install --upgrade pip setuptools wheel
conda install -y -c conda-forge ffmpeg=4.3.2
```

If the environment already exists and `python -m pip` says `No module named pip`, repair it from inside the active env:

```bash
conda activate worldmem
conda install -y pip setuptools wheel
python -m pip install --upgrade pip setuptools wheel
```

If `conda install pip` is not available on that machine for some reason, try the Python bootstrap:

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
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

On CECSL, a completed download has been observed at about `421G` with all three split directories present. Exact size can vary slightly with cache and partial files, but a much smaller directory, missing `validation`, or remaining `.lock`/`.incomplete` files means the download is not fully clean yet.

Download the released Minecraft dataset into the CECSL data directory. One practical route is the Hugging Face CLI:

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download zeqixiao/worldmem_minecraft_dataset \
  --repo-type dataset \
  --local-dir "$WORLDMEM_DATA/minecraft"
```

After download, confirm that each split contains `.mp4` files and matching `.npz` files. The loader looks for videos under each split and then opens the same path with a `.npz` suffix for actions and poses.

To check whether an interrupted download is actually complete:

```bash
cd /data/ab575577/worldmem/data/minecraft

du -sh .
find training -type f -name '*.mp4' | wc -l
find training -type f -name '*.npz' | wc -l
find validation -type f -name '*.mp4' | wc -l
find validation -type f -name '*.npz' | wc -l
find test -type f -name '*.mp4' | wc -l
find test -type f -name '*.npz' | wc -l
find .cache/huggingface/download \( -name '*.lock' -o -name '*.incomplete' \) -print | head
```

If the download was interrupted, first make sure no old downloader is still running, remove stale `.lock` files only, and then resume with one `hf download` process:

```bash
pgrep -af 'hf download|huggingface-cli|snapshot_download|huggingface_hub'
kill <PID>                # only for old download PIDs that are still running
kill -9 <PID>             # only if the old PID refuses to stop

find /data/ab575577/worldmem/data/minecraft/.cache/huggingface/download \
  -name '*.lock' -type f -delete

hf download zeqixiao/worldmem_minecraft_dataset \
  --repo-type dataset \
  --local-dir /data/ab575577/worldmem/data/minecraft
```

Keep `.incomplete` files when resuming; they are partial download files that the Hugging Face client may be able to continue from.

For a long CECSL resume, run the downloader inside tmux with a log and a smaller worker count so it is easier to monitor and less likely to create many competing locks:

```bash
tmux new -s worldmem_dl
conda activate worldmem

export WORLDMEM_ROOT=/data/ab575577/worldmem
export HF_HOME=$WORLDMEM_ROOT/hf_cache
export TMPDIR=$WORLDMEM_ROOT/tmp
mkdir -p "$HF_HOME" "$TMPDIR" "$WORLDMEM_ROOT/logs"

hf download zeqixiao/worldmem_minecraft_dataset \
  --repo-type dataset \
  --local-dir /data/ab575577/worldmem/data/minecraft \
  --max-workers 4 \
  2>&1 | tee "$WORLDMEM_ROOT/logs/hf_dataset_download_$(date +%F_%H%M).log"
```

Detach from tmux with `Ctrl-b`, then `d`. Reattach with `tmux attach -t worldmem_dl`.

If the restarted command immediately prints `Still waiting to acquire lock`, stop it with `Ctrl-C`, confirm no downloader remains, delete the lock files again, and verify the lock count is zero before restarting:

```bash
pgrep -af 'hf download|huggingface-cli|snapshot_download|huggingface_hub'
find /data/ab575577/worldmem/data/minecraft/.cache/huggingface/download \
  -name '*.lock' -type f -delete
find /data/ab575577/worldmem/data/minecraft/.cache/huggingface/download \
  -name '*.lock' -type f | wc -l
```

If the count is still nonzero, inspect one stuck lock with `ls -l <lock-file>` and remove the specific file with `rm -f <lock-file>` after confirming no Hugging Face downloader is running.

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

## Memory Policy Smoke Runs

WorldMem now has a MemCam-style fixed-budget memory test hook for evaluation/test generation. The default behavior is still unbounded memory. Budgeted policies first retain a bounded memory bank, then WorldMem's original FOV-overlap retrieval selects reference frames from that retained bank.

Supported policies:

```text
unbounded
fifo
rarity_irreplaceability
slam_covisibility
kcenter_coreset
```

On CECSL, run from `~/WorldMem`. If your shell has `WORLDMEM_ROOT=/data/ab575577/worldmem` from the storage setup, that is fine; the runner uses `WORLDMEM_REPO_ROOT` for the repository path and `WORLDMEM_STORAGE_ROOT` for large outputs.

```bash
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
MEMORY_POLICY=fifo \
MEMORY_BUDGET=32 \
bash scripts/run_worldmem_memory_policy_smoke.sh
```

RI:

```bash
MEMORY_POLICY=rarity_irreplaceability \
MEMORY_BUDGET=32 \
bash scripts/run_worldmem_memory_policy_smoke.sh
```

SLAM-style covisibility:

```bash
MEMORY_POLICY=slam_covisibility \
MEMORY_BUDGET=32 \
bash scripts/run_worldmem_memory_policy_smoke.sh
```

K-center coreset:

```bash
MEMORY_POLICY=kcenter_coreset \
MEMORY_BUDGET=32 \
bash scripts/run_worldmem_memory_policy_smoke.sh
```

WorldMem K-center uses pooled latent features plus pose distance by default, matching the spirit of the MemCam `kcenter_b*_dino_pose` runs while staying native to WorldMem's latent memory representation. Default weights are:

```text
KCENTER_VISUAL_WEIGHT=0.5
KCENTER_POSE_WEIGHT=0.5
KCENTER_TIME_WEIGHT=0.0
KCENTER_ARCHIVE_STRIDE=1
```

The script uses `/data/ab575577/worldmem` automatically on CECSL. On Newton, it does not assume that path; set the data and output roots explicitly if needed:

```bash
WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_DATA_DIR=/path/on/newton/minecraft \
WORLDMEM_STORAGE_ROOT=$HOME/worldmem_results \
MEMORY_POLICY=fifo \
MEMORY_BUDGET=32 \
bash scripts/run_worldmem_memory_policy_smoke.sh
```

Newton Slurm wrapper:

```bash
MEMORY_POLICY=fifo MEMORY_BUDGET=32 sbatch slurm/newton_worldmem_memory_policy_smoke.sbatch
```

Each run writes a JSONL access trace under:

```text
<output_dir>/access_traces/<run_name>.jsonl
```

The trace records retrieval events and eviction events, including selected memory frames, FOV-overlap confidence, retained memory size, and policy-specific eviction scores where available.

### WorldMem Base Memory Semantics

WorldMem's base method is memory-augmented, but the reported model does not impose a fixed storage budget. During validation/test generation, it initializes `xs_pred` with the full context window, then appends each newly generated chunk to that latent history. For each next chunk it uses a short local sliding window plus `memory_condition_length` retrieved reference frames.

The key distinction is:

- The model does **not** attend to every past frame at once. It attends to a small retrieved set, usually `memory_condition_length=8`.
- The candidate memory bank for the base method is effectively all previous frames. In this repo, `memory_policy=unbounded` creates no bounded buffer, so retrieval candidates default to `range(curr_frame)`.
- Retrieval is FOV/pose based: it samples 3D points near the upcoming pose horizon, scores previous frames by FOV overlap with that future region, applies a small recency penalty, and greedily selects memory frames.
- In the interactive API, the explicit memory arrays (`memory_latent_frames`, actions, poses, c2w, frame indices) are concatenated with newly generated frames after each call, so that path also grows memory unless an external policy prunes it.

So the paper's base method is best described as **unbounded storage with bounded per-step retrieval**. The budgeted policies here bound the storage/candidate bank before WorldMem's own FOV-overlap retriever chooses the per-step reference frames.

### GPU Memory Profiling

To compare inference-time peak GPU memory between unbounded WorldMem and a bounded policy, use:

```bash
cd ~/WorldMem
conda activate worldmem

GPU=0 \
FUTURE_SECONDS=60 \
NUM_VIDEOS=1 \
MINE_POLICY=rarity_irreplaceability \
MINE_BUDGETS=32 \
LOG_VIDEO=false \
bash scripts/profile_worldmem_gpu_memory.sh
```

This profiles one 60s video for unbounded and one 60s video for RI b32. To profile all RI budgets:

```bash
GPU=0 \
FUTURE_SECONDS=60 \
NUM_VIDEOS=1 \
MINE_POLICY=rarity_irreplaceability \
MINE_BUDGETS=16,32,64,128 \
LOG_VIDEO=false \
bash scripts/profile_worldmem_gpu_memory.sh
```

The profiler writes a timestamped directory under:

```text
/data/ab575577/worldmem/outputs/memory_policy/gpu_memory_profiles/
```

Important files:

```text
summary.csv                         # one row per profiled run
nvidia_smi/*.csv                    # external GPU polling trace
access_traces/*.jsonl               # WorldMem trace with PyTorch CUDA peak events
logs/*.log                          # full command logs
```

`summary.csv` includes:

- `peak_nvidia_smi_used_mib`: whole-device peak memory from `nvidia-smi`.
- `net_peak_nvidia_smi_used_mib`: peak minus baseline at the start of the run.
- `peak_torch_allocated_mib`: process-level PyTorch peak allocated memory.
- `peak_torch_reserved_mib`: process-level PyTorch peak reserved/cached memory.

Note: in the current validation/generation path, WorldMem keeps most generated latent history on CPU and moves only the sliding window plus retrieved reference frames to GPU. So peak GPU memory may be similar between unbounded and bounded policies; the unbounded-vs-budgeted difference can show up more strongly in candidate-bank size, CPU/RAM behavior, retrieval cost, and output quality. This profiling still gives the clean GPU-memory evidence.

Observed CECSL CPU-bank GPU-memory profile for one 60s video on GPU 0, from `/data/ab575577/worldmem/outputs/memory_policy/gpu_memory_profiles/2026-07-17_134759/summary.csv` on 2026-07-17:

| Run | Policy | Budget | Peak `nvidia-smi` MiB | Net peak MiB | Peak PyTorch allocated MiB | Peak PyTorch reserved MiB | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `worldmem_gpu_profile_unbounded_60s_n1` | unbounded | | 10921.000 | 10906.000 | 9640.697 | 10204.000 | 0 |
| `worldmem_gpu_profile_rarity_irreplaceability_b32_60s_n1` | RI | 32 | 10921.000 | 10906.000 | 9640.697 | 10204.000 | 0 |

Interpretation:

- Current WorldMem validation uses a CPU-resident latent/history bank, so unbounded does **not** grow GPU memory in this released path.
- RI b32 and unbounded have identical peak GPU memory for this run because both send only the active sliding window plus `memory_condition_length=8` retrieved references to GPU.
- The useful claim is therefore not "unbounded OOMs GPU in the released implementation." The useful claim is: unbounded avoids GPU growth by CPU offloading, while bounded memory allows a GPU-resident bank with constant memory. The Pareto analysis should compare CPU-bank and GPU-bank variants using latency, peak GPU memory, and quality.

Latency note: the original retrieval code computed FOV overlap for all previous frames and then masked to the bounded candidates. That made bounded policies pay most of the unbounded retrieval cost. The current local code changes `_generate_condition_indices` so bounded policies score only their retained candidate union, while unbounded still scores all previous frames. New profiler runs also write `wall_seconds`, `total_seconds`, `retrieval_seconds`, `sampling_seconds`, `memory_update_seconds`, and `decode_seconds` to `summary.csv`.

Observed CECSL optimized CPU-bank latency profile for one 60s video on GPU 0, from `/data/ab575577/worldmem/outputs/memory_policy/gpu_memory_profiles/2026-07-17_142116/summary.csv` on 2026-07-17:

| Run | Policy | Budget | Wall sec | Total sec | Retrieval sec | Sampling sec | Decode sec | Peak `nvidia-smi` MiB | Peak PyTorch allocated MiB | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `worldmem_gpu_profile_unbounded_60s_n1` | unbounded | | 745 | 730.193 | 43.691 | 677.146 | 0.915 | 10921.000 | 9640.697 | 0 |
| `worldmem_gpu_profile_rarity_irreplaceability_b32_60s_n1` | RI | 32 | 718 | 703.340 | 3.280 | 689.320 | 0.918 | 10915.000 | 9640.697 | 0 |

Interpretation:

- In CPU-bank mode, RI b32 has essentially the same peak GPU memory as unbounded (`10915` vs `10921` MiB) but lower wall time (`718s` vs `745s`).
- Retrieval time drops from `43.691s` to `3.280s`, about a 13.3x retrieval speedup and `40.411s` saved inside retrieval.
- End-to-end wall time improves by `27s`, about 3.6%, while total traced runtime improves by `26.853s`.
- Sampling still dominates runtime (`~677-689s`), so retrieval savings do not translate one-for-one into wall-clock speed. The defensible CPU-bank systems claim is: RI improves quality, keeps GPU peak unchanged, and reduces retrieval overhead enough to improve total latency modestly.

For the full CPU-bank versus GPU-bank Pareto profile:

```bash
cd ~/WorldMem
conda activate worldmem

GPU=0 \
FUTURE_SECONDS=60 \
NUM_VIDEOS=1 \
MINE_POLICY=rarity_irreplaceability \
MINE_BUDGETS=32 \
MEMORY_BANK_DEVICES=cpu,gpu \
LOG_VIDEO=false \
bash scripts/profile_worldmem_gpu_memory.sh
```

This runs four points:

```text
cpu-bank unbounded
cpu-bank RI b32
gpu-bank unbounded
gpu-bank RI b32
```

The GPU-bank mode is an analysis variant controlled by `+algorithm.memory_bank_device=gpu`. It keeps the full generated output/history on CPU for decoding, while keeping a separate retrieval/reference bank on GPU. For unbounded, that GPU bank grows with the horizon; for budgeted policies, that GPU bank is capped by the memory budget.

The resulting `summary.csv` includes `memory_bank_device`, `peak_bank_mib`, and `peak_bank_frames` in addition to latency and peak GPU-memory columns. For the Pareto plot, use `wall_seconds` or `total_seconds` on X and `peak_nvidia_smi_used_mib` or `peak_torch_allocated_mib` on Y; use `peak_bank_mib` to show the actual resident-bank growth.

Observed CECSL 10s CPU/GPU-bank smoke profile, one video on GPU 0, from `/data/ab575577/worldmem/outputs/memory_policy/gpu_memory_profiles/2026-07-17_144821/summary.csv` on 2026-07-17:

| Run | Bank | Policy | Budget | Wall sec | Retrieval sec | Sampling sec | Peak bank MiB | Peak bank frames | Peak `nvidia-smi` MiB | Peak PyTorch allocated MiB | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `worldmem_gpu_profile_cpu_bank_unbounded_10s_n1` | CPU | unbounded | | 132 | 5.452 | 111.630 | | 0 | 8555.000 | 7478.448 | 0 |
| `worldmem_gpu_profile_cpu_bank_rarity_irreplaceability_b32_10s_n1` | CPU | RI | 32 | 131 | 0.626 | 114.280 | | 0 | 8555.000 | 7478.448 | 0 |
| `worldmem_gpu_profile_gpu_bank_unbounded_10s_n1` | GPU | unbounded | | 134 | 5.371 | 112.558 | 14.062 | 700 | 8571.000 | 7493.319 | 0 |
| `worldmem_gpu_profile_gpu_bank_rarity_irreplaceability_b32_10s_n1` | GPU | RI | 32 | 129 | 0.551 | 113.737 | 0.738 | 32 | 8557.000 | 7480.030 | 0 |

Interpretation:

- The GPU-bank mode works for the 10s smoke. Unbounded stores 700 latent frames on GPU; RI b32 stores exactly 32.
- The latent memory bank is small in absolute GPU memory: unbounded is `14.062` MiB at 700 frames, RI b32 is `0.738` MiB at 32 frames. So the WorldMem latent-bank GPU-OOM story is weak for this released resolution/model path.
- The structural constant-memory result is still clear: GPU-bank unbounded scales with horizon, GPU-bank RI is capped by budget.
- RI reduces retrieval time in both CPU-bank and GPU-bank modes. At 10s, retrieval drops from about `5.4s` to about `0.6s`; end-to-end wall time changes only slightly because sampling dominates.

Observed CECSL 60s CPU/GPU-bank Pareto profile, one video on GPU 0, from `/data/ab575577/worldmem/outputs/memory_policy/gpu_memory_profiles/2026-07-17_145930/summary.csv` on 2026-07-17:

| Run | Bank | Policy | Budget | Wall sec | Retrieval sec | Sampling sec | Peak bank MiB | Peak bank frames | Peak `nvidia-smi` MiB | Peak PyTorch allocated MiB | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `worldmem_gpu_profile_cpu_bank_unbounded_60s_n1` | CPU | unbounded | | 748 | 44.010 | 680.023 | | 0 | 10921.000 | 9640.697 | 0 |
| `worldmem_gpu_profile_cpu_bank_rarity_irreplaceability_b32_60s_n1` | CPU | RI | 32 | 716 | 3.050 | 688.344 | | 0 | 10915.000 | 9640.697 | 0 |
| `worldmem_gpu_profile_gpu_bank_unbounded_60s_n1` | GPU | unbounded | | 746 | 44.440 | 678.447 | 31.641 | 1200 | 10953.000 | 9673.181 | 0 |
| `worldmem_gpu_profile_gpu_bank_rarity_irreplaceability_b32_60s_n1` | GPU | RI | 32 | 711 | 2.705 | 685.963 | 0.826 | 32 | 10917.000 | 9642.367 | 0 |

Interpretation:

- RI b32 improves end-to-end wall time in both bank modes. CPU-bank RI is `32s` faster than CPU-bank unbounded (`716s` vs `748s`), and GPU-bank RI is `35s` faster than GPU-bank unbounded (`711s` vs `746s`).
- Retrieval time drops from about `44s` for unbounded to about `3s` for RI b32. The retrieval speedup is roughly 14-16x, depending on bank mode.
- GPU-bank RI b32 is the fastest of the four points (`711s`) and keeps the resident GPU bank capped at 32 frames (`0.826` MiB).
- GPU-bank unbounded stores all 1200 latent frames (`31.641` MiB). The resident bank grows with horizon, but the absolute latent-bank memory is small relative to the model's ~10.9 GiB peak.
- The honest systems claim for WorldMem is therefore: bounded RI/SLAM improve quality, cap the resident bank, and drastically reduce retrieval overhead, but total GPU memory is model-dominated because WorldMem stores compact latents.

Local plots for the 60s Pareto profile are generated by:

```bash
python utils/plot_worldmem_pareto_profile.py
```

Outputs:

```text
assets/plots/worldmem_pareto_walltime_gpu_memory_60s.png
assets/plots/worldmem_pareto_walltime_gpu_memory_60s.pdf
assets/plots/worldmem_retrieval_vs_bank_frames_60s.png
assets/plots/worldmem_retrieval_vs_bank_frames_60s.pdf
assets/plots/worldmem_memory_bank_pareto_combined_60s.png
assets/plots/worldmem_memory_bank_pareto_combined_60s.pdf
```

Speculative duration-scaling plot:

```bash
python utils/plot_worldmem_speculative_gpu_scaling.py
```

Outputs:

```text
assets/plots/worldmem_speculative_peak_gpu_scaling.png
assets/plots/worldmem_speculative_peak_gpu_scaling.pdf
assets/plots/worldmem_speculative_peak_gpu_scaling.csv
```

This plot uses the measured CECSL 10s and 60s profile anchors, then extrapolates a fitted model peak plus resident-bank size out to 180s. It shows two things at once: WorldMem's actual latent bank is tiny relative to the model peak, but an unbounded GPU-resident bank still grows with horizon while RI b32 stays capped. The RGB curves are speculative what-if variants, not measured WorldMem behavior.

Local videos are saved while each batch finishes, not only at the end of a long test run. For a run named `worldmem_unbounded_60s_n30`, inspect:

```text
/data/ab575577/worldmem/outputs/memory_policy/worldmem_unbounded_60s_n30/videos/test_vis/pred
/data/ab575577/worldmem/outputs/memory_policy/worldmem_unbounded_60s_n30/videos/test_vis/gt
```

The per-batch filenames look like `video_batch00000_0_rank0.mp4`, `video_batch00001_0_rank0.mp4`, and so on. This behavior is enabled by default in `scripts/run_worldmem_memory_policy_smoke.sh` with `SAVE_LOCAL_PER_BATCH=true`.

The memory-policy runner is restart-friendly by default. Before starting a run, it counts contiguous completed prediction MP4s in:

```text
<output_dir>/videos/test_vis/pred
```

If all requested videos are already present, it skips that run. If only the first `N` batch videos are present, it resumes from dataset index `N`, writes the next file as `video_batchNNNNN_0_rank0.mp4`, appends to the access trace, and only runs the remaining batches. This is controlled by:

```bash
RESUME_PARTIAL=1
SKIP_COMPLETED=1
```

Both are enabled by default. To force a clean rerun into the same output directory, use `RESUME_PARTIAL=0 SKIP_COMPLETED=0`, but be aware that this can overwrite existing per-batch videos and trace content.

Memory-policy sweep runs are video-only by default: W&B is disabled, eval metrics are not computed, test dataloading uses `TEST_NUM_WORKERS=0`, and only prediction MP4s are saved. This avoids the LPIPS/MSE/PSNR pass, avoids persistent worker buildup, skips unnecessary GT decoding, and prevents long `60s_n30` runs from holding decoded videos in RAM just to print a metrics table. To opt back into metrics later, set:

```bash
COMPUTE_EVAL_METRICS=true STREAM_EVAL_METRICS=true WANDB_MODE=offline
```

To also save ground-truth videos beside predictions, set:

```bash
SAVE_GT_VIDEO=true
```

For paper-style grids matching the MemCam setup, use 30 videos, durations 10/20/30/60 seconds, and budgets 32/64 for budgeted policies:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
NUM_VIDEOS=30 \
DURATIONS=10,20,30,60 \
POLICIES=unbounded,fifo,rarity_irreplaceability,slam_covisibility \
BUDGETS=32,64 \
bash scripts/run_worldmem_memory_policy_grid.sh
```

The duration is the generated future horizon at 10 FPS:

```text
10s -> 100 generated frames, N_FRAMES_VALID=700 with CONTEXT_FRAMES=600
20s -> 200 generated frames, N_FRAMES_VALID=800
30s -> 300 generated frames, N_FRAMES_VALID=900
60s -> 600 generated frames, N_FRAMES_VALID=1200
```

This full grid is large: 4 durations x 1 unbounded + 4 durations x 3 policies x 2 budgets = 28 runs, each over 30 videos. On CECSL, start with a small subset before launching the full grid:

```bash
NUM_VIDEOS=2 DURATIONS=10 POLICIES=unbounded,fifo BUDGETS=32 \
bash scripts/run_worldmem_memory_policy_grid.sh
```

To add the MemCam-style K-center baseline to the 60s WorldMem grid, run it explicitly so existing completed policy grids are not disturbed:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
NUM_VIDEOS=30 \
DURATIONS=60 \
POLICIES=kcenter_coreset \
BUDGETS=16,32,64,128 \
bash scripts/run_worldmem_memory_policy_grid.sh
```

Expected run directories:

```text
worldmem_kcenter_coreset_b16_60s_n30
worldmem_kcenter_coreset_b32_60s_n30
worldmem_kcenter_coreset_b64_60s_n30
worldmem_kcenter_coreset_b128_60s_n30
```

Check K-center completion:

```bash
cd /data/ab575577/worldmem/outputs/memory_policy

for r in \
  worldmem_kcenter_coreset_b16_60s_n30 \
  worldmem_kcenter_coreset_b32_60s_n30 \
  worldmem_kcenter_coreset_b64_60s_n30 \
  worldmem_kcenter_coreset_b128_60s_n30
do
  n=$(find "$r/videos/test_vis/pred" -name '*.mp4' 2>/dev/null | wc -l)
  if [ "$n" -ge 30 ]; then status="OK"; else status="MISSING"; fi
  printf "%-65s %3s  %s\n" "$r" "$n" "$status"
done
```

## 180s Pivot Runs

For the bounded-memory story, a useful next target is 180-second generation with fewer videos per run. With 600 context frames and 10 FPS, 180 seconds means:

```text
future frames = 1800
N_FRAMES_VALID = 600 + 1800 = 2400
required source frames/actions/poses = 100 initial skip + 2400 = 2500
```

Before launching the run, check whether the downloaded Minecraft test clips can actually supply that much ground-truth action/pose/video:

```bash
cd ~/WorldMem
conda activate worldmem

python utils/check_worldmem_horizon_availability.py \
  --data_dir data/minecraft \
  --future_seconds 180 \
  --context_frames 600 \
  --num_videos 15
```

If the first 15 selected videos are all `OK`, launch the tight 180s grid:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
NUM_VIDEOS=15 \
DURATIONS=180 \
POLICIES=unbounded,fifo,rarity_irreplaceability,slam_covisibility \
BUDGETS=32,64 \
bash scripts/run_worldmem_memory_policy_180s_tight.sh
```

That default tight grid produces seven runs:

```text
worldmem_unbounded_180s_n15
worldmem_fifo_b32_180s_n15
worldmem_fifo_b64_180s_n15
worldmem_rarity_irreplaceability_b32_180s_n15
worldmem_rarity_irreplaceability_b64_180s_n15
worldmem_slam_covisibility_b32_180s_n15
worldmem_slam_covisibility_b64_180s_n15
```

To match the MemCam 180s benchmark exactly, use budgets 16/32/64/128 for FIFO, RI, and SLAM-style covisibility, plus the unbounded baseline:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
NUM_VIDEOS=15 \
DURATIONS=180 \
POLICIES=unbounded,fifo,rarity_irreplaceability,slam_covisibility \
BUDGETS=16,32,64,128 \
bash scripts/run_worldmem_memory_policy_180s_tight.sh
```

This produces 13 runs:

```text
worldmem_unbounded_180s_n15
worldmem_fifo_b16_180s_n15
worldmem_fifo_b32_180s_n15
worldmem_fifo_b64_180s_n15
worldmem_fifo_b128_180s_n15
worldmem_rarity_irreplaceability_b16_180s_n15
worldmem_rarity_irreplaceability_b32_180s_n15
worldmem_rarity_irreplaceability_b64_180s_n15
worldmem_rarity_irreplaceability_b128_180s_n15
worldmem_slam_covisibility_b16_180s_n15
worldmem_slam_covisibility_b32_180s_n15
worldmem_slam_covisibility_b64_180s_n15
worldmem_slam_covisibility_b128_180s_n15
```

For a cheaper pilot before the full MemCam-matched grid, run 3 videos with only unbounded and the two smart policies at budget 32:

```bash
WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
NUM_VIDEOS=3 \
DURATIONS=180 \
POLICIES=unbounded,rarity_irreplaceability,slam_covisibility \
BUDGETS=32 \
bash scripts/run_worldmem_memory_policy_180s_tight.sh
```

To run the MemCam-matched 180s grid in round-robin order, use the dedicated round-robin runner. It keeps output directories stable at `..._180s_n15`, and each sweep asks every run to reach one additional completed video. This means sweep 1 generates video 1 for every run, sweep 2 generates video 2 for every run, and so on:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
TOTAL_VIDEOS=15 \
DURATIONS=180 \
POLICIES=unbounded,fifo,rarity_irreplaceability,slam_covisibility \
BUDGETS=16,32,64,128 \
bash scripts/run_worldmem_memory_policy_round_robin.sh
```

To stop after 7 preliminary videos per run, use:

```bash
TOTAL_VIDEOS=15 \
END_VIDEO=7 \
DURATIONS=180 \
POLICIES=unbounded,fifo,rarity_irreplaceability,slam_covisibility \
BUDGETS=16,32,64,128 \
bash scripts/run_worldmem_memory_policy_round_robin.sh
```

If the availability check reports `SHORT`, do not treat a 180s GT-backed run as valid yet. Options are: reduce the future horizon to the longest supported value, reduce the context length, generate longer Minecraft trajectories, or build a custom action/pose rollout for qualitative/self-consistency tests without future-GT metrics.

For FVD on the 180s runs:

```bash
cd ~/WorldMem
conda activate worldmem

RUNS_180=worldmem_unbounded_180s_n15,worldmem_fifo_b16_180s_n15,worldmem_fifo_b32_180s_n15,worldmem_fifo_b64_180s_n15,worldmem_fifo_b128_180s_n15,worldmem_rarity_irreplaceability_b16_180s_n15,worldmem_rarity_irreplaceability_b32_180s_n15,worldmem_rarity_irreplaceability_b64_180s_n15,worldmem_rarity_irreplaceability_b128_180s_n15,worldmem_slam_covisibility_b16_180s_n15,worldmem_slam_covisibility_b32_180s_n15,worldmem_slam_covisibility_b64_180s_n15,worldmem_slam_covisibility_b128_180s_n15

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
RUNS=$RUNS_180 \
EVAL_DURATIONS=30,60,120,180 \
METRICS_DIR=/data/ab575577/worldmem/outputs/memory_policy/metrics/fvd_prefix_180s_n15 \
bash scripts/evaluate_worldmem_fvd.sh
```

For LPIPS on the 180s runs:

```bash
cd ~/WorldMem
conda activate worldmem

RUNS_180=worldmem_unbounded_180s_n15,worldmem_fifo_b16_180s_n15,worldmem_fifo_b32_180s_n15,worldmem_fifo_b64_180s_n15,worldmem_fifo_b128_180s_n15,worldmem_rarity_irreplaceability_b16_180s_n15,worldmem_rarity_irreplaceability_b32_180s_n15,worldmem_rarity_irreplaceability_b64_180s_n15,worldmem_rarity_irreplaceability_b128_180s_n15,worldmem_slam_covisibility_b16_180s_n15,worldmem_slam_covisibility_b32_180s_n15,worldmem_slam_covisibility_b64_180s_n15,worldmem_slam_covisibility_b128_180s_n15

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
RUNS=$RUNS_180 \
EVAL_DURATIONS=30,60,120,180 \
METRICS_DIR=/data/ab575577/worldmem/outputs/memory_policy/metrics/lpips_prefix_180s_n15 \
bash scripts/evaluate_worldmem_lpips.sh
```

For CUT3R on the 180s runs:

```bash
cd ~/WorldMem
conda activate worldmem

RUNS_180=worldmem_unbounded_180s_n15,worldmem_fifo_b16_180s_n15,worldmem_fifo_b32_180s_n15,worldmem_fifo_b64_180s_n15,worldmem_fifo_b128_180s_n15,worldmem_rarity_irreplaceability_b16_180s_n15,worldmem_rarity_irreplaceability_b32_180s_n15,worldmem_rarity_irreplaceability_b64_180s_n15,worldmem_rarity_irreplaceability_b128_180s_n15,worldmem_slam_covisibility_b16_180s_n15,worldmem_slam_covisibility_b32_180s_n15,worldmem_slam_covisibility_b64_180s_n15,worldmem_slam_covisibility_b128_180s_n15

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
CUT3R_ROOT=$HOME/MemCam/CUT3R \
CUT3R_MODEL=$HOME/MemCam/CUT3R/src/cut3r_512_dpt_4_64.pth \
RUNS=$RUNS_180 \
DURATION_SEC=180 \
FRAME_STRIDE=30 \
MAX_FRAMES=120 \
RECON_DIR=/data/ab575577/worldmem/outputs/memory_policy/metrics/cut3r_pose_recon_180s_n15 \
METRICS_DIR=/data/ab575577/worldmem/outputs/memory_policy/metrics/cut3r_camera_metrics_180s_n15 \
bash scripts/run_worldmem_cut3r_metrics.sh
```

Newton Slurm array for the full 28-run grid:

```bash
WORLDMEM_DATA_DIR=/path/on/newton/minecraft \
WORLDMEM_STORAGE_ROOT=$HOME/worldmem_results \
NUM_VIDEOS=30 \
sbatch slurm/newton_worldmem_memory_policy_grid.sbatch
```

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

## Metrics For Memory-Policy Videos

WorldMem's released `evaluate.sh`/`infer.sh` setup uses `dataset.n_frames_valid=700` with `algorithm.context_frames=600`. At 10 FPS, that means their reproduced quantitative setup is 600 context/memory frames plus 100 generated future frames: 60 seconds of history and 10 seconds of generated video. This is different from the 60-second memory-policy runs here, which use `n_frames_valid=1200` and generate 600 future frames.

After the 60s/30-video policy runs have generated prediction MP4s, compute FVD prefix curves from the 60s videos. As of the CECSL check below, all seven target runs are complete:

```text
worldmem_unbounded_60s_n30
worldmem_fifo_b32_60s_n30
worldmem_fifo_b64_60s_n30
worldmem_rarity_irreplaceability_b32_60s_n30
worldmem_rarity_irreplaceability_b64_60s_n30
worldmem_slam_covisibility_b32_60s_n30
worldmem_slam_covisibility_b64_60s_n30
```

Verify completion on CECSL with:

```bash
cd /data/ab575577/worldmem/outputs/memory_policy

for r in \
  worldmem_unbounded_60s_n30 \
  worldmem_fifo_b32_60s_n30 \
  worldmem_fifo_b64_60s_n30 \
  worldmem_rarity_irreplaceability_b32_60s_n30 \
  worldmem_rarity_irreplaceability_b64_60s_n30 \
  worldmem_slam_covisibility_b32_60s_n30 \
  worldmem_slam_covisibility_b64_60s_n30
do
  printf "%-55s " "$r"
  find "$r/videos/test_vis/pred" -name '*.mp4' 2>/dev/null | wc -l
done
```

Then run FVD:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
EVAL_DURATIONS=10,20,30,60 \
bash scripts/evaluate_worldmem_fvd.sh
```

Outputs:

```text
/data/ab575577/worldmem/outputs/memory_policy/metrics/fvd_prefix/summary.csv
/data/ab575577/worldmem/outputs/memory_policy/metrics/fvd_prefix/summary.json
```

Observed CECSL FVD prefix results for the 60s/30-video runs, from `/data/ab575577/worldmem/outputs/memory_policy/metrics/fvd_prefix/summary.csv` on 2026-07-08. Lower is better.

| Run | FVD@10s | FVD@20s | FVD@30s | FVD@60s | FVD@60s - FVD@10s | Gap vs unbounded @60s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `worldmem_slam_covisibility_b32_60s_n30` | 672.178 | 583.125 | 629.683 | 696.808 | 24.630 | -2271.743 |
| `worldmem_rarity_irreplaceability_b64_60s_n30` | 629.728 | 578.193 | 676.294 | 766.039 | 136.311 | -2202.512 |
| `worldmem_rarity_irreplaceability_b32_60s_n30` | 620.664 | 580.070 | 625.066 | 778.176 | 157.512 | -2190.375 |
| `worldmem_slam_covisibility_b64_60s_n30` | 669.509 | 573.441 | 593.596 | 785.111 | 115.602 | -2183.440 |
| `worldmem_unbounded_60s_n30` | 743.733 | 820.271 | 1298.265 | 2968.551 | 2224.817 | 0.000 |
| `worldmem_fifo_b64_60s_n30` | 752.786 | 914.158 | 1491.072 | 3413.941 | 2661.155 | 445.390 |
| `worldmem_fifo_b32_60s_n30` | 872.445 | 1526.977 | 2152.052 | 3611.819 | 2739.373 | 643.268 |

Interpretation for the current paper story:

- Smart bounded memory is not just matching unbounded on FVD; RI and SLAM-style covisibility beat unbounded at every prefix and by more than 2000 FVD points at 60s.
- FIFO is a useful negative control: simply bounding memory is not enough. The choice of retained memories matters.
- The unbounded run degrades sharply with horizon (`60s - 10s = 2224.817`), while RI/SLAM remain much flatter. This supports the story that unbounded memory can accumulate harmful or conflicting retrieval candidates during long generation.
- This is FVD evidence only. Use CUT3R trajectory metrics, revisit consistency, and qualitative grids before making the stronger claim that bounded memory is generally better.

Observed CECSL FVD prefix results for the MemCam-matched 60s budget grid, first 15 videos per run, from `/data/ab575577/worldmem/outputs/memory_policy/metrics/fvd_prefix_60s_n15/summary.csv` on 2026-07-16. Lower is better.

| Run | FVD@10s | FVD@20s | FVD@30s | FVD@60s | Gap vs unbounded @60s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `worldmem_slam_covisibility_b16_60s_n30` | 1175.593 | 1086.814 | 1110.516 | 1041.757 | -2035.843 |
| `worldmem_slam_covisibility_b32_60s_n30` | 1179.954 | 1064.023 | 1048.373 | 1116.925 | -1960.675 |
| `worldmem_slam_covisibility_b64_60s_n30` | 1155.894 | 1125.879 | 1123.141 | 1128.462 | -1949.138 |
| `worldmem_rarity_irreplaceability_b32_60s_n30` | 1137.080 | 1085.518 | 1089.522 | 1160.428 | -1917.172 |
| `worldmem_rarity_irreplaceability_b64_60s_n30` | 1140.951 | 1119.389 | 1117.093 | 1165.354 | -1912.245 |
| `worldmem_rarity_irreplaceability_b16_60s_n30` | 1159.996 | 1138.900 | 1201.591 | 1238.744 | -1838.856 |
| `worldmem_rarity_irreplaceability_b128_60s_n30` | 1238.575 | 1221.720 | 1174.987 | 1250.561 | -1827.039 |
| `worldmem_slam_covisibility_b128_60s_n30` | 1254.551 | 1290.373 | 1372.669 | 1601.814 | -1475.786 |
| `worldmem_fifo_b128_60s_n30` | 1278.413 | 1399.131 | 1665.320 | 2604.960 | -472.640 |
| `worldmem_unbounded_60s_n30` | 1294.640 | 1376.429 | 1699.578 | 3077.600 | 0.000 |
| `worldmem_fifo_b32_60s_n30` | 1432.371 | 2022.649 | 2373.870 | 3554.909 | 477.309 |
| `worldmem_fifo_b64_60s_n30` | 1290.119 | 1480.887 | 1958.381 | 3821.737 | 744.137 |
| `worldmem_fifo_b16_60s_n30` | 1327.137 | 1644.809 | 2087.115 | 4205.032 | 1127.432 |

Interpretation of the 15-video all-budget FVD grid:

- The FVD result matches the LPIPS result: SLAM-style covisibility and RI beat unbounded across all tested budgets at 60s.
- The top 60s run is again `worldmem_slam_covisibility_b16_60s_n30`, with FVD `1041.757` versus unbounded `3077.600`, a gap of `-2035.843`.
- SLAM b16/b32/b64 are the top three by FVD@60s. RI b32/b64 follow closely.
- FIFO remains the control. FIFO b128 improves over unbounded, but FIFO b16/b32/b64 are worse at 60s, so the main win is not simply from bounding memory.
- As with LPIPS, b128 is not best for the selective policies. Smaller budgets can be cleaner because they remove stale or conflicting candidates before WorldMem's FOV retriever makes its local selection.

To compute LPIPS prefix curves on the same saved videos:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
EVAL_DURATIONS=10,20,30,60 \
bash scripts/evaluate_worldmem_lpips.sh
```

Outputs:

```text
/data/ab575577/worldmem/outputs/memory_policy/metrics/lpips_prefix/summary.csv
/data/ab575577/worldmem/outputs/memory_policy/metrics/lpips_prefix/summary.json
```

Print the LPIPS table and gaps against unbounded:

```bash
cd /data/ab575577/worldmem/outputs/memory_policy/metrics/lpips_prefix

python - <<'PY'
import pandas as pd
df = pd.read_csv("summary.csv")
p = df.pivot(index="run_name", columns="duration_sec", values="lpips")
u = p.loc["worldmem_unbounded_60s_n30"]
for d in [10, 20, 30, 60]:
    p[f"gap_vs_unbounded_{d}s"] = p[d] - u[d]
print(p.sort_values(60).to_string())
PY
```

This post-hoc LPIPS evaluator compares saved prediction MP4s to frame-aligned dataset GT, resizing GT to the prediction video resolution when needed. It is the right apples-to-apples comparison across memory policies. It is close to, but not identical to, WorldMem's original internal LPIPS path because the 60s memory-policy runs did not save VAE-decoded GT videos.

Observed CECSL LPIPS prefix results for the 60s/30-video runs, from `/data/ab575577/worldmem/outputs/memory_policy/metrics/lpips_prefix/summary.csv` on 2026-07-08. Lower is better.

| Run | LPIPS@10s | LPIPS@20s | LPIPS@30s | LPIPS@60s | Gap vs unbounded @60s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `worldmem_slam_covisibility_b32_60s_n30` | 0.514968 | 0.540618 | 0.561542 | 0.562467 | -0.102720 |
| `worldmem_rarity_irreplaceability_b64_60s_n30` | 0.514690 | 0.549945 | 0.566514 | 0.565072 | -0.100115 |
| `worldmem_slam_covisibility_b64_60s_n30` | 0.512701 | 0.542228 | 0.558144 | 0.567476 | -0.097711 |
| `worldmem_rarity_irreplaceability_b32_60s_n30` | 0.512983 | 0.551988 | 0.570109 | 0.573680 | -0.091507 |
| `worldmem_unbounded_60s_n30` | 0.512044 | 0.568492 | 0.604595 | 0.665187 | 0.000000 |
| `worldmem_fifo_b64_60s_n30` | 0.518849 | 0.565213 | 0.607281 | 0.698907 | 0.033720 |
| `worldmem_fifo_b32_60s_n30` | 0.528668 | 0.587703 | 0.634050 | 0.722850 | 0.057663 |

Interpretation:

- At 10s, unbounded and smart bounded policies are nearly tied, which is important because 10s is closest to WorldMem's released quantitative horizon.
- The gap opens with horizon. By 60s, RI/SLAM-style policies are about 0.09 to 0.10 LPIPS better than unbounded.
- FIFO again acts as a negative control: it is worse than unbounded at 30s/60s. The gain is from selective retention, not merely bounding memory.
- These numbers should not be compared directly to WorldMem's paper LPIPS (`0.1429`) because the paper uses its internal 10s eval path, while this is post-hoc MP4-vs-raw-GT evaluation. The policy comparison here is still apples-to-apples.

Observed CECSL LPIPS prefix results for the MemCam-matched 60s budget grid, first 15 videos per run, from `/data/ab575577/worldmem/outputs/memory_policy/metrics/lpips_prefix_60s_n15/summary.csv` on 2026-07-16. Lower is better.

| Run | LPIPS@10s | LPIPS@20s | LPIPS@30s | LPIPS@60s | Gap vs unbounded @60s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `worldmem_slam_covisibility_b16_60s_n30` | 0.495760 | 0.514547 | 0.535107 | 0.524506 | -0.127763 |
| `worldmem_slam_covisibility_b32_60s_n30` | 0.496085 | 0.517781 | 0.543519 | 0.533678 | -0.118591 |
| `worldmem_slam_covisibility_b64_60s_n30` | 0.495189 | 0.527610 | 0.548124 | 0.545439 | -0.106830 |
| `worldmem_rarity_irreplaceability_b32_60s_n30` | 0.492754 | 0.527723 | 0.550394 | 0.545953 | -0.106316 |
| `worldmem_rarity_irreplaceability_b64_60s_n30` | 0.498620 | 0.535419 | 0.555312 | 0.548573 | -0.103696 |
| `worldmem_rarity_irreplaceability_b16_60s_n30` | 0.500104 | 0.534284 | 0.560150 | 0.565720 | -0.086549 |
| `worldmem_rarity_irreplaceability_b128_60s_n30` | 0.497253 | 0.536124 | 0.558436 | 0.566730 | -0.085539 |
| `worldmem_slam_covisibility_b128_60s_n30` | 0.500675 | 0.543008 | 0.571432 | 0.577360 | -0.074909 |
| `worldmem_fifo_b128_60s_n30` | 0.501106 | 0.559536 | 0.595753 | 0.647241 | -0.005028 |
| `worldmem_unbounded_60s_n30` | 0.505903 | 0.568854 | 0.601760 | 0.652269 | 0.000000 |
| `worldmem_fifo_b64_60s_n30` | 0.507510 | 0.559912 | 0.602527 | 0.687605 | 0.035336 |
| `worldmem_fifo_b32_60s_n30` | 0.523885 | 0.576564 | 0.612585 | 0.688773 | 0.036504 |
| `worldmem_fifo_b16_60s_n30` | 0.518241 | 0.563879 | 0.607389 | 0.717445 | 0.065176 |

Interpretation of the 15-video all-budget LPIPS grid:

- Every selective bounded-memory policy, RI and SLAM-style covisibility at budgets 16/32/64/128, beats unbounded at every prefix from 10s through 60s.
- The strongest 60s run is `worldmem_slam_covisibility_b16_60s_n30`: LPIPS `0.524506`, an absolute improvement of `0.127763` over unbounded (`0.652269`), about a 19.6% relative reduction.
- SLAM-style covisibility is strongest in this WorldMem grid, especially at budgets 16 and 32. RI remains consistently better than unbounded, with b32/b64 closest to SLAM.
- FIFO stays useful as a negative control. FIFO b128 nearly ties unbounded at 60s, but FIFO b16/b32/b64 are worse at 60s. This supports the claim that selective retention matters, not merely imposing a smaller memory cap.
- Bigger budget is not automatically better. For SLAM and RI, b128 is worse than smaller budgets, which matches the hypothesis that too many retained frames can reintroduce confusing or conflicting retrieval candidates.

Local plots for the all-budget 60s/15-video grid are generated from the pasted CECSL summary tables by:

```bash
python utils/plot_worldmem_memory_policy_metrics.py
```

Outputs:

```text
assets/plots/worldmem_lpips_prefix_60s_n15.png
assets/plots/worldmem_lpips_prefix_60s_n15.pdf
assets/plots/worldmem_fvd_prefix_60s_n15.png
assets/plots/worldmem_fvd_prefix_60s_n15.pdf
```

For CUT3R camera trajectory metrics, use the MemCam/CUT3R checkout and checkpoint. Start with a smoke subset:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
CUT3R_ROOT=$HOME/MemCam/CUT3R \
CUT3R_MODEL=$HOME/MemCam/CUT3R/src/cut3r_512_dpt_4_64.pth \
LIMIT=1 \
bash scripts/run_worldmem_cut3r_metrics.sh
```

If the smoke passes, run the full 30-video set:

```bash
WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
CUT3R_ROOT=$HOME/MemCam/CUT3R \
CUT3R_MODEL=$HOME/MemCam/CUT3R/src/cut3r_512_dpt_4_64.pth \
bash scripts/run_worldmem_cut3r_metrics.sh
```

CUT3R outputs:

```text
/data/ab575577/worldmem/outputs/memory_policy/metrics/cut3r_pose_recon/
/data/ab575577/worldmem/outputs/memory_policy/metrics/cut3r_camera_metrics/cut3r_camera_summary.csv
/data/ab575577/worldmem/outputs/memory_policy/metrics/cut3r_camera_metrics/cut3r_camera_metrics.csv
```

The WorldMem metric scripts reconstruct matching ground-truth frames/poses from the Minecraft test split using the same deterministic seed (`42`), initial skipped frames (`100`), and context length (`600`) used for generation. The saved prediction videos contain only the generated future; the scripts compare frame `k` of each MP4 to dataset frame `100 + 600 + k`.

## Revisit Candidates

Before using pixel-level revisit matching, check whether the selected WorldMem test trajectories actually revisit earlier context poses. This script scans the GT pose traces for future frames that return near a pose from the 600-frame context:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
NUM_VIDEOS=30 \
FUTURE_SECONDS=60 \
CONTEXT_FRAMES=600 \
POS_THRESHOLD=1.0 \
YAW_THRESHOLD=20 \
bash scripts/analyze_worldmem_revisits.sh
```

Outputs:

```text
/data/ab575577/worldmem/outputs/memory_policy/metrics/revisit_candidates/revisit_summary.csv
/data/ab575577/worldmem/outputs/memory_policy/metrics/revisit_candidates/revisit_pairs.csv
/data/ab575577/worldmem/outputs/memory_policy/metrics/revisit_candidates/revisit_details.jsonl
```

Use `revisit_pairs.csv` as the candidate set for later pixel/LPIPS/L2 matching. If this scan finds few or no pairs, then the WorldMem Minecraft test set is weak for a revisit-place metric and CUT3R/trajectory consistency should carry more of the geometry story.

## Common Issues

- `ModuleNotFoundError`: reactivate `conda activate worldmem`, then reinstall missing packages.
- `pyrealsense2` install failure: it is listed in `requirements.txt`, but the inspected train/infer/eval paths do not import it. If it blocks setup, install the remaining requirements and revisit only if data generation needs it.
- CUDA architecture errors: reinstall a newer PyTorch CUDA wheel, especially on the CECSL A6000 Pro/newer GPU.
- `ImportError: cannot import name 'read_video' from 'torchvision.io'`: newer `torchvision` versions removed the eager `read_video` import. This repo has been patched to lazily import `read_video` and fall back to OpenCV in `algorithms/worldmem/models/utils.py`.
- `AttributeError: 'float' object has no attribute 'detach'` in `_accumulate_stream_metrics`: LPIPS can return a Python float while MSE/PSNR return tensors. This repo's streaming metric path now handles both, but memory-policy sweeps disable eval metrics by default.
- Home directory fills up: re-check `HF_HOME`, `WANDB_DIR`, `WANDB_CACHE_DIR`, `TMPDIR`, and `+output_dir`.
- Dataset has zero samples: check that `training`, `validation`, and `test` contain `.mp4` files, and that every video has a matching `.npz` action/pose file.
- W&B entity error: pass `wandb.entity=local` for offline tests or set your real W&B entity for online logging.
- Interrupted Hugging Face download keeps waiting on `.lock` files: find the old downloader with `pgrep -af 'hf download|huggingface-cli|snapshot_download|huggingface_hub'`, stop it with `kill <PID>`, then use `kill -9 <PID>` only if it ignores the first kill. After no downloader remains, remove stale lock files with `find /data/ab575577/worldmem/data/minecraft/.cache/huggingface/download -name '*.lock' -type f -delete`.
