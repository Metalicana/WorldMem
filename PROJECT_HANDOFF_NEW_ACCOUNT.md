# WorldMem Project Handoff

Last updated: 2026-08-22

Read this file first in a new Codex/GPT account. Then read
`RUNNING_CECSL_NEWTON.md`, which is the exhaustive command log and runbook.

## Collaboration Contract

The source of truth is the local checkout:

```text
/Users/metalicana/projects_summer_2026/WorldMem
```

The normal workflow is:

1. Codex studies and edits the local checkout.
2. Codex performs all feasible local static checks and focused tests.
3. The user reviews, commits, and pushes the changes. Do not push on the user's
   behalf unless explicitly requested.
4. The user pulls the repository on CECSL or Newton and runs the GPU commands.
5. The user pastes logs/results back; Codex diagnoses them and updates the local
   code and runbook.

Important working preferences:

- Keep `RUNNING_CECSL_NEWTON.md` updated with durable commands, results, and
  caveats. Do not leave important project state only in chat.
- Give complete copy-pasteable shell commands, usually including `cd`, conda
  activation, GPU selection, storage roots, and a timestamped `tee` log.
- Never use the shell `column` command for displaying CSV results. Use a short
  pandas command with `to_string(index=False)`, or `python -m json.tool` for JSON.
- Be direct when the user asks for a command or status. Do not repeatedly explain
  background they already understand.
- Think carefully before making a mechanism claim. Distinguish correlation,
  matched observational evidence, and causal intervention.
- Do not jump from one successful example to a paper-level conclusion. State the
  sample size and protocol limitations.
- Long generation must save one video as each batch completes and be resumable.
  The user has lost long jobs before and does not want all outputs buffered until
  the end.
- Generation sweeps normally disable W&B and online metrics. The user wants the
  videos first, with metrics computed offline afterward.
- Do not assume GPU 0 is free. Other projects, especially VMem, may occupy it.
  Honor the requested GPU; recent WorldMem mechanism runs use GPU 1.
- Do not run destructive git commands or discard unrelated user changes.

This local machine is an inspection/editing sandbox and does not run the full
WorldMem model. Full inference is run by the user on CECSL or Newton.

## Repositories

Local sibling repositories:

```text
../WorldMem
../MemCam
../vmem
../spmem
../VBench
```

Relevant CECSL checkouts:

```text
~/WorldMem
~/MemCam
~/vmem
```

The WorldMem remote is:

```text
https://github.com/Metalicana/WorldMem
```

Current branch at handoff: `main`.

The proposal/context PDF is `Radi-Summer-2026-1.pdf`. MemCam is the main policy
reference implementation. CUT3R is under `~/MemCam/CUT3R` on CECSL.

## Machines And Storage

### CECSL PC

Login observed in logs:

```text
ab575577@CECSL4622128797
```

Repository and environment:

```text
Repo:  ~/WorldMem
Conda: worldmem
Python: 3.10
```

Observed working CUDA stack:

```text
torch: 2.11.0+cu128
torch CUDA: 12.8
GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
compute capability: 12.0
```

The CECSL machine has two GPUs. A previous `nvidia-smi` showed about 97.9 GiB per
GPU. Always check occupancy before launching:

```bash
nvidia-smi
```

All large data, caches, outputs, and logs belong under:

```text
/data/ab575577/worldmem
```

Standard CECSL environment variables:

```bash
export WORLDMEM_ROOT=/data/ab575577/worldmem
export WORLDMEM_DATA=$WORLDMEM_ROOT/data
export HF_HOME=$WORLDMEM_ROOT/hf_cache
export WANDB_DIR=$WORLDMEM_ROOT/wandb
export WANDB_CACHE_DIR=$WORLDMEM_ROOT/wandb/cache
export TMPDIR=$WORLDMEM_ROOT/tmp
```

Repository data link:

```text
~/WorldMem/data/minecraft -> /data/ab575577/worldmem/data/minecraft
```

Main output root:

```text
/data/ab575577/worldmem/outputs
```

Main log root:

```text
/data/ab575577/worldmem/logs
```

### Newton

Newton is a regular cluster and does not have `/data/ab575577`. Never put that
path in a Newton command. Use an explicit project/scratch root, for example:

```bash
export WORLDMEM_STORAGE_ROOT=$SCRATCH/worldmem
export WORLDMEM_DATA_DIR=$SCRATCH/worldmem/data/minecraft
```

The scripts generally detect whether `/data/ab575577` exists, but explicit
Newton paths are safer. This checkout does not currently include a complete
`configurations/cluster/*.yaml`, so cluster jobs may need direct shell/Slurm
wrapping according to Newton policy.

## Dataset And Horizon

Hugging Face dataset:

```text
zeqixiao/worldmem_minecraft_dataset
```

CECSL data root:

```text
/data/ab575577/worldmem/data/minecraft
```

Expected splits:

```text
training/
validation/
test/
```

The completed local dataset was approximately 421 GiB. A final check showed
12,307 MP4 files and 12,307 NPZ files overall after stale incomplete files were
removed.

Minecraft test clips contain 1,501 usable frames at 10 FPS. The WorldMem runs use
600 context frames. Therefore:

- 60-second future generation is supported with `n_frames_valid=1200`.
- 180-second future generation is not GT-supported. It would require 2,500
  source frames/actions/poses, and zero of 473 test videos met that horizon.
- MemCam experiments used 180 seconds, but WorldMem comparisons must use 60
  seconds unless a new action/pose source and evaluation protocol are designed.

The saved prediction MP4 contains only the generated future. Offline GT alignment
maps prediction frame `k` to dataset frame `100 + 600 + k`, matching the loader's
100-frame initial skip and 600-frame context.

## Research Goal

The project benchmarks budgeted external memory for autoregressive long-video
generation. MemCam, WorldMem, and VMem use a growing past-memory bank and retrieve
a small subset for the next generated chunk. The project asks whether selective
bounded memory can match or beat that unbounded bank while reducing retrieval
cost and guaranteeing constant bank size.

MemCam's matched protocol was:

```text
Policies: FIFO, rarity x irreplaceability (RI), SLAM-style, unbounded
Budgets: 16, 32, 64, 128
Videos: 15
Horizon: 180 seconds
```

WorldMem uses the same policy/budget logic where possible, but its supported
evaluation horizon is 60 seconds. K-center and MCE were later added as additional
selective baselines.

The current paper story is not merely "smaller memory saves resources." The
stronger observed phenomenon is:

> Selective bounded memory can outperform unbounded memory because eviction can
> prevent corrupted autoregressive history from remaining available for future
> retrieval and conditioning.

FIFO is essential as a negative control because FIFO often loses to unbounded.
This shows that bounding alone is not sufficient; retained-memory quality and
coverage matter.

## WorldMem Memory Semantics

The released WorldMem method is best described as unbounded storage with bounded
per-step retrieval:

- The initial 600-frame latent context is retained.
- Newly generated latent chunks are appended to history.
- The unbounded candidate bank therefore grows with time.
- The model does not condition on the entire bank at once. It retrieves
  `memory_condition_length=8` frames for each next chunk.
- Retrieval is pose/FOV-overlap based, with a small recency preference.
- Budgeted policies first restrict the retained candidate bank; WorldMem's own
  retriever then selects eight references from that bank.

Supported policies in `algorithms/worldmem/memory_policies.py`:

```text
unbounded
random_cap
fifo
rarity_irreplaceability
slam_covisibility
kcenter_coreset
mce
```

Budgets normally tested: `16,32,64,128`.

### RI Representations

Default WorldMem RI is Latent-RI, not DINO-RI:

```text
generated VAE latent [16,H,W]
  -> adaptive average pool [16,4,4]
  -> flatten [256]
  -> cosine distances
```

Both rarity and irreplaceability use this pooled latent space. Label it
`Latent-RI` in analysis.

Optional backends are implemented:

```text
MEMORY_FEATURE_BACKEND=latent
MEMORY_FEATURE_BACKEND=dino
MEMORY_FEATURE_BACKEND=dino_rgb
```

The DINO path correctly decodes the scaled WorldMem latent through the VAE,
converts to RGB `[0,1]`, runs frozen `facebook/dinov2-base`, L2 normalizes the
feature, and caches it by frame index. `dino_rgb` uses DINO for rarity and 64x64
RGB distance for irreplaceability to match MemCam more closely.

## Implemented Engineering

Core implementation files:

```text
algorithms/worldmem/df_video.py
algorithms/worldmem/memory_policies.py
algorithms/worldmem/memory_diagnostics.py
algorithms/worldmem/models/utils.py
```

Important capabilities now implemented:

- FIFO, random-cap, Latent/DINO RI, SLAM covisibility, K-center, MCE, unbounded.
- CPU-resident and GPU-resident memory-bank analysis modes.
- Per-batch local video saving and resume/skip of completed videos.
- Output batch offsets and deterministic dataset/per-video/policy seeds.
- JSONL access traces for retrieval, eviction, candidate diagnostics, bank state,
  timing, and memory quality.
- Retrieval cost optimization: bounded policies compute FOV overlap only for
  retained candidates, while unbounded scores all prior frames.
- Offline FVD and LPIPS prefix evaluation at 10/20/30/60 seconds.
- GPU peak-memory and timing profiling plus Pareto plotting.
- Revisit-candidate analysis.
- CUT3R wrapper and GT sanity evaluator.
- VBench and VBench-Long wrappers.
- Retrieved-memory quality tracing using decoded actual stored latents.
- Fixed-history GT memory-cleaning causal replay.

Relevant scripts:

```text
scripts/run_worldmem_memory_policy_smoke.sh
scripts/run_worldmem_memory_policy_grid.sh
scripts/run_worldmem_memory_policy_round_robin.sh
scripts/audit_worldmem_memory_policy_runs.sh
scripts/evaluate_worldmem_fvd.sh
scripts/evaluate_worldmem_lpips.sh
scripts/profile_worldmem_gpu_memory.sh
scripts/run_worldmem_memory_mechanisms.sh
scripts/analyze_worldmem_memory_mechanisms.sh
scripts/run_worldmem_retrieved_memory_quality.sh
scripts/analyze_worldmem_retrieved_memory_quality.sh
scripts/run_worldmem_gt_memory_replay.sh
```

Focused tests:

```text
tests/test_worldmem_memory_policies.py
tests/test_worldmem_memory_diagnostics.py
```

Full model/GPU execution cannot be validated locally. Previous focused policy
and diagnostic tests, Python compilation, Bash syntax, Hydra dry runs, and diff
checks passed. Run a one-video CECSL pilot after behaviorally meaningful changes.

## Main Quality Results

The fair all-budget comparison uses the first 15 videos from every run, 60-second
future generation, and eight retrieved memory frames. Some directory names end
in `_n30` even when only the first 15 videos are used for matched metrics.

### LPIPS At 60 Seconds

Lower is better.

| Policy | b16 | b32 | b64 | b128 |
| --- | ---: | ---: | ---: | ---: |
| FIFO | 0.717 | 0.689 | 0.688 | 0.647 |
| RI | 0.566 | 0.546 | 0.549 | 0.567 |
| SLAM covisibility | **0.525** | **0.534** | **0.545** | 0.577 |
| K-center | 0.545 | 0.559 | 0.575 | **0.559** |
| MCE | 0.576 | 0.575 | 0.596 | 0.604 |
| Unbounded | - | - | - | 0.652 |

Best observed result: SLAM b16, `0.524506`, versus unbounded `0.652269`, a
`0.127763` absolute or approximately 19.6% relative LPIPS reduction.

### FVD At 60 Seconds

Lower is better. The first-15-video matched results include:

| Policy | Budget | FVD@60s |
| --- | ---: | ---: |
| SLAM covisibility | 16 | **1041.757** |
| SLAM covisibility | 32 | 1116.925 |
| SLAM covisibility | 64 | 1128.462 |
| RI | 32 | 1160.428 |
| RI | 64 | 1165.354 |
| RI | 16 | 1238.744 |
| RI | 128 | 1250.561 |
| SLAM covisibility | 128 | 1601.814 |
| FIFO | 128 | 2604.960 |
| Unbounded | - | 3077.600 |
| FIFO | 32 | 3554.909 |
| FIFO | 64 | 3821.737 |
| FIFO | 16 | 4205.032 |

Absolute FVD values are very high because this is a long 60-second post-hoc
protocol, not WorldMem's reported short-horizon evaluation. Use FVD for the
matched policy comparison and horizon trend, not as a claim that the number is
directly comparable to another paper.

WorldMem's paper LPIPS (`0.1429`) is also not directly comparable to the current
post-hoc MP4-versus-raw-GT LPIPS. The paper's reproduced setup uses 600 context
frames plus only 100 generated frames, approximately 10 seconds.

Local plots:

```text
assets/plots/worldmem_lpips_prefix_60s_n15.{png,pdf}
assets/plots/worldmem_fvd_prefix_60s_n15.{png,pdf}
```

Generate them with:

```bash
python utils/plot_worldmem_memory_policy_metrics.py
```

## Systems Results

The released inference path keeps most latent history on CPU, so unbounded does
not cause GPU memory to grow in the default path. Do not claim that released
unbounded WorldMem OOMs the GPU.

Measured one-video 60-second profile:

| Bank | Policy | Wall sec | Retrieval sec | Peak bank | Device peak |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | unbounded | 748 | 44.010 | CPU | 10921 MiB |
| CPU | RI b32 | 716 | 3.050 | CPU | 10915 MiB |
| GPU | unbounded | 746 | 44.440 | 31.641 MiB / 1200 frames | 10953 MiB |
| GPU | RI b32 | 711 | 2.705 | 0.826 MiB / 32 frames | 10917 MiB |

Honest claim:

- Bounded memory guarantees constant bank size.
- RI reduces retrieval time by roughly 14-16x.
- End-to-end latency improves modestly, roughly 32-35 seconds or 4-5%, because
  diffusion sampling dominates.
- Total peak GPU memory is model-dominated because WorldMem stores compact
  latents. The latent-bank OOM story is weak for this model and resolution.
- RGB-bank scaling figures are speculative extrapolations and must be labeled as
  such.

Profiling/plotting utilities:

```text
scripts/profile_worldmem_gpu_memory.sh
utils/plot_worldmem_pareto_profile.py
utils/plot_worldmem_speculative_gpu_scaling.py
```

## Geometry Metrics Caveats

### Revisit Metric

The first 30 selected WorldMem test trajectories produced zero revisit candidates
under position threshold `1.0` and yaw threshold `20` degrees. Do not force a
pixel-revisit metric onto these trajectories. A custom action trajectory would be
needed for deliberate revisits.

### CUT3R

CUT3R inference was made runnable, including its CUDA extension and PyTorch 2.6+
checkpoint-loading compatibility. However, CUT3R produced huge camera errors and
zero WorldScore camera-control score even on GT Minecraft frames. The GT sanity
check therefore failed.

Do not use current CUT3R numbers in a paper claim until pose convention/alignment
is fixed and GT sanity produces sensible errors. The full troubleshooting and
commands are in `RUNNING_CECSL_NEWTON.md`.

## Mechanism Findings

### Zero-Overlap Fallback

Earlier diagnostics found:

- Roughly 70-80% of retrieval calls have exactly zero geometric overlap for all
  candidates under the current WorldMem retrieval geometry.
- Unbounded and FIFO then inherit recency-biased tie behavior.
- Content-based eviction removes fresh redundant frames, so its zero-overlap
  fallback can reach farther into history.
- SLAM median fallback age across budgets 16/32/64/128 was approximately
  `213 / 81 / 28 / 13` frames, matching its LPIPS trend.
- Raising Monte Carlo overlap precision from 1x to 50x did not remove winner
  flips. These appear to be geometric ties, not only sampling noise.
- Current WorldMem geometry uses radius 30, 10,000 samples, and 52.5 by 37.5
  degree half-FOV. Radius 50 remains an open controlled ablation.

This finding is useful but did not fully explain the quality gap. The newer
corrupted-memory experiment is stronger.

### Retrieved-Memory Quality Pilot

One matched 60-second trajectory was traced. The late window is 45-60 seconds.
The table below evaluates only retrieved references that came from generated
frames, then scores the following generated chunk.

| Policy | Generated reference fraction | Retrieved PSNR | Retrieved SSIM | Retrieved LPIPS | Next-chunk LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| SLAM b16 | 0.744 | **15.09** | **0.453** | **0.421** | **0.661** |
| RI b32 | 0.892 | 11.01 | 0.345 | 0.693 | 0.775 |
| FIFO b128 | 1.000 | 7.36 | 0.217 | 0.760 | 0.761 |
| Unbounded | 0.935 | 6.60 | 0.099 | 0.914 | 0.914 |

Worst-decile retrieved-memory LPIPS was `0.694` for SLAM, `0.781` for RI,
`0.802` for FIFO, and `0.996` for unbounded.

This is matched observational evidence that unbounded retrieves much more
corrupted generated latents and that the immediately following chunks are worse.
Because the generated-only comparison still favors SLAM, the result is not
explained solely by SLAM retrieving more clean initial-context frames.

### Fixed-History GT Memory-Cleaning Replay

Pilot intervention:

```text
trajectory/batch: 0
target frame: 1054
selected frames: 1053,1052,1051,1050,1049,1048,1047,1046
```

The control used the originally selected generated memory latents. The cleaned
branch kept the same selected IDs, prior generated history, actions, poses,
noise, diffusion schedule, and RNG, but replaced those eight memory contents with
VAE-encoded GT frames from the same indices.

Validity checks passed. Immediate next-chunk deltas, GT-cleaned minus control:

| Metric | Delta | Improvement direction |
| --- | ---: | --- |
| PSNR | +2.727 dB | positive |
| SSIM | +0.0479 | positive |
| LPIPS | -0.273 | negative |

This is causal evidence that corrupted retrieved memory harmed the next chunk at
this selected event. It is only one event, so it is not an average treatment
effect. Its printed confidence interval is degenerate and must not be treated as
inferential uncertainty.

Pilot output roots:

```text
/data/ab575577/worldmem/outputs/memory_quality_60s_pilot
/data/ab575577/worldmem/outputs/gt_memory_replay_60s_pilot_v2
```

### Full 15-Trajectory Observational Result

The 15-video run completed for unbounded, FIFO b128, RI b32, and SLAM b16. Late
45-60 second paired differences against unbounded were:

| Policy | Retrieved PSNR | Retrieved SSIM | Retrieved LPIPS | Next PSNR | Next SSIM | Next LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RI b32 | +4.464 | +0.115 | -0.250 | +6.777 | +0.150 | -0.270 |
| SLAM b16 | +9.980 | +0.217 | -0.416 | +5.413 | +0.184 | -0.331 |
| FIFO b128 | -1.177 | -0.082 | +0.046 | -0.481 | -0.069 | +0.019 |

All six late-window RI and SLAM 95% trajectory-bootstrap intervals exclude zero
in the beneficial direction. FIFO does not improve, preserving the negative
control. The raw retrieval-to-following correlations use 9,000 one-frame steps
per run, but those steps are clustered and autocorrelated within 15 trajectories;
do not claim `n=9000` independent samples.

The generated-only late-window comparison is also decisive:

| Policy | Generated-reference fraction | Generated-only PSNR | Generated-only SSIM | Generated-only LPIPS | Worst-decile LPIPS | Next LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SLAM b16 | 0.875 | **22.928** | **0.611** | **0.175** | **0.521** | **0.273** |
| RI b32 | 0.864 | 16.615 | 0.491 | 0.369 | 0.771 | 0.335 |
| Unbounded | 0.977 | 13.255 | 0.399 | 0.587 | 0.949 | 0.604 |
| FIFO b128 | 1.000 | 12.453 | 0.324 | 0.621 | 0.977 | 0.623 |

Thus, SLAM's gain is not only from retrieving fewer generated references. Among
generated references themselves, it retrieves much cleaner content. RI has a
slightly lower generated-reference fraction than SLAM but worse generated-only
quality, further separating exposure frequency from retained-content quality.

The replay manifest selects high-corruption late events from four distinct
unbounded trajectories: batches `0,12,7,14`, at 45.4-59.7 seconds, with
retrieved-generated LPIPS `1.003,0.991,0.941,0.915`.

## Highest-Priority Next Work

The full 15-video retrieved-memory quality run is complete. Its generation
command is retained below for reproducibility or resume:

```bash
cd ~/WorldMem
conda activate worldmem
mkdir -p /data/ab575577/worldmem/logs

GPU=1 \
WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
OUTPUT_ROOT=/data/ab575577/worldmem/outputs/memory_quality_60s \
NUM_VIDEOS=15 \
GLOBAL_SEED=101 \
DATASET_SEED=42 \
POLICY_SPECS=unbounded:,fifo:128,rarity_irreplaceability:32,slam_covisibility:16 \
bash scripts/run_worldmem_retrieved_memory_quality.sh \
  2>&1 | tee /data/ab575577/worldmem/logs/memory_quality_60s_n15_gpu1_$(date +%F_%H%M).log
```

The runner is resume-aware. The completed analysis command that selected four
high-corruption unbounded events from distinct trajectories was:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
OUTPUT_ROOT=/data/ab575577/worldmem/outputs/memory_quality_60s \
METRICS_DIR=/data/ab575577/worldmem/outputs/memory_quality_60s/metrics/retrieved_memory_quality \
LATE_START_SEC=45 \
REPLAY_COUNT=4 \
bash scripts/analyze_worldmem_retrieved_memory_quality.sh
```

Then run the four fixed-history causal replays:

```bash
cd ~/WorldMem
conda activate worldmem

GPU=1 \
WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
QUALITY_ROOT=/data/ab575577/worldmem/outputs/memory_quality_60s \
MANIFEST=/data/ab575577/worldmem/outputs/memory_quality_60s/metrics/retrieved_memory_quality/gt_replay_manifest.json \
OUTPUT_ROOT=/data/ab575577/worldmem/outputs/gt_memory_replay_60s \
COMPUTE_DINO=false \
bash scripts/run_worldmem_gt_memory_replay.sh \
  2>&1 | tee /data/ab575577/worldmem/logs/gt_memory_replay_gpu1_$(date +%F_%H%M).log
```

Success criteria:

- Replay validity passes for every event.
- PSNR and SSIM deltas are positive.
- LPIPS delta is negative.
- Report per-event results, mean/median, source-video bootstrap or a clearly
  labeled small-sample interval, and the number of trajectories.

Four events are an exploratory replication gate. For a final paper claim, prefer
one prespecified late event from all 15 trajectories if compute allows. Consider
adding neutral/low-corruption events as a specificity control rather than
selecting only extreme events.

Other open work:

1. Radius-50 controlled retrieval probe.
2. VBench and VBench-Long runs; wrappers exist but no results are recorded.
3. Repair CUT3R GT pose sanity before using CUT3R metrics.
4. Explain why `rarity_neighbors=8` helped MemCam but barely changed WorldMem MCE.
5. Trace K-center/MCE fallback age if needed.

## New Admission + Retention Policy

Update: the learned causal-consistency gate below is a superseded experiment.
MemCam's no-reference estimators and pose-calibrated DINO consistency both
failed held-out validation and must not be used for reported runs.

The latest method is `causal_consistency_coverage_ri`, initially at budget 32.
It is not the old MemCam `reliable_slam_ri` heuristic. It has two separate
decisions:

1. Admission uses a pose-calibrated, overlap-weighted pooled-latent residual
   against the exact multiple parents retrieved by WorldMem.
2. Retention independently min-max normalizes WorldMem's existing geometric
   coverage and RI scores, then uses `0.75 * G + 0.25 * RI` for Top-B retention.

Implementation files:

- `algorithms/worldmem/causal_memory_gate.py`
- `algorithms/worldmem/memory_policies.py`
- `algorithms/worldmem/df_video.py`
- `utils/calibrate_worldmem_causal_gate.py`
- `scripts/calibrate_worldmem_causal_gate.sh`

The workflow is shadow generation, offline trajectory-disjoint calibration,
then enforced generation. Shadow mode admits every candidate and records GT
labels only for calibration. Enforced mode requires an approved frozen artifact
and never reads GT when deciding admission. Full CECSL and Newton commands are
in `RUNNING_CECSL_NEWTON.md`.

The current hypothesis is coverage-hysteretic admission: preserve an older
persistent representative when a later generated frame already covers its view.
Before implementing it, run
`scripts/validate_worldmem_coverage_hysteresis.sh` on the existing unbounded
60-second videos. The validator derives WorldMem's actual one-frame chunks from
access traces when available. The original unbounded rollout predates retrieval
tracing, so its fallback reconstructs chunks from the 600-frame saved prediction
using the repository's configured `chunk_size: 1` and records this source in
`video_inventory.csv`. It excludes the clean input context, sweeps geometric thresholds
0.80/0.85/0.90/0.95, compares each older/later frame against its own exact-index
GT using PSNR/SSIM, and bootstraps trajectory means. Runtime policy work was
gated on a positive, threshold-robust result, including the final quarter.

That validation is now complete and positive on all 30 trajectories as a
population-level result. Every threshold in 0.80/0.85/0.90/0.95 had positive
trajectory-bootstrap PSNR/SSIM confidence intervals, including 45-60 seconds.
At the primary threshold 0.90, older representatives were better by +0.847 dB
PSNR/+0.047 SSIM overall and +1.207 dB/+0.065 SSIM late, with median temporal
gaps of 43 and 77 frames respectively. The effect is heterogeneous rather than
universal across trajectories.

WorldMem now implements `coverage_hysteresis`, following MemCam: camera-only
sequential admission at threshold 0.90, same-chunk admitted candidates becoming
immediate references, normalized 0.75 SLAM-coverage/0.25 latent-RI retention,
and older-incumbent tie preservation. WorldMem does not need MemCam's transient
endpoint bank slot because its native recent sliding window remains unchanged.
The primary CECSL B32/60s/n15 command is documented in
`RUNNING_CECSL_NEWTON.md`.

## Common Operational Lessons

- Activate `conda activate worldmem` before every run.
- Use `python -m pip`, not bare `pip`, to ensure the active environment is used.
- Keep only one Hugging Face downloader process. Stale `.lock` files caused long
  waits during the 421 GiB dataset download.
- New torchvision removed eager `torchvision.io.read_video`; this repo lazily
  imports it and falls back to OpenCV.
- The config key is misspelled upstream as `seperate_load`; preserve that spelling
  when writing raw Hydra commands.
- PyTorch 2.6+ changed `torch.load` to `weights_only=True`; the trusted local
  CUT3R wrapper explicitly handles the old checkpoint.
- CUT3R's old `curope` CUDA code needed the deprecated `tokens.type()` dispatch
  fixed for modern PyTorch and an explicit Blackwell architecture build.
- GT replay once failed because GT latents were FP16 and the control tensor was
  FP32. The replay now casts GT references to the destination device/dtype before
  masked replacement.
- A status of `0` means success in the profiling CSV; a status of `1` means the
  command failed.
- Sampling dominates runtime. A large retrieval speedup produces only a modest
  end-to-end speedup.

## Git State At Handoff

At the time this handoff was created:

```text
branch: main
HEAD: 9f2e12c
```

`HEAD` contains the GT replay dtype fix. `RUNNING_CECSL_NEWTON.md` has additional
uncommitted notes recording the successful one-event causal replay, and this
handoff file is new/uncommitted. The user should review, commit, and push them,
then pull on CECSL before continuing.
