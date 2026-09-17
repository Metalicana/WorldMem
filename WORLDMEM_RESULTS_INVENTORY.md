# WorldMem Results Inventory

Updated: 2026-09-15

Paper: **KEEPSAKE: Selective Spatial Memory for Long-Horizon Video Generation**.
The final KEEPSAKE method is the SLAM-style Geometric Coverage policy; its
strongest WorldMem configuration is B16.

This file consolidates the measured WorldMem results currently recorded in the
repository and CECSL logs. It separates usable paper results from pilots,
speculative extrapolations, and invalid metrics.

## Evaluation Protocol

- Dataset: WorldMem Minecraft test split.
- Prediction horizon: 60 seconds (600 generated frames at 10 FPS).
- Initial context: 600 frames.
- Retrieved conditioning set: 8 frames.
- Main matched sample: exact generated batch IDs 0 through 14.
- Budgets: 16, 32, 64, and 128 retained frames.
- LPIPS: frame-aligned generated MP4 versus exact-index dataset GT.
- FVD: cached StyleGAN-V I3D, 16-frame clips, 4 clips per video, stride 4,
  image size 224.
- Standard VBench: six prompt-independent dimensions evaluated on exact batch
  IDs 0 through 14 for every cell.
- Lower LPIPS and FVD are better. Higher VBench scores are better.

The post-hoc 60-second LPIPS/FVD values are comparable across these WorldMem
runs, but not directly to MemCam or the original short-horizon WorldMem paper.

## Complete 60-Second Budget Sweep

| Policy | Budget | LPIPS | FVD |
| --- | ---: | ---: | ---: |
| Unbounded | - | 0.652269 | 3077.599804 |
| FIFO | 16 | 0.717445 | 4205.032292 |
| FIFO | 32 | 0.688773 | 3554.909072 |
| FIFO | 64 | 0.687605 | 3821.736563 |
| FIFO | 128 | 0.647241 | 2604.960196 |
| Latent-RI | 16 | 0.565720 | 1238.743749 |
| Latent-RI | 32 | 0.545953 | 1160.427844 |
| Latent-RI | 64 | 0.548573 | 1165.354452 |
| Latent-RI | 128 | 0.566730 | 1250.560853 |
| Geometric Coverage | 16 | **0.524506** | **1041.756572** |
| Geometric Coverage | 32 | 0.533678 | 1116.924792 |
| Geometric Coverage | 64 | 0.545439 | 1128.461550 |
| Geometric Coverage | 128 | 0.577360 | 1601.813935 |
| K-center | 16 | 0.544560 | 1166.073559 |
| K-center | 32 | 0.558800 | 1419.578751 |
| K-center | 64 | 0.574565 | 1664.465529 |
| K-center | 128 | 0.558839 | 1533.881086 |
| MCE | 16 | 0.575549 | 1459.740206 |
| MCE | 32 | 0.574797 | 1923.971752 |
| MCE | 64 | 0.595627 | 2258.660089 |
| MCE | 128 | 0.603571 | 2173.207329 |

Best overall is Geometric Coverage B16. Relative to Unbounded, it reduces
LPIPS by 0.127763 (19.6%) and FVD by 2035.843232 (66.1%). At the fixed B32
cross-system comparison, Geometric Coverage is also best.

## Standard VBench, 15 Matched Videos

| Policy | Budget | Subject | Background | Motion | Dynamic | Aesthetic | Imaging |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unbounded | - | 0.7387 | 0.8753 | 0.9725 | 0.6667 | 0.3837 | 0.6201 |
| FIFO | 16 | 0.6842 | 0.8321 | 0.9738 | 0.8000 | 0.3526 | 0.6106 |
| FIFO | 32 | 0.7247 | 0.8615 | 0.9735 | 0.7333 | 0.3692 | 0.6302 |
| FIFO | 64 | 0.7350 | 0.8654 | 0.9736 | 0.7333 | 0.3690 | 0.6327 |
| FIFO | 128 | 0.7617 | 0.8816 | 0.9754 | 0.6000 | 0.3864 | 0.6206 |
| Latent-RI | 16 | 0.8094 | 0.9295 | 0.9734 | **0.8667** | 0.4332 | **0.6668** |
| Latent-RI | 32 | 0.8116 | 0.9246 | 0.9750 | **0.8667** | 0.4294 | 0.6493 |
| Latent-RI | 64 | 0.8027 | 0.9223 | 0.9746 | 0.8000 | 0.4311 | 0.6536 |
| Latent-RI | 128 | 0.8028 | 0.9224 | 0.9734 | 0.7333 | 0.4293 | 0.6617 |
| Geometric Coverage | 16 | **0.8230** | **0.9346** | 0.9716 | 0.7333 | **0.4409** | 0.6626 |
| Geometric Coverage | 32 | 0.8171 | 0.9295 | 0.9739 | 0.7333 | 0.4372 | 0.6598 |
| Geometric Coverage | 64 | 0.8102 | 0.9192 | 0.9747 | 0.7333 | 0.4286 | 0.6356 |
| Geometric Coverage | 128 | 0.7941 | 0.9063 | 0.9743 | 0.6667 | 0.4135 | 0.6496 |
| K-center | 16 | 0.7955 | 0.9156 | 0.9755 | 0.8000 | 0.4218 | 0.6214 |
| K-center | 32 | 0.7960 | 0.9158 | 0.9739 | 0.7333 | 0.4304 | 0.6338 |
| K-center | 64 | 0.7956 | 0.9119 | 0.9748 | 0.7333 | 0.4214 | 0.6296 |
| K-center | 128 | 0.8043 | 0.9189 | 0.9727 | 0.7333 | 0.4246 | 0.6604 |
| MCE | 16 | 0.7762 | 0.9059 | 0.9750 | 0.7333 | 0.4113 | 0.6311 |
| MCE | 32 | 0.7912 | 0.9030 | 0.9749 | 0.8000 | 0.4154 | 0.6443 |
| MCE | 64 | 0.7835 | 0.8990 | 0.9754 | 0.8000 | 0.4027 | 0.6370 |
| MCE | 128 | 0.7838 | 0.8986 | **0.9764** | 0.8000 | 0.4057 | 0.6310 |

All 21 cells completed with exactly 15 matched videos. Geometric Coverage B16
is best on subject consistency, background consistency, and aesthetic quality.
Latent-RI B16/B32 is best on dynamic degree, Latent-RI B16 is best on imaging
quality, and MCE B128 narrowly leads motion smoothness. Unbounded does not lead
any measured VBench dimension.

Paper-ready budget-sweep figures are generated from
`assets/results/worldmem_budget_sweep_60s_n15.csv` with:

```bash
python utils/plot_worldmem_complete_metric_sweep.py
```

Outputs in `assets/plots/`:

- `worldmem_lpips_fvd_budget_sweep_60s_n15.{png,pdf}`
- `worldmem_vbench_budget_sweep_60s_n15.{png,pdf}`
- `worldmem_vbench_radar_highlights_60s_n15.{png,pdf}`
- `worldmem_complete_metric_budget_sweep_60s_n15.{png,pdf}`

The radar chart is the cleaner paper-facing VBench summary. It compares
Unbounded with the strongest Latent-RI and Geometric Coverage configurations.
Each axis is expressed relative to Unbounded at 100; the radial axis starts at
90 and should therefore be described as a relative profile, not an absolute
cross-metric scale. The six-panel line plot remains the complete budget-sweep
record for the appendix.

## LPIPS Prefix Curves

| Policy | Budget | 10s | 20s | 30s | 60s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unbounded | - | 0.505903 | 0.568854 | 0.601760 | 0.652269 |
| FIFO | 16 | 0.518241 | 0.563879 | 0.607389 | 0.717445 |
| FIFO | 32 | 0.523885 | 0.576564 | 0.612585 | 0.688773 |
| FIFO | 64 | 0.507510 | 0.559912 | 0.602527 | 0.687605 |
| FIFO | 128 | 0.501106 | 0.559536 | 0.595753 | 0.647241 |
| Latent-RI | 16 | 0.500104 | 0.534284 | 0.560150 | 0.565720 |
| Latent-RI | 32 | 0.492754 | 0.527723 | 0.550394 | 0.545953 |
| Latent-RI | 64 | 0.498620 | 0.535418 | 0.555312 | 0.548573 |
| Latent-RI | 128 | 0.497253 | 0.536124 | 0.558436 | 0.566730 |
| Geometric Coverage | 16 | 0.495760 | 0.514547 | 0.535106 | 0.524506 |
| Geometric Coverage | 32 | 0.496085 | 0.517781 | 0.543519 | 0.533678 |
| Geometric Coverage | 64 | 0.495189 | 0.527610 | 0.548124 | 0.545439 |
| Geometric Coverage | 128 | 0.500675 | 0.543008 | 0.571433 | 0.577360 |
| K-center | 16 | 0.494573 | 0.520839 | 0.549154 | 0.544560 |
| K-center | 32 | 0.496180 | 0.526247 | 0.552772 | 0.558800 |
| K-center | 64 | 0.488704 | 0.526353 | 0.553901 | 0.574565 |
| K-center | 128 | 0.489405 | 0.533085 | 0.558121 | 0.558839 |
| MCE | 16 | 0.506518 | 0.547532 | 0.577127 | 0.575549 |
| MCE | 32 | 0.505700 | 0.542828 | 0.567654 | 0.574797 |
| MCE | 64 | 0.498490 | 0.547219 | 0.572505 | 0.595627 |
| MCE | 128 | 0.496287 | 0.546787 | 0.568957 | 0.603571 |

## FVD Prefix Curves Recorded Locally

The repository currently records all prefixes for Unbounded, FIFO, Latent-RI,
and Geometric Coverage. The complete CECSL summary also contains K-center and
MCE prefixes, but only their 60-second values have been copied into this local
inventory so far.

| Policy | Budget | 10s | 20s | 30s | 60s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unbounded | - | 1294.640098 | 1376.429092 | 1699.577615 | 3077.599804 |
| FIFO | 16 | 1327.136552 | 1644.809272 | 2087.115469 | 4205.032292 |
| FIFO | 32 | 1432.370689 | 2022.648811 | 2373.869870 | 3554.909072 |
| FIFO | 64 | 1290.118592 | 1480.887390 | 1958.380615 | 3821.736563 |
| FIFO | 128 | 1278.412947 | 1399.130980 | 1665.320119 | 2604.960196 |
| Latent-RI | 16 | 1159.996193 | 1138.900076 | 1201.590948 | 1238.743749 |
| Latent-RI | 32 | 1137.079588 | 1085.517834 | 1089.521562 | 1160.427844 |
| Latent-RI | 64 | 1140.950734 | 1119.389292 | 1117.093006 | 1165.354452 |
| Latent-RI | 128 | 1238.575079 | 1221.719983 | 1174.987174 | 1250.560853 |
| Geometric Coverage | 16 | 1175.592628 | 1086.813921 | 1110.516295 | 1041.756572 |
| Geometric Coverage | 32 | 1179.953901 | 1064.023359 | 1048.372693 | 1116.924792 |
| Geometric Coverage | 64 | 1155.893563 | 1125.879012 | 1123.141370 | 1128.461550 |
| Geometric Coverage | 128 | 1254.551435 | 1290.372697 | 1372.669186 | 1601.813935 |

To print and copy the missing K-center/MCE prefixes from CECSL without using
`column`:

```bash
python - <<'PY'
import pandas as pd

path = "/data/ab575577/worldmem/outputs/memory_policy/metrics/fvd_budget_sweep_60s_n15/summary.csv"
df = pd.read_csv(path)
names = df["run_name"].str.contains("kcenter|mce", regex=True)
print(df.loc[names, ["run_name", "duration_sec", "fvd"]].to_string(index=False))
PY
```

## Fixed-B32 Paper Comparison

| Policy | LPIPS@60s | FVD@60s |
| --- | ---: | ---: |
| Unbounded | 0.652269 | 3077.599804 |
| FIFO-32 | 0.688773 | 3554.909072 |
| Latent-RI-32 | 0.545953 | 1160.427844 |
| Geometric Coverage-32 | **0.533678** | **1116.924792** |
| K-center-32 | 0.558800 | 1419.578751 |
| MCE-32 | 0.574797 | 1923.971752 |

This table is the locked cross-system comparison. The full budget sweep is a
separate sensitivity result; do not substitute each policy's test-optimal
budget into the fixed-B32 table.

## Systems Profile

Measured on one 60-second video. Device peak is total `nvidia-smi` usage.

| Bank | Policy | Wall sec | Retrieval sec | Sampling sec | Resident bank | Device peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CPU | Unbounded | 748 | 44.010 | 680.023 | CPU | 10921 MiB |
| CPU | Latent-RI B32 | 716 | 3.050 | 688.344 | CPU | 10915 MiB |
| GPU | Unbounded | 746 | 44.440 | 678.447 | 31.641 MiB / 1200 frames | 10953 MiB |
| GPU | Latent-RI B32 | 711 | 2.705 | 685.963 | 0.826 MiB / 32 frames | 10917 MiB |

RI reduces retrieval time by roughly 14-16x and wall time by 32-35 seconds
(about 4-5%). Total GPU memory is model-dominated; the honest claim is constant
bank size and lower retrieval overhead, not that released Unbounded necessarily
OOMs the GPU.

## Retrieved-Memory Quality, 15 Trajectories

Late window: 45-60 seconds. The deltas below are bounded minus Unbounded.
Positive PSNR/SSIM and negative LPIPS are improvements. Brackets are
trajectory-bootstrap 95% confidence intervals.

| Policy | Retrieved PSNR delta | Retrieved SSIM delta | Retrieved LPIPS delta | Following PSNR delta | Following SSIM delta | Following LPIPS delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Latent-RI B32 | +4.464 [0.477, 9.189] | +0.115 [0.033, 0.212] | -0.250 [-0.384, -0.135] | +6.777 [2.360, 12.288] | +0.150 [0.060, 0.260] | -0.270 [-0.417, -0.138] |
| Geometric Coverage B16 | +9.980 [6.560, 13.782] | +0.217 [0.122, 0.317] | -0.416 [-0.543, -0.287] | +5.413 [2.977, 8.223] | +0.184 [0.083, 0.294] | -0.331 [-0.476, -0.201] |
| FIFO B128 | -1.177 [-3.336, 0.828] | -0.082 [-0.174, -0.005] | +0.046 [-0.060, 0.155] | -0.481 [-2.628, 1.547] | -0.069 [-0.162, 0.010] | +0.019 [-0.087, 0.127] |

Generated-reference-only late-window values:

| Policy | Generated fraction | PSNR | SSIM | LPIPS | Worst-decile LPIPS | Following LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Geometric Coverage B16 | 0.875 | **22.928** | **0.611** | **0.175** | **0.521** | **0.273** |
| Latent-RI B32 | 0.864 | 16.615 | 0.491 | 0.369 | 0.771 | 0.335 |
| Unbounded | 0.977 | 13.255 | 0.399 | 0.587 | 0.949 | 0.604 |
| FIFO B128 | 1.000 | 12.453 | 0.324 | 0.621 | 0.977 | 0.623 |

Descriptive late LPIPS correlations between retrieved quality and following
quality are 0.951 for RI, 0.668 for Geometric Coverage, and 0.987 for
Unbounded. The 9,000 frame steps per run are clustered within 15 trajectories
and are not 9,000 independent observations.

## Fixed-History GT Memory-Cleaning Replay

One validated causal pilot used batch 0 at target frame 1054. It retained the
same retrieved IDs, history, actions, poses, diffusion noise, and RNG, replacing
only eight generated memory latents with VAE-encoded exact-index GT latents.

| Metric | GT-cleaned minus control |
| --- | ---: |
| PSNR | +2.727 dB |
| SSIM | +0.0479 |
| LPIPS | -0.273 |

This is causal evidence at one selected event, not an average treatment effect.
The four-event manifest targets batches 0, 12, 7, and 14 at 45.4, 55.4, 59.7,
and 59.6 seconds, with retrieved-generated LPIPS 1.003, 0.991, 0.941, and 0.915.

## Geometry And Retrieval Diagnostics

- Zero-overlap retrieval calls: approximately 70-80% across tested policies.
- Geometric Coverage median fallback age at B16/B32/B64/B128:
  213/81/28/13 frames.
- Latent-RI B32 fallback age: 111 frames.
- Increasing Monte Carlo overlap precision from 1x to 50x did not eliminate
  winner flips.
- Current geometry: radius 30, 10,000 samples, half-FOV 52.5 by 37.5 degrees.
- Revisit candidates: 0 of 30 trajectories at position threshold 1.0 and yaw
  threshold 20 degrees.

Coverage-hysteresis older-versus-later validation on 30 Unbounded trajectories:

| Threshold | Subset | Pairs | PSNR delta | SSIM delta |
| ---: | --- | ---: | ---: | ---: |
| 0.80 | all | 15494 | +1.004728 | +0.059143 |
| 0.80 | 45-60s | 4045 | +1.464320 | +0.093911 |
| 0.85 | all | 15223 | +0.897847 | +0.052760 |
| 0.85 | 45-60s | 3955 | +1.252252 | +0.078520 |
| 0.90 | all | 14436 | +0.846829 | +0.046951 |
| 0.90 | 45-60s | 3748 | +1.207168 | +0.065213 |
| 0.95 | all | 13723 | +0.461980 | +0.025954 |
| 0.95 | 45-60s | 3534 | +0.654487 | +0.035569 |

The offline diagnostic was positive, but Coverage-Hysteresis is excluded from
the final queue because the corresponding completed MemCam runtime policy lost
on the broader metric suite.

## Result Status And Exclusions

| Item | Status |
| --- | --- |
| Complete LPIPS budget sweep | Valid, 21 cells, 15 matched videos |
| Complete FVD budget sweep | Valid, 21 cells, 15 matched videos |
| Standard VBench | Valid, 21 cells, 15 matched videos |
| VBench-Long | Corrected original-video grouping and full 21-cell evaluator ready; results pending |
| CUT3R generated-video metrics | Invalid: GT Minecraft sanity failed |
| Pixel revisit metric | Unavailable: zero candidates in selected trajectories |
| Rarity-only B32 | 15-video generation exists; metrics not recorded here |
| SLAM-rarity 75/25 B32 | Implementation exists; completion not recorded here |
| Coverage-Hysteresis B32 | 15-video generation exists; excluded from final roster |
| RGB memory scaling | Speculative extrapolation, not measured behavior |

## Completing The WorldMem Metric Matrix

The target follows the MemCam budget grid: Unbounded plus FIFO, Latent-RI,
Geometric Coverage, K-center, and MCE at budgets 16, 32, 64, and 128. Every
metric uses generated batch IDs 0 through 14, for 21 cells and 315 matched
policy-video evaluations. Existing extra videos in the older N30 directories
are excluded by staging.

LPIPS and FVD are already complete. Do not regenerate videos or rerun those
metrics unless their saved summaries fail the final status audit.

CECSL does not currently have a `vbench` Conda environment. Create it once,
separately from the working `worldmem` environment. The pinned VBench checkout
has a stale `setup.py` guard that rejects CUDA newer than 12.1, so do not run
`pip install .`; the evaluation wrappers execute `evaluate.py` directly.

```bash
cd ~

if [ ! -d "$HOME/VBench/.git" ]; then
  git clone https://github.com/Vchitect/VBench.git "$HOME/VBench"
fi

cd "$HOME/VBench"
git fetch origin
git checkout 45e79ec14e69a2187202c675d2dbce1a71843d53

conda create -n vbench python=3.10 pip -y
conda activate vbench

python -m pip install --upgrade pip wheel
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install "moviepy==1.0.3" av \
  "dreamsim==0.2.1" "peft==0.7.1" "transformers==4.33.2"
```

DreamSim does not cap PEFT, so an unconstrained install can pull a newer PEFT
that is incompatible with VBench's pinned `transformers==4.33.2`. If the
environment already has `peft==0.20.0`, repair it in place with:

```bash
conda activate vbench
python -m pip install "peft==0.7.1" "transformers==4.33.2"
python -m pip check
```

Verify imports and Blackwell GPU execution before the full evaluation:

```bash
conda activate vbench
cd "$HOME/VBench"

python - <<'PY'
import torch
import vbench
import av
import decord
import dreamsim
from moviepy.editor import VideoFileClip

print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("decord:", decord.__version__)
x = torch.ones(1, device="cuda")
print("CUDA tensor:", x)
print("VBench imports: OK")
PY
```

The wrappers place VBench checkpoints and framework caches under
`/data/ab575577/worldmem/` on CECSL. Newton automatically uses
`$HOME/worldmem_results/` instead.

Run standard VBench on CECSL GPU 0:

```bash
cd ~/WorldMem
conda activate vbench
export CUDA_VISIBLE_DEVICES=0

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
bash scripts/run_worldmem_vbench.sh \
  2>&1 | tee /data/ab575577/worldmem/logs/vbench_budget_sweep_60s_n15_$(date +%F_%H%M).log
```

Smoke-test corrected VBench-Long on Unbounded:

```bash
cd ~/WorldMem
conda activate vbench
export CUDA_VISIBLE_DEVICES=0

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
RUNS=worldmem_unbounded_60s_n30 \
bash scripts/run_worldmem_vbench_long.sh \
  2>&1 | tee /data/ab575577/worldmem/logs/vbench_long_unbounded_smoke_$(date +%F_%H%M).log
```

After the smoke succeeds, run the full VBench-Long grid. The completed
Unbounded cell is verified and skipped:

```bash
cd ~/WorldMem
conda activate vbench
export CUDA_VISIBLE_DEVICES=0

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
bash scripts/run_worldmem_vbench_long.sh \
  2>&1 | tee /data/ab575577/worldmem/logs/vbench_long_budget_sweep_60s_n15_$(date +%F_%H%M).log
```

After both finish, build the complete machine-readable status table without
using `column`:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
bash scripts/build_worldmem_final_metric_status.sh

python - <<'PY'
import pandas as pd

path = "/data/ab575577/worldmem/outputs/memory_policy/metrics/final_status/worldmem_final_metric_status.csv"
print(pd.read_csv(path).to_string(index=False))
PY
```

CUT3R must remain absent/invalid in the final table until the GT Minecraft
sanity run produces sensible camera errors and writes an explicit passing
`validity.json`. A completed reconstruction process alone is not evidence that
the camera metric is valid.

## CPU-Only Retrieval-Failure Figures

`utils/visualize_worldmem_retrieval_failures.py` scans the completed 15-video
retrieved-memory-quality runs over 45-60 seconds. It creates two complementary
five-column galleries plus individual strips:

- `actual_history`: each policy's selected memory is read from its own rollout,
  showing what the model actually consumed.
- `common_source`: both selected frame identities are read from the same
  Unbounded rollout, isolating retrieval choice from prior rollout quality.

Each WorldMem chunk retrieves eight frames. Chunks are ranked using the mean
quality gap across all generated retrieved items, while the displayed image is
explicitly labeled as the worst retrieved generated item. The images are
selected extremes, not estimates of failure frequency.

Run on CECSL without using either GPU:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
MAX_EXAMPLES=5 \
bash scripts/visualize_worldmem_retrieval_failures.sh \
  2>&1 | tee /data/ab575577/worldmem/logs/retrieval_failure_gallery_$(date +%F_%H%M).log
```

Default output:

```text
/data/ab575577/worldmem/outputs/memory_quality_60s/metrics/retrieval_failure_gallery_geocov_b16/
```

The output contains combined PNG/PDF figures, individual case strips, complete
candidate CSVs, selected-case CSVs, and a protocol report. The wrapper sets
`CUDA_VISIBLE_DEVICES` to empty and imports no Torch models.

For the main paper, use the simpler same-frame KEEPSAKE snowballing figure:

```bash
cd ~/WorldMem
conda activate worldmem

WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
bash scripts/visualize_worldmem_keepsake_snowballing.sh \
  2>&1 | tee /data/ab575577/worldmem/logs/keepsake_snowballing_4col_$(date +%F_%H%M).log
```

This produces `Ground truth | Unbounded | FIFO | KEEPSAKE (ours)` using the
same batch and exact future frame in every column. The five-column selected
memory figure remains a retrieval-mechanism diagnostic for the supplement.

The original WorldMem paper LPIPS value recorded in the handoff is 0.1429 at
its short evaluation horizon. It is not directly comparable to this post-hoc
60-second MP4-versus-raw-GT protocol. The original paper did not provide the
same 60-second FVD experiment.
