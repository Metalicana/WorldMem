# WorldMem Final Metrics

Updated: 2026-09-18.

Paper: **KEEPSAKE: Selective Spatial Memory for Long-Horizon Video Generation**.
KEEPSAKE is `slam_covisibility`, labeled Geometric Coverage in the source CSV.

## Scope

- 60-second generation, 600 generated frames at 10 FPS, 600 initial context frames.
- 15 videos per configuration, generated batch IDs 0 through 14.
- Unbounded plus five bounded policies at budgets 16, 32, 64, and 128: 21 configurations.
- Final quality metrics: LPIPS, FVD, and six standard VBench dimensions.
- VBench-Long is excluded by request. CUT3R is excluded because the GT sanity test failed.

Source: [saved metric CSV](assets/results/worldmem_budget_sweep_60s_n15.csv).
These are recorded aggregate results, not a new verification of remote metric files.
The retrieval-trace provenance audit is separate and remains unresolved for some historical runs.
The inventory contains prefix curves and mechanism results:
[WorldMem results inventory](WORLDMEM_RESULTS_INVENTORY.md).

## Complete Quality Table

LPIPS and FVD: lower is better. All VBench dimensions: higher is better.
Subject = subject consistency; Background = background consistency;
Motion = motion smoothness; Dynamic = dynamic degree;
Aesthetic = aesthetic quality; Imaging = imaging quality.
VBench dimensions remain on their recorded 0-1 scale. `VBench-6` is a derived
custom-input aggregate: each of the six available dimensions is normalized with
the VBench leaderboard bounds, Dynamic receives weight 0.5, the other dimensions
receive weight 1.0, and the weighted sum is divided by 5.5. It is not the full
official VBench Quality or Total Score because temporal flickering and the
semantic dimensions were not evaluated.

| Policy | Budget | Videos | LPIPS | FVD | Subject | Background | Motion | Dynamic | Aesthetic | Imaging |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unbounded | - | 15 | 0.652269 | 3077.599804 | 0.7387 | 0.8753 | 0.9725 | 0.6667 | 0.3837 | 0.6201 |
| FIFO | 16 | 15 | 0.717445 | 4205.032292 | 0.6842 | 0.8321 | 0.9738 | 0.8000 | 0.3526 | 0.6106 |
| FIFO | 32 | 15 | 0.688773 | 3554.909072 | 0.7247 | 0.8615 | 0.9735 | 0.7333 | 0.3692 | 0.6302 |
| FIFO | 64 | 15 | 0.687605 | 3821.736563 | 0.7350 | 0.8654 | 0.9736 | 0.7333 | 0.3690 | 0.6327 |
| FIFO | 128 | 15 | 0.647241 | 2604.960196 | 0.7617 | 0.8816 | 0.9754 | 0.6000 | 0.3864 | 0.6206 |
| Latent-RI | 16 | 15 | 0.565720 | 1238.743749 | 0.8094 | 0.9295 | 0.9734 | 0.8667 | 0.4332 | 0.6668 |
| Latent-RI | 32 | 15 | 0.545953 | 1160.427844 | 0.8116 | 0.9246 | 0.9750 | 0.8667 | 0.4294 | 0.6493 |
| Latent-RI | 64 | 15 | 0.548573 | 1165.354452 | 0.8027 | 0.9223 | 0.9746 | 0.8000 | 0.4311 | 0.6536 |
| Latent-RI | 128 | 15 | 0.566730 | 1250.560853 | 0.8028 | 0.9224 | 0.9734 | 0.7333 | 0.4293 | 0.6617 |
| KEEPSAKE | 16 | 15 | 0.524506 | 1041.756572 | 0.8230 | 0.9346 | 0.9716 | 0.7333 | 0.4409 | 0.6626 |
| KEEPSAKE | 32 | 15 | 0.533678 | 1116.924792 | 0.8171 | 0.9295 | 0.9739 | 0.7333 | 0.4372 | 0.6598 |
| KEEPSAKE | 64 | 15 | 0.545439 | 1128.461550 | 0.8102 | 0.9192 | 0.9747 | 0.7333 | 0.4286 | 0.6356 |
| KEEPSAKE | 128 | 15 | 0.577360 | 1601.813935 | 0.7941 | 0.9063 | 0.9743 | 0.6667 | 0.4135 | 0.6496 |
| K-center | 16 | 15 | 0.544560 | 1166.073559 | 0.7955 | 0.9156 | 0.9755 | 0.8000 | 0.4218 | 0.6214 |
| K-center | 32 | 15 | 0.558800 | 1419.578751 | 0.7960 | 0.9158 | 0.9739 | 0.7333 | 0.4304 | 0.6338 |
| K-center | 64 | 15 | 0.574565 | 1664.465529 | 0.7956 | 0.9119 | 0.9748 | 0.7333 | 0.4214 | 0.6296 |
| K-center | 128 | 15 | 0.558839 | 1533.881086 | 0.8043 | 0.9189 | 0.9727 | 0.7333 | 0.4246 | 0.6604 |
| MCE | 16 | 15 | 0.575549 | 1459.740206 | 0.7762 | 0.9059 | 0.9750 | 0.7333 | 0.4113 | 0.6311 |
| MCE | 32 | 15 | 0.574797 | 1923.971752 | 0.7912 | 0.9030 | 0.9749 | 0.8000 | 0.4154 | 0.6443 |
| MCE | 64 | 15 | 0.595627 | 2258.660089 | 0.7835 | 0.8990 | 0.9754 | 0.8000 | 0.4027 | 0.6370 |
| MCE | 128 | 15 | 0.603571 | 2173.207329 | 0.7838 | 0.8986 | 0.9764 | 0.8000 | 0.4057 | 0.6310 |

## VBench-6 Main-Table Aggregate

These values use the recorded B32 dimension means above. Recalculate from the
raw CECSL result JSONs for the final paper values so rounding in the recorded
four-decimal CSV cannot affect the last displayed decimal.

| Model | VBench-6 (%) |
| --- | ---: |
| WorldMem | 68.66 |
| WorldMem + FIFO B32 | 68.61 |
| WorldMem + MCE B32 | 72.84 |
| WorldMem + K-center B32 | 72.67 |
| **WorldMem + RI B32** | **74.77** |
| WorldMem + KEEPSAKE B32 | 74.05 |

Recalculate and export from the matched raw VBench outputs on CECSL:

```bash
cd ~/WorldMem
conda activate vbench

python utils/calculate_worldmem_vbench6.py \
  --root /data/ab575577/worldmem/outputs/memory_policy/metrics/vbench_budget_sweep_60s_n15 \
  --limit 15 \
  --output /data/ab575577/worldmem/outputs/memory_policy/metrics/vbench_budget_sweep_60s_n15/worldmem_vbench6_main.csv
```

## Reading the Results

KEEPSAKE B16 has the lowest LPIPS and FVD, and the highest subject consistency,
background consistency, and aesthetic quality across this sweep. Latent-RI B16
has the highest imaging quality; Latent-RI B16/B32 tie for the highest dynamic
degree. MCE B128 has the highest motion smoothness. These are aggregate rankings,
not significance tests, and dynamic degree does not by itself establish fidelity.

Use B32 for the locked cross-system comparison. Keep the complete sweep separate
from that fixed-budget table rather than choosing every policy's best test budget.

## Five Figure Variations

Generate locally on CPU:

```bash
python -m utils.plot_worldmem_metric_variations
```

Output directory: `assets/plots/metric_variations/`. Each treatment has PNG/PDF
exports; `00_comparison_sheet.png` previews all five.

1. `01_budget_sweep`: LPIPS, FVD, and background consistency versus budget;
   absolute scores, with dashed unbounded references.
2. `02_full_dashboard`: all eight metrics versus budget, with the same scales
   and styles per metric as the main sweep.
3. `03_relative_heatmap`: all 20 bounded cells and all eight metrics. Values are
   signed relative differences from unbounded: `(baseline-value)/baseline` for
   LPIPS/FVD and `(value-baseline)/baseline` for VBench, multiplied by 100.
   This is not a composite score or a significance test.
4. `04_fixed_b32_dumbbells`: every metric at the locked B32 budget, with absolute
   scores connected to the unbounded reference.
5. `05_tradeoff_scatter`: LPIPS versus FVD and aesthetic versus imaging quality;
   every bounded budget is shown and labeled, with marker area encoding budget.

All figures use saved aggregates without confidence intervals; per-video data
would be required to estimate uncertainty. KEEPSAKE remains green throughout.

## Bar-Based Alternatives

The second design pass separates the fixed-budget comparison from the full
sweep and uses zero-based bar axes:

```bash
python -m utils.plot_worldmem_metric_bars
```

Outputs in `assets/plots/metric_bars/`, with PNG/PDF exports:

- `01_editorial_b32`: horizontal LPIPS/FVD bars plus subject, background,
  and aesthetic VBench differences relative to unbounded, all at B32.
- `02_grouped_budget_bars`: the full budget sweep for LPIPS, FVD, and
  background consistency. Within each policy, bars are B16/B32/B64/B128,
  ordered left to right and distinguished by opacity.
- `03_vbench_horizontal_bars`: all six standard VBench metrics at B32;
  absolute scores displayed as percentages with zero-based axes.
- `00_bar_comparison.png`: preview sheet of all three layouts.

These views do not replace the complete numeric table or establish statistical
significance. VBench percent scaling is only a display-unit change.

## Retrieval Latency Column

The separate matched CPU-bank latency pilot completed for all six B32 table rows.
The reported main-table quantity is the equal-weight trajectory mean over all
600 synchronized retrieval queries per completed rollout. Encoding, denoising,
decoding, bank updates, and latent gathering/transfer are excluded. This is not
end-to-end latency. Native candidate scoring uses 10,000 FOV samples; the
generator receives eight memories. The roster reader verifies these settings,
logged per-trajectory generation seeds, candidate counts, and GPU model.

| Model | Stored items at rollout end | Retrieval ms/query |
| --- | ---: | ---: |
| WorldMem | 1,200 | 77.921 |
| WorldMem + FIFO B32 | 32 | 4.670 |
| WorldMem + MCE B32 | 32 | 4.708 |
| WorldMem + K-center B32 | 32 | 5.004 |
| WorldMem + RI B32 | 32 | 4.638 |
| WorldMem + KEEPSAKE B32 | 32 | **4.541** |

Latency is measured from one matched trajectory per method (`n=1`), with 600
queries per trajectory. The brackets printed by the summarizer are degenerate
for `n=1` and must not be presented as confidence intervals. The 600 queries are
repeated measurements within one trajectory, not 600 independent trajectories.
Relative to unbounded WorldMem, KEEPSAKE reduces mean retrieval latency by
94.2% (17.2x speedup) in this pilot.

Run on an idle CECSL GPU after pulling the updated code:

```bash
cd ~/WorldMem
conda activate worldmem
mkdir -p /data/ab575577/worldmem/logs

GPU=0 \
CUDA_VISIBLE_DEVICES=0 \
WORLDMEM_REPO_ROOT=$HOME/WorldMem \
WORLDMEM_STORAGE_ROOT=/data/ab575577/worldmem \
NUM_VIDEOS=15 \
GLOBAL_SEED=101 \
DATASET_SEED=42 \
bash scripts/run_worldmem_retrieval_latency_roster.sh \
  2>&1 | tee /data/ab575577/worldmem/logs/retrieval_latency_roster_gpu0_$(date +%F_%H%M).log
```

With `NUM_VIDEOS=1`, this profiles six rollouts and produces the pilot reported
above. Increasing `NUM_VIDEOS` produces a trajectory-level uncertainty estimate.
Existing quality runs are not overwritten. Video-based resumption is available,
but a video without its completed 600-query profile does not satisfy the
summarizer.

Main-table output:
`/data/ab575577/worldmem/outputs/retrieval_latency_roster_60s_n15/summary/worldmem_retrieval_latency_main.csv`.
The same directory includes early/late windows and a protocol record. On
Newton omit the CECSL storage override or set an appropriate cluster path;
the default storage root is `$HOME/worldmem_results`.
