# DFoT and FramePack Minecraft Evaluation Handoff

Updated: 2026-09-20.

Use this document as the experiment contract for the DFoT and FramePack Codex
sessions. The goal is to reuse the WorldMem Minecraft test bed without silently
changing trajectories, time indexing, memory semantics, or metrics.

## Non-Negotiable Rules

1. Inspect the target repository before editing. Do not assume that its native
   conditioning, frame rate, latent format, or generation chunk size matches
   WorldMem.
2. Do not launch the full experiment until a one-trajectory smoke test has been
   decoded and inspected.
3. Build and save an immutable cohort manifest before generation. Every output
   must record the source MP4/NPZ, source-frame interval, seed, model revision,
   checkpoint, conditioning mode, output-frame mapping, and policy settings.
4. Use the exact 15 trajectories below. Do not identify trajectories only by a
   generated batch number and do not allow retry-on-error to substitute another
   trajectory.
5. Export evaluation videos at 10 FPS. WorldMem accidentally encoded its
   600-frame files at 15 FPS; do not repeat that container-rate error.
6. Report model-native differences. In particular, do not describe FramePack as
   action-conditioned unless the inspected implementation actually consumes the
   Minecraft action sequence.
7. Keep quality evaluation and retrieval-latency profiling separate. Quality
   uses 15 trajectories. Retrieval latency may use one matched trajectory and
   must be labeled `n=1`, averaged over all retrieval queries.

## CECSL and Newton Storage

The code is edited locally, pushed by the user, and pulled on the execution
machine. Do not attempt to run the remote code from the local sandbox.

- CECSL dataset: `/data/ab575577/worldmem/data/minecraft`
- CECSL experiment root: choose a model-specific directory under
  `/data/ab575577/`, not the home filesystem.
- Newton has no `/data/ab575577` path. Require a configurable storage root and
  default to `$HOME/<project>_results` there.
- Never hard-code the CECSL path inside Python modules.

## Locked Minecraft Cohort

Dataset: `zeqixiao/worldmem_minecraft_dataset`, test split.

For every trajectory:

- Source FPS: 10.
- Source start index: 100, because the first 100 dataset frames are skipped.
- Full source interval: `[100, 1300)`, exactly 1,200 frames.
- Observed history: source indices `[100, 700)`, 600 frames.
- Prediction target: source indices `[700, 1300)`, 600 frames.
- Generated duration: nominal 60 seconds at 10 FPS.
- Exact-index GT for generated output frame `k`: source frame `700 + k`.
- Action and pose for generated output frame `k`: NPZ entry `700 + k`.

Use these exact relative test paths, in this order:

```text
val_savanna/3/000041.mp4
val_desert/3/000029.mp4
val_desert/0/000033.mp4
val_savanna/1/000039.mp4
val_iceplains/0/000028.mp4
val_desert/3/000035.mp4
val_plains/0/000004.mp4
val_iceplains/3/000010.mp4
val_savanna/0/000017.mp4
val_desert/0/000038.mp4
val_iceplains/0/000011.mp4
val_iceplains/1/000041.mp4
val_plains/0/000002.mp4
val_iceplains/1/000026.mp4
val_desert/2/000028.mp4
```

Machine-readable copy:
`assets/results/worldmem_minecraft_cohort_60s_n15.json`.

Each MP4 must have a same-stem NPZ. Before generation, decode frames 100 through
1299 and verify that video, actions, and poses all cover the interval. Fail
closed on any missing/corrupt sample; do not advance to a replacement file.

## Minecraft Actions

WorldMem converts the NPZ action array into 25 channels. Reuse and test the
mapping in `WorldMem/datasets/video/minecraft_video_dataset.py`, including
forward/back, left/right, camera X/Y, drop/event, and hotbar handling. Do not
invent a different action interpretation.

If a target model cannot consume actions:

- it may still be evaluated on the same source clips as an image/video-
  conditioned generation baseline;
- its conditioning must be labeled explicitly;
- LPIPS against the action-defined exact-index GT is diagnostic and should not
  be interpreted as trajectory-control accuracy;
- do not claim a controlled, action-conditioned backbone comparison.

## Locked Memory Contract

For a memory-enabled adaptation, preserve this interface unless the paper table
explicitly labels a model-native alternative:

- Native recent context: eight most recent frames, separate from the persistent
  memory budget.
- Persistent memory begins from the 600 observed-history frames.
- Retrieval happens before generating the next frame/chunk and returns eight
  distinct eligible past items.
- An unbounded archive admits each generated frame and grows from 600 to 1,200
  items over the rollout.
- Bounded policies retain at most `B` persistent items, including protected
  items. Use `B in {16, 32, 64, 128}` for the sweep and `B=32` for the locked
  cross-system comparison.
- The latest section endpoint is protected during updates.
- RI, MCE, K-center, and KEEPSAKE protect the initial history frame; FIFO does
  not.
- KEEPSAKE is the existing WorldMem `slam_covisibility` / Geometric Coverage
  policy. Do not substitute coverage hysteresis or the 75/25 experimental blend.
- Reuse the policy implementation and tie-breaking behavior from this WorldMem
  repository rather than reimplementing names from memory.
- One stored item must be one temporally indexed frame representation. If the
  target model stores a multi-frame pack/token, document its frame-equivalent
  accounting and do not call 32 packs “32 frames.”

Start with this six-cell B32 roster:

```text
Unbounded
FIFO B32
MCE B32
K-center B32
RI B32
KEEPSAKE B32
```

Only after those six cells pass should the session run the full budget sweep.

## Determinism and Output Identity

- Dataset/cohort order is fixed by the manifest above, not shuffling.
- Use one documented base generation seed and derive per-trajectory seeds as
  `base_seed + trajectory_id`.
- Use the same per-trajectory seed across policies within a model.
- Memory-policy randomness gets a separate recorded seed, held constant across
  matched policy comparisons.
- Save one prediction MP4 immediately after each trajectory.
- Use names containing model, policy, budget, horizon, trajectory ID, and seed.
- Resume by manifest completion status, not merely by counting arbitrary MP4s.
- Save access traces with every query's eligible bank IDs, selected IDs,
  candidate count, retained-bank size, query time, and source-frame mapping.

## Quality Metrics

Run on exact matched trajectory IDs 0 through 14:

1. LPIPS against exact-index GT, frame aligned over 600 output frames.
2. FVD using the same implementation/settings as WorldMem: cached StyleGAN-V
   I3D, 16-frame clips, four clips per video, stride 4, image size 224.
3. Standard VBench prompt-independent dimensions: subject consistency,
   background consistency, motion smoothness, dynamic degree, aesthetic
   quality, and imaging quality.

Do not run VBench-Long. Do not report CUT3R until a model-specific exact-GT
sanity test demonstrates sensible camera reconstruction; WorldMem's prior CUT3R
sanity test failed.

For LPIPS/FVD, evaluate by frame index, not MP4 wall-clock timestamps. For
VBench, stage copies encoded at exactly 10 FPS and verify that each contains 600
frames. Save machine-readable per-video values as well as aggregate summaries.

## Retrieval Latency

Use one matched trajectory per method for the paper's descriptive latency
column. Time only candidate scoring and selection, synchronized around GPU work.
Exclude candidate-bank updates, denoising, encoding/decoding, and memory-content
transfer. Report:

- mean ms/query across the trajectory;
- number of queries;
- early and late windows separately;
- hardware, bank device, candidate count, selected count, and timing scope.

Do not print a confidence interval for one trajectory. The many queries are
repeated measurements within one rollout, not independent trajectories.

## DFoT Session Instructions

Paste the entire document into the DFoT session, followed by:

> First audit this repository and identify whether it is the same action-
> conditioned DFoT backbone used by WorldMem, the checkpoint expected for
> Minecraft, its native chunk/frame-stack semantics, and where external memory
> enters conditioning. Compare its code against the WorldMem repository rather
> than guessing. Then implement the locked cohort manifest and a strict loader.
> Run one no-memory/native-baseline smoke trajectory and verify 600 generated
> frames at 10 FPS. If the repository supports WorldMem-compatible memory
> conditioning, port the existing policy interface without changing model
> weights and run the six-cell B32 roster one trajectory at a time. Stop and
> report if no compatible trained checkpoint or memory-conditioning interface
> exists; do not manufacture one and call it zero-shot evaluation. Provide
> CECSL commands first, with all storage paths configurable for Newton.

DFoT is the most plausible candidate for exact action-conditioned parity, but
the session must establish that from code and checkpoint metadata.

## FramePack Session Instructions

Paste the entire document into the FramePack session, followed by:

> Audit FramePack's actual input contract, FPS, context representation, chunk
> overlap, prompt requirements, and whether it supports per-frame actions. Do
> not assume WorldMem action conditioning. Build the locked Minecraft cohort
> manifest first. If FramePack is image/text-conditioned only, use source frame
> 699 (the frame immediately before the locked target) or the longest native
> prefix ending at source frame 699 as conditioning. Label the reduced context
> prominently, and treat exact-index LPIPS as diagnostic rather
> than action-control accuracy. Generate a full 60 seconds in model-native
> chunks, then create a deterministic 600-frame 10-FPS evaluation rendering
> with an explicit timestamp mapping; never merely relabel the container FPS.
> Before porting memory policies, identify whether a stored FramePack item is a
> frame, latent frame, or multi-frame pack and report frame-equivalent storage.
> Do not claim the same B32 memory budget until that accounting is valid. Run
> one trajectory end to end before launching the six-cell roster. Provide
> resumable CECSL commands and configurable Newton paths.

FramePack can share the Minecraft clips and evaluation framework, but it is not
automatically an action-conditioned apples-to-apples WorldMem comparison.

## Required Smoke-Test Report

Before any full run, each session must print:

```text
model/checkpoint/revision
conditioning mode and inputs
source MP4 and NPZ
source interval
observed source indices
generated-target source indices
native generated frame count and native FPS
evaluation frame count and evaluation FPS
action tensor shape (or ACTIONS UNSUPPORTED)
memory item unit
persistent bank size at start/end
retrieved items per query
output path
trace path
```

It must also decode the saved MP4 and assert the expected frame count and FPS.

## Order of Work

1. Repository/checkpoint/conditioning audit.
2. Immutable cohort manifest and strict data preflight.
3. One native baseline trajectory.
4. One Unbounded and one KEEPSAKE B32 trajectory, if memory compatibility is
   established.
5. Inspect outputs and traces.
6. Complete the six-cell B32 roster for 15 matched trajectories.
7. Compute LPIPS, FVD, and standard VBench.
8. Run one-trajectory retrieval-latency profiling.
9. Only then consider the four-budget sweep.

The session must not begin at step 6 merely because the dataset already exists.
