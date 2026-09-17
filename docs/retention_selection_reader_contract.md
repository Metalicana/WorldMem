# WorldMem Retention/Selection: Reader Contract and Audit Gate

Scope: the 21 existing 60s configurations, batches 0-14. No regeneration,
feature extraction, gap calculation, or favorable result is implied by this audit.
Source-code facts below describe the current checkout, not an independently
verified historical revision for every existing rollout.

## Verified Current Reader Contract

- One stored item is a VAE latent frame with a full-history frame ID. It is not
  an RGB image, temporal block, or merged state. See `encode` and `validation_step`
  in [df_video.py](../algorithms/worldmem/df_video.py).
- `validation_step` reads before sampling the current chunk and before updating
  the bank. It appends noise for the current chunk but eligibility excludes those
  current/future frames. Default `chunk_size` is 1 in
  [df_base.yaml](../configurations/algorithm/df_base.yaml); audit logs must confirm
  one-frame horizons rather than borrowing a MemCam stride.
- `_generate_condition_indices` filters IDs to `0 <= i < curr_frame`, sorts and
  deduplicates candidates, and greedily selects eight IDs. Each winner is masked
  out. With at least eight candidates the selected IDs are distinct. Warmup and
  bank-exhaustion fallback can repeat/pad IDs; the proposed suite audit rejects
  those queries, rather than silently applying a distinct best-eight oracle.
- There are no forced historical retrieval slots or hard geometric set
  constraints in the normal reader. Geometric overlap and a time penalty score
  candidates; removing occupied FOV changes later greedy scores but does not
  prohibit any distinct eight-ID set. A best-eight independent-appearance oracle
  is therefore feasible, but not a model of complementary conditioning utility.
- The actual selected `random_idx` IDs are gathered into the sampling input as
  eight latent references, on either CPU-bank or GPU-bank paths. Logged scored
  alternatives are not generation inputs.
- Unbounded eligible history at query q is `H = range(q)`. At context 600 and
  rollout index k, q is 600+k. This is checked against real candidate-count logs,
  not supplied as a fabricated bank trace.
- `_build_memory_buffers` adds all 600 initial IDs and logs initialization
  evictions BEFORE `memory_run_start`. The audit associates that eviction burst
  with the following explicit trajectory identifier. At generation updates,
  `_update_memory_buffers` admits all new frames for the six requested policies
  and logs removals AFTER the write. Both boundaries matter for reconstruction.
- FIFO has no permanently pinned initial item. RI, geometric coverage/Ours,
  K-center and MCE pin history ID 0. All six budget-sweep families temporarily
  protect the latest observed endpoint during each write, including ID 599 at
  initialization. These are eviction protections, not forced retrieval choices.
- Protected items and retained initial context count INSIDE B. The bounded
  retrieval path reads only `buffer.candidates()`; it does not append an
  uncounted 600-frame initial store to the candidate list.
- The native recent token context remains separate from the external bank.
  Full-history `xs_pred`, preencoded GT latents, and policy-scoring archives can
  still occupy host memory. Thus B bounds the persistent retrievable candidate
  bank, NOT total RAM or every tensor stored by the application. Predicted-memory
  reads cannot retrieve evicted IDs merely because their tensors remain allocated.
- K-center and some scoring routines consult archival descriptors during bank
  maintenance. These are not additional historical latent references furnished
  to the generator by the query reader. Report this maintenance storage/cost
  separately; do not turn a candidate-count bound into a total-memory claim.

## Index and Pixel Protocol

The dataset loader skips the first 100 dataset frames. In the stated protocol:

| Quantity | Initial context | Generated rollout |
| --- | --- | --- |
| Full-history ID | 0-599 | 600-1199 |
| Generated MP4 index | Not present | 0-599 |
| Exact dataset index | 100-699 | 700-1299 |

Dataset retries are a critical identity limitation: `MinecraftVideoDataset`
may advance past a failing sample without logging the actual returned source
path. An equal batch number does not independently prove equal scene identity.
Current traces also omit checkpoint/config identity. Audit those separately
before declaring a matched cohort. Do not infer source seed 101 matches the
primary suite solely from similar run names.

Generated pixels in the proposed shared-source experiment are decoded MP4
proxies for the stored latents, not exact latent decoding. Initial-context
pixels/features come from observed dataset inputs; they are not generated or
assigned a quality ceiling. All policy banks are scored against ONE common
source, while their admission/eviction/selection IDs come from their own rollouts.

## First Deliverable: CPU Audit

`scripts/audit_worldmem_retention_selection.sh` audits the existing primary run
names and the proposed seed-101 common source. It writes `coverage.csv` and
`audit.json`, including actual bank/count validation, query coverage, initial
context exposure counts, source seed status, source/cache hashes, and explicit
unverified dataset/checkpoint identities. It never starts generation or DINO.

Existing paired DINO caches can be reused only if their saved identities,
source/GT hashes, encoder revision/processor hash, index map, float32 dimensions,
normalization, and initial-context origin pass. Filename/array length alone is
insufficient. The existing cache stores source and GT in separate named arrays
within one NPZ; this is an explicitly labeled legacy cache layout.

The tool does not mark analysis ready merely because read reconstruction passes.
Legacy traces may omit `memory_reference_source`; this does not by itself
invalidate historical bank IDs. The audit labels the source metadata as
`unlogged_legacy_metadata` and preserves the original record. Revision `339722c`
gathers directly from `xs_pred`, and revision `6a33105` introduces the configurable
source field. This evidence does not identify the runtime revision of each old
trace, so it is not used to silently assert a missing value. Explicit GT-memory
mode remains a primary-protocol failure.

An absent `memory_run_end` is reported separately from structural read validity.
A last attempt with all 600 valid reads can establish bank/read reconstruction
without asserting that generation completed or that its MP4 association was
verified. Duplicate completed attempts or a later partial attempt after a
completed one remain ambiguous and fail audit. Generation seeds are reported
as matched, mismatched, or missing, rather than combining those cases.

After the real CECSL audit is returned, resolve scene/start/config identities,
missing traces, seed mismatches and feature compatibility. Missing features call
for feature extraction, not regeneration. Missing traces need explicit approval
before any expensive regeneration. Next-stage gap and figure code must retain
all null/unfavorable results and use the handoff's best-eight and paired
trajectory-bootstrap protocol.
