# CLAUDE.md - Project context and work log

Read this file first. It is the shared memory for everyone working on this repo.
Two people use Claude Code on this project, so keep it current (see "Work log" and
"Two-person protocol" below).

---

## 1. The project in one paragraph

**Research question:** How accurately can the *pretrained metric* MASt3R checkpoint estimate
real-world distances from uncalibrated image pairs, and what factors affect that accuracy?

**Pipeline:** image pair -> MASt3R -> point maps + dense descriptors -> pixel correspondences
-> corresponding 3D points -> Euclidean distance -> compare with ground truth -> error analysis.

**Application framing (decided 2026-10-06, owner: raghunandan).** Motivating use case: an
Amazon-style "see it in your home" check, but from only two ordinary photos and no AR
hardware. The user marks where an object would go (distance between two small objects or
points); the pipeline estimates that metric gap and answers "does the object fit: yes / no /
uncertain", using the error bound as a margin. This is the *motivation*, not a change to the
research question. A fit-check phase would come after Phase 8 and only if the accuracy numbers
make it viable (a fit check needs roughly 2-5 cm on 1-3 m gaps). Phase 8 result: with a known-length reference in the photos the held-out median error is 1.3 cm and MAE 4.9 cm on gaps of 0.1-0.95 m; without one, MAE is about 11 cm. Gaps of 1-3 m are untested. Do not call the idea
"unique" in any write-up without a literature check (room-measuring and furniture-fit apps
exist). Dataset: **7-Scenes** (indoor RGB-D; chosen 2026-10-06), first scene `chess/seq-01`. It is
believed (from memory, unverified) to be outside MASt3R's training mix (ARKitScenes, ScanNet++,
Habitat, etc., see context.md Q5). Optional tape-measured home pairs may be added later as a
real-world check.

The full brief is in `ORIGINAL_BRIEF.md` (the original prompt, verbatim). Scope notes and the
running decision log are in `context.md`.

---

## 2. Repository layout

```
main.ipynb          THE work. All phases live here. One notebook, no phase scripts.
Setup.md            How to set up on Kaggle or Colab (read this before running anything).
CLAUDE.md           This file.
ORIGINAL_BRIEF.md   Original project brief (verbatim).
context.md          Running notes and scope decisions.
src/paths.py        Path constants and the MASt3R import bootstrap.
phases/             Early Phase 1 script from before the single-notebook decision. Reference only.
results/figures/    Saved plots (matplotlib, static).
results/metrics/    Saved numbers.
mast3r/             NOT in git. Official naver/mast3r clone. Setup cell re-clones it.
checkpoints/        NOT in git. 2.75 GB metric checkpoint. Setup cell re-downloads it.
data/               NOT in git. Ground-truth scenes (data/scene_XX/...) and data/_raw_7scenes/, both written by Phase 6 on the runtime.
.venv/              NOT in git. Local only.
```

---

## 3. Working rules (set by the project owner - do not break)

1. **All work goes into `main.ipynb`, phase by phase.** Do NOT create standalone scripts for
   phases.
2. **The user runs each cell and reports back.** Do not move to the next phase until the user
   says "Proceed with phase N".
3. **Keep the notebook structure for each phase:** Objective (simple + technical explanation),
   "What MASt3R is doing internally", numbered sub-sections (N.1, N.2, ...), a code cell each,
   then "Phase N summary" with: *What was established*, *What is still NOT established*,
   *Assumptions added to the record (question 13)*, and *Next*.
4. **Use the official repo's functions** (`load_images`, `inference`, `fast_reciprocal_NNs`).
   Do not reinvent them.
5. **Static matplotlib figures only.** No plotly or other interactive 3D.
6. **Smoke-test any Python** before presenting it as done.
7. **Conventions that must not break:**
   - Matches are `(x, y)`. Point maps are indexed `[y, x]`. Assert it.
   - Output units are what the model predicts (expected metres, tested in Phase 2.6). Only
     ground truth can confirm the scale.
8. **Never claim a number is accurate** because it is self-consistent. The cross-view residual
   measures agreement with itself, not truth.

---

## 4. Phase status

| Phase | Status | Notes |
|---|---|---|
| 1. Setup and verification | DONE (user ran) | Checkpoint verified byte-exact. Metric recipe fingerprinted. |
| 2. Inference | DONE (user ran) | `load_pair`, `run_mast3r`, `SCENES` (chateau, nle_tower), `IMG_SIZE=512`. Both point maps are in view 1's frame. |
| 3. Point-map extraction | DONE (user ran) | `extract_pointmaps`. Percentile confidence filter. Bbox IoU frame check. Focal estimate. |
| 4. Correspondence matching | DONE (user ran) | `find_correspondences`, `filter_border`, `filter_matches`, `cross_view_residuals`. |
| 5. Distance estimation | DONE (user ran on Kaggle T4, no errors, 2026-10-06) | `estimate_distance` (+ `resolve_endpoint`, `build_context`, `analyse_pair`, `original_to_model_xy`, `get_3d_point`). Error floor + refusal policy (constants `REFUSE_FRAC=0.10` etc. are choices). Demo on bundled scenes, "model estimate only". Ground-truth hook `data/scene_*/metadata.json`; 7-Scenes helpers (`sample_gt_pairs`, `write_gt_scene`) with assumed file conventions NOT yet checked against real data. Cells 60-75 of `main.ipynb`. |
| 6. Ground-truth validation | DONE (user ran on Kaggle T4, 2026-10-08; identical to the local CPU preview) | `data/_raw_7scenes` streamed frames of chess/seq-01, conventions verified on real files (pose = camera-to-world, f about 585), `robust_gt_pairs`, 4 scenes x 10 GT pairs, scale test per scene. Cells 76-87. Local CPU preview only (see work log). |
| 7. Evaluation metrics | DONE (user ran on Kaggle T4, 2026-10-08) | `calculate_metrics()` (MAE, RMSE, rel/pct error, median, p90, bias, scale ratio, rescaled MAE, bootstrap CIs) with a synthetic self-test; summary table, per-pair CSV, 3 plots. Cells 88-95. Reads `ALL_ROWS` or `results/metrics/phase6_gt_results.json`. |
| 8. Experiments / ablation | DONE (user ran on Kaggle T4, no errors, 2026-10-08) | Cells 96-111. Innovation 1 (8-vote ensemble: swap + mirror + patch median) was tested and NOT adopted: no gain for 4x compute. Innovation 2 (known-length scale anchor, 3 depth-matched anchors, refusal when anchors disagree) is the result. Settings chosen on dev scenes (chess, office), frozen, then run once on held-out redkitchen+pumpkin+fire (110 pairs): baseline median 5.8 / MAE 10.8 cm -> final median 1.3 cm [0.6, 2.0] / MAE 4.9 cm [2.0, 8.4]. Pre-registered criteria (both <= 5 cm) MET, MAE only just. See work log and `context.md` section 14. Not covered: image resolution, texture, occlusion. |

**Known open labels.** The notes use labels "B" (input resolution) and "E" (confidence
percentile). The brief lists the full set above. Confirm with the owner before using the
letter labels.

**Assumptions on record (from Phases 2-4):**
1. Overlapping, mostly rigid, static scene.
2. View 1 is the anchor frame.
3. Long side 512 (the trained regime).
4. Static scene.
5. Confidence percentile is a choice, tested in Phase 8.
6. Bbox IoU frame check is coarse.
7. One-pixel quantisation error grows with depth.
8. Reciprocal NN assumes the point is visible in both views (occlusion failure mode).

---

## 5. Runtime

Kaggle (Tesla T4; last run: Python 3.13.15, torch 2.11.0+cu128, numpy 2.1.3) or Google Colab (T4). Both are used alternately, so the setup
cells at the top of `main.ipynb` handle either one. **Full steps are in `Setup.md`.**

- Runtime files are temporary. Copy anything worth keeping back to git before a session ends.
- The checkpoint must be re-downloaded (about 2.75 GB) on every new runtime.
- Phase 6 builds ground truth from 7-Scenes depth (Kinect-derived, not hand-measured). Distances on the bundled scenes stay "model estimate only".
- Committed notebook carries only the Phase 1-4 outputs; do not commit newer outputs (10 MB+). Cells 60+ are Phase 5 onward; Phase 1-4 cell numbers in `context.md` are +2.

---

## 6. Two-person protocol

Both people push to `main`. Because `main.ipynb` is one large JSON file, two people editing it
at the same time will conflict badly. Use these rules:

1. **Pull before you start:** `git pull --rebase`.
2. **Claim a phase before editing the notebook.** Add a line under "Current claims" below,
   commit it, and push it. Do this first, then work.
3. **Only the claimant edits `main.ipynb`** until the claim is released. The other person works
   on something else (a different phase, `Setup.md`, `CLAUDE.md`, or reading results).
4. **Release the claim** by deleting your line, updating the phase status table, adding a log
   entry, and pushing.
5. **Commit and push after each stable step,** not at the end of a long session. Commit
   messages must end with: `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
6. **Never commit** `checkpoints/`, `.venv/`, `mast3r/`, `data/`, or any token.

### Current claims

_(none)_

---

## 7. Work log

Each person writes only in their own section. Add new entries at the top of your section.
Format for each entry:

```
- YYYY-MM-DD | Phase N | runtime: Kaggle|Colab
  Did: what changed
  Verified: what was run and the result (real output, not expected output)
  Next: the very next step
```

### raghunandan

- 2026-10-08 | Phase 8 | runtime: Kaggle T4 (owner's run); local CPU for the dev cross-check
  Did: appended Phase 8 (cells 96-111): per-scene convention checks, 5 scenes split into dev (chess, office) and
  held-out (redkitchen, pumpkin, fire); training-free 8-vote ensemble (innovation 1); known-length scale anchor with
  depth-matched anchors, endpoint-sharing guard and anchor-agreement refusal (innovation 2); factor analysis.
  Verified (owner's Kaggle run, all cells, no errors; dev numbers identical to my local CPU run):
  DEV (80 pairs): ensemble NOT adopted (MAE 11.5 vs 11.7 cm, median 9.0 vs 7.4; vote spread has rho 0.00 with error).
  Frozen: plain estimator, k=3 depth-matched anchors, refuse when anchor scales disagree (threshold 0.157, ~70% coverage).
  HELD-OUT (110 pairs from 11 frame pairs; fire_100-160 skipped, no robust pairs), errors in cm:
  baseline MAE 10.8 / median 5.8 / <=5cm 44%; ensemble 10.7 / 6.8; naive 1-anchor 5.7 / 1.8; FINAL 4.9 [2.0, 8.4] /
  1.3 [0.6, 2.0] / 88%; final on the 86% answered: MAE 3.7 / median 1.0 / 94%.
  Pre-registered criteria (median <= 5 and MAE <= 5) MET; the MAE criterion only just, with a 95% interval up to 8.4 cm.
  Per scene MAE: redkitchen 5.9, pumpkin 6.5, fire 1.5 (two of three above 5 cm). About 3% of pairs have errors > 40 cm.
  CAVEATS: the result needs a known-length reference marked in the photos (here another ground-truth pair, with Kinect
  noise); all scenes are 7-Scenes/Kinect; ground-truth pairs are smooth-region and depth-shift-robust (easier than
  average); 80 dev pairs were used to choose settings (alternatives counted in 8.4).
  Next: owner decides: fit-check phase, or first more scenes / phone photos with tape-measured lengths.

- 2026-10-07 | Docs sync | runtime: local
  Did: brought `CLAUDE.md` and `context.md` up to date for Phases 5-7 (new `context.md` section 14 with
  results, verified 7-Scenes conventions, open items; old plan sections marked superseded).
  Verified: all Phase 5-7 commits are on `origin/main`. The owner's local `main.ipynb` still carries
  Kaggle outputs for Phases 5 onward; deliberately NOT committed (10 MB+).
  Next: owner runs Phases 6-7 on Kaggle; then "Proceed with phase 8".

- 2026-10-07 | Phase 7 | runtime: local CPU preview only (not yet run on Kaggle)
  Did: appended Phase 7 to `main.ipynb`: `calculate_metrics` with a self-test on synthetic cases,
  summary metrics table, per-pair table (`phase7_pairs.csv`), `phase7_metrics.json`, plots of
  ground truth vs predicted / absolute error / relative error.
  Verified: self-test passes; table and plots ran on the saved local Phase 6 results (40 pairs).
  Preview (local CPU, all pairs): MAE 0.130, RMSE 0.223, median abs error 0.062, median relative
  error 12.5% (bootstrap 95%: 11.0-14.3%, optimistic), bias +0.077, pooled scale ratio 1.03 but per
  scene 0.89 / 0.93 / 1.13 / 1.36. Rescaling per scene would cut MAE only to 0.128 pooled, so most of
  the pooled error is NOT a plain scale error (outliers in scenes 3-4). The 3 largest errors are all
  pairs the 5.2 policy refused.
  Next: user runs Phase 7 cells; Phase 8 on "Proceed with phase 8".

- 2026-10-06 | Phase 6 | runtime: local CPU preview only (not yet run on Kaggle)
  Did: appended Phase 6 to `main.ipynb`: streams a few 7-Scenes `chess/seq-01` frames (134 MB, not
  3 GB), verifies the dataset conventions on real files, builds 4 ground-truth scenes x 10 pairs that
  tolerate an 8 px depth-registration error, runs the 5.6 hook, scale test and error table.
  Verified (local CPU, real model, real frames; the Kaggle run is the one that counts): pose is
  camera-to-world (84.1% depth agreement vs 0.4% inverse), f about 585 px. Policy accepted 4 of 40
  pairs (it needs conf >= median at both endpoints in both views: too strict for random points).
  Using all 40: median D_est/D_gt per scene 0.89 / 0.93 / 1.13 / 1.36 (baselines 0.10-0.82 m), so NO
  single consistent scale; MAE 0.130, median abs error 0.062, median relative error 12.5%.
  Depth-to-colour registration is NOT verified (ground truth made tolerant to 8 px instead).
  Next: user runs the Phase 6 cells; Phase 7 follows.

- 2026-10-06 | Phase 5 follow-up | runtime: Kaggle (user run), local CPU (checks)
  Did: refusal message shows one decimal (a 10.04% floor no longer reads "10% (limit 10%)");
  selected-points figure staggers labels. Read the user's Kaggle outputs.
  Verified: Kaggle run: all cells through Phase 5 ran without error; chateau 505 reliable matches,
  focal 930 px; 5 of 8 demo distances reported (D 0.55-0.73, floors 6-9% of D), 3 refused.
  CAUTION: the bundled scenes are outdoor buildings yet the model puts them 2-4 units away, so the
  absolute scale is very likely wrong there (plausibility argument only; ground truth needed).
  Next: user says "Proceed with phase 6" (7-Scenes, indoor).
- 2026-10-06 | Phase 5 | runtime: local CPU only (no Kaggle/Colab run yet)
  Did: appended Phase 5 to `main.ipynb` (distance estimation with error floor and refusal, demo on
  the two bundled scenes, optional ground-truth hook, 7-Scenes helpers). Recorded the application
  framing; chose 7-Scenes for Phase 6.
  Verified: synthetic tests passed (x/y indexing, refusals, pixel mapping vs real `load_images`
  for 6 photo sizes, 7-Scenes geometry on a synthetic wall). Full pipeline ran on CPU on the
  chateau pair: 505 reliable matches, focal 929.6 px, threshold 2.104 (all match Phases 3-4), 3 of 4
  demo distances reported (D about 0.58-0.70, floors 6-9% of D), 1 refused. Not run on GPU; the
  new cells are not run in the notebook itself.
  Next: user runs the Phase 5 cells and reports; then "Proceed with phase 6" (download 7-Scenes,
  verify assumed conventions, build `data/scene_XX/`).

- 2026-10-05 | Repo setup | runtime: Kaggle
  Did: created `.gitignore`, `CLAUDE.md`, `Setup.md`, and the Colab and Kaggle setup cells at the
  top of `main.ipynb`. Initialised git.
  Verified: notebook JSON valid (61 cells); both setup cells compile and skip correctly off-platform.
  NOT yet run on Colab or Kaggle - the clone, pip and GPU branches are untested.
  Next: user runs the Colab and Kaggle setup cells and reports back; then Phase 5, once the
  user approves the plan.

### Person 2 (name to be added)

- _(no entries yet)_
