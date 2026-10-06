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
make it viable (a fit check needs roughly 2-5 cm on 1-3 m gaps; unproven). Do not call the idea
"unique" in any write-up without a literature check (room-measuring and furniture-fit apps
exist). Dataset leaning: an **indoor** dataset, since the target is small-object distances at
room scale. Prefer held-out sets (7-Scenes, NYU Depth V2, ScanNet) over MASt3R training sets
(ARKitScenes, ScanNet++, Habitat, etc., see context.md Q5). Not final; the owner has not yet
picked a dataset. Optional tape-measured home pairs may be added later as a real-world check.

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
data/               NOT in git. Ground-truth scenes (data/scene_XX/...). Empty for now.
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
| 5. Distance estimation | WRITTEN, awaiting user run on Kaggle/Colab (2026-10-06) | `estimate_distance` (+ `resolve_endpoint`, `build_context`, `analyse_pair`, `original_to_model_xy`, `get_3d_point`). Error floor + refusal policy (constants `REFUSE_FRAC=0.10` etc. are choices). Demo on bundled scenes, "model estimate only". Ground-truth hook `data/scene_*/metadata.json`; 7-Scenes helpers (`sample_gt_pairs`, `write_gt_scene`) with assumed file conventions NOT yet checked against real data. Cells 60-75 of `main.ipynb`. |
| 6. Ground-truth validation | NOT STARTED | Dataset chosen: **7-Scenes** (indoor RGB-D, owner decision 2026-10-06). Must verify its file conventions (intrinsics, depth registration, pose direction) and ask before downloading. |
| 7. Evaluation metrics | NOT STARTED | MAE, RMSE, relative error, percentage error (per ORIGINAL_BRIEF Section 6). |
| 8. Experiments / ablation | NOT STARTED | Factors from the brief: viewpoint difference, physical distance, image resolution, descriptor match quality, confidence filtering on/off, which points are chosen, texture richness, occlusion. |

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

Kaggle (Tesla T4, Python 3.12) or Google Colab (T4). Both are used alternately, so the setup
cells at the top of `main.ipynb` handle either one. **Full steps are in `Setup.md`.**

- Runtime files are temporary. Copy anything worth keeping back to git before a session ends.
- The checkpoint must be re-downloaded (about 2.75 GB) on every new runtime.
- No ground truth exists yet. Every distance is "model estimate only".

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

- Phase 5 cosmetic fixes (2 cells in `main.ipynb`) - raghunandan with Claude Code - since 2026-10-06

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
