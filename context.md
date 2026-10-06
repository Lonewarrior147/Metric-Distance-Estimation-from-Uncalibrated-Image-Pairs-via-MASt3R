# Context: `mast3r_metric_distance`

Handover document for continuing this project with a fresh Claude (or any engineer). Read it fully before touching anything.

**Provenance.** The original chat transcript is lost (see §11). This document is reconstructed from `main.ipynb` (code, markdown and the saved cell outputs), `src/paths.py`, `phases/phase1_setup.py`, the folder contents, and the author's prompt history. Where something is not recorded anywhere, it is marked **UNKNOWN** rather than guessed.

---

## 1. What the project is

**Research question:** how accurately can a *pretrained metric* MASt3R model estimate real-world distances from **uncalibrated** image pairs, and what factors influence that accuracy?

**Pipeline (the throughline of every phase):**

```
two uncalibrated images
  -> MASt3R (pretrained, off-the-shelf, metric checkpoint; never fine-tuned)
  -> 3D point maps + dense 24-D descriptors
  -> pixel-to-pixel correspondences (reciprocal nearest neighbours)
  -> corresponding 3D points
  -> Euclidean distance  D = sqrt((X1-X2)^2 + (Y1-Y2)^2 + (Z1-Z2)^2)
  -> compare with tape-measured ground truth -> error analysis + experiments
```

The model is **frozen**. "Retraining" is never required: new data only means re-running evaluation.

## 2. How the author wants to work (follow these)

1. All work goes **directly into `main.ipynb`**, phase by phase. Do **not** produce standalone per-phase scripts. (An early attempt did, in `phases/phase1_setup.py`. The author asked for the notebook instead.)
2. The author **runs each cell personally** and reports errors or output. Do not start the next phase until they say so ("Proceed with phase N").
3. Notebook style per phase: `## PHASE N - TITLE`, then `### Objective` (a "Simple explanation" and a "Technical explanation"), then `### What MASt3R is doing internally at this stage`, then numbered sub-sections `### N.1`, `### N.2`, ... each followed by one code cell, then `### Phase N summary` with *What was established*, *What is still NOT established*, *Assumptions added to the record (question 13)*, and a `### Next` note.
4. Use the **official repo functions** (`load_images`, `inference`, `fast_reciprocal_NNs`). Do not reinvent them. Verify claims against the actual source at the pinned commit rather than from memory.
5. The author **cannot use interactive 3D** (plotly etc.), so use static matplotlib figures. Phase 3.6 interactive 3D is optional and was skipped.
6. Assertions over assumptions: coordinate order, shapes and ranges are asserted in code.
7. Smoke-test any code before presenting it as done.
8. Tell the author the plan before big steps. Their latest instruction was: skip tape-measured data for now, leave a hook so it can be added later, and **tell the plan first**.

## 3. Folder layout, and what each part does

Root: `/home/raghunandan/3d/` (the parent folder `3d/` also holds `3d.zip`).

```
3d/
├── 3d.zip                         3.6 GB archive of the project (see below)
└── mast3r_metric_distance/        <- the project root (PROJECT_ROOT)
    ├── main.ipynb                 THE deliverable. 59 cells, Phases 1-4 written and run, outputs saved (8.5 MB).
    ├── src/
    │   └── paths.py               Path constants + bootstrap_mast3r_imports(). Used by phase1_setup.py.
    ├── phases/
    │   └── phase1_setup.py        Early standalone script version of Phase 1 (superseded by the notebook).
    ├── mast3r/                    Official `naver/mast3r` clone with submodules (dust3r, croco). Do not edit.
    ├── checkpoints/
    │   └── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth   (2.75 GB, the metric model)
    ├── data/                      EMPTY. Reserved for the author's own tape-measured scenes (scene_01/, ...).
    ├── results/
    │   ├── figures/               EMPTY locally (see caveat below)
    │   └── metrics/               EMPTY locally
    ├── .venv/                     Local Python 3.10 virtualenv (CPU torch 2.13.0+cpu). Not the real runtime.
    ├── RESUME_PROMPT.md           A prompt for resuming from Phase 5 in a new session.
    ├── summary.md                 Shareable results summary.
    └── context.md                 This file.
```

- **`main.ipynb`**: structured as a research notebook. The cell map:

| Cells | Content |
|---|---|
| 0 | Title, research question, pipeline, environment note (Kaggle T4, Python 3.12) |
| 1-17 | **Phase 1** setup: env check (2), paths/config (4), clone repo (6), deps (8), imports + device (10), download checkpoint (12), inspect checkpoint (14), load model (16), and cell 17 = the Phase 1 findings Q1-Q9 |
| 18-31 | **Phase 2** inference: choose pairs (19-20), input viz (21-22), run inference (23-24), dissect outputs (25-26), confidence maps (27-28), scale test (29-30), summary (31) |
| 32-45 | **Phase 3** point maps: extract (33-34), depth maps (35-36), confidence filtering (37-38), 3D viz (39-40), geometry checks (41-42), optional interactive 3D (43-44), summary (45) |
| 46-57 | **Phase 4** correspondences: matcher (47-48), border filter + validity (49-50), correspondence viz (51-52), `filter_matches` (53-54), strict 3D consistency test (55-56), summary (57) |
| 58 | Empty code cell. This is where Phase 5 begins. |

- **`3d.zip`**: a snapshot of the project folder dated Aug 19 (it contains `.venv` and `phases/`, with a Python 3.10 `.pyc`). It is probably what was used to move the project around. Treat it as a backup, not a source of truth.
- **`.venv/`**: local Python 3.10 with CPU-only PyTorch. The machine has **no NVIDIA GPU** (`nvidia-smi` is absent). The project cannot run end to end locally; see §4.
- **`src/paths.py`**: defines `PROJECT_ROOT`, `MAST3R_REPO`, `DUST3R_REPO`, `CROCO_REPO`, `CHECKPOINT_DIR`, `DATA_DIR`, `RESULTS_DIR`, `FIGURES_DIR`, `METRICS_DIR`, `METRIC_CHECKPOINT(_NAME/_URL)`, and `bootstrap_mast3r_imports()`, which puts the repo root on `sys.path` (appended, not inserted at 0) and gives actionable errors if a submodule is missing. The notebook defines its own equivalent constants in cell 4 rather than importing this file.

## 4. Runtime: where it actually ran

**The notebook ran on a Kaggle remote kernel** (Tesla T4, 14.6 GB, Python 3.12.13, torch 2.10.0+cu128, numpy 2.0.2, scipy 1.16.3, opencv 4.13.0, matplotlib 3.10.0, einops 0.8.2, trimesh 5.0.0, plus `roma`). Saved outputs show paths like `/kaggle/working/mast3r_metric_distance/...`.

Caveats to know about:
- Cell 4 sets `WORK = /kaggle/working` if it exists, else `Path.cwd()`. Off Kaggle, paths follow the current working directory.
- Cell 10 has `assert DEVICE == "cuda"`, so the notebook will not run on CPU as written.
- The notebook `.ipynb` on disk may be more current than the files in `results/`: `results/figures` is empty here, although cells report saving figures (for example `phase3_conf_mask_chateau.png`). Those were written on Kaggle's disk, not synced back. Figures can be regenerated by re-running.
- Cell 4 checks free disk (about 4 GB is needed) and `CKPT_EXPECTED_BYTES = 2,754,910,614`.
- Cell 6 needs Internet ON in Kaggle settings (git clone). Cell 12 uses `wget -c` (resumable) and checks the byte count.

**Pinned revisions (for reproducibility):** `mast3r f5209af` (2025-06-30), `dust3r 3cc8c88` (2025-06-26), `croco d7de070` (2025-05-22).

**Checkpoint:** `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`, from `https://download.europe.naverlabs.com/ComputerVision/MASt3R/`. It is the only public ViT-L MASt3R checkpoint and is the metric one. Supports 512x384, 512x336, 512x288, 512x256 and 512x160. Head CatMLP+DPT, encoder ViT-L, decoder ViT-B, 688.6 M parameters, 2.58 GB VRAM loaded, 3.13 GB peak during inference. Check `mast3r/CHECKPOINTS_NOTICE` and `LICENSE` for usage terms before publishing anything.

## 5. Phase-by-phase: what exists and what was found

### Phase 1: Setup and verification (done)
The checkpoint's architecture string matches the README's metric training recipe on all 15 kwargs and differs from the demo recipe. All keys load. **Source-verified findings (Q1-Q9):**
- Q1: per view the head returns `pts3d [B,H,W,3]`, `conf [B,H,W]`, `desc [B,H,W,24]`, `desc_conf [B,H,W]`. View 2's point map key is renamed `pts3d_in_other_view`.
- Q2: point maps live in the **camera frame of image 1** (origin at its optical centre, Z forward).
- Q3/4: "metric" comes from the training objective (`ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2)`). The leading `?` means normalisation is skipped on metric-depth datasets. It is a learned statistical prior, not a measurement, so it should degrade on out-of-distribution scenes (this is what Phase 8 is meant to quantify).
- Q5: units expected to be metres (training sets ARKitScenes, ScanNet++, Habitat, TartanAir, VirtualKitti, WildRGB-D, NianticMapFree, UnrealStereo4K). **Not yet confirmed.**
- Q6: both point maps are already in one shared frame, so no pose alignment or global optimisation is needed for a single pair.
- Q7: descriptors come from concatenated encoder and decoder features through an MLP with pixel-shuffle. They are L2-normalised, so a dot product is cosine similarity.
- Q8: matching uses the repo's `mast3r/fast_nn.py::fast_reciprocal_NNs` (seed grid `subsample_or_initxy1=8`).
- Q9: `conf = 1 + exp(x)` in (1, inf), `desc_conf = exp(x)` in (0, inf). Both are unbounded and uncalibrated, so filtering must be **percentile-based**.
- Questions 10-14 (thresholding policy, pixel->3D lookup, validity of raw Euclidean distance, assumptions, failure cases) are answered progressively in Phases 3-5.

### Phase 2: Inference (done)
- Bundled scenes: `SCENES = {"chateau": croco/assets/Chateau{1,2}.png, "nle_tower": mast3r/assets/NLE_tower/ photos #0 and #3}`. `IMG_SIZE = 512`.
- `load_pair(paths)` returns two view dicts. `run_mast3r(view1, view2, model, device)` calls the official `inference([(view1, view2)], model, device, batch_size=1)`. Output keys: `view1, view2, pred1, pred2, loss(None)`.
- Pixel->3D is `pts3d[0, y, x]` (y first).
- **Scale probe:** mean `||pts3d||` = 3.005 (chateau) vs 2.368 (nle_tower), a 1.27x ratio. A scale-normalised model would give 1.0 for both, so the output carries an absolute scale. **That it is *correct* is not established.**

### Phase 3: Point maps and geometry (done)
- `extract_pointmaps(output)` returns `pm` with `pts1, pts2 [H,W,3]`, `conf1, conf2`, `dconf1, dconf2 [H,W]`, `col1, col2 [H,W,3]`. Grid is 384x512 = 196,608 points per view.
- Confidence filtering table (chateau, view 1): keep 50% -> conf >= 2.104; 25% -> 3.433; 10% -> 6.612. Default used is the 50th percentile. Non-finite points: 0.
- 5/5 geometry checks pass: all confident Z > 0 (cheirality); bbox IoU 0.235 (> 0.1) with a centroid gap of 0.782 against scene size 2.635 (shared frame, coarse); implied focal about 929.6 px on a 512 px-wide image (about 30.8 deg horizontal FOV; intrinsics are baked into the representation, so no calibration step is needed); depth varies (no collapse).

### Phase 4: Correspondences (done; run on the chateau pair)
- `extract_descriptors`, `find_correspondences(desc1, desc2, device, subsample=8)` (`dist="dot"`, `block_size=2**13`), `filter_border(m0, m1, shape0, shape1, margin=3)`, `filter_matches(m0, m1, dconf1, dconf2, percentile, extra_mask)` (score = min of the two endpoint `desc_conf`), `cross_view_residuals(pm, m0, m1)`.
- Results: 1,048 raw matches (34.1% of 3,072 seeds converged), 1,009 after the 3 px border filter (all unique). Matched-descriptor cosine similarity: min 0.646, median 0.968, max 0.996. Pixel displacement: median 56.6 px, 90th percentile 79.9 px, max 338.4 px.
- Match score: min 0.120, median 1.892, max 5.550. Percentile thresholds 25/50/75/90 keep 757/505/253/101 matches.
- **Cross-view residual `||P_view1 - P_view2||`** (the error floor for any distance), on 1,009 matches at median scene distance 2.819 model units:

| pct | residual | % of scene |
|---|---|---|
| 10 | 0.0079 | 0.28 |
| 25 | 0.0123 | 0.44 |
| 50 | 0.0224 | 0.80 |
| 75 | 0.0927 | 3.29 |
| 90 | 0.6392 | 22.68 |
| 99 | 1.0983 | 38.97 |

  Mean 0.1589, max 4.8195. The median is about 0.8% of scene scale, but the tail is heavy, so unfiltered matches can be badly wrong.
- **Secondary question D (does confidence predict consistency?):** Spearman rho(match score, residual) = -0.329 (p = 5.8e-27), so confidence carries real signal. Median residual by kept fraction: top 100% 0.0224 (0.80%), 75% 0.0197, 50% 0.0186, 25% 0.0179, top 10% 0.0156 (0.55%).

## 6. Conventions that fail silently if wrong
- Match coordinates from `fast_reciprocal_NNs` are **(x, y)** (column 0 = x). Point maps and confidence maps are indexed **`[y, x]`**. Assert both.
- View 1 key is `pts3d`. View 2 key is `pts3d_in_other_view`. There is no bare `pts3d` in `pred2`.
- `rgb(view["img"][0])` undoes ImgNorm for display. Images are normalised to about [-1, 1].
- Shapes are `(H, W) = (384, 512)` for these images. `true_shape` comes from the view dicts.
- Import order: `import mast3r.model` first (it triggers the `path_to_dust3r` and `path_to_croco` shims). Append the repo root to `sys.path` and run from a state where it resolves.
- All coordinates are in the camera frame of **image 1** (the anchor).
- Distances inherit units from the model output. Treat them as "model units, expected metres" until ground truth confirms.

## 7. Assumptions already on the record (question 13)
1. The images overlap and show a genuinely, mostly rigid scene.
2. View 1 is the anchor frame.
3. Long side is 512 (the trained regime). Other sizes are out of distribution (Phase 8, experiment B).
4. The scene is static between shots. Motion breaks the shared frame silently.
5. The confidence percentile is a choice, not a property of the scene (Phase 8, experiment E).
6. Bounding-box IoU is a coarse proxy for agreement.
7. Matched pixels are treated as the same physical point. One pixel spans a real distance that grows with depth (quantisation error floor).
8. Reciprocal NN assumes visibility in both views. Occlusion can yield confident but wrong matches (the Phase 8 occlusion failure case).

## 8. What is NOT established
- **Absolute scale correctness.** All checks so far are scale-invariant or self-consistency checks. Only tape-measured ground truth (Phase 6) can settle it.
- Units being metres. Behaviour on the author's own photos. Behaviour under resolution change, baseline change, occlusion, scene type.

## 9. The remaining jobs

**Phase 5: distance estimation (NEXT; cell 58 is empty and ready).** The last thing the author said was: they asked how to supply tape-measured input, then said "for now can we skip this part?? if we need it necessarily then maybe leave a space or add some kinda thing that'll enable us to add this and retrain it again or smth. Tell me the plan first."
So, in this order:
1. Present a short plan and get approval before writing code.
2. Implement: choose point pairs; look up 3D points `pts1[y0,x0]` and `pts2[y1,x1]`; compute `D`; report each distance with its error floor (the cross-view residual and the pixel-quantisation error); refuse (or flag) low-confidence points rather than quoting a number the model distrusts. Make selection pluggable, for example a `PAIRS` list of image-1 pixels mapped to image 2 via the nearest reliable match.
3. Demo on the bundled scenes, labelled "model estimate only, no ground truth".
4. Add an **optional ground-truth hook**: `data/scene_XX/{image1.jpg, image2.jpg, metadata.json}`. The notebook must run fine when no scene exists and pick scenes up automatically when added. Proposed `metadata.json`: pixel coordinates of each measured point in both images, the true distance in metres, and notes. (Not yet agreed; propose and confirm.)
5. Follow the notebook style in §2, then stop and wait for the author to run it.

**Phase 6: ground-truth validation.** Per the original brief: create or use image pairs whose real-world distance is known and compare the prediction against it. Per-scene layout `scene_XX/{image1.jpg, image2.jpg, metadata.json}` with e.g. `{"distance_m": 2.0}`. Prioritise controlled scenes, clearly identifiable points, measurable distances and good overlap. The author has no such data, so this needs a public dataset (see §12). The brief says "do not select a dataset just because it contains RGB images; it must allow meaningful metric-distance ground truth."

**Phase 7: evaluation.** Defined in the original brief (recovered, see §13): compute MAE, RMSE, relative error, percentage error and other useful metrics. Formulas: `AbsErr = |D_est - D_gt|`, `RelErr = |D_est - D_gt| / D_gt`, `PctErr = RelErr x 100`. Planned function: `calculate_metrics()`.

**Phase 8: experiments / ablation.** Defined in the brief. Factors to study: viewpoint difference, actual distance, image resolution, descriptor matching quality, confidence filtering, point selection, texture richness, occlusion. Experiments: (1) viewpoint baselines, (2) physical distances, (3) image resolutions, (4) with vs without confidence filtering, (5) high vs low quality correspondences, (6) texture-rich vs texture-poor. NOTE: the notebook's "Phase 8 experiment B" and "experiment E" are really the brief's **secondary questions B (resolution) and E (does filtering low-confidence correspondences help)**, not Experiments 2 and 4.

## 10. Practical notes for the next session
- Open `main.ipynb` and read cells 0-58 first. Variables live in the kernel (`model`, `output`, `pm`, `matches_im0/1`, `dconf1/2`, `DEVICE`, ...). If the kernel restarted, re-run cells in order. Phases 2-4 depend on the Phase 1 model load.
- Keep new code cells self-contained and defensively written (assertions, sensible refusal). Save figures to `FIGURES_DIR` and metrics to `METRICS_DIR` in the same style as earlier cells.
- Because the real runtime is Kaggle, the author must upload any new images (to `data/scene_XX/`) there. Writing the notebook locally does not execute it.
- Update `summary.md` as phases finish.

## 11. Known gaps and risks
- **Lost transcript.** The original Phase 1-4 chat sessions (`2d0ece51-4b54-4e2e-a819-45a114943515`, plus the shorter `b20dcc53-...` and `ba8eaa3a-...`, all run from `/home/raghunandan/3d`) have no transcript on disk, so the design discussion from those is lost beyond what the notebook records. Session `f9d58c52-7448-4705-a88e-d4a02b662156` DOES exist now (`~/.claude/projects/-home-raghunandan-3d-mast3r-metric-distance/`), but it is the *newer* session that started from the resume prompt (see §12), not the original.
- **Original brief:** the 416-line paste itself is not in the paste cache, but the author has since supplied what appears to be the original prompt (see §13), which defines Q1-Q14, secondary questions A-G and Phases 6-8.
- **`results/` is empty locally** and the local `.venv` is CPU-only Python 3.10, which differs from the Kaggle runtime.
- **`main.ipynb` is 8.5 MB** because outputs and figures are embedded. Clear outputs before putting it in git or sending by email.
- No git repo covers the project folder itself (only the nested `mast3r/` clone has `.git`). Nothing has been committed.

## 12. Decisions and discussion since the resume (session `f9d58c52`, 2026-10-05)

This comes from the newer Claude session started in this folder. Nothing here has been written into `main.ipynb` yet, and the Phase 5 plan has **not been approved** by the author.

**Proposed Phase 5 plan (awaiting "approved"):**
- `PAIRS`, pluggable: named points in image 1 given as pixel `(x, y)`. The second endpoint is found by `"match"` (nearest reliable reciprocal match within a search radius, using `pts2[y1, x1]`) or `"direct"` (user supplies the image-2 pixel too, and the cross-view residual is checked). Distances are `D(a, b)` for declared pairs. Assert `(x, y)` vs `[y, x]` on every lookup.
- Refuse (don't report) when: no reliable match within the radius; descriptor score below threshold; geometry confidence below the Phase 3 percentile threshold; depth not positive; or the error bound exceeds a fraction of `D` (proposed 10%, to go on the assumptions record).
- Per-distance error floor: the *local* median cross-view residual near each endpoint (not the global Phase 4 median), plus a quantisation term of about `Z / f` per endpoint (f from the Phase 3 focal estimate), combined in quadrature, and also report the safer linear sum.
- Demo on `chateau` and `nle_tower`, labelled "model estimate only, no ground truth".
- Optional ground-truth hook: glob `data/scene_*/metadata.json`. If none, print "no ground-truth scenes found" and run the demo only. Proposed layout `data/scene_01/{image1.jpg, image2.jpg, metadata.json}` with schema:
  `{"scene_id", "image1", "image2", "pixel_space": "original", "points": {"A": {"image1_xy": [x,y], "image2_xy": [x,y]}, ...}, "pairs": [{"a","b","true_distance_m","uncertainty_m","method","notes"}], "notes"}`.
  `pixel_space: "original"` means pixels are on the saved photo, so the code must map them through the resize-to-512-long-side and crop-to-multiple-of-16 that `load_images` applies. Verify this against the repo's `load_images` before relying on it.
- Two choices the author may change: the 10% refusal fraction, and defaulting to "direct" mode when both pixels are supplied.

**Facts the author stated:**
- They **do not have their own tape-measured dataset**.
- **Rendering synthetic scenes is not possible** for them, so a *real* dataset with 3D ground truth is wanted.

**Dataset candidates suggested (none chosen or downloaded yet):** ETH3D (suggested first: DSLR photos closest to phone quality, laser-scan ground truth), ScanNet (indoor RGB-D, needs a signed terms-of-use form), NYU Depth V2 (easy, low-res, older), DTU MVS (clean turntable objects), 7-Scenes (small indoor RGB-D), KITTI (metric LiDAR, car viewpoint, unlike hand-held pairs), Tanks and Temples (laser-scan, larger outdoor scale, harder setup). Check each licence (several are non-commercial research only). Converting a chosen dataset into the `data/scene_XX/` layout, including ground-truth distances computed from its 3D data, is a new work item sitting between Phase 5 and Phase 6.

**Estimate given to the author:** about 3-5 weeks part-time. Phase 5: a few days. Dataset preparation: 1-2 weeks (the biggest unknown). Phase 6: 3-5 days after data is ready. Phase 7: cannot estimate (undefined). Phase 8: 1-2 weeks. Compute is not the bottleneck.

**Still needed from the author:** approval of the Phase 5 plan, and a choice of dataset. Phase 7 and the Phase 8 experiments are now defined by the original brief (§13).

## 12b. Application framing and dataset direction (2026-10-06)

**Application.** The author wants to recreate the idea behind Amazon's "view in your room" AR, but from **two photos only**: estimate the metric distance between two places/objects in the user's home, then say whether a given object fits (yes / no / uncertain). The fit decision uses the Phase 5 error bound as its margin. This motivates the research question; it does not replace it.

**Assessment given to the author (keep honest in write-ups):**
- Strengths: clear practical framing; no AR hardware (Amazon's AR relies on ARKit/ARCore tracking, IMU, sometimes LiDAR); a thresholded fit decision with a margin is more forgiving than quoting an exact distance.
- Cautions: not provably "unique" (room-measuring and furniture-fit apps exist; two-image feed-forward metric reconstruction for this is less common, but no literature search has been done). Fit needs more than a point-to-point distance (free volume or floor area); start with point-to-point width, then height. A fit check needs about 2-5 cm accuracy on 1-3 m gaps, unproven. Metric scale from uncalibrated pairs is the weakest link (plausible 5-15% scale error). Textureless walls and floors hurt matching.
- Possible new late phase (after Phase 8): fit check with inputs = object dimensions + two chosen points, output = fits / does not fit / uncertain. Only worth building once Phases 6-8 give real accuracy numbers.

**Dataset direction.** Ground truth does not require the author's own tape measurements. True distances are derived from a dataset's depth/3D data (back-project two pixels with GT depth, intrinsics and pose, then take the 3D distance); accuracy is bounded by the sensor. The author prefers an **indoor** dataset because the target is distance between small objects. Candidates: 7-Scenes, NYU Depth V2 (Kinect RGB-D, roughly 1-2 cm noise), ScanNet (needs terms-of-use form). Avoid datasets in MASt3R's training mix (ARKitScenes, ScanNet++, Habitat, BlendedMVS, MegaDepth, CO3D, Waymo, etc.) to avoid flattering results; this list is from memory and **must be verified against the MASt3R paper** (see Q5 for the repo-recorded list). ETH3D (laser GT) stays an option for a high-accuracy check. No dataset is chosen or downloaded. Tape-measured home pairs remain an optional later real-world test.

**Still open:** the specific indoor dataset; whether the ground-truth hook goes into Phase 5 or Phase 5 stays model-only; the Phase 5 plan is still unapproved.

## 13. The original brief (recovered, verbatim in `ORIGINAL_BRIEF.md`)

The author supplied the exact original prompt. It is saved verbatim as `ORIGINAL_BRIEF.md` in this folder: **read it first**. It is the spec. Summary and how the project has deviated from it:

- **Role and scope:** CV researcher plus hands-on implementation mentor. NOT a reimplementation of MASt3R and NOT a MASt3R-SfM project. No MASt3R-SfM, COLMAP, calibration or bundle adjustment unless discussed first as an optional extension. Hard constraints: no hallucinated APIs, no silent DUSt3R substitution, never assume coordinates are metres without verification, no premature optimisation, and **never start a new phase without the author's explicit go-ahead**, even if the last one clearly succeeded.
- **Phases 1-8:** Setup; Inference; Pointmap extraction; Correspondence matching; Distance estimation; Ground-truth validation (known real distances, `scene_XX/{image1.jpg, image2.jpg, metadata.json}` with e.g. `{"distance_m": 2.0}`); Evaluation (MAE, RMSE, relative and percentage error, other useful metrics); Experiments/ablation (viewpoint difference, actual distance, resolution, matching quality, confidence filtering, point selection, texture, occlusion).
- **Questions Q1-Q14** (Q1-Q9 answered in Phase 1, Q10 in Phase 3, Q11 in Phase 2; Q12-14 open until Phases 5-8) and **secondary questions A-G**: A viewpoint change; B resolution; C physical distance vs relative error; D confidence vs accuracy (early read in Phase 4, rho = -0.329); E filtering low-confidence matches; F texture-rich vs poor; G which scenes give the largest errors. **Planned experiments 1-6:** baselines, physical distances, resolutions, with/without confidence filtering, high vs low quality correspondences, texture-rich vs poor. The notebook's "Phase 8 experiment B / E" mean secondary questions B and E.
- **Required final outputs (11):** input pair viz [done]; point-map viz [done]; correspondence viz [done]; selected measurement points; predicted 3D coordinates; ground-truth distance; predicted distance; absolute error; relative error; summary metrics table; plots (GT vs predicted, GT vs absolute error, GT vs relative error, viewpoint difference vs error, confidence vs error).
- **Planned functions:** `load_model, load_images, run_mast3r, extract_pointmaps, extract_descriptors, find_correspondences, filter_matches, get_3d_point, estimate_distance, calculate_metrics, visualize_results`. The first seven exist in some form in `main.ipynb`. The last four are still to write.
- **Per-phase delivery:** simple objective, technical concept, what MASt3R does internally, implementation, expected output, sanity checks, then a stop for the go-ahead. Keep a **Simple explanation / Technical explanation** pair for everything (the author must present to a professor). Final story: Problem -> Approach -> 9-step Method -> Research question.
- **Style:** modular code, comments explain WHY, print shapes and intermediates, explain every tensor shape such as `[B,H,W,3]`. Debugging: explain the error, trace it to its real source, inspect the installed repo code, fix the smallest component, re-run only the relevant phase.

**Deliberate deviations from the brief (made by the author during the work):**
1. The brief said the AI should run code itself via a terminal on the Kaggle GPU (through VS Code) and lay out `phases/phaseN_*.py` plus `src/{inference,matching,distance,evaluation,visualization}.py`. In practice the author did **not** want separate scripts: they asked for everything to be written directly into `main.ipynb` and said they would run each cell themselves and report errors. Only `phases/phase1_setup.py` and `src/paths.py` exist from the original layout.
2. The brief says stop after each phase and ask for a go-ahead. The author now says "proceed with phase N" explicitly. Keep that gate.
3. If the next Claude *does* have terminal access to a GPU, the brief's original intent (run it yourself and show real output) is still the stated spec. Otherwise follow the notebook workflow. Ask the author which applies.
