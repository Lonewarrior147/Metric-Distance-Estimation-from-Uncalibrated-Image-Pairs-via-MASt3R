# Continuation Brief: MASt3R Metric Distance Estimation, Phases 5-8

You are continuing a project that is already partly built. Phases 1-4 are done and verified. Do not redo them.

## 0. Read first (before writing anything)

1. `context.md`: full state of the folder, results so far, conventions, decisions.
2. `ORIGINAL_BRIEF.md`: the original spec (Q1-Q14, secondary questions A-G, required outputs, hard constraints).
3. `main.ipynb` cells 0-58 and `src/paths.py`: the existing code and style. New code must reuse the existing variable and function names.

Then confirm in two or three lines that you have read them, and give the Phase 5 plan (section 5) **before writing any code**.

## 1. Role

Act as both a computer vision researcher who understands DUSt3R/MASt3R deeply and a practical implementation mentor. Build incrementally and scientifically, one verified phase at a time.

## 2. Project in one paragraph

Research question: *How accurately can a pretrained metric MASt3R model estimate real-world distances from uncalibrated image pairs, and what factors influence this accuracy?*
Pipeline: image pair -> MASt3R (pretrained, frozen, metric checkpoint) -> 3D point maps + 24-D descriptors -> reciprocal-NN pixel correspondences -> corresponding 3D points -> `D_est = sqrt((X1-X2)^2 + (Y1-Y2)^2 + (Z1-Z2)^2)` -> compare with `D_gt` (`AbsErr = |D_est-D_gt|`, `RelErr = AbsErr/D_gt`, `PctErr = RelErr*100`) -> error analysis and experiments.
Not in scope: reimplementing MASt3R, MASt3R-SfM, COLMAP, calibration, bundle adjustment (only as an optional extension, discussed first).

## 3. Working rules (the author's own, follow exactly)

1. **All work goes into `main.ipynb`**, appended after the existing cells (cell 58 is an empty code cell where Phase 5 begins). No separate per-phase scripts.
2. **The author runs each cell** (on a Kaggle T4 kernel) and reports errors or output. Do not claim a cell works until they report it. If this session has working GPU access, say so and ask whether to run cells yourself.
3. **Phase gate:** write one phase, stop, and wait for the author to say "proceed with phase N". Never start the next phase on your own, even if the last one clearly succeeded.
4. **Plan first** for each phase: give a short plan in plain terms and get approval before coding.
5. **Notebook style per phase:** `## PHASE N - TITLE`; `### Objective` with a *Simple explanation* and a *Technical explanation*; `### What MASt3R is doing internally at this stage`; numbered sub-sections `### N.1`, `### N.2`, ...; one code cell each; then `### Phase N summary` with *What was established*, *What is still NOT established*, *Assumptions added to the record (question 13)* (continue numbering from 9), and a `### Next` note.
6. **Dual explanation everywhere** (simple + technical). The author must present this to a professor.
7. Use the official repo functions (`load_images`, `inference`, `fast_reciprocal_NNs`). Never invent APIs. If unsure, read the pinned source in `mast3r/` (mast3r f5209af, dust3r 3cc8c88, croco d7de070).
8. Static matplotlib figures only (the author cannot use interactive 3D). Save figures to `FIGURES_DIR` and metrics to `METRICS_DIR`.
9. Modular code, not one huge cell. Comments explain WHY. Print tensor shapes and key intermediates. Add assertions and sanity checks. Explain every tensor shape such as `[B,H,W,3]`.
10. Debugging: explain what the error means, find its true source in the installed code, fix the smallest component, re-run only the relevant part. Do not rewrite the pipeline.
11. Smoke-test any Python before presenting it (at minimum `python -c`/syntax and shape checks on synthetic arrays).
12. Hard constraints: no hallucinated APIs; no silent DUSt3R substitution; never assume the output is in metres without evidence; no premature optimisation.

## 4. Current state (done and verified)

| Phase | Done | Key names in the kernel |
|---|---|---|
| 1 Setup | model loaded on T4, fp32 | `model`, `DEVICE`, paths in cell 4 |
| 2 Inference | `load_pair`, `run_mast3r`, `SCENES` (chateau, nle_tower), `IMG_SIZE=512` | `output`, `view1`, `view2` |
| 3 Point maps | `extract_pointmaps`, percentile confidence filtering | `pm` (pts1, pts2, conf1, conf2, dconf1, dconf2, col1, col2) |
| 4 Matching | `extract_descriptors`, `find_correspondences`, `filter_border`, `filter_matches`, `cross_view_residuals` | `matches_im0/1`, `dconf1/2`, `resid`, `mask1` |

Measured so far (chateau pair, 384x512):
- Scale probe: mean `||pts3d||` = 3.005 (chateau) vs 2.368 (nle_tower), so the output is not scale-normalised. **Correctness of the scale is not established.**
- Both point maps are in image 1's camera frame (`pred2` uses `pts3d_in_other_view`).
- 1,009 matches after the 3 px border filter. Cross-view residual median 0.0224 (0.80% of the 2.819 scene distance), p75 3.3%, p90 22.7%, max 4.82. Heavy tail, so unfiltered matches can be badly wrong.
- Spearman(match score, residual) = -0.329: confidence carries signal. Keeping the top 10% of matches lowers the median residual to 0.0156 (0.55%).
- Focal implied by the point map: about 930 px on a 512 px-wide image.

Conventions that fail silently: match coordinates are **(x, y)**, point maps are indexed **`[y, x]`**. Assert both. All coordinates are in image 1's camera frame. Units are "model units, expected metres, unconfirmed".

Assumptions already on record (1-8): overlapping mostly rigid scene; image 1 is the anchor; 512 long side; static scene; confidence percentile is a choice; bbox IoU is a coarse check; pixel quantisation grows with depth; reciprocal NN assumes visibility in both views.

## 5. Remaining work

### Phase 5: distance estimation (start here; plan first)
Proposed plan (revise with the author, then get "approved"):
- **Point selection, pluggable.** `PAIRS` of named points in image 1 given as pixel `(x, y)`. The image-2 endpoint is found by `"match"` (nearest reliable reciprocal match within a search radius, using `pts2[y1, x1]`) or `"direct"` (author supplies the image-2 pixel, and the cross-view residual is checked). Declare distances as `D(a, b)`.
- **Functions to add:** `get_3d_point()`, `estimate_distance()`.
- **Refuse, don't report,** when: no reliable match in the radius; descriptor score below threshold; geometry confidence below the Phase 3 percentile threshold; depth not positive; or the error bound exceeds a fraction of `D` (proposed 10%, to be recorded as an assumption).
- **Per-distance error floor:** local median cross-view residual near each endpoint (not the global one), plus a quantisation term of about `Z / f` per endpoint (f from the Phase 3 focal estimate), combined in quadrature, and also the safer linear sum.
- **Demo** on `chateau` and `nle_tower`, every result labelled "model estimate only, no ground truth".
- **Ground-truth hook (optional, no retraining):** glob `data/scene_*/metadata.json`; if none, print "no ground-truth scenes found" and run the demo only. Proposed schema:
  ```json
  {"scene_id": "scene_01", "image1": "image1.jpg", "image2": "image2.jpg",
   "pixel_space": "original",
   "points": {"A": {"image1_xy": [412, 233], "image2_xy": [398, 240]}},
   "pairs": [{"a": "A", "b": "B", "true_distance_m": 1.25, "uncertainty_m": 0.005, "method": "tape", "notes": ""}]}
  ```
  `pixel_space: "original"` means pixels refer to the saved photo, so map them through the resize-to-512-long-side and crop-to-multiple-of-16 that `load_images` applies. Verify this against the repo's `load_images`, do not assume.
- Also include the brief's required outputs 4-7 for this phase: selected points drawn on both images, predicted 3D coordinates, predicted distance.

### Dataset (decide before Phase 6)
The author has **no tape-measured data and cannot render synthetic scenes**. A real public dataset with 3D ground truth is needed, one that actually allows metric distance ground truth (not just RGB). Candidates already suggested: ETH3D (suggested first; DSLR images, laser-scan GT), ScanNet (needs a signed terms form), NYU Depth V2, DTU MVS, 7-Scenes, KITTI, Tanks and Temples. Check licences. Plan a conversion step that turns the chosen dataset into `data/scene_XX/` with distances computed from its 3D data. Discuss and get the author's choice first. Do not download anything large without asking.

### Phase 6: ground-truth validation
Run the pipeline on the prepared scenes and compare `D_est` with `D_gt`. Outputs: ground-truth distance, predicted distance, absolute error, relative error, per scene. Also confirm or refute that the units are metres (Q5) and answer Q12 (is raw Euclidean distance valid for the chosen measurement setup).

### Phase 7: evaluation
`calculate_metrics()`: MAE, RMSE, relative error, percentage error, plus other useful measures (median error, bias/signed error, scale ratio `D_est/D_gt`). Summary metrics table. Plots: GT vs predicted distance; GT vs absolute error; GT vs relative error.

### Phase 8: experiments and ablation
Answer secondary questions A-G with experiments: (1) viewpoint baselines, (2) physical distances, (3) image resolutions (off the trained 512 long side), (4) with vs without confidence filtering (and the percentile choice), (5) high vs low quality correspondences, (6) texture-rich vs texture-poor. Also point selection and occlusion. Plots: viewpoint difference vs error; confidence vs error. Finish with Q13 (full assumptions list) and Q14 (failure cases, with evidence) and `visualize_results()`.

### Final deliverable checklist (from the brief)
1 input pair viz [done] · 2 point-map viz [done] · 3 correspondence viz [done] · 4 selected points · 5 predicted 3D coordinates · 6 ground-truth distance · 7 predicted distance · 8 absolute error · 9 relative error · 10 summary metrics table · 11 the five plots listed above.
Final story to be able to present: Problem (metric distance from images usually needs calibration, explicit geometry or special sensors) -> Approach (MASt3R's learned 3D representation and dense correspondence on uncalibrated pairs) -> Method (the 9 steps) -> Research question.

## 6. Housekeeping

- Keep `summary.md` and `context.md` updated as phases finish.
- `main.ipynb` is large because outputs are embedded; do not print whole cells back, read what you need.
- The notebook was written for Kaggle (`/kaggle/working`, GPU asserted). The local machine has no GPU.

## 7. Your first reply

1. Confirm you have read the three sources (two or three lines, no summary of the whole project).
2. Present the Phase 5 plan in simple terms, noting the open choices (10% refusal fraction; "direct" as default when both pixels are given; `metadata.json` schema).
3. Ask one thing: approval to write Phase 5, and which dataset the author wants for Phase 6.
4. Do not write notebook code until the author approves.
