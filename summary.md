# Metric Distance Estimation with MASt3R: Project Summary

**Status:** Phases 1-4 complete and run. Phase 5 (distance estimation) is next. Phases 6-8 are planned but not yet written.
**Where the work lives:** `main.ipynb` (all phases, with saved outputs), `src/paths.py` (paths and import bootstrap), and `phases/phase1_setup.py` (an early standalone version of Phase 1, since superseded by the notebook).

> Note on provenance: the original chat transcript was lost, so this summary is reconstructed from `main.ipynb` (including its saved cell outputs) and the prompt history. Anything not recorded in the notebook is marked as unknown rather than guessed.

## 1. Goal

**Research question:** how accurately can a *pretrained, metric* MASt3R model estimate real-world distances from **uncalibrated** image pairs, and what factors influence that accuracy?

**Pipeline:**

```
two uncalibrated images
  -> MASt3R (pretrained metric checkpoint, off the shelf, no fine-tuning)
  -> 3D point maps + dense descriptors
  -> pixel-to-pixel correspondences
  -> corresponding 3D points
  -> Euclidean distance D = sqrt(dX^2 + dY^2 + dZ^2)
  -> compare with ground truth -> error analysis + experiments
```

## 2. Setup

- **Environment:** Kaggle remote kernel, Tesla T4 (14.6 GB), Python 3.12, fp32. Inference takes about 0.8 s per pair and uses about 2.6 GB of VRAM for the model.
- **Model:** official `naver/mast3r` repo, with the `dust3r` and `croco` submodules. Checkpoint `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` (about 2.75 GB, 688.6 M parameters).
- **Working method:** every phase is written directly into `main.ipynb`. The user runs the cells and reports errors before the next phase is written. The interactive 3D view was not usable, so static matplotlib figures are used.

## 3. What has been established (Phases 1-4)

### Phase 1: Setup and verification
- The checkpoint's recorded architecture string matches the README's *metric* training recipe on all 15 kwargs, and differs from the demo recipe.
- Model loads cleanly (`All keys matched`) on the GPU.
- Output heads: `pts3d` is unbounded and never rescaled, `conf = 1 + exp(x)` in (1, inf), and `desc_conf = exp(x)` in (0, inf).
- Findings read from source:
  - Both point maps are expressed in **view 1's camera frame**, so no pose alignment is needed for a single pair.
  - Descriptors are 24-D and L2-normalised, so a dot product is cosine similarity.
  - "Metric" comes from the training objective (`Regr3D` with `?avg_dis`, which skips normalisation on metric-depth datasets). It is a learned prior, not a measurement, so it may degrade on scenes unlike the training data.

### Phase 2: Inference
- The official `load_images` -> `inference` path runs end to end. Long side is 512, the trained regime.
- Pixel -> 3D is a plain lookup, `pts3d[0, y, x]` (y first).
- **Scale probe on two bundled scenes:** mean `||pts3d||` was 3.005 (chateau) and 2.368 (NLE tower), a 1.27x ratio. Neither sits at 1.0, so the output is **not** scale-normalised and does carry an absolute scale.
- This shows the scale exists. It does not show that the scale is **correct**.

### Phase 3: Point maps and 3D geometry
- Dense point maps extract cleanly and are pixel-aligned with confidence and colour.
- Confidence filtering must be **percentile-based**, because `conf` is unbounded and uncalibrated.
- Geometry checks (5/5 passed):
  - All confident points are in front of the camera (Z > 0).
  - Bounding-box IoU between the two views' clouds is 0.235 (> 0.1), consistent with a shared frame.
  - The implied focal length is about 930 px on a 512 px-wide image (about 30.8 deg horizontal FOV), recovered from the point map alone.
  - Depth varies across the scene, with no collapse.

### Phase 4: Correspondence matching
- Matches come from the repo's own `fast_reciprocal_NNs` (iterative mutual nearest neighbours on the 24-D descriptors, seeded on an 8 px grid). A 3 px border filter follows the README.
- `filter_matches` scores each match by `min(desc_conf)` over its two endpoints.
- **Cross-view consistency test** (the error floor for later distances), on 1,009 matches at a median scene distance of 2.819 model units:

| Percentile | Residual (model units) | % of scene distance |
|---|---|---|
| 10th | 0.0079 | 0.28% |
| 50th (median) | 0.0224 | 0.80% |
| 75th | 0.0927 | 3.29% |
| 90th | 0.6392 | 22.68% |
| 99th | 1.0983 | 38.97% |

  The median is small, but the tail is heavy, with a mean of 0.159 and a max of 4.82. Good matches agree to under 1%, while a minority of matches are badly wrong. Distances should therefore be taken from confident matches only.
- The notebook also computes a Spearman correlation between match confidence and residual (secondary question D). I did not re-read its result for this summary, so check the cell output.

## 4. What is NOT established
- **Accuracy against the real world.** Everything so far measures the model's agreement with itself. A model can be confidently, consistently wrong in absolute scale. Only tape-measured ground truth can settle this.
- Whether the output units are metres is expected but unconfirmed.

## 5. Assumptions on record
1. The two images overlap and show a mostly rigid scene.
2. View 1 is the anchor frame.
3. Long side is 512 (other sizes are out of distribution).
4. The scene is static between shots.
5. The confidence percentile is a choice, not a scene property.
6. Bounding-box IoU is only a coarse proxy for agreement.
7. One pixel spans a real distance that grows with depth, which is a quantisation error floor.
8. Reciprocal NN assumes the point is visible in both views. Occlusion can give confident but wrong matches.

## 6. Conventions (easy to get wrong)
- Match coordinates are **(x, y)**. Point maps are indexed **`[y, x]`**. Mixing them up fails silently.
- `pred1` uses key `pts3d`. `pred2` uses `pts3d_in_other_view`.

## 7. Next steps
- **Phase 5, distance estimation:** the user asked to skip tape-measured data for now. The agreed direction is to select corresponding point pairs, look up their 3D points, compute D, and report each distance with its error floor, using the bundled scenes. Add an optional ground-truth hook for later: `data/scene_XX/{image1.jpg, image2.jpg, metadata.json}`. The model is frozen, so adding data later only means re-running evaluation, not retraining. A plan was to be agreed before code is written.
- **Phase 6:** accuracy against tape-measured ground truth. Needs the user's own scenes, with a spread of true distances and generous overlap.
- **Phase 8 experiments:** B (input resolution off 512) and E (confidence-percentile filtering) are known. The rest of Phases 7-8 is not recorded in the notebook and should be taken from the original brief.
