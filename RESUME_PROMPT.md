I'm resuming a project from an earlier chat whose transcript was lost. Everything below is reconstructed from `main.ipynb` and my prompt history. Read `main.ipynb` (cells 0-58) and `src/paths.py` before writing anything, so your code matches the existing style and variable names.

## Project
Metric distance estimation from uncalibrated image pairs via pretrained metric MASt3R.
Research question: how accurately can the pretrained metric MASt3R checkpoint estimate real-world distances from uncalibrated image pairs, and what factors affect that accuracy?

Pipeline: image pair -> MASt3R -> point maps + dense descriptors -> pixel correspondences -> corresponding 3D points -> Euclidean distance -> compare with ground truth -> error analysis.

Location: `/home/raghunandan/3d/mast3r_metric_distance/`. Layout: `main.ipynb`, `src/paths.py`, `phases/`, `mast3r/` (official repo), `checkpoints/` (metric .pth already downloaded), `data/` (empty), `results/{figures,metrics}`.
Runtime: the notebook was written for a Kaggle remote kernel (Tesla T4, Python 3.12, fp32).

## Working rules (from my earlier instructions)
- Put all work directly into `main.ipynb`, phase by phase. Do NOT create standalone scripts for phases.
- I run each cell myself and report back errors or output. Do not move on to the next phase until I say so ("Proceed with phase N").
- Keep the existing notebook structure for each phase: Objective (simple + technical explanation), "What MASt3R is doing internally", numbered sub-sections (N.1, N.2, ...), a code cell each, then a "Phase N summary" with "What was established", "What is still NOT established", "Assumptions added to the record (question 13)", and a "Next" note.
- Use the official repo's own functions (`load_images`, `inference`, `fast_reciprocal_NNs`). Don't reinvent them.
- I can't use interactive 3D (plotly etc.), so use static matplotlib figures.
- Smoke-test any Python you write (`python -c` / `node --check` equivalents) before presenting it as done.

## State: Phases 1-4 are done and run by me
- Phase 1: setup, checkpoint inspection, model instantiation.
- Phase 2: inference. Defined `load_pair`, `run_mast3r`, `SCENES` (chateau, nle_tower), `IMG_SIZE=512`. Confirmed that both point maps are in view 1's camera frame.
- Phase 3: `extract_pointmaps(output)` returns `pm` with keys pts1, pts2 [H,W,3], conf1/2, dconf1/2, col1/2. Confidence filtering is percentile-based (conf is unbounded). Added a bounding-box IoU frame check and a focal-length estimate.
- Phase 4: `find_correspondences`, `filter_border` (3 px margin), `filter_matches` (score = min desc_conf of the two endpoints), and `cross_view_residuals(pm, m0, m1)`. The median residual is the error floor for any distance quoted.

Conventions that must not be broken:
- Matches are (x, y) but point maps are indexed [y, x]. Assert it.
- Output is in units as predicted by the model (expected metres, which Phase 2.6 tested). A model can be self-consistent yet uniformly wrong in scale, and only ground truth settles that.

Existing assumptions on record: 1. overlapping, mostly rigid scene; 2. view 1 is the anchor frame; 3. long side 512; 4. static scene; 5. confidence percentile is a choice, tested in Phase 8 experiment E; 6. bbox IoU is coarse; 7. one-pixel quantisation error grows with depth; 8. reciprocal NN assumes the point is visible in both views (the occlusion failure mode).

## Where we stopped
I asked how to supply tape-measured ground truth for Phase 5. Then I said: "for now can we skip this part?? if we need it necessarily then maybe leave a space or add some kinda thing that'll enable us to add this and retrain it again or smth. Tell me the plan first."

## What I want now
1. **Do not write code yet. First give me a short plan** for Phase 5 (distance estimation) that works WITHOUT my own tape-measured data:
   - Select specific corresponding point pairs, look up their 3D coordinates (`pts1[y0,x0]` and `pts2[y1,x1]`), and compute D = sqrt(dX^2 + dY^2 + dZ^2).
   - Report each distance together with its error floor from the cross-view residual and the quantisation error.
   - Make point selection pluggable: for example a `PAIRS` list of pixel coordinates in image 1, mapped to image 2 via the nearest reliable match, with refusal when confidence is low.
   - Use bundled scenes (chateau, nle_tower) for demonstration, where no ground truth exists and results are labelled "model estimate only".
   - Leave a clearly marked, optional ground-truth hook: `data/scene_XX/{image1.jpg, image2.jpg, metadata.json}`. Propose a `metadata.json` schema (point pixel coordinates in each image, true distance in metres, notes). The Phase 5 code should run fine when no `data/scene_*` exists and automatically pick up scenes when they are added later. Note that "retrain" is not needed, since the model is pretrained and frozen: only the evaluation is re-run.
2. After I approve the plan, write Phase 5 into `main.ipynb`, then stop and wait for me to run it.
3. Later phases, in order, each only on request:
   - Phase 6: accuracy against ground truth. Needs my data, so it is deferred and the hook above covers it.
   - Phase 7 (inferred, check against the notebook intro).
   - Phase 8: failure and robustness experiments. Known so far: B = input resolution off the trained 512 long side; E = confidence-percentile filtering. A, C, D are not recorded in the notebook, so ask me rather than guessing.
   Also note that the original 416-line brief I pasted in the first chat is not recoverable. If the notebook intro doesn't define Phases 6-8, ask me.
