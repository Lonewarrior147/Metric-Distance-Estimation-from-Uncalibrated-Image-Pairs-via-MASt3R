# Project Brief: Metric Distance Estimation from Uncalibrated Image Pairs via MASt3R

> Verbatim copy of the original project prompt, as pasted by the author. Later deviations (single `main.ipynb`, user runs cells) are recorded in `context.md`, not here.

## 0. Your Role

You are acting in two capacities simultaneously, and you should maintain both throughout this project:

1. **A computer vision researcher** with deep, current knowledge of DUSt3R, MASt3R, and dense 3D correspondence methods — someone who understands the underlying theory, not just the API surface.
2. **A hands-on implementation mentor** guiding a real, working implementation on a Kaggle GPU session — someone who writes runnable code, checks it against the actual current repository, and helps debug failures methodically rather than guessing.

**You have direct terminal/execution access to this environment** (Kaggle GPU, accessed through VS Code). Use it. For each phase: implement the code, **run it yourself**, inspect the actual output, confirm the sanity checks pass against real results, and debug any failures yourself using the debugging strategy in Section 11 — do not just write code and ask me to run it. The only things that should come back to me are: (a) results/decisions that genuinely require my judgment (e.g., confirming a ground-truth measurement, visually approving a plot, choosing between two valid approaches), and (b) the phase-completion checkpoint described below.

You are not being asked to write a one-shot final implementation. You are being asked to build this project **incrementally, phase by phase, with verification at each step**, the way a careful research collaborator would.

---

## 1. Project Title and Objective

**Title:** Metric Distance Estimation from Uncalibrated Image Pairs via MASt3R

**Objective:** Investigate whether MASt3R — used as a pretrained, off-the-shelf model — can estimate real-world metric distances between corresponding points or objects, given only an uncalibrated pair of images (no known camera intrinsics, no calibration target, no depth sensor).

**Scope clarification (read carefully):**
- This is **NOT** a project to reimplement MASt3R's architecture or training procedure from scratch.
- This is **NOT** primarily a MASt3R-SfM (multi-view Structure-from-Motion) project.
- The actual work is: run official pretrained MASt3R inference on image pairs → extract its 3D point maps and dense descriptors → establish pixel-level correspondences → estimate metric distances between corresponding 3D points → evaluate those estimates against ground truth → analyze error behavior across conditions.

---

## 2. Core Conceptual Pipeline

The end-to-end pipeline, at a conceptual level, is:

```
Two uncalibrated images
        ↓
      MASt3R
        ↓
3D PointMaps + Dense Descriptors
        ↓
Pixel-to-pixel correspondences
        ↓
Corresponding 3D points
        ↓
Euclidean distance
        ↓
Estimated metric distance
        ↓
Compare with ground truth
        ↓
Error analysis + experiments
```

**Core mathematics.** For a pair of corresponding 3D points recovered from the two views:

- P1 = (X1, Y1, Z1)
- P2 = (X2, Y2, Z2)

The estimated distance is:

```
D_est = sqrt((X1 - X2)^2 + (Y1 - Y2)^2 + (Z1 - Z2)^2)
```

This is then compared against the known ground-truth distance `D_gt` using:

- **Absolute Error** = `|D_est - D_gt|`
- **Relative Error** = `|D_est - D_gt| / D_gt`
- **Percentage Error** = `Relative Error × 100`

Keep this pipeline visible as the throughline of the whole project — every phase below exists to correctly implement one link in this chain.

---

## 3. Conceptual Foundations You Must Keep Distinct

It is easy to blur these three related-but-different things. Do not let that happen at any point in the project, in code, or in explanations.

**DUSt3R**
- Predicts dense 3D point maps directly from an image pair.
- Its key contribution is *reducing dependence on traditional explicit geometric reconstruction* (no need for calibration, no explicit triangulation pipeline).
- Every pixel in an image can be associated with a predicted 3D point.

**MASt3R**
- Builds on DUSt3R by adding **learned dense descriptors/features** on top of the point-map prediction.
- Because of this, MASt3R answers *two* questions simultaneously for every pixel:
  1. "Where is this point in 3D space?" (from the point map)
  2. "Which point in the other image does this correspond to?" (from the dense descriptors)
- This dual capability — geometry *and* correspondence — is exactly why MASt3R, not DUSt3R, is the right tool for this project's distance-estimation task.

**MASt3R-SfM**
- A full multi-image Structure-from-Motion pipeline built around MASt3R.
- **Not required** for this project, since our problem is defined over *image pairs*, not multi-view scenes.
- Do not introduce MASt3R-SfM, COLMAP, explicit camera calibration, or bundle adjustment unless they later become useful as an *optional* comparison/extension — and even then, only after the core pairwise pipeline is working and only with explicit discussion first.

---

## 4. Computational Environment

The implementation environment is fixed as:

- **Kaggle GPU, accessed through a remote VS Code session** — you have direct terminal access to this environment through your execution tools. Use it to run installation commands, inference scripts, and tests directly, rather than only producing code for me to run manually.
- Python
- PyTorch
- The **official** MASt3R repository
- The **official pretrained MASt3R metric checkpoint**

**Before writing any implementation code**, you must:

1. Inspect the *current* official MASt3R repository and its documentation — do not rely on possibly outdated code patterns from training data or from generic tutorials found online.
2. Confirm the correct model-loading API as it exists in the repository today.
3. Confirm the correct inference API.
4. Confirm the actual output structure returned by inference (tensor shapes, keys, dtypes).
5. Confirm how the official repository itself performs descriptor matching (don't invent a matching procedure — use theirs, or explicitly justify a deviation).
6. Identify the correct **metric** checkpoint specifically (as opposed to a non-metric / relative-scale checkpoint).
7. Verify all code paths are compatible with the current repository version, not a historical one.

If internet or repository access is available, **inspect the actual source code and docs** rather than guessing. If what you find in the real repository differs from what tutorials or memory suggest, **trust the real repository.**

---

## 5. Dataset Strategy

Start deliberately small and controlled — this is not a "collect lots of images" project, it is a "get accurate ground truth on a few good pairs" project.

- Do **not** start with a large dataset.
- Begin with a small, controlled set of image pairs where ground-truth distances can be measured reliably (e.g., with a tape measure, known object dimensions, or a controlled rig).
- Each scene should be organized as:

```
scene_01/
    image1.jpg
    image2.jpg
    metadata.json
```

Example `metadata.json`:

```json
{
    "distance_m": 2.0
}
```

**Selection priorities, in order:**
- Controlled scenes
- Clearly identifiable points/objects to measure between
- Measurable, trustworthy real-world distances
- Sufficient visual overlap between the two views

Only after the core pipeline is validated on this small controlled set should you investigate suitable public datasets. **A dataset is not suitable just because it contains RGB image pairs** — it must also provide (or allow deriving) reliable metric-distance ground truth.

---

## 6. Implementation Phases (Phase-Gated Workflow)

Implement the project in **exactly** these phases, in this order, and do not skip ahead. For each phase: implement it, **run it yourself in this environment via the terminal**, and confirm it actually works using the sanity checks — using real output you obtained, not hypothetical output. Once a phase is verified working, **stop and explicitly ask me for a go-ahead before starting the next phase.** Do not proceed automatically, even if you're confident the phase succeeded.

**PHASE 1 — SETUP**
Install MASt3R and its dependencies on Kaggle GPU. Verify that GPU access, PyTorch, CUDA, the cloned repository, and the downloaded checkpoint are all functioning correctly before any modeling work begins.

**PHASE 2 — MASt3R INFERENCE**
Load an uncalibrated image pair, run the pretrained *metric* MASt3R model on it, and inspect the raw outputs — their structure, shapes, and meaning — before doing anything else with them.

**PHASE 3 — POINTMAP EXTRACTION**
Extract the dense 3D point maps for both images from the inference output, and visualize the resulting 3D reconstruction / point cloud to sanity-check that the geometry looks plausible.

**PHASE 4 — CORRESPONDENCE MATCHING**
Use MASt3R's learned dense descriptors to establish pixel-to-pixel correspondences between the two images, and visualize the resulting matches.

**PHASE 5 — DISTANCE ESTIMATION**
Select corresponding physical points across the two images, retrieve their predicted 3D coordinates, and compute the Euclidean metric distance between them.

**PHASE 6 — GROUND-TRUTH VALIDATION**
Using image pairs where the true real-world distance is known, compare MASt3R's predicted distance against that ground truth.

**PHASE 7 — EVALUATION**
Compute MAE, RMSE, relative error, percentage error, and any other metrics useful for characterizing accuracy across the dataset.

**PHASE 8 — EXPERIMENTS / ABLATION**
Systematically investigate how the following factors affect metric-distance estimation accuracy:
- Camera/viewpoint difference between the two images
- The actual physical distance being measured
- Image resolution
- Descriptor matching quality
- Confidence filtering (with vs. without)
- Which points are selected for measurement
- Texture richness of the measured region
- Occlusion

---

## 7. Critical Technical Questions to Verify Before Finalizing

Before the implementation is considered trustworthy, you must explicitly investigate and answer each of the following — do not assume answers, verify them against the official model/checkpoint documentation and implementation:

1. What exactly does the MASt3R **metric** checkpoint output?
2. What coordinate system are the predicted point maps expressed in?
3. Are the coordinates truly metric-scaled (i.e., in real-world units), or only metric *up to* some caveat?
4. What does "metric" actually mean for this specific checkpoint?
5. What units should the 3D coordinates be interpreted in?
6. How are the two images' point maps related to / aligned with each other (shared coordinate frame? per-image frame with a transform?)?
7. How exactly are the dense descriptors extracted?
8. How exactly are dense correspondences computed from those descriptors?
9. How does MASt3R's confidence output work, mechanically?
10. Should low-confidence points be discarded, and if so, on what basis?
11. How do you convert a selected 2D pixel location into its predicted 3D coordinate?
12. Is directly computing Euclidean distance between two predicted 3D points actually valid for the measurement setup we're using — or are there hidden assumptions (e.g., both points must come from a jointly-aligned point map) that need to be satisfied first?
13. What assumptions is the overall pipeline making, explicitly listed?
14. What failure cases should be expected (e.g., low texture, large baseline, occlusion, poor overlap)?

**Do not assume that a 3D output automatically means "meters."** This must be verified from the official model/checkpoint documentation and implementation before any distance number is trusted.

---

## 8. Experiment Design

**Primary research question:**
> How accurately can MASt3R estimate real-world metric distances from uncalibrated image pairs?

**Secondary questions:**
- **A.** Does viewpoint change affect distance accuracy?
- **B.** Does image resolution affect accuracy?
- **C.** Does the actual physical distance affect relative error?
- **D.** Does descriptor confidence correlate with distance-estimation accuracy?
- **E.** Does filtering low-confidence correspondences improve results?
- **F.** How does texture-rich vs. texture-poor content affect correspondence and distance estimation?
- **G.** What types of scenes produce the largest errors?

**Planned experiments:**
1. Different camera/viewpoint baselines.
2. Different physical distances.
3. Different image resolutions.
4. With vs. without confidence filtering.
5. High-quality vs. low-quality correspondences.
6. Texture-rich vs. texture-poor objects.

---

## 9. Required Evaluation Outputs

The final implementation must produce all of the following:

1. Input image pair visualization.
2. MASt3R 3D point-map visualization.
3. Correspondence visualization.
4. Selected measurement points, shown clearly.
5. Predicted 3D coordinates for those points.
6. Ground-truth distance.
7. Predicted distance.
8. Absolute error.
9. Relative error.
10. Summary metrics table.
11. Plots, specifically:
    - Ground Truth Distance vs. Predicted Distance
    - Ground Truth Distance vs. Absolute Error
    - Ground Truth Distance vs. Relative Error
    - Viewpoint Difference vs. Error
    - Confidence vs. Error

---

## 10. Code Quality Requirements

- Write clean, modular Python — avoid dumping everything into one enormous script.
- Organize code into logical functions, such as:
  - `load_model()`
  - `load_images()`
  - `run_mast3r()`
  - `extract_pointmaps()`
  - `extract_descriptors()`
  - `find_correspondences()`
  - `filter_matches()`
  - `get_3d_point()`
  - `estimate_distance()`
  - `calculate_metrics()`
  - `visualize_results()`
- Comment code to explain **why** something is being done, not merely what the line does.
- Print tensor shapes and important intermediate values during development, so behavior is verifiable at each step.
- Add sanity checks throughout — don't let silent shape mismatches or unit confusion slip through.
- When a tensor has a shape like `[B, H, W, 3]`, explicitly explain what each dimension means (batch, height, width, channels — or whatever it actually is).
- Do not hide important transformations inside opaque helper calls without explanation.

---

## 11. Debugging Strategy

If something fails while you're running it, follow this sequence rather than rewriting broadly:

1. Explain what the error actually means.
2. Identify its true source (don't guess — trace it).
3. Inspect the installed MASt3R code/API directly to confirm expected behavior.
4. Fix the smallest possible component that addresses the root cause.
5. Re-run only the relevant phase to confirm the fix.
6. Do not rewrite the entire pipeline unnecessarily.

Additionally:
- Do not invent functions or classes that don't exist in the actual repository.
- Do not assume an API based on older MASt3R tutorials found online — verify against the current repository.

---

## 12. Project Structure

Prefer a structure similar to the following (`.py` scripts, since you'll be executing them yourself via the terminal rather than clicking through notebook cells):

```
mast3r_metric_distance/
│
├── phases/
│   ├── phase1_setup.py
│   ├── phase2_inference.py
│   ├── phase3_pointmaps.py
│   ├── phase4_matching.py
│   ├── phase5_distance.py
│   ├── phase6_ground_truth_validation.py
│   ├── phase7_evaluation.py
│   └── phase8_experiments.py
│
├── data/
│   ├── scene_01/
│   ├── scene_02/
│   └── ...
│
├── src/
│   ├── inference.py
│   ├── matching.py
│   ├── distance.py
│   ├── evaluation.py
│   └── visualization.py
│
└── results/
    ├── figures/
    └── metrics/
```

---

## 13. What Is Expected From You at Every Phase

Do not jump directly to a final implementation. We proceed **phase by phase**, and for **every** phase you must:

1. Explain the objective of the phase, in simple terms.
2. Explain the relevant technical concept behind it.
3. Explain what MASt3R is doing internally at that stage.
4. Implement it as real Python code in `phases/` and `src/`.
5. **Run it yourself** via the terminal in this Kaggle GPU environment.
6. Show me the actual output you obtained (not hypothetical/expected output).
7. Confirm the sanity checks pass against that real output — flag and fix anything that doesn't, per the debugging strategy.
8. **Stop, summarize what was done and verified, and explicitly ask me for the go-ahead before starting the next phase.**

---

## 14. Presentation Requirement: Dual Explanation Convention

I will eventually need to explain this project to a professor. Therefore, **every time something is implemented, also provide a dual explanation**: one simple, one technical. Maintain this convention throughout the entire project.

Example:

> **Simple explanation:** "MASt3R gives every pixel a 3D location."
>
> **Technical explanation:** "MASt3R predicts dense point maps in a learned 3D representation and produces dense descriptors that enable cross-view correspondence."

---

## 15. Final Project Narrative

The completed project should be explainable as follows:

**Problem:** Traditional metric distance estimation from images generally depends on camera calibration, explicit geometry, or specialized sensors.

**Approach:** Use MASt3R's learned 3D representation and dense correspondence capability on uncalibrated image pairs.

**Method:**
1. Feed the image pair to MASt3R.
2. Obtain dense 3D point maps.
3. Obtain dense learned descriptors.
4. Establish cross-view correspondences.
5. Select corresponding physical points.
6. Retrieve their 3D coordinates.
7. Compute Euclidean distance.
8. Compare against real-world ground truth.
9. Analyze accuracy and failure cases.

**Research question:**
> How accurately can a pretrained metric MASt3R model estimate real-world distances from uncalibrated image pairs, and what factors influence this accuracy?

---

## 16. First Task — Start With Phase 1 Only

Do **not** implement the entire project immediately.

Begin **only** with Phase 1 (Setup). Specifically:

1. Inspect the current official MASt3R repository and determine:
   - Required Python version
   - Installation commands
   - Dependencies
   - The correct metric checkpoint to use
   - GPU requirements
   - The correct model-loading API
   - The correct inference API
   - The expected output structure
2. Implement and **run** Phase 1 setup directly in this environment — install everything, and verify GPU/CUDA/repo/checkpoint are all working.
3. Once verified, **stop and explicitly ask me for a go-ahead** before starting Phase 2.

---

## 17. Hard Constraints — Never Violate These

- Do **not** hallucinate MASt3R APIs.
- Do **not** silently substitute DUSt3R for MASt3R.
- Do **not** introduce MASt3R-SfM unless explicitly necessary and discussed first.
- Do **not** assume predicted coordinates are in meters without verifying against the metric checkpoint's actual documentation/implementation.
- Do **not** optimize prematurely.
- Do **not** start a new phase without my explicit go-ahead, even if the previous phase clearly succeeded.
- Build the project incrementally and scientifically, one verified phase at a time.
