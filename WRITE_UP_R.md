# Metric distance from two photos with MASt3R: finding the scale error and fixing it with one known length

**Full account of our part of the project (`main.ipynb`, Phases 1-8, plus the phone-photo check).**

This document explains everything we did, in order, with the reasons for each decision and every result that was measured. It is
written for a reader (a person or an AI assistant building slides) who has not seen the project. Technical terms are explained the
first time they appear. Numbers are copied from notebook outputs from the Kaggle T4 runs; where a number comes from somewhere else
(a local run, a plausibility argument, a rough estimate) it says so. It is the counterpart of `WRITE_UP.md`, which covers the teammate's
fine-tuning work (notebooks 01-04).

**How to use this for slides.** Section 0 gives a one-minute story, the key numbers and a suggested slide plan. Every later section starts
with a **Takeaway** line (one slide title or message) and the figures or tables worth showing. Figures are in `results/figures/`.

---

## 0. The one-minute version

**Question.** How accurately can the pretrained *metric* MASt3R model measure real-world distances from ordinary photos, and what limits it?

**What we did.**
1. Built the measuring pipeline in one notebook, phase by phase (Phases 1-5).
2. Tested it against real distances computed from indoor RGB-D data (7-Scenes) and scored it (Phases 6-7).
3. Found that the model's overall **scale drifts** between photos, then tried two ideas to fix it (Phase 8):
   - an **ensemble** of 8 readings (no extra information): it did **not** help;
   - a **known-length scale anchor** (the user marks one or more things of known real length): it **did**.
4. Tested the frozen method once on three scenes never used for any decision, then on real phone photos with a tape measure.

**What we found.**
- Without a reference, the model's distances are typically about 6 cm off (mean about 11 cm) on indoor gaps under 1 m.
- The main systematic problem is the overall size of the scene: it drifts between photo pairs (median ratio 0.89 to 1.36 across scenes).
- One known length in the photos removes most of that: typical error 5.8 cm to 1.3 cm, mean 10.8 cm to 4.9 cm on held-out scenes.
- On two phone scenes, every tape-measured segment ended up within 5 cm with the anchor.

**What it is not.** A known length is required. The evidence is one dataset family, short gaps, easy (smooth-region) pairs and two near-flat phone scenes.

### Key numbers (held-out 7-Scenes: redkitchen, pumpkin, fire; 110 pairs, errors in cm)

| Method | Median | Mean (MAE) | Within 5 cm | Within 10 cm |
|---|---|---|---|---|
| Baseline (no known length) | 5.8 | 10.8 | 44% | 69% |
| 8-vote ensemble (no known length) | 6.8 | 10.7 | 45% | 70% |
| One anchor chosen by length | 1.8 | 5.7 | 83% | 94% |
| **Final: 3 depth-matched anchors** | **1.3** [0.6, 2.0] | **4.9** [2.0, 8.4] | **88%** | **94%** |
| Final, on the 86% of pairs where anchors agree | 1.0 [0.4, 1.5] | 3.7 [1.5, 6.6] | 94% | 96% |

(Ranges in brackets are 95% intervals from resampling whole photo pairs.)

### Suggested slide plan

| # | Slide | From section | Visual |
|---|---|---|---|
| 1 | The question and the application idea | 1 | one sentence + a sketch of "two photos + a known length" |
| 2 | The pipeline | 3 | flow diagram: photos, MASt3R, 3D points, distance |
| 3 | How we built and ran it | 2 | the 8-phase table |
| 4 | Test data: where true distances come from | 4 | formula + a table of the 5 scenes |
| 5 | How we kept the test fair | 5 | dev / held-out split, pre-registered criteria |
| 6 | Step 1: how good is it as it stands? | 6 | scale-ratio table + `results/figures/phase7_error_plots.png` |
| 7 | Step 2: where does the error come from? | 7 | "scale drifts; averaging does not help" |
| 8 | The fix: a known-length scale anchor | 8 | formula + the 4 design choices |
| 9 | Held-out result | 9 | key-numbers table + `results/figures/phase8_heldout_results.png` |
| 10 | What drives the remaining error | 9 | binned tables |
| 11 | Real phone photos | 10 | your own photos of the two scenes + the result tables |
| 12 | Measuring a real object: the A4 sheet | 10 | the 12-measurement table |
| 13 | Compared with the teammate's fine-tuning | 11 | the ablation table |
| 14 | Honest limits | 13 | bullet list |
| 15 | What next | 14 | bullet list |

---

## 1. The task

**Takeaway:** measure the real distance between two points in a photo, with the model's scale pinned by a known length.

- **Input:** two photos of the same static scene, two clicked points in photo 1, and one or more *known lengths* (an A4 sheet's long edge is 29.7 cm, or any tape-measured gap) marked in the same photo.
- **Output:** the real-world distance in metres between the two points.
- **Core model:** MASt3R (github.com/naver/mast3r), pinned to commit `f5209af`, with the public checkpoint `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` (2.75 GB, 688.6 million parameters). **The model is never retrained or edited.**
- **Research question:** how accurately can the pretrained metric model estimate real distances from uncalibrated image pairs (no camera calibration), and what affects the accuracy?
- **Application that motivated it:** a "does this object fit here?" check from two phone photos, in the spirit of retail AR previews but without AR hardware. That is a motivation, not a result: a fit check needs roughly 2-5 cm on 1-3 m gaps, and gaps above 1 m are untested.
- **Constraints:** one notebook (`main.ipynb`) that holds all phases; static figures only; official MASt3R functions (`load_images`, `inference`, `fast_reciprocal_NNs`); the owner runs each phase and reports before the next starts.

---

## 2. How we built and ran it

**Takeaway:** an 8-phase notebook, each phase run and checked before the next.

- **Where things ran:** Kaggle (Tesla T4, 14.6 GB; last run Python 3.13.15, torch 2.11.0+cu128) and Google Colab. The setup cells at the top of the notebook handle either. The 2.75 GB checkpoint is re-downloaded every session. A local machine without a GPU was used only for smoke tests on CPU (the same code, the real model).
- **Repository:** one GitHub repo. Two people work in it, with a "claim a phase before editing the notebook" protocol recorded in `CLAUDE.md`, and a work log. Notebook outputs are not committed (10 MB+).
- **Working rules:** assertions on coordinate order and shapes; every new function is smoke-tested before it is presented; no number is called "accurate" just because it is self-consistent.

### The eight phases

| Phase | What it does | Key result |
|---|---|---|
| 1. Setup and verification | Loads the model; checks the checkpoint byte-for-byte and its architecture against the README's metric recipe | 688.6M parameters; the "metric" behaviour comes from the training loss, so it is a learned prior, not a measurement |
| 2. Inference | One forward pass on a photo pair (about 0.7 s on the T4) | Both 3D point maps are in image 1's camera frame; scale probe: the output carries an absolute scale, whose correctness is unknown |
| 3. Point maps | Extracts the per-pixel 3D points and confidences; checks the geometry | 384x512 point map; implied focal length about 930 px; all geometry checks passed |
| 4. Correspondences | Matches pixels across the two photos (mutual nearest neighbours on the 24-number descriptors) | 1,009 matches; the model's two heads disagree about one physical point by a median of 0.8% of scene size, with a heavy tail (90th percentile 22.7%) |
| 5. Distance estimation | `D = ||P_a - P_b||`, with an error floor and a refusal policy | On two outdoor demo scenes the model put buildings only 2-4 units away: a plausibility red flag for scale |
| 6. Ground-truth validation | Compares against true distances from 7-Scenes | The scale ratio `D_est / D_gt` differs between scenes: 0.89, 0.93, 1.13, 1.36 |
| 7. Evaluation metrics | `calculate_metrics()`: MAE, RMSE, relative error, bias, scale ratio, bootstrap intervals | Median error 6.2 cm, mean 13.0 cm, median relative error 12.5% (40 pairs) |
| 8. Improvement and test | Ensemble (not adopted), scale anchor (adopted), held-out test, factor analysis | Held-out median 5.8 cm to 1.3 cm; mean 10.8 cm to 4.9 cm |

---

## 3. How a distance is computed

**Takeaway:** the distance is a subtraction of two 3D points; the model's job is to put the points in the right place.

1. **Preprocessing:** an exact copy of MASt3R's loader (`load_images(size=512)`): resize so the long side is 512, then centre-crop to multiples of 16. A 640x480 photo becomes 512x384 with no crop; a portrait 575x1280 photo becomes 224x512 (3 columns cropped on each side).
2. **Forward pass:** MASt3R takes **two** images. Our method runs it on (photo 1, photo 2) and reads the 3D point of every pixel of photo 1 (`pred1['pts3d']`, metres in photo 1's camera frame, indexed `[y, x]`).
3. **Clicked pixels:** points are marked on the *original* photo in `(x, y)` and mapped to the model grid by `original_to_model_xy`, which re-derives the loader's resize and crop. It was checked against the real loader for six photo sizes.
4. **Distance:** the straight-line distance between the 3D points at the two pixels. Only photo 1's pixels are needed; photo 2 provides the geometry. (A test showed that using image-1 pixels alone is as good as also using matched pixels in image 2.)
5. **Anchor correction** (the new part, section 8): multiply by a scale taken from known lengths.

**Difference from the teammate's setup:** their main study feeds the *same photo twice* (single-image mode). Ours uses two different photos. That is why the two approaches are not directly comparable (section 11).

---

## 4. Data and ground truth

**Takeaway:** we never measured objects in 7-Scenes. True lengths are computed from depth.

**Dataset: 7-Scenes** (indoor RGB-D video of small rooms, Kinect depth camera, with a camera pose per frame). Five scenes, split as development and held-out:

| Scene | Role | Frame pairs used (baseline between the two cameras) |
|---|---|---|
| chess | development | 0.10 / 0.41 / 0.54 / 0.82 m |
| office | development | 0.15 / 0.32 / 0.84 / 0.63 m (one pair with 42 degrees of rotation) |
| redkitchen | **held out** | 0.05 / 0.35 / 0.40 / 0.52 m |
| pumpkin | **held out** | 0.05 / 0.34 / 0.61 / 0.66 m |
| fire | **held out** | 0.21 / 0.36 / 0.81 m (one pair, frames 100-160, had no usable points and was skipped) |

Frames were downloaded by **streaming**: the server supports range requests, so about 100 MB per scene was read instead of the 3 GB zip. Frame pairs are the same four for every scene (frames 0-30, 40-80, 100-160, 0-100).

**How a true length is computed** (this is what "ground truth" means here):
1. Pick two pixels in photo 1 that have valid depth, are at least 40 px apart, and lie in smooth regions.
2. Turn each into a 3D point with its Kinect depth `z` and the camera constants (focal length 585 px, centre 320, 240): `X = (x - 320) * z / 585`, `Y = (y - 240) * z / 585`, `Z = z`.
3. The true length is the straight-line distance between the two points, kept only if it is between 0.10 and 1.00 m.
4. Poses are only used to find the same points in photo 2, with an occlusion check.

**Verified on the real files (Phase 6 and again for every new scene in Phase 8):** the pose files are camera-to-world (depth maps agree 84-93% when read that way against 0.4-35% when inverted), the focal length 585 px fits (the sweep peaks at 585-600), and the depth is 16-bit millimetres with 65535 meaning invalid (about 21% of pixels).

**Not verified: whether depth and colour images line up pixel for pixel.** A test for it was inconclusive. So the ground-truth pairs were chosen to survive an **8-pixel depth shift**: a point is kept only if shifting the depth lookup by 8 px moves its image-2 location by at most 3 px, and a pair only if its length changes by at most 2 cm. This selects smooth regions, so the pairs are easier than average (section 13).

**Other ground truth:** the lengths on real phone photos (section 10) are tape-measured.

---

## 5. How we kept the test fair

**Takeaway:** settings were frozen on two scenes, then tested once on three others, against criteria written beforehand.

- **Development scenes (chess, office; 80 pairs)** were used to choose settings. **Held-out scenes (redkitchen, pumpkin, fire; 110 pairs)** were not built, run or looked at until the single test run in section 9.
- **Success criteria fixed in advance** (from the 3-5 cm goal): median absolute error at most 5 cm **and** mean absolute error at most 5 cm on the held-out scenes, with the share of pairs answered stated.
- **Coverage is always reported** next to any error computed after refusing hard pairs, because refusing hard pairs lowers error by construction. The "all pairs" row is the one that cannot be gamed.
- **Uncertainty:** 95% intervals from a **cluster bootstrap**: resample whole photo pairs (11 held-out photo pairs), not individual pairs, because pairs inside one photo pair share a viewpoint and share errors.
- **Leakage guards:** a target is never its own anchor, and no anchor may share an endpoint with its target (a duplicate would hand over the true answer; this guard was added after a dry run exposed the hazard).
- **Counting alternatives:** every option examined on the 80 development pairs is listed in the notebook (2 estimator families, 5 patch radii, 6 vote subsets, 3 anchor counts, 2 anchor modes, 3 refusal signals), because each is a chance to over-fit.

---

## 6. Step 1: how good is the model as it stands?

**Takeaway:** about 6 cm typical error, but the overall scale is not stable.

(Phases 6-7, chess scene, 40 ground-truth pairs; before the improvement work.)

**Scale ratio `D_est / D_gt` per scene** (1.0 would mean the model's units are exactly metres; below 1 means too small):

| Frame pair | Camera baseline | Median ratio |
|---|---|---|
| scene 1 | 0.10 m | 0.89 |
| scene 2 | 0.41 m | 0.93 |
| scene 3 | 0.54 m | 1.13 |
| scene 4 | 0.82 m | 1.36 |

The pooled median (1.03) hides opposite biases. The notebook's verdict, using stated thresholds: **no single consistent scale**.

**Error metrics (all 40 pairs, truth in metres, estimate in model units):** mean absolute error 13.0 cm; RMSE 22.3 cm; median 6.2 cm; 90th percentile 37.0 cm; median relative error 12.5% (95% interval 11.0-14.3%); within each scene the estimates track the truth (Pearson r 0.99, 0.97, 0.68, 0.78).

**The refusal policy from Phase 5** (answer only if both endpoints in both views are at or above the median confidence) accepted just **4 of 40** pairs: random points pass that rule about 1 time in 16. So the policy, not the point maps, explains the high refusal rate. The scale test therefore used every pair's unfiltered value, with the accepted subset shown beside it. The five largest errors were all pairs the policy had refused.

Figures: `results/figures/phase6_gt_vs_estimate.png`, `results/figures/phase7_error_plots.png`.

---

## 7. Step 2: where does the error come from, and what does not help?

**Takeaway:** the error is systematic (a drifting scale), so averaging more readings does not fix it.

**Idea 1, a training-free ensemble.** Four forward passes per photo pair (plain and mirrored images, each in both orders), eight distance readings (from image 1's or image 2's points), optionally a patch median instead of one pixel, fused by the median. A distance does not change when the frame is rotated, shifted or mirrored, so the readings need no alignment. The bookkeeping (which pixel to read in a mirrored or swapped run) is tested against a synthetic scene with known answers, including a deliberately wrong mapping that must be detected.

**Rule fixed before the result was seen:** adopt the ensemble only if it lowers the development mean error by at least 5% and does not raise the median.

**Result on development scenes: not adopted.**

| | Mean error | Median error |
|---|---|---|
| Plain single-pixel estimator | 11.7 cm | 7.4 cm |
| Best ensemble (8 votes, 11x11 patch) | 11.5 cm | 9.0 cm |

- Patch medians from 3x3 to 11x11 gave no gain.
- The spread between votes has **zero** rank correlation with the error (rho = 0.00): all votes come from the same frozen model, so they share its bias. Averaging removes noise, not a shared mistake.
- Using image-1 pixels only (no matcher at all) is as good as using matched pixels in image 2 (pooled mean error 11.4 cm against 11.2 cm).

**What this tells us:** the dominant error is the scale, which is a property of the model's output for a given photo pair, not noise that more reads can average away. Another independent analysis (the teammate's, on NYU and DIODE) reached the same conclusion about local smoothing.

---

## 8. The fix: a known-length scale anchor

**Takeaway:** one known length in the photos fixes most of the scale error; choosing anchors at the target's depth and checking that they agree makes it robust.

**Idea.** The user marks things of known real length in the photos: an A4 sheet's long edge (29.7 cm), a tape-measured gap. For each, the model's own length for that segment gives a scale:

`scale = true_length / model_length`, and then `distance = scale * model_distance`.

**The four design choices** (all fixed on the development scenes):

| Choice | Setting | Why |
|---|---|---|
| How many anchors | 3 (scale = **median** of their scales) | the median is a real anchor's scale and resists one bad anchor |
| Which anchors | the ones **nearest in depth** to the target (depth = mean Z of the endpoints, from the model's own point map) | the scale is not uniform inside a photo pair; matching depth lowered the dev mean error from 5.6 cm (length-matched) to 4.6 cm |
| Minimum anchor length | 25 cm | a short anchor amplifies its own error |
| Trust signal | **anchor disagreement**: `(max - min) / median` of the 3 scales, answered if at most 0.157 (the development 70th percentile) | when anchors disagree, the scale is not uniform here, so the answer is not trusted; it correlated with the error far better than vote spread (rho +0.59 against +0.17) |

**What the experiment simulates, stated plainly.** In a real app the user would mark a printed A4 sheet, a credit card or a door. In the 7-Scenes experiment the "known length" is another ground-truth pair of the same photos, which carries Kinect noise (about 1-2 cm). A printed sheet is exact, so the experiment is, if anything, pessimistic about the anchor.

**Development results (80 pairs, cm; mean / median):**

| | Mean / median |
|---|---|
| No anchor | 11.7 / 7.4 |
| Anchors by length, 1 / 3 / 5 | 7.4 / 2.2, 5.6 / 1.4, 5.8 / 1.3 |
| Anchors by depth, 1 / 3 / 5 | 4.3 / 1.3, **4.6 / 1.2**, 4.5 / 1.1 |
| Oracle: one perfect scale per photo pair (uses the answer) | 4.9 / 0.9 |

**Frozen configuration:** plain estimator, k = 3, depth-matched anchors, minimum length 25 cm, refusal threshold 0.157.

---

## 9. Result: the held-out test

**Takeaway:** typical error 5.8 cm to 1.3 cm; mean 10.8 cm to 4.9 cm; criteria met, the mean only just.

Run once, on redkitchen, pumpkin and fire (110 pairs from 11 photo pairs), with the configuration frozen above. The table is in section 0. Figure: `results/figures/phase8_heldout_results.png` (error distribution and risk-versus-coverage curve).

**Against the pre-registered criteria**

| | Median at most 5 cm | Mean at most 5 cm |
|---|---|---|
| All pairs | **met** (1.3 cm) | **met, only just** (4.9 cm; 95% interval up to 8.4 cm) |
| The 86% answered | met (1.0 cm) | met (3.7 cm) |

**Per scene (median / mean, cm):**

| Scene | Baseline | Final |
|---|---|---|
| redkitchen | 7.6 / 13.5 | 1.5 / 5.9 |
| pumpkin | 3.0 / 9.5 | 1.3 / 6.5 |
| fire | 6.7 / 8.9 | 1.0 / 1.5 |

Two of three scenes are above 5 cm mean error. About 3% of pairs are off by more than 40 cm; they dominate the mean, which is why the median is the sturdier number.

### What drives the remaining error (development and held-out pooled, 190 pairs; median / mean, cm)

| Factor | Bin | Baseline | Final |
|---|---|---|---|
| Camera baseline | under 0.2 m | 9.0 / 11.2 | 1.3 / 3.3 |
| | 0.2 to 0.5 m | 7.2 / 12.1 | 1.6 / 5.8 |
| | 0.5 m and up | 4.4 / 10.4 | 0.6 / 4.7 |
| Rotation between photos | under 8 degrees | 7.7 / 8.4 | 0.9 / 2.0 |
| | 8 to 15 degrees | 7.2 / 13.0 | 2.4 / 7.0 |
| | 15 degrees and up | 5.2 / 12.1 | 0.9 / 5.5 |
| True length | under 0.4 m | 3.9 / 10.1 | 1.1 / 5.5 |
| | 0.4 to 0.7 m | 8.1 / 9.9 | 0.8 / 3.6 |
| | 0.7 m and up | 10.9 / 14.3 | 2.2 / 5.3 |
| Depth from camera | under 2 m | 7.2 / 11.1 | 1.1 / 4.5 |
| | 2 to 3 m | 6.7 / 10.1 | 1.0 / 4.8 |
| | 3 m and up | 5.7 / 13.9 | 1.6 / 5.5 |

No factor shows a clean trend. The bins are small, so these are descriptions, not findings. **Confidence filtering:** the Phase 5 rule answers only 25% of pairs; at equal 70% coverage the anchor-agreement rule and a plain confidence rule give similar errors (mean 3.0 cm against 2.9 cm).

---

## 10. Real phone photos

**Takeaway:** on two real scenes, with a tape measure, the anchor brought every segment within 5 cm; and a real object (the A4 sheet) was measured to within about 1 cm.

**Setup.** Two scenes photographed on a phone, two photos each, the second after a sideways step of 30-40 cm:
- **Scene 1, tile floor:** coloured pins as markers, an A4 sheet, three tape-measured segments (35, 50 and 80 cm).
- **Scene 2, mattress:** sticky notes as markers, an A4 sheet, Rubik's cubes as clutter, three tape-measured segments (40, 60 and 45 cm). The tape was measured between the *facing edges* of each pair of notes (inferred from the data; the owner is asked to confirm).

The points were located automatically with a colour detector (pins, notes and the A4 corners), and each overlay was checked by eye. The two photos agree through a plane-to-plane mapping to about 1 px. The photos used were chat-resized to 1280 px, not the originals.

**Result 1: leave-one-out anchoring** (each segment predicted from the other three; each scene run with either photo as image 1; mean error in cm, raw model against anchored):

| Scene | Image 1 | Raw model | Anchored | Segments within 5 cm |
|---|---|---|---|---|
| Tile, pins | photo 1 | 4.4 | 1.2 | 4 of 4 |
| Tile, pins | photo 2 | 11.6 | 0.4 | 4 of 4 |
| Mattress (facing edges) | photo 1 | 11.8 | 0.2 | 4 of 4 |
| Mattress (facing edges) | photo 2 | 9.5 | 0.6 | 4 of 4 |
| Mattress (outer edges) | photo 1 | 14.7 | 0.9 | 4 of 4 |
| Mattress (outer edges) | photo 2 | 12.1 | 2.0 | 4 of 4 |

**Result 2: measuring a real object.** The A4 sheet has an exactly known size, 29.7 x 21.0 cm. It was **not** used as an anchor here: the scale came from the three tape-measured segments only, then the sheet's two edges were measured.

| Scene and photo order | Scale from tape | Long edge (true 29.7) | Short edge (true 21.0) |
|---|---|---|---|
| Tile, photo 1 | x0.91 | 29.7 (+0.0) | 21.3 (+0.3) |
| Tile, photo 2 | x0.80 | 29.6 (-0.1) | 21.4 (+0.4) |
| Mattress (facing), photo 1 | x0.78 | 29.1 (-0.6) | 20.1 (-0.9) |
| Mattress (facing), photo 2 | x0.82 | 28.8 (-0.9) | 20.2 (-0.8) |

Over all 12 measurements (including the outer-edge reading): the raw model was off by a median 5.1 cm (too long); after the tape-set scale, the median error was **0.8 cm**, the worst 2.7 cm. The facing-edge reading of the mattress notes fits the A4 sheet within about 1 cm, while the outer-edge reading is off by 2-2.7 cm: independent support for the facing-edge convention.

**What the model alone does:** it gets the shape roughly right but the size is biased (3-7 cm too long on the sheet). The known measurement is what turns it into an accurate metric reading.

**Caveats (important).**
- Two scenes, 8 segments: a preliminary check, not statistics.
- Both scenes are nearly flat and seen from above, where scale is almost uniform. A plain ruler-from-the-A4 calculation gets within about 2-6 cm on the same photos, so this is an easy case. It does not yet test taller objects or oblique views.
- Small objects are hard: below about 10 cm one pixel at the model's resolution is about 0.5 cm.
- Chat-resized copies; the mattress photo pair had different heights (1280x537 and 1280x575), so one was probably cropped.

---

## 11. Compared with the teammate's fine-tuning

**Takeaway:** two independent routes to the same problem; they agree on the diagnosis; they are complementary and the combination is untested.

Full tables and caveats are in `ABLATION_COMPARISON.md`. In short:

| | Teammate | Ours |
|---|---|---|
| Idea | fine-tune the 3D-point head (20.2M of 688.6M parameters) | keep the model frozen, add a known length |
| Data | NYU, DIODE, TUM | 7-Scenes, phone photos |
| Main result | NYU typical error 8.5% to 7.3%; no gain on DIODE; none outdoors | held-out typical error 5.8 cm to 1.3 cm |
| Extra experiment | a second photo helps (about 20% to 15%); adding the known camera movement halves it (about 9-10%) | the anchor is the "known length" version of the same idea |

**What they agree on:** unanchored indoor error is about 8-20% depending on the data; local smoothing and extra readings do not help (the error is systematic); the scale is badly wrong outside indoor training scenes (their outdoor scenes are about 3.8x too small; our outdoor demo scenes looked implausible too); and **one real-world length, in whatever form, removes most of the scale error.**

**Not directly comparable:** different data, relative against absolute error, random pairs against smooth-region pairs, one photo against two.

**The experiment that would tie them together (not run):** a 2x2 grid (original or fine-tuned head, with or without the anchor) on identical clicks. `scale.md` specifies how to run it in his Gradio demo, which already has a simple one-reference mode that ours extends with several anchors, depth matching and an agreement check.

---

## 12. Problems hit, and how they were handled

1. **The first refusal policy was too strict:** it accepted 4 of 40 random pairs. We kept every pair's unfiltered value for the scale test, reported both sets side by side, and chose a different trust signal (anchor agreement) in Phase 8.
2. **The ensemble did not help.** We kept it in the tables as a negative result, with the rule that decided it written beforehand.
3. **A leakage hazard in the anchors.** A dry run on duplicated records showed that a pair can be its own anchor if records repeat, and two pairs from one photo pair can share an endpoint. We added an endpoint-sharing guard, tested it, and rebuilt the development records.
4. **Unverified ground-truth alignment.** We could not confirm depth-to-colour registration, so the ground-truth pairs were made tolerant to an 8 px shift instead.
5. **Our own misreadings, corrected along the way:** the first phone photo looked rotated 90 degrees but was tilted about 20-25 degrees; the early marker advice (A to B, B to C, C to D) shared endpoints, which the code correctly excludes; and an overstated "sheet within 5 cm" reading was replaced by an A4-only strict test.
6. **A working folder was cleaned up mid-session,** so the official held-out run was executed again from the notebook's own cells (the Kaggle run is the one that counts, and its development numbers were identical to the local CPU run).

---

## 13. Honest limits

- **A known length is required.** Without one the error is about 11 cm (mean) and 6 cm (median).
- **One dataset family.** 7-Scenes is Kinect data of small rooms. Phones, other rooms, lighting and texture are only touched by two scenes.
- **Short gaps only.** Tested lengths were 0.1-0.95 m; a fit check needs 1-3 m.
- **Easier-than-average pairs.** Smooth-region, depth-shift-robust pairs; a person's clicks are harder.
- **The ground truth is not exact.** Kinect noise (about 1-2 cm, not subtracted) and unverified depth-to-colour alignment sit under every number.
- **The simulated anchor** is another ground-truth pair, not a printed sheet.
- **The mean is borderline.** 4.9 cm sits right at the 5 cm target and its 95% interval reaches 8.4 cm. Two of three held-out scenes were above 5 cm. The median (1.3 cm) is the solid number.
- **Phone check is small and easy:** 2 scenes, 8 segments, nearly flat, top-down, chat-resized photos.
- **Training-data overlap cannot be ruled out.** The MASt3R metric model lists an unreleased internal dataset; we only know 7-Scenes is not in the published list.
- **Untested factors from the brief:** image resolution, texture richness, occlusion.
- **Novelty is not claimed.** Room-measuring and furniture-fit apps exist; no literature search was done.

---

## 14. What we would do next

1. **More phone scenes** shot at an angle, with taller objects, gaps of 1-3 m and printed A4 anchors, using the original full-size photos.
2. **The 2x2** (fine-tuned head x anchor) once the teammate's weights and his two-photo check are available (`scale.md` has the spec).
3. **A safety check** for the anchor: a minimum number of anchors, and a plausibility range for the scale.
4. **The fit-check feature:** two points plus an object's size gives fits / does not fit / uncertain, using anchor disagreement as the margin.
5. **Untested factors:** image resolution, texture, occlusion, a repeat of the evaluation on other datasets.

---

## 15. Where everything is

**In the repository:**
- `main.ipynb`: the work. Cells 0-95: setup and Phases 1-7 (Phase 1-4 cell numbers are +2 from the older notes); cells 96-111: Phase 8. Find cells by content, not index.
- `CLAUDE.md`: project rules, phase status, two-person protocol, work log.
- `context.md`: running notes; section 14 has the results and the open items.
- `ABLATION_COMPARISON.md`: comparison with the teammate's work.
- `scale.md`: integration spec and tested code for putting the anchor in the Gradio demo.
- `WRITE_UP_R.md`: this document.
- `results/figures/`: `phase5_points_chateau.png`, `phase5_points_nle_tower.png`, `phase6_gt_vs_estimate.png`, `phase7_error_plots.png`, `phase8_heldout_results.png`.

**Not in git:** `data/` (the streamed 7-Scenes frames, the scene folders and the phone labels at `data/phone/scene_02/labels_draft.json`), the checkpoint, the MASt3R clone, and the per-run outputs saved on Kaggle (`results/metrics/*.json`, `phase8_records_*.json`).

---

## 16. Questions we may be asked

**Why a known length? Isn't the model supposed to be metric?** It is only approximately metric: its overall scale drifts between photo pairs (0.89 to 1.36 median ratio across our scenes, 3.8x too small on outdoor scenes in the teammate's data). One known length pins the scale for that photo pair.

**Why didn't the ensemble help?** All the readings come from one frozen model, so they share its mistake. Averaging removes noise, not a consistent bias.

**How do you know the test was fair?** Settings were frozen on two scenes; three other scenes were run once, against criteria written before the run; intervals come from resampling whole photo pairs; anchors can never be the target or share a point with it.

**Is 4.9 cm accurate?** Do not say so. It is the mean on held-out scenes, at the 5 cm line, with an interval up to 8.4 cm and two scenes above 5 cm. The typical (median) error is 1.3 cm.

**Is the ground truth exact?** No. It is computed from Kinect depth (about 1-2 cm noise), with depth-to-colour alignment unverified. The phone-scene lengths are tape-measured.

**Why did you not retrain the model?** Ten or so labelled distances cannot teach a 688M-parameter model; the anchor is a one-parameter correction that needs no training. The teammate's fine-tune used hundreds of photos with dense depth labels, and it did not transfer to another dataset.

**Would it work on my phone?** Two phone scenes say yes, in the easy case. Taller objects, oblique views and 1-3 m gaps are untested.

**What if there is no known object?** Then the method does nothing, and the raw model error (about 6 cm typical) stands. A known camera movement (the teammate's experiment) is the other way to supply scale.

**Is this novel?** We do not claim so. Room-measuring and furniture-fit apps exist, and we have not done a literature search.

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **MASt3R** | The AI model that guesses a 3D point for every pixel of a photo. |
| **Metric** | In real units (metres), not just relative shape. |
| **Point map** | The array of those 3D points, one per pixel. |
| **Ground truth** | The true answer, from a sensor or a tape measure. |
| **Scale** | The overall size of a reconstructed scene (dollhouse versus real house). |
| **Scale ratio** | Estimated distance divided by true distance: 1.0 is perfect, below 1 is too small. |
| **Anchor** | A segment of known real length marked in the photos, used to set the scale. |
| **Depth-matched anchors** | The anchors whose distance from the camera is closest to the target's. |
| **Anchor disagreement** | How far apart the anchors' scales are; used to decide whether to trust an answer. |
| **Baseline (camera)** | How far the camera moved between the two photos. |
| **Held-out** | Kept aside for testing; never used for any choice. |
| **Development scenes** | The scenes used to choose settings. |
| **Pre-registered criteria** | Success thresholds written down before running the test. |
| **MAE** | Mean absolute error: the average size of the errors. |
| **Median error** | The middle error when sorted; not thrown off by a few disasters. |
| **Coverage** | The share of pairs the method answers (it can refuse). |
| **Cluster bootstrap** | Re-scoring many times on random re-selections of whole photo pairs, to see how much a result could vary. |
| **Oracle** | A method that uses the true answer; a diagnostic, not a usable method. |
| **Kinect** | A camera that also measures depth with infrared light. |
| **7-Scenes** | The public indoor RGB-D dataset we used. |
| **Leave-one-out** | Predicting each segment from all the others. |
| **Frozen** | Not changed: the model weights, and the settings after the development phase. |
