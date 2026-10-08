# Ablation and comparison: ways of improving MASt3R's metric distances

Two independent analyses of the same pretrained model, compared side by side. **No new experiments were run for
this document.** Our numbers come from `main.ipynb` (Phases 6-8, run on a Kaggle T4) and the phone-photo check.
The teammate's numbers are taken from their written summary (including its two-photo / TUM section) and have **not**
been re-run or independently verified.

- **Teammate:** fine-tune MASt3R's 3D-point head, and test whether a second photo and a known camera movement help.
- **Ours:** keep the model frozen, use two photos, and fix the scale with a known length placed in the scene.

## 1. The approaches

| | Teammate | Ours |
|---|---|---|
| Input | 1 photo paired with itself (main study); 2 photos in the extra experiment; 2 clicks | 2 photos, 2 clicks in photo 1 |
| Extra scale information | none, or the known camera movement (extra experiment) | at least 1 known length in the scene (A4 sheet, tape) |
| Model change | DPT head fine-tuned (20.2M of 688.6M parameters), rest frozen | none, model frozen |
| Data | NYU Depth v2 (547 test images), DIODE (771), TUM RGB-D (4 scenes, 161 photos) | 7-Scenes (5 scenes: 2 for choosing settings, 3 held out), 2 phone scenes |
| Ground truth | Kinect depth (NYU, TUM), laser scanner (DIODE), tracked camera poses (TUM) | Kinect depth + camera poses (7-Scenes), tape measure (phone) |
| Pairs | 300 random per image, at least 0.25 m and 16 px apart | 10 per photo pair, 0.1-0.95 m, chosen in smooth regions |
| Metric | relative error | absolute error (cm) |
| Held-out rule | split by scene | split by scene; settings fixed before the held-out run |
| Uncertainty | resampling whole scenes | resampling whole photo pairs |

## 2. Headline results

| | Before | After | Change |
|---|---|---|---|
| Teammate, NYU test (fine-tuned head) | median 8.5%, 56% within 10% | 7.3%, 63% | real gain (interval excludes zero) |
| Teammate, DIODE indoor (fine-tuned head) | median 29.8%, 12% within 10% | 30.1%, 10% | no gain |
| Teammate, DIODE outdoor (fine-tuned head) | median 75.7% | 75.5% | no gain |
| Teammate, TUM: second photo | one photo 19.9%, 30% within 10% | two photos, moved 10 / 25 / 50 cm: 15.4 / 14.9 / 13.3%, 35 / 38 / 40% | small, reliable gain |
| Teammate, TUM: second photo + known movement | same | moved 10 / 25 / 50 cm: 16.2 / **10.2** / **8.8**%, 33 / **49** / **55**% | large gain from 25 cm, but mean error worsens |
| Teammate, TUM: fine-tuned head | median 19.9%, mean 35% | 18.7%, mean 25% | mixed (better in 2 of 4 scenes) |
| Ours, 7-Scenes held-out (110 pairs) | median 5.8 cm, MAE 10.8 cm, 44% within 5 cm | median 1.3 cm, MAE 4.9 cm, 88% within 5 cm | large gain, needs a known length |
| Ours, 2 phone scenes (8 segments) | 4-15 cm average error | 0.2-2 cm | see caveats in section 6 |

Relative error was not computed on our held-out set. As a rough estimate only, our unanchored median is about
12% (Phase 7, chess) and the anchored one a few percent. The 50 cm row of the teammate's TUM test uses 143 photos.

## 3. Ablation: what each factor did

| Factor | Teammate (measured) | Ours (measured) | Verdict |
|---|---|---|---|
| Fix the overall scale | oracle scale per image: NYU 8.6 to 6.4%, DIODE indoor 29.8 to 17.3%, outdoor 75.7 to 24.3% | 3 depth-matched anchors: held-out median 5.8 to 1.3 cm; scale ratio drifted 0.89-1.36 between photo pairs | Both find scale matters; ours gains more |
| **A known real-world length** | known camera movement as the length: median 19.9% (one photo) to 10.2% (25 cm) and 8.8% (50 cm) | known length in the scene: median error down about 78% | **Agree: one known length fixes most of the scale error** |
| Second photo, no scale information | TUM: 19.9 to about 15%, in all 4 scenes; weakest where overall size is the main error | used by default, never ablated against one photo | Theirs fills our gap |
| Local smoothing (patch median) | 5x5 patch: no change | 3x3 to 11x11 patches: no gain (MAE 11.7 to 11.5 cm) | **Agree** |
| Multiple readings (swap, mirror, 8 votes) | not tested | no gain; vote spread does not predict error (rho = 0.00) | Ours only |
| Lens / focal length | true focal alone slightly worse on NYU | not tested (no intrinsics are used) | Gap |
| Per-pixel depth | oracle depth gives 3.5%; fine-tuned head gives 7.3% on NYU only | not changed; about 3% of pairs remain off by more than 40 cm | Complementary |
| Camera rotation | recovered to about 0.3 degrees; movement distance uncertain by about 20% | not used | Theirs only |
| Points near depth edges | edge pairs 12% against 8% elsewhere | avoided by design (smooth-region pairs) | Our numbers are optimistic |
| Transfer to new data | fine-tune: no gain on DIODE, small mixed gain on TUM | frozen model; same dataset family plus 2 phone scenes | Different questions |
| Outdoor scenes | overall size about 3.8x too small | bundled outdoor scenes placed 2-4 units away, implausible | **Agree** |

**What the two analyses agree on:** unanchored indoor error is about 8-20% depending on the data; local smoothing and extra
readings do not help (so the error is systematic, not noise); the scale is badly wrong outside the indoor training domain; and
**giving the model one real-world length, in whatever form, removes most of the scale error.**

## 3b. The scale-information ladder

The same idea appears in both studies: how much real-world size information is supplied.

| Scale information supplied | Teammate (TUM, indoor video) | Ours (7-Scenes, indoor) |
|---|---|---|
| None, one photo | median 19.9% | median 5.8 cm, MAE 10.8 cm (two photos used, no anchor) |
| None, two photos | 14.9% (25 cm move) | same as the row above |
| Known camera movement | 10.2% (25 cm), 8.8% (50 cm); mean error worsens | not used |
| Known length in the scene | not tested | median 1.3 cm, MAE 4.9 cm |

The rows are not comparable in size (different data, pairs and metrics). The shape is the same: more supplied scale, less error.
The practical difference: a known object needs a person to place it; a known movement needs a phone motion sensor, and the movement
must be at least about 25 cm.

## 4. Why the numbers are not directly comparable

- **Pair selection.** They test random pairs, including edges and far points. We use smooth-region pairs, which removes much of
  the per-pixel depth error that dominates in their data. Our error is lower partly by design.
- **How scale is fixed.** Their oracle uses one scale per image and needs the ground truth. Our anchor is estimated from known
  lengths, matched by depth, and local to the photo pair, so it corrects more.
- **Metric and range.** Relative error over metres-scale pairs against absolute error over 0.1-0.95 m gaps.
- **Task.** Their fine-tune changes the model for one photo; their extra experiment and our method both use two photos.

## 5. The experiments that would tie them together (not run)

A grid on our held-out set: **original or fine-tuned head**, each with **no scale information, a known movement, or a known length**.
- The anchor should fix scale and their head should fix per-pixel depth, so the two may add up.
- Their head was trained on features from a photo paired with itself, and our two-photo features may differ, so a gain might not carry over.
- It needs their checkpoint (`dpt_head_ft_best.pt`) and roughly 1-2 hours.
- A second test would be **known length plus known movement**, to see whether two scale sources agree and one can flag a bad estimate of the other.

## 6. Limits to state honestly

- **Ours:** one dataset family (7-Scenes, Kinect), gaps under 1 m, smooth-region pairs, and a known length is required.
  The held-out MAE of 4.9 cm sits right at the 5 cm target, and its 95% interval reaches 8.4 cm.
- **Phone check:** 2 scenes, 4 segments each, both nearly flat and viewed from above, where scale is almost uniform. Results were
  within 5 cm for all segments with the anchor. The tape-to-photo edge convention for the mattress scene is still unconfirmed
  (facing edges fit slightly better than outer edges). Not statistics, just a preliminary check.
- **Training data (both projects):** the MASt3R metric model lists an unreleased internal dataset, so a clean "not in the training data" claim is impossible for NYU, DIODE, TUM or 7-Scenes. We only know none of them is in the published list.
- **Teammate, main study:** one training run and one seed; random pairs rather than human clicks; DIODE intervals rest on only 10
  scanner positions per type; the DIODE focal length comes from the paper's stated field of view.
- **Teammate, two-photo test:** only 4 scenes, and photos from one video resemble each other, so its ranges are probably too
  optimistic. The camera movement was **known from tracked ground-truth poses**; a phone's motion sensors are noisier, which
  is untested. With known movement the typical answer improves but the average does not (34-35% to 38-40%), because a badly
  misjudged movement makes a few answers much worse. No safety check was built.
- **This comparison:** the teammate's figures are from their summary only.

## 7. Verification pass on the teammate's work (his assistant read his code and Drive files)

Status after his reply. "Verified" means checked in his code or printed outputs, not re-run by us.

| # | Item | Status |
|---|---|---|
| 1 | Weights file, keys and load line | **Not verified by running.** From the saving code: a dict with keys `dpt`, `epoch`, `cfg`, `val`; load with `model.downstream_head1.dpt.load_state_dict(ck['dpt'])`. File size (80,835,115 bytes) fits a plain copy of the 20,201,668-parameter head. |
| 2 | Image size and resizing | Verified: the same function as MASt3R's own `load_images(size=512)` in training and evaluation. Training ran the frozen parts in bf16, evaluation in fp32. |
| 3 | Fine-tuned head on two photos | **Not run.** The head was trained only on a photo fed twice. His notebook `05_verification.ipynb` (about 15-20 min on an L4) runs it. |
| 4 | Metric definitions, DIODE scale | Verified. "Within 10%" is strict `<`. Scale bias is the median of true over predicted depth. **DIODE indoor is about 18.5% too small, not 23%** (median ratio 1.227); outdoor 3.8x too small (ratio 3.766); NYU test about 3.7% too large. He corrected his write-up. The separate -27.6% pair-distance bias includes the focal-length effect. |
| 5 | Confidence output after fine-tuning | **Not checked.** Only distances were saved, and the loss never touched confidence. Run by notebook 05. |
| 6 | Hook-bug order | Verified from code: the cached baseline used the buggy hook, which affected only view 2's final decoder output, so view 1 was not affected. Caveat: a file cannot prove which version ran the "difference 0.0" check. That check was also only on two photos (`nyu_0000`, `nyu_0002`). |
| 7 | Training-data list | Searching the MASt3R repo found no NYU, DIODE or TUM dataset. The metric model's list includes Habitat, BlendedMVS, MegaDepth, ARKitScenes, Co3d, StaticThings3D, Stereo4K, VirtualKitti, WildRGBD, NianticMapFree, DL3DV and an **unreleased internal dataset**. Overlap cannot be ruled out, for either project's data. |
| 8 | Edge pairs, NYU test | Verified: median error 12.2% for edge pairs against 8.4% otherwise; about 6% of pairs are edge pairs. |
| 9 | TUM known movement | Verified: taken from ground-truth poses (motion capture), interpolated to each frame; reference photos every 1.5 s; the partner frame closest to 10/25/50 cm within a tolerance. |
| 10 | Plausibility check on the movement | Verified absent: the code only rescales by `move_true / max(move_pred, 1e-6)`. |
| 11 | Depth-dependent scale oracle | **Not run.** Needs per-pixel depths; run by notebook 05. Note the "6.4%" figure is NYU train; NYU test is 6.3%. |

**Still blocking the 2x2 experiment:** items 1, 3 and 5 (a working load check, the two-photo test, and what happened to the confidence output).

## 8. Sentences for the demo

"Two independent analyses of the same model agree on three things: indoor error is roughly 8-20% without help, local smoothing does
not fix it, and the dominant systematic problem is scale. Giving the model one real length fixes most of it: a known camera movement in
my teammate's test, a known object in the scene in mine. The approaches are complementary, and combining a fine-tuned head with a known
length is the obvious next experiment."
