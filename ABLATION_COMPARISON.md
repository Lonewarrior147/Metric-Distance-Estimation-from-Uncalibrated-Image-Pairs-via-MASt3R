# Ablation and comparison: two ways of improving MASt3R's metric distances

Two independent analyses of the same pretrained model, compared side by side. **No new experiments were run for
this document.** Our numbers come from `main.ipynb` (Phases 6-8, run on a Kaggle T4) and the phone-photo check.
The teammate's numbers are taken from their written summary and have **not** been re-run or independently verified.

- **Teammate:** fine-tune MASt3R's 3D-point head, one photo at a time, no extra information.
- **Ours:** keep the model frozen, use two photos, and fix the scale with a known length in the photos.

## 1. The two approaches

| | Teammate | Ours |
|---|---|---|
| Input | 1 photo paired with itself, 2 clicks | 2 photos, 2 clicks in photo 1 |
| Extra information | none | at least 1 known length (A4 sheet, tape) |
| Model change | DPT head fine-tuned (20.2M of 688.6M parameters), rest frozen | none, model frozen |
| Data | NYU Depth v2 (547 held-out images), DIODE (771) | 7-Scenes (5 scenes: 2 for choosing settings, 3 held out), 2 phone scenes |
| Ground truth | raw Kinect depth (NYU), laser scanner (DIODE) | Kinect depth + camera poses (7-Scenes), tape measure (phone) |
| Pairs | 300 random per image, at least 0.25 m and 16 px apart | 10 per photo pair, 0.1-0.95 m, chosen in smooth regions |
| Metric | relative error | absolute error (cm) |
| Held-out rule | split by scene | split by scene; settings fixed before the held-out run |
| Uncertainty | resampling whole scenes | resampling whole photo pairs |

## 2. Headline results

| | Before | After | Change |
|---|---|---|---|
| Teammate, NYU test | median 8.5%, 56% within 10% | 7.3%, 63% | real gain (interval excludes zero) |
| Teammate, DIODE indoor | median 29.8%, 12% within 10% | 30.1%, 10% | no gain |
| Teammate, DIODE outdoor | median 75.7% | 75.5% | no gain |
| Ours, 7-Scenes held-out (110 pairs) | median 5.8 cm, MAE 10.8 cm, 44% within 5 cm | median 1.3 cm, MAE 4.9 cm, 88% within 5 cm | large gain, needs a known length |
| Ours, 2 phone scenes (8 segments) | 4-15 cm average error | 0.2-2 cm | see caveats in section 6 |

Relative error was not computed on our held-out set. As a rough estimate only, our unanchored median is about
12% (Phase 7, chess) and the anchored one a few percent.

## 3. Ablation: what each factor did

| Factor | Teammate (measured) | Ours (measured) | Verdict |
|---|---|---|---|
| Fix the overall scale | oracle scale per image: NYU 8.6 to 6.4%, DIODE indoor 29.8 to 17.3%, outdoor 75.7 to 24.3% | 3 depth-matched anchors: held-out median 5.8 to 1.3 cm; scale ratio drifted 0.89-1.36 between photo pairs | Both find scale matters; ours gains much more |
| Local smoothing (patch median) | 5x5 patch: no change | 3x3 to 11x11 patches: no gain (MAE 11.7 to 11.5 cm) | **Agree** |
| Multiple readings (swap, mirror, 8 votes) | not tested | no gain; vote spread does not predict error (rho = 0.00) | Ours only |
| Lens / focal length | true focal alone slightly worse on NYU | not tested (no intrinsics are used) | Gap |
| Per-pixel depth | oracle depth gives 3.5%; fine-tuned head gives 7.3% on NYU only | not changed; about 3% of pairs remain off by more than 40 cm | Complementary |
| Second view | not used | used by default, never ablated against one view | Gap for both |
| Points near depth edges | edge pairs 12% against 8% elsewhere | avoided by design (smooth-region pairs) | Our numbers are optimistic |
| Transfer to new data | fine-tune does not transfer to DIODE | frozen model; same dataset family plus 2 phone scenes | Different questions |
| Outdoor scenes | overall size about 3.8x too small | bundled outdoor scenes placed 2-4 units away, implausible | **Agree** |

**What the two analyses agree on:** unanchored indoor error is about 8-12%; local smoothing and extra readings do
not help (so the error is systematic, not noise); the scale is badly wrong outside the indoor training domain.

## 4. Why the numbers are not directly comparable

- **Pair selection.** They test random pairs, including edges and far points. We use smooth-region pairs, which removes much of
  the per-pixel depth error that dominates in their data. Our error is lower partly by design.
- **How scale is fixed.** Their oracle uses one scale per image and needs the ground truth. Our anchor is estimated from known
  lengths, matched by depth, and local to the photo pair, so it corrects more.
- **Metric and range.** Relative error over metres-scale pairs against absolute error over 0.1-0.95 m gaps.
- **Task.** Their fix changes the model for one photo. Ours adds one known length to two photos.

## 5. The experiment that would tie them together (not run)

A 2x2 grid on our held-out set: **original head or fine-tuned head**, each **with or without the anchor**. The anchor
should fix scale and their head should fix per-pixel depth, so the two may add up.
One risk: their head was trained on features from a photo paired with itself, and our two-photo features may look different,
so the gain might not carry over. It also needs their checkpoint (`dpt_head_ft_best.pt`) and roughly 1-2 hours.

## 6. Limits to state honestly

- **Ours:** one dataset family (7-Scenes, Kinect), gaps under 1 m, smooth-region pairs, and a known length is required.
  The held-out MAE of 4.9 cm sits right at the 5 cm target, and its 95% interval reaches 8.4 cm.
- **Phone check:** 2 scenes, 4 segments each, both nearly flat and viewed from above, where scale is almost uniform. Results were
  within 5 cm for all segments with the anchor. The tape-to-photo edge convention for the mattress scene is still unconfirmed
  (facing edges fit slightly better than outer edges). Not statistics, just a preliminary check.
- **Teammate:** one training run and one seed; random pairs rather than human clicks; DIODE intervals rest on only 10 scanner
  positions per type; the DIODE focal length comes from the paper's stated field of view.
- **This comparison:** the teammate's figures are from their summary only.

## 7. Questions to ask the teammate

1. DIODE indoor shows a pair-distance bias of -27.6% but a per-image scale bias of +22.7%. Different statistics, but are they consistent?
2. Was the fine-tuned head evaluated only on self-paired photos?
3. Is "within 10%" defined relative to the true distance?

## 8. Two sentences for the demo

"Two independent analyses of the same model agree that indoor error is about 8-12%, that scale is the dominant systematic problem,
and that local smoothing does not help. They change the model and we change the readout, so the approaches should be
complementary, and the combined version is the obvious next experiment."
