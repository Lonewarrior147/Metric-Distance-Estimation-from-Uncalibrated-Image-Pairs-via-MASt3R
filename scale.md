# scale.md - integrating the known-length scale anchor into the Gradio tool

**Audience:** the teammate (and his coding assistant) who owns the Gradio UI for the fine-tuned MASt3R head.
**Goal:** add our **scale anchor** to that UI, so the two ideas can be compared in an ablation:
(A) the fine-tuned head, (B) the scale anchor, (C) both. This file contains everything needed. Read it top to bottom.
Do not retrain anything. The anchor is pure post-processing on MASt3R's 3D points.

Where the original code lives (our repo, `main.ipynb`; find cells by content, not index):
- pixel mapping `original_to_model_xy`, `get_3d_point`: Phase 5, cell 5.1 (around cell 62)
- `anchor_correct`, `final_system`, `CONFIG8`: Phase 8, cell 8.4 (around cell 106)
- background and results: `CLAUDE.md`, `context.md` (section 14), `ABLATION_COMPARISON.md`

---

## 1. What the method does, in plain words

MASt3R gives every pixel a 3D point, so the distance between two clicked pixels is the straight-line distance between their 3D points.
That raw distance is roughly right in shape but its **overall size drifts** (we measured median ratios from 0.89x to 1.36x between photo pairs; his work
measured the same effect on NYU and DIODE). The fix:

1. The user marks one or more **anchors**: two points whose real distance they know (an A4 sheet's long edge = 29.7 cm, a tape-measured gap).
2. For each anchor compute `scale = true_length / model_length`.
3. Multiply the target's model distance by that scale.

That is all. The details that make it work well (which anchors to use, when to distrust the answer) are below. With it, on 7-Scenes
held-out scenes the typical error went from 5.8 cm to 1.3 cm (mean 10.8 to 4.9 cm), and on two phone scenes from 4-15 cm to 0.2-2 cm.
**A known length is required.** Without one, nothing changes.

---

## 2. Inputs and outputs

**Per measurement the function receives**
- `pts1`: float array `[H, W, 3]`, the 3D points of **image 1** from the model (camera frame of image 1; any head works). Index as `pts1[y, x]`.
- `grid_hw`: `(H, W)` of that array (for example `(384, 512)` for a 640x480 photo).
- `orig_wh`: `(width, height)` of image 1 **after EXIF rotation** (see section 6).
- `target`: `((x1, y1), (x2, y2))` the two points to measure, in **original pixels of image 1**.
- `anchors`: list of `(((x1, y1), (x2, y2)), length_m)` with known lengths in **metres**, same pixel convention.

**It returns a dict**
```
distance_m      anchored distance in metres (or the raw one if no usable anchor)
raw_m           the model's own distance, unanchored
scale           the factor applied (1.0 if no anchor)
n_anchors       how many anchors were used (at most k)
anchor_scales   list of the individual scales
disagreement    (max - min) / median of anchor_scales, None if fewer than 2 anchors
status          "OK" | "LOW_AGREEMENT" | "SINGLE_ANCHOR" | "NO_ANCHOR"
```
Status meaning: `OK` = at least 2 anchors and they agree (disagreement <= threshold). `LOW_AGREEMENT` = anchors disagree, so do not
trust the number. `SINGLE_ANCHOR` = only one anchor, agreement cannot be checked (show the value, mark unverified). `NO_ANCHOR` = raw value only.

---

## 3. Frozen settings (chosen on development scenes, then tested once on held-out scenes; do not re-tune on your own test set)

| Setting | Value | Meaning |
|---|---|---|
| estimator | plain, single pixel | `pts1[y, x]` at the clicked pixel. No patch median, no ensemble (both tested, no gain). |
| `k` | 3 | use up to 3 anchors; scale = **median** of their scales |
| anchor choice | by depth | take the anchors whose mean depth is nearest the target's |
| `anchor_min_m` | 0.25 | an anchor shorter than 25 cm is ignored (a short anchor amplifies its own error) |
| refusal signal | anchor disagreement | `(max - min) / median` of the k scales |
| `refusal_threshold` | 0.157 | answer only if disagreement <= this (chosen so ~70% of development pairs were answered) |
| `share_tol_px` | 3 | an anchor may not share an endpoint (within 3 px) with the target, nor be the target |

Depth used for matching = the **mean of the Z coordinates** (third component of `pts1`) of the segment's two endpoints, from the same model
output, **before** scaling. Never use ground-truth depth here.

---

## 4. Reference implementation (tested; copy as is)

<!-- REF-IMPL-START -->
```python
import numpy as np

CFG = dict(k=3, anchor_min_m=0.25, refusal_threshold=0.157, share_tol_px=3.0, size=512, patch=16)


def orig_to_grid(xy, orig_wh, grid_hw, size=512, patch=16):
    """Map an (x, y) pixel on the ORIGINAL photo to the (x, y) grid the model actually saw.
    Mirrors dust3r.utils.image.load_images: resize the long side to `size`, then centre-crop to a multiple of `patch`.
    Returns float (x, y) on the model grid."""
    W1, H1 = orig_wh
    S = max(W1, H1)
    W, H = int(round(W1 * size / S)), int(round(H1 * size / S))
    cx, cy = W // 2, H // 2
    halfw = ((2 * cx) // patch) * patch / 2
    halfh = ((2 * cy) // patch) * patch / 2
    if W == H:
        halfh = 3 * halfw / 4
    left, top = cx - halfw, cy - halfh
    assert (int(2 * halfh), int(2 * halfw)) == tuple(int(v) for v in grid_hw), \
        f"grid {tuple(grid_hw)} does not match the loader's crop {(int(2*halfh), int(2*halfw))}"
    x = (xy[0] + 0.5) * W / W1 - 0.5 - left
    y = (xy[1] + 0.5) * H / H1 - 0.5 - top
    return float(x), float(y)


def segment_raw(pts1, seg, orig_wh, cfg=CFG):
    """(raw length in model units, mean depth Z) of a segment given by two ORIGINAL-pixel points."""
    H, W = pts1.shape[:2]
    P = []
    for xy in seg:
        gx, gy = orig_to_grid(xy, orig_wh, (H, W), cfg["size"], cfg["patch"])
        x, y = int(round(gx)), int(round(gy))
        assert 0 <= x < W and 0 <= y < H, f"point {xy} falls outside the model grid (cropped away?)"
        P.append(np.asarray(pts1[y, x], float))          # NOTE: [y, x] indexing; input is (x, y)
    assert np.isfinite(P[0]).all() and np.isfinite(P[1]).all(), "non-finite 3D point"
    return float(np.linalg.norm(P[0] - P[1])), float(0.5 * (P[0][2] + P[1][2]))


def _shares_endpoint(a, b, tol):
    return any(np.hypot(p[0] - q[0], p[1] - q[1]) <= tol for p in a for q in b)


def measure_with_anchors(pts1, orig_wh, target, anchors, cfg=CFG):
    """Anchored metric distance of `target`. See scale.md section 2 for the meaning of every field."""
    raw, z_t = segment_raw(pts1, target, orig_wh, cfg)
    pool = []
    for seg, length_m in anchors:
        if length_m < cfg["anchor_min_m"]:
            continue                                         # too short to be a reliable anchor
        if _shares_endpoint(seg, target, cfg["share_tol_px"]):
            continue                                         # same segment, or shares a point with the target
        a_raw, z_a = segment_raw(pts1, seg, orig_wh, cfg)
        if a_raw <= 0:
            continue
        pool.append((abs(z_a - z_t), length_m / a_raw))
    if not pool:
        return dict(distance_m=raw, raw_m=raw, scale=1.0, n_anchors=0, anchor_scales=[], disagreement=None, status="NO_ANCHOR")
    pool.sort(key=lambda t: t[0])                            # nearest in depth first
    scales = [s for _, s in pool[:cfg["k"]]]
    s = float(np.median(scales))
    disagreement = float((max(scales) - min(scales)) / s) if len(scales) >= 2 else None
    if disagreement is None:
        status = "SINGLE_ANCHOR"
    else:
        status = "OK" if disagreement <= cfg["refusal_threshold"] else "LOW_AGREEMENT"
    return dict(distance_m=raw * s, raw_m=raw, scale=s, n_anchors=len(scales), anchor_scales=scales,
                disagreement=disagreement, status=status)
```
<!-- REF-IMPL-END -->

Notes on the code:
- It is deliberately independent of any model code. Feed it the `[H, W, 3]` 3D points that your head produces for **image 1**.
- `segment_raw` uses a single pixel. Do not add smoothing: we tested patch medians (3x3 to 11x11) and extra votes, and neither helped.
- If you run the **two-photo** mode, image 1 is the photo whose pixels the user clicks, and `pts1` is the first head's output for it.

---

## 5. Tests the integration must pass

Copy these next to the reference code and run them before wiring the UI.

<!-- TESTS-START -->
```python
def test_pixel_mapping():
    # 640x480 photo -> 512x384 grid, no crop: the photo corners land on the grid corners
    x0, y0 = orig_to_grid((0, 0), (640, 480), (384, 512)); x1, y1 = orig_to_grid((639, 479), (640, 480), (384, 512))
    assert abs(x0) < 0.5 and abs(y0) < 0.5 and abs(x1 - 511) < 0.5 and abs(y1 - 383) < 0.5
    # portrait 575x1280 photo -> 224x512 grid (3 columns are cropped away on each side): the centre pixel lands on the grid centre
    cx, cy = orig_to_grid((287, 639), (575, 1280), (512, 224))
    assert abs(cx - 111.5) < 0.6 and abs(cy - 255.5) < 0.6
    # a wrong grid shape must be rejected, not silently accepted
    try:
        orig_to_grid((10, 10), (640, 480), (384, 500)); raise SystemExit("accepted a wrong grid shape")
    except AssertionError:
        pass


def _plane(scale=1.0, H=384, W=512):
    # a plane 2 m from the camera; the "model" output is `scale` times too large
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    return np.stack([(xx - W / 2) / 250.0, (yy - H / 2) / 250.0, np.full((H, W), 2.0)], -1) * scale


def test_pure_scale_error_is_removed():
    wh = (640, 480); truth_plane, model_plane = _plane(1.0), _plane(1.2)       # the model is 1.2x too large
    true_len = lambda s: segment_raw(truth_plane, s, wh)[0]
    segs = [((100, 100), (300, 120)), ((50, 300), (250, 330)), ((400, 100), (600, 150)), ((300, 400), (500, 380))]
    target, others = segs[0], segs[1:]
    r = measure_with_anchors(model_plane, wh, target, [(s, true_len(s)) for s in others], dict(CFG, anchor_min_m=0.0))
    assert abs(r["distance_m"] - true_len(target)) < 1e-9 and abs(r["scale"] - 1 / 1.2) < 1e-9
    assert r["status"] == "OK" and r["n_anchors"] == 3 and r["disagreement"] < 1e-9


def test_edge_cases():
    pts, wh, t = _plane(), (640, 480), ((100, 100), (300, 120))
    a1, a2, a3 = ((50, 300), (250, 330)), ((400, 100), (600, 150)), ((300, 400), (500, 380))
    r = measure_with_anchors(pts, wh, t, []);                       assert r["status"] == "NO_ANCHOR" and r["scale"] == 1.0
    r = measure_with_anchors(pts, wh, t, [(a1, 0.10)]);             assert r["status"] == "NO_ANCHOR"        # shorter than 25 cm: ignored
    r = measure_with_anchors(pts, wh, t, [(t, 0.5)]);               assert r["status"] == "NO_ANCHOR"        # the target is never its own anchor
    r = measure_with_anchors(pts, wh, t, [(((100, 101), (400, 300)), 0.5)]); assert r["status"] == "NO_ANCHOR"   # shares an endpoint
    r = measure_with_anchors(pts, wh, t, [(a1, 0.5)]);              assert r["status"] == "SINGLE_ANCHOR"
    r = measure_with_anchors(pts, wh, t, [(a1, 0.5), (a2, 1.5), (a3, 0.5)]); assert r["status"] == "LOW_AGREEMENT"   # one wild anchor


def test_regression_against_our_phone_runs():
    # Raw model lengths (cm) and the anchored answers our run printed (leave-one-out, k=3, 4 segments: all 3 others are always used).
    cases = [([38.6, 56.5, 84.7, 32.8], [35.0, 50.0, 80.0, 29.7], [35.0, 51.3, 76.7, 29.7]),    # tile scene, image 1 = photo 1
             ([50.9, 76.6, 57.4, 37.2], [40.0, 60.0, 45.0, 29.7], [39.9, 60.1, 45.1, 29.1])]    # mattress scene (facing edges), photo 1
    for raw, true, expected in cases:
        for i in range(4):
            scales = [true[j] / raw[j] for j in range(4) if j != i]
            assert abs(raw[i] * float(np.median(scales)) - expected[i]) < 0.35, (i, expected[i])   # printed values are rounded to 0.1 cm


for t in (test_pixel_mapping, test_pure_scale_error_is_removed, test_edge_cases, test_regression_against_our_phone_runs):
    t(); print("PASS", t.__name__)
```
<!-- TESTS-END -->

The last test reproduces numbers that our notebook run printed (see `ABLATION_COMPARISON.md`, phone-photo check). Passing it means the
arithmetic of the anchor logic matches ours. It does not test your pixel mapping, which is the likeliest place for a bug; test that
separately by clicking a known corner of an image and printing where it lands on the grid.

---

## 6. Pixel conventions (the usual source of silent bugs)

1. **Clicks are `(x, y)`; the point array is `pts1[y, x]`.** The reference code asserts the bounds.
2. **Clicks must be in ORIGINAL pixels of image 1**, not in whatever size the UI shows. Gradio's `Image.select` event provides `evt.index = (x, y)`; check
   which coordinates it reports (displayed or original) by clicking a known corner and printing the value, and convert back to original pixels if the UI scales the image for display.
3. **EXIF rotation:** MASt3R's loader applies `exif_transpose`. Apply it to the image for display *and* for the model, and use the
   transposed size as `orig_wh`. Phone photos are often rotated by EXIF only. Both photos of a pair should be held the same way.
4. **Grid size:** the model sees a long side of 512, cropped to a multiple of 16 (640x480 -> 512x384; portrait 575x1280 -> 224x512).
   `grid_hw` must be the shape of the array you pass. The function asserts it matches the loader's crop.
5. **Do not crop or re-save photos unevenly.** If the two photos have different sizes, each uses its own mapping (the model handles it).
6. **Units:** the model's output is in metres only approximately. Lengths you pass as anchors must be in **metres**.

---

## 7. UI changes (Gradio)

Add a mode "Use known-length anchors". Suggested minimum:

- **Inputs:** photo 1 (required), photo 2 (optional but recommended: our results used two photos), a click mode selector
  (`target` / `anchor`), and a length box (cm) shown when the mode is `anchor`.
- **Flow:** the user clicks two points. If the mode is `anchor`, they type the real length and the segment is stored in a table.
  If it is `target`, the segment is measured. Show a table of stored anchors with a delete button.
- **Outputs:** the anchored distance in cm, the raw model distance in cm, the scale factor, the number of anchors used, the anchor
  disagreement and the status. Draw all segments on the image: anchors in one colour, the target in another.
- **Wording by status:** `OK` normal; `SINGLE_ANCHOR` "one anchor: value not cross-checked"; `LOW_AGREEMENT` "anchors disagree: do not trust";
  `NO_ANCHOR` "no usable anchor (needs a known length of at least 25 cm)".
- **Advice shown to the user:** use a flat A4 sheet (29.7 cm long) near the thing being measured, at about the same distance from the
  camera, and add one or two more tape-measured lengths of at least 25 cm. Take the second photo 10-50 cm to the side.

State to keep: the model output for the current photo pair (`pts1`, `orig_wh`) so that adding anchors does not re-run the model.

---

## 8. Ablation: which performs better

Run all four arms on **identical clicks and identical anchors**:

| Arm | Head | Anchor |
|---|---|---|
| A | original | none |
| B | fine-tuned | none |
| C | original | yes |
| D | fine-tuned | yes |

Both heads are used the same way: take `pts1` from that head, then either use the raw distance (A, B) or `measure_with_anchors` (C, D).
The anchor does not care which head produced `pts1`.

**Evaluation data (use ground truth only to score and to supply anchor lengths, never to choose anchors):**
1. **Phone scenes with tape measurements:** the best evidence for the application. Each scene: 2 photos, at least 4 tape-measured segments of 25 cm or more, no shared endpoints. Leave-one-out: each segment is the target once and the others are anchors.
2. **7-Scenes held-out (`redkitchen`, `pumpkin`, `fire`):** our protocol and numbers are in notebook Phase 8; reproduce the frozen results first (section 10).
3. **NYU / TUM from your side:** anchors can be built from ground-truth depth: pick other random pixel pairs in the same photo with true length of at least 0.25 m and use those lengths as anchors (same endpoint rule). Note this makes the anchor lengths carry sensor noise.

**Report, per arm:** median and mean absolute error (cm), median and mean relative error (%), share within 5 cm and within 10%, and for the anchor arms the
share of pairs with status `OK`. Always show the **coverage** (share answered) next to any error computed after refusal.
Give 95% intervals by resampling whole scenes (or whole photo pairs), not individual pairs.

**Rules that keep it honest**
- Fix all settings (section 3) before looking at the evaluation data.
- Report arms where an idea does not help. In our own tests the ensemble and patch smoothing gave nothing, and his fine-tune did not transfer to DIODE.
- Say what each arm needs: the anchor needs a known length; the fine-tuned head needs nothing at use time.
- Two-photo versus self-pair (the same photo twice): our anchor results are from **two different photos**. His head was trained on self-pair features. Run both input modes for both heads if possible, and report them separately.

**What we expect, to sanity-check, not to aim at:** from our data, anchors should help far more than the fine-tuned head wherever the model's scale is off
(DIODE-like or unusual scenes), and the fine-tuned head may add a little on NYU-like scenes. If you see the opposite, look for a pixel-mapping bug first.

---

## 9. Edge cases and what to do

| Situation | Behaviour |
|---|---|
| No anchor, or all shorter than 25 cm | `NO_ANCHOR`: show the raw value, labelled unanchored |
| Exactly one anchor | `SINGLE_ANCHOR`: value shown, not cross-checked |
| Anchors disagree (> 0.157) | `LOW_AGREEMENT`: warn, do not present as reliable |
| Anchor shares an endpoint with the target (within 3 px) | ignored |
| A click lands in the cropped-away strip | assertion error: show "point outside the analysed area" |
| Non-finite 3D point | assertion error: ask the user to pick another pixel |
| Raw anchor length of 0 | skipped |
| More than 3 usable anchors | the 3 nearest in depth are used |

---

## 10. Reproducing our numbers (do this once so you trust the integration)

1. Run our notebook `main.ipynb` through Phase 8 (needs a GPU, Internet on). The held-out table in 8.5 is the reference:
   baseline median 5.8 cm / mean 10.8 cm; final system median 1.3 cm / mean 4.9 cm (95% interval for the mean 2.0-8.4); 88% within 5 cm.
2. Our two phone scenes gave, with `facing` edges for the sticky notes: all segments within 5 cm, with anchored mean error 0.2-2.0 cm (raw model errors 4-15 cm).
3. Your integration should match the anchor arithmetic exactly on identical raw lengths (section 5, last test).

---

## 11. Limits to keep in your write-up

- One known length is required. Results are from 7-Scenes (Kinect) and two near-flat phone scenes; gaps were 0.1-0.95 m.
- Anchors in our experiment came from other ground-truth pairs of the same photos (Kinect noise 1-2 cm); a printed A4 sheet is exact, so the anchor is, if anything, pessimistic.
- Ground-truth pairs were chosen in smooth regions, so the pairs are easier than a person's clicks.
- The refusal threshold 0.157 was calibrated on two development scenes; recalibrate it on your own data if you change the setup, and say so.
- The 4.9 cm held-out mean sits at the 5 cm target and its interval reaches 8.4 cm. Do not describe it as "accurate to 5 cm".
