---
license: cc-by-sa-4.0
task_categories:
  - object-detection
  - image-classification
language:
  - en
tags:
  - dice
  - polyhedral-dice
  - ttrpg
  - tabletop
  - object-detection
  - webcam
size_categories:
  - n<1K
pretty_name: dieCamera dice frames
---

# dieCamera — physical dice, read by webcam

351 webcam frames of physical polyhedral dice on a tray, with per-die **type**, **face
value** and **bounding box**. Collected to train the offline reader in
[dieCamera](https://github.com/eschatus/diecamera), an app that watches your dice tray
and posts the roll into a virtual tabletop.

**1,079 labelled dice** across the six standard types (d4, d6, d8, d10, d12, d20).
587 of those carry face values that are ground truth — confirmed by a human in the app,
or placed deliberately to a prompt. The rest are trustworthy for **type and box only**.
That distinction is a column, not a footnote — see *Label trust* below.

## Why it is laid out by rig and date

```
data/<camera>/<capture-date>/<frame-id>.jpg
data/metadata.jsonl
```

Both levels of that path are a domain boundary, and mixing across either is the main way
to get a misleading number out of this data.

**Rig** is the obvious one: a gooseneck webcam over a gray tray, a phone camera over
felt, and a Nintendo Switch camera are three different problems. A model trained on one
and evaluated on another loses most of its apparent accuracy.

**Date** is the one that cost us. The capture rig changed under the data. The app used to
ask the camera for 1080p and leave the lens wherever autofocus abandoned it; on
2026-08-01 it started requesting full sensor resolution and sweeping the lens for the
sharpest focus. Same dice, same camera, same table — and the crop across a die went from
~930px to ~1700px, with measured sharpness (variance of Laplacian at 224px) going 94 →
294 → 326. Train across that boundary without knowing it is there and the model learns
the blur rather than the numeral.

`metadata.jsonl` carries the exact timestamp and crop dimensions per frame, so any other
split — by sharpness, session, or lens era — is a filter away.

## Contents

| rig                                 | dates             | frames | dice | trusted faces |
| ----------------------------------- | ----------------- | -----: | ---: | ------------: |
| `hue-hd-camera-0c45-6341`           | 2026-07-10 → 07-15 |    105 |  336 |           198 |
| `android-webcam-18d1-4eed`          | 2026-07-10        |     70 |  238 |             6 |
| `hd-usb-camera-05a3-9520`           | 2026-07-19 → 08-01 |     89 |  225 |           206 |
| `unknown-rig`                       | 2026-07-09 → 07-10 |     65 |  219 |           116 |
| `triveni-s-iphone-2-camera`         | 2026-07-15, 07-22 |     17 |   56 |            56 |
| `nintendo-switch-camera-057e-206d`  | 2026-07-15 → 07-16 |      5 |    5 |             5 |

`unknown-rig` is the earliest capture generation, from before the app recorded which
camera took a frame. It is believed to be the HUE gooseneck but the frames do not say so,
and guessing would defeat the point of splitting by rig.

Frames are already cropped to the dice tray (the app's region-of-interest), which is why
image dimensions vary within a rig.

## Fields

| field            | meaning                                                                   |
| ---------------- | ------------------------------------------------------------------------- |
| `file_name`      | image path, relative to `data/`                                            |
| `camera`         | raw device label as the OS reported it                                     |
| `epoch`          | UTC capture date — the directory level above the frame                     |
| `captured_at`    | full ISO timestamp                                                         |
| `width`,`height` | crop dimensions in pixels                                                  |
| `label_source`   | `human` (confirmed in the app) or `teacher` (a batch vision-model pass)     |
| `values_trusted` | whether the **face values** may be trained on                              |
| `dice`           | `[{type, value, box:{x,y,w,h}, confidence}]`; boxes are frame fractions 0–1 |

```python
from datasets import load_dataset
ds = load_dataset("G-G-Games/diecamera-dice", split="train")

# Faces are only safe to train on where the flag says so.
faces = ds.filter(lambda r: r["values_trusted"])

# The sharp, full-resolution era on the current rig.
recent = ds.filter(lambda r: r["epoch"] >= "2026-08-01")
```

## Label trust

Labels arrive by three routes, and they are not equally good:

1. **In-app confirmation.** Every roll the app reads goes through a correct-step where a
   human clicks any die that was read wrong before it posts. What lands here is what a
   person signed off on. Best quality; these are `label_source: human`.
2. **Guided collection.** The app prompts for a specific set ("roll 2×d6 + 1×d20") and
   captures on settle, so the prompted set *is* the type/count ground truth with no model
   in the loop. In value-sweep mode the prompt names the exact faces to place, which makes
   the values ground truth too.
3. **Teacher passes.** A frontier vision model labelled the backlog. Usable for **types
   and boxes**, which it gets right; its face reads are exactly what the local model
   exists to replace, so they are never marked trusted.

`values_trusted` encodes the outcome of that. **Type and box labels are usable on every
row; face values are only usable where `values_trusted` is true.** A model trained on the
untrusted faces is being trained on another model's guesses.

## Known issues

- **d10 6-vs-9 is genuinely ambiguous** on some dice sets and is the single largest
  source of face error. Where a set marks orientation with a dot or underline the label
  follows the mark; ornate sets use a fleur-de-lis flourish, which is easy to mistake for
  a mislabel and is not one.
- **Thin, but evenly thin.** All 60 (type, face) combinations have trusted labels, and
  the rarest has 7 examples against the commonest's 20. Depth is the constraint, not
  balance — 7 examples of a d12 showing 9 is not many pictures of a numeral.
- **One d100 die in the whole corpus**, so percentile is effectively uncovered.
- **`values_trusted` is false for most of `android-webcam-18d1-4eed`** — that generation
  was type-labelled by guided collection and never had its faces confirmed.
- Boxes on human-confirmed frames originate from a model and were corrected only when
  visibly wrong, so box tightness is not uniform.

## Caveat on any accuracy number

Everything published from this corpus so far was trained *and* evaluated on it with
splits that are not recorded here. Treat single-number accuracies with suspicion and cut
your own held-out split — by rig, or by date, so the test set is a domain the model has
not seen. In-domain depth, not corpus size, is the binding constraint on this problem.

## Provenance and credit

Captured by [@eschatus](https://github.com/eschatus) across five rigs, with frames
contributed by **[@trivenigandhi](https://github.com/trivenigandhi)** (the
`triveni-s-iphone-2-camera` rig). Labels are human confirmations plus teacher passes as
described above.

Regenerate this layout from the source repo with `npm run dataset:export`.

## License

**CC BY-SA 4.0.** Share-alike: any redistribution or derivative dataset built from this
corpus must carry attribution and the same license forward. The
[dieCamera](https://github.com/eschatus/diecamera) application code is licensed separately
(see its own repo).

*(Provisional — this replaces an earlier CC BY 4.0 license on this card, to match the
"opt-in, copyleft" data-sharing terms agreed in principle on Aug 6. Not yet cleared by
counsel.)*
