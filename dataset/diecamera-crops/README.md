---
license: other
license_name: cc-by-nc-4.0-ours-cc-by-4.0-roboflow
license_link: https://github.com/eschatus/diecamera/blob/main/LICENSE-NC.md
task_categories:
  - image-classification
language:
  - en
tags:
  - dice
  - polyhedral-dice
  - ttrpg
  - tabletop
  - dice-reading
pretty_name: dieCamera dice crops
size_categories:
  - n<10K
---

# dieCamera — per-die crops

One cropped image per physical die, labelled with its **type** and **face value**. This is
the deliberately-simple training set for dieCamera's offline value reader — the app that
watches a dice tray and posts the roll into a virtual tabletop
([source](https://github.com/eschatus/diecamera)).

For the full frames these crops were cut from (and the multi-die detector-training data), see
the companion repo **[G-G-Games/diecamera-frames](https://huggingface.co/datasets/G-G-Games/diecamera-frames)**.

## Schema

Standard 🤗 `imagefolder` layout — `load_dataset("G-G-Games/diecamera-crops")` needs no config.

```
data/<file>.jpg        one die, cropped to its bounding box + a small margin
data/metadata.jsonl    one row per crop
```

| column      | type   | meaning                                                                                                |
| ----------- | ------ | ------------------------------------------------------------------------------------------------------ |
| `file_name` | string | the crop image                                                                                         |
| `source`    | string | `rig` (our webcam) or `roboflow:<fork>` (a third-party image, see below)                               |
| `type`      | string | die type — `d4`, `d6`, `d8`, `d10`, `d12`, `d20`                                                       |
| `value`     | int    | the up-face value read (d10 may be 0)                                                                  |
| `date`      | string | capture date (`rig`), or publish date when the source has none                                         |
| `added_at`  | string | the day this crop first entered the dataset — filter `added_at > last_run` to train only on what's new |
| `holdout`   | bool   | `true` = reserved for evaluation; **filter these out when training**                                   |

> **Training tip.** Exclude eval frames and (optionally) skip what you've already trained on:
> `ds.filter(lambda r: not r["holdout"])`. `added_at` lets an incremental finetune pick up only
> rows added since its last run, instead of reprocessing the whole set.

Every die here has a **trusted** face value: rig dice are human-confirmed or placed to a
prompt; roboflow dice are the ones a human reviewed and confirmed by hand.

## Licence — read this before commercial use

This dataset is **mixed-licence**, and the `source` column tells you which applies per row:

| `source`          | licence                                                                          |
| ----------------- | -------------------------------------------------------------------------------- |
| `rig`             | **CC BY-NC 4.0 © G-G-Games** — free personal use; commercial needs a licence     |
| `roboflow:<fork>` | **CC BY 4.0 © the fork's original author** (commercial OK, attribution required) |

The **labels** on every row are G-G-Games' own work (CC BY-NC 4.0). Full terms:
[LICENSE-NC.md](https://github.com/eschatus/diecamera/blob/main/LICENSE-NC.md).

### Roboflow attribution (CC BY 4.0)

Crops with a `roboflow:` source derive from these [Roboflow Universe](https://universe.roboflow.com)
datasets, used with modifications (cropped; our own top-face type/value labels added):

| `source`                   | original author → dataset                                           |
| -------------------------- | ------------------------------------------------------------------- |
| `roboflow:200_dataset`     | **vkr-55xr7** → https://universe.roboflow.com/vkr-55xr7/200_dataset |
| `roboflow:d4-turbo-rad-v4` | **turbo-rad** → https://universe.roboflow.com/turbo-rad/d4-bmzdm    |

See [ATTRIBUTIONS.md](https://github.com/eschatus/diecamera/blob/main/ATTRIBUTIONS.md) for the
full provenance record.
