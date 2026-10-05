# trackdedup
[![semantic-release](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--release-e10079.svg)](https://github.com/semantic-release/semantic-release)

Hungarian-algorithm de-duplication of in-situ particle detections in Planktivore data.

The same physical particle can appear in multiple images during a single
imaging burst (detector overlap) or across successive bursts as the camera
platform moves. `trackdedup` identifies those repeat detections and groups
them into a single canonical track, so downstream analyses count each
particle only once.

![Duplicate tracks animated overview](docs/imgs/dedup_visualization.gif)

---

## How it works

Detection metadata (box position, depth, timestamp) is parsed directly from
image filenames:

```
low_mag_cam-{timestamp_us}-{session}-…-{y}-{x}-{h}-{w}-{ignored}_rawcolor.jpg
```

The box center `(cx, cy)` is computed as `(x + w/2, y + h/2)`.
Depth and label information come from an accompanying `localizations.csv`.

Matching runs in two passes using `scipy.spatial.distance.cdist` and
`scipy.optimize.linear_sum_assignment`:

| Pass | Scope | Method |
|------|-------|--------|
| 1 — within-burst | Detections in the same imaging burst (< `--frame-gap` s apart) | Exhaustive pairwise distance |
| 2 — cross-burst | Burst pairs whose midpoints are within `--time-gate` s | Hungarian (globally optimal bijective matching) |

Both passes work in a normalised feature space `[cx/σ_xy, cy/σ_xy, depth/σ_depth]`
and accept matches whose Euclidean distance is ≤ `--max-cost`.
A Union-Find structure propagates track identity through all accepted matches,
so a particle seen three or more times is grouped into a single track.

The output CSV adds four columns to the input:

| Column | Description |
|--------|-------------|
| `track_id` | Integer ID shared by all detections of the same particle |
| `canonical_uuid` | UUID of the first-seen detection in the track |
| `is_duplicate` | `True` for every detection that is not the canonical one |
| `frame_id` | Imaging-burst index (detections with the same `frame_id` are < `--frame-gap` s apart) |

---

## Full example

From the repository root (`planktivore-trackdedup/`):

```bash
# 1. Extract the sample dataset
cd data
tar -zxf data.tar.gz
cd ..

# This creates data/ptvr_lm/ with:
#   localizations.csv          — input detections
#   crops/Velella_velella/     — crop images keyed by UUID
#   images/                    — full-frame images

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install numpy pandas scipy Pillow matplotlib

# 4. Run de-duplication
python src/hungarian_dedup.py data/ptvr_lm/localizations.csv \
    -o data/ptvr_lm/localizations_dedup.csv \
    --verbose

# 5. Generate the animated GIF
python src/visualize_dedup.py \
    --dedup data/ptvr_lm/localizations_dedup.csv \
    --crops data/ptvr_lm/crops/Velella_velella \
    --out docs/imgs/dedup_visualization.gif
```

Expected output from step 4 (default thresholds):

```
177 detections → 83 unique track(s)
 94 duplicate(s) identified (53.1%)
```

The GIF written in step 5 is the same animation shown at the top of this README.

---

## Usage

```bash
python src/hungarian_dedup.py data/ptvr_lm/localizations.csv \
    -o data/ptvr_lm/localizations_dedup.csv

# Tune matching thresholds:
python src/hungarian_dedup.py data/ptvr_lm/localizations.csv \
    --sigma-xy 300 \
    --sigma-depth 5 \
    --max-cost 1.5 \
    --frame-gap 2.0 \
    --time-gate 600

# Visualise results:
python src/visualize_dedup.py \
    --dedup data/ptvr_lm/localizations_dedup.csv \
    --crops data/ptvr_lm/crops/Velella_velella \
    --out docs/imgs/dedup_visualization.gif
```

## Dependencies

```
numpy
pandas
scipy
Pillow
matplotlib
```
 
## AI Disclosure

This project was developed with the assistance of AI tools, including Claude and Cursor. 
AI was used for code generation, refactoring, and documentation assistance.
