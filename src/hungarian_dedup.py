#!/usr/bin/env python3
"""
Hungarian-based de-duplication of particle detections.

Detections of the same physical particle across different time bursts are
identified by matching on box center (cx, cy), depth, and time using the
Hungarian algorithm (linear_sum_assignment) with a normalized Euclidean cost.

Filename format:
  low_mag_cam-{timestamp_us}-{session}-{frame}-{roi_idx}-{x}-{y}-{w}-{h}_rawcolor.jpg
  Last 4 dash-separated numbers before _rawcolor are x, y, w, h (pixels).

Inputs:
  localizations.csv — depth and label metadata, box from the media filename.
  CFE lab *.parquet — filename, epoch_seconds, time, and depth columns.
  Parquet filename values are relative; pass --base-path to build the full
  image path (base-path / filename).
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_EXPECTED_PATTERN = re.compile(
    r'^low_mag_cam-(\d+)-\d+-.+-(\d+)-(\d+)-(\d+)-(\d+)_rawcolor\.\w+$',
    re.IGNORECASE,
)


def parse_filename(fname: str) -> dict | None:
    """
    Extract box coordinates and timestamp from an image filename.

    Returns dict with keys: x, y, w, h, cx, cy, timestamp_us, time_s,
    or None if the filename does not match the expected pattern.

    Expected pattern:
      low_mag_cam-{timestamp_us}-{session}-{...}-{x}-{y}-{w}-{h}_rawcolor.ext
    The last four dash-separated numbers before _rawcolor are x, y, w, h.
    """
    basename = Path(fname).name
    stem = re.sub(r'_rawcolor\.\w+$', '', basename, flags=re.IGNORECASE)
    parts = stem.split('-')

    # Need at least: cam_name, timestamp, session, …, x, y, w, h  (≥6 parts)
    # and all of the last 4 + parts[1] must be integers.
    # parts[-5] is the zero-padded ROI index within the camera frame, not a coordinate.
    if len(parts) < 6:
        return None
    try:
        x, y, w, h = int(parts[-4]), int(parts[-3]), int(parts[-2]), int(parts[-1])
        timestamp_us = int(parts[1])
    except (ValueError, IndexError):
        return None

    return {
        'box_x': x, 'box_y': y, 'box_w': w, 'box_h': h,
        'cx': x + w / 2.0,
        'cy': y + h / 2.0,
        'timestamp_us': timestamp_us,
        'time_s': timestamp_us / 1e6,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def resolve_image_path(filename: str, base_path: str | None) -> str:
    """Join a relative filename onto base_path. Absolute paths are unchanged."""
    name = str(filename)
    if not base_path:
        return name
    path = Path(name)
    if path.is_absolute():
        return name
    return str(Path(base_path) / path)


def epoch_to_time_s(epoch: pd.Series) -> pd.Series:
    """Convert an epoch column to Unix seconds.

    CFE lab parquet names this column ``epoch_seconds``, but the stored
    values are the camera timestamp in microseconds — the same integer
    embedded in the image filename. Magnitudes at or above 1e14 are treated
    as microseconds, at or above 1e11 as milliseconds, and smaller values
    as seconds.
    """
    values = pd.to_numeric(epoch, errors='coerce')
    median = values.dropna().median()
    if pd.isna(median):
        return values
    if median >= 1e14:
        return values / 1e6
    if median >= 1e11:
        return values / 1e3
    return values


def _datetime_to_time_s(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors='coerce')
    epoch = pd.Timestamp('1970-01-01', tz='UTC')
    return (dt - epoch).dt.total_seconds()


def find_dinov3_label_column(columns) -> str | None:
    """Return the dinov3 model label column (not its -score column), if any."""
    label_cols = sorted(
        str(name) for name in columns
        if str(name).lower().startswith('dinov3')
        and not str(name).lower().endswith('-score')
    )
    if not label_cols:
        return None
    if len(label_cols) > 1:
        print(f"  Warning: several dinov3 label columns {label_cols}; using {label_cols[0]!r}.")
    return label_cols[0]


def _map_model_label(df: pd.DataFrame) -> pd.DataFrame:
    """Copy a dinov3 label/score pair onto Label and score when those are absent."""
    if 'label' in df.columns and 'Label' not in df.columns:
        df['Label'] = df['label']
    if 'Label' in df.columns:
        return df

    label_col = find_dinov3_label_column(df.columns)
    if label_col is None:
        return df

    df['Label'] = df[label_col]
    score_col = f'{label_col}-score'
    if score_col in df.columns and 'score' not in df.columns:
        df['score'] = pd.to_numeric(df[score_col], errors='coerce')
    return df


def _ensure_filename_column(df: pd.DataFrame) -> pd.DataFrame:
    """Require a filename column, or adopt the first column when it holds paths."""
    if 'filename' in df.columns:
        return df
    first = df.columns[0]
    sample = df[first].astype(str).head(20)
    if sample.str.contains(r'cam-\d+-', regex=True).any():
        print(f"  Warning: no 'filename' column; using first column {first!r}.")
        return df.rename(columns={first: 'filename'})
    raise SystemExit(
        "Error: parquet is missing required column 'filename' "
        f"(columns: {list(df.columns)})"
    )


def _attach_filename_fields(df: pd.DataFrame, filename_col: str) -> pd.DataFrame:
    """Parse box geometry from filenames and drop rows that do not match."""
    parsed_raw = df[filename_col].apply(parse_filename)
    unparseable = parsed_raw.isna()
    if unparseable.any():
        n_skip = int(unparseable.sum())
        skipped = df.loc[unparseable, filename_col].astype(str).tolist()
        print(f"  Warning: skipping {n_skip} row(s) with non-standard filenames:")
        for name in skipped[:10]:
            print(f"    {name}")
        if n_skip > 10:
            print(f"    … and {n_skip - 10} more")
        df = df.loc[~unparseable].reset_index(drop=True)
        parsed_raw = parsed_raw.loc[~unparseable].reset_index(drop=True)

    parsed = parsed_raw.apply(pd.Series)
    # Column clocks override the timestamp embedded in the filename.
    parsed = parsed.drop(columns=['time_s'], errors='ignore')
    return pd.concat([df.reset_index(drop=True), parsed], axis=1)


def _assign_time(df: pd.DataFrame) -> pd.DataFrame:
    """Set time_s and datetime_utc from epoch_seconds, time, or the filename."""
    if 'epoch_seconds' in df.columns:
        df['time_s'] = epoch_to_time_s(df['epoch_seconds'])
    elif 'timestamp_us' in df.columns:
        df['time_s'] = df['timestamp_us'] / 1e6

    if 'time' in df.columns:
        from_time = _datetime_to_time_s(df['time'])
        if 'time_s' not in df.columns:
            df['time_s'] = from_time
        df['datetime_utc'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    elif 'iso_datetime.1' in df.columns:
        df['datetime_utc'] = pd.to_datetime(df['iso_datetime.1'], utc=True, errors='coerce')
    elif 'iso_datetime' in df.columns:
        df['datetime_utc'] = pd.to_datetime(df['iso_datetime'], utc=True, errors='coerce')
    elif 'time_s' in df.columns:
        df['datetime_utc'] = pd.to_datetime(df['time_s'], unit='s', utc=True, errors='coerce')

    if 'time_s' not in df.columns:
        raise SystemExit(
            "Error: could not determine detection time. "
            "Expected epoch_seconds, time, or a camera timestamp in the filename."
        )
    return df


def load_localizations(path: str, base_path: str | None = None) -> pd.DataFrame:
    """
    Load a localizations CSV or a CFE lab parquet file.

    CSV rows use ``media`` for the image name. The CSV may have two columns
    both named 'iso_datetime'; pandas renames the second to 'iso_datetime.1'.
    The UTC value (contains '+00:00') is preferred.

    Parquet rows use ``filename`` (relative), ``epoch_seconds``, ``time``,
    and ``depth``. ``--base-path`` is joined onto each relative filename to
    form ``image_path``. ``epoch_seconds`` is the camera timestamp in
    microseconds. ``uuid`` falls back to the relative filename when absent.
    """
    source = Path(path)
    if source.suffix.lower() == '.parquet':
        try:
            df = pd.read_parquet(source)
        except ImportError as exc:
            raise SystemExit(
                "Error: reading parquet requires pyarrow. "
                "Install it with: pip install pyarrow"
            ) from exc
        df = _ensure_filename_column(df)
        df['filename'] = df['filename'].astype(str)
        if not base_path:
            print("  Warning: no --base-path given; image_path keeps the relative filename.")
        df['image_path'] = df['filename'].map(lambda name: resolve_image_path(name, base_path))
        df['media'] = df['image_path']
        if 'uuid' not in df.columns:
            df['uuid'] = df['filename']
        df = _map_model_label(df)
        filename_col = 'filename'
    else:
        df = pd.read_csv(source)
        if 'media' not in df.columns:
            raise SystemExit(
                "Error: CSV is missing required column 'media' "
                f"(columns: {list(df.columns)})"
            )
        df['media'] = df['media'].astype(str)
        if base_path:
            df['image_path'] = df['media'].map(lambda name: resolve_image_path(name, base_path))
            df['media'] = df['image_path']
        filename_col = 'media'

    if 'depth' in df.columns:
        df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    else:
        df['depth'] = np.nan
    df = _attach_filename_fields(df, filename_col)
    if df.empty or 'cx' not in df.columns:
        raise SystemExit("Error: no detections left after parsing filenames.")
    df = _assign_time(df)
    return df.sort_values('time_s').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Frame grouping
# ---------------------------------------------------------------------------

def assign_frames(df: pd.DataFrame, frame_gap_s: float = 2.0) -> pd.DataFrame:
    """
    Assign a frame_id to each detection.

    Detections separated by less than frame_gap_s seconds are in the
    same frame (same imaging burst). Larger gaps start a new frame.
    """
    df = df.copy()
    frame_ids = np.zeros(len(df), dtype=int)
    fid = 0
    for i in range(1, len(df)):
        if df.loc[i, 'time_s'] - df.loc[i - 1, 'time_s'] > frame_gap_s:
            fid += 1
        frame_ids[i] = fid
    df['frame_id'] = frame_ids
    return df


# ---------------------------------------------------------------------------
# Union-Find (for track graph)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Core: Hungarian matching between two frame groups
# ---------------------------------------------------------------------------

def label_mismatch(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """Boolean matrix, True where both labels are present and differ."""
    known_a = pd.notna(labels_a)[:, None]
    known_b = pd.notna(labels_b)[None, :]
    differ = labels_a[:, None] != labels_b[None, :]
    return known_a & known_b & differ


def match_frames(
    indices_a: list[int],
    indices_b: list[int],
    features: np.ndarray,
    max_cost: float,
    labels: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], int]:
    """
    Match detections in frame_a against detections in frame_b using
    the Hungarian algorithm on the cdist cost matrix.

    Parameters
    ----------
    indices_a, indices_b : row indices into `features`
    features : (N, D) normalized feature array
    max_cost : pairs with cost > max_cost are not linked
    labels   : optional per-row class labels; pairs whose labels differ are
               never linked

    Returns the (idx_a, idx_b) pairs matched below max_cost, and the number
    of within-max_cost candidate pairs rejected for differing labels.
    """
    feat_a = features[indices_a]
    feat_b = features[indices_b]

    cost_matrix = cdist(feat_a, feat_b, metric='euclidean')

    n_label_rejects = 0
    if labels is not None:
        mismatch = label_mismatch(labels[indices_a], labels[indices_b])
        n_label_rejects = int((mismatch & (cost_matrix <= max_cost)).sum())
        # Finite so linear_sum_assignment stays feasible; still above max_cost.
        cost_matrix[mismatch] = max_cost + 1e6

    # linear_sum_assignment minimizes total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= max_cost:
            matches.append((indices_a[r], indices_b[c]))

    return matches, n_label_rejects


# ---------------------------------------------------------------------------
# Main de-duplication pipeline
# ---------------------------------------------------------------------------

def deduplicate(
    df: pd.DataFrame,
    sigma_xy: float = 300.0,
    sigma_depth: float = 5.0,
    sigma_time: float = 120.0,
    max_cost: float = 1.5,
    frame_gap_s: float = 2.0,
    time_gate_s: float = 600.0,
    use_time_in_cost: bool = False,
    label_col: str | None = None,
) -> pd.DataFrame:
    """
    Run Hungarian de-duplication.

    Strategy — two passes
    ---------------------
    Pass 1 (within-frame): detections in the same imaging burst that are
      within max_cost of each other in normalised [cx, cy, depth] space are
      merged as duplicates via exhaustive pairwise comparison.
    Pass 2 (cross-frame): for each pair of bursts whose midpoints are within
      time_gate_s of each other, build a cost matrix and run
      linear_sum_assignment (Hungarian) to find the globally optimal bijective
      matching; pairs below max_cost are merged.
    Union-Find propagates track identity through all matches.
    Returns df with track_id, canonical_uuid, is_duplicate columns.

    Parameters
    ----------
    sigma_xy    : spatial scale (pixels) — detections within ~1σ are "close"
    sigma_depth : depth scale (meters)
    sigma_time  : time scale (seconds) for optional temporal cost term
    max_cost    : normalized-distance threshold for accepting a match
    frame_gap_s : time gap (s) that separates imaging bursts
    time_gate_s : only compare frame pairs whose midpoints are ≤ this apart
    use_time_in_cost : include |Δt|/σ_time in the feature vector
    label_col   : optional class-label column (e.g. dinov3_v32_v3); detections
                  with different labels are never merged. Missing labels
                  do not block a match.
    """
    df = df.copy()

    # Fill missing depth with median so it doesn't dominate the cost
    depth_median = df['depth'].median() if df['depth'].notna().any() else 0.0
    df['depth_filled'] = df['depth'].fillna(depth_median)

    # Assign imaging-burst frame IDs
    df = assign_frames(df, frame_gap_s=frame_gap_s)

    n = len(df)
    uf = UnionFind(n)

    # Build normalized feature matrix
    if use_time_in_cost:
        features = np.column_stack([
            df['cx'].values / sigma_xy,
            df['cy'].values / sigma_xy,
            df['depth_filled'].values / sigma_depth,
            df['time_s'].values / sigma_time,
        ])
    else:
        features = np.column_stack([
            df['cx'].values / sigma_xy,
            df['cy'].values / sigma_xy,
            df['depth_filled'].values / sigma_depth,
        ])

    # Group row indices by frame
    frame_groups: dict[int, list[int]] = {}
    for idx, fid in enumerate(df['frame_id']):
        frame_groups.setdefault(fid, []).append(idx)

    frame_ids_sorted = sorted(frame_groups.keys())

    # Representative time for each frame (median of its detections)
    frame_times = {
        fid: df.loc[frame_groups[fid], 'time_s'].median()
        for fid in frame_ids_sorted
    }

    labels = df[label_col].to_numpy(dtype=object) if label_col else None

    within_matches = 0
    cross_pairs = 0
    cross_matches = 0
    label_rejects = 0

    # --- Pass 1: within-frame duplicates ---
    # Two detections in the same burst are duplicates if their normalised
    # distance is below max_cost.  We use exhaustive pairwise comparison
    # (not bijective Hungarian) because both detections survive the burst;
    # we just need to identify which ones overlap.
    for fid in frame_ids_sorted:
        idx_list = frame_groups[fid]
        if len(idx_list) < 2:
            continue
        feat = features[idx_list]
        dists = cdist(feat, feat, metric='euclidean')
        if labels is not None:
            mismatch = label_mismatch(labels[idx_list], labels[idx_list])
            label_rejects += int(np.triu(mismatch & (dists <= max_cost), k=1).sum())
            dists[mismatch] = np.inf
        for ii in range(len(idx_list)):
            for jj in range(ii + 1, len(idx_list)):
                if dists[ii, jj] <= max_cost:
                    uf.union(idx_list[ii], idx_list[jj])
                    within_matches += 1

    # --- Pass 2: cross-frame duplicates (Hungarian) ---
    # For each pair of bursts within time_gate_s, find the globally
    # optimal bijective matching using linear_sum_assignment.
    for i, fid_a in enumerate(frame_ids_sorted):
        for fid_b in frame_ids_sorted[i + 1:]:
            dt = abs(frame_times[fid_b] - frame_times[fid_a])
            if dt > time_gate_s:
                break

            idx_a = frame_groups[fid_a]
            idx_b = frame_groups[fid_b]
            cross_pairs += 1

            matches, n_rejects = match_frames(idx_a, idx_b, features, max_cost, labels)
            for a, b in matches:
                uf.union(a, b)
            cross_matches += len(matches)
            label_rejects += n_rejects

    print(f"  Within-frame: {within_matches} duplicate pair(s) merged.")
    print(f"  Cross-frame:  compared {cross_pairs} burst pair(s), "
          f"found {cross_matches} match(es).")
    if labels is not None:
        print(f"  Label gate:   rejected {label_rejects} close pair(s) "
              f"with different {label_col} labels.")

    # Assign track IDs (0-based, ordered by first appearance)
    raw_roots = [uf.find(i) for i in range(n)]
    unique_roots = dict.fromkeys(raw_roots)  # preserves insertion order
    root_to_track = {r: tid for tid, r in enumerate(unique_roots)}
    df['track_id'] = [root_to_track[r] for r in raw_roots]

    # Canonical detection = first-seen in each track
    canonical = df.groupby('track_id')['uuid'].first().rename('canonical_uuid')
    df = df.merge(canonical, on='track_id')
    df['is_duplicate'] = df['uuid'] != df['canonical_uuid']

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        'input_path',
        nargs='?',
        default='ptvr_lm/localizations.csv',
        help='Path to localizations.csv or a CFE lab .parquet file '
             '(default: ptvr_lm/localizations.csv)',
    )
    p.add_argument(
        '--base-path',
        default=None,
        help='Directory joined onto each relative filename to form the full '
             'image path. Example: '
             '/mnt/DeepSea-AI/data/Planktivore/raw/'
             '2026_April_20_Ahi-Planktivore/low_mag_cam/',
    )
    p.add_argument(
        '-o', '--output',
        default=None,
        help='Output path; written as parquet when it ends in .parquet, '
             'otherwise CSV. Default: <input_stem>_dedup.csv',
    )
    p.add_argument(
        '--sigma-xy',
        type=float, default=300.0,
        help='Spatial scale in pixels (default: 300). Detections within '
             '~1 sigma_xy pixels are considered spatially close.',
    )
    p.add_argument(
        '--sigma-depth',
        type=float, default=5.0,
        help='Depth scale in meters (default: 5). Detections within '
             '~1 sigma_depth metres are considered depth-close.',
    )
    p.add_argument(
        '--sigma-time',
        type=float, default=120.0,
        help='Time scale in seconds (default: 120), used only with '
             '--use-time-in-cost.',
    )
    p.add_argument(
        '--max-cost',
        type=float, default=1.5,
        help='Maximum normalised distance to accept a Hungarian match '
             '(default: 1.5). Lower = stricter de-duplication.',
    )
    p.add_argument(
        '--frame-gap',
        type=float, default=2.0,
        help='Time gap in seconds that separates imaging bursts '
             '(default: 2.0 s).',
    )
    p.add_argument(
        '--time-gate',
        type=float, default=600.0,
        help='Maximum seconds between burst midpoints to attempt matching '
             '(default: 600 s = 10 min).',
    )
    p.add_argument(
        '--use-time-in-cost',
        action='store_true',
        help='Include Δtime / sigma_time in the cost function.',
    )
    p.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print matched pairs.',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    input_path = args.input_path
    if not Path(input_path).exists():
        sys.exit(f"Error: input not found: {input_path}")

    output_path = args.output or str(
        Path(input_path).with_name(Path(input_path).stem + '_dedup.csv')
    )

    print(f"Loading {input_path} ...")
    if args.base_path:
        print(f"  base-path: {args.base_path}")
    df = load_localizations(input_path, base_path=args.base_path)
    print(f"  {len(df)} detections loaded.")

    depth_valid = df['depth'].notna().sum()
    print(f"  {depth_valid}/{len(df)} detections have depth values "
          f"(others filled with median={df['depth'].median():.2f} m).")

    df = assign_frames(df, frame_gap_s=args.frame_gap)
    n_frames = df['frame_id'].nunique()
    print(f"  {n_frames} imaging burst(s) detected "
          f"(frame_gap={args.frame_gap} s).")

    label_col = find_dinov3_label_column(df.columns)
    if label_col:
        print(f"  Label gate on: only detections with the same {label_col} label are merged.")
    else:
        print("  Label gate skipped: no dinov3 label column.")

    print("\nRunning Hungarian de-duplication ...")
    print(f"  sigma_xy={args.sigma_xy} px  sigma_depth={args.sigma_depth} m  "
          f"max_cost={args.max_cost}  time_gate={args.time_gate} s")

    result = deduplicate(
        df,
        sigma_xy=args.sigma_xy,
        sigma_depth=args.sigma_depth,
        sigma_time=args.sigma_time,
        max_cost=args.max_cost,
        frame_gap_s=args.frame_gap,
        time_gate_s=args.time_gate,
        use_time_in_cost=args.use_time_in_cost,
        label_col=label_col,
    )

    n_tracks = result['track_id'].nunique()
    n_dups = result['is_duplicate'].sum()
    print(f"\nResults:")
    print(f"  {len(result)} detections → {n_tracks} unique track(s)")
    print(f"  {n_dups} duplicate(s) identified "
          f"({n_dups / len(result) * 100:.1f}%)")

    if args.verbose:
        dup_df = result[result['is_duplicate']][
            ['media', 'track_id', 'canonical_uuid', 'cx', 'cy',
             'depth', 'time_s', 'frame_id']
        ]
        if len(dup_df):
            print("\nDuplicate detections:")
            print(dup_df.to_string(index=False))
        else:
            print("\nNo duplicates found with current thresholds.")

    # Show per-track summary
    track_counts = result.groupby('track_id').size()
    multi = (track_counts > 1).sum()
    if multi:
        print(f"\n  {multi} track(s) have >1 detection (true duplicates):")
        top = track_counts[track_counts > 1].sort_values(ascending=False).head(10)
        for tid, cnt in top.items():
            first_uuid = result.loc[result['track_id'] == tid, 'canonical_uuid'].iloc[0]
            print(f"    track {tid:4d}: {cnt} detections  canonical={first_uuid}")

    # Save output
    preferred = [
        'media', 'image_path', 'filename', 'uuid', 'Label', 'depth',
        'iso_datetime', 'time', 'epoch_seconds',
        'cx', 'cy', 'box_x', 'box_y', 'box_w', 'box_h',
        'time_s', 'frame_id', 'track_id', 'canonical_uuid', 'is_duplicate',
        'latitude', 'longitude', 'score',
    ]
    if Path(input_path).suffix.lower() == '.parquet':
        hidden = {'depth_filled'}
        output_cols = [c for c in preferred if c in result.columns]
        output_cols += [c for c in result.columns if c not in output_cols and c not in hidden]
    else:
        output_cols = [c for c in preferred if c in result.columns]
    if Path(output_path).suffix.lower() == '.parquet':
        result[output_cols].to_parquet(output_path, index=False)
    else:
        result[output_cols].to_csv(output_path, index=False)
    print(f"\nOutput written to: {output_path}")


if __name__ == '__main__':
    main()
