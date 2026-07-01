#!/usr/bin/env python3
"""
Hungarian-based de-duplication of particle detections.

Detections of the same physical particle across different time bursts are
identified by matching on box center (cx, cy), depth, and time using the
Hungarian algorithm (linear_sum_assignment) with a normalized Euclidean cost.

Filename format:
  low_mag_cam-{timestamp_us}-{session}-{...}-{x}-{y}-{w}-{h}_rawcolor.jpg
  Last 4 dash-separated numbers before _rawcolor are x, y, w, h (pixels).

Depth and label metadata come from localizations.csv.
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
    # and all of the last 4 + parts[1] must be integers
    if len(parts) < 6:
        return None
    # Format: …-{y}-{x}-{h}-{w}-{ignored}_rawcolor.ext
    # parts[-5]=y, parts[-4]=x, parts[-3]=h, parts[-2]=w; parts[-1] is discarded.
    try:
        y, x, h, w = int(parts[-5]), int(parts[-4]), int(parts[-3]), int(parts[-2])
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

def load_localizations(csv_path: str) -> pd.DataFrame:
    """
    Load localizations.csv and enrich with parsed filename fields.

    The CSV may have two columns both named 'iso_datetime'; pandas will
    rename the second one to 'iso_datetime.1'. We prefer the UTC one
    (contains '+00:00').
    """
    df = pd.read_csv(csv_path)

    # Parse filenames; rows that don't match the pattern are dropped
    parsed_raw = df['media'].apply(parse_filename)
    unparseable = parsed_raw.isna()
    if unparseable.any():
        n_skip = unparseable.sum()
        skipped = df.loc[unparseable, 'media'].tolist()
        print(f"  Warning: skipping {n_skip} row(s) with non-standard filenames:")
        for s in skipped[:10]:
            print(f"    {s}")
        if n_skip > 10:
            print(f"    … and {n_skip - 10} more")
        df = df[~unparseable].reset_index(drop=True)
        parsed_raw = parsed_raw[~unparseable].reset_index(drop=True)

    parsed = parsed_raw.apply(pd.Series)
    df = pd.concat([df.reset_index(drop=True), parsed], axis=1)

    # Choose the UTC datetime column
    if 'iso_datetime.1' in df.columns:
        df['datetime_utc'] = pd.to_datetime(df['iso_datetime.1'], utc=True, errors='coerce')
    else:
        df['datetime_utc'] = pd.to_datetime(df['iso_datetime'], utc=True, errors='coerce')

    df = df.sort_values('time_s').reset_index(drop=True)
    return df


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

def match_frames(
    indices_a: list[int],
    indices_b: list[int],
    features: np.ndarray,
    max_cost: float,
) -> list[tuple[int, int]]:
    """
    Match detections in frame_a against detections in frame_b using
    the Hungarian algorithm on the cdist cost matrix.

    Parameters
    ----------
    indices_a, indices_b : row indices into `features`
    features : (N, D) normalized feature array
    max_cost : pairs with cost > max_cost are not linked

    Returns list of (idx_a, idx_b) pairs that are matched below max_cost.
    """
    feat_a = features[indices_a]
    feat_b = features[indices_b]

    cost_matrix = cdist(feat_a, feat_b, metric='euclidean')

    # linear_sum_assignment minimizes total cost
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= max_cost:
            matches.append((indices_a[r], indices_b[c]))

    return matches


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

    within_matches = 0
    cross_pairs = 0
    cross_matches = 0

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

            matches = match_frames(idx_a, idx_b, features, max_cost)
            for a, b in matches:
                uf.union(a, b)
            cross_matches += len(matches)

    print(f"  Within-frame: {within_matches} duplicate pair(s) merged.")
    print(f"  Cross-frame:  compared {cross_pairs} burst pair(s), "
          f"found {cross_matches} match(es).")

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
        'csv_path',
        nargs='?',
        default='ptvr_lm/localizations.csv',
        help='Path to localizations.csv (default: ptvr_lm/localizations.csv)',
    )
    p.add_argument(
        '-o', '--output',
        default=None,
        help='Output CSV path. Default: <input_stem>_dedup.csv',
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

    csv_path = args.csv_path
    if not Path(csv_path).exists():
        sys.exit(f"Error: CSV not found: {csv_path}")

    output_path = args.output or (
        str(Path(csv_path).with_stem(Path(csv_path).stem + '_dedup'))
    )

    print(f"Loading {csv_path} ...")
    df = load_localizations(csv_path)
    print(f"  {len(df)} detections loaded.")

    depth_valid = df['depth'].notna().sum()
    print(f"  {depth_valid}/{len(df)} detections have depth values "
          f"(others filled with median={df['depth'].median():.2f} m).")

    df = assign_frames(df, frame_gap_s=args.frame_gap)
    n_frames = df['frame_id'].nunique()
    print(f"  {n_frames} imaging burst(s) detected "
          f"(frame_gap={args.frame_gap} s).")

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
    output_cols = [
        'media', 'uuid', 'Label', 'depth', 'iso_datetime',
        'cx', 'cy', 'box_x', 'box_y', 'box_w', 'box_h',
        'time_s', 'frame_id', 'track_id', 'canonical_uuid', 'is_duplicate',
        'latitude', 'longitude', 'score',
    ]
    output_cols = [c for c in output_cols if c in result.columns]
    result[output_cols].to_csv(output_path, index=False)
    print(f"\nOutput written to: {output_path}")


if __name__ == '__main__':
    main()
