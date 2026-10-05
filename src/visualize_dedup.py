#!/usr/bin/env python3
"""
Visualize Hungarian de-duplication results as an animated GIF.

Frames:
  0     — Overview scatter plot: all detections in (cx, cy) space,
           depth-coloured, duplicate pairs connected by coloured lines.
  1…N   — One frame per duplicate track: crop thumbnails side-by-side,
           metadata panel, and an inset showing the pair location on the map.

Usage:
    python3 visualize_dedup.py [--dedup ptvr_lm/localizations_dedup.csv]
                               [--crops ptvr_lm/crops/Velella_velella]
                               [--out dedup_visualization.gif]

    # CFE lab parquet output: each ROI image is the crop, at base-path/filename
    python3 visualize_dedup.py --dedup tiny_dedup.parquet
                               --base-path /Volumes/DeepSea-AI/.../low_mag_cam/
                               [--out dedup_visualization.gif]
"""

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
    '#fabed4', '#469990',
]
BG_COLOR    = '#0d1117'
TEXT_COLOR  = '#e6edf3'
GRID_COLOR  = '#21262d'
DIM_COLOR   = '#484f58'
ACCENT      = '#58a6ff'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dedup(path: str) -> pd.DataFrame:
    if Path(path).suffix.lower() == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    df['depth']  = pd.to_numeric(df['depth'],  errors='coerce')
    return df


def crop_candidates(row: pd.Series, crops_dir: Path | None,
                    base_path: str | None) -> list[Path]:
    """Possible crop locations for a detection, in priority order.

    Parquet rows: base_path / filename, then the stored image_path.
    CSV rows: crops_dir / <uuid>.{jpg,jpeg,png}.
    """
    paths: list[Path] = []
    filename = row.get('filename')
    if base_path and isinstance(filename, str):
        paths.append(Path(base_path) / filename)
    image_path = row.get('image_path')
    if isinstance(image_path, str):
        paths.append(Path(image_path))
    uuid = row.get('uuid')
    if crops_dir is not None and isinstance(uuid, str):
        paths.extend(crops_dir / (uuid + ext) for ext in ('.jpg', '.jpeg', '.png'))
    return paths


def load_crop(row: pd.Series, crops_dir: Path | None,
              base_path: str | None) -> Image.Image | None:
    candidates = crop_candidates(row, crops_dir, base_path)
    for p in candidates:
        if p.exists():
            return Image.open(p).convert('RGB')
    tried = ', '.join(str(p) for p in candidates[:2]) or 'no candidate paths'
    print(f"    Warning: crop not found (tried {tried})")
    return None


def thumb(img: Image.Image, size: int = 256) -> Image.Image:
    """Resize to square thumbnail, preserving aspect ratio with padding."""
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new('RGB', (size, size), (20, 20, 20))
    ox = (size - img.width)  // 2
    oy = (size - img.height) // 2
    canvas.paste(img, (ox, oy))
    return canvas


def fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return Image.open(buf).copy()


def timedelta_str(dt_s: float) -> str:
    dt_s = abs(dt_s)
    if dt_s < 60:
        return f"{dt_s:.1f} s"
    elif dt_s < 3600:
        return f"{dt_s/60:.1f} min"
    else:
        return f"{dt_s/3600:.2f} h"


# ---------------------------------------------------------------------------
# Frame 0: Overview scatter
# ---------------------------------------------------------------------------

def make_overview(df: pd.DataFrame, dup_tracks: list[int]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    depth_vals = df['depth'].values
    depth_min  = np.nanmin(depth_vals)
    depth_max  = np.nanmax(depth_vals)

    # All non-duplicate detections — dim dots
    non_dup = df[~df['is_duplicate'] & ~df['track_id'].isin(dup_tracks)]
    sc = ax.scatter(
        non_dup['cx'], non_dup['cy'],
        c=non_dup['depth'].fillna(depth_min),
        cmap='viridis', vmin=depth_min, vmax=depth_max,
        s=35, alpha=0.55, linewidths=0, zorder=2,
    )

    # Draw each duplicate track with a distinct colour + connecting line
    for i, tid in enumerate(dup_tracks):
        color = PALETTE[i % len(PALETTE)]
        group = df[df['track_id'] == tid].sort_values('time_s')
        xs, ys = group['cx'].values, group['cy'].values

        # Line connecting the detections
        ax.plot(xs, ys, '-', color=color, linewidth=1.8,
                alpha=0.85, zorder=3)

        # Canonical (first) detection — filled star
        ax.scatter(xs[0], ys[0], s=160, marker='*', color=color,
                   edgecolors='white', linewidths=0.6, zorder=5,
                   label=f'Track {tid}')

        # Duplicate detections — open circle
        ax.scatter(xs[1:], ys[1:], s=80, marker='o', facecolors='none',
                   edgecolors=color, linewidths=1.8, zorder=5)

        # Label near the midpoint
        mx, my = xs.mean(), ys.mean()
        ax.annotate(f'T{tid}', (mx, my),
                    xytext=(6, 6), textcoords='offset points',
                    color=color, fontsize=8, fontweight='bold', zorder=6)

    cbar = fig.colorbar(sc, ax=ax, pad=0.01, shrink=0.85)
    cbar.set_label('Depth (m)', color=TEXT_COLOR, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)
    cbar.outline.set_edgecolor(GRID_COLOR)

    ax.set_xlabel('Box centre  cx  (px)', color=TEXT_COLOR, fontsize=10)
    ax.set_ylabel('Box centre  cy  (px)', color=TEXT_COLOR, fontsize=10)
    ax.set_title('Particle detection map — duplicate tracks highlighted',
                 color=TEXT_COLOR, fontsize=12, pad=12)

    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5)

    # Legend
    star = Line2D([0], [0], marker='*', color='w', markerfacecolor='w',
                  markersize=10, label='Canonical detection', linestyle='None')
    circ = Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                  markeredgecolor='w', markersize=8,
                  label='Duplicate detection', linestyle='None')
    line = Line2D([0], [0], color='w', linewidth=1.5, label='Matched pair')
    leg = ax.legend(handles=[star, circ, line],
                    facecolor='#161b22', edgecolor=GRID_COLOR,
                    labelcolor=TEXT_COLOR, fontsize=8, loc='upper right')

    n_det = len(df)
    n_dup = df['is_duplicate'].sum()
    ax.text(0.01, 0.01,
            f'{n_det} detections  •  {n_dup} duplicate(s)  •  '
            f'{len(dup_tracks)} track(s) with repeats',
            transform=ax.transAxes, color=DIM_COLOR, fontsize=8, va='bottom')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-pair frames
# ---------------------------------------------------------------------------

def make_pair_frame(
    df: pd.DataFrame,
    tid: int,
    pair_idx: int,
    n_pairs: int,
    crops_dir: Path | None,
    all_dup_tracks: list[int],
    base_path: str | None = None,
) -> plt.Figure:
    group = df[df['track_id'] == tid].sort_values('time_s').reset_index(drop=True)
    canonical = group.iloc[0]
    duplicate = group.iloc[1]

    dt   = duplicate['time_s'] - canonical['time_s']
    ddep = (duplicate['depth'] - canonical['depth']
            if pd.notna(canonical['depth']) and pd.notna(duplicate['depth'])
            else float('nan'))
    dcx  = duplicate['cx'] - canonical['cx']
    dcy  = duplicate['cy'] - canonical['cy']
    dist_px = np.sqrt(dcx**2 + dcy**2)

    color = PALETTE[pair_idx % len(PALETTE)]

    # Layout: [crop_A | crop_B | metadata] top row
    #         [              map          ] bottom row
    fig = plt.figure(figsize=(14, 8), facecolor=BG_COLOR)
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1.6, 1],
        hspace=0.35, wspace=0.3,
    )

    ax_img_a = fig.add_subplot(gs[0, 0])
    ax_img_b = fig.add_subplot(gs[0, 1])
    ax_meta  = fig.add_subplot(gs[0, 2])
    ax_map   = fig.add_subplot(gs[1, :])

    for ax in (ax_img_a, ax_img_b, ax_meta, ax_map):
        ax.set_facecolor(BG_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)

    # --- Crop A (canonical) ---
    img_a = load_crop(canonical, crops_dir, base_path)
    if img_a:
        ax_img_a.imshow(thumb(img_a, 240), aspect='auto')
    else:
        ax_img_a.text(0.5, 0.5, 'image\nnot found', transform=ax_img_a.transAxes,
                      ha='center', va='center', color=DIM_COLOR)
    ax_img_a.set_title('Canonical (first seen)', color=TEXT_COLOR, fontsize=9, pad=4)
    ax_img_a.tick_params(left=False, bottom=False,
                         labelleft=False, labelbottom=False)
    ax_img_a.set_xlabel(
        f"Frame {int(canonical['frame_id'])}  •  t = {canonical['time_s']:.0f} s",
        color=DIM_COLOR, fontsize=7,
    )

    # --- Crop B (duplicate) ---
    img_b = load_crop(duplicate, crops_dir, base_path)
    if img_b:
        ax_img_b.imshow(thumb(img_b, 240), aspect='auto')
    else:
        ax_img_b.text(0.5, 0.5, 'image\nnot found', transform=ax_img_b.transAxes,
                      ha='center', va='center', color=DIM_COLOR)
    ax_img_b.set_title('Duplicate', color=TEXT_COLOR, fontsize=9, pad=4)
    ax_img_b.tick_params(left=False, bottom=False,
                         labelleft=False, labelbottom=False)
    ax_img_b.set_xlabel(
        f"Frame {int(duplicate['frame_id'])}  •  t = {duplicate['time_s']:.0f} s",
        color=DIM_COLOR, fontsize=7,
    )

    # --- Metadata panel ---
    ax_meta.axis('off')
    ax_meta.set_title(f'Track {tid}  ({pair_idx+1}/{n_pairs})',
                      color=color, fontsize=11, fontweight='bold', pad=6)

    rows = [
        ('Δ time',   timedelta_str(dt)),
        ('Δ depth',  f'{ddep:+.2f} m' if not np.isnan(ddep) else 'n/a'),
        ('Δ cx',     f'{dcx:+.1f} px'),
        ('Δ cy',     f'{dcy:+.1f} px'),
        ('Δ spatial',f'{dist_px:.1f} px'),
        ('',         ''),
        ('cx (canonical)', f'{canonical["cx"]:.0f} px'),
        ('cy (canonical)', f'{canonical["cy"]:.0f} px'),
        ('depth (canonical)', f'{canonical["depth"]:.2f} m'
                               if pd.notna(canonical['depth']) else 'n/a'),
        ('',         ''),
        ('cx (dup)', f'{duplicate["cx"]:.0f} px'),
        ('cy (dup)', f'{duplicate["cy"]:.0f} px'),
        ('depth (dup)', f'{duplicate["depth"]:.2f} m'
                         if pd.notna(duplicate['depth']) else 'n/a'),
    ]
    if 'Label' in df.columns:
        rows += [
            ('',         ''),
            ('label (canonical)', str(canonical['Label'])),
            ('label (dup)',       str(duplicate['Label'])),
        ]

    n_text = sum(1 for label, _ in rows if label)
    n_gap = len(rows) - n_text
    step = min(0.075, (0.95 - 0.04 * n_gap) / n_text)
    y = 0.97
    for label, val in rows:
        if label == '':
            y -= 0.04
            continue
        ax_meta.text(0.05, y, label, transform=ax_meta.transAxes,
                     color=DIM_COLOR, fontsize=8, va='top')
        ax_meta.text(0.95, y, val,   transform=ax_meta.transAxes,
                     color=TEXT_COLOR, fontsize=8, va='top', ha='right',
                     fontweight='bold')
        y -= step

    # --- Mini-map ---
    depth_vals = df['depth'].values
    depth_min  = np.nanmin(depth_vals)
    depth_max  = np.nanmax(depth_vals)

    non_focus = df[~df['track_id'].isin(all_dup_tracks)]
    ax_map.scatter(non_focus['cx'], non_focus['cy'],
                   c=non_focus['depth'].fillna(depth_min),
                   cmap='viridis', vmin=depth_min, vmax=depth_max,
                   s=18, alpha=0.35, linewidths=0, zorder=2)

    # Other dup tracks (dim)
    for other_tid in all_dup_tracks:
        if other_tid == tid:
            continue
        g = df[df['track_id'] == other_tid]
        ax_map.scatter(g['cx'], g['cy'], s=22, color=DIM_COLOR,
                       alpha=0.5, linewidths=0, zorder=3)

    # Current pair (bright)
    xs = group['cx'].values
    ys = group['cy'].values
    ax_map.plot(xs, ys, '-', color=color, linewidth=2, zorder=4)
    ax_map.scatter(xs[0], ys[0], s=180, marker='*', color=color,
                   edgecolors='white', linewidths=0.8, zorder=5, label='Canonical')
    ax_map.scatter(xs[1:], ys[1:], s=90, marker='o', facecolors='none',
                   edgecolors=color, linewidths=2, zorder=5, label='Duplicate')

    # Zoom in to ±500 px around the pair, clamped to data range
    pad = 600
    cx_lo  = max(df['cx'].min() - 50, min(xs) - pad)
    cx_hi  = min(df['cx'].max() + 50, max(xs) + pad)
    cy_lo  = max(df['cy'].min() - 50, min(ys) - pad)
    cy_hi  = min(df['cy'].max() + 50, max(ys) + pad)
    ax_map.set_xlim(cx_lo, cx_hi)
    ax_map.set_ylim(cy_lo, cy_hi)

    ax_map.set_xlabel('cx (px)', color=TEXT_COLOR, fontsize=9)
    ax_map.set_ylabel('cy (px)', color=TEXT_COLOR, fontsize=9)
    ax_map.set_title('Position in frame space', color=TEXT_COLOR, fontsize=9)
    ax_map.tick_params(colors=TEXT_COLOR, labelsize=7)
    ax_map.grid(True, color=GRID_COLOR, linewidth=0.4)
    ax_map.legend(facecolor='#161b22', edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR, fontsize=7, loc='upper right')

    fig.suptitle(
        f'De-duplication  —  duplicate pair {pair_idx+1} of {n_pairs}',
        color=TEXT_COLOR, fontsize=13, y=1.01,
    )

    return fig


# ---------------------------------------------------------------------------
# Assemble GIF
# ---------------------------------------------------------------------------

def build_gif(frames: list[Image.Image], output_path: str,
              hold_overview_ms: int = 3000, hold_pair_ms: int = 2500) -> None:
    durations = [hold_overview_ms] + [hold_pair_ms] * (len(frames) - 1)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dedup',  default='ptvr_lm/localizations_dedup.csv',
                   help='De-dup CSV or parquet produced by hungarian_dedup.py')
    p.add_argument('--crops',  default='ptvr_lm/crops/Velella_velella',
                   help='Directory of crop images named <uuid>.jpg (CSV input)')
    p.add_argument('--base-path', default=None,
                   help='Directory joined onto the filename column to locate '
                        'each crop (parquet input). Falls back to image_path.')
    p.add_argument('--out',    default='dedup_visualization.gif',
                   help='Output GIF path')
    p.add_argument('--dpi',    type=int, default=100)
    return p.parse_args()


def main() -> None:
    args  = parse_args()
    crops = Path(args.crops)

    print(f"Loading {args.dedup} ...")
    df = load_dedup(args.dedup)

    # Identify tracks with more than one detection
    track_counts = df.groupby('track_id').size()
    dup_tracks   = sorted(track_counts[track_counts > 1].index.tolist())

    if not dup_tracks:
        print("No duplicate tracks found — nothing to visualize.")
        return

    print(f"Found {len(dup_tracks)} duplicate track(s): {dup_tracks}")

    gif_frames: list[Image.Image] = []

    # --- Frame 0: overview ---
    print("Rendering overview frame ...")
    fig0 = make_overview(df, dup_tracks)
    gif_frames.append(fig_to_pil(fig0))
    plt.close(fig0)

    # --- One frame per duplicate pair ---
    for i, tid in enumerate(dup_tracks):
        print(f"  Rendering pair frame {i+1}/{len(dup_tracks)}  (track {tid}) ...")
        fig = make_pair_frame(df, tid, i, len(dup_tracks), crops, dup_tracks,
                              base_path=args.base_path)
        gif_frames.append(fig_to_pil(fig))
        plt.close(fig)

    # Resize all frames to the same size (use the largest)
    max_w = max(f.width  for f in gif_frames)
    max_h = max(f.height for f in gif_frames)
    gif_frames = [
        f.resize((max_w, max_h), Image.LANCZOS) if (f.width, f.height) != (max_w, max_h)
        else f
        for f in gif_frames
    ]

    # Convert to palette mode for smaller GIF
    gif_frames_p = [f.convert('P', palette=Image.ADAPTIVE, colors=256)
                    for f in gif_frames]

    print(f"Writing GIF ({len(gif_frames)} frames) → {args.out} ...")
    build_gif(gif_frames_p, args.out)
    print("Done.")


if __name__ == '__main__':
    main()
