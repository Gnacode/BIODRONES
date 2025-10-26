#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DK-Rend-4e (fixed) — Clean satellite basemap PNG renderer for OpenDrift frame GeoJSONs.

- Renders Esri World Imagery (satellite only, no washed-out label overlay).
- Locks aspect so Denmark is not stretched in X.
- Uses true lon/lat tick labels even though the basemap is Web Mercator.
- Adds frame index and timestamp in the title.
- Colors particles by `source`.

Inputs (per-frame GeoJSON from the simulation exporter):
  frames/frame_####_points.geojson
  frames/frame_####_tails.geojson   (optional)

Outputs:
  frames_png/frame_####.png

Usage:
  python DK-Rend-4e.py \
    --frames_dir y:/RGXSIM/frames \
    --out_dir    y:/RGXSIM/frames_png \
    --width      1920 \
    --height     1080 \
    --dpi        120
"""

from __future__ import annotations
import os, json, glob, argparse, math, warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

# =========================
# CONSTANTS (easy to tweak)
# =========================
BASE_DIR = Path(__file__).resolve().parent
FRAMES_DIR_DEFAULT   = str(BASE_DIR / "frames")
OUT_DIR_DEFAULT      = str(BASE_DIR / "frames_png")

POINTS_GLOB          = "frame_*_points.geojson"
TAILS_TEMPLATE       = "frame_{:04d}_tails.geojson"   # optional

# Output PNG resolution in *pixels*
FIG_WIDTH_PX         = 1920
FIG_HEIGHT_PX        = 1080
DPI                  = 120

# Text styling (absolute point sizes, not scaled with DPI)
TITLE_PREFIX         = "OpenDrift Flow"
TITLE_SIZE_PT        = 12
AXIS_LABEL_SIZE_PT   = 10
TICK_SIZE_PT         = 9
LEGEND_SIZE_PT       = 9

# Plot + colors
BG_COLOR             = "white"
FG_COLOR             = "black"

POINT_RADIUS_PX      = 3.0     # visual radius of each particle dot
POINT_ALPHA          = 0.95
POINT_EDGE_COLOR     = None    # e.g. "#222222" for outlined dots
TAIL_LINEWIDTH_PX    = 0.8
TAIL_ALPHA           = 0.75

# How many degree ticks along each axis
DEG_TICKS_N          = 8

# Basemap mode:
#   "imagery"  satellite only (default, sharpest, best visual)
#   "none"     no basemap, just lon/lat grid
BASEMAP_MODE_DEFAULT = "imagery"

# Esri imagery tiles (Web Mercator)
ESRI_WORLD_IMAGERY_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# ==============================
# Optional basemap dependency
# ==============================
_HAS_CTX = False
Transformer = None
try:
    import contextily as cx
    from pyproj import Transformer as _Transformer
    Transformer = _Transformer
    _HAS_CTX = True
except Exception as e:
    warnings.warn(
        f"Contextily / pyproj not available ({e}). "
        "Basemap will be disabled and axes will be plain lon/lat."
    )
    _HAS_CTX = False

# -----------------
# Utility helpers
# -----------------
def _read_geojson(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _frame_index_from_name(path: str) -> Optional[int]:
    # expects .../frame_####_points.geojson
    base = os.path.basename(path)
    try:
        num = base.split("_")[1]
        return int(num)
    except Exception:
        return None

def _scan_extent(frames_dir: str, points_glob: str) -> Tuple[float, float, float, float, float]:
    """
    Scan all point frames to compute overall lon/lat extent and mean lat.
    Returns (minlon, maxlon, minlat, maxlat, mean_lat).
    We pad a little so data isn't glued to the border.
    """
    pts_files = sorted(glob.glob(os.path.join(frames_dir, points_glob)))
    if not pts_files:
        raise RuntimeError("No point frame files found.")

    minlon =  1e9
    maxlon = -1e9
    minlat =  1e9
    maxlat = -1e9
    lat_sum = 0.0
    lat_count = 0

    for p in pts_files:
        gj = _read_geojson(p)
        if not gj or "features" not in gj:
            continue
        for feat in gj["features"]:
            try:
                x, y = feat["geometry"]["coordinates"]
            except Exception:
                continue
            if x is None or y is None:
                continue
            if isinstance(x, float) and math.isnan(x):
                continue
            if isinstance(y, float) and math.isnan(y):
                continue
            minlon = min(minlon, x); maxlon = max(maxlon, x)
            minlat = min(minlat, y); maxlat = max(maxlat, y)
            lat_sum += y; lat_count += 1

    if lat_count == 0:
        raise RuntimeError("No valid coordinates found in frames. Check your frames folder.")

    # pad ~8% so we see coastline around plume
    pad_x = (maxlon - minlon) * 0.08 if maxlon > minlon else 0.5
    pad_y = (maxlat - minlat) * 0.08 if maxlat > minlat else 0.5
    minlon -= pad_x; maxlon += pad_x
    minlat -= pad_y; maxlat += pad_y

    mean_lat = lat_sum / lat_count
    return minlon, maxlon, minlat, maxlat, mean_lat

def _collect_sources(points_gj: dict) -> List[str]:
    """Collect distinct 'source' values from features (e.g. CPH, AAL, BLL)."""
    sources: List[str] = []
    if not points_gj or "features" not in points_gj:
        return sources
    for feat in points_gj["features"]:
        src = feat.get("properties", {}).get("source", "")
        if src and src not in sources:
            sources.append(src)
    return sources

# color palette per source
PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]
FALLBACK_COLOR = "#e6e212"

def _color_for_source(src: str) -> str:
    if not src:
        return FALLBACK_COLOR
    idx = abs(hash(src)) % len(PALETTE)
    return PALETTE[idx]

# degree label formatters
def _fmt_lon(v, pos=None):
    hemi = "E" if v >= 0 else "W"
    return f"{abs(v):.2f}°{hemi}"

def _fmt_lat(v, pos=None):
    hemi = "N" if v >= 0 else "S"
    return f"{abs(v):.2f}°{hemi}"

# convert our desired pixel sizes to matplotlib units
def _px_to_points(px: float, dpi: float) -> float:
    # 1 point = 1/72 inch. pixels_per_inch = dpi.
    # so px pixels = (px / dpi) inches = (px / dpi)*72 points
    return px * 72.0 / dpi

def _scatter_s_from_pixel_radius(radius_px: float, dpi: float) -> float:
    """
    Matplotlib scatter 's' is marker area in points^2.
    We'll interpret POINT_RADIUS_PX as pixel radius.
    """
    r_pt = _px_to_points(radius_px, dpi)
    return (r_pt ** 2)

# =========================
# Basemap plumbing
# =========================
def _get_transformers():
    """
    Returns forward (lon,lat->x,y in EPSG:3857) and inverse (x,y->lon,lat)
    Only valid if contextily/pyproj is available.
    """
    fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    return fwd.transform, inv.transform

def _estimate_zoom(extent_meters: float, fig_width_px: int) -> int:
    """
    Choose a Web Mercator zoom level so the tiles are reasonably detailed
    for the requested output width. Bigger zoom => sharper.
    156543.03392804097 m/px at zoom 0 (equator), halves each zoom.
    """
    if extent_meters <= 0 or fig_width_px <= 0:
        return 6
    target_mpp = extent_meters / float(fig_width_px)
    z_float = math.log2(156543.03392804097 / max(target_mpp, 1e-9))
    # small positive bias for crispness
    z_int = int(math.floor(z_float + 1))
    # clamp to typical provider range
    return max(0, min(19, z_int))

def _apply_lonlat_ticks_on_mercator(ax,
                                    lon_min, lon_max,
                                    lat_min, lat_max,
                                    n=8,
                                    fg="black"):
    """
    We are plotting in EPSG:3857 meters, but we want ticks labeled in degrees.
    Strategy:
    - Pick n evenly spaced lon/lat values in *degrees*
    - Project each to meters to set tick positions
    - Format tick labels as °E/°N
    """
    to3857, _ = _get_transformers()

    # longitude ticks => X axis ticks
    lon_ticks = np.linspace(lon_min, lon_max, n)
    mid_lat   = 0.5 * (lat_min + lat_max)
    x_tickpos = [to3857(lon, mid_lat)[0] for lon in lon_ticks]

    ax.xaxis.set_major_locator(FixedLocator(x_tickpos))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: _fmt_lon(lon_ticks[pos]) if 0 <= pos < len(lon_ticks) else ""
    ))

    # latitude ticks => Y axis ticks
    lat_ticks = np.linspace(lat_min, lat_max, n)
    mid_lon   = 0.5 * (lon_min + lon_max)
    y_tickpos = [to3857(mid_lon, lat)[1] for lat in lat_ticks]

    ax.yaxis.set_major_locator(FixedLocator(y_tickpos))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: _fmt_lat(lat_ticks[pos]) if 0 <= pos < len(lat_ticks) else ""
    ))

    ax.tick_params(axis='both', colors=fg, labelsize=TICK_SIZE_PT)

def _draw_basemap(ax,
                  lon_min, lon_max,
                  lat_min, lat_max,
                  fig_width_px: int,
                  fig_height_px: int,
                  basemap_mode: str) -> Tuple[str, callable]:
    """
    Draw Esri satellite imagery in Web Mercator using contextily.
    Returns:
        backend: "mercator" or "none"
        project_fn: function to project lon/lat -> x/y for plotting
    """
    if basemap_mode == "none" or not _HAS_CTX:
        # No basemap, stick to lon/lat in degrees
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect("auto")
        return "none", (lambda x, y: (x, y))

    # We *do* have contextily and basemap_mode == "imagery"
    to3857, _ = _get_transformers()
    Xmin, Ymin = to3857(lon_min, lat_min)
    Xmax, Ymax = to3857(lon_max, lat_max)

    # lock aspect so map isn't squashed
    ax.set_xlim(Xmin, Xmax)
    ax.set_ylim(Ymin, Ymax)
    ax.set_aspect("equal", adjustable="box")

    # choose a tile zoom based on horizontal extent
    zoom = _estimate_zoom(abs(Xmax - Xmin), fig_width_px)

    try:
        cx.add_basemap(
            ax,
            source=ESRI_WORLD_IMAGERY_URL,   # direct URL to Esri World Imagery
            attribution=False,
            reset_extent=False,
            zoom=zoom,
        )
    except Exception as e:
        warnings.warn(
            f"Adding imagery failed ({e}); falling back to plain lon/lat axes."
        )
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect("auto")
        return "none", (lambda x, y: (x, y))

    def project_fn(lons, lats):
        X, Y = to3857(lons, lats)
        return X, Y

    return "mercator", project_fn

# ------------------------
# Main rendering function
# ------------------------
def render_frames(
    frames_dir: str,
    out_dir: str,
    width: int = FIG_WIDTH_PX,
    height: int = FIG_HEIGHT_PX,
    dpi: int = DPI,
    points_glob: str = POINTS_GLOB,
    tails_template: Optional[str] = TAILS_TEMPLATE,
    bg_color: str = BG_COLOR,
    fg_color: str = FG_COLOR,
    basemap_mode: str = BASEMAP_MODE_DEFAULT,
):
    os.makedirs(out_dir, exist_ok=True)

    # figure out global lon/lat extent
    lon_min, lon_max, lat_min, lat_max, _ = _scan_extent(frames_dir, points_glob)

    # grab all frame_####_points.geojson
    pts_files = sorted(glob.glob(os.path.join(frames_dir, points_glob)))
    if not pts_files:
        raise RuntimeError("No point frame files found.")

    # prep scatter / line sizes in figure units
    scatter_area_pts2 = _scatter_s_from_pixel_radius(POINT_RADIUS_PX, dpi)
    tail_linewidth_pts = _px_to_points(TAIL_LINEWIDTH_PX, dpi)

    # color mapping per source (collect from first frame to seed dict)
    first_gj = _read_geojson(pts_files[0])
    src_color: Dict[str, str] = {
        s: _color_for_source(s) for s in _collect_sources(first_gj)
    }

    for pf in pts_files:
        idx = _frame_index_from_name(pf)
        if idx is None:
            continue

        points_gj = _read_geojson(pf)
        tails_path = os.path.join(
            frames_dir,
            (tails_template.format(idx) if tails_template else "")
        )
        tails_gj = (
            _read_geojson(tails_path)
            if tails_template and os.path.exists(tails_path)
            else None
        )

        # update colors for any new sources this frame
        for s in _collect_sources(points_gj):
            if s not in src_color:
                src_color[s] = _color_for_source(s)

        # timestamp for title (take it from first point feature)
        frame_time = ""
        if points_gj and "features" in points_gj and points_gj["features"]:
            frame_time = points_gj["features"][0].get("properties", {}).get("time", "")

        # --- build figure
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # set font sizes (constant point sizes, not scaled by dpi)
        matplotlib.rcParams.update({
            "font.size":           AXIS_LABEL_SIZE_PT,
            "axes.labelsize":      AXIS_LABEL_SIZE_PT,
            "xtick.labelsize":     TICK_SIZE_PT,
            "ytick.labelsize":     TICK_SIZE_PT,
            "legend.fontsize":     LEGEND_SIZE_PT,
            "axes.titlesize":      TITLE_SIZE_PT,
        })

        # basemap
        backend, project_fn = _draw_basemap(
            ax,
            lon_min, lon_max,
            lat_min, lat_max,
            fig_width_px=width,
            fig_height_px=height,
            basemap_mode=basemap_mode if _HAS_CTX else "none",
        )

        # ticks / labels
        for side in ("bottom", "top", "left", "right"):
            ax.spines[side].set_color(fg_color)

        if backend == "mercator":
            _apply_lonlat_ticks_on_mercator(
                ax,
                lon_min, lon_max,
                lat_min, lat_max,
                n=DEG_TICKS_N,
                fg=fg_color,
            )
            ax.set_xlabel("Longitude", color=fg_color)
            ax.set_ylabel("Latitude",  color=fg_color)
        else:
            # plain lon/lat axis (no basemap)
            xticks = np.linspace(lon_min, lon_max, DEG_TICKS_N)
            yticks = np.linspace(lat_min, lat_max, DEG_TICKS_N)
            ax.xaxis.set_major_locator(FixedLocator(xticks))
            ax.yaxis.set_major_locator(FixedLocator(yticks))
            ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lon))
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_lat))
            ax.tick_params(axis='both', colors=fg_color, labelsize=TICK_SIZE_PT)
            ax.set_xlabel("Longitude", color=fg_color)
            ax.set_ylabel("Latitude",  color=fg_color)

        # --- draw tails first so points sit on top
        if tails_gj and "features" in tails_gj:
            for feat in tails_gj["features"]:
                geom = feat.get("geometry", {})
                props = feat.get("properties", {})
                src = props.get("source", "")
                color = src_color.get(src, FALLBACK_COLOR)

                if geom.get("type") == "LineString":
                    coords = geom.get("coordinates", [])
                    if coords:
                        xs = np.array([c[0] for c in coords])
                        ys = np.array([c[1] for c in coords])
                        if backend == "mercator":
                            xs, ys = project_fn(xs, ys)
                        ax.plot(
                            xs, ys,
                            linewidth=tail_linewidth_pts,
                            alpha=TAIL_ALPHA,
                            color=color,
                        )
                elif geom.get("type") == "Point":
                    try:
                        x, y = geom.get("coordinates", [None, None])
                        if x is not None and y is not None:
                            if backend == "mercator":
                                x, y = project_fn(x, y)
                            ax.plot(
                                [x], [y],
                                marker=".",
                                color=color,
                                alpha=TAIL_ALPHA,
                                markersize=_px_to_points(POINT_RADIUS_PX, dpi),
                            )
                    except Exception:
                        pass

        # --- draw points
        if points_gj and "features" in points_gj:
            buckets: Dict[str, List[Tuple[float, float]]] = {}
            for feat in points_gj["features"]:
                geom = feat.get("geometry", {})
                props = feat.get("properties", {})
                if geom.get("type") != "Point":
                    continue
                try:
                    x, y = geom.get("coordinates", [None, None])
                    if x is None or y is None:
                        continue
                    src = props.get("source", "")
                    buckets.setdefault(src, []).append((x, y))
                except Exception:
                    continue

            for src, xy in buckets.items():
                if not xy:
                    continue
                xs = np.array([c[0] for c in xy])
                ys = np.array([c[1] for c in xy])
                if backend == "mercator":
                    xs, ys = project_fn(xs, ys)

                face_c = src_color.get(src, FALLBACK_COLOR)
                edge_c = POINT_EDGE_COLOR if POINT_EDGE_COLOR else face_c
                lw_pts = 0 if POINT_EDGE_COLOR is None else _px_to_points(0.6, dpi)

                ax.scatter(
                    xs, ys,
                    s=scatter_area_pts2,
                    alpha=POINT_ALPHA,
                    facecolors=face_c,
                    edgecolors=edge_c,
                    linewidths=lw_pts,
                    label=src or None,
                )

        # --- title
        title = f"{TITLE_PREFIX}  |  Frame {idx:04d}"
        if frame_time:
            title += f"  |  {frame_time}"
        ax.set_title(title, color=fg_color)

        # --- legend (only if we have named sources)
        if any(k for k in src_color.keys()):
            handles, labels = [], []
            used = set()
            for src in sorted(src_color.keys()):
                if src in used:
                    continue
                used.add(src)
                handles.append(
                    matplotlib.lines.Line2D(
                        [], [], color=src_color[src], marker='o',
                        linestyle='None', markersize=6
                    )
                )
                labels.append(src)
            if handles:
                leg = ax.legend(handles, labels, loc="upper right", frameon=True)
                for text in leg.get_texts():
                    text.set_color(fg_color)

        # save PNG
        out_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
        plt.savefig(out_path, dpi=dpi, facecolor=bg_color)
        plt.close(fig)
        print(f"Saved {out_path}")

# ----------
# CLI entry
# ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", default=FRAMES_DIR_DEFAULT)
    ap.add_argument("--out_dir",   default=OUT_DIR_DEFAULT)
    ap.add_argument("--width",    type=int, default=FIG_WIDTH_PX)
    ap.add_argument("--height",   type=int, default=FIG_HEIGHT_PX)
    ap.add_argument("--dpi",      type=int, default=DPI)
    ap.add_argument("--basemap_mode",
                    default=BASEMAP_MODE_DEFAULT,
                    choices=["imagery","none"],
                    help="imagery = Esri satellite (default); none = plain lon/lat, no basemap")
    args = ap.parse_args()

    render_frames(
        frames_dir=str(args.frames_dir),
        out_dir=str(args.out_dir),
        width=int(args.width),
        height=int(args.height),
        dpi=int(args.dpi),
        basemap_mode=str(args.basemap_mode),
    )

if __name__ == "__main__":
    main()
