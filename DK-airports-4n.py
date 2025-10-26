#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenDrift + ERA5 10 m winds (CF) — DK seeding (airports + optional offshore) with Leeway
Adds FLOW SNAPSHOTS exports (one GeoJSON per frame) for smooth animation in QGIS.

Enhancements vs previous:
- horizontal_diffusivity = 25.0 m^2/s  (stronger turbulent spread)
- "vertical dispersion proxy":
    we scale the 10 m winds upward based on wind speed magnitude to mimic lofting
    into faster winds aloft. Stronger surface wind => more mixing => faster transport.

Tested with OpenDrift 1.14.x
"""

from __future__ import annotations
import sys, json, logging, shutil
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# OpenDrift
from opendrift.models.leeway import Leeway
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers import reader_global_landmask

# -----------------------
# PATHS / CONFIG (Linux-friendly)
# -----------------------
BASE_DIR = Path(__file__).resolve().parent

# Input ERA5 winds (10m) and derived/working files
ERA5_10M_CF      = BASE_DIR / "era5_10m_wind_CF.nc"   # <-- your ERA5 file
OUT_CF_RAW       = BASE_DIR / "_opendrift_rawwind.nc" # normalized raw 10m wind
OUT_CF_BOOSTED   = BASE_DIR / "_opendrift_boostedwind.nc"  # boosted (shear/mixing proxy)
OUT_CURR         = BASE_DIR / "_zero_currents.nc"     # zero currents for Leeway

# Simulation outputs
OUT_NETCDF  = BASE_DIR / "opendrift_air_aerosol_tracks.nc"
OUT_GEOJSON = BASE_DIR / "opendrift_air_aerosol_tracks.geojson"
OUT_GJ_PTS  = BASE_DIR / "opendrift_air_aerosol_points.geojson"
OUT_CSV_PTS = BASE_DIR / "opendrift_air_aerosol_points.csv"

# FLOW snapshots directory (will be wiped/created each run)
FRAMES_DIR           = BASE_DIR / "frames"
FRAME_PT_TEMPLATE    = "frame_{:04d}_points.geojson"
FRAME_TAIL_TEMPLATE  = "frame_{:04d}_tails.geojson"

# Simulation control
PARTICLES_PER_SITE = 2000
TIME_STEP_SECONDS  = 1800   # 30 min
MAX_DAYS           = 5

# Exports thinning
POINTS_EVERY_N_STEPS = 1
PARTICLE_SUBSAMPLE   = 1
WRITE_CSV_POINTS     = True

# Flow snapshot options
SNAPSHOT_EVERY_N_STEPS = 1     # write every step; raise to thin frames
TAIL_STEPS             = 4     # 0 = off; otherwise short path of last N steps

# Toggle: include offshore sources?
INCLUDE_OFFSHORE = False

# Shear / vertical mixing proxy
# wind_boost_factor = 1 + MIX_ALPHA * wind_speed_10m
MIX_ALPHA = 0.3   # tune this up/down. 0.3 means ~+30% per m/s

# SEEDS (lon, lat)
AIRPORTS: Dict[str, tuple[float, float]] = {
    "CPH": (12.6560, 55.6180),
    "AAL": (9.8492, 57.0928),
    "BLL": (9.1518, 55.7403),
}
OFFSHORE: Dict[str, tuple[float, float]] = {
    "CPH_off": (12.80, 55.60),
    "AAL_off": (10.20, 57.30),
    "BLL_off": (8.20, 55.60),
}

# -----------------------
# LOGGING
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("OpenDrift-ERA5")

# -----------------------
# HELPERS
# -----------------------
def normalize_cf_10m_winds(infile: Path, outfile: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Ensure ERA5 10m wind is CF-ish:
    - coords named lat/lon
    - vars named eastward_wind/northward_wind with CF attrs
    Returns (t0, t1).
    """
    infile = Path(infile)
    outfile = Path(outfile)
    log.info(f"Checking ERA5 NetCDF: {infile}")
    if not infile.exists():
        raise FileNotFoundError(str(infile))

    ds = xr.open_dataset(infile)

    # rename coords to lat/lon if needed
    ren_c = {}
    if "latitude" in ds.coords:  ren_c["latitude"]  = "lat"
    if "longitude" in ds.coords: ren_c["longitude"] = "lon"
    if ren_c:
        ds = ds.rename(ren_c)
        log.info(f"Renamed coords: {ren_c}")

    # rename dims x/y -> lon/lat if they exist
    if "x" in ds.dims and "x" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"x": "lon"})
    if "y" in ds.dims and "y" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"y": "lat"})

    # rename wind vars to CF-ish names
    ren_v = {}
    if "x_wind" in ds and "eastward_wind" not in ds:   ren_v["x_wind"] = "eastward_wind"
    if "y_wind" in ds and "northward_wind" not in ds: ren_v["y_wind"] = "northward_wind"
    if ren_v:
        ds = ds.rename(ren_v)
        log.info(f"Renamed wind variables: {ren_v}")

    # Attach attrs
    if "eastward_wind" in ds:
        ds["eastward_wind"] = ds["eastward_wind"].assign_attrs(
            standard_name="eastward_wind", long_name="Eastward 10m wind", units="m s-1"
        )
    if "northward_wind" in ds:
        ds["northward_wind"] = ds["northward_wind"].assign_attrs(
            standard_name="northward_wind", long_name="Northward 10m wind", units="m s-1"
        )

    # decode_cf -> make sure 'time' is datetime64
    if "time" in ds.coords:
        try:
            ds = xr.decode_cf(ds)
        except Exception:
            pass

    outfile.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(outfile)
    log.info(f"Wrote normalized CF file for OpenDrift: {outfile}")

    if "time" not in ds:
        raise ValueError("No time coordinate found after normalization.")
    t0 = pd.to_datetime(ds.time.values[0]).tz_localize(None)
    t1 = pd.to_datetime(ds.time.values[-1]).tz_localize(None)
    log.info(f"Dataset time coverage: {t0} → {t1} (UTC)")
    ds.close()
    return t0, t1


def apply_vertical_mixing_proxy(raw_cf: Path, boosted_cf: Path, alpha: float = MIX_ALPHA):
    """
    Create boosted wind file:
    u_boost = u10 * (1 + alpha * |U10|)
    v_boost = v10 * (1 + alpha * |U10|)
    This mimics: stronger surface wind -> more lofting -> particles feel faster shear flow aloft.
    """
    raw_cf = Path(raw_cf)
    boosted_cf = Path(boosted_cf)
    ds = xr.open_dataset(raw_cf)

    if not all(v in ds for v in ("eastward_wind", "northward_wind")):
        raise ValueError("Expected eastward_wind / northward_wind in normalized wind file")

    # Original 10 m winds
    u10 = ds["eastward_wind"]         # DataArray [time, lat, lon]
    v10 = ds["northward_wind"]        # DataArray [time, lat, lon]

    # Wind speed magnitude at 10 m
    speed10 = np.sqrt(u10**2 + v10**2)

    # Scale factor for 'lofted wind'
    factor = 1.0 + alpha * speed10

    # Boosted components
    u_boost = u10 * factor
    v_boost = v10 * factor

    # xarray doesn't like directly stuffing DataArrays into Dataset in some cases.
    # We'll explicitly pass .data and dims.
    dims_u = u10.dims        # expected ('time','lat','lon')
    dims_v = v10.dims

    ds_boost = xr.Dataset(
        data_vars={
            "eastward_wind": (
                dims_u,
                u_boost.data.astype("float32"),
                {
                    "standard_name": "eastward_wind",
                    "long_name": "Boosted eastward wind (vertical mixing proxy)",
                    "units": "m s-1",
                },
            ),
            "northward_wind": (
                dims_v,
                v_boost.data.astype("float32"),
                {
                    "standard_name": "northward_wind",
                    "long_name": "Boosted northward wind (vertical mixing proxy)",
                    "units": "m s-1",
                },
            ),
        },
        coords={
            "time": (
                "time",
                ds["time"].values,
            ),
            "lat": (
                "lat",
                ds["lat"].values,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            "lon": (
                "lon",
                ds["lon"].values,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
        },
        attrs={
            "title": "10m wind boosted to emulate lofted aerosol advection",
            "note": f"u,v scaled by (1 + {alpha} * |U10|)",
            "Conventions": "CF-1.8",
        },
    )

    boosted_cf.parent.mkdir(parents=True, exist_ok=True)
    ds_boost.to_netcdf(boosted_cf)

    ds.close()
    ds_boost.close()
    log.info(f"Wrote boosted wind file (vertical mixing proxy): {boosted_cf}")



def write_zero_currents_like(wind_cf_path: Path, out_curr_path: Path):
    """
    Leeway normally expects ocean currents.
    We supply a zero-current field that matches the wind grid/time so the model runs.
    """
    wind_cf_path = Path(wind_cf_path)
    out_curr_path = Path(out_curr_path)
    ds_w = xr.open_dataset(wind_cf_path)

    for c in ("time", "lat", "lon"):
        if c not in ds_w.coords:
            raise ValueError(f"{wind_cf_path} is missing coord '{c}'")

    time = ds_w["time"].values
    lat = ds_w["lat"].values
    lon = ds_w["lon"].values
    zeros = np.zeros((len(time), len(lat), len(lon)), dtype="float32")

    ds_c = xr.Dataset(
        data_vars=dict(
            x_sea_water_velocity=(
                ("time", "lat", "lon"),
                zeros,
                dict(
                    standard_name="eastward_sea_water_velocity",
                    long_name="Zero ocean current (east)",
                    units="m s-1",
                ),
            ),
            y_sea_water_velocity=(
                ("time", "lat", "lon"),
                zeros,
                dict(
                    standard_name="northward_sea_water_velocity",
                    long_name="Zero ocean current (north)",
                    units="m s-1",
                ),
            ),
        ),
        coords=dict(
            time=("time", time),
            lat=("lat", lat, dict(standard_name="latitude", units="degrees_north")),
            lon=("lon", lon, dict(standard_name="longitude", units="degrees_east")),
        ),
        attrs=dict(
            title="Zero ocean currents for Leeway",
            Conventions="CF-1.8",
        ),
    )
    out_curr_path.parent.mkdir(parents=True, exist_ok=True)
    ds_c.to_netcdf(out_curr_path)
    ds_w.close()
    ds_c.close()
    log.info(f"Wrote zero-currents file for Leeway: {out_curr_path}")


def make_reader(cf_path: Path):
    cf_path = Path(cf_path)
    log.info(f"Opening file with xr.open_dataset: {cf_path}")
    return reader_netCDF_CF_generic.Reader(str(cf_path))


# ---- export helpers ----
def _sanitize_value_for_netcdf(v):
    import numpy as np
    if isinstance(v, (str, bytes, int, float, bool)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (np.dtype, type)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return type(v)(_sanitize_value_for_netcdf(x) for x in v)
    if isinstance(v, np.ndarray):
        return v if v.dtype.kind in "buifcUOS" else v.astype("float64", copy=False)
    return str(v)


def _sanitize_attrs_inplace(ds: xr.Dataset) -> xr.Dataset:
    ds.attrs = {k: _sanitize_value_for_netcdf(v) for k, v in list(ds.attrs.items())}
    for _, var in ds.variables.items():
        var.attrs = {k: _sanitize_value_for_netcdf(v) for k, v in list(var.attrs.items())}
        if hasattr(var, "encoding"):
            var.encoding = {}
    if hasattr(ds, "encoding"):
        ds.encoding = {}
    return ds


def export_to_netcdf_via_result(o, path: Path):
    path = Path(path)
    ds: xr.Dataset = o.result
    # make sure time is nanosecond precision to avoid float64-as-attrs issues
    if "time" in ds.coords:
        try:
            ds = ds.assign_coords(time=ds.time.astype("datetime64[ns]"))
        except Exception:
            pass
    ds = ds.assign_attrs(
        title="OpenDrift tracks (Leeway, land+sea)",
        featureType="trajectory",
        Conventions="CF-1.8",
    )
    ds = _sanitize_attrs_inplace(ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def _to_time_traj(da: xr.DataArray) -> xr.DataArray:
    # Ensure shape is [time, trajectory]
    dims = list(da.dims)
    if "time" in dims and "trajectory" in dims:
        return da.transpose("time", "trajectory") if da.dims != ("time", "trajectory") else da
    if "time" in dims:
        other = [d for d in dims if d != "time"][0]
        return da.transpose("time", other)
    return da.transpose(...)


def export_to_geojson_from_result(o, path: Path):
    path = Path(path)
    ds: xr.Dataset = o.result
    L = _to_time_traj(ds["lon"]).values
    A = _to_time_traj(ds["lat"]).values
    ntime, npart = L.shape

    feats = []
    for p in range(npart):
        coords = [
            [float(L[t, p]), float(A[t, p])]
            for t in range(ntime)
            if not (np.isnan(L[t, p]) or np.isnan(A[t, p]))
        ]
        if len(coords) >= 2:
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"particle": int(p)},
                    "geometry": {"type": "LineString", "coordinates": coords},
                }
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)


def export_points_geojson_temporal(
    o,
    path: Path,
    every_n_steps: int = 1,
    particle_subsample: int = 1,
    per_site: int = PARTICLES_PER_SITE,
    site_names: List[str] | None = None,
    also_csv: Path | None = None,
):
    path = Path(path)
    ds: xr.Dataset = o.result
    L = _to_time_traj(ds["lon"]).values
    A = _to_time_traj(ds["lat"]).values
    time = pd.to_datetime(ds["time"].values)

    ntime, npart = L.shape
    if site_names is None:
        site_names = []

    path.parent.mkdir(parents=True, exist_ok=True)
    csv_f = None
    if also_csv:
        also_csv = Path(also_csv)
        also_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_f = also_csv.open("w", encoding="utf-8")
        csv_f.write("time,lon,lat,particle,step,source\n")

    count_features = 0
    with path.open("w", encoding="utf-8") as f:
        f.write('{"type":"FeatureCollection","features":[')
        first = True
        for p in range(0, npart, particle_subsample):
            block = p // per_site
            src = site_names[min(block, len(site_names) - 1)] if site_names else ""
            for t in range(0, ntime, every_n_steps):
                x = float(L[t, p])
                y = float(A[t, p])
                if np.isnan(x) or np.isnan(y):
                    continue
                when = (
                    pd.Timestamp(time[t])
                    .tz_localize("UTC")
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                feat = {
                    "type": "Feature",
                    "properties": {
                        "particle": int(p),
                        "step": int(t),
                        "time": when,
                        "source": src,
                    },
                    "geometry": {"type": "Point", "coordinates": [x, y]},
                }
                if not first:
                    f.write(",")
                else:
                    first = False
                f.write(json.dumps(feat, ensure_ascii=False))
                count_features += 1
                if csv_f:
                    csv_f.write(f"{when},{x},{y},{p},{t},{src}\n")
        f.write("]}")
    if csv_f:
        csv_f.close()
    log.info(f"Wrote {count_features:,} point features to {path}")


# -------- FLOW SNAPSHOTS --------
def _ensure_clean_dir(d: Path):
    d = Path(d)
    if d.is_dir():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)


def export_flow_frames_points(o, out_dir: Path, template: str, step_stride=1):
    """
    One GeoJSON per frame with ONLY the particles at that step (Points).
    """
    out_dir = Path(out_dir)
    _ensure_clean_dir(out_dir)

    ds: xr.Dataset = o.result
    L = _to_time_traj(ds["lon"]).values
    A = _to_time_traj(ds["lat"]).values
    times = pd.to_datetime(ds["time"].values)

    ntime, npart = L.shape
    frames = 0
    for t in range(0, ntime, step_stride):
        feats = []
        when_iso = (
            pd.Timestamp(times[t]).tz_localize("UTC").isoformat().replace("+00:00", "Z")
        )
        for p in range(npart):
            x = float(L[t, p])
            y = float(A[t, p])
            if np.isnan(x) or np.isnan(y):
                continue
            feats.append(
                {
                    "type": "Feature",
                    "properties": {
                        "particle": int(p),
                        "time": when_iso,
                        "step": int(t),
                    },
                    "geometry": {"type": "Point", "coordinates": [x, y]},
                }
            )
        out_path = out_dir / template.format(t)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
        frames += 1
    log.info(f"Flow frames (points): wrote {frames} files to {out_dir}")


def export_flow_frames_tails(
    o, out_dir: Path, template: str, tail_steps=4, step_stride=1
):
    """
    One GeoJSON per frame with short LineString tails (last `tail_steps` positions) per particle.
    Set tail_steps <= 0 to skip.
    """
    if tail_steps <= 0:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds: xr.Dataset = o.result
    L = _to_time_traj(ds["lon"]).values
    A = _to_time_traj(ds["lat"]).values
    times = pd.to_datetime(ds["time"].values)

    ntime, npart = L.shape
    frames = 0
    for t in range(0, ntime, step_stride):
        t0 = max(0, t - tail_steps + 1)
        feats = []
        when_iso = (
            pd.Timestamp(times[t]).tz_localize("UTC").isoformat().replace("+00:00", "Z")
        )
        for p in range(npart):
            coords = []
            for tt in range(t0, t + 1):
                x = float(L[tt, p])
                y = float(A[tt, p])
                if np.isnan(x) or np.isnan(y):
                    continue
                coords.append([x, y])
            if coords:
                geom = (
                    {"type": "LineString", "coordinates": coords}
                    if len(coords) > 1
                    else {"type": "Point", "coordinates": coords[0]}
                )
                feats.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "particle": int(p),
                            "time": when_iso,
                            "step": int(t),
                        },
                        "geometry": geom,
                    }
                )
        out_path = out_dir / template.format(t)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f)
        frames += 1
    log.info(f"Flow frames (tails={tail_steps}): wrote {frames} files to {out_dir}")


# -----------------------
# MAIN
# -----------------------
def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--era5", default=str(ERA5_10M_CF), help="ERA5 10m wind NetCDF path")
    ap.add_argument("--out-raw", default=str(OUT_CF_RAW), help="Normalized raw wind file")
    ap.add_argument(
        "--out-boost",
        default=str(OUT_CF_BOOSTED),
        help="Boosted wind file (vertical mixing proxy)",
    )
    ap.add_argument("--out-curr", default=str(OUT_CURR), help="Zero currents file")
    ap.add_argument("--out-nc", default=str(OUT_NETCDF), help="Tracks NetCDF")
    ap.add_argument("--out-gj", default=str(OUT_GEOJSON), help="Tracks GeoJSON (LineString)")
    ap.add_argument(
        "--out-gj-pts", default=str(OUT_GJ_PTS), help="Temporal points GeoJSON"
    )
    ap.add_argument(
        "--out-csv-pts", default=str(OUT_CSV_PTS), help="Temporal points CSV"
    )
    ap.add_argument(
        "--frames-dir", default=str(FRAMES_DIR), help="Per-frame flow outputs dir"
    )
    ap.add_argument(
        "--include-offshore",
        action="store_true" if not INCLUDE_OFFSHORE else "store_false",
    )
    ap.add_argument("--max-days", type=int, default=MAX_DAYS)
    ap.add_argument("--dt", type=int, default=TIME_STEP_SECONDS)
    ap.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY_N_STEPS)
    ap.add_argument("--tail-steps", type=int, default=TAIL_STEPS)
    ap.add_argument("--points-every", type=int, default=POINTS_EVERY_N_STEPS)
    ap.add_argument("--particle-subsample", type=int, default=PARTICLE_SUBSAMPLE)
    args = ap.parse_args()

    # normalize flags/paths
    era5 = Path(args.era5)
    out_raw = Path(args.out_raw)
    out_boost = Path(args.out_boost)
    out_curr = Path(args.out_curr)
    out_nc = Path(args.out_nc)
    out_gj = Path(args.out_gj)
    out_gj_pts = Path(args.out_gj_pts)
    out_csv_pts = Path(args.out_csv_pts)
    frames_dir = Path(args.frames_dir)
    include_offshore = (
        args.include_offshore
        if args.include_offshore != (not INCLUDE_OFFSHORE)
        else INCLUDE_OFFSHORE
    )
    max_days = max(1, int(args.max_days))
    dt_sec = max(60, int(args.dt))  # at least 1 minute

    # Clean up single-file outputs so we don't append stale data
    for f in [out_raw, out_boost, out_curr, out_nc, out_gj, out_gj_pts, out_csv_pts]:
        try:
            if f.exists():
                f.unlink()
        except Exception as e:
            log.warning(f"Could not remove {f}: {e}")

    # 1) Normalize winds -> out_raw, record time coverage
    t0, t1 = normalize_cf_10m_winds(era5, out_raw)

    # 2) Apply mixing/shear proxy -> out_boost
    apply_vertical_mixing_proxy(out_raw, out_boost, alpha=MIX_ALPHA)

    # 3) Build zero-currents matching boosted wind grid/time
    write_zero_currents_like(out_boost, out_curr)

    # 4) Readers
    #    NOTE: we now feed the BOOSTED winds to OpenDrift,
    #    which makes particles advect faster when winds are stronger.
    r_wind = make_reader(out_boost)
    r_curr = make_reader(out_curr)  # required by Leeway
    r_land = reader_global_landmask.Reader()  # provides land_binary_mask

    # 5) Leeway model (LAND + SEA enabled)
    o = Leeway(loglevel=20)
    o.set_config("general:use_auto_landmask", False)
    o.set_config("seed:ocean_only", False)
    o.set_config("general:coastline_action", "none")

    # Stronger turbulent spread (you changed this to 25 already)
    o.set_config("drift:horizontal_diffusivity", 25.0)  # m^2/s
    o.set_config("drift:wind_uncertainty", 0.5)        # m/s random wind jitter

    o.add_reader(r_curr)
    o.add_reader(r_wind)
    o.add_reader(r_land)

    # 6) Seeding
    seed_sites = dict(AIRPORTS)
    if include_offshore:
        seed_sites.update(OFFSHORE)

    start_time = pd.Timestamp(t0).to_pydatetime()
    for name, (lon, lat) in seed_sites.items():
        log.info(f"Seeding {PARTICLES_PER_SITE} at {name}: lon={lon:.4f}, lat={lat:.4f}")
        o.seed_elements(lon=lon, lat=lat, number=PARTICLES_PER_SITE, time=start_time)

    # 7) Run
    total_hours_data = max(1, int((t1 - t0).total_seconds() // 3600))
    total_hours_sim = min(max_days * 24, total_hours_data)
    duration = timedelta(hours=total_hours_sim)

    log.info(f"Simulation: {t0} → {t0 + duration} | dt={dt_sec}s")
    o.run(duration=duration, time_step=timedelta(seconds=dt_sec))

    # 8) Exports (single files)
    try:
        log.info(f"Writing {out_nc.name}")
        export_to_netcdf_via_result(o, out_nc)
    except Exception as e:
        log.error(f"Could not export NetCDF: {e}")

    try:
        log.info(f"Writing {out_gj.name}")
        export_to_geojson_from_result(o, out_gj)
    except Exception as e:
        log.error(f"GeoJSON export failed: {e}")

    try:
        log.info(f"Writing {out_gj_pts.name} (temporal points)")
        export_points_geojson_temporal(
            o,
            out_gj_pts,
            every_n_steps=args.points_every,
            particle_subsample=args.particle_subsample,
            per_site=PARTICLES_PER_SITE,
            site_names=list(seed_sites.keys()),
            also_csv=out_csv_pts if WRITE_CSV_POINTS else None,
        )
    except Exception as e:
        log.error(f"Temporal Points export failed: {e}")

    # 9) FLOW SNAPSHOTS (per-frame files)
    try:
        export_flow_frames_points(
            o, frames_dir, FRAME_PT_TEMPLATE, step_stride=args.snapshot_every
        )
        export_flow_frames_tails(
            o,
            frames_dir,
            FRAME_TAIL_TEMPLATE,
            tail_steps=args.tail_steps,
            step_stride=args.snapshot_every,
        )
    except Exception as e:
        log.error(f"Flow snapshot export failed: {e}")

    log.info("✅ Finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.exception(exc)
        sys.exit(1)
