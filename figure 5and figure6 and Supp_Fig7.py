#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result 2 | direct observational analysis redesign (rev7)
--------------------------------------------------------
Fixes relative to rev6:
1) Figure1_result2_main.png panel b:
   - move shared y-label further left
   - remove x-axis titles from all three subpanels
2) Figure1_result2_main.png panel d and Supp_Fig1_result2_day3_rr.png panel a:
   - force any filename containing “连续” to Continuous
   - use stronger fallback settings so Continuous curves are drawn whenever valid data exist
3) Add event-type diagnostics to console and summary text.
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_PYGEOS", "0")

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re
import unicodedata

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_DIR = Path(r"D:\nature\全球数据分析最终版\欧美两地分析最先版\最新包含每个温度的全部数据")
OUTPUT_DIRNAME = "_result2_direct_redesign_rev7_2000_2024"
FIG_DPI = 320
YEAR_MIN = 2000
YEAR_MAX = 2024
PRIMARY_RESPONSE = "max_rr"
ROBUST_RESPONSE = "day3_rr"
BOOT_N = 40
BINS_2D_X = 18
BINS_2D_Y = 16
RANDOM_SEED = 42
LANDCOVER_ORDER = ["Forest", "Cropland", "Grassland", "Urban", "Wetland", "Bareland"]
LINE_MAIN = 3.2
LINE_COMPARE = 3.0
LINE_SUPPORT = 2.8
CI_ALPHA = 0.22


def configure_style() -> Dict[str, str]:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 21,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return {
        "na": "#2F5D9B",
        "na_soft": "#B7C9E8",
        "eu": "#B35C37",
        "eu_soft": "#E7C8BA",
        "iso": "#2F5D9B",
        "cont": "#C47A2C",
        "all": "#2B2B2B",
        "rain": "#2F5D9B",
        "sm": "#3F8A4E",
        "temp": "#B35C37",
        "cool": "#2F5D9B",
        "mid": "#8B8B8B",
        "hot": "#C0642F",
        "forest": "#2E7D32",
        "crop": "#C47A2C",
        "grass": "#7BAA47",
        "urban": "#7E57C2",
        "wet": "#1F9AC0",
        "bare": "#8D6E63",
        "ink": "#202124",
        "muted": "#7C7C7C",
        "grid": "#DDDDDD",
    }


PALETTE = configure_style()
SURFACE_CMAP = LinearSegmentedColormap.from_list("surface", ["#2F5D9B", "#F7F7F7", "#B35C37"])


def parse_filename_metadata(path: Path) -> Dict[str, str]:
    stem = path.stem
    continent = "Unknown"
    if "美洲" in stem:
        continent = "North America"
    elif "欧洲" in stem:
        continent = "Europe"

    event_type = "Unknown"
    if "单独" in stem:
        event_type = "Isolated"
    elif "连续" in stem:
        event_type = "Continuous"

    lc = "Unknown"
    if "草地" in stem:
        lc = "Grassland"
    elif "城市区域" in stem or "城市" in stem:
        lc = "Urban"
    elif "裸地" in stem:
        lc = "Bareland"
    elif "农田" in stem:
        lc = "Cropland"
    elif "森林" in stem:
        lc = "Forest"
    elif "湿地" in stem:
        lc = "Wetland"

    return {
        "continent": continent,
        "event_type": event_type,
        "land_cover_group": lc,
        "source_file": stem,
    }


COLUMN_ALIASES = {
    "周期开始日期": "cycle_start_date",
    "周期结束日期": "cycle_end_date",
    "第一次降雨事件日期": "first_rain_date",
    "第一次降雨当天rainfall_ratio": "firstday_rr",
    "最后一次有效降雨事件日期": "last_effective_rain_date",
    "第一天ratio": "day1_rr",
    "第二天ratio": "day2_rr",
    "第三天ratio": "day3_rr",
    "第四天ratio": "day4_rr",
    "第五天ratio": "day5_rr",
    "第六天ratio": "day6_rr",
    "周期最大ratio": "max_rr",
    "总降雨量（>5）": "total_rain_gt5",
    "总降雨量(>5)": "total_rain_gt5",
    "周期最大降雨量": "max_rain",
    "表面温度平均值": "surface_temp_mean",
    "温度条件": "temp_group",
    "saturation": "saturation",
    "clay_fraction": "clay_fraction",
    "organic_carbon": "organic_carbon",
    "sand_fraction": "sand_fraction",
    "silt_fraction": "silt_fraction",
    "land_cover": "land_cover_raw",
    "climate": "climate",
    "soil_moisture": "soil_moisture",
    "station": "station",
    "lon": "lon",
    "lat": "lat",
    "longitude": "lon",
    "latitude": "lat",
}


def normalize_colname(name: str) -> str:
    s = unicodedata.normalize("NFKC", str(name))
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("：", ":").replace("，", ",")
    s = re.sub(r"\s+", " ", s).strip()
    return s


NORMALIZED_ALIASES = {normalize_colname(k): v for k, v in COLUMN_ALIASES.items()}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_colname(c) for c in out.columns]
    out = out.rename(columns={c: NORMALIZED_ALIASES.get(c, c) for c in out.columns})
    if out.columns.duplicated().any():
        merged = pd.DataFrame(index=out.index)
        for col in pd.unique(out.columns):
            same = out.loc[:, out.columns == col]
            merged[col] = same.bfill(axis=1).iloc[:, 0]
        out = merged
    return out.copy()


def plausible_bounds_for_continent(continent: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if continent == "North America":
        return (-170, -45), (10, 85)
    if continent == "Europe":
        return (-15, 45), (30, 72)
    return (-180, 180), (-90, 90)


def fix_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "lon" not in out.columns or "lat" not in out.columns:
        return out
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")

    fixed_lon, fixed_lat = [], []
    for continent, lon, lat in out[["continent", "lon", "lat"]].itertuples(index=False, name=None):
        if pd.isna(lon) or pd.isna(lat):
            fixed_lon.append(lon)
            fixed_lat.append(lat)
            continue
        lon_bounds, lat_bounds = plausible_bounds_for_continent(continent)
        ok_as_is = lon_bounds[0] <= lon <= lon_bounds[1] and lat_bounds[0] <= lat <= lat_bounds[1]
        ok_swapped = lon_bounds[0] <= lat <= lon_bounds[1] and lat_bounds[0] <= lon <= lat_bounds[1]
        if (not ok_as_is and ok_swapped) or (abs(lon) <= 90 and abs(lat) > 90):
            fixed_lon.append(lat)
            fixed_lat.append(lon)
        else:
            fixed_lon.append(lon)
            fixed_lat.append(lat)
    out["lon"] = fixed_lon
    out["lat"] = fixed_lat
    return out


def force_event_type_from_source(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    src = out.get("source_file", pd.Series(index=out.index, dtype=str)).astype(str)
    et = out.get("event_type", pd.Series(index=out.index, dtype=str)).astype(str)
    out["event_type"] = np.where(
        src.str.contains("连续", na=False), "Continuous",
        np.where(src.str.contains("单独", na=False), "Isolated", et)
    )
    return out


def load_excel_files(input_dir: Path) -> pd.DataFrame:
    files = sorted([p for p in input_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in: {input_dir}")
    print(f"Loading Excel files from: {input_dir}")
    frames: List[pd.DataFrame] = []
    for path in files:
        meta = parse_filename_metadata(path)
        df = pd.read_excel(path)
        df = standardize_columns(df)
        for k, v in meta.items():
            df[k] = v
        frames.append(df)
    events = pd.concat(frames, ignore_index=True, sort=False)

    if "station" not in events.columns:
        events["station"] = events["source_file"]
    events["station"] = events["station"].astype(str).str.strip()

    year_series = None
    for c in ["cycle_start_date", "first_rain_date", "last_effective_rain_date", "cycle_end_date"]:
        if c in events.columns:
            dt = pd.to_datetime(events[c], errors="coerce")
            if dt.notna().sum() > 0:
                year_series = dt.dt.year
                break
    if year_series is None:
        raise ValueError("Could not infer year from date columns.")
    events["year"] = year_series.astype("Int64")

    for c in ["max_rr", "day3_rr", "surface_temp_mean", "soil_moisture", "total_rain_gt5", "max_rain"]:
        if c in events.columns:
            events[c] = pd.to_numeric(events[c], errors="coerce")

    events = fix_coordinates(events)
    events = force_event_type_from_source(events)
    events = events.dropna(subset=["continent", "event_type", "station", "year"]).copy()
    events["year"] = events["year"].astype(int)
    events = events[(events["year"] >= YEAR_MIN) & (events["year"] <= YEAR_MAX)].copy()

    if "climate" not in events.columns:
        events["climate"] = "Unknown"

    if "max_rr" in events.columns:
        events = events[events["max_rr"].between(-5, 10, inclusive="both") | events["max_rr"].isna()].copy()
    if "day3_rr" in events.columns:
        events = events[events["day3_rr"].between(-5, 10, inclusive="both") | events["day3_rr"].isna()].copy()
    if "soil_moisture" in events.columns:
        events = events[(events["soil_moisture"].isna()) | events["soil_moisture"].between(0, 1, inclusive="both")].copy()
    if "total_rain_gt5" in events.columns:
        events = events[(events["total_rain_gt5"].isna()) | events["total_rain_gt5"].between(0, 500, inclusive="both")].copy()

    return events.reset_index(drop=True)


def diagnose_missing_by_event_type(events: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    print(f"Missing-value diagnosis for {label} before sample filtering:")
    for et, sub in events.groupby("event_type", dropna=False):
        parts = [f"{c}_na={int(sub[c].isna().sum())}" for c in cols if c in sub.columns]
        print(f"  {et}: n={len(sub)} | " + " | ".join(parts))



def build_sample(events: pd.DataFrame, response: str, required: Optional[Sequence[str]] = None,
                 optional: Optional[Sequence[str]] = None, clip_response: bool = True) -> pd.DataFrame:
    if required is None:
        required = [response, "total_rain_gt5", "soil_moisture", "surface_temp_mean", "station", "continent", "event_type", "land_cover_group"]
    if optional is None:
        optional = ["climate", "year"]
    keep_cols = [c for c in list(required) + list(optional) if c in events.columns]
    df = events[keep_cols].copy()
    req = [c for c in required if c in df.columns]
    if req:
        df = df.dropna(subset=req).copy()
    if clip_response and response in df.columns and not df.empty:
        q_low, q_high = df[response].quantile([0.005, 0.995])
        df[response] = df[response].clip(lower=q_low, upper=q_high)
    return df.reset_index(drop=True)



def build_core_sample(events: pd.DataFrame, response: str) -> pd.DataFrame:
    diagnose_missing_by_event_type(events, [response, "total_rain_gt5", "soil_moisture", "surface_temp_mean"], response)
    return build_sample(events, response=response)



def build_eventtype_sample(events: pd.DataFrame, response: str) -> pd.DataFrame:
    required = [response, "total_rain_gt5", "station", "event_type"]
    optional = ["continent", "year"]
    diagnose_missing_by_event_type(events, [response, "total_rain_gt5"], f"{response} event-type")
    return build_sample(events, response=response, required=required, optional=optional)



def build_continent_sample(events: pd.DataFrame, response: str) -> pd.DataFrame:
    required = [response, "total_rain_gt5", "station", "continent"]
    optional = ["event_type", "year"]
    return build_sample(events, response=response, required=required, optional=optional)



def build_temp_sample(events: pd.DataFrame, response: str) -> pd.DataFrame:
    required = [response, "total_rain_gt5", "station", "surface_temp_mean"]
    optional = ["event_type", "continent", "year"]
    return build_sample(events, response=response, required=required, optional=optional)



def build_landcover_sample(events: pd.DataFrame, response: str) -> pd.DataFrame:
    required = [response, "total_rain_gt5", "station", "land_cover_group"]
    optional = ["event_type", "continent", "year"]
    return build_sample(events, response=response, required=required, optional=optional)


def quantile_edges(series: pd.Series, n_bins: int, q_lo: float = 0.02, q_hi: float = 0.98) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if x.size < 6:
        return np.array([])
    qs = np.linspace(q_lo, q_hi, n_bins + 1)
    edges = np.nanquantile(x, qs)
    edges = np.unique(edges)
    return edges if len(edges) >= 3 else np.array([])


def adaptive_curve_settings(n_obs: int, target_bins: int = 16) -> Tuple[int, int]:
    if n_obs < 40:
        return 6, 3
    if n_obs < 80:
        return 7, 4
    if n_obs < 160:
        return 9, 5
    if n_obs < 300:
        return 11, 6
    if n_obs < 600:
        return 13, 8
    return target_bins, 10


def binned_curve(df: pd.DataFrame, predictor: str, response: str,
                 subset: Optional[pd.Series] = None,
                 target_bins: int = 16,
                 min_bin_count: Optional[int] = None,
                 boot_n: int = BOOT_N,
                 random_state: int = RANDOM_SEED) -> pd.DataFrame:
    sub = df.copy()
    if subset is not None:
        sub = sub.loc[subset].copy()
    sub = sub.dropna(subset=[predictor, response, "station"])
    if sub.empty:
        return pd.DataFrame(columns=["x", "y", "lo", "hi", "n"])

    n_bins, default_min = adaptive_curve_settings(len(sub), target_bins=target_bins)
    if min_bin_count is None:
        min_bin_count = default_min

    edges = quantile_edges(sub[predictor], n_bins=n_bins)
    if edges.size == 0:
        return pd.DataFrame(columns=["x", "y", "lo", "hi", "n"])

    sub["_bin"] = pd.cut(sub[predictor], bins=edges, include_lowest=True, duplicates="drop")
    raw_counts = sub.groupby("_bin", observed=True)[response].size().reset_index(name="n")
    keep_bins = raw_counts.loc[raw_counts["n"] >= min_bin_count, "_bin"].tolist()
    if len(keep_bins) < 3:
        min_bin_count = max(1, min_bin_count - 2)
        keep_bins = raw_counts.loc[raw_counts["n"] >= min_bin_count, "_bin"].tolist()
    if len(keep_bins) < 2:
        return pd.DataFrame(columns=["x", "y", "lo", "hi", "n"])

    grouped = sub.groupby("_bin", observed=True).agg(
        x=(predictor, "median"),
        y=(response, "median"),
        n=(response, "size"),
    )
    grouped = grouped.reindex(keep_bins).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    stations = np.array(sorted(sub["station"].astype(str).unique()))
    boot_vals: List[np.ndarray] = []
    for _ in range(boot_n):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        boot_df = pd.concat([sub[sub["station"].astype(str) == s] for s in sampled], ignore_index=True)
        boot_df["_bin"] = pd.cut(boot_df[predictor], bins=edges, include_lowest=True, duplicates="drop")
        tmp = boot_df.groupby("_bin", observed=True)[response].median().reindex(keep_bins)
        boot_vals.append(tmp.to_numpy(dtype=float))
    boot_arr = np.vstack(boot_vals) if boot_vals else np.empty((0, len(grouped)))
    grouped["lo"] = np.nanpercentile(boot_arr, 5, axis=0) if boot_arr.size else grouped["y"]
    grouped["hi"] = np.nanpercentile(boot_arr, 95, axis=0) if boot_arr.size else grouped["y"]
    return grouped[["x", "y", "lo", "hi", "n"]].reset_index(drop=True)


def rolling_fallback_curve(df: pd.DataFrame, predictor: str, response: str,
                           subset: Optional[pd.Series] = None,
                           boot_n: int = BOOT_N,
                           random_state: int = RANDOM_SEED) -> pd.DataFrame:
    sub = df.copy()
    if subset is not None:
        sub = sub.loc[subset].copy()
    sub = sub.dropna(subset=[predictor, response, "station"])
    if len(sub) < 8:
        return pd.DataFrame(columns=["x", "y", "lo", "hi", "n"])

    sub = sub.sort_values(predictor).reset_index(drop=True)
    n = len(sub)
    win = max(4, min(20, n // 5))
    if win >= n:
        win = max(3, n // 2)

    idxs = np.arange(0, n - win + 1, max(1, win // 3))
    rows = []
    for start in idxs:
        chunk = sub.iloc[start:start+win]
        rows.append({
            "x": float(np.nanmedian(chunk[predictor].to_numpy(dtype=float))),
            "y": float(np.nanmedian(chunk[response].to_numpy(dtype=float))),
            "n": len(chunk),
        })
    base = pd.DataFrame(rows)
    if len(base) < 2:
        return pd.DataFrame(columns=["x", "y", "lo", "hi", "n"])

    rng = np.random.default_rng(random_state)
    stations = np.array(sorted(sub["station"].astype(str).unique()))
    boot_matrix = []
    for _ in range(boot_n):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        boot_df = pd.concat([sub[sub["station"].astype(str) == s] for s in sampled], ignore_index=True)
        boot_df = boot_df.sort_values(predictor).reset_index(drop=True)
        if len(boot_df) < win:
            continue
        vals = []
        for start in idxs:
            end = min(start + win, len(boot_df))
            chunk = boot_df.iloc[start:end]
            if len(chunk) < 3:
                vals.append(np.nan)
            else:
                vals.append(float(np.nanmedian(chunk[response].to_numpy(dtype=float))))
        boot_matrix.append(vals)
    if boot_matrix:
        arr = np.asarray(boot_matrix, dtype=float)
        base["lo"] = np.nanpercentile(arr, 5, axis=0)
        base["hi"] = np.nanpercentile(arr, 95, axis=0)
    else:
        base["lo"] = base["y"]
        base["hi"] = base["y"]
    return base[["x", "y", "lo", "hi", "n"]]


def robust_binned_curve(df: pd.DataFrame, predictor: str, response: str,
                        subset: Optional[pd.Series] = None,
                        settings: Optional[List[Tuple[int, int]]] = None,
                        boot_n: int = BOOT_N) -> pd.DataFrame:
    if settings is None:
        settings = [(16, 10), (12, 6), (10, 4), (8, 3), (6, 2), (5, 1), (4, 1), (3, 1)]
    for bins, minc in settings:
        out = binned_curve(df, predictor, response, subset=subset, target_bins=bins, min_bin_count=minc, boot_n=boot_n)
        if not out.empty and len(out) >= 2:
            return out
    return rolling_fallback_curve(df, predictor, response, subset=subset, boot_n=boot_n)


def binned_surface(df: pd.DataFrame, xvar: str, yvar: str, response: str,
                   n_x: int = BINS_2D_X, n_y: int = BINS_2D_Y,
                   min_count: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df.dropna(subset=[xvar, yvar, response]).copy()
    if sub.empty:
        return np.array([]), np.array([]), np.array([[]])
    x_edges = quantile_edges(sub[xvar], n_bins=n_x)
    y_edges = quantile_edges(sub[yvar], n_bins=n_y, q_lo=0.03, q_hi=0.97)
    if x_edges.size == 0 or y_edges.size == 0:
        return np.array([]), np.array([]), np.array([[]])
    sub["_xbin"] = pd.cut(sub[xvar], bins=x_edges, include_lowest=True, duplicates="drop")
    sub["_ybin"] = pd.cut(sub[yvar], bins=y_edges, include_lowest=True, duplicates="drop")
    med = sub.pivot_table(index="_ybin", columns="_xbin", values=response, aggfunc="median", observed=True)
    cnt = sub.pivot_table(index="_ybin", columns="_xbin", values=response, aggfunc="size", observed=True)
    x_centers = np.array([iv.mid for iv in med.columns.tolist()], dtype=float)
    y_centers = np.array([iv.mid for iv in med.index.tolist()], dtype=float)
    Z = med.to_numpy(dtype=float)
    C = cnt.reindex(index=med.index, columns=med.columns).to_numpy(dtype=float)
    Z[C < min_count] = np.nan
    return x_centers, y_centers, Z


def derivative_breakpoint(curve: pd.DataFrame) -> Optional[float]:
    if curve.empty or len(curve) < 3:
        return None
    x = curve["x"].to_numpy(dtype=float)
    y = curve["y"].to_numpy(dtype=float)
    try:
        dy = np.gradient(y, x)
        neg = np.where(dy < 0)[0]
        if len(neg) > 0:
            return float(x[int(neg[0])])
    except Exception:
        pass
    return float(x[int(np.nanargmax(y))])


def breakpoint_table(df: pd.DataFrame, response: str) -> pd.DataFrame:
    specs: List[Tuple[str, pd.Series]] = [
        ("All events", pd.Series(True, index=df.index)),
        ("North America", df["continent"] == "North America"),
        ("Europe", df["continent"] == "Europe"),
        ("Isolated", df["event_type"] == "Isolated"),
        ("Continuous", df["event_type"] == "Continuous"),
    ]
    for lc in LANDCOVER_ORDER:
        specs.append((lc, df["land_cover_group"] == lc))

    rows = []
    for name, mask in specs:
        sub = df.loc[mask].copy()
        if len(sub) < 8:
            rows.append({"group": name, "bp": np.nan, "lo": np.nan, "hi": np.nan, "n": len(sub)})
            continue
        base_curve = robust_binned_curve(sub, "total_rain_gt5", response, settings=[(12, 4), (8, 3), (6, 2), (4, 1)], boot_n=0)
        bp = derivative_breakpoint(base_curve)
        if bp is None:
            rows.append({"group": name, "bp": np.nan, "lo": np.nan, "hi": np.nan, "n": len(sub)})
            continue
        rng = np.random.default_rng(RANDOM_SEED)
        stations = np.array(sorted(sub["station"].astype(str).unique()))
        bps = []
        for _ in range(BOOT_N):
            sampled = rng.choice(stations, size=len(stations), replace=True)
            boot_df = pd.concat([sub[sub["station"].astype(str) == s] for s in sampled], ignore_index=True)
            curve = robust_binned_curve(boot_df, "total_rain_gt5", response, settings=[(12, 4), (8, 3), (6, 2), (4, 1)], boot_n=0)
            bpi = derivative_breakpoint(curve)
            if bpi is not None and np.isfinite(bpi):
                bps.append(float(bpi))
        lo, hi = (np.nan, np.nan)
        if len(bps) >= 5:
            lo, hi = np.nanpercentile(np.asarray(bps, dtype=float), [5, 95]).tolist()
        rows.append({"group": name, "bp": float(bp), "lo": float(lo), "hi": float(hi), "n": len(sub)})
    return pd.DataFrame(rows)


def bootstrap_eta2(df: pd.DataFrame, response: str,
                   features: Sequence[str], boot_n: int = BOOT_N,
                   random_state: int = RANDOM_SEED) -> pd.DataFrame:
    sub = df.dropna(subset=[response, "station"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["feature", "eta2", "boot"])
    sub["_y"] = sub[response] - sub.groupby("station")[response].transform("median")
    rng = np.random.default_rng(random_state)
    stations = np.array(sorted(sub["station"].astype(str).unique()))

    def _eta2(work: pd.DataFrame, feat: str) -> float:
        yy = work["_y"].to_numpy(dtype=float)
        if np.nanvar(yy) <= 0:
            return np.nan
        if feat in ["total_rain_gt5", "soil_moisture", "surface_temp_mean"]:
            edges = quantile_edges(work[feat], n_bins=12)
            if edges.size == 0:
                return np.nan
            bins = pd.cut(work[feat], bins=edges, include_lowest=True, duplicates="drop")
            g = work.groupby(bins, observed=True)["_y"].agg(["mean", "size"])
        else:
            g = work.groupby(feat, observed=True)["_y"].agg(["mean", "size"])
        if g.empty:
            return np.nan
        ss_between = np.nansum(g["size"].to_numpy(dtype=float) * (g["mean"].to_numpy(dtype=float) ** 2))
        ss_total = np.nansum(yy ** 2)
        return float(ss_between / ss_total) if ss_total > 0 else np.nan

    rows = []
    for feat in features:
        rows.append({"feature": feat, "eta2": _eta2(sub, feat), "boot": -1})
    for b in range(boot_n):
        sampled = rng.choice(stations, size=len(stations), replace=True)
        boot_df = pd.concat([sub[sub["station"].astype(str) == s] for s in sampled], ignore_index=True)
        for feat in features:
            rows.append({"feature": feat, "eta2": _eta2(boot_df, feat), "boot": b})
    return pd.DataFrame(rows)


def temp_group_series(df: pd.DataFrame) -> pd.Series:
    x = df["surface_temp_mean"].to_numpy(dtype=float)
    q1, q2 = np.nanquantile(x, [1/3, 2/3])
    return pd.Series(np.where(x <= q1, "Cool", np.where(x <= q2, "Intermediate", "Hot")), index=df.index)


def tidy_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis in ("x", "both"):
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.75)
    if grid_axis in ("y", "both"):
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.75)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=32, fontweight="bold", ha="left", va="bottom")


def draw_surface(ax: plt.Axes, xgrid: np.ndarray, ygrid: np.ndarray, Z: np.ndarray,
                 xlab: str, ylab: str, title: str, cbar_label: str) -> None:
    if Z.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return
    med = np.nanmedian(Z)
    abs95 = max(float(np.nanpercentile(np.abs(Z - med), 95)), 0.05)
    norm = TwoSlopeNorm(vcenter=med, vmin=med - abs95, vmax=med + abs95)
    img = ax.contourf(xgrid, ygrid, Z, levels=18, cmap=SURFACE_CMAP, norm=norm)
    ax.contour(xgrid, ygrid, Z, levels=8, colors="white", linewidths=0.45, alpha=0.7)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left")
    tidy_axes(ax, grid_axis="both")
    cbar = plt.colorbar(img, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=18)


def draw_single_curve(ax: plt.Axes, curve: pd.DataFrame, color: str, xlabel: str = "", title: Optional[str] = None,
                      annotate_bp: bool = True, show_xticklabels: bool = True) -> None:
    if curve.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return
    x = curve["x"].to_numpy(dtype=float)
    y = curve["y"].to_numpy(dtype=float)
    lo = curve["lo"].to_numpy(dtype=float)
    hi = curve["hi"].to_numpy(dtype=float)
    ax.fill_between(x, lo, hi, color=color, alpha=CI_ALPHA, linewidth=0)
    ax.plot(x, y, color=color, linewidth=LINE_MAIN)
    if annotate_bp:
        bp = derivative_breakpoint(curve)
        if bp is not None and np.isfinite(bp):
            ax.axvline(bp, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
            ax.text(bp, ax.get_ylim()[1], f" {bp:.1f}", va="top", fontsize=13, color=PALETTE["muted"])
    if title:
        ax.text(0.01, 0.84, title, transform=ax.transAxes, fontsize=17, color=PALETTE["ink"])
    ax.set_xlabel(xlabel, labelpad=6)
    if not show_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)
    tidy_axes(ax, grid_axis="both")


def draw_curve_compare(ax: plt.Axes, curves: List[Tuple[pd.DataFrame, str, str, str]],
                       xlabel: str, ylabel: str, title: str,
                       show_legend: bool = True) -> None:
    handles = []
    for curve, color, label, ls in curves:
        if curve.empty:
            continue
        x = curve["x"].to_numpy(dtype=float)
        y = curve["y"].to_numpy(dtype=float)
        lo = curve["lo"].to_numpy(dtype=float)
        hi = curve["hi"].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color=color, alpha=CI_ALPHA, linewidth=0)
        ln, = ax.plot(x, y, color=color, linewidth=LINE_COMPARE, linestyle=ls, label=label)
        handles.append(ln)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    tidy_axes(ax, grid_axis="both")
    if show_legend and handles:
        ax.legend(handles=handles, frameon=False, loc="best")


def draw_eta2_violin(ax: plt.Axes, eta_df: pd.DataFrame) -> None:
    order = ["total_rain_gt5", "surface_temp_mean", "continent", "event_type", "soil_moisture", "land_cover_group", "climate"]
    labels = {
        "total_rain_gt5": "Rainfall",
        "surface_temp_mean": "Temperature",
        "continent": "Continent",
        "event_type": "Event type",
        "soil_moisture": "Soil moisture",
        "land_cover_group": "Land cover",
        "climate": "Climate",
    }
    colors = {
        "total_rain_gt5": PALETTE["rain"],
        "soil_moisture": PALETTE["sm"],
        "surface_temp_mean": PALETTE["temp"],
        "continent": PALETTE["eu"],
        "event_type": PALETTE["all"],
        "land_cover_group": PALETTE["urban"],
        "climate": PALETTE["bare"],
    }
    y = np.arange(len(order))[::-1] + 1
    base = eta_df[eta_df["boot"] == -1].set_index("feature")
    for yi, feat in zip(y, order):
        arr = eta_df[(eta_df["feature"] == feat) & (eta_df["boot"] >= 0)]["eta2"].dropna().to_numpy(dtype=float)
        if arr.size >= 5:
            vp = ax.violinplot([arr], positions=[yi], vert=False, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(colors[feat])
                body.set_alpha(0.16)
                body.set_edgecolor("none")
            q10, q25, med, q75, q90 = np.nanpercentile(arr, [10, 25, 50, 75, 90])
            ax.hlines(yi, q10, q90, color=colors[feat], linewidth=1.6)
            ax.hlines(yi, q25, q75, color=colors[feat], linewidth=6.0)
            ax.scatter([med], [yi], s=80, color=colors[feat], edgecolor="white", linewidth=0.8, zorder=3)
        if feat in base.index:
            ax.scatter([base.loc[feat, "eta2"]], [yi], s=58, color=colors[feat], edgecolor="white", linewidth=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[o] for o in order])
    ax.set_xlabel("Bootstrapped association strength (η²)")
    tidy_axes(ax, grid_axis="x")


def draw_breakpoint_summary(ax: plt.Axes, bp_table: pd.DataFrame, response_label: str) -> None:
    order = ["All events", "North America", "Europe", "Isolated", "Continuous"] + LANDCOVER_ORDER
    cmap = {
        "All events": PALETTE["all"],
        "North America": PALETTE["na"],
        "Europe": PALETTE["eu"],
        "Isolated": PALETTE["iso"],
        "Continuous": PALETTE["cont"],
        "Forest": PALETTE["forest"],
        "Cropland": PALETTE["crop"],
        "Grassland": PALETTE["grass"],
        "Urban": PALETTE["urban"],
        "Wetland": PALETTE["wet"],
        "Bareland": PALETTE["bare"],
    }
    y = np.arange(len(order))[::-1]
    for yi, label in zip(y, order):
        sub = bp_table[bp_table["group"] == label]
        if sub.empty:
            continue
        r = sub.iloc[0]
        if not np.isfinite(r["bp"]):
            continue
        col = cmap[label]
        if np.isfinite(r["lo"]) and np.isfinite(r["hi"]):
            ax.hlines(yi, r["lo"], r["hi"], color=col, linewidth=1.7)
        ax.scatter(r["bp"], yi, s=72, color=col, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel(f"Empirical breakpoint in rainfall (mm) | {response_label}")
    tidy_axes(ax, grid_axis="x")


def draw_landcover_multiline(ax: plt.Axes, curves: Dict[str, pd.DataFrame], response_label: str) -> None:
    color_map = {
        "Forest": PALETTE["forest"],
        "Cropland": PALETTE["crop"],
        "Grassland": PALETTE["grass"],
        "Urban": PALETTE["urban"],
        "Wetland": PALETTE["wet"],
        "Bareland": PALETTE["bare"],
    }
    handles = []
    for lc in LANDCOVER_ORDER:
        curve = curves.get(lc, pd.DataFrame())
        if curve.empty:
            continue
        x = curve["x"].to_numpy(dtype=float)
        y = curve["y"].to_numpy(dtype=float)
        lo = curve["lo"].to_numpy(dtype=float) if "lo" in curve.columns else y
        hi = curve["hi"].to_numpy(dtype=float) if "hi" in curve.columns else y
        ax.fill_between(x, lo, hi, color=color_map[lc], alpha=CI_ALPHA, linewidth=0)
        ln, = ax.plot(x, y, color=color_map[lc], linewidth=LINE_SUPPORT, label=lc)
        handles.append(ln)
    ax.set_xlabel("Rainfall > 5 mm (mm)")
    ax.set_ylabel(f"Binned median {response_label}")
    ax.set_title("Land-cover-specific rainfall sensitivity", loc="left")
    tidy_axes(ax, grid_axis="both")
    if handles:
        ax.legend(handles=handles, frameon=False, ncol=2, loc="best", fontsize=15)


def make_main_figure(output_dir: Path,
                     surface_x: np.ndarray, surface_y: np.ndarray, surface_Z: np.ndarray,
                     curve_rain: pd.DataFrame, curve_sm: pd.DataFrame, curve_temp: pd.DataFrame,
                     eta_df: pd.DataFrame,
                     curve_eventtypes: Dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.84, 1.16], height_ratios=[1.0, 0.95], wspace=0.58, hspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    draw_surface(ax1, surface_x, surface_y, surface_Z,
                 xlab="Rainfall > 5 mm (mm)",
                 ylab="Antecedent soil moisture",
                 title="All rainfall events",
                 cbar_label="Binned median Max RR")
    add_panel_label(ax1, "a")

    inner = gs[0, 1].subgridspec(3, 1, hspace=0.42)
    axb1 = fig.add_subplot(inner[0, 0])
    draw_single_curve(axb1, curve_rain, PALETTE["rain"], xlabel="", title="Rainfall", show_xticklabels=True)
    axb2 = fig.add_subplot(inner[1, 0])
    draw_single_curve(axb2, curve_sm, PALETTE["sm"], xlabel="", title="Soil moisture", show_xticklabels=True)
    axb3 = fig.add_subplot(inner[2, 0])
    draw_single_curve(axb3, curve_temp, PALETTE["temp"], xlabel="", title="Temperature", show_xticklabels=True)
    for ax in [axb1, axb2, axb3]:
        ax.set_ylabel("")
        ax.set_xlabel("")
    fig.text(axb1.get_position().x0 - 0.040,
             (axb1.get_position().y1 + axb3.get_position().y0) / 2,
             "Binned median\nMax RR", rotation=90, va="center", ha="center", fontsize=22)
    add_panel_label(axb1, "b")

    ax3 = fig.add_subplot(gs[1, 0])
    draw_eta2_violin(ax3, eta_df)
    add_panel_label(ax3, "c")

    ax4 = fig.add_subplot(gs[1, 1])
    compare = [
        (curve_eventtypes.get("Isolated", pd.DataFrame()), PALETTE["iso"], "Isolated", "-"),
        (curve_eventtypes.get("Continuous", pd.DataFrame()), PALETTE["cont"], "Continuous", "--"),
    ]
    draw_curve_compare(ax4, compare, "Rainfall > 5 mm (mm)", "Binned median Max RR", "Event-type contrast in rainfall sensitivity")
    add_panel_label(ax4, "d")

    fig.savefig(output_dir / "Figure1_result2_main.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def make_structure_figure(output_dir: Path,
                          continent_curves: Dict[str, pd.DataFrame],
                          temp_curves: Dict[str, pd.DataFrame],
                          lc_curves: Dict[str, pd.DataFrame],
                          bp_table: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.20, hspace=0.24)

    ax1 = fig.add_subplot(gs[0, 0])
    curves = [
        (continent_curves.get("North America", pd.DataFrame()), PALETTE["na"], "North America", "-"),
        (continent_curves.get("Europe", pd.DataFrame()), PALETTE["eu"], "Europe", "-"),
    ]
    draw_curve_compare(ax1, curves, "Rainfall > 5 mm (mm)", "Binned median Max RR", "Continental contrast in rainfall sensitivity")
    add_panel_label(ax1, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    temp_compare = [
        (temp_curves.get("Cool", pd.DataFrame()), PALETTE["cool"], "Cool", "-"),
        (temp_curves.get("Intermediate", pd.DataFrame()), PALETTE["mid"], "Intermediate", "-"),
        (temp_curves.get("Hot", pd.DataFrame()), PALETTE["hot"], "Hot", "-"),
    ]
    draw_curve_compare(ax2, temp_compare, "Rainfall > 5 mm (mm)", "Binned median Max RR", "Temperature-conditioned rainfall sensitivity")
    add_panel_label(ax2, "b")

    ax3 = fig.add_subplot(gs[1, 0])
    draw_landcover_multiline(ax3, lc_curves, response_label="Max RR")
    add_panel_label(ax3, "c")

    ax4 = fig.add_subplot(gs[1, 1])
    draw_breakpoint_summary(ax4, bp_table, response_label="Max RR")
    add_panel_label(ax4, "d")

    fig.savefig(output_dir / "Figure2_result2_structure.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def make_day3_figure(output_dir: Path,
                     day3_event_curves: Dict[str, pd.DataFrame],
                     day3_continent_curves: Dict[str, pd.DataFrame],
                     day3_temp_curves: Dict[str, pd.DataFrame],
                     day3_lc_curves: Dict[str, pd.DataFrame]) -> None:
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.22, hspace=0.24)

    ax1 = fig.add_subplot(gs[0, 0])
    c1 = [
        (day3_event_curves.get("Isolated", pd.DataFrame()), PALETTE["iso"], "Isolated", "-"),
        (day3_event_curves.get("Continuous", pd.DataFrame()), PALETTE["cont"], "Continuous", "--"),
    ]
    draw_curve_compare(ax1, c1, "Rainfall > 5 mm (mm)", "Binned median Day 3 RR", "Day 3 RR | event-type contrast")
    add_panel_label(ax1, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    c2 = [
        (day3_continent_curves.get("North America", pd.DataFrame()), PALETTE["na"], "North America", "-"),
        (day3_continent_curves.get("Europe", pd.DataFrame()), PALETTE["eu"], "Europe", "-"),
    ]
    draw_curve_compare(ax2, c2, "Rainfall > 5 mm (mm)", "Binned median Day 3 RR", "Day 3 RR | continental contrast")
    add_panel_label(ax2, "b")

    ax3 = fig.add_subplot(gs[1, 0])
    c3 = [
        (day3_temp_curves.get("Cool", pd.DataFrame()), PALETTE["cool"], "Cool", "-"),
        (day3_temp_curves.get("Intermediate", pd.DataFrame()), PALETTE["mid"], "Intermediate", "-"),
        (day3_temp_curves.get("Hot", pd.DataFrame()), PALETTE["hot"], "Hot", "-"),
    ]
    draw_curve_compare(ax3, c3, "Rainfall > 5 mm (mm)", "Binned median Day 3 RR", "Day 3 RR | temperature conditioning")
    add_panel_label(ax3, "c")

    ax4 = fig.add_subplot(gs[1, 1])
    draw_landcover_multiline(ax4, day3_lc_curves, response_label="Day 3 RR")
    ax4.set_title("Day 3 RR | land-cover curves", loc="left")
    add_panel_label(ax4, "d")

    fig.savefig(output_dir / "Supp_Fig1_result2_day3_rr.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def export_tables(output_dir: Path, events: pd.DataFrame, max_df: pd.DataFrame, day3_df: pd.DataFrame,
                  eta_df: pd.DataFrame, bp_max: pd.DataFrame, bp_day3: pd.DataFrame,
                  lc_counts: pd.DataFrame) -> None:
    events.to_csv(output_dir / "events_cleaned_2000_2024.csv", index=False, encoding="utf-8-sig")
    max_df.to_csv(output_dir / "result2_all_events_max_rr.csv", index=False, encoding="utf-8-sig")
    day3_df.to_csv(output_dir / "result2_all_events_day3_rr.csv", index=False, encoding="utf-8-sig")
    eta_df.to_csv(output_dir / "result2_eta2_bootstrap.csv", index=False, encoding="utf-8-sig")
    bp_max.to_csv(output_dir / "result2_breakpoints_max_rr.csv", index=False, encoding="utf-8-sig")
    bp_day3.to_csv(output_dir / "result2_breakpoints_day3_rr.csv", index=False, encoding="utf-8-sig")
    lc_counts.to_csv(output_dir / "result2_landcover_event_counts.csv", index=False, encoding="utf-8-sig")


def count_curve_points(curve: pd.DataFrame) -> int:
    return 0 if curve is None or curve.empty else int(len(curve))


def main() -> None:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_DIR
    output_dir = input_dir / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading and cleaning input files...")
    events = load_excel_files(input_dir)

    core_max_df = build_core_sample(events, response=PRIMARY_RESPONSE)
    core_day3_df = build_core_sample(events, response=ROBUST_RESPONSE) if ROBUST_RESPONSE in events.columns else pd.DataFrame()

    event_max_df = build_eventtype_sample(events, response=PRIMARY_RESPONSE)
    event_day3_df = build_eventtype_sample(events, response=ROBUST_RESPONSE) if ROBUST_RESPONSE in events.columns else pd.DataFrame()
    continent_max_df = build_continent_sample(events, response=PRIMARY_RESPONSE)
    continent_day3_df = build_continent_sample(events, response=ROBUST_RESPONSE) if ROBUST_RESPONSE in events.columns else pd.DataFrame()
    temp_max_df = build_temp_sample(events, response=PRIMARY_RESPONSE)
    temp_day3_df = build_temp_sample(events, response=ROBUST_RESPONSE) if ROBUST_RESPONSE in events.columns else pd.DataFrame()
    lc_max_df = build_landcover_sample(events, response=PRIMARY_RESPONSE)
    lc_day3_df = build_landcover_sample(events, response=ROBUST_RESPONSE) if ROBUST_RESPONSE in events.columns else pd.DataFrame()

    if core_max_df.empty:
        raise ValueError("Core Max RR sample is empty after cleaning.")

    print("Event-type counts after cleaning:")
    print(events["event_type"].value_counts(dropna=False).to_string())
    print("Event-type sample counts for Max RR panel d:")
    print(event_max_df["event_type"].value_counts(dropna=False).to_string())
    if not event_day3_df.empty:
        print("Event-type sample counts for Day3 RR panel a:")
        print(event_day3_df["event_type"].value_counts(dropna=False).to_string())

    print("[2/4] Building direct-analysis summaries for Max RR...")
    sx, sy, SZ = binned_surface(core_max_df, "total_rain_gt5", "soil_moisture", PRIMARY_RESPONSE)
    curve_rain = robust_binned_curve(core_max_df, "total_rain_gt5", PRIMARY_RESPONSE, settings=[(16, 10), (12, 6), (10, 4), (8, 3), (6, 2), (5, 1), (4, 1)])
    curve_sm = robust_binned_curve(core_max_df, "soil_moisture", PRIMARY_RESPONSE, settings=[(16, 10), (12, 6), (10, 4), (8, 3), (6, 2), (5, 1), (4, 1)])
    curve_temp = robust_binned_curve(core_max_df, "surface_temp_mean", PRIMARY_RESPONSE, settings=[(16, 10), (12, 6), (10, 4), (8, 3), (6, 2), (5, 1), (4, 1)])

    eta_features = ["total_rain_gt5", "surface_temp_mean", "continent", "event_type", "soil_moisture", "land_cover_group", "climate"]
    eta_df = bootstrap_eta2(core_max_df, PRIMARY_RESPONSE, eta_features)

    event_curves = {
        "Isolated": robust_binned_curve(event_max_df, "total_rain_gt5", PRIMARY_RESPONSE, subset=event_max_df["event_type"] == "Isolated", settings=[(12, 5), (10, 4), (8, 3), (6, 2), (5, 1), (4, 1), (3, 1)]),
        "Continuous": robust_binned_curve(event_max_df, "total_rain_gt5", PRIMARY_RESPONSE, subset=event_max_df["event_type"] == "Continuous", settings=[(12, 5), (10, 4), (8, 3), (6, 2), (5, 1), (4, 1), (3, 1)]),
    }

    continent_curves = {
        "North America": robust_binned_curve(continent_max_df, "total_rain_gt5", PRIMARY_RESPONSE, subset=continent_max_df["continent"] == "North America", settings=[(13, 6), (10, 4), (8, 3)]),
        "Europe": robust_binned_curve(continent_max_df, "total_rain_gt5", PRIMARY_RESPONSE, subset=continent_max_df["continent"] == "Europe", settings=[(10, 4), (8, 3), (6, 2), (4, 1)]),
    }

    temp_max_df = temp_max_df.copy()
    temp_max_df["temp_group3"] = temp_group_series(temp_max_df)
    temp_curves = {
        "Cool": robust_binned_curve(temp_max_df, "total_rain_gt5", PRIMARY_RESPONSE, subset=temp_max_df["temp_group3"] == "Cool", settings=[(12, 5), (10, 4), (8, 3)]),
        "Intermediate": robust_binned_curve(temp_max_df, "total_rain_gt5", PRIMARY_RESPONSE, subset=temp_max_df["temp_group3"] == "Intermediate", settings=[(12, 5), (10, 4), (8, 3)]),
        "Hot": robust_binned_curve(temp_max_df, "total_rain_gt5", PRIMARY_RESPONSE, subset=temp_max_df["temp_group3"] == "Hot", settings=[(12, 5), (10, 4), (8, 3)]),
    }

    lc_curves = {lc: robust_binned_curve(lc_max_df, "total_rain_gt5", PRIMARY_RESPONSE, subset=lc_max_df["land_cover_group"] == lc, settings=[(8, 3), (6, 2), (5, 1), (4, 1), (3, 1)]) for lc in LANDCOVER_ORDER}
    bp_max = breakpoint_table(core_max_df, PRIMARY_RESPONSE)

    print("Max RR curve points:")
    for k, v in event_curves.items():
        print(f"  {k}: {len(v)}")

    print("[3/4] Building direct-analysis summaries for Day 3 RR...")
    if core_day3_df.empty:
        day3_event_curves = {k: pd.DataFrame() for k in ["Isolated", "Continuous"]}
        day3_continent_curves = {k: pd.DataFrame() for k in ["North America", "Europe"]}
        day3_temp_curves = {k: pd.DataFrame() for k in ["Cool", "Intermediate", "Hot"]}
        day3_lc_curves = {lc: pd.DataFrame() for lc in LANDCOVER_ORDER}
        bp_day3 = pd.DataFrame(columns=["group", "bp", "lo", "hi", "n"])
    else:
        day3_event_curves = {
            "Isolated": robust_binned_curve(event_day3_df, "total_rain_gt5", ROBUST_RESPONSE, subset=event_day3_df["event_type"] == "Isolated", settings=[(12, 5), (10, 4), (8, 3), (6, 2), (5, 1), (4, 1), (3, 1)]),
            "Continuous": robust_binned_curve(event_day3_df, "total_rain_gt5", ROBUST_RESPONSE, subset=event_day3_df["event_type"] == "Continuous", settings=[(12, 5), (10, 4), (8, 3), (6, 2), (5, 1), (4, 1), (3, 1)]),
        }
        day3_continent_curves = {
            "North America": robust_binned_curve(continent_day3_df, "total_rain_gt5", ROBUST_RESPONSE, subset=continent_day3_df["continent"] == "North America", settings=[(13, 6), (10, 4), (8, 3)]),
            "Europe": robust_binned_curve(continent_day3_df, "total_rain_gt5", ROBUST_RESPONSE, subset=continent_day3_df["continent"] == "Europe", settings=[(10, 4), (8, 3), (6, 2), (4, 1)]),
        }
        temp_day3_df = temp_day3_df.copy()
        temp_day3_df["temp_group3"] = temp_group_series(temp_day3_df)
        day3_temp_curves = {
            "Cool": robust_binned_curve(temp_day3_df, "total_rain_gt5", ROBUST_RESPONSE, subset=temp_day3_df["temp_group3"] == "Cool", settings=[(12, 5), (10, 4), (8, 3)]),
            "Intermediate": robust_binned_curve(temp_day3_df, "total_rain_gt5", ROBUST_RESPONSE, subset=temp_day3_df["temp_group3"] == "Intermediate", settings=[(12, 5), (10, 4), (8, 3)]),
            "Hot": robust_binned_curve(temp_day3_df, "total_rain_gt5", ROBUST_RESPONSE, subset=temp_day3_df["temp_group3"] == "Hot", settings=[(12, 5), (10, 4), (8, 3)]),
        }
        day3_lc_curves = {lc: robust_binned_curve(lc_day3_df, "total_rain_gt5", ROBUST_RESPONSE, subset=lc_day3_df["land_cover_group"] == lc, settings=[(8, 3), (6, 2), (5, 1), (4, 1), (3, 1)]) for lc in LANDCOVER_ORDER}
        bp_day3 = breakpoint_table(core_day3_df, ROBUST_RESPONSE)

        print("Day3 RR curve points:")
        for k, v in day3_event_curves.items():
            print(f"  {k}: {len(v)}")

    lc_counts = events.groupby(["land_cover_group", "event_type", "continent"], dropna=False).size().reset_index(name="n_events")

    print("[4/4] Writing figures and tables...")
    export_tables(output_dir, events, core_max_df, core_day3_df, eta_df, bp_max, bp_day3, lc_counts)
    make_main_figure(output_dir, sx, sy, SZ, curve_rain, curve_sm, curve_temp, eta_df, event_curves)
    make_structure_figure(output_dir, continent_curves, temp_curves, lc_curves, bp_max)
    if not core_day3_df.empty:
        make_day3_figure(output_dir, day3_event_curves, day3_continent_curves, day3_temp_curves, day3_lc_curves)

    with open(output_dir / "result2_readme_summary.txt", "w", encoding="utf-8") as f:
        f.write("Result 2 direct-analysis redesign rev7\n")
        f.write(f"Core sample (Max RR): {len(core_max_df)} rows\n")
        f.write(f"Core sample (Day 3 RR): {len(core_day3_df)} rows\n")
        f.write("\nEvent-type counts after cleaning:\n")
        f.write(events.groupby('event_type').size().to_string())
        f.write("\n\nEvent-type counts for panel d (Max RR):\n")
        f.write(event_max_df.groupby('event_type').size().to_string())
        if not event_day3_df.empty:
            f.write("\n\nEvent-type counts for Supp Fig a (Day3 RR):\n")
            f.write(event_day3_df.groupby('event_type').size().to_string())
        f.write("\n\nLand-cover counts:\n")
        f.write(lc_counts.to_string(index=False))

    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
