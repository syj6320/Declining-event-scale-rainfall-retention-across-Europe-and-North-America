#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result 1 | refined figure suite for event-scale rainfall retention efficiency
Nature/NCC-oriented graphics, stability-first implementation
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_PYGEOS", "0")

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import geopandas as gpd
except Exception:
    gpd = None

# ------------------------------
# User settings
# ------------------------------
INPUT_DIR = Path(r"D:\nature\全球数据分析最终版\欧美两地分析最先版\最新包含每个温度的全部数据")
OUTPUT_DIRNAME = "_result1_refined_ncc_style_2000_2024_fixed"
MIN_YEARS_FOR_SLOPE = 4
EARLY_LATE_WINDOW_YEARS = 3
FIG_DPI = 320
YEAR_MIN = 2000
YEAR_MAX = 2024
MAP_XMIN, MAP_XMAX = -128, 42
MAP_YMIN, MAP_YMAX = 24, 72

# ------------------------------
# Style
# ------------------------------

def configure_style() -> Dict[str, str]:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 21,
        "axes.titlesize": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "legend.fontsize": 18,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
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
        "na_soft": "#AEC3E3",
        "eu": "#B35C37",
        "eu_soft": "#E6C5B7",
        "ink": "#202124",
        "muted": "#8C8C8C",
        "grid": "#DDDDDD",
        "land": "#F6F6F4",
        "border": "#C8C8C8",
    }

PALETTE = configure_style()
SLOPE_CMAP = LinearSegmentedColormap.from_list("slope", ["#3B4CC0", "#F7F7F7", "#B40426"])
DELTA_CMAP = LinearSegmentedColormap.from_list("delta", ["#3B4CC0", "#F7F7F7", "#C92A2A"])

# ------------------------------
# Metadata parsing
# ------------------------------

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


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df.columns}).copy()

# ------------------------------
# Coordinate QA
# ------------------------------

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

    fixed_lon, fixed_lat, swapped = [], [], []
    for continent, lon, lat in out[["continent", "lon", "lat"]].itertuples(index=False, name=None):
        if pd.isna(lon) or pd.isna(lat):
            fixed_lon.append(lon)
            fixed_lat.append(lat)
            swapped.append(False)
            continue
        lon_bounds, lat_bounds = plausible_bounds_for_continent(continent)
        ok_as_is = lon_bounds[0] <= lon <= lon_bounds[1] and lat_bounds[0] <= lat <= lat_bounds[1]
        ok_swapped = lon_bounds[0] <= lat <= lon_bounds[1] and lat_bounds[0] <= lon <= lat_bounds[1]
        do_swap = False
        if not ok_as_is and ok_swapped:
            do_swap = True
        elif abs(lon) <= 90 and abs(lat) > 90:
            do_swap = True
        elif continent == "North America" and (lon > -20 and lat < 0):
            do_swap = True
        elif continent == "Europe" and (lat < 0 and -15 <= lon <= 72):
            do_swap = True
        if do_swap:
            fixed_lon.append(lat)
            fixed_lat.append(lon)
            swapped.append(True)
        else:
            fixed_lon.append(lon)
            fixed_lat.append(lat)
            swapped.append(False)
    out["lon"] = fixed_lon
    out["lat"] = fixed_lat
    out["coord_swapped"] = swapped
    return out

# ------------------------------
# Data loading
# ------------------------------

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

    for c in ["max_rr", "day3_rr", "lon", "lat", "surface_temp_mean", "soil_moisture"]:
        if c in events.columns:
            events[c] = pd.to_numeric(events[c], errors="coerce")

    events = fix_coordinates(events)

    if "max_rr" in events.columns:
        events = events[events["max_rr"].between(-5, 10, inclusive="both") | events["max_rr"].isna()]
    if "day3_rr" in events.columns:
        events = events[events["day3_rr"].between(-5, 10, inclusive="both") | events["day3_rr"].isna()]

    events = events.dropna(subset=["continent", "event_type", "station", "year", "max_rr"]).copy()
    events["year"] = events["year"].astype(int)
    events = events[(events["year"] >= YEAR_MIN) & (events["year"] <= YEAR_MAX)].copy()
    events = events[(events["lon"].isna()) | ((events["lon"] >= MAP_XMIN) & (events["lon"] <= MAP_XMAX) & (events["lat"] >= MAP_YMIN) & (events["lat"] <= MAP_YMAX))].copy()
    return events.reset_index(drop=True)

# ------------------------------
# Summary tables
# ------------------------------

def safe_group_label(continent: str, event_type: str) -> str:
    left = "NA" if continent == "North America" else "EU"
    right = "Iso" if event_type == "Isolated" else "Cont"
    return f"{left} {right}"


def build_station_year_summary(events: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "max_rr": "median",
        "lon": "median",
        "lat": "median",
        "source_file": "first",
    }
    if "day3_rr" in events.columns:
        agg["day3_rr"] = "median"
    if "surface_temp_mean" in events.columns:
        agg["surface_temp_mean"] = "median"
    if "soil_moisture" in events.columns:
        agg["soil_moisture"] = "median"

    summary = (
        events.groupby(["continent", "event_type", "land_cover_group", "station", "year"], dropna=False)
        .agg(agg)
        .rename(columns={"source_file": "example_source_file"})
        .reset_index()
    )
    counts = (
        events.groupby(["continent", "event_type", "land_cover_group", "station", "year"], dropna=False)
        .size().rename("event_count").reset_index()
    )
    summary = summary.merge(counts, on=["continent", "event_type", "land_cover_group", "station", "year"], how="left")
    summary["group"] = [safe_group_label(c, e) for c, e in zip(summary["continent"], summary["event_type"])]
    summary["weight"] = np.sqrt(summary["event_count"].clip(lower=1))
    return summary

# ------------------------------
# Trend estimation
# ------------------------------

def sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slopes: List[float] = []
    for i in range(len(x) - 1):
        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]
        valid = dx != 0
        if np.any(valid):
            slopes.extend((dy[valid] / dx[valid]).tolist())
    return float(np.nanmedian(slopes)) if slopes else np.nan


def ols_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return np.nan, np.nan
    m, b = np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1)
    return float(m), float(b)


def station_slopes(summary: pd.DataFrame, metric: str, min_years: int = MIN_YEARS_FOR_SLOPE) -> pd.DataFrame:
    rows = []
    cols = [
        "continent", "event_type", "station", "land_cover_group", "metric", "n_years",
        "year_min", "year_max", "lon", "lat", "ols_slope", "ols_intercept", "sen_slope",
        "early_value", "late_value", "late_minus_early"
    ]
    for (continent, event_type, station), sub in summary.groupby(["continent", "event_type", "station"], dropna=False):
        s = sub[["year", metric, "lon", "lat", "land_cover_group"]].dropna(subset=[metric]).sort_values("year")
        if s["year"].nunique() < min_years:
            continue
        x = s["year"].to_numpy(dtype=float)
        y = s[metric].to_numpy(dtype=float)
        m, b = ols_line(x, y)
        sen = sen_slope(x, y)
        years_sorted = np.sort(s["year"].unique())
        early_years = years_sorted[:min(EARLY_LATE_WINDOW_YEARS, len(years_sorted))]
        late_years = years_sorted[-min(EARLY_LATE_WINDOW_YEARS, len(years_sorted)):]
        early_val = float(np.nanmedian(s.loc[s["year"].isin(early_years), metric].to_numpy(dtype=float)))
        late_val = float(np.nanmedian(s.loc[s["year"].isin(late_years), metric].to_numpy(dtype=float)))
        lc_mode = s["land_cover_group"].mode(dropna=True)
        lc_value = lc_mode.iloc[0] if len(lc_mode) else "Unknown"
        rows.append({
            "continent": continent,
            "event_type": event_type,
            "station": station,
            "land_cover_group": lc_value,
            "metric": metric,
            "n_years": int(s["year"].nunique()),
            "year_min": int(np.min(x)),
            "year_max": int(np.max(x)),
            "lon": float(np.nanmedian(s["lon"].to_numpy(dtype=float))),
            "lat": float(np.nanmedian(s["lat"].to_numpy(dtype=float))),
            "ols_slope": m,
            "ols_intercept": b,
            "sen_slope": sen,
            "early_value": early_val,
            "late_value": late_val,
            "late_minus_early": late_val - early_val,
        })
    return pd.DataFrame(rows, columns=cols)

# ------------------------------
# Figure summaries
# ------------------------------

def yearly_station_medians(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = (
        summary.groupby(["continent", "event_type", "year"], dropna=False)[metric]
        .agg(median="median",
             q25=lambda x: np.nanpercentile(np.asarray(x, dtype=float), 25),
             q75=lambda x: np.nanpercentile(np.asarray(x, dtype=float), 75),
             n_station="count")
        .reset_index()
    )
    out["group"] = [safe_group_label(c, e) for c, e in zip(out["continent"], out["event_type"])]
    return out


def coverage_heatmap_table(summary: pd.DataFrame) -> pd.DataFrame:
    cov = (
        summary.groupby(["continent", "event_type", "year"], dropna=False)["station"]
        .nunique().rename("n_station").reset_index()
    )
    cov["group"] = [safe_group_label(c, e) for c, e in zip(cov["continent"], cov["event_type"])]
    return cov


def slope_interval_table(slopes: pd.DataFrame) -> pd.DataFrame:
    cols = ["continent", "event_type", "metric", "median_slope", "q10", "q25", "q75", "q90", "n_station", "group"]
    if slopes.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (continent, event_type, metric), sub in slopes.groupby(["continent", "event_type", "metric"], dropna=False):
        arr = sub["sen_slope"].dropna().to_numpy(dtype=float)
        if arr.size == 0:
            continue
        rows.append({
            "continent": continent,
            "event_type": event_type,
            "metric": metric,
            "median_slope": float(np.nanmedian(arr)),
            "q10": float(np.nanpercentile(arr, 10)),
            "q25": float(np.nanpercentile(arr, 25)),
            "q75": float(np.nanpercentile(arr, 75)),
            "q90": float(np.nanpercentile(arr, 90)),
            "n_station": int(arr.size),
            "group": safe_group_label(continent, event_type),
        })
    return pd.DataFrame(rows, columns=cols)


def landcover_matrix(slopes: pd.DataFrame, metric: str = "max_rr") -> pd.DataFrame:
    cols = ["continent", "event_type", "land_cover_group", "group", "median_slope", "n_station", "q25", "q75"]
    if slopes.empty:
        return pd.DataFrame(columns=cols)
    sub = slopes[slopes["metric"] == metric].copy()
    rows = []
    for (continent, event_type, lc), g in sub.groupby(["continent", "event_type", "land_cover_group"], dropna=False):
        vals = g["sen_slope"].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append({
            "continent": continent,
            "event_type": event_type,
            "land_cover_group": lc,
            "group": safe_group_label(continent, event_type),
            "median_slope": float(np.nanmedian(vals)),
            "n_station": int(vals.size),
            "q25": float(np.nanpercentile(vals, 25)),
            "q75": float(np.nanpercentile(vals, 75)),
        })
    return pd.DataFrame(rows, columns=cols)

# ------------------------------
# Plot helpers
# ------------------------------

def tidy_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis in ("x", "both"):
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.7)
    if grid_axis in ("y", "both"):
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.7)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.04) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=32, fontweight="bold", ha="left", va="bottom")


def robust_symmetric_limits(values: np.ndarray, floor: float) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -floor, floor
    q = np.nanpercentile(np.abs(arr), 95)
    lim = max(float(q), floor)
    return -lim, lim


def lon_label(x: float) -> str:
    if x < 0:
        return f"{int(abs(x))}°W"
    if x > 0:
        return f"{int(x)}°E"
    return "0°"


def lat_label(y: float) -> str:
    if y < 0:
        return f"{int(abs(y))}°S"
    if y > 0:
        return f"{int(y)}°N"
    return "0°"


def apply_map_ticks(ax: plt.Axes) -> None:
    xticks = [-120, -90, -60, -30, 0, 30]
    yticks = [25, 35, 45, 55, 65]
    ax.set_xticks([x for x in xticks if MAP_XMIN <= x <= MAP_XMAX])
    ax.set_yticks([y for y in yticks if MAP_YMIN <= y <= MAP_YMAX])
    ax.set_xticklabels([lon_label(x) for x in ax.get_xticks()], color="black")
    ax.set_yticklabels([lat_label(y) for y in ax.get_yticks()], color="black")
    ax.tick_params(axis="both", labelsize=18, length=4, width=0.8, colors="black")


def filter_map_points(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if "lon" in out.columns and "lat" in out.columns:
        out = out[(out["lon"] >= MAP_XMIN) & (out["lon"] <= MAP_XMAX) & (out["lat"] >= MAP_YMIN) & (out["lat"] <= MAP_YMAX)].copy()
    return out


def get_world() -> Optional["gpd.GeoDataFrame"]:
    if gpd is None:
        return None
    try:
        candidates = []
        if hasattr(gpd.datasets, "get_path"):
            try:
                candidates.append(gpd.datasets.get_path("naturalearth_lowres"))
            except Exception:
                pass
        for path in candidates:
            if path:
                return gpd.read_file(path)
    except Exception:
        return None
    return None


def draw_basemap(ax: plt.Axes) -> None:
    world = get_world()
    ax.set_facecolor("white")
    if world is not None:
        try:
            world = world.cx[MAP_XMIN:MAP_XMAX, MAP_YMIN:MAP_YMAX]
        except Exception:
            pass
        world.plot(ax=ax, color=PALETTE["land"], edgecolor=PALETTE["border"], linewidth=0.55, zorder=0)
    else:
        rects = [(MAP_XMIN, MAP_YMIN, MAP_XMAX-MAP_XMIN, MAP_YMAX-MAP_YMIN)]
        for x0, y0, w, h in rects:
            ax.add_patch(Rectangle((x0, y0), w, h, facecolor=PALETTE["land"], edgecolor=PALETTE["border"], linewidth=0.8, zorder=0))
    ax.set_xlim(MAP_XMIN, MAP_XMAX)
    ax.set_ylim(MAP_YMIN, MAP_YMAX)
    apply_map_ticks(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color("#9E9E9E")
        ax.spines[side].set_linewidth(0.8)


def draw_station_slope_map(ax: plt.Axes, slopes: pd.DataFrame, event_type: str, metric: str, value_col: str,
                           cmap, colorbar_label: str, size_by_years: bool = True) -> None:
    draw_basemap(ax)
    sub = slopes[(slopes["event_type"] == event_type) & (slopes["metric"] == metric)].dropna(subset=["lon", "lat", value_col]).copy()
    sub = filter_map_points(sub)
    if sub.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return
    vmin, vmax = robust_symmetric_limits(sub[value_col].to_numpy(dtype=float), floor=0.02 if value_col == "sen_slope" else 0.10)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    if size_by_years:
        years_used = sub["n_years"].clip(lower=MIN_YEARS_FOR_SLOPE).to_numpy(dtype=float)
        sizes = 14 + (years_used - MIN_YEARS_FOR_SLOPE) * 5.0
    else:
        sizes = np.repeat(34.0, len(sub))
    sc = ax.scatter(
        sub["lon"].to_numpy(dtype=float),
        sub["lat"].to_numpy(dtype=float),
        c=sub[value_col].to_numpy(dtype=float),
        s=sizes,
        cmap=cmap,
        norm=norm,
        edgecolor="white",
        linewidth=0.65,
        alpha=0.95,
        zorder=2,
    )
    ax.text(-95, 69.2, "North America", fontsize=17, color="black", ha="center")
    ax.text(15, 69.2, "Europe", fontsize=17, color="black", ha="center")
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.08, pad=0.12)
    cbar.set_label(colorbar_label, color="black")
    cbar.ax.tick_params(labelsize=17, colors="black")
    if size_by_years:
        vals = [4, 8, 12]
        handles = [
            plt.scatter([], [], s=14 + (v - MIN_YEARS_FOR_SLOPE) * 5.0, facecolor="#BDBDBD", edgecolor="white", linewidth=0.7)
            for v in vals
        ]
        leg = ax.legend(
            handles,
            [str(v) for v in vals],
            title="Years/station",
            loc="center",
            bbox_to_anchor=(0.48, 0.21),
            frameon=False,
            ncol=1,
            fontsize=15,
            title_fontsize=16,
            handletextpad=0.6,
            columnspacing=1.0,
            borderaxespad=0.0,
        )
        plt.setp(leg.get_texts(), color="black")
        if leg.get_title() is not None:
            leg.get_title().set_color("black")


def draw_trend_small_multiples(fig: plt.Figure, subspec, yearly: pd.DataFrame) -> None:
    inner = subspec.subgridspec(2, 2, wspace=0.18, hspace=0.36)
    order = [("North America", "Isolated"), ("North America", "Continuous"), ("Europe", "Isolated"), ("Europe", "Continuous")]
    titles = ["North America | isolated", "North America | continuous", "Europe | isolated", "Europe | continuous"]
    axes = []
    for idx, ((cont, ev), title) in enumerate(zip(order, titles)):
        ax = fig.add_subplot(inner[idx // 2, idx % 2])
        axes.append(ax)
        sub = yearly[(yearly["continent"] == cont) & (yearly["event_type"] == ev)].sort_values("year")
        if sub.empty:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue
        color = PALETTE["na"] if cont == "North America" else PALETTE["eu"]
        fill = PALETTE["na_soft"] if cont == "North America" else PALETTE["eu_soft"]
        x = sub["year"].to_numpy(dtype=float)
        med = sub["median"].to_numpy(dtype=float)
        q25 = sub["q25"].to_numpy(dtype=float)
        q75 = sub["q75"].to_numpy(dtype=float)
        nst = sub["n_station"].to_numpy(dtype=float)
        ax.fill_between(x, q25, q75, color=fill, alpha=0.85, linewidth=0)
        ax.plot(x, med, color=color, linewidth=2.5)
        ax.scatter(x, med, s=14 + 0.42 * nst, color=color, alpha=0.28, edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_title(title, fontsize=18, loc="left", pad=2)
        ax.set_xlim(max(YEAR_MIN, x.min()) - 0.3, min(YEAR_MAX, x.max()) + 0.3)
        if idx // 2 == 1:
            ax.set_xlabel("Year", labelpad=4)
        else:
            ax.set_xlabel("")
        ax.set_ylabel("")
        tidy_axes(ax, grid_axis="both")
        ax.text(0.98, 0.08, f"n max = {int(np.nanmax(nst))}", transform=ax.transAxes, ha="right", color=PALETTE["muted"], fontsize=14)
    if axes:
        left = min(ax.get_position().x0 for ax in axes)
        right = max(ax.get_position().x1 for ax in axes)
        bottom = min(ax.get_position().y0 for ax in axes)
        top = max(ax.get_position().y1 for ax in axes)
        fig.text(left - 0.045, (bottom + top) / 2, "Station-year median Max RR", rotation=90, va="center", ha="center", fontsize=20)


def kde_curve(values: np.ndarray, xgrid: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return np.zeros_like(xgrid)
    sd = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
    bw = 1.06 * sd * (vals.size ** (-1/5)) if sd > 0 else max((xgrid.max() - xgrid.min()) / 50, 1e-3)
    bw = max(bw, 1e-3)
    z = (xgrid[:, None] - vals[None, :]) / bw
    dens = np.exp(-0.5 * z**2).sum(axis=1) / (vals.size * bw * np.sqrt(2*np.pi))
    return dens


def draw_slope_ridges(ax: plt.Axes, slopes: pd.DataFrame, metric: str = "max_rr") -> None:
    order = ["NA Iso", "NA Cont", "EU Iso", "EU Cont"]
    label_to_filter = {
        "NA Iso": ("North America", "Isolated"),
        "NA Cont": ("North America", "Continuous"),
        "EU Iso": ("Europe", "Isolated"),
        "EU Cont": ("Europe", "Continuous"),
    }
    sub = slopes[slopes["metric"] == metric].copy()
    vals_all = sub["sen_slope"].dropna().to_numpy(dtype=float)
    xlim = robust_symmetric_limits(vals_all, floor=0.03)
    xgrid = np.linspace(xlim[0], xlim[1], 300)
    y_positions = np.arange(len(order))[::-1] + 1
    for pos, lab in zip(y_positions, order):
        cont, ev = label_to_filter[lab]
        arr = sub[(sub["continent"] == cont) & (sub["event_type"] == ev)]["sen_slope"].dropna().to_numpy(dtype=float)
        if arr.size == 0:
            continue
        dens = kde_curve(arr, xgrid)
        if dens.max() > 0:
            dens = dens / dens.max() * 0.65
        color = PALETTE["na"] if cont == "North America" else PALETTE["eu"]
        fill = PALETTE["na_soft"] if cont == "North America" else PALETTE["eu_soft"]
        ax.fill_between(xgrid, pos, pos + dens, color=fill, alpha=0.95, linewidth=0)
        ax.plot(xgrid, pos + dens, color=color, linewidth=2.0)
        med = float(np.nanmedian(arr))
        q25, q75 = np.nanpercentile(arr, [25, 75])
        ax.hlines(pos, q25, q75, color=color, linewidth=5)
        ax.scatter([med], [pos], s=85, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.text(xlim[0] - 0.02 * (xlim[1] - xlim[0]), pos, lab, ha="right", va="center", fontsize=18)
    ax.axvline(0, color=PALETTE["muted"], linewidth=1.0, linestyle="--")
    ax.set_yticks([])
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xlabel("Station-level Sen slope of Max RR (year$^{-1}$)")
    tidy_axes(ax, grid_axis="x")


def draw_early_late_raincloud(ax: plt.Axes, slopes: pd.DataFrame, continent: str, metric: str = "max_rr", event_type: str = "Isolated") -> None:
    sub = slopes[(slopes["continent"] == continent) & (slopes["metric"] == metric) & (slopes["event_type"] == event_type)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return
    color = PALETTE["na"] if continent == "North America" else PALETTE["eu"]
    fill = PALETTE["na_soft"] if continent == "North America" else PALETTE["eu_soft"]
    rng = np.random.default_rng(0)
    early = sub["early_value"].dropna().to_numpy(dtype=float)
    late = sub["late_value"].dropna().to_numpy(dtype=float)
    x_positions = [0.9, 1.9]
    data = [early, late]
    vp = ax.violinplot(data, positions=x_positions, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(fill)
        body.set_edgecolor("none")
        body.set_alpha(0.7)
    for xpos, arr in zip(x_positions, data):
        if arr.size == 0:
            continue
        jitter = rng.normal(0, 0.045, size=arr.size)
        ax.scatter(np.full(arr.size, xpos) + jitter, arr, s=16, color=color, alpha=0.20, edgecolor="none")
        q25, med, q75 = np.nanpercentile(arr, [25, 50, 75])
        ax.vlines(xpos, q25, q75, color=color, linewidth=3)
        ax.hlines(med, xpos-0.16, xpos+0.16, color="white", linewidth=3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Early", "Late"])
    ax.set_title(continent, fontsize=20, loc="left")
    ax.set_ylabel("Station-year median Max RR")
    tidy_axes(ax, grid_axis="y")


def draw_interval_summary(ax: plt.Axes, intervals: pd.DataFrame) -> None:
    if intervals.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return
    order = [
        ("North America", "Isolated", "max_rr", "NA Iso, Max"),
        ("North America", "Isolated", "day3_rr", "NA Iso, Day 3"),
        ("North America", "Continuous", "max_rr", "NA Cont, Max"),
        ("North America", "Continuous", "day3_rr", "NA Cont, Day 3"),
        ("Europe", "Isolated", "max_rr", "EU Iso, Max"),
        ("Europe", "Isolated", "day3_rr", "EU Iso, Day 3"),
        ("Europe", "Continuous", "max_rr", "EU Cont, Max"),
        ("Europe", "Continuous", "day3_rr", "EU Cont, Day 3"),
    ]
    ypos = np.arange(len(order))[::-1]
    for y, (cont, ev, metric, label) in zip(ypos, order):
        sub = intervals[(intervals["continent"] == cont) & (intervals["event_type"] == ev) & (intervals["metric"] == metric)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        color = PALETTE["na"] if cont == "North America" else PALETTE["eu"]
        alpha = 1.0 if metric == "max_rr" else 0.55
        ax.hlines(y, r["q10"], r["q90"], color=color, linewidth=1.8, alpha=alpha)
        ax.hlines(y, r["q25"], r["q75"], color=color, linewidth=6.0, alpha=alpha)
        ax.scatter(r["median_slope"], y, s=68, color=color, alpha=alpha, edgecolor="white", linewidth=0.8, zorder=4)
    ax.axvline(0, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([t[3] for t in order])
    ax.set_xlabel("Station-level Sen slope (year$^{-1}$)")
    tidy_axes(ax, grid_axis="x")


def draw_coverage_heatmap(ax: plt.Axes, coverage: pd.DataFrame) -> None:
    order = ["NA Iso", "NA Cont", "EU Iso", "EU Cont"]
    pivot = coverage.pivot(index="group", columns="year", values="n_station").reindex(order)
    if pivot.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return
    img = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    years = list(pivot.columns)
    sel = np.linspace(0, len(years)-1, min(6, len(years))).round().astype(int)
    ax.set_xticks(sel)
    ax.set_xticklabels([str(years[i]) for i in sel])
    ax.set_xlabel("Year")
    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Stations represented")
    cbar.ax.tick_params(labelsize=18)
    for i in sel:
        ax.axvline(i - 0.5, color="white", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_landcover_dot_matrix(ax: plt.Axes, matrix_df: pd.DataFrame) -> None:
    if matrix_df.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return
    landcovers = ["Forest", "Cropland", "Grassland", "Urban", "Wetland", "Bareland"]
    groups = ["NA Iso", "NA Cont", "EU Iso", "EU Cont"]
    grid = matrix_df.copy()
    vmin, vmax = robust_symmetric_limits(grid["median_slope"].to_numpy(dtype=float), floor=0.015)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    for yi, lc in enumerate(landcovers[::-1]):
        ax.axhline(yi, color=PALETTE["grid"], linewidth=0.6, zorder=0)
    for xi, g in enumerate(groups):
        ax.axvline(xi, color=PALETTE["grid"], linewidth=0.6, zorder=0)
    for _, row in grid.iterrows():
        if row["land_cover_group"] not in landcovers or row["group"] not in groups:
            continue
        x = groups.index(row["group"])
        y = landcovers[::-1].index(row["land_cover_group"])
        ax.scatter(x, y, s=40 + row["n_station"] * 20, c=[row["median_slope"]], cmap=SLOPE_CMAP, norm=norm,
                   edgecolor="white", linewidth=0.9, alpha=0.95, zorder=3)
        ax.hlines(y, x-0.18, x+0.18, color=PALETTE["ink"], linewidth=0.9, alpha=0.55)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_yticks(range(len(landcovers)))
    ax.set_yticklabels(landcovers[::-1])
    ax.set_xlim(-0.5, len(groups)-0.5)
    ax.set_ylim(-0.5, len(landcovers)-0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=SLOPE_CMAP, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.08, pad=0.10)
    cbar.set_label("Median station Sen slope", fontsize=19)
    cbar.ax.tick_params(labelsize=15)


def draw_station_trajectory_small_multiples(fig: plt.Figure, subspec, summary: pd.DataFrame, metric: str = "max_rr") -> None:
    ax = fig.add_subplot(subspec)
    pooled = summary[(summary["event_type"] == "Isolated") & np.isfinite(summary[metric])][metric].to_numpy(dtype=float)
    if pooled.size:
        ymin = float(np.nanpercentile(pooled, 1))
        ymax = float(np.nanpercentile(pooled, 99))
        pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.1
        ylim = (ymin - pad, ymax + pad)
    else:
        ylim = (0.0, 1.0)

    for continent, color, label in [
        ("North America", PALETTE["na"], "North America"),
        ("Europe", PALETTE["eu"], "Europe"),
    ]:
        sub = summary[(summary["continent"] == continent) & (summary["event_type"] == "Isolated")].copy()
        if sub.empty:
            continue
        for _, ss in sub.groupby("station"):
            sss = ss.sort_values("year")
            x = sss["year"].to_numpy(dtype=float)
            y = sss[metric].to_numpy(dtype=float)
            ax.plot(x, y, color=color, alpha=0.045, linewidth=1.0, zorder=1)
        yearly = sub.groupby("year")[metric].median().reset_index().sort_values("year")
        x = yearly["year"].to_numpy(dtype=float)
        y = yearly[metric].to_numpy(dtype=float)
        if len(y) >= 5:
            ys = pd.Series(y).rolling(window=5, center=True, min_periods=1).median().to_numpy(dtype=float)
        else:
            ys = y
        ax.plot(x, ys, color=color, linewidth=3.8, zorder=4, label=label)

    ax.set_xlim(YEAR_MIN - 0.5, YEAR_MAX + 0.5)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Year")
    ax.set_ylabel("Station-year median Max RR")
    tidy_axes(ax, grid_axis="both")
    ax.legend(frameon=False, loc="upper right", ncol=1, fontsize=18)


def selected_year_ridges(ax: plt.Axes, summary: pd.DataFrame, continent: str, metric: str = "max_rr") -> None:
    sub = summary[(summary["continent"] == continent) & (summary["event_type"] == "Isolated")].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return
    counts = sub.groupby("year")["station"].nunique().reset_index(name="n_station").sort_values("year")
    min_stations = 20 if continent == "North America" else 10
    valid_years = counts.loc[counts["n_station"] >= min_stations, "year"].to_numpy(dtype=int)
    if valid_years.size == 0:
        valid_years = counts["year"].to_numpy(dtype=int)
    picks_idx = np.linspace(0, len(valid_years)-1, min(5, len(valid_years))).round().astype(int)
    picks = valid_years[picks_idx]
    vals_all = sub[metric].dropna().to_numpy(dtype=float)
    xmin, xmax = np.nanpercentile(vals_all, [1, 99]) if vals_all.size else (0, 1)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        xmin, xmax = -0.2, 1.2
    xgrid = np.linspace(xmin, xmax, 260)
    positions = np.arange(len(picks))[::-1] + 1
    color = PALETTE["na"] if continent == "North America" else PALETTE["eu"]
    fill = PALETTE["na_soft"] if continent == "North America" else PALETTE["eu_soft"]
    for y0, yr in zip(positions, picks):
        arr = sub.loc[sub["year"] == yr, metric].dropna().to_numpy(dtype=float)
        if arr.size == 0:
            continue
        dens = kde_curve(arr, xgrid)
        if dens.max() > 0:
            dens = dens / dens.max() * 0.72
        ax.fill_between(xgrid, y0, y0 + dens, color=fill, alpha=0.95)
        ax.plot(xgrid, y0 + dens, color=color, linewidth=2.2)
        ax.text(xgrid[0], y0 + 0.06, str(int(yr)), color=PALETTE["muted"], fontsize=16)
    ax.set_title(continent, loc="left", fontsize=22)
    ax.set_xlabel("Station-year median Max RR")
    ax.set_yticks([])
    tidy_axes(ax, grid_axis="x")


def _draw_sensitivity_intervals_metric(ax: plt.Axes, summary: pd.DataFrame, metric: str, title: str) -> None:
    settings = [
        (1, 4, "Main"),
        (2, 4, "≥2 events"),
        (3, 4, "≥3 events"),
        (1, 6, "≥6 years"),
        (1, 8, "≥8 years"),
        (2, 6, "≥2 events & ≥6 years"),
    ]
    ypos = np.arange(len(settings))[::-1]
    for y0, (thr_events, thr_years, label) in zip(ypos, settings):
        ss = summary[summary["event_count"] >= thr_events]
        slopes = station_slopes(ss, metric=metric, min_years=thr_years)
        if slopes.empty:
            continue
        for continent, color, dy in [("North America", PALETTE["na"], -0.10), ("Europe", PALETTE["eu"], 0.10)]:
            arr = slopes[(slopes["continent"] == continent) & (slopes["event_type"] == "Isolated")]["sen_slope"].dropna().to_numpy(dtype=float)
            if arr.size == 0:
                continue
            q05, q10, q25, med, q75, q90, q95 = np.nanpercentile(arr, [5, 10, 25, 50, 75, 90, 95])
            ax.hlines(y0 + dy, q05, q95, color=color, linewidth=1.2, alpha=0.55)
            ax.hlines(y0 + dy, q10, q90, color=color, linewidth=2.0, alpha=0.75)
            ax.hlines(y0 + dy, q25, q75, color=color, linewidth=6.0, alpha=1.0)
            ax.scatter([med], [y0 + dy], s=80, color=color, edgecolor="white", linewidth=0.9, zorder=4)
    ax.axvline(0, color=PALETTE["muted"], linestyle="--", linewidth=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for _, _, lab in settings])
    ax.set_xlabel("Station-level Sen slope (year$^{-1}$)")
    ax.set_title(title, loc="left", fontsize=22)
    tidy_axes(ax, grid_axis="x")

def draw_sensitivity_intervals(fig: plt.Figure, subspec, summary: pd.DataFrame) -> None:
    inner = subspec.subgridspec(1, 2, wspace=0.20)
    ax1 = fig.add_subplot(inner[0, 0])
    _draw_sensitivity_intervals_metric(ax1, summary, metric="max_rr", title="Max RR")
    ax1.legend(handles=[
        Line2D([0], [0], color=PALETTE["na"], linewidth=5, label="North America"),
        Line2D([0], [0], color=PALETTE["eu"], linewidth=5, label="Europe"),
    ], frameon=False, loc="lower right")
    ax2 = fig.add_subplot(inner[0, 1], sharey=ax1)
    if "day3_rr" in summary.columns and summary["day3_rr"].notna().any():
        _draw_sensitivity_intervals_metric(ax2, summary, metric="day3_rr", title="Day 3 RR")
    else:
        ax2.text(0.5, 0.5, "Day 3 RR not available", transform=ax2.transAxes, ha="center", va="center")
        ax2.set_title("Day 3 RR", loc="left", fontsize=22)
    ax2.tick_params(labelleft=False)


# ------------------------------
# Figure builders
# ------------------------------

def make_main_figure(output_dir: Path, slopes_all: pd.DataFrame, yearly_max: pd.DataFrame, intervals: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.96, 1.10], height_ratios=[1.05, 0.95], wspace=0.34, hspace=0.26)

    ax1 = fig.add_subplot(gs[0, 0])
    draw_station_slope_map(ax1, slopes_all, event_type="Isolated", metric="max_rr", value_col="sen_slope", cmap=SLOPE_CMAP,
                           colorbar_label="Station Sen slope of Max RR (year$^{-1}$)")
    add_panel_label(ax1, "a")

    draw_trend_small_multiples(fig, gs[0, 1], yearly_max)
    trend_axes = fig.axes[-4:]
    if trend_axes:
        add_panel_label(trend_axes[0], "b")

    ax3 = fig.add_subplot(gs[1, 0])
    draw_slope_ridges(ax3, slopes_all, metric="max_rr")
    add_panel_label(ax3, "c")

    ax4 = fig.add_subplot(gs[1, 1])
    draw_interval_summary(ax4, intervals)
    add_panel_label(ax4, "d")

    fig.savefig(output_dir / "Figure1_result1_main_refined.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def make_structure_figure(output_dir: Path, slopes_all: pd.DataFrame, coverage: pd.DataFrame, matrix_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.98, 1.10], height_ratios=[1.00, 1.00], wspace=0.28, hspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    draw_station_slope_map(ax1, slopes_all, event_type="Isolated", metric="max_rr", value_col="late_minus_early", cmap=DELTA_CMAP,
                           colorbar_label="Late minus early station median Max RR")
    add_panel_label(ax1, "a")

    ax2 = fig.add_subplot(gs[0, 1])
    draw_coverage_heatmap(ax2, coverage)
    add_panel_label(ax2, "b")

    ax3 = fig.add_subplot(gs[1, 0])
    draw_landcover_dot_matrix(ax3, matrix_df)
    add_panel_label(ax3, "c")

    inner = gs[1, 1].subgridspec(1, 2, wspace=0.32)
    ax4a = fig.add_subplot(inner[0, 0])
    draw_early_late_raincloud(ax4a, slopes_all, continent="North America")
    add_panel_label(ax4a, "d")
    ax4b = fig.add_subplot(inner[0, 1])
    draw_early_late_raincloud(ax4b, slopes_all, continent="Europe")
    ax4b.set_ylabel("")

    fig.savefig(output_dir / "Figure2_result1_structure_refined.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def make_supplementary_figures(output_dir: Path, summary: pd.DataFrame, slopes_all: pd.DataFrame) -> None:
    # S1 trajectories
    fig = plt.figure(figsize=(20, 9.5))
    outer = GridSpec(1, 1, figure=fig)
    draw_station_trajectory_small_multiples(fig, outer[0], summary, metric="max_rr")
    if len(fig.axes) >= 1:
        add_panel_label(fig.axes[0], "a")
    fig.savefig(output_dir / "Supp_Fig1_result1_station_trajectories_refined.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # S2 selected-year ridges
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(1, 2, figure=fig, wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    selected_year_ridges(ax1, summary, continent="North America", metric="max_rr")
    add_panel_label(ax1, "a")
    ax2 = fig.add_subplot(gs[0, 1])
    selected_year_ridges(ax2, summary, continent="Europe", metric="max_rr")
    add_panel_label(ax2, "b")
    fig.savefig(output_dir / "Supp_Fig2_result1_selected_year_ridges_refined.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # S3 sensitivity
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(1, 1, figure=fig)
    draw_sensitivity_intervals(fig, gs[0], summary)
    if len(fig.axes) >= 1:
        add_panel_label(fig.axes[0], "a")
    if len(fig.axes) >= 2:
        add_panel_label(fig.axes[1], "b")
    fig.savefig(output_dir / "Supp_Fig3_result1_sensitivity_refined.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # S4 early-late maps Max and Day3
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(1, 2, figure=fig, wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    slopes_max = slopes_all[slopes_all["metric"] == "max_rr"].copy()
    draw_station_slope_map(ax1, slopes_max, event_type="Isolated", metric="max_rr", value_col="late_minus_early", cmap=DELTA_CMAP,
                           colorbar_label="Late minus early Max RR")
    add_panel_label(ax1, "a")
    ax2 = fig.add_subplot(gs[0, 1])
    slopes_day3 = slopes_all[slopes_all["metric"] == "day3_rr"].copy() if "day3_rr" in slopes_all["metric"].unique() else pd.DataFrame()
    if slopes_day3.empty:
        ax2.text(0.5, 0.5, "Day 3 RR not available", transform=ax2.transAxes, ha="center", va="center")
    else:
        draw_station_slope_map(ax2, slopes_day3, event_type="Isolated", metric="day3_rr", value_col="late_minus_early", cmap=DELTA_CMAP,
                               colorbar_label="Late minus early Day 3 RR")
    add_panel_label(ax2, "b")
    fig.savefig(output_dir / "Supp_Fig4_result1_early_late_maps_refined.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

# ------------------------------
# Export tables
# ------------------------------

def export_tables(output_dir: Path, events: pd.DataFrame, summary: pd.DataFrame, slopes_all: pd.DataFrame,
                  yearly_max: pd.DataFrame, intervals: pd.DataFrame, coverage: pd.DataFrame, matrix_df: pd.DataFrame) -> None:
    events.to_csv(output_dir / "events_cleaned.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "station_year_summary.csv", index=False, encoding="utf-8-sig")
    slopes_all.to_csv(output_dir / "station_slopes_all_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_max.to_csv(output_dir / "yearly_station_medians_maxrr.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(output_dir / "slope_interval_table.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / "coverage_heatmap_table.csv", index=False, encoding="utf-8-sig")
    matrix_df.to_csv(output_dir / "landcover_slope_matrix.csv", index=False, encoding="utf-8-sig")

# ------------------------------
# Main
# ------------------------------

def main() -> None:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_DIR
    output_dir = input_dir / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_excel_files(input_dir)
    summary = build_station_year_summary(events)

    slope_frames = []
    for metric in [m for m in ["max_rr", "day3_rr"] if m in summary.columns]:
        slope_frames.append(station_slopes(summary, metric=metric, min_years=MIN_YEARS_FOR_SLOPE))
    slopes_all = pd.concat(slope_frames, ignore_index=True, sort=False) if slope_frames else pd.DataFrame()

    yearly_max = yearly_station_medians(summary, metric="max_rr")
    intervals = slope_interval_table(slopes_all)
    coverage = coverage_heatmap_table(summary)
    matrix_df = landcover_matrix(slopes_all, metric="max_rr")

    export_tables(output_dir, events, summary, slopes_all, yearly_max, intervals, coverage, matrix_df)
    make_main_figure(output_dir, slopes_all, yearly_max, intervals)
    make_structure_figure(output_dir, slopes_all[slopes_all["metric"] == "max_rr"].copy(), coverage, matrix_df)
    make_supplementary_figures(output_dir, summary, slopes_all)
    print(f"Done. Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
