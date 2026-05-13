#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Result 3 main analysis for an NCC-style manuscript:
Land cover and soil texture reshape the response surface of event-scale
rainfall retention efficiency (Max RR).

Primary design choices
----------------------
1) Primary sample: all available rainfall-event records (isolated + continuous).
2) Primary outcome: Max RR (周期最大ratio).
3) Event-type contrasts and Day 3 RR are treated as robustness checks.
4) Main inference: GEE with station-level clustering + spline interactions.
5) Main figure: land-cover response curves + within-land-cover texture modifiers.

The script is written to work directly with the user's Excel files under:
D:\nature\全球数据分析最终版\欧美两地分析最先版\最新包含每个温度的全部数据

Outputs
-------
- Result3_main_ncc.png
- Supp_Fig_Result3_sample_structure.png
- Supp_Fig_Result3_robustness.png
- result3_analysis_dataset.csv
- result3_landcover_gee_summary.txt
- result3_landcover_wald_terms.csv
- result3_texture_wald_terms.csv
- result3_key_contrasts.csv
- result3_sample_counts.csv

Author: OpenAI ChatGPT
"""

from __future__ import annotations

import os
import re
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import statsmodels.api as sm
from patsy import bs
from statsmodels.genmod.cov_struct import Exchangeable

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------
@dataclass
class Config:
    input_root: str = r"D:\nature\全球数据分析最终版\欧美两地分析最先版\最新包含每个温度的全部数据"
    output_root: str = r"D:\nature\全球数据分析最终版\欧美两地分析最先版\最新包含每个温度的全部数据\Result3_NCC_outputs"

    primary_event_type: str = "All"        # primary sample: use all data unless changed
    min_sites_main: int = 8                 # robust support threshold for inferential GEE
    min_events_main: int = 300              # robust support threshold for inferential GEE
    min_events_texture: int = 500           # robust threshold for texture-specific GEE fits
    min_sites_texture: int = 6              # robust threshold for texture-specific GEE fits
    min_events_texture_bin: int = 8         # fallback descriptive bin threshold for texture curves
    display_curve_min_events: int = 8        # minimum events to draw descriptive land-cover curves
    display_curve_min_sites: int = 1         # minimum station count to draw descriptive land-cover curves
    display_surface_min_events: int = 12     # minimum events to draw descriptive 2D surfaces
    climate_min_sites: int = 4              # minimum station count for climate summary figure
    climate_top_n: int = 8                  # maximum number of climate classes to show

    max_plausible_event_rain_mm: float = 300.0
    max_plausible_temp_c: float = 50.0
    min_plausible_temp_c: float = -20.0
    max_plausible_sm: float = 1.0
    min_plausible_sm: float = 0.0

    # Robust filtering on Max RR using MAD, not arbitrary sign truncation.
    mad_cutoff: float = 8.0

    random_seed: int = 42

    # Styling
    dpi: int = 400
    font_family: str = "Arial"
    base_fontsize: int = 34


CFG = Config()
np.random.seed(CFG.random_seed)


# -----------------------------------------------------------------------------
# Plot style (clean NCC-like, restrained, no heavy decorations)
# -----------------------------------------------------------------------------
def set_plot_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [CFG.font_family, "DejaVu Sans", "Liberation Sans"],
        "font.size": CFG.base_fontsize,
        "axes.labelsize": CFG.base_fontsize + 2,
        "axes.titlesize": CFG.base_fontsize + 2,
        "axes.linewidth": 0.8,
        "xtick.labelsize": CFG.base_fontsize,
        "ytick.labelsize": CFG.base_fontsize,
        "legend.fontsize": CFG.base_fontsize - 2,
        "figure.titlesize": CFG.base_fontsize + 4,
        "savefig.dpi": CFG.dpi,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "lines.linewidth": 4.2,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


LAND_COLORS = {
    "Cropland": "#B07D36",
    "Forest": "#2E7D32",
    "Grassland": "#8E9A3A",
    "Urban": "#6D6D6D",
    "Wetland": "#2A9D8F",
    "Bareland": "#8C564B",
}

TEXTURE_COLORS = {
    "Sand": "#D9A066",
    "Loam": "#8C6D4F",
    "Clay": "#5A4637",
}

PANEL_LETTERS = list("abcdefghi")


# -----------------------------------------------------------------------------
# Column handling
# -----------------------------------------------------------------------------
COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "start_date": ["周期开始日期", "start_date"],
    "end_date": ["周期结束日期", "end_date"],
    "first_rain_date": ["第一次降雨事件日期", "first_rain_date"],
    "first_day_rr": ["第一次降雨当天rainfall_ratio", "first_day_rr", "第一天ratio"],
    "last_effective_rain_date": ["最后一次有效降雨事件日期", "last_effective_rain_date"],
    "day1_rr": ["第一天ratio", "day1_rr"],
    "day2_rr": ["第二天ratio", "day2_rr"],
    "day3_rr": ["第三天ratio", "day3_rr"],
    "day4_rr": ["第四天ratio", "day4_rr"],
    "day5_rr": ["第五天ratio", "day5_rr"],
    "day6_rr": ["第六天ratio", "day6_rr"],
    "max_rr": ["周期最大ratio", "max_rr", "周期最大 ratio"],
    "event_rain": ["总降雨量（>5）", "总降雨量(>5)", "event_rain", "总降雨量>5"],
    "event_max_rain": ["周期最大降雨量", "event_max_rain"],
    "temp_mean": ["表面温度平均值", "surface_temp_mean", "temp_mean"],
    "temp_condition": ["温度条件", "temp_condition"],
    "saturation": ["saturation", "饱和度"],
    "clay_fraction": ["clay_fraction", "clay"],
    "organic_carbon": ["organic_carbon", "oc", "SOC"],
    "sand_fraction": ["sand_fraction", "sand"],
    "silt_fraction": ["silt_fraction", "silt"],
    "land_cover_raw": ["land_cover", "landcover"],
    "climate": ["climate"],
    "soil_moisture": ["soil_moisture", "sm", "soil moisture"],
    "station": ["station", "site", "站点"],
    "lon": ["lon", "longitude", "经度"],
    "lat": ["lat", "latitude", "纬度"],
}


def normalize_colname(col: str) -> str:
    s = str(col).strip()
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("\n", "").replace("\r", "")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def find_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    norm_map = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for std_name, cands in COLUMN_CANDIDATES.items():
        hit = find_column(df, cands)
        if hit is not None:
            rename_map[hit] = std_name
    return df.rename(columns=rename_map)


# -----------------------------------------------------------------------------
# File-level metadata
# -----------------------------------------------------------------------------
def infer_region_from_filename(name: str) -> str:
    if "美洲" in name:
        return "North America"
    if "欧洲" in name:
        return "Europe"
    return "Unknown"


def infer_event_type_from_filename(name: str) -> str:
    if "单独" in name:
        return "Isolated"
    if "连续" in name:
        return "Continuous"
    return "Unknown"


def infer_landcover_from_filename(name: str) -> str:
    if "草地" in name:
        return "Grassland"
    if "城市" in name:
        return "Urban"
    if "裸地" in name:
        return "Bareland"
    if "农田" in name:
        return "Cropland"
    if "森林" in name:
        return "Forest"
    if "湿地" in name:
        return "Wetland"
    return "Unknown"


# -----------------------------------------------------------------------------
# Texture classification
# -----------------------------------------------------------------------------
def classify_texture_simple(sand: float, clay: float, silt: float) -> str:
    """
    Simplified but robust three-class texture grouping for manuscript-level analysis.
    The goal is stable inference, not full USDA taxonomic granularity.
    """
    if pd.isna(sand) or pd.isna(clay) or pd.isna(silt):
        return np.nan
    if clay >= 35:
        return "Clay"
    if sand >= 60 and clay < 20:
        return "Sand"
    return "Loam"


# -----------------------------------------------------------------------------
# Data loading and QC
# -----------------------------------------------------------------------------
def load_single_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = rename_columns(df)
    df["source_file"] = path.name
    df["region"] = infer_region_from_filename(path.name)
    df["event_type"] = infer_event_type_from_filename(path.name)
    df["land_cover_file"] = infer_landcover_from_filename(path.name)
    return df


def coalesce_land_cover(row: pd.Series) -> str:
    raw = str(row.get("land_cover_raw", "")).strip().lower()
    if raw:
        if any(k in raw for k in ["grass", "grassland", "草地"]):
            return "Grassland"
        if any(k in raw for k in ["urban", "city", "城市"]):
            return "Urban"
        if any(k in raw for k in ["bare", "裸地"]):
            return "Bareland"
        if any(k in raw for k in ["crop", "farmland", "agri", "农田"]):
            return "Cropland"
        if any(k in raw for k in ["forest", "wood", "森林"]):
            return "Forest"
        if any(k in raw for k in ["wet", "marsh", "湿地"]):
            return "Wetland"
    return row.get("land_cover_file", "Unknown")


def mad_filter(series: pd.Series, cutoff: float) -> pd.Series:
    s = series.dropna()
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series(True, index=series.index)
    robust_z = 0.6745 * (series - med) / mad
    return robust_z.abs() <= cutoff


def load_all_data(cfg: Config) -> pd.DataFrame:
    root = Path(cfg.input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root}")

    files = sorted(root.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No xlsx files found under: {root}")

    frames = []
    for fp in files:
        try:
            frames.append(load_single_file(fp))
        except Exception as e:
            print(f"[WARN] Failed to read {fp.name}: {e}")

    if not frames:
        raise RuntimeError("No readable Excel files were loaded.")

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Parse dates if present
    for col in ["start_date", "end_date", "first_rain_date", "last_effective_rain_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric columns
    numeric_cols = [
        "first_day_rr", "day1_rr", "day2_rr", "day3_rr", "day4_rr", "day5_rr", "day6_rr",
        "max_rr", "event_rain", "event_max_rain", "temp_mean", "saturation",
        "clay_fraction", "organic_carbon", "sand_fraction", "silt_fraction",
        "soil_moisture", "lon", "lat"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Core harmonization
    df["land_cover"] = df.apply(coalesce_land_cover, axis=1)
    df["texture3"] = [
        classify_texture_simple(s, c, si)
        for s, c, si in zip(df.get("sand_fraction"), df.get("clay_fraction"), df.get("silt_fraction"))
    ]

    if "station" not in df.columns:
        raise KeyError("A station column is required but was not found.")
    df["station"] = df["station"].astype(str).str.strip()
    df["station_code"] = pd.factorize(df["station"])[0].astype(int)

    if "start_date" in df.columns:
        df["year"] = df["start_date"].dt.year
        df["month"] = df["start_date"].dt.month
        df["doy"] = df["start_date"].dt.dayofyear
    else:
        df["year"] = np.nan
        df["month"] = np.nan
        df["doy"] = np.nan

    # Primary plausibility filters
    required = ["max_rr", "event_rain", "temp_mean", "soil_moisture", "station", "land_cover", "region", "event_type"]
    required = [c for c in required if c in df.columns]
    df = df.dropna(subset=required).copy()

    df = df[df["event_rain"].between(0.01, cfg.max_plausible_event_rain_mm)]
    df = df[df["temp_mean"].between(cfg.min_plausible_temp_c, cfg.max_plausible_temp_c)]
    df = df[df["soil_moisture"].between(cfg.min_plausible_sm, cfg.max_plausible_sm)]
    df = df[df["land_cover"].isin(["Cropland", "Forest", "Grassland", "Urban", "Wetland", "Bareland"])]
    df = df[df["region"].isin(["North America", "Europe"])]
    df = df[df["event_type"].isin(["Isolated", "Continuous"])]

    # Robust outlier removal on Max RR only; retain negatives because the metric is not bounded to positive values.
    keep = mad_filter(df["max_rr"], cfg.mad_cutoff)
    df = df[keep].copy()

    # Derived variables
    df["y_asinh"] = np.arcsinh(df["max_rr"])
    df["month"] = df["month"].fillna(7).astype(int)
    df["year"] = df["year"].fillna(-1).astype(int)

    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Category selection and summaries
# -----------------------------------------------------------------------------
def choose_main_landcovers(df_primary: pd.DataFrame, cfg: Config) -> List[str]:
    """
    For the revised Result 3 display we keep all six land-cover types whenever
    they are present in the data. Sparse classes are retained visually, but they
    should still be interpreted cautiously in the manuscript text.
    """
    present = set(df_primary["land_cover"].dropna().astype(str).unique().tolist())
    preferred = ["Cropland", "Forest", "Grassland", "Urban", "Wetland", "Bareland"]
    main = [x for x in preferred if x in present]
    return main


def write_sample_counts(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    counts = (
        df.groupby(["region", "event_type", "land_cover", "texture3"])
        .agg(n_events=("max_rr", "size"), n_sites=("station", "nunique"))
        .reset_index()
        .sort_values(["event_type", "region", "land_cover", "texture3"])
    )
    counts.to_csv(out_dir / "result3_sample_counts.csv", index=False, encoding="utf-8-sig")
    return counts


# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------
def fit_landcover_gee(df: pd.DataFrame, main_landcovers: List[str]):
    data = df[df["land_cover"].isin(main_landcovers)].copy()
    data["land_cover"] = pd.Categorical(data["land_cover"], categories=main_landcovers, ordered=True)
    data["texture3"] = pd.Categorical(data["texture3"], categories=["Sand", "Loam", "Clay"], ordered=True)
    data["region"] = pd.Categorical(data["region"], categories=["North America", "Europe"], ordered=True)

    formula = (
        "y_asinh ~ C(land_cover) * ("
        "bs(event_rain, df=5, include_intercept=False) + "
        "bs(soil_moisture, df=5, include_intercept=False) + "
        "bs(temp_mean, df=4, include_intercept=False)"
        ") + C(texture3) + C(region) + C(month)"
    )

    model = sm.GEE.from_formula(
        formula=formula,
        groups="station_code",
        data=data,
        family=sm.families.Gaussian(),
        cov_struct=Exchangeable(),
    )
    result = model.fit(maxiter=200)
    return data, result


def fit_texture_gees(df: pd.DataFrame, main_landcovers: List[str]) -> Dict[str, Tuple[pd.DataFrame, object]]:
    out: Dict[str, Tuple[pd.DataFrame, object]] = {}
    for land in main_landcovers:
        sub = df[df["land_cover"] == land].copy()
        texture_counts = sub["texture3"].value_counts(dropna=False)
        texture_sites = sub.groupby("texture3")["station"].nunique()
        keep_textures = [
            t for t in ["Sand", "Loam", "Clay"]
            if texture_counts.get(t, 0) >= CFG.min_events_texture and texture_sites.get(t, 0) >= CFG.min_sites_texture
        ]
        if len(keep_textures) < 2:
            continue

        sub = sub[sub["texture3"].isin(keep_textures)].copy()
        sub["texture3"] = pd.Categorical(sub["texture3"], categories=keep_textures, ordered=True)
        sub["region"] = pd.Categorical(sub["region"], categories=["North America", "Europe"], ordered=True)

        formula = (
            "y_asinh ~ C(texture3) * ("
            "bs(soil_moisture, df=4, include_intercept=False) + "
            "bs(event_rain, df=4, include_intercept=False)"
            ") + C(region) + C(month)"
        )
        try:
            model = sm.GEE.from_formula(
                formula=formula,
                groups="station_code",
                data=sub,
                family=sm.families.Gaussian(),
                cov_struct=Exchangeable(),
            )
            result = model.fit(maxiter=200)
            out[land] = (sub, result)
        except Exception as e:
            print(f"[WARN] Texture GEE failed for {land}: {e}")
    return out


# -----------------------------------------------------------------------------
# Predictions and contrasts
# -----------------------------------------------------------------------------
def safe_prediction_frame(result, new_df: pd.DataFrame) -> pd.DataFrame:
    pred = result.get_prediction(new_df)
    sf = pred.summary_frame(alpha=0.05)
    sf = sf.rename(columns={
        "mean": "mean_link",
        "mean_ci_lower": "lower_link",
        "mean_ci_upper": "upper_link",
    })
    sf["mean"] = np.sinh(sf["mean_link"])
    sf["lower"] = np.sinh(sf["lower_link"])
    sf["upper"] = np.sinh(sf["upper_link"])
    return sf


def make_reference_row(df: pd.DataFrame) -> Dict[str, object]:
    ref = {
        "event_rain": float(df["event_rain"].median()),
        "soil_moisture": float(df["soil_moisture"].median()),
        "temp_mean": float(df["temp_mean"].median()),
        "texture3": df["texture3"].mode(dropna=True).iloc[0] if df["texture3"].notna().any() else "Loam",
        "region": df["region"].mode(dropna=True).iloc[0],
        "month": int(df["month"].mode(dropna=True).iloc[0]),
        "station_code": int(df["station_code"].mode(dropna=True).iloc[0]),
    }
    return ref


def _interp_curve(points: pd.DataFrame, predictor: str, n: int = 180) -> pd.DataFrame:
    pts = points.sort_values(predictor).copy()
    for col in ["mean", "lower", "upper"]:
        pts[col] = (
            pd.to_numeric(pts[col], errors="coerce")
            .rolling(3, center=True, min_periods=1)
            .mean()
        )
    x = pd.to_numeric(pts[predictor], errors="coerce").to_numpy(dtype=float)
    mean = pd.to_numeric(pts["mean"], errors="coerce").to_numpy(dtype=float)
    lower = pd.to_numeric(pts["lower"], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(pts["upper"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(mean) & np.isfinite(lower) & np.isfinite(upper)
    x = x[ok]; mean = mean[ok]; lower = lower[ok]; upper = upper[ok]
    if x.size < 2:
        return pd.DataFrame()
    order = np.argsort(x)
    x = x[order]; mean = mean[order]; lower = lower[order]; upper = upper[order]
    x_new = np.linspace(float(x.min()), float(x.max()), n)
    mean_new = np.interp(x_new, x, mean)
    lower_new = np.interp(x_new, x, lower)
    upper_new = np.interp(x_new, x, upper)
    lower_new = np.minimum(lower_new, mean_new)
    upper_new = np.maximum(upper_new, mean_new)
    return pd.DataFrame({predictor: x_new, "mean": mean_new, "lower": lower_new, "upper": upper_new})


def descriptive_landcover_curve_grid(df: pd.DataFrame, landcovers: List[str], predictor: str, n_bins: int = 8, n: int = 180) -> pd.DataFrame:
    rows = []
    use = df[df["land_cover"].isin(landcovers)].copy()
    use = use[np.isfinite(pd.to_numeric(use[predictor], errors="coerce"))].copy()
    for lc in landcovers:
        sub = use[use["land_cover"] == lc].copy()
        if len(sub) < CFG.display_curve_min_events or sub["station"].nunique() < CFG.display_curve_min_sites:
            continue
        x = pd.to_numeric(sub[predictor], errors="coerce")
        y = pd.to_numeric(sub["max_rr"], errors="coerce")
        sub = sub[np.isfinite(x) & np.isfinite(y)].copy()
        if sub.empty:
            continue
        lo_x = float(sub[predictor].quantile(0.02))
        hi_x = float(sub[predictor].quantile(0.98))
        sub = sub[(sub[predictor] >= lo_x) & (sub[predictor] <= hi_x)].copy()
        if sub.empty:
            continue
        pts = []
        uniq = int(pd.to_numeric(sub[predictor], errors="coerce").nunique())
        if uniq >= 3 and len(sub) >= 12:
            try:
                q = min(n_bins, max(2, uniq))
                sub["bin"] = pd.qcut(sub[predictor], q=q, duplicates="drop")
            except Exception:
                sub["bin"] = pd.cut(sub[predictor], bins=min(6, max(2, uniq)), duplicates="drop")
            for _, sb in sub.groupby("bin", observed=True):
                center = float(pd.to_numeric(sb[predictor], errors="coerce").median())
                per_station = sb.groupby("station", observed=True)["max_rr"].median().dropna().to_numpy(dtype=float)
                if per_station.size == 0:
                    continue
                med, lo, hi = _bootstrap_median_ci(per_station, n_boot=300)
                pts.append({predictor: center, "mean": med, "lower": lo, "upper": hi})
        if len(pts) < 2:
            per_station = sub.groupby("station", observed=True)["max_rr"].median().dropna().to_numpy(dtype=float)
            if per_station.size == 0:
                continue
            med, lo, hi = _bootstrap_median_ci(per_station, n_boot=300)
            pts = [
                {predictor: float(lo_x), "mean": med, "lower": lo, "upper": hi},
                {predictor: float(hi_x), "mean": med, "lower": lo, "upper": hi},
            ]
        pts = pd.DataFrame(pts)
        grid = _interp_curve(pts, predictor, n=n)
        if grid.empty:
            continue
        grid["land_cover"] = lc
        grid["predictor"] = predictor
        rows.append(grid)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_landcover_display_grid(df: pd.DataFrame, landcovers: List[str], predictor: str, n: int = 180) -> pd.DataFrame:
    return descriptive_landcover_curve_grid(df, landcovers, predictor=predictor, n_bins=8, n=n)


def landcover_response_grid(data: pd.DataFrame, result, main_landcovers: List[str], predictor: str, n: int = 200) -> pd.DataFrame:
    ref = make_reference_row(data)
    x = np.linspace(float(data[predictor].quantile(0.02)), float(data[predictor].quantile(0.98)), n)

    frames = []
    for lc in main_landcovers:
        grid = pd.DataFrame({predictor: x})
        for col, val in ref.items():
            if col not in grid.columns:
                grid[col] = val
        grid["land_cover"] = lc

        pred = safe_prediction_frame(result, grid)
        tmp = pd.concat([grid.reset_index(drop=True), pred[["mean", "lower", "upper"]]], axis=1)
        tmp["land_cover"] = lc
        tmp["predictor"] = predictor
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


def texture_response_grid(texture_models: Dict[str, Tuple[pd.DataFrame, object]], predictor: str = "soil_moisture", n: int = 160) -> pd.DataFrame:
    frames = []
    for land, (sub, result) in texture_models.items():
        ref = make_reference_row(sub)
        textures = list(sub["texture3"].cat.categories)
        q_lo = []
        q_hi = []
        for tx0 in textures:
            s0 = sub.loc[sub["texture3"] == tx0, predictor].dropna()
            if len(s0) == 0:
                continue
            q_lo.append(float(s0.quantile(0.05)))
            q_hi.append(float(s0.quantile(0.95)))
        if q_lo and q_hi and max(q_lo) < min(q_hi):
            lo_x, hi_x = max(q_lo), min(q_hi)
        else:
            lo_x, hi_x = float(sub[predictor].quantile(0.10)), float(sub[predictor].quantile(0.90))
        x = np.linspace(lo_x, hi_x, n)

        for tx in textures:
            grid = pd.DataFrame({predictor: x})
            for col, val in ref.items():
                if col not in grid.columns:
                    grid[col] = val
            if "event_rain" not in grid.columns:
                grid["event_rain"] = ref["event_rain"]
            if "soil_moisture" not in grid.columns:
                grid["soil_moisture"] = ref["soil_moisture"]
            grid["texture3"] = tx
            pred = safe_prediction_frame(result, grid)
            tmp = pd.concat([grid.reset_index(drop=True), pred[["mean", "lower", "upper"]]], axis=1)
            tmp["land_cover"] = land
            tmp["texture3"] = tx
            frames.append(tmp)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def descriptive_texture_grid(df: pd.DataFrame, landcovers: List[str], predictor: str = "soil_moisture", n_bins: int = 5) -> pd.DataFrame:
    rows = []
    use = df[df["land_cover"].isin(landcovers) & df["texture3"].isin(["Sand", "Loam", "Clay"])].copy()
    for land in landcovers:
        sub_land = use[use["land_cover"] == land].copy()
        if sub_land.empty:
            continue
        for tx in ["Sand", "Loam", "Clay"]:
            sub = sub_land[sub_land["texture3"] == tx].copy()
            if len(sub) < CFG.min_events_texture_bin or sub["station"].nunique() < 1:
                continue
            try:
                sub["bin"] = pd.qcut(sub[predictor], q=min(n_bins, max(2, sub[predictor].nunique())), duplicates="drop")
            except Exception:
                continue
            if sub["bin"].nunique() < 2:
                continue
            pts = []
            for _, sb in sub.groupby("bin", observed=True):
                center = float(pd.to_numeric(sb[predictor], errors="coerce").median())
                per_station = sb.groupby("station", observed=True)["max_rr"].median().dropna().to_numpy(dtype=float)
                if per_station.size == 0:
                    continue
                med, lo, hi = _bootstrap_median_ci(per_station, n_boot=400)
                pts.append({predictor: center, "mean": med, "lower": lo, "upper": hi})
            pts = pd.DataFrame(pts)
            if len(pts) < 2:
                continue
            grid = _interp_curve(pts, predictor, n=120)
            if grid.empty:
                continue
            grid["land_cover"] = land
            grid["texture3"] = tx
            grid["source"] = "descriptive_bin"
            rows.append(grid)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_texture_display_grid(df: pd.DataFrame, texture_models: Dict[str, Tuple[pd.DataFrame, object]], landcovers: List[str], predictor: str = "soil_moisture", n: int = 160) -> pd.DataFrame:
    model_grid = texture_response_grid(texture_models, predictor=predictor, n=n)
    if not model_grid.empty:
        # Filter obviously unstable model outputs before plotting.
        model_grid = model_grid[np.isfinite(model_grid["mean"]) & np.isfinite(model_grid["lower"]) & np.isfinite(model_grid["upper"])].copy()
        model_grid = model_grid[model_grid[["mean", "lower", "upper"]].abs().max(axis=1) < 5].copy()
        model_grid["source"] = "model"
    desc_grid = descriptive_texture_grid(df, landcovers, predictor=predictor, n_bins=6)
    pieces = []
    available_model_lands = set(model_grid["land_cover"].unique()) if not model_grid.empty else set()
    if not model_grid.empty:
        pieces.append(model_grid)
    if not desc_grid.empty:
        desc_grid = desc_grid[~desc_grid["land_cover"].isin(available_model_lands)].copy()
        if not desc_grid.empty:
            pieces.append(desc_grid)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def descriptive_surface_grid(df: pd.DataFrame, landcovers: List[str], n_rain: int = 24, n_sm: int = 24, min_cell_events: int = 2) -> pd.DataFrame:
    rows = []
    use = df[df["land_cover"].isin(landcovers)].copy()
    use = use[np.isfinite(pd.to_numeric(use["event_rain"], errors="coerce")) & np.isfinite(pd.to_numeric(use["soil_moisture"], errors="coerce")) & np.isfinite(pd.to_numeric(use["max_rr"], errors="coerce"))].copy()
    for lc in landcovers:
        sub = use[use["land_cover"] == lc].copy()
        if len(sub) < CFG.display_surface_min_events:
            continue
        x_lo, x_hi = float(sub["event_rain"].quantile(0.05)), float(sub["event_rain"].quantile(0.95))
        y_lo, y_hi = float(sub["soil_moisture"].quantile(0.05)), float(sub["soil_moisture"].quantile(0.95))
        if not (np.isfinite(x_lo) and np.isfinite(x_hi) and np.isfinite(y_lo) and np.isfinite(y_hi)):
            continue
        if x_hi <= x_lo or y_hi <= y_lo:
            continue
        sub = sub[(sub["event_rain"] >= x_lo) & (sub["event_rain"] <= x_hi) & (sub["soil_moisture"] >= y_lo) & (sub["soil_moisture"] <= y_hi)].copy()
        n_rain_use = min(n_rain, max(6, int(np.sqrt(len(sub)))))
        n_sm_use = min(n_sm, max(6, int(np.sqrt(len(sub)))))
        x_edges = np.linspace(x_lo, x_hi, n_rain_use + 1)
        y_edges = np.linspace(y_lo, y_hi, n_sm_use + 1)
        sub["x_bin"] = pd.cut(sub["event_rain"], bins=x_edges, labels=False, include_lowest=True)
        sub["y_bin"] = pd.cut(sub["soil_moisture"], bins=y_edges, labels=False, include_lowest=True)
        grp = sub.groupby(["y_bin", "x_bin"], observed=True).agg(n=("max_rr", "size"), med=("max_rr", "median")).reset_index()
        grp = grp[grp["n"] >= min_cell_events].copy()
        if grp.empty:
            continue
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        for _, r in grp.iterrows():
            xb = int(r["x_bin"]); yb = int(r["y_bin"])
            rows.append({
                "land_cover": lc,
                "event_rain": float(x_centers[xb]),
                "soil_moisture": float(y_centers[yb]),
                "mean": float(r["med"]),
            })
    return pd.DataFrame(rows)


def compute_key_contrasts(data: pd.DataFrame, main_landcovers: List[str], land_result, texture_models) -> pd.DataFrame:
    rows = []

    # Land-cover contrasts across the 10th-90th percentile range of each predictor
    for predictor in ["event_rain", "soil_moisture", "temp_mean"]:
        grid = landcover_response_grid(data, land_result, main_landcovers, predictor, n=120)
        for lc, g in grid.groupby("land_cover"):
            g = g.sort_values(predictor)
            lo = g.iloc[0]
            hi = g.iloc[-1]
            rows.append({
                "analysis": "land_cover_curve_span",
                "group": lc,
                "predictor": predictor,
                "contrast": f"{predictor}: 2nd->98th percentile",
                "delta_max_rr": hi["mean"] - lo["mean"],
            })

    # Texture contrasts within land covers (soil moisture only; most stable for interpretation)
    for land, (sub, res) in texture_models.items():
        g = texture_response_grid({land: (sub, res)}, predictor="soil_moisture", n=120)
        if g.empty:
            continue
        ref_sm = np.median(sub["soil_moisture"])
        idx = (g["soil_moisture"] - ref_sm).abs().groupby([g["land_cover"], g["texture3"]]).idxmin()
        at_ref = g.loc[idx]
        pivot = at_ref.pivot(index="land_cover", columns="texture3", values="mean")
        cols = list(pivot.columns)
        if len(cols) >= 2:
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    rows.append({
                        "analysis": "texture_reference_contrast",
                        "group": land,
                        "predictor": "soil_moisture",
                        "contrast": f"{cols[j]} - {cols[i]} @ median SM",
                        "delta_max_rr": float(pivot.loc[land, cols[j]] - pivot.loc[land, cols[i]]),
                    })

    out = pd.DataFrame(rows)
    return out


def _bootstrap_median_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    med = float(np.nanmedian(values))
    if values.size == 1:
        return med, med, med
    idx = np.random.randint(0, values.size, size=(n_boot, values.size))
    boots = np.nanmedian(values[idx], axis=1)
    lo = float(np.nanpercentile(boots, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return med, lo, hi


def summarize_metric_with_ci(df: pd.DataFrame, group_cols: list[str], metric_col: str, station_col: str = "station", n_boot: int = 400) -> pd.DataFrame:
    """
    Fast descriptive summary with 95% CIs.

    Important: support-figure uncertainty is computed at the station level,
    not by re-sampling every event row. For each group, we first collapse to one
    median value per station, then bootstrap those station medians. This is much
    faster than row-level cluster bootstrap and is aligned with the manuscript's
    descriptive aim for grouped support figures.
    """
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(sub[metric_col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue

        if station_col in sub.columns:
            tmp = sub[[station_col, metric_col]].copy()
            tmp[station_col] = tmp[station_col].astype(str)
            tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
            tmp = tmp.dropna(subset=[station_col, metric_col])
            station_meds = tmp.groupby(station_col, observed=True)[metric_col].median().to_numpy(dtype=float)
        else:
            station_meds = np.array([], dtype=float)

        if station_meds.size >= 5:
            med = float(np.nanmedian(station_meds))
            idx = np.random.randint(0, station_meds.size, size=(n_boot, station_meds.size))
            boots = np.nanmedian(station_meds[idx], axis=1)
            lo = float(np.nanpercentile(boots, 2.5))
            hi = float(np.nanpercentile(boots, 97.5))
        else:
            med, lo, hi = _bootstrap_median_ci(vals, n_boot=max(300, n_boot))

        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: k for c, k in zip(group_cols, keys)}
        row.update({"median_rr": med, "ci_low": lo, "ci_high": hi, "n": int(vals.size)})
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def add_panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=CFG.base_fontsize + 4,
            fontweight="bold", va="bottom", ha="left")


def format_axes(ax, xlabel: str, ylabel: str = "Predicted Max RR") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(length=3.5)


def plot_landcover_curves(ax, grid: pd.DataFrame, predictor: str, xlabel: str, main_landcovers: List[str], legend: bool = False) -> None:
    for lc in main_landcovers:
        sub = grid[(grid["land_cover"] == lc) & (grid["predictor"] == predictor)].sort_values(predictor)
        if sub.empty:
            continue
        color = LAND_COLORS.get(lc, "#333333")
        x = pd.to_numeric(sub[predictor], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["mean"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(sub["lower"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(sub["upper"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if ok.sum() < 2:
            continue
        ax.plot(x[ok], y[ok], color=color, lw=4.6, label=lc)
        ax.fill_between(x[ok], lo[ok], hi[ok], color=color, alpha=0.18, linewidth=0)
    format_axes(ax, xlabel)
    if legend:
        ax.legend(frameon=False, ncol=1, loc="best")



def plot_texture_panel(ax, tex_grid: pd.DataFrame, land_cover: str, xlabel: str = "Antecedent soil moisture (m$^3$ m$^{-3}$)") -> None:
    sub0 = tex_grid[tex_grid["land_cover"] == land_cover].copy()
    if sub0.empty:
        format_axes(ax, xlabel)
        ax.set_title(land_cover, pad=4)
        ax.text(0.5, 0.5, "Sparse texture\nsupport", transform=ax.transAxes, ha="center", va="center", fontsize=CFG.base_fontsize-4)
        return
    for tx in ["Sand", "Loam", "Clay"]:
        sub = sub0[sub0["texture3"] == tx].sort_values("soil_moisture")
        if sub.empty:
            continue
        color = TEXTURE_COLORS.get(tx, "#555555")
        x = pd.to_numeric(sub["soil_moisture"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub["mean"], errors="coerce").to_numpy(dtype=float)
        lo = pd.to_numeric(sub["lower"], errors="coerce").to_numpy(dtype=float)
        hi = pd.to_numeric(sub["upper"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        if ok.sum() < 2:
            continue
        ax.plot(x[ok], y[ok], color=color, lw=4.6, label=tx)
        ax.fill_between(x[ok], lo[ok], hi[ok], color=color, alpha=0.18, linewidth=0)
    format_axes(ax, xlabel)
    ax.set_title(land_cover, pad=4)



def make_main_figure(curves: Dict[str, pd.DataFrame], tex_grid: pd.DataFrame,
                     main_landcovers: List[str], out_dir: Path) -> None:
    fig = plt.figure(figsize=(30.0, 32.0))
    gs = fig.add_gridspec(3, 3, left=0.06, right=0.985, bottom=0.12, top=0.97, wspace=0.34, hspace=0.42)

    # Top row: land-cover response curves (all available land covers)
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    plot_landcover_curves(axes_top[0], curves["event_rain"], "event_rain", "Event rainfall (mm)", main_landcovers, legend=True)
    plot_landcover_curves(axes_top[1], curves["soil_moisture"], "soil_moisture", "Antecedent soil moisture (m$^3$ m$^{-3}$)", main_landcovers)
    plot_landcover_curves(axes_top[2], curves["temp_mean"], "temp_mean", "Mean surface temperature (°C)", main_landcovers)

    # Lower two rows: one texture panel per land-cover type
    tex_axes = [fig.add_subplot(gs[1, i]) for i in range(3)] + [fig.add_subplot(gs[2, i]) for i in range(3)]
    for ax, lc in zip(tex_axes, main_landcovers):
        plot_texture_panel(ax, tex_grid, lc)
    for ax in tex_axes[len(main_landcovers):]:
        ax.axis("off")

    all_axes = axes_top + tex_axes
    for i, ax in enumerate(all_axes[:len(PANEL_LETTERS)]):
        add_panel_label(ax, PANEL_LETTERS[i])

    tex_handles = [Line2D([0], [0], color=TEXTURE_COLORS[k], lw=3.0, label=k) for k in ["Sand", "Loam", "Clay"]]
    fig.legend(handles=tex_handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.50, 0.03))

    fig.savefig(out_dir / "Result3_main_ncc.png", dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)



def make_sample_structure_figure(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Station counts must be based on unique stations, not summed across isolated/continuous
    subsets. Event counts are shown separately by texture using all records.
    """
    plot_df = df[df["land_cover"].isin(["Cropland", "Forest", "Grassland", "Urban", "Wetland", "Bareland"])].copy()

    land_summary = (
        plot_df.groupby("land_cover")
        .agg(n_sites=("station", "nunique"), n_events=("max_rr", "size"))
        .reset_index()
        .sort_values("n_sites", ascending=False)
    )

    tex_summary = (
        plot_df[plot_df["texture3"].isin(["Sand", "Loam", "Clay"])]
        .groupby(["land_cover", "texture3"])
        .agg(n_events=("max_rr", "size"))
        .reset_index()
    )

    fig = plt.figure(figsize=(24.0, 11.0))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, bottom=0.26, top=0.94, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    order = land_summary["land_cover"].tolist()

    ax1.bar(order, land_summary["n_sites"].to_numpy(dtype=float), color=[LAND_COLORS.get(x, "#888") for x in order], alpha=0.92)
    ax1.set_ylabel("Number of stations")
    ax1.set_xlabel("Land cover")
    ax1.tick_params(axis="x", rotation=26)
    for lab in ax1.get_xticklabels():
        lab.set_ha("right")
    add_panel_label(ax1, "a")

    bottoms = np.zeros(len(order), dtype=float)
    for tx in ["Sand", "Loam", "Clay"]:
        vals = []
        for lc in order:
            hit = tex_summary[(tex_summary["land_cover"] == lc) & (tex_summary["texture3"] == tx)]
            vals.append(float(hit["n_events"].sum()) if not hit.empty else 0.0)
        vals = np.asarray(vals, dtype=float)
        ax2.bar(order, vals, bottom=bottoms, color=TEXTURE_COLORS[tx], label=tx)
        bottoms = bottoms + vals
    ax2.set_ylabel("Number of events")
    ax2.set_xlabel("Land cover")
    ax2.tick_params(axis="x", rotation=26)
    for lab in ax2.get_xticklabels():
        lab.set_ha("right")
    ax2.legend(frameon=False, ncol=3, loc="upper right")
    add_panel_label(ax2, "b")

    fig.savefig(out_dir / "Supp_Fig_Result3_sample_structure.png", dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)



def make_robustness_figure(df: pd.DataFrame, out_dir: Path) -> None:
    """
    A compact support figure to show that the main Result 3 pattern is stable to:
    1) event type (isolated vs continuous)
    2) response metric (Max RR vs Day 3 RR)
    Median summaries are accompanied by 95% station-cluster bootstrap CIs.
    """
    plot_df = df.copy()
    plot_df = plot_df[plot_df["land_cover"].isin(["Cropland", "Forest", "Grassland", "Urban", "Wetland", "Bareland"])].copy()
    plot_df["metric"] = "Max RR"
    plot_df["y_plot"] = pd.to_numeric(plot_df["max_rr"], errors="coerce")

    pieces = [plot_df]
    if "day3_rr" in plot_df.columns:
        alt = plot_df.copy()
        alt["metric"] = "Day 3 RR"
        alt["y_plot"] = pd.to_numeric(alt["day3_rr"], errors="coerce")
        pieces.append(alt)
    plot_df = pd.concat(pieces, ignore_index=True)
    plot_df = plot_df[np.isfinite(plot_df["y_plot"])].copy()

    summ = summarize_metric_with_ci(plot_df, ["event_type", "metric", "land_cover"], "y_plot", station_col="station")

    fig = plt.figure(figsize=(24.0, 11.0))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, bottom=0.26, top=0.94, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Max RR by event type
    for et, ls in zip(["Isolated", "Continuous"], ["-", "--"]):
        s = summ[(summ["metric"] == "Max RR") & (summ["event_type"] == et)].copy()
        order = [x for x in ["Cropland", "Forest", "Grassland", "Urban", "Wetland", "Bareland"] if x in s["land_cover"].tolist()]
        if s.empty or not order:
            continue
        med = [float(s.loc[s["land_cover"] == x, "median_rr"].iloc[0]) for x in order]
        lo = [float(s.loc[s["land_cover"] == x, "ci_low"].iloc[0]) for x in order]
        hi = [float(s.loc[s["land_cover"] == x, "ci_high"].iloc[0]) for x in order]
        med_a = np.asarray(med, dtype=float)
        lo_a = np.asarray(lo, dtype=float)
        hi_a = np.asarray(hi, dtype=float)
        yerr = np.vstack([med_a - lo_a, hi_a - med_a])
        ax1.errorbar(order, med_a, yerr=yerr, linestyle=ls, marker="o", color="#333333", label=et, capsize=4, elinewidth=2.4)
    ax1.set_ylabel("Median Max RR")
    ax1.set_xlabel("Land cover")
    ax1.tick_params(axis="x", rotation=28)
    for lab in ax1.get_xticklabels():
        lab.set_ha("right")
    ax1.legend(frameon=False, loc="best")
    add_panel_label(ax1, "a")

    # Max RR vs Day 3 RR within isolated events
    for metric, color in zip(["Max RR", "Day 3 RR"], ["#333333", "#A23B72"]):
        s = summ[(summ["metric"] == metric) & (summ["event_type"] == "Isolated")].copy()
        if s.empty:
            continue
        order = [x for x in ["Cropland", "Forest", "Grassland", "Urban", "Wetland", "Bareland"] if x in s["land_cover"].tolist()]
        if not order:
            continue
        med = [float(s.loc[s["land_cover"] == x, "median_rr"].iloc[0]) for x in order]
        lo = [float(s.loc[s["land_cover"] == x, "ci_low"].iloc[0]) for x in order]
        hi = [float(s.loc[s["land_cover"] == x, "ci_high"].iloc[0]) for x in order]
        med_a = np.asarray(med, dtype=float)
        lo_a = np.asarray(lo, dtype=float)
        hi_a = np.asarray(hi, dtype=float)
        yerr = np.vstack([med_a - lo_a, hi_a - med_a])
        ax2.errorbar(order, med_a, yerr=yerr, marker="o", color=color, label=metric, capsize=4, elinewidth=2.4)
    ax2.set_ylabel("Median rainfall-ratio metric")
    ax2.set_xlabel("Land cover")
    ax2.tick_params(axis="x", rotation=28)
    for lab in ax2.get_xticklabels():
        lab.set_ha("right")
    ax2.legend(frameon=False, loc="best")
    add_panel_label(ax2, "b")

    fig.savefig(out_dir / "Supp_Fig_Result3_robustness.png", dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)




def make_region_stratified_figure(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df[df["land_cover"].isin(["Cropland", "Forest", "Grassland", "Urban", "Wetland", "Bareland"])].copy()
    order = [x for x in ["Cropland", "Forest", "Grassland", "Urban", "Wetland", "Bareland"] if x in plot_df["land_cover"].unique()]
    fig = plt.figure(figsize=(24.0, 11.0))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, bottom=0.26, top=0.94, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    summ1 = summarize_metric_with_ci(plot_df, ["region", "land_cover"], "max_rr", station_col="station")
    reg_colors = {"North America": "#333333", "Europe": "#8E3B6A"}
    for region in ["North America", "Europe"]:
        s = summ1[summ1["region"] == region]
        if s.empty:
            continue
        vals = [float(s.loc[s["land_cover"] == x, "median_rr"].iloc[0]) if x in s["land_cover"].tolist() else np.nan for x in order]
        lo = [float(s.loc[s["land_cover"] == x, "ci_low"].iloc[0]) if x in s["land_cover"].tolist() else np.nan for x in order]
        hi = [float(s.loc[s["land_cover"] == x, "ci_high"].iloc[0]) if x in s["land_cover"].tolist() else np.nan for x in order]
        vals_a = np.array(vals, dtype=float); lo_a = np.array(lo, dtype=float); hi_a = np.array(hi, dtype=float)
        yerr = np.vstack([vals_a - lo_a, hi_a - vals_a])
        ax1.errorbar(order, vals_a, yerr=yerr, marker="o", lw=4.0, color=reg_colors[region], label=region, capsize=4, elinewidth=2.4)
    ax1.set_ylabel("Median Max RR")
    ax1.set_xlabel("Land cover")
    ax1.tick_params(axis="x", rotation=28)
    for lab in ax1.get_xticklabels():
        lab.set_ha("right")
    ax1.legend(frameon=False, loc="best")
    add_panel_label(ax1, "a")

    summ2 = summarize_metric_with_ci(plot_df, ["region", "event_type", "land_cover"], "max_rr", station_col="station")
    ls_map = {"Isolated": "-", "Continuous": "--"}
    for (region, et), s in summ2.groupby(["region", "event_type"]):
        vals = [float(s.loc[s["land_cover"] == x, "median_rr"].iloc[0]) if x in s["land_cover"].tolist() else np.nan for x in order]
        lo = [float(s.loc[s["land_cover"] == x, "ci_low"].iloc[0]) if x in s["land_cover"].tolist() else np.nan for x in order]
        hi = [float(s.loc[s["land_cover"] == x, "ci_high"].iloc[0]) if x in s["land_cover"].tolist() else np.nan for x in order]
        vals_a = np.array(vals, dtype=float); lo_a = np.array(lo, dtype=float); hi_a = np.array(hi, dtype=float)
        yerr = np.vstack([vals_a - lo_a, hi_a - vals_a])
        ax2.errorbar(order, vals_a, yerr=yerr, marker="o", lw=3.8, linestyle=ls_map.get(et, '-'), color=reg_colors.get(region, '#444444'), label=f"{region} | {et}", capsize=3, elinewidth=2.2)
    ax2.set_ylabel("Median Max RR")
    ax2.set_xlabel("Land cover")
    ax2.tick_params(axis="x", rotation=28)
    for lab in ax2.get_xticklabels():
        lab.set_ha("right")
    ax2.legend(frameon=False, ncol=1, loc="upper left", bbox_to_anchor=(0.01, 0.99))
    add_panel_label(ax2, "b")

    fig.savefig(out_dir / "Supp_Fig_Result3_region_stratified.png", dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)


def landcover_surface_grid(data: pd.DataFrame, result, main_landcovers: List[str], n_rain: int = 120, n_sm: int = 120) -> pd.DataFrame:
    ref = make_reference_row(data)
    rain = np.linspace(float(data["event_rain"].quantile(0.05)), float(data["event_rain"].quantile(0.95)), n_rain)
    smv = np.linspace(float(data["soil_moisture"].quantile(0.05)), float(data["soil_moisture"].quantile(0.95)), n_sm)
    rr, ss = np.meshgrid(rain, smv)
    frames = []
    for lc in main_landcovers:
        grid = pd.DataFrame({
            "event_rain": rr.ravel(),
            "soil_moisture": ss.ravel(),
            "temp_mean": ref["temp_mean"],
            "texture3": ref["texture3"],
            "region": ref["region"],
            "month": ref["month"],
            "station_code": ref["station_code"],
            "land_cover": lc,
        })
        pred = safe_prediction_frame(result, grid)
        tmp = pd.concat([grid.reset_index(drop=True), pred[["mean"]]], axis=1)
        tmp["land_cover"] = lc
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_surface_figure(surface_df: pd.DataFrame, landcovers: List[str], out_dir: Path) -> None:
    if surface_df.empty:
        return
    landcovers = [x for x in landcovers if x in ["Cropland", "Forest", "Grassland", "Urban", "Wetland", "Bareland"]]
    n = len(landcovers)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(24.0, 10.0 * nrows), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.10, top=0.95, wspace=0.25, hspace=0.28)
    vmin = float(surface_df["mean"].quantile(0.02))
    vmax = float(surface_df["mean"].quantile(0.98))
    mappable = None
    for i, lc in enumerate(landcovers):
        ax = axes.ravel()[i]
        sub = surface_df[surface_df["land_cover"] == lc].copy()
        piv = sub.pivot_table(index="soil_moisture", columns="event_rain", values="mean")
        piv = piv.sort_index().sort_index(axis=1)
        piv = piv.interpolate(axis=0, limit_direction="both").interpolate(axis=1, limit_direction="both")
        z = piv.to_numpy(dtype=float)
        z = np.clip(z, vmin, vmax)
        x = piv.columns.to_numpy(dtype=float)
        y = piv.index.to_numpy(dtype=float)
        mappable = ax.pcolormesh(x, y, z, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Event rainfall (mm)")
        ax.set_ylabel("Antecedent soil moisture (m$^3$ m$^{-3}$)")
        ax.set_title(lc, pad=6)
        add_panel_label(ax, PANEL_LETTERS[i])
    for ax in axes.ravel()[len(landcovers):]:
        ax.axis("off")
    if mappable is not None:
        cax = fig.add_axes([0.94, 0.18, 0.020, 0.64])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label("Predicted Max RR")
    fig.savefig(out_dir / "Supp_Fig_Result3_two_way_surfaces.png", dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)


def make_texture_summary_figure(df: pd.DataFrame, main_landcovers: List[str], out_dir: Path) -> None:
    plot_df = df[df["land_cover"].isin(main_landcovers)].copy()
    plot_df = plot_df[plot_df["texture3"].isin(["Sand", "Loam", "Clay"])].copy()
    summ = summarize_metric_with_ci(plot_df, ["land_cover", "texture3"], "max_rr", station_col="station")
    n_events = plot_df.groupby(["land_cover", "texture3"]).size().rename("n_events").reset_index()
    summ = summ.merge(n_events, on=["land_cover", "texture3"], how="left")
    if summ.empty:
        return
    fig = plt.figure(figsize=(24.0, 11.0))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, bottom=0.24, top=0.94, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    order = [x for x in main_landcovers if x in ["Cropland", "Forest", "Grassland", "Urban", "Wetland", "Bareland"]]
    for tx in ["Sand", "Loam", "Clay"]:
        s = summ[summ["texture3"] == tx]
        vals = [float(s.loc[s["land_cover"] == x, "median_rr"].iloc[0]) if x in s["land_cover"].tolist() else np.nan for x in order]
        lo = [float(s.loc[s["land_cover"] == x, "ci_low"].iloc[0]) if x in s["land_cover"].tolist() else np.nan for x in order]
        hi = [float(s.loc[s["land_cover"] == x, "ci_high"].iloc[0]) if x in s["land_cover"].tolist() else np.nan for x in order]
        vals_a = np.array(vals, dtype=float); lo_a = np.array(lo, dtype=float); hi_a = np.array(hi, dtype=float)
        yerr = np.vstack([vals_a - lo_a, hi_a - vals_a])
        ax1.errorbar(order, vals_a, yerr=yerr, marker="o", lw=4.0, color=TEXTURE_COLORS[tx], label=tx, capsize=4, elinewidth=2.4)
    ax1.set_ylabel("Median Max RR")
    ax1.set_xlabel("Land cover")
    ax1.tick_params(axis="x", rotation=28)
    for lab in ax1.get_xticklabels():
        lab.set_ha("right")
    ax1.legend(frameon=False, ncol=3, loc="best")
    add_panel_label(ax1, "a")

    width = 0.24
    xpos = np.arange(len(order))
    for j, tx in enumerate(["Sand", "Loam", "Clay"]):
        s = summ[summ["texture3"] == tx]
        vals = [float(s.loc[s["land_cover"] == x, "n_events"].iloc[0]) if x in s["land_cover"].tolist() else 0.0 for x in order]
        ax2.bar(xpos + (j-1)*width, vals, width=width, color=TEXTURE_COLORS[tx], label=tx)
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(order, rotation=28, ha="right")
    ax2.set_ylabel("Number of events")
    ax2.set_xlabel("Land cover")
    ax2.legend(frameon=False, ncol=1, loc="upper right")
    add_panel_label(ax2, "b")

    fig.savefig(out_dir / "Supp_Fig_Result3_texture_summary.png", dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)


def make_climate_summary_figure(df: pd.DataFrame, out_dir: Path) -> None:
    plot_df = df.copy()
    plot_df["climate"] = plot_df["climate"].astype(str).str.strip()
    plot_df = plot_df[plot_df["climate"].notna() & (plot_df["climate"] != "") & (plot_df["climate"].str.lower() != "nan")].copy()
    if plot_df.empty:
        return
    climate_counts = (
        plot_df.groupby("climate")
        .agg(n_events=("max_rr", "size"), n_sites=("station", "nunique"))
        .reset_index()
    )
    climate_keep = (
        climate_counts.loc[climate_counts["n_sites"] >= CFG.climate_min_sites]
        .sort_values(["n_sites", "n_events"], ascending=False)
        .head(CFG.climate_top_n)["climate"].tolist()
    )
    if len(climate_keep) < 3:
        climate_keep = climate_counts.sort_values(["n_sites", "n_events"], ascending=False).head(min(CFG.climate_top_n, len(climate_counts)))["climate"].tolist()
    plot_df = plot_df[plot_df["climate"].isin(climate_keep)].copy()
    order = climate_keep
    summ = summarize_metric_with_ci(plot_df, ["climate"], "max_rr", station_col="station")
    counts = plot_df.groupby("climate").size().rename("n_events").reset_index()
    summ = summ.merge(counts, on="climate", how="left")
    fig = plt.figure(figsize=(24.0, 11.0))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, bottom=0.28, top=0.94, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    vals = [float(summ.loc[summ["climate"] == x, "median_rr"].iloc[0]) if x in summ["climate"].tolist() else np.nan for x in order]
    lo = [float(summ.loc[summ["climate"] == x, "ci_low"].iloc[0]) if x in summ["climate"].tolist() else np.nan for x in order]
    hi = [float(summ.loc[summ["climate"] == x, "ci_high"].iloc[0]) if x in summ["climate"].tolist() else np.nan for x in order]
    vals_a = np.array(vals, dtype=float); lo_a = np.array(lo, dtype=float); hi_a = np.array(hi, dtype=float)
    yerr = np.vstack([vals_a - lo_a, hi_a - vals_a])
    ax1.errorbar(order, vals_a, yerr=yerr, marker="o", lw=4.0, color="#333333", capsize=4, elinewidth=2.4)
    ax1.set_ylabel("Median Max RR")
    ax1.set_xlabel("Climate type")
    ax1.tick_params(axis="x", rotation=28)
    for lab in ax1.get_xticklabels():
        lab.set_ha("right")
    add_panel_label(ax1, "a")

    count_vals = [float(summ.loc[summ["climate"] == x, "n_events"].iloc[0]) if x in summ["climate"].tolist() else 0.0 for x in order]
    ax2.bar(order, count_vals, color="#777777", alpha=0.9)
    ax2.set_ylabel("Number of events")
    ax2.set_xlabel("Climate type")
    ax2.tick_params(axis="x", rotation=28)
    for lab in ax2.get_xticklabels():
        lab.set_ha("right")
    add_panel_label(ax2, "b")

    fig.savefig(out_dir / "Supp_Fig_Result3_climate_summary.png", dpi=CFG.dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Text outputs
# -----------------------------------------------------------------------------
def write_model_outputs(result, texture_models, out_dir: Path) -> None:
    with open(out_dir / "result3_landcover_gee_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(result.summary()))
        f.write("\n\n")
        try:
            f.write("Wald tests by term\n")
            f.write(str(result.wald_test_terms()))
        except Exception:
            pass

    # Wald tables
    try:
        wt = result.wald_test_terms(skip_single=False).summary_frame()
        wt.to_csv(out_dir / "result3_landcover_wald_terms.csv", encoding="utf-8-sig")
    except Exception:
        pass

    tex_rows = []
    for land, (_, res) in texture_models.items():
        try:
            wf = res.wald_test_terms(skip_single=False).summary_frame().reset_index().rename(columns={"index": "term"})
            wf.insert(0, "land_cover", land)
            tex_rows.append(wf)
        except Exception:
            continue
    if tex_rows:
        pd.concat(tex_rows, ignore_index=True).to_csv(out_dir / "result3_texture_wald_terms.csv", index=False, encoding="utf-8-sig")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(cfg: Config = CFG) -> None:
    set_plot_style()

    out_dir = Path(cfg.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading and harmonizing Excel files...")
    df = load_all_data(cfg)
    df.to_csv(out_dir / "result3_analysis_dataset.csv", index=False, encoding="utf-8-sig")

    print(f"    Loaded rows after QC: {len(df):,}")
    print(f"    Unique stations: {df['station'].nunique():,}")

    print("[2/7] Writing sample counts...")
    counts = write_sample_counts(df, out_dir)

    print("[3/7] Selecting primary sample (all rainfall-event records)...")
    if str(cfg.primary_event_type).lower() == "all":
        primary = df.copy()
    else:
        primary = df[df["event_type"] == cfg.primary_event_type].copy()
    model_landcovers = choose_main_landcovers(primary, cfg)
    figure_landcovers = [x for x in ["Cropland", "Forest", "Grassland", "Urban", "Wetland", "Bareland"] if x in primary["land_cover"].unique()]
    print(f"    Model land covers: {model_landcovers}")
    print(f"    Figure land covers: {figure_landcovers}")

    print("[4/7] Fitting main GEE with spline × land-cover interactions...")
    model_data, land_res = fit_landcover_gee(primary, model_landcovers)

    print("[5/7] Fitting within-land-cover texture models...")
    texture_models = fit_texture_gees(primary[primary["land_cover"].isin(model_landcovers)].copy(), model_landcovers)

    print("[6/7] Building manuscript figures...")
    print("    -> response curves (data-constrained six-class display)")
    curves = {
        "event_rain": build_landcover_display_grid(primary, figure_landcovers, "event_rain", n=180),
        "soil_moisture": build_landcover_display_grid(primary, figure_landcovers, "soil_moisture", n=180),
        "temp_mean": build_landcover_display_grid(primary, figure_landcovers, "temp_mean", n=180),
    }
    tex_grid = build_texture_display_grid(primary, texture_models, figure_landcovers, predictor="soil_moisture", n=150)

    make_main_figure(curves, tex_grid, figure_landcovers, out_dir)
    print("    -> sample structure")
    make_sample_structure_figure(df, out_dir)
    print("    -> robustness with station-level CIs")
    make_robustness_figure(df, out_dir)
    print("    -> regional stratification with station-level CIs")
    make_region_stratified_figure(df, out_dir)
    print("    -> two-way response surfaces")
    surface_df = descriptive_surface_grid(primary, figure_landcovers, n_rain=22, n_sm=22, min_cell_events=2)
    make_surface_figure(surface_df, figure_landcovers, out_dir)
    print("    -> texture summary with station-level CIs")
    make_texture_summary_figure(df, figure_landcovers, out_dir)
    print("    -> climate summary")
    make_climate_summary_figure(df, out_dir)

    print("[7/7] Writing model summaries and key contrasts...")
    write_model_outputs(land_res, texture_models, out_dir)
    key_contrasts = compute_key_contrasts(model_data, model_landcovers, land_res, texture_models)
    key_contrasts.to_csv(out_dir / "result3_key_contrasts.csv", index=False, encoding="utf-8-sig")

    # Short execution log for the manuscript workflow.
    with open(out_dir / "README_result3_outputs.txt", "w", encoding="utf-8") as f:
        f.write(
            "Result 3 outputs generated successfully.\n\n"
            "Main figure: Result3_main_ncc.png\n"
            "Support figures: Supp_Fig_Result3_sample_structure.png, Supp_Fig_Result3_robustness.png, Supp_Fig_Result3_region_stratified.png, Supp_Fig_Result3_two_way_surfaces.png, Supp_Fig_Result3_texture_summary.png, Supp_Fig_Result3_climate_summary.png\n"
            "Primary sample: all rainfall-event records (isolated + continuous)\n"
            "Primary outcome: Max RR\n"
            "Main inference: GEE with station-level clustering and spline × land-cover interactions\n"
            "Display strategy for six land-cover classes: data-constrained descriptive curves/surfaces to avoid spline extrapolation artefacts in sparse classes\n"
            "Texture panels: within-land-cover soil-moisture response by simplified texture class (Sand/Loam/Clay)\n"
        )

    print("Done.")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
