"""
MLB WAR Prediction Pipeline — Pitcher Data Scraper
===================================================
Reads and merges three Baseball Reference pitching tables per year:
  - pitching_YEAR.xls       (value/WAR table)
  - pitch_standard_YEAR.xls (ERA, FIP, K/9, BB/9, IP, etc.)
  - pitch_advanced_YEAR.xls (K%, BB%, GB%, etc.)
"""

import os
import pandas as pd
import numpy as np

VALUE_COL_MAP = {
    "Player": "name", "Age": "age", "Team": "team",
    "IP": "IP", "G": "G", "GS": "GS",
    "RA9": "RA9", "RAA": "RAA", "WAA": "WAA",
    "WAR": "WAR", "RAR": "RAR",
}

STANDARD_COL_MAP = {
    "Player": "name", "Age": "age", "Team": "team",
    "W": "W", "L": "L", "ERA": "ERA", "G": "G", "GS": "GS",
    "GF": "GF", "SV": "SV", "IP": "IP",
    "H": "H", "R": "R", "ER": "ER", "HR": "HR",
    "BB": "BB", "SO": "SO", "BF": "BF",
    "ERA+": "ERA_plus", "FIP": "FIP", "WHIP": "WHIP",
    "H9": "H9", "HR9": "HR9", "BB9": "BB9",
    "SO9": "SO9", "SO/BB": "SO_BB",
}

ADVANCED_COL_MAP = {
    "Player": "name", "Age": "age", "Team": "team",
    "IP": "IP", "K%": "K_pct", "BB%": "BB_pct",
    "BAbip": "BABIP", "HR%": "HR_pct",
    "GB%": "GB_pct", "FB%": "FB_pct",
    "EV": "exit_velo", "HardH%": "hard_hit_pct",
}

def fetch_pitcher_data(start_year=2000, end_year=2025, data_dir=None):
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📂  Reading pitcher files from: {data_dir}\n")

    seasons = []
    for year in range(start_year, end_year + 1):
        print(f"  Loading {year}...", end=" ")
        df = _load_year(data_dir, year)
        if df is not None and len(df) > 0:
            seasons.append(df)
            print(f"{len(df)} pitchers loaded ✅")
        else:
            print("⚠️  Skipped")

    if not seasons:
        raise FileNotFoundError("No pitching data files found.")

    df = pd.concat(seasons, ignore_index=True)
    df = _engineer_features(df)

    print(f"\n✅  Pitcher data loaded: {len(df):,} pitcher-seasons | {df['player_id'].nunique():,} pitchers")
    print(f"    Seasons: {df['season'].min()}–{df['season'].max()}")
    print(f"    WAR range: {df['WAR'].min():.1f} → {df['WAR'].max():.1f}\n")
    return df


def _load_year(data_dir, year):
    value    = _read_table(data_dir, f"pitching_{year}",       VALUE_COL_MAP)
    standard = _read_table(data_dir, f"pitch_standard_{year}", STANDARD_COL_MAP)
    advanced = _read_table(data_dir, f"pitch_advanced_{year}", ADVANCED_COL_MAP)

    if value is None:
        return None

    value["season"] = year

    # Coerce numerics
    for col in ["age","IP","G","GS","WAR","RAA","WAA","RAR","RA9"]:
        if col in value.columns:
            value[col] = pd.to_numeric(value[col], errors="coerce")

    value = value.dropna(subset=["WAR","IP"])
    value = value[value["IP"] >= 20]  # min 20 innings pitched

    # Classify role: starter vs reliever
    if "GS" in value.columns:
        value["role"] = value.apply(
            lambda r: "SP" if pd.notna(r["GS"]) and r["GS"] >= r["G"] * 0.5 else "RP",
            axis=1
        )
    else:
        value["role"] = "SP"

    # Handle traded players
    if "team" in value.columns:
        multi_team_tags = {"TOT","2TM","3TM","4TM"}
        has_tot = set(value[value["team"].isin(multi_team_tags)]["name"].unique())
        no_splits = value[~((value["name"].isin(has_tot)) & (~value["team"].isin(multi_team_tags)))].copy()
        no_splits["team"] = no_splits["team"].apply(lambda t: "Multiple" if t in multi_team_tags else t)
        multi = value[~value["name"].isin(has_tot)]
        dupes = multi[multi.duplicated("name", keep=False)]["name"].unique()
        if len(dupes) > 0:
            deduped = (multi[multi["name"].isin(dupes)]
                       .sort_values("IP", ascending=False)
                       .drop_duplicates("name", keep="first"))
            agg = multi[multi["name"].isin(dupes)].groupby("name")[["WAR","IP"]].sum().reset_index()
            deduped = deduped.drop(columns=["WAR","IP"]).merge(agg, on="name")
            deduped["team"] = "Multiple"
            single = multi[~multi["name"].isin(dupes)]
            value = pd.concat([no_splits[~no_splits["name"].isin(dupes)], single, deduped], ignore_index=True)
        else:
            value = no_splits

    # Merge standard
    if standard is not None:
        for col in ["ERA","FIP","WHIP","SO9","BB9","HR9","ERA_plus","SO_BB","IP","GS","SV"]:
            if col in standard.columns:
                standard[col] = pd.to_numeric(standard[col], errors="coerce")
        std_cols = ["name"] + [c for c in ["ERA","FIP","WHIP","SO9","BB9","HR9","H9",
                               "ERA_plus","SO_BB","GS","SV","BB","SO","IP"] if c in standard.columns]
        std_merge = standard[std_cols].copy()
        if "IP" in std_merge.columns:
            std_merge["IP"] = pd.to_numeric(std_merge["IP"], errors="coerce").fillna(0)
            std_merge = std_merge.sort_values("IP", ascending=False).drop_duplicates("name", keep="first")
        else:
            std_merge = std_merge.drop_duplicates("name", keep="first")
        value = value.merge(std_merge, on="name", how="left")

    # Merge advanced
    if advanced is not None:
        for col in ["K_pct","BB_pct","BABIP","HR_pct","GB_pct","FB_pct","exit_velo","hard_hit_pct"]:
            if col in advanced.columns:
                advanced[col] = pd.to_numeric(
                    advanced[col].astype(str).str.replace("%","",regex=False).str.strip(),
                    errors="coerce")
        for col in ["K_pct","BB_pct","hard_hit_pct","GB_pct","FB_pct","HR_pct"]:
            if col in advanced.columns:
                # Convert from percentage to decimal if > 1
                mask = advanced[col] > 1
                advanced.loc[mask, col] = advanced.loc[mask, col] / 100
        adv_cols = ["name"] + [c for c in ["K_pct","BB_pct","BABIP","HR_pct",
                                "GB_pct","FB_pct","exit_velo","hard_hit_pct"] if c in advanced.columns]
        adv_merge = advanced[adv_cols].drop_duplicates("name", keep="first")
        value = value.merge(adv_merge, on="name", how="left")

    # Fill defaults
    defaults = {
        "ERA": 4.50, "FIP": 4.50, "WHIP": 1.35,
        "SO9": 8.0, "BB9": 3.0, "HR9": 1.2,
        "ERA_plus": 100, "K_pct": 0.22, "BB_pct": 0.085,
        "GB_pct": 0.44, "BABIP": 0.295,
    }
    for col, val in defaults.items():
        if col not in value.columns: value[col] = val
        else: value[col] = value[col].fillna(val)

    value["salary_M"]      = np.nan
    value["partial_season"] = (year == 2025)

    return value.reset_index(drop=True)


def _read_table(data_dir, filename_base, col_map, name_col="Player"):
    raw = None
    for ext in ["xls","xlsx"]:
        path = os.path.join(data_dir, f"{filename_base}.{ext}")
        if not os.path.exists(path): continue
        for method in ["html","xlrd","openpyxl"]:
            try:
                if method == "html": raw = pd.read_html(path, header=0)[0]
                elif method == "xlrd": raw = pd.read_excel(path, header=0, engine="xlrd")
                else: raw = pd.read_excel(path, header=0, engine="openpyxl")
                break
            except Exception: continue
        if raw is not None: break
    if raw is None: return None

    if name_col not in raw.columns and raw.iloc[0].astype(str).str.contains(name_col).any():
        raw.columns = raw.iloc[0].tolist()
        raw = raw.iloc[1:].reset_index(drop=True)

    if name_col in raw.columns:
        raw = raw[raw[name_col] != name_col].copy()
        raw = raw[raw[name_col].notna()].copy()
        raw = raw[~raw[name_col].astype(str).str.contains("League|Total|Avg|Rk", na=False)]

    raw = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})
    if "name" in raw.columns:
        raw["name"] = raw["name"].astype(str).str.replace(r"[*#]","",regex=True).str.strip()
    return raw


def _engineer_features(df):
    df["name"]      = df["name"].astype(str).str.replace(r"[*#]","",regex=True).str.strip()
    df["player_id"] = ("P_" + df["name"]).astype("category").cat.codes

    # Key derived features
    df["K_BB_ratio"] = (df["SO9"] / df["BB9"].clip(lower=0.1)).round(2) if "SO9" in df.columns else 2.5
    df["FIP_minus_ERA"] = (df["FIP"] - df["ERA"]).round(2) if "FIP" in df.columns else 0.0

    df = df.sort_values(["player_id","season"]).reset_index(drop=True)
    df["service_years"] = df.groupby("player_id").cumcount()
    df["salary_M"] = df.apply(
        lambda r: r["salary_M"] if pd.notna(r["salary_M"])
                  else _estimate_salary(r["WAR"], r["service_years"]), axis=1)

    df["WAR_prev"]    = df.groupby("player_id")["WAR"].shift(1)
    df["WAR_next"]    = df.groupby("player_id")["WAR"].shift(-1)
    df["WAR_delta"]   = (df["WAR"] - df["WAR_prev"]).round(2)
    # Normalize 2020 WAR to full-season equivalent before computing 3yr avg
    war_normalized = df["WAR"].copy()
    war_normalized[df["season"] == 2020] = (war_normalized[df["season"] == 2020] * 2.7).clip(upper=12)
    df["WAR_3yr_avg"] = df.groupby("player_id").apply(
        lambda g: war_normalized.loc[g.index].rolling(3, min_periods=1).mean()
    ).reset_index(level=0, drop=True).round(2)
    df["career_WAR_to_date"] = df.groupby("player_id")["WAR"].cumsum().round(2)
    df["IP_per_start"] = (df["IP"] / df["GS"].clip(lower=1)).round(1) if "GS" in df.columns else 6.0
    return df


def _estimate_salary(war, service):
    if service < 3: return 0.74
    elif service < 6: return round(max(0.74, war * 7.5 * 0.35), 2)
    else: return round(max(0.74, war * 7.5 * 0.85), 2)
