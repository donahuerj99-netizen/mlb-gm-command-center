"""
MLB WAR Prediction Pipeline — Local Data Scraper
"""

import os
import pandas as pd
import numpy as np

VALUE_COL_MAP = {
    "Player": "name", "Age": "age", "Team": "team", "Pos": "position",
    "PA": "PA", "Rbat": "Rbat", "Rbaser": "BsR", "Rfield": "Def",
    "RAA": "RAA", "WAA": "WAA", "RAR": "RAR", "WAR": "WAR",
    "oWAR": "oWAR", "dWAR": "dWAR",
}
STANDARD_COL_MAP = {
    "Player": "name", "Age": "age", "Team": "team",
    "PA": "PA", "AB": "AB", "H": "H", "2B": "2B", "3B": "3B",
    "HR": "HR", "SB": "SB", "CS": "CS", "BB": "BB", "SO": "SO",
    "BA": "AVG", "OBP": "OBP", "SLG": "SLG", "OPS": "OPS",
    "R": "R", "RBI": "RBI", "Pos": "position_std",
}
ADVANCED_COL_MAP = {
    "Player": "name", "Age": "age", "Team": "team",
    "BB%": "BB_pct", "SO%": "K_pct", "ISO": "ISO",
    "BAbip": "BABIP", "rOBA": "wOBA",
    "EV": "exit_velo", "HardH%": "hard_hit_pct",
    "Rbat+": "wRC_plus", "HR%": "HR_rate_adv",
}
FIELDING_COL_MAP = {
    "Player": "name", "Age": "age", "Team": "team",
    "G": "G_field", "Inn": "Inn",
    "Rtot": "Rtot", "Rtot/yr": "Rtot_per_yr",
    "Rdrs": "Rdrs", "Fld%": "Fld_pct",
    "RF/9": "RF9", "PO": "PO",
    "A": "assists", "E": "errors", "DP": "DP",
}
POS_MAP = {"C":"C","1B":"1B","2B":"2B","3B":"3B","SS":"SS","LF":"LF","CF":"CF","RF":"RF","DH":"DH","OF":"RF","IF":"2B","UT":"DH","P":None}
NUM_TO_POS = {"2":"C","3":"1B","4":"2B","5":"3B","6":"SS","7":"LF","8":"CF","9":"RF","0":"DH","1":"DH"}
POS_SPEED  = {"C":27.0,"1B":26.5,"2B":27.8,"3B":27.2,"SS":28.2,"LF":27.5,"CF":28.8,"RF":27.6,"DH":26.2}
POS_HEIGHT = {"C":72.5,"1B":74.5,"2B":71.5,"3B":73.5,"SS":72.5,"LF":73.0,"CF":72.5,"RF":73.5,"DH":74.0}
POS_WEIGHT = {"C":210,"1B":225,"2B":185,"3B":210,"SS":190,"LF":210,"CF":195,"RF":215,"DH":230}

def fetch_real_data(start_year=2000, end_year=2025, data_dir=None):
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📂  Reading Baseball Reference files from: {data_dir}\n")
    seasons = []
    for year in range(start_year, end_year + 1):
        print(f"  Loading {year}...", end=" ")
        df = _load_year(data_dir, year)
        if df is not None and len(df) > 0:
            seasons.append(df)
            print(f"{len(df)} players loaded ✅")
        else:
            print("⚠️  Skipped")
    if not seasons:
        raise FileNotFoundError("No data files found.")
    df = pd.concat(seasons, ignore_index=True)
    df = _engineer_features(df)
    print(f"\n✅  Real data loaded: {len(df):,} player-seasons | {df['player_id'].nunique():,} players")
    print(f"    Seasons: {df['season'].min()}–{df['season'].max()}")
    print(f"    WAR range: {df['WAR'].min():.1f} → {df['WAR'].max():.1f}\n")
    return df

def _parse_position(raw):
    raw = str(raw).replace("*","").replace("#","").strip()
    first = raw.split("/")[0].strip()
    letters = "".join(c for c in first if c.isalpha()).upper()
    letter_map = {"D":"DH","H":"DH","O":"RF","M":"DH","Y":"DH","U":"DH"}
    if letters in POS_MAP: return POS_MAP.get(letters)
    if letters in letter_map: return letter_map[letters]
    digits = "".join(c for c in first if c.isdigit())
    for d in digits:
        if d in NUM_TO_POS: return NUM_TO_POS[d]
    return None

def _load_year(data_dir, year):
    value    = _read_table(data_dir, f"batting_{year}",  VALUE_COL_MAP)
    standard = _read_table(data_dir, f"standard_{year}", STANDARD_COL_MAP)
    advanced = _read_table(data_dir, f"advanced_{year}", ADVANCED_COL_MAP)
    fielding = _read_table(data_dir, f"fielding_{year}", FIELDING_COL_MAP)
    if value is None: return None

    value["season"] = year
    for col in ["age","PA","WAR","oWAR","dWAR","Def","BsR","RAA","WAA","RAR"]:
        if col in value.columns:
            value[col] = pd.to_numeric(value[col], errors="coerce")
    value = value.dropna(subset=["WAR","PA"])
    value = value[value["PA"] >= 50]

    if "position" in value.columns:
        value["position"] = value["position"].apply(_parse_position)
        value = value[value["position"].notna()]
    else:
        value["position"] = "DH"

    # Handle traded players
    if "team" in value.columns:
        multi_team_tags = {"TOT","2TM","3TM","4TM"}
        has_tot = set(value[value["team"].isin(multi_team_tags)]["name"].unique())
        no_splits = value[~((value["name"].isin(has_tot)) & (~value["team"].isin(multi_team_tags)))].copy()
        no_splits["team"] = no_splits["team"].apply(lambda t: "Multiple" if t in multi_team_tags else t)
        multi_team = value[~value["name"].isin(has_tot)]
        dupes = multi_team[multi_team.duplicated("name", keep=False)]["name"].unique()
        if len(dupes) > 0:
            deduped = (multi_team[multi_team["name"].isin(dupes)]
                       .sort_values("PA", ascending=False)
                       .drop_duplicates("name", keep="first"))
            agg = (multi_team[multi_team["name"].isin(dupes)]
                   .groupby("name")[["WAR","PA"]].sum().reset_index())
            deduped = deduped.drop(columns=["WAR","PA"]).merge(agg, on="name")
            deduped["team"] = "Multiple"
            single_team = multi_team[~multi_team["name"].isin(dupes)]
            value = pd.concat([no_splits[~no_splits["name"].isin(dupes)], single_team, deduped], ignore_index=True)
        else:
            value = no_splits

    # Merge standard
    if standard is not None:
        for col in ["AVG","OBP","SLG","OPS","HR","SB","CS","BB","SO","AB","H"]:
            if col in standard.columns:
                standard[col] = pd.to_numeric(standard[col], errors="coerce")
        std_cols = ["name"] + [c for c in ["AVG","OBP","SLG","OPS","HR","SB","CS","BB","SO","AB","H"] if c in standard.columns]
        std_merge = standard[std_cols].copy()
        if "AB" in std_merge.columns:
            std_merge["AB"] = pd.to_numeric(std_merge["AB"], errors="coerce").fillna(0)
            std_merge = std_merge.sort_values("AB", ascending=False).drop_duplicates("name", keep="first")
        else:
            std_merge = std_merge.drop_duplicates("name", keep="first")
        value = value.merge(std_merge, on="name", how="left")

    # Merge advanced
    if advanced is not None:
        if "name" not in advanced.columns and advanced.iloc[0].astype(str).str.contains("Player").any():
            advanced.columns = advanced.iloc[0].tolist()
            advanced = advanced.iloc[1:].reset_index(drop=True)
            advanced = advanced.rename(columns={k: v for k, v in ADVANCED_COL_MAP.items() if k in advanced.columns})
            if "name" in advanced.columns:
                advanced["name"] = advanced["name"].astype(str).str.replace(r"[*#]","",regex=True).str.strip()
        for col in ["BB_pct","K_pct","ISO","BABIP","wOBA","wRC_plus","exit_velo","hard_hit_pct"]:
            if col in advanced.columns:
                advanced[col] = pd.to_numeric(advanced[col].astype(str).str.replace("%","",regex=False).str.strip(), errors="coerce")
        for col in ["BB_pct","K_pct","hard_hit_pct"]:
            if col in advanced.columns:
                advanced[col] = advanced[col] / 100
        adv_cols = ["name"] + [c for c in ["BB_pct","K_pct","ISO","BABIP","wOBA","wRC_plus","exit_velo","hard_hit_pct","HR_rate_adv"] if c in advanced.columns]
        adv_merge = advanced[adv_cols].drop_duplicates("name", keep="first")
        value = value.merge(adv_merge, on="name", how="left")
        if "wRC_plus" in value.columns:
            fallback = (100 + value["RAA"] / value["PA"] * 100).clip(0, 250)
            value["wRC_plus"] = value["wRC_plus"].fillna(fallback)
        else:
            value["wRC_plus"] = (100 + value["RAA"] / value["PA"] * 100).clip(0, 250)

    # Merge fielding
    if fielding is not None:
        for col in ["Rtot","Rtot_per_yr","Rdrs","RF9","Inn","errors","DP","assists","PO"]:
            if col in fielding.columns:
                fielding[col] = pd.to_numeric(fielding[col], errors="coerce")
        if "Fld_pct" in fielding.columns:
            fielding["Fld_pct"] = pd.to_numeric(fielding["Fld_pct"].astype(str).str.replace("%","",regex=False), errors="coerce")
        field_num = [c for c in ["Rtot","Rdrs","Inn","errors","DP","assists","PO"] if c in fielding.columns]
        field_agg = fielding.groupby("name")[field_num].sum().reset_index()
        if "Fld_pct" in fielding.columns:
            fld_pct = fielding.groupby("name")["Fld_pct"].mean().reset_index()
            field_agg = field_agg.merge(fld_pct, on="name", how="left")
        value = value.merge(field_agg, on="name", how="left")

    # Defaults
    defaults = {
        "AVG":0.250,"OBP":0.320,"SLG":0.420,"OPS":0.740,
        "BB_pct":0.085,"K_pct":0.215,"ISO":0.150,"BABIP":0.295,
        "wRC_plus":100,"HR":15,"SB":5,
        "Rtot":0.0,"Rdrs":0.0,"Inn":0.0,"Fld_pct":0.980,
    }
    for col, val in defaults.items():
        if col not in value.columns: value[col] = val
        else: value[col] = value[col].fillna(val)

    value["height_in"]  = value["position"].map(POS_HEIGHT).fillna(73.0)
    value["weight_lbs"] = value["position"].map(POS_WEIGHT).fillna(205.0)
    value["bats"]       = "R"
    value["throws"]     = "R"
    value["salary_M"]   = np.nan
    value["partial_season"] = (year == 2025)

    pos_base = value["position"].map(POS_SPEED).fillna(27.0)
    if "SB" in value.columns and "CS" in value.columns:
        sb = pd.to_numeric(value["SB"], errors="coerce").fillna(0)
        cs = pd.to_numeric(value["CS"], errors="coerce").fillna(0)
        pa = pd.to_numeric(value["PA"], errors="coerce").fillna(1)
        attempts = sb + cs
        speed_adj = (attempts / pa) * 150 * (sb / (attempts + 0.001))
        value["sprint_speed"] = (pos_base + speed_adj).clip(24.0, 31.5).round(2)
    else:
        value["sprint_speed"] = pos_base

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
    df["name"]         = df["name"].astype(str).str.replace(r"[*#]","",regex=True).str.strip()
    df["player_id"]    = df["name"].astype("category").cat.codes
    df["OPS"]          = (df["OBP"] + df["SLG"]).round(3)
    df["contact_rate"] = (1 - df["K_pct"]).round(3)
    df["power_speed"]  = (df["ISO"] * df["sprint_speed"]).round(4)
    df["bmi"]          = (df["weight_lbs"] / (df["height_in"] ** 2) * 703).round(1)
    if "HR" in df.columns:
        df["HR_rate"] = (pd.to_numeric(df["HR"], errors="coerce") / df["PA"]).round(4)
    if "SB" in df.columns:
        df["SB_rate"] = (pd.to_numeric(df["SB"], errors="coerce") / df["PA"]).round(4)
    df = df.sort_values(["player_id","season"]).reset_index(drop=True)
    df["service_years"] = df.groupby("player_id").cumcount()
    df["salary_M"] = df.apply(lambda r: r["salary_M"] if pd.notna(r["salary_M"]) else _estimate_salary(r["WAR"], r["service_years"]), axis=1)
    df["WAR_prev"]    = df.groupby("player_id")["WAR"].shift(1)
    df["WAR_next"]    = df.groupby("player_id")["WAR"].shift(-1)
    df["WAR_delta"]   = (df["WAR"] - df["WAR_prev"]).round(2)
    # Normalize 2020 WAR to full-season equivalent (60 games -> 162 games = 2.7x)
    # This prevents the shortened COVID season from distorting 3yr averages
    war_normalized = df["WAR"].copy()
    war_normalized[df["season"] == 2020] = (war_normalized[df["season"] == 2020] * 2.7).clip(upper=12)
    df["WAR_3yr_avg"] = df.groupby("player_id")["WAR"].transform(lambda x: x.rolling(3, min_periods=1).mean()).round(2)
    # Use normalized WAR for 3yr avg only (don't change actual WAR values)
    df["WAR_3yr_avg"] = df.groupby("player_id").apply(
        lambda g: war_normalized.loc[g.index].rolling(3, min_periods=1).mean()
    ).reset_index(level=0, drop=True).round(2)
    df["career_WAR_to_date"] = df.groupby("player_id")["WAR"].cumsum().round(2)

    # Peak WAR — best single season to date (key signal for elite players)
    df["peak_WAR"] = df.groupby("player_id")["WAR"].transform(
        lambda x: x.expanding().max()
    ).round(2)

    # 5-year rolling average (smoother signal for veterans)
    df["WAR_5yr_avg"] = df.groupby("player_id").apply(
        lambda g: war_normalized.loc[g.index].rolling(5, min_periods=1).mean()
    ).reset_index(level=0, drop=True).round(2)

    # Average of best 3 seasons to date (captures ceiling, not just recent form)
    def top3_avg(x):
        return x.expanding().apply(lambda v: sorted(v)[-min(3,len(v)):] and
               sum(sorted(v)[-min(3,len(v)):]) / min(3,len(v)), raw=True)
    df["WAR_top3_avg"] = df.groupby("player_id")["WAR"].transform(top3_avg).round(2)

    return df

def _estimate_salary(war, service):
    if service < 3: return 0.74
    elif service < 6: return round(max(0.74, war * 7.5 * 0.35), 2)
    else: return round(max(0.74, war * 7.5 * 0.85), 2)
