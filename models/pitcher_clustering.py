"""
MLB WAR Prediction Pipeline — Pitcher Archetype Clustering
===========================================================
Discovers natural pitcher archetypes via KMeans clustering.

Archetypes expected:
  - Ace / Frontline Starter     (high WAR, low ERA, high K%)
  - Mid-Rotation Starter        (solid WAR, average ERA)
  - Back-End Starter            (below-average WAR, high ERA)
  - High-Leverage Closer        (low IP, high K%, saves)
  - Setup/High-Leverage RP      (moderate IP, solid ERA)
  - Situational/Mop-up RP       (low WAR, many appearances)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

CLUSTER_FEATURES_SP = [
    "ERA", "FIP", "WHIP", "SO9", "BB9", "HR9",
    "K_pct", "BB_pct", "GB_pct", "WAR_3yr_avg",
    "IP_per_start", "K_BB_ratio",
]

CLUSTER_FEATURES_RP = [
    "ERA", "FIP", "WHIP", "SO9", "BB9",
    "K_pct", "BB_pct", "WAR_3yr_avg", "K_BB_ratio",
]


def build_pitcher_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build pitcher profiles using recency-weighted averages.
    Recent seasons (last 3 years) are weighted 3x vs earlier seasons.
    This ensures current performance drives archetype classification.
    """
    max_season = df["season"].max()

    def weighted_mean(group, col):
        if col not in group.columns:
            return np.nan
        # Weight: 3 for last 2 seasons, 2 for 3-4 seasons ago, 1 for older
        weights = group["season"].apply(
            lambda s: 3 if s >= max_season - 1 else 2 if s >= max_season - 3 else 1
        )
        vals = pd.to_numeric(group[col], errors="coerce")
        mask = vals.notna()
        if mask.sum() == 0:
            return np.nan
        return float((vals[mask] * weights[mask]).sum() / weights[mask].sum())

    records = []
    for pid, grp in df.groupby("player_id"):
        grp = grp.sort_values("season")
        rec = {
            "player_id":      pid,
            "name":           grp["name"].iloc[-1],
            "role":           grp["role"].iloc[-1],
            "team":           grp["team"].iloc[-1],
            "ERA":            weighted_mean(grp, "ERA"),
            "FIP":            weighted_mean(grp, "FIP"),
            "WHIP":           weighted_mean(grp, "WHIP"),
            "SO9":            weighted_mean(grp, "SO9"),
            "BB9":            weighted_mean(grp, "BB9"),
            "HR9":            weighted_mean(grp, "HR9"),
            "K_pct":          weighted_mean(grp, "K_pct"),
            "BB_pct":         weighted_mean(grp, "BB_pct"),
            "GB_pct":         weighted_mean(grp, "GB_pct") if "GB_pct" in grp.columns else 0.44,
            "WAR_3yr_avg":    float(grp["WAR_3yr_avg"].iloc[-1]) if "WAR_3yr_avg" in grp.columns else 0,
            "IP_per_start":   weighted_mean(grp, "IP_per_start") if "IP_per_start" in grp.columns else 6.0,
            "K_BB_ratio":     weighted_mean(grp, "K_BB_ratio"),
            "peak_WAR":       float(grp["WAR"].max()),
            "career_WAR":     float(grp["WAR"].sum()),
            "career_seasons": len(grp),
            "avg_IP":         float(grp["IP"].mean()) if "IP" in grp.columns else 0.0,
            # Recent form: last 2 seasons average
            "recent_WAR":     float(grp[grp["season"] >= max_season - 1]["WAR"].mean()) if len(grp[grp["season"] >= max_season - 1]) > 0 else float(grp["WAR"].iloc[-1]),
        }
        records.append(rec)

    return pd.DataFrame(records).reset_index(drop=True)


def fit_pitcher_archetypes(df: pd.DataFrame) -> tuple:
    print("\n🔍  Building pitcher profiles for clustering...")
    profiles = build_pitcher_profiles(df)

    # Cluster starters and relievers separately
    sp = profiles[profiles["role"] == "SP"].copy()
    rp = profiles[profiles["role"] == "RP"].copy()

    sp_profiles, sp_bundle = _cluster_group(sp, CLUSTER_FEATURES_SP, "SP", n_clusters=6)
    rp_profiles, rp_bundle = _cluster_group(rp, CLUSTER_FEATURES_RP, "RP", n_clusters=3)

    all_profiles = pd.concat([sp_profiles, rp_profiles], ignore_index=True)

    print("\n📊  Pitcher Archetype Summary:")
    summary = _build_summary(all_profiles)
    print(summary[["archetype_label","n_pitchers","avg_peak_WAR","avg_ERA","avg_FIP","avg_SO9","avg_BB9"]].to_string(index=False))

    # ── Compute aging curves per pitcher archetype ──────────────────
    arch_map = all_profiles[['player_id','archetype_label']].drop_duplicates()
    merged_seasons = df.merge(arch_map, on='player_id', how='left')
    aging_curves = {}
    for arch in all_profiles['archetype_label'].dropna().unique():
        arch_data = merged_seasons[merged_seasons['archetype_label'] == arch]
        curve = arch_data.groupby('age')['WAR'].agg(['mean','count']).reset_index()
        curve = curve[(curve['age'] >= 20) & (curve['age'] <= 42) & (curve['count'] >= 5)]
        aging_curves[arch] = dict(zip(curve['age'], curve['mean']))

    bundle = {
        "sp_bundle": sp_bundle,
        "rp_bundle": rp_bundle,
        "summary":   summary,
        "features_sp": CLUSTER_FEATURES_SP,
        "features_rp": CLUSTER_FEATURES_RP,
        "aging_curves": aging_curves,
    }
    return all_profiles, bundle


def _cluster_group(profiles, features, role, n_clusters):
    available = [f for f in features if f in profiles.columns]
    feat_df   = profiles[available].copy()
    valid     = feat_df.dropna()
    prof_clean = profiles.loc[valid.index].copy()

    if len(prof_clean) < n_clusters:
        prof_clean["archetype_id"]    = 0
        prof_clean["archetype_label"] = f"{role} Pitcher"
        prof_clean["pca_x"] = 0.0
        prof_clean["pca_y"] = 0.0
        return prof_clean, {}

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(valid)

    # Find optimal k
    best_k = n_clusters
    best_score = -1
    for k in range(2, min(n_clusters + 3, len(prof_clean))):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        if len(set(labels)) > 1:
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_score = score
                best_k = k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    prof_clean["archetype_id"] = km.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    prof_clean["pca_x"] = coords[:,0]
    prof_clean["pca_y"] = coords[:,1]

    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=available
    )
    centroids["archetype_id"] = range(best_k)
    labels = _label_pitchers(centroids, role)
    prof_clean["archetype_label"] = prof_clean["archetype_id"].map(labels)

    bundle = {
        "scaler": scaler, "kmeans": km, "pca": pca,
        "centroids": centroids, "labels": labels,
        "features": available, "role": role,
    }
    return prof_clean, bundle


def _label_pitchers(centroids, role):
    labels = {}
    for _, row in centroids.iterrows():
        aid  = int(row["archetype_id"])
        era  = row.get("ERA", 4.5)
        fip  = row.get("FIP", 4.5)
        so9  = row.get("SO9", 8.0)
        bb9  = row.get("BB9", 3.0)
        war  = row.get("WAR_3yr_avg", 1.0)
        ips  = row.get("IP_per_start", 6.0)
        kpct = row.get("K_pct", 0.22)

        if role == "SP":
            # Use FIP as primary since it better predicts future performance
            if fip <= 3.00 or war >= 4.5:
                label = "Ace / Frontline Starter"
            elif fip <= 3.50 and so9 >= 9.5:
                label = "Power Arm Starter"
            elif fip <= 3.80 and so9 >= 8.5:
                label = "No. 2 / Quality Starter"
            elif fip <= 4.20:
                label = "Mid-Rotation Starter"
            elif fip <= 4.80:
                label = "Back-End Starter"
            else:
                label = "Rotation Filler"
        else:  # RP
            if fip <= 2.50 or (war >= 1.5 and so9 >= 12.0):
                label = "Elite Closer / Dominant RP"
            elif fip <= 3.20 and so9 >= 10.0:
                label = "High-Leverage Setup Man"
            elif fip <= 3.80:
                label = "Solid Middle Reliever"
            else:
                label = "Situational / Mop-up RP"

        labels[aid] = label
    return labels


def get_secondary_tags(pitcher_row):
    """
    Generate secondary descriptor tags for a pitcher based on their style.
    Returns a list of tags like ['Power Arm', 'Ground Ball Specialist']
    """
    tags = []
    so9  = float(pitcher_row.get("SO9", 8.0) or 8.0)
    bb9  = float(pitcher_row.get("BB9", 3.0) or 3.0)
    gb   = float(pitcher_row.get("GB_pct", 0.44) or 0.44)
    hr9  = float(pitcher_row.get("HR9", 1.2) or 1.2)
    fip  = float(pitcher_row.get("FIP", 4.5) or 4.5)
    era  = float(pitcher_row.get("ERA", 4.5) or 4.5)

    if so9 >= 11.0:           tags.append("Elite Strikeout Artist")
    elif so9 >= 9.5:          tags.append("Power Arm")
    if bb9 <= 2.0:            tags.append("Pinpoint Control")
    elif bb9 >= 4.5:          tags.append("Control Issues")
    if gb >= 0.52:            tags.append("Ground Ball Specialist")
    elif gb <= 0.35:          tags.append("Fly Ball Pitcher")
    if hr9 <= 0.8:            tags.append("Home Run Suppressor")
    if (era - fip) <= -0.50: tags.append("Outperforms Metrics")
    if (era - fip) >= 0.75:  tags.append("Underperforms Metrics")

    return tags[:2]  # max 2 secondary tags


def _build_summary(profiles):
    grp = profiles.groupby("archetype_label")
    return pd.DataFrame({
        "archetype_label": grp["archetype_label"].first(),
        "n_pitchers":      grp["player_id"].count(),
        "avg_peak_WAR":    grp["peak_WAR"].mean().round(2),
        "avg_career_WAR":  grp["career_WAR"].mean().round(2),
        "avg_ERA":         grp["ERA"].mean().round(2),
        "avg_FIP":         grp["FIP"].mean().round(2),
        "avg_SO9":         grp["SO9"].mean().round(1),
        "avg_BB9":         grp["BB9"].mean().round(1),
        "avg_IP":          grp["avg_IP"].mean().round(0),
    }).reset_index(drop=True).sort_values("avg_peak_WAR", ascending=False)


def classify_pitcher_row(row):
    """Classify a single pitcher row using rule-based logic. Used for training data labeling."""
    role = row.get("role", "SP")
    fip = float(row.get("FIP", 4.5) or 4.5)
    war = float(row.get("WAR", 0) or 0)
    so9 = float(row.get("SO9", 8.0) or 8.0)

    if role == "SP":
        if fip <= 2.80 or war >= 5.0:
            return "Ace / Frontline Starter"
        elif fip <= 3.30 and so9 >= 9.0:
            return "Power Arm Starter"
        elif fip <= 3.70:
            return "No. 2 / Quality Starter"
        elif fip <= 4.20:
            return "Mid-Rotation Starter"
        elif fip >= 5.20 or war <= 0.2:
            return "Rotation Filler"
        else:
            return "Back-End Starter"
    else:
        if fip <= 2.80 and so9 >= 10.0 and war >= 0.8:
            return "Elite Closer / Dominant RP"
        elif fip <= 3.40:
            return "High-Leverage Setup Man"
        elif fip <= 4.20:
            return "Solid Middle Reliever"
        else:
            return "Situational / Mop-up RP"

def classify_pitcher(history, bundle):
    role = history["role"].iloc[-1] if "role" in history.columns else "SP"
    sub_bundle = bundle["sp_bundle"] if role == "SP" else bundle["rp_bundle"]

    # Use recent seasons (last 2) for classification
    max_season = history["season"].max()
    recent = history[history["season"] >= max_season - 1]
    if len(recent) == 0:
        recent = history

    # Override with stats-based label for clearly elite/weak pitchers
    avg_fip = float(recent["FIP"].mean()) if "FIP" in recent.columns else 4.5
    avg_war = float(recent["WAR"].mean()) if "WAR" in recent.columns else 0
    avg_so9 = float(recent["SO9"].mean()) if "SO9" in recent.columns else 8.0
    avg_era = float(recent["ERA"].mean()) if "ERA" in recent.columns else 4.5

    # Require minimum sample size for override labels
    n_seasons = len(recent)
    avg_ip = float(recent["IP"].mean()) if "IP" in recent.columns else 0

    if role == "SP":
        if avg_fip <= 2.80 or avg_war >= 5.0:
            return {"archetype_label": "Ace / Frontline Starter", "archetype_id": -1, "centroid_distance": 0, "role": role}
        elif avg_fip <= 3.30 and avg_so9 >= 9.0:
            return {"archetype_label": "Power Arm Starter", "archetype_id": -1, "centroid_distance": 0, "role": role}
        elif avg_fip <= 3.70:
            return {"archetype_label": "No. 2 / Quality Starter", "archetype_id": -1, "centroid_distance": 0, "role": role}
        elif avg_fip <= 4.20:
            return {"archetype_label": "Mid-Rotation Starter", "archetype_id": -1, "centroid_distance": 0, "role": role}
        elif avg_fip >= 5.20 or avg_war <= 0.2:
            return {"archetype_label": "Rotation Filler", "archetype_id": -1, "centroid_distance": 0, "role": role}
    else:
        if avg_fip <= 2.80 and avg_so9 >= 10.0 and avg_war >= 0.8:
            return {"archetype_label": "Elite Closer / Dominant RP", "archetype_id": -1, "centroid_distance": 0, "role": role}
        elif avg_fip <= 3.40:
            return {"archetype_label": "High-Leverage Setup Man", "archetype_id": -1, "centroid_distance": 0, "role": role}
        elif avg_fip <= 4.20:
            return {"archetype_label": "Solid Middle Reliever", "archetype_id": -1, "centroid_distance": 0, "role": role}

    # Fall back to cluster-based label
    if not sub_bundle:
        return {"archetype_label": f"{role} Pitcher", "archetype_id": 0, "centroid_distance": 0}

    features  = sub_bundle["features"]
    available = [f for f in features if f in history.columns]
    profile   = history[available].mean().values.reshape(1,-1)
    scaled    = sub_bundle["scaler"].transform(profile)
    aid       = sub_bundle["kmeans"].predict(scaled)[0]
    label     = sub_bundle["labels"][aid]
    dist      = float(np.linalg.norm(scaled - sub_bundle["kmeans"].cluster_centers_[aid]))
    return {"archetype_label": label, "archetype_id": int(aid), "centroid_distance": round(dist,3), "role": role}
