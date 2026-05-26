"""
MLB WAR Prediction Pipeline — Archetype Clustering
====================================================
Discovers natural player archetypes via KMeans clustering,
then profiles each archetype with aging curves & statistics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ── Archetype Labels (assigned post-hoc by profiling) ────────────────────────
# These will be auto-assigned based on cluster centroid characteristics.
ARCHETYPE_DESCRIPTIONS = {
    "Power Masher":       "High ISO/SLG, below-avg speed, corner position. Ages well offensively but declines defensively fast.",
    "Elite Two-Way SS/CF":"High WAR both sides, above-avg speed. Peak value 24-29. Premium trade asset.",
    "Speedy Slap Hitter": "High sprint speed, low ISO, high contact. Ages quickly as speed fades.",
    "Patient OBP Machine":"High BB%, moderate power. Durable archetype, ages gracefully.",
    "Glove-First Catcher":"Strong Def rating, below-avg offense. Consistent but capped upside.",
    "Five-Tool Star":     "Elite across the board. Rare. Commands max contract value.",
    "High-K Power Threat":"Big ISO/SLG, elevated K%. Boom-or-bust profile.",
    "Utility/Bench Role": "Below-avg WAR, positional flexibility. Short career arc.",
}


# ── Feature Selection ─────────────────────────────────────────────────────────

CLUSTER_FEATURES = [
    "ISO", "BB_pct", "K_pct", "wRC_plus",
    "Def", "BsR", "sprint_speed",
    "OBP", "SLG", "WAR_3yr_avg",
    "HR_rate", "SB_rate", "contact_rate", "power_speed",
    "Rtot", "Rdrs",
]

def build_player_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate season-level data to player-level profiles for clustering.
    Uses peak-season and career-average stats.
    """
    grp = df.groupby("player_id")

    profiles = pd.DataFrame({
        "player_id":      grp["player_id"].first(),
        "name":           grp["name"].first(),
        "position":       grp["position"].first(),
        "bats":           grp["bats"].first(),
        "height_in":      grp["height_in"].first(),
        "weight_lbs":     grp["weight_lbs"].first(),
        "sprint_speed":   grp["sprint_speed"].first(),
        "bmi":            grp["bmi"].first(),
        # Career averages (weighted by PA)
        "ISO":            grp["ISO"].mean() if grp["ISO"].mean().notna().any() else 0.15,
        "BB_pct":         grp["BB_pct"].mean(),
        "K_pct":          grp["K_pct"].mean(),
        "wRC_plus":       grp["wRC_plus"].mean(),
        "Def":            grp["Def"].mean(),
        "BsR":            grp["BsR"].mean(),
        "OBP":            grp["OBP"].mean(),
        "SLG":            grp["SLG"].mean(),
        "sprint_speed":   grp["sprint_speed"].mean(),
        "contact_rate":   grp["contact_rate"].mean() if "contact_rate" in grp.obj.columns else 0.785,
        "power_speed":    grp["power_speed"].mean()  if "power_speed"  in grp.obj.columns else 0.0,
        "HR_rate":        grp["HR_rate"].mean()       if "HR_rate"      in grp.obj.columns else 0.03,
        "SB_rate":        grp["SB_rate"].mean()       if "SB_rate"      in grp.obj.columns else 0.01,
        "Rtot":           grp["Rtot"].mean()           if "Rtot"         in grp.obj.columns else 0.0,
        "Rdrs":           grp["Rdrs"].mean()           if "Rdrs"         in grp.obj.columns else 0.0,
        # Peak season WAR
        "peak_WAR":       grp["WAR"].max(),
        "career_WAR":     grp["WAR"].sum(),
        "career_seasons": grp["WAR"].count(),
        "WAR_3yr_avg":    grp["WAR_3yr_avg"].mean(),
        # Debut/age info
        "debut_age":      grp["age"].min(),
        "final_age":      grp["age"].max(),
    }).reset_index(drop=True)

    return profiles


# ── Optimal K Selection ───────────────────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(4, 16)) -> int:
    """Elbow + silhouette method to find best K."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Pick k with best silhouette score (simple, robust)
    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"    Silhouette scores by k: { {k: round(s,3) for k,s in zip(k_range, silhouettes)} }")
    print(f"    ➜  Optimal k = {best_k}")
    return best_k


# ── Main Clustering Function ──────────────────────────────────────────────────

def fit_archetypes(df: pd.DataFrame, n_clusters: int = None) -> tuple:
    """
    Fit KMeans archetypes on player profiles.

    Returns
    -------
    profiles_df  : player-level DataFrame with 'archetype' and 'archetype_label' columns
    model_bundle : dict with scaler, kmeans, pca, centroid_df
    """
    print("\n🔍  Building player profiles for clustering...")
    profiles = build_player_profiles(df)

    # Drop rows with any NaN in cluster features
    # Only use features that actually exist in the profiles
    available_features = [f for f in CLUSTER_FEATURES if f in profiles.columns]
    if len(available_features) < 4:
        raise ValueError(f"Too few cluster features available: {available_features}")
    feat_df = profiles[available_features].copy()
    valid   = feat_df.dropna()
    profiles_clean = profiles.loc[valid.index].copy()

    # Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(valid)

    # Find optimal K if not specified
    if n_clusters is None:
        print("  Finding optimal number of archetypes...")
        n_clusters = find_optimal_k(X_scaled)

    # Final clustering
    print(f"\n🧬  Fitting KMeans with k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    profiles_clean["archetype_id"] = km.fit_predict(X_scaled)

    # PCA for 2-D visualization
    pca     = PCA(n_components=2, random_state=42)
    coords  = pca.fit_transform(X_scaled)
    profiles_clean["pca_x"] = coords[:, 0]
    profiles_clean["pca_y"] = coords[:, 1]

    # Profile each cluster by centroid characteristics → auto-label
    centroids_scaled = km.cluster_centers_
    centroids_raw    = pd.DataFrame(
        scaler.inverse_transform(centroids_scaled),
        columns=CLUSTER_FEATURES
    )
    centroids_raw["archetype_id"] = range(n_clusters)
    archetype_labels = _auto_label_archetypes(centroids_raw)
    profiles_clean["archetype_label"] = profiles_clean["archetype_id"].map(archetype_labels)

    # Summary
    summary = _build_archetype_summary(profiles_clean, df)
    print("\n📊  Archetype Summary:")
    print(summary[["archetype_label","n_players","avg_peak_WAR",
                    "avg_career_WAR","avg_ISO","avg_sprint_speed",
                    "avg_Def","avg_wRC_plus"]].to_string(index=False))

    # ── Compute aging curves per archetype ──────────────────────────
    arch_map = profiles_clean[['player_id','archetype_label']].drop_duplicates()
    merged_seasons = df.merge(arch_map, on='player_id', how='left')
    aging_curves = {}
    for arch in profiles_clean['archetype_label'].dropna().unique():
        arch_data = merged_seasons[merged_seasons['archetype_label'] == arch]
        curve = arch_data.groupby('age')['WAR'].agg(['mean','count']).reset_index()
        curve = curve[(curve['age'] >= 20) & (curve['age'] <= 42) & (curve['count'] >= 5)]
        aging_curves[arch] = dict(zip(curve['age'], curve['mean']))

    model_bundle = {
        "scaler":       scaler,
        "kmeans":       km,
        "pca":          pca,
        "centroids":    centroids_raw,
        "labels":       archetype_labels,
        "features":     CLUSTER_FEATURES,
        "summary":      summary,
        "n_clusters":   n_clusters,
        "aging_curves": aging_curves,
    }

    return profiles_clean, model_bundle


# ── Archetype Auto-Labeling ───────────────────────────────────────────────────

def _auto_label_archetypes(centroids: pd.DataFrame) -> dict:
    """
    Two-tier archetype system with 6 classes and 12 specific archetypes.
    Thresholds calibrated against actual data distributions.
    """
    labels = {}
    for _, row in centroids.iterrows():
        aid   = int(row["archetype_id"])
        iso   = row.get("ISO", 0.15)
        bb    = row.get("BB_pct", 0.085)
        kk    = row.get("K_pct", 0.215)
        speed = row.get("sprint_speed", 28.0)
        def_  = row.get("Def", 0.0)
        war   = row.get("WAR_3yr_avg", 1.0)
        wrc   = row.get("wRC_plus", 100)
        rtot  = row.get("Rtot", 0.0)

        # ── ELITE CLASS ───────────────────────────────────────────────────────
        if war >= 3.5 or wrc >= 125:
            if def_ >= 3.0 and speed >= 28.5:
                label = "Two-Way Superstar"
            else:
                label = "Franchise Cornerstone"

        # ── POWER CLASS ──────────────────────────────────────────────────────
        elif iso >= 0.19 and kk < 0.22:
            label = "Pure Power Masher"
        elif iso >= 0.17 and kk >= 0.22:
            label = "High-K Power Threat"

        # ── CONTACT & OBP CLASS ──────────────────────────────────────────────
        elif kk < 0.17 and wrc >= 92:
            label = "Elite Contact Hitter"
        elif bb >= 0.10 and wrc >= 92:
            label = "Patient OBP Machine"

        # ── SPEED & DEFENSE CLASS ─────────────────────────────────────────────
        elif speed >= 29.5 and iso < 0.14:
            label = "Speedy Slap Hitter"
        elif (def_ >= 3.0 or rtot >= 4.0) and wrc < 100:
            label = "Glove-First Defender"
        elif def_ >= 1.5 and speed >= 28.5 and war >= 1.2:
            label = "Two-Way Threat"

        # ── REGULAR CLASS ────────────────────────────────────────────────────
        elif war >= 1.5 and wrc >= 98:
            label = "All-Around Regular"
        elif war >= 0.8:
            label = "Solid Contributor"

        # ── DEPTH CLASS ──────────────────────────────────────────────────────
        else:
            label = "Bench/Utility Role"

        labels[aid] = label
    return labels


def label_player_archetype(player_row):
    """
    Label an individual player based on their own stats.
    Priority order: Elite > Glove/Speed > Power > Contact > Regular > Depth
    """
    iso   = float(player_row.get("ISO", 0.15) or 0.15)
    bb    = float(player_row.get("BB_pct", 0.085) or 0.085)
    kk    = float(player_row.get("K_pct", 0.215) or 0.215)
    speed = float(player_row.get("sprint_speed", 28.0) or 28.0)
    def_  = float(player_row.get("Def", 0.0) or 0.0)
    war   = float(player_row.get("WAR_3yr_avg", 1.0) or 1.0)
    wrc   = float(player_row.get("wRC_plus", 100) or 100)
    rtot  = float(player_row.get("Rtot", 0.0) or 0.0)
    seasons_exp = int(player_row.get("seasons_300pa", 99) or 99)
    is_emerging = seasons_exp <= 2

    # ── ELITE CLASS ───────────────────────────────────────────────────────────
    if war >= 5.0 or (war >= 3.5 and wrc >= 120):
        if is_emerging:
            if def_ >= 3.0 and speed >= 28.5:
                return "Two-Way Superstar"
            return "Emerging Superstar"
        if def_ >= 3.0 and speed >= 28.5:
            return "Two-Way Superstar"
        return "Franchise Cornerstone"

    # EMERGING SUPERSTAR — elite production in first 2 full seasons
    if war >= 3.0 and wrc >= 115 and is_emerging:
        if def_ >= 3.0 and speed >= 28.5:
            return "Two-Way Superstar"
        return "Emerging Superstar"

    # ── SPEED & DEFENSE CLASS — check BEFORE power ───────────────────────────
    # Pure speed profiles
    if speed >= 29.5 and iso < 0.15 and wrc < 100:
        return "Speedy Slap Hitter"

    # Elite defenders — check Def first, even before power
    if (def_ >= 5.0 or rtot >= 6.0) and iso < 0.20:
        if wrc >= 95 and war >= 1.5:
            return "Two-Way Threat"
        return "Glove-First Defender"

    # Broader glove-first — good defense, below-avg offense
    if (def_ >= 2.5 or rtot >= 4.0) and wrc < 98 and iso < 0.16:
        return "Glove-First Defender"

    # Two-Way Threat — speed + defense + decent bat
    if def_ >= 2.5 and speed >= 29.0 and war >= 1.5 and wrc >= 95 and 0.13 <= iso <= 0.20:
        return "Two-Way Threat"

    # ── POWER CLASS ──────────────────────────────────────────────────────────
    # Require real power, controlled K%, solid production AND meaningful WAR
    if iso >= 0.21 and kk < 0.26 and wrc >= 100 and war >= 1.5:
        return "Pure Power Masher"
    if iso >= 0.17 and kk >= 0.24:
        return "High-K Power Threat"

    # ── CONTACT & OBP CLASS ──────────────────────────────────────────────────
    if kk < 0.16 and wrc >= 92:
        return "Elite Contact Hitter"
    if bb >= 0.10 and wrc >= 92:
        return "Patient OBP Machine"

    # ── REGULAR CLASS ────────────────────────────────────────────────────────
    if war >= 1.5 and wrc >= 98:
        return "All-Around Regular"
    if war >= 0.8:
        return "Solid Contributor"

    return "Bench/Utility Role"


def get_hitter_secondary_tags(player_row):
    """
    Generate secondary descriptor tags for a hitter.
    Returns up to 2 tags like ['Elite Defender', 'Speed Dimension']
    """
    tags = []
    rtot  = float(player_row.get("Rtot", 0) or 0)
    rdrs  = float(player_row.get("Rdrs", 0) or 0)
    spd   = float(player_row.get("sprint_speed", 27.0) or 27.0)
    iso   = float(player_row.get("ISO", 0.15) or 0.15)
    bb    = float(player_row.get("BB_pct", 0.085) or 0.085)
    kk    = float(player_row.get("K_pct", 0.215) or 0.215)
    wrc   = float(player_row.get("wRC_plus", 100) or 100)
    def_  = float(player_row.get("Def", 0) or 0)

    # Defensive tags (highest priority)
    if rtot >= 10 or def_ >= 8:   tags.append("Elite Defender")
    elif rtot >= 5 or def_ >= 4:  tags.append("Above-Avg Defender")
    elif rtot <= -8:               tags.append("Defensive Liability")

    # Offensive style tags
    if spd >= 29.5 and iso < 0.16: tags.append("Speed Dimension")
    elif iso >= 0.220:              tags.append("Raw Power")
    if bb >= 0.11:                  tags.append("Elite Plate Discipline")
    if kk <= 0.12:                  tags.append("Elite Contact")
    elif kk >= 0.30:                tags.append("High Strikeout Risk")
    if wrc >= 140:                  tags.append("Elite Offensive Force")

    return tags[:2]  # max 2 secondary tags


def _build_archetype_summary(profiles: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Build a rich summary table for each archetype."""
    grp = profiles.groupby("archetype_label")
    summary = pd.DataFrame({
        "archetype_label":   grp["archetype_label"].first(),
        "n_players":         grp["player_id"].count(),
        "avg_peak_WAR":      grp["peak_WAR"].mean().round(2),
        "avg_career_WAR":    grp["career_WAR"].mean().round(2),
        "avg_ISO":           grp["ISO"].mean().round(3),
        "avg_BB_pct":        grp["BB_pct"].mean().round(3),
        "avg_K_pct":         grp["K_pct"].mean().round(3),
        "avg_wRC_plus":      grp["wRC_plus"].mean().round(1),
        "avg_Def":           grp["Def"].mean().round(2),
        "avg_sprint_speed":  grp["sprint_speed"].mean().round(1),
        "avg_debut_age":     grp["debut_age"].mean().round(1),
        "avg_career_length": grp["career_seasons"].mean().round(1),
    }).reset_index(drop=True)

    return summary.sort_values("avg_peak_WAR", ascending=False)


# ── Classify a New Player ─────────────────────────────────────────────────────

def classify_player(history: pd.DataFrame, bundle: dict, seasons_300pa_override=None) -> dict:
    """Classify a player using their recent stats directly."""
    # Use last 2 seasons weighted average for classification
    max_season = history["season"].max()
    recent = history[history["season"] >= max_season - 1]
    if len(recent) == 0:
        recent = history.tail(2)

    # Count seasons with meaningful playing time (300+ PA)
    if seasons_300pa_override is not None:
        seasons_300pa = seasons_300pa_override
    elif "PA" in history.columns:
        seasons_300pa = int((pd.to_numeric(history["PA"], errors="coerce").fillna(0) >= 300).sum())
    else:
        seasons_300pa = 99

    # Build a weighted average profile
    profile = {}
    weight_cols = ["ISO","BB_pct","K_pct","sprint_speed","Def","WAR_3yr_avg","wRC_plus","Rtot"]
    for col in weight_cols:
        if col in recent.columns:
            vals = pd.to_numeric(recent[col], errors="coerce").dropna()
            profile[col] = float(vals.mean()) if len(vals) > 0 else 0.0

    profile["seasons_300pa"] = seasons_300pa
    label = label_player_archetype(profile)

    # Still use cluster for archetype_id (for model features)
    try:
        features = bundle.get("features", [])
        available = [f for f in features if f in history.columns]
        if available and bundle.get("scaler") and bundle.get("kmeans"):
            feat_vals = history[available].mean().values.reshape(1, -1)
            scaled = bundle["scaler"].transform(feat_vals)
            aid = int(bundle["kmeans"].predict(scaled)[0])
        else:
            aid = 0
    except Exception:
        aid = 0

    return {
        "archetype_label": label,
        "archetype_id": aid,
        "archetype_class": _get_archetype_class(label),
    }


def _get_archetype_class(label):
    classes = {
        "Franchise Cornerstone": "ELITE",
        "Two-Way Superstar": "ELITE",
        "Pure Power Masher": "POWER",
        "High-K Power Threat": "POWER",
        "Elite Contact Hitter": "CONTACT",
        "Patient OBP Machine": "CONTACT",
        "Speedy Slap Hitter": "SPEED/DEF",
        "Glove-First Defender": "SPEED/DEF",
        "Two-Way Threat": "SPEED/DEF",
        "All-Around Regular": "REGULAR",
        "Solid Contributor": "REGULAR",
        "Bench/Utility Role": "DEPTH",
    }
    return classes.get(label, "REGULAR")


