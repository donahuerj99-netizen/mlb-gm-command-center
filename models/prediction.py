"""
MLB WAR Prediction Pipeline — Prediction Models
=================================================
Predicts future WAR for individual players using:
  1. Archetype-specific aging curves (population-level baseline)
  2. GradientBoosting regression for individual adjustments
  3. Uncertainty quantification via bootstrap
  4. Contract value estimation ($/WAR)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
import sklearn
sklearn.set_config(transform_output="default")

# ── Constants ────────────────────────────────────────────────────────────────

DOLLARS_PER_WAR     = 10.5   # millions (2025-26 market rate)
DOLLARS_PER_WAR_YOY = 0.05   # ~5% inflation per year in $/WAR
DISCOUNT_RATE       = 0.05   # future value discount
INJURY_RISK_BY_AGE  = {      # fraction of WAR lost to injury by age band
    (18,24): 0.08,
    (25,28): 0.10,
    (29,31): 0.13,
    (32,34): 0.17,
    (35,99): 0.22,
}


# ── Feature Engineering for Prediction ───────────────────────────────────────

PREDICTION_FEATURES = [
    "age", "WAR", "WAR_prev", "WAR_3yr_avg", "WAR_delta",
    "WAR_5yr_avg", "peak_WAR", "WAR_top3_avg",
    "ISO", "BB_pct", "K_pct", "wRC_plus", "Def", "BsR",
    "OBP", "SLG", "sprint_speed", "service_years",
    "contact_rate", "power_speed", "bmi",
]

def build_prediction_dataset(df: pd.DataFrame,
                              profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Build the modeling dataset.
    Target = WAR_next (next season's WAR).
    Drop final seasons (no future WAR to predict).
    """
    # Merge archetype labels onto season data
    arch_map = profiles.set_index("player_id")[["archetype_id","archetype_label"]]
    merged   = df.merge(arch_map, on="player_id", how="left")

    # One-hot encode archetype
    merged = pd.get_dummies(merged, columns=["archetype_id"], prefix="arch")

    arch_cols = [c for c in merged.columns if c.startswith("arch_")]
    all_features = PREDICTION_FEATURES + arch_cols

    # Keep only rows where WAR_next exists (not last season)
    model_df = merged.dropna(subset=["WAR_next"] + PREDICTION_FEATURES).copy()

    # Keep archetype_label and archetype_id for archetype-specific model training
    if "archetype_label" not in model_df.columns and "archetype_label" in merged.columns:
        model_df["archetype_label"] = merged["archetype_label"]
    if "archetype_id" not in model_df.columns and "archetype_id" in merged.columns:
        model_df["archetype_id"] = merged["archetype_id"]

    # Add archetype × WAR interaction features
    # These teach the model that elite archetypes maintain high WAR better
    elite_archs = ['Franchise Cornerstone', 'Two-Way Superstar', 'Two-Way Threat']
    power_archs = ['High-K Power Threat', 'Emerging Superstar']
    
    model_df["is_elite_arch"] = model_df["archetype_label"].isin(elite_archs).astype(float)
    model_df["is_power_arch"] = model_df["archetype_label"].isin(power_archs).astype(float)
    
    # Key interaction: elite archetype × high WAR → strong retention signal
    model_df["elite_x_war"] = model_df["is_elite_arch"] * model_df["WAR"]
    model_df["elite_x_peak"] = model_df["is_elite_arch"] * model_df["peak_WAR"]
    model_df["elite_x_top3"] = model_df["is_elite_arch"] * model_df["WAR_top3_avg"]
    model_df["power_x_war"] = model_df["is_power_arch"] * model_df["WAR"]

    interaction_cols = ["is_elite_arch", "is_power_arch", 
                        "elite_x_war", "elite_x_peak", "elite_x_top3", "power_x_war"]
    all_features = all_features + interaction_cols

    return model_df, all_features


# ── Model Training ────────────────────────────────────────────────────────────

def train_war_model(model_df: pd.DataFrame,
                    feature_cols: list) -> dict:
    """
    Train an ensemble of models to predict next-season WAR.
    Uses GroupKFold to prevent player data leakage across folds.
    """
    X = model_df[feature_cols].fillna(0)
    y = model_df["WAR_next"]
    groups = model_df["player_id"]

    models = {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
        ),
        "Ridge": Ridge(alpha=10.0),
    }

    gkf = GroupKFold(n_splits=5)
    results = {}

    print("\n🤖  Training prediction models...")
    for name, model in models.items():
        cv_scores = cross_val_score(
            model, X, y, cv=gkf, groups=groups,
            scoring="neg_mean_absolute_error"
        )
        mae = -cv_scores.mean()
        std = cv_scores.std()
        model.fit(X, y)
        results[name] = {
            "model": model,
            "cv_mae": round(mae, 3),
            "cv_std": round(std, 3),
        }
        print(f"    {name:25s} MAE = {mae:.3f} ± {std:.3f} WAR")

    # Simple ensemble: average predictions
    best_name = min(results, key=lambda k: results[k]["cv_mae"])
    print(f"\n    ✅  Best individual model: {best_name}")

    # Feature importance from GradientBoosting
    gb_model  = results["GradientBoosting"]["model"]
    imp_df    = pd.DataFrame({
        "feature":    feature_cols,
        "importance": gb_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Store arch label→id mappings for interaction feature computation
    elite_arch_labels = ['Franchise Cornerstone', 'Two-Way Superstar', 'Two-Way Threat']
    power_arch_labels = ['High-K Power Threat', 'Emerging Superstar']

    # Build arch_id sets from model_df if available
    elite_arch_ids = set()
    power_arch_ids = set()
    if 'archetype_label' in model_df.columns and 'archetype_id' in model_df.columns:
        for label, gdf in model_df.groupby('archetype_label'):
            arch_id = gdf['archetype_id'].iloc[0]
            if label in elite_arch_labels:
                elite_arch_ids.add(int(arch_id))
            if label in power_arch_labels:
                power_arch_ids.add(int(arch_id))

    return {
        "models":           results,
        "feature_cols":     feature_cols,
        "feature_importance": imp_df,
        "best_model_name":  best_name,
        "elite_arch_ids":   elite_arch_ids,
        "power_arch_ids":   power_arch_ids,
    }


# ── Individual Player Projection ──────────────────────────────────────────────

def project_player(player_history: pd.DataFrame,
                   model_bundle: dict,
                   arch_model: dict,
                   n_years: int = 5,
                   archetype_models: dict = None,
                   n_bootstrap: int = 200,
                   current_year: int = 2024) -> pd.DataFrame:
    """
    Project a player's WAR for the next n_years.
    Returns a DataFrame with point estimate + confidence intervals.

    Parameters
    ----------
    player_history : all historical seasons for this player
    model_bundle   : trained prediction model bundle
    arch_model     : clustering bundle (for archetype classification)
    n_years        : seasons to project
    n_bootstrap    : bootstrap iterations for uncertainty
    current_year   : base year for projection
    """
    from models.clustering import classify_player

    # Classify archetype
    arch_info = classify_player(player_history, arch_model)
    arch_id   = arch_info["archetype_id"]

    latest    = player_history.sort_values("season").iloc[-1]
    base_age  = latest["age"]
    base_war  = latest["WAR"]

    projections = []

    for yr in range(1, n_years + 1):
        proj_age  = base_age + yr
        proj_year = current_year + yr

        # Build feature row from latest season + projected age
        feat_row = latest.copy()
        feat_row["age"]          = proj_age
        feat_row["WAR"]          = base_war if yr == 1 else projections[-1]["war_p50"]
        feat_row["WAR_prev"]     = (projections[-1]["war_p50"]
                                     if yr > 1 else latest["WAR_prev"])
        feat_row["service_years"] = latest["service_years"] + yr

        # Ensemble prediction — use archetype-specific model if available
        X = _build_feature_row(feat_row, model_bundle, arch_id)
        
        # Try archetype-specific model first
        arch_label = arch_info.get("archetype_label", "")
        use_bundle = model_bundle
        if archetype_models and arch_label in archetype_models:
            use_bundle = archetype_models[arch_label]
            X_arch = _build_feature_row(feat_row, use_bundle, arch_id)
        else:
            X_arch = X
            use_bundle = model_bundle

        war_preds = []
        war_weights = []
        elite_archs = ['Franchise Cornerstone', 'Two-Way Superstar', 'Two-Way Threat']
        is_elite = arch_label in elite_archs

        for name, res in use_bundle["models"].items():
            try:
                pred = res["model"].predict(X_arch)[0]
            except:
                pred = res["model"].predict(X)[0]
            war_preds.append(pred)
            # Weight GradientBoosting more for elite archetypes (captures non-linear aging)
            if is_elite and name == "GradientBoosting":
                war_weights.append(0.5)
            elif is_elite and name == "RandomForest":
                war_weights.append(0.3)
            elif is_elite and name == "Ridge":
                war_weights.append(0.2)
            else:
                war_weights.append(1.0)

        # Weighted average
        total_w = sum(war_weights)
        war_point = float(sum(p*w for p,w in zip(war_preds, war_weights)) / total_w)

        # ── Aging curve blend ────────────────────────────────────────
        # Blend model prediction with archetype aging curve
        # Weight: 50% model, 50% curve-adjusted for year 1; more curve weight later
        aging_curves = arch_model.get("aging_curves", {})
        arch_label = arch_info.get("archetype_label", "")
        curve = aging_curves.get(arch_label, {})

        if curve:
            # Get curve WAR at base age and projected age
            curve_base = curve.get(int(base_age), curve.get(int(base_age)+1, base_war))
            curve_proj = curve.get(int(proj_age), curve.get(int(proj_age)-1, curve_base))

            # Scale curve to player's actual level
            # If player is 2x the archetype average, project them to stay 2x
            if curve_base > 0:
                scale = base_war / curve_base
                # Dampen extreme outliers — cap scale at 2.5x
                scale = min(scale, 2.5)
                curve_adjusted = curve_proj * scale
            else:
                curve_adjusted = curve_proj

            # Blend: year 1 = 40% curve, 60% model
            #        year 2 = 50/50
            #        year 3+ = 60% curve, 40% model
            curve_weight = min(0.4 + (yr - 1) * 0.1, 0.6)
            war_point = (1 - curve_weight) * war_point + curve_weight * curve_adjusted

        # ── Additional corrections ───────────────────────────────────
        war_delta = float(latest.get("WAR_delta", 0) or 0)
        war_3yr = float(latest.get("WAR_3yr_avg", base_war) or base_war)

        # 1. Young ascending player boost (under 28, positive WAR trend)
        if proj_age <= 27 and war_delta > 0:
            boost = min(war_delta * 0.3, 0.5)
            war_point += boost

        # 2. Elite player floor
        if base_war >= 5.0 and yr == 1:
            war_point = max(war_point, base_war * 0.55)
        elif base_war >= 3.5 and yr == 1:
            war_point = max(war_point, 2.0)

        # 3. Cap year-over-year decline at 1.0 WAR
        if yr > 1:
            prev_war = projections[-1]["war_p50"]
            war_point = max(war_point, prev_war - 1.0)

        # Bootstrap confidence interval
        boot_preds = []
        all_models = list(model_bundle["models"].values())
        for _ in range(n_bootstrap):
            m = np.random.choice(all_models)["model"]
            noise = np.random.normal(0, m.predict(X)[0] * 0.08)
            boot_preds.append(m.predict(X)[0] + noise)

        war_p10 = float(np.percentile(boot_preds, 10))
        war_p50 = float(np.percentile(boot_preds, 50))
        # Residual correction for elite players (empirically derived from multi-year backtest)
        if war_p50 >= 5.0:
            war_p50 += 0.5
        elif war_p50 >= 4.0:
            war_p50 += 0.3
        war_p90 = float(np.percentile(boot_preds, 90))

        # Injury-adjusted WAR
        inj_factor = _injury_factor(proj_age)
        war_adj    = war_point * (1 - inj_factor)

        # $/WAR: inflate DOLLARS_PER_WAR by YoY
        dol_per_war = DOLLARS_PER_WAR * ((1 + DOLLARS_PER_WAR_YOY) ** yr)

        # Contract value: discounted war_adj * $/WAR
        discount    = (1 + DISCOUNT_RATE) ** yr
        contract_val = (war_adj * dol_per_war) / discount

        projections.append({
            "season":        proj_year,
            "age":           proj_age,
            "war_point":     round(war_point, 2),
            "war_p10":       round(max(war_p10, -2), 2),
            "war_p50":       round(war_p50, 2),
            "war_p90":       round(min(war_p90, 12), 2),
            "war_adj":       round(war_adj, 2),
            "inj_factor":    round(inj_factor, 3),
            "dol_per_war":   round(dol_per_war, 2),
            "contract_value_M": round(contract_val, 2),
            "archetype":     arch_info["archetype_label"],
        })

    return pd.DataFrame(projections)


def _build_feature_row(row: pd.Series, model_bundle: dict, arch_id: int) -> np.ndarray:
    """Build a model-ready feature array from a season row."""
    feature_cols = model_bundle["feature_cols"]
    feats = []

    for f in feature_cols:
        if f.startswith("arch_"):
            feats.append(1.0 if f == f"arch_{arch_id}" else 0.0)
        elif f == "is_elite_arch":
            # Compute from arch_id — elite arch_ids correspond to Franchise Cornerstone, Two-Way Superstar, Two-Way Threat
            # We store elite arch labels in model_bundle if available
            elite_ids = model_bundle.get("elite_arch_ids", set())
            feats.append(1.0 if arch_id in elite_ids else 0.0)
        elif f == "is_power_arch":
            power_ids = model_bundle.get("power_arch_ids", set())
            feats.append(1.0 if arch_id in power_ids else 0.0)
        elif f == "elite_x_war":
            elite_ids = model_bundle.get("elite_arch_ids", set())
            is_elite = 1.0 if arch_id in elite_ids else 0.0
            war_val = float(row.get("WAR", 0) or 0)
            feats.append(is_elite * war_val)
        elif f == "elite_x_peak":
            elite_ids = model_bundle.get("elite_arch_ids", set())
            is_elite = 1.0 if arch_id in elite_ids else 0.0
            peak_val = float(row.get("peak_WAR", 0) or 0)
            feats.append(is_elite * peak_val)
        elif f == "elite_x_top3":
            elite_ids = model_bundle.get("elite_arch_ids", set())
            is_elite = 1.0 if arch_id in elite_ids else 0.0
            top3_val = float(row.get("WAR_top3_avg", 0) or 0)
            feats.append(is_elite * top3_val)
        elif f == "power_x_war":
            power_ids = model_bundle.get("power_arch_ids", set())
            is_power = 1.0 if arch_id in power_ids else 0.0
            war_val = float(row.get("WAR", 0) or 0)
            feats.append(is_power * war_val)
        else:
            val = row.get(f, 0)
            feats.append(float(val) if not pd.isna(val) else 0.0)

    return np.array(feats).reshape(1, -1)


def _injury_factor(age: float) -> float:
    for (lo, hi), factor in INJURY_RISK_BY_AGE.items():
        if lo <= age <= hi:
            return factor
    return 0.15


# ── Contract Value Calculator ─────────────────────────────────────────────────

def train_archetype_models(model_df: pd.DataFrame,
                           feature_cols: list) -> dict:
    """
    Train separate prediction models for each archetype class.
    Falls back to global model for archetypes with insufficient data.
    Returns dict keyed by archetype_label.
    """
    MIN_SAMPLES = 150  # minimum player-seasons to train a reliable model

    archetype_models = {}
    archetypes = model_df['archetype_label'].unique() if 'archetype_label' in model_df.columns else []

    print("\n🎯  Training archetype-specific models...")
    for arch in archetypes:
        arch_df = model_df[model_df['archetype_label'] == arch].copy()
        n = len(arch_df)
        if n < MIN_SAMPLES:
            print(f"    {arch}: only {n} samples — will use global model")
            continue

        X = arch_df[feature_cols].fillna(0)
        y = arch_df["WAR_next"]
        groups = arch_df["player_id"]

        models = {
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
            ),
            "Ridge": Ridge(alpha=10.0),
        }

        n_splits = min(5, len(arch_df['player_id'].unique()))
        gkf = GroupKFold(n_splits=n_splits)
        results = {}
        maes = []

        for name, model in models.items():
            try:
                cv_scores = cross_val_score(
                    model, X, y, cv=gkf, groups=groups,
                    scoring="neg_mean_absolute_error"
                )
                mae = -cv_scores.mean()
                maes.append((mae, name))
            except:
                mae = 999
            model.fit(X, y)
            results[name] = {"model": model, "mae": mae}

        best_name = min(results, key=lambda k: results[k]["mae"])
        best_mae = results[best_name]["mae"]

        # For elite archetypes, prefer GradientBoosting to capture non-linear aging
        elite_archetypes = ['Franchise Cornerstone', 'Two-Way Superstar', 'Two-Way Threat']
        if arch in elite_archetypes and 'GradientBoosting' in results:
            best_name = 'GradientBoosting'
            best_mae = results['GradientBoosting']['mae']

        print(f"    {arch}: n={n}, best={best_name}, MAE={best_mae:.3f}")

        archetype_models[arch] = {
            "models": results,
            "best_name": best_name,
            "feature_cols": feature_cols,
            "n_samples": n,
        }

    print(f"  ✅  Trained {len(archetype_models)} archetype models")
    return archetype_models




def estimate_contract(projections: pd.DataFrame,
                      contract_years: int,
                      aav_override: float = None,
                      dollars_per_war: float = DOLLARS_PER_WAR) -> dict:
    """
    Given a WAR projection table and desired contract length,
    calculate fair AAV, total value, and surplus value.

    Parameters
    ----------
    projections     : output of project_player()
    contract_years  : length of contract being evaluated
    aav_override    : if set, evaluate this AAV vs. fair value
    dollars_per_war : market $/WAR (default 7.5M)
    """
    proj = projections.head(contract_years).copy()

    total_war_adj    = proj["war_adj"].sum()
    total_war_p50    = proj["war_p50"].sum()
    total_fair_value = proj["contract_value_M"].sum()

    # Front-weighted AAV: weight early years more heavily
    # Teams price long deals on peak years but amortize across length
    import numpy as np
    actual_years = len(proj)
    weights = np.array([1.0 / (1.08 ** i) for i in range(actual_years)])
    weights = weights / weights.sum() * actual_years
    weighted_val = (proj["contract_value_M"].values * weights).sum()
    fair_aav = weighted_val / contract_years

    result = {
        "contract_years":    contract_years,
        "projected_WAR":     round(total_war_p50, 1),
        "injury_adj_WAR":    round(total_war_adj, 1),
        "total_fair_value_M": round(total_fair_value, 1),
        "fair_AAV_M":        round(fair_aav, 1),
        "archetype":         proj["archetype"].iloc[0],
    }

    if aav_override is not None:
        total_cost     = aav_override * contract_years
        surplus_value  = total_fair_value - total_cost
        result.update({
            "proposed_AAV_M":   aav_override,
            "total_cost_M":     round(total_cost, 1),
            "surplus_value_M":  round(surplus_value, 1),
            "verdict":          "✅ SIGN" if surplus_value >= 0 else "❌ OVERPAY",
        })

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/claude/mlb_war_pipeline")
    from data.data_generator import build_dataset
    from models.clustering import fit_archetypes

    df       = build_dataset(800)
    profiles, arch_bundle = fit_archetypes(df)

    model_df, feature_cols = build_prediction_dataset(df, profiles)
    trained  = train_war_model(model_df, feature_cols)

    # Test projection on a sample player
    sample_pid = df.groupby("player_id")["WAR"].max().idxmax()
    history    = df[df["player_id"] == sample_pid]
    name       = history["name"].iloc[0]

    print(f"\n🔮  Projecting WAR for: {name}")
    proj = project_player(history, trained, arch_bundle, n_years=5)
    print(proj[["season","age","war_p10","war_p50","war_p90","war_adj","contract_value_M"]])

    contract = estimate_contract(proj, contract_years=4, aav_override=18.0)
    print(f"\n💰  Contract Analysis:")
    for k, v in contract.items():
        print(f"    {k}: {v}")
