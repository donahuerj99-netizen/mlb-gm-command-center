"""
MLB WAR Prediction Pipeline — Pitcher WAR Prediction
=====================================================
Predicts future WAR for pitchers using archetype-aware modeling.
Key differences from hitters:
  - Pitchers peak earlier (age 24-26) and decline faster
  - Injury risk is higher and more volatile
  - Starters and relievers have different aging curves
  - FIP is more predictive than ERA for future performance
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, GroupKFold
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DOLLARS_PER_WAR  = 10.5   # millions (2025-26 market rate)
DISCOUNT_RATE    = 0.05

PITCHER_INJURY_RISK = {
    (18, 25): 0.12,
    (26, 28): 0.14,
    (29, 31): 0.18,
    (32, 34): 0.22,
    (35, 99): 0.28,
}

PREDICTION_FEATURES = [
    "age", "WAR", "WAR_prev", "WAR_3yr_avg", "WAR_delta",
    "ERA", "FIP", "WHIP", "SO9", "BB9", "HR9",
    "K_pct", "BB_pct", "K_BB_ratio", "service_years",
]


def build_pitcher_prediction_dataset(df, profiles):
    arch_map = profiles.set_index("player_id")[["archetype_id","archetype_label"]]
    merged   = df.merge(arch_map, on="player_id", how="left")
    merged   = pd.get_dummies(merged, columns=["archetype_id"], prefix="arch")
    arch_cols = [c for c in merged.columns if c.startswith("arch_")]
    all_features = PREDICTION_FEATURES + arch_cols
    model_df = merged.dropna(subset=["WAR_next"] + PREDICTION_FEATURES).copy()
    return model_df, all_features


def train_pitcher_model(model_df, feature_cols):
    X      = model_df[feature_cols].fillna(0)
    y      = model_df["WAR_next"]
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

    gkf     = GroupKFold(n_splits=5)
    results = {}

    print("\n🤖  Training pitcher prediction models...")
    for name, model in models.items():
        cv_scores = cross_val_score(
            model, X, y, cv=gkf, groups=groups,
            scoring="neg_mean_absolute_error"
        )
        mae = -cv_scores.mean()
        model.fit(X, y)
        results[name] = {"model": model, "cv_mae": round(mae,3)}
        print(f"    {name:25s} MAE = {mae:.3f} WAR")

    best_name = min(results, key=lambda k: results[k]["cv_mae"])
    print(f"\n    ✅  Best model: {best_name}")

    imp_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": results["GradientBoosting"]["model"].feature_importances_,
    }).sort_values("importance", ascending=False)

    return {"models": results, "feature_cols": feature_cols,
            "feature_importance": imp_df, "best_model_name": best_name}


def project_pitcher(history, model_bundle, arch_bundle, n_years=5, current_year=2025):
    from models.pitcher_clustering import classify_pitcher
    arch_info = classify_pitcher(history, arch_bundle)
    arch_id   = arch_info["archetype_id"]
    latest    = history.sort_values("season").iloc[-1]
    base_age  = latest["age"]

    projections = []
    for yr in range(1, n_years + 1):
        proj_age  = base_age + yr
        proj_year = current_year + yr

        feat_row = latest.copy()
        feat_row["age"]           = proj_age
        feat_row["WAR"]           = projections[-1]["war_p50"] if yr > 1 else latest["WAR"]
        feat_row["WAR_prev"]      = projections[-1]["war_p50"] if yr > 1 else latest.get("WAR_prev", latest["WAR"])
        feat_row["service_years"] = latest["service_years"] + yr

        X = _build_feature_row(feat_row, model_bundle, arch_id)
        war_preds = [res["model"].predict(X)[0] for res in model_bundle["models"].values()]
        war_point = float(np.mean(war_preds))

        # ── Aging curve blend ────────────────────────────────────────
        base_war = float(latest.get("WAR", 0) or 0)
        war_delta = float(latest.get("WAR_delta", 0) or 0)

        aging_curves = arch_bundle.get("aging_curves", {})
        arch_label = arch_info.get("archetype_label", "")
        curve = aging_curves.get(arch_label, {})

        if curve:
            curve_base = curve.get(int(base_age), curve.get(int(base_age)+1, base_war))
            curve_proj = curve.get(int(proj_age), curve.get(int(proj_age)-1, curve_base))
            if curve_base > 0:
                scale = min(base_war / curve_base, 2.5)
                curve_adjusted = curve_proj * scale
            else:
                curve_adjusted = curve_proj
            curve_weight = min(0.4 + (yr - 1) * 0.1, 0.6)
            war_point = (1 - curve_weight) * war_point + curve_weight * curve_adjusted

        # ── Additional corrections ───────────────────────────────────
        # 1. Young ascending pitcher boost
        if proj_age <= 27 and war_delta > 0:
            boost = min(war_delta * 0.3, 0.4)
            war_point += boost

        # 2. Elite pitcher floor
        if base_war >= 4.0 and yr == 1:
            war_point = max(war_point, base_war * 0.55)
        elif base_war >= 2.5 and yr == 1:
            war_point = max(war_point, 1.5)

        # 3. Cap year-over-year decline at 1.0 WAR
        if yr > 1:
            prev_war = projections[-1]["war_p50"]
            war_point = max(war_point, prev_war - 1.0)

        boot = [np.random.choice(list(model_bundle["models"].values()))["model"].predict(X)[0]
                + np.random.normal(0, 0.15) for _ in range(50)]
        war_p10 = float(np.percentile(boot, 10))
        war_p50 = float(np.percentile(boot, 50))
        # Residual correction for elite pitchers (empirically derived from backtest)
        if war_p50 >= 3.0:
            war_p50 += 0.3
        war_p90 = float(np.percentile(boot, 90))

        inj_factor = _pitcher_injury_factor(proj_age)
        war_adj    = war_point * (1 - inj_factor)
        dol_per_war = DOLLARS_PER_WAR * ((1.05) ** yr)
        contract_val = (war_adj * dol_per_war) / ((1 + DISCOUNT_RATE) ** yr)

        projections.append({
            "season": proj_year, "age": proj_age,
            "war_point": round(war_point, 2),
            "war_p10": round(max(war_p10, -2), 2),
            "war_p50": round(war_p50, 2),
            "war_p90": round(min(war_p90, 12), 2),
            "war_adj": round(war_adj, 2),
            "inj_factor": round(inj_factor, 3),
            "contract_value_M": round(contract_val, 2),
            "archetype": arch_info["archetype_label"],
            "role": arch_info.get("role","SP"),
        })
    return pd.DataFrame(projections)


def _build_feature_row(row, model_bundle, arch_id):
    base_feats = [float(row.get(f, 0)) if not pd.isna(row.get(f, 0)) else 0.0
                  for f in PREDICTION_FEATURES]
    arch_cols = [c for c in model_bundle["feature_cols"] if c.startswith("arch_")]
    arch_vec  = [1.0 if c == f"arch_{arch_id}" else 0.0 for c in arch_cols]
    return np.array(base_feats + arch_vec).reshape(1,-1)


def _pitcher_injury_factor(age):
    for (lo, hi), factor in PITCHER_INJURY_RISK.items():
        if lo <= age <= hi:
            return factor
    return 0.20


def estimate_pitcher_contract(projections, contract_years, aav_override=None):
    proj = projections.head(contract_years)
    total_war_adj   = proj["war_adj"].sum()
    total_war_p50   = proj["war_p50"].sum()
    total_fair_val  = proj["contract_value_M"].sum()

    # Front-weighted AAV: weight early years more heavily
    # Teams price long deals on peak years but amortize across length
    # Use weighted average where year 1 gets full weight, declining by 8% per year
    import numpy as np
    actual_years = len(proj)
    weights = np.array([1.0 / (1.08 ** i) for i in range(actual_years)])
    weights = weights / weights.sum() * actual_years
    weighted_val = (proj["contract_value_M"].values * weights).sum()
    fair_aav = weighted_val / contract_years

    result = {
        "contract_years":     contract_years,
        "projected_WAR":      round(total_war_p50, 1),
        "injury_adj_WAR":     round(total_war_adj, 1),
        "total_fair_value_M": round(total_fair_val, 1),
        "fair_AAV_M":         round(fair_aav, 1),
        "archetype":          proj["archetype"].iloc[0],
        "role":               proj["role"].iloc[0],
    }
    if aav_override is not None:
        total_cost   = aav_override * contract_years
        surplus      = total_fair_val - total_cost
        result.update({
            "proposed_AAV_M":  aav_override,
            "total_cost_M":    round(total_cost, 1),
            "surplus_value_M": round(surplus, 1),
            "verdict": "✅ SIGN" if surplus >= 0 else "❌ OVERPAY",
        })
    return result
