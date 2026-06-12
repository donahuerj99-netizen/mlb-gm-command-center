import unicodedata
"""
MLB Front Office Dashboard — Flask Backend
==========================================
Run with: python3 app.py
Then open: http://localhost:5000
"""

import os, sys, json, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="dashboard")


roster_cache = {}
team_stats_cache = {}  # League-wide team summary stats

# ── Global state (loaded once on startup) ────────────────────────────────────
DATA      = None
PROFILES  = None
ARCH_MODELS = {}
ARCH_BUNDLE = None
TRAINED   = None

def load_pipeline():
    global DATA, PROFILES, ARCH_BUNDLE, TRAINED, ARCH_MODELS
    print("Loading pipeline from pickle...")
    try:
        import pickle
        bundle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "model_bundle.pkl")
        print(f"Bundle path: {bundle_path}, exists: {os.path.exists(bundle_path)}")
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)
        DATA = bundle['DATA']
        PROFILES = bundle['PROFILES']
        ARCH_BUNDLE = bundle['ARCH_BUNDLE']
        TRAINED = bundle['TRAINED']
        ARCH_MODELS = bundle.get('ARCH_MODELS', {})
        print(f"Hitter pipeline loaded: {len(DATA):,} player-seasons")
        load_pitcher_pipeline()
        if os.environ.get('RENDER') == 'true':
            # On Render: just load proj caches into memory, skip MLB API calls
            load_proj_caches_only()
        else:
            prewarm_cache()
    except Exception as e:
        import traceback
        print(f"PIPELINE ERROR: {e}")
        traceback.print_exc()


def get_proj_war(hist, name):
    """Get projected WAR for a player, adding pitcher WAR for TWP players."""
    try:
        from models.prediction import project_player as _pp
        proj = _pp(hist, TRAINED, ARCH_BUNDLE, n_years=3, archetype_models=ARCH_MODELS)
        war = round(float(proj["war_p50"].iloc[0]), 1)
        # TWP check: if player also has pitcher data, add pitcher projection
        if PITCHER_DATA is not None:
            import unicodedata as _ud
            def _norm(s): return _ud.normalize('NFD', str(s)).encode('ascii','ignore').decode().lower().strip()
            p_hist = PITCHER_DATA[PITCHER_DATA['name'].apply(_norm) == _norm(name)]
            if not p_hist.empty and len(p_hist) >= 2:
                from models.pitcher_prediction import project_pitcher
                p_proj = project_pitcher(p_hist, PITCHER_TRAINED, PITCHER_BUNDLE, n_years=1, archetype_models=PITCHER_ARCH_MODELS)
                war = round(war + float(p_proj["war_p50"].iloc[0]), 1)
        return war
    except:
        return None

def get_twp_latest_war(name, season, base_war):
    """For TWP players, add pitcher WAR to latest_WAR display."""
    try:
        if PITCHER_DATA is None: return base_war
        import unicodedata as _ud
        def _norm(s): return _ud.normalize('NFD', str(s)).encode('ascii','ignore').decode().lower().strip()
        p_hist = PITCHER_DATA[(PITCHER_DATA['name'].apply(_norm) == _norm(name)) & 
                              (PITCHER_DATA['season'] == season)]
        if p_hist.empty: return base_war
        p_war = round(float(p_hist['WAR'].iloc[0]), 1)
        return round(base_war + p_war, 1)
    except:
        return base_war

# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")


@app.route("/api/players")
def list_players():
    """Return all unique player names for autocomplete, with encoding fixed."""
    def fix_encoding(name):
        try:
            return name.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return name
    names = sorted(set(fix_encoding(n) for n in DATA["name"].unique().tolist()))
    return jsonify(names)


@app.route("/api/player/<name>")
def get_player(name):
    """Return full projection + contract analysis for a player."""
    from models.prediction import project_player, estimate_contract
    from models.clustering import classify_player

    # Find player using normalized name matching (handles accents)
    name_clean = name.strip()
    history = find_player_in_data(DATA, name_clean)

    if history.empty:
        return jsonify({"error": f"Player '{name}' not found"}), 404
    if history["name"].nunique() > 1:
        matches = history["name"].unique().tolist()
        return jsonify({"ambiguous": True, "matches": matches})

    player_name = history["name"].iloc[0]
    history = DATA[DATA["name"] == player_name].copy()

    # Get archetype
    arch_info = classify_player(history, ARCH_BUNDLE)

    # Get latest season stats
    latest = history.sort_values("season").iloc[-1]

    contract_years = request.args.get("contract_years", 4, type=int)
    proposed_aav   = request.args.get("aav", None, type=float)
    n_years = max(contract_years, 5)  # always project at least 5 years

    proj = project_player(history, TRAINED, ARCH_BUNDLE, n_years=n_years, archetype_models=ARCH_MODELS)

    # TWP (Two-Way Player) handling: add pitcher projection on top of hitter projection
    import unicodedata as _ud
    def _norm(s): return _ud.normalize('NFD', str(s)).encode('ascii','ignore').decode().lower().strip()
    twp_pitcher_hist = PITCHER_DATA[PITCHER_DATA['name'].apply(_norm) == _norm(player_name)] if PITCHER_DATA is not None else None
    if twp_pitcher_hist is not None and not twp_pitcher_hist.empty and len(twp_pitcher_hist) >= 2:
        try:
            from models.pitcher_prediction import project_pitcher
            twp_proj = project_pitcher(twp_pitcher_hist, PITCHER_TRAINED, PITCHER_BUNDLE, n_years=n_years, archetype_models=PITCHER_ARCH_MODELS)
            # Add pitcher WAR to each projection year
            for i in range(min(len(proj), len(twp_proj))):
                proj.iloc[i, proj.columns.get_loc('war_p50')] = round(
                    proj.iloc[i]['war_p50'] + twp_proj.iloc[i]['war_p50'], 2)
                proj.iloc[i, proj.columns.get_loc('war_adj')] = round(
                    proj.iloc[i]['war_adj'] + twp_proj.iloc[i]['war_adj'], 2)
        except Exception as e:
            print(f"TWP pitcher projection failed: {e}")

    from models.prediction import estimate_contract
    contract = estimate_contract(
        proj,
        contract_years=contract_years,
        aav_override=proposed_aav
    )

    # Career summary
    # Calculate current age: last known age + years elapsed since last season
    current_year = 2026
    latest_season = int(latest["season"])
    last_known_age = int(latest["age"])
    current_age = last_known_age + (current_year - latest_season)

    # Determine if player is active (played in 2024 or 2025)
    is_active = latest_season >= 2024

    # TWP: add pitcher WAR to latest_WAR and career_WAR for display
    twp_latest_war = 0.0
    twp_career_war = 0.0
    if twp_pitcher_hist is not None and not twp_pitcher_hist.empty:
        twp_latest = twp_pitcher_hist[twp_pitcher_hist['season'] == latest_season]
        if not twp_latest.empty:
            twp_latest_war = round(float(twp_latest['WAR'].iloc[0]), 1)
        twp_career_war = round(float(twp_pitcher_hist['WAR'].sum()), 1)

    career = {
        "seasons":     int(history["season"].nunique()),
        "career_WAR":  round(float(history["WAR"].sum()) + twp_career_war, 1),
        "peak_WAR":    round(float(history["WAR"].max()), 1),
        "avg_WAR":     round(float(history["WAR"].mean()), 2),
        "latest_year": latest_season,
        "latest_WAR":  round(float(latest["WAR"]) + twp_latest_war, 1),
        "age":         current_age,
        "last_season_age": last_known_age,
        "is_active":   is_active,
        "position":    str(latest["position"]),
        "OBP":         round(float(latest.get("OBP", 0.320)), 3),
        "SLG":         round(float(latest.get("SLG", 0.420)), 3),
        "ISO":         round(float(latest.get("ISO", 0.150)), 3),
        "BB_pct":      round(float(latest.get("BB_pct", 0.085)) * 100, 1),
        "K_pct":       round(float(latest.get("K_pct", 0.215)) * 100, 1),
        "wRC_plus":    int(latest.get("wRC_plus", 100)),
        "Def":         round(float(latest.get("Def", 0)), 1),
        "Rtot":        round(float(latest.get("Rtot", 0)), 1),
        "Rdrs":        round(float(latest.get("Rdrs", 0)), 1),
        "Fld_pct":     round(float(latest.get("Fld_pct", 0.980)), 3),
        "sprint_speed":round(float(latest.get("sprint_speed", 27.0)), 1),
    }

    # WAR history for chart
    war_history = history.sort_values("season")[["season","age","WAR"]].to_dict("records")

    # Projection table
    proj_table = proj[["season","age","war_p10","war_p50","war_p90",
                        "war_adj","contract_value_M","archetype"]].to_dict("records")

    from models.clustering import get_hitter_secondary_tags
    secondary_tags = get_hitter_secondary_tags(latest)

    return jsonify({
        "name":           player_name,
        "archetype":      arch_info,
        "secondary_tags": secondary_tags,
        "career":         career,
        "war_history":    war_history,
        "projection":     proj_table,
        "contract":       contract,
    })


@app.route("/api/compare")
def compare_players():
    """Compare two players side by side."""
    p1 = request.args.get("p1", "")
    p2 = request.args.get("p2", "")
    results = {}
    for name in [p1, p2]:
        with app.test_request_context(f"/api/player/{name}"):
            r = get_player(name)
            if hasattr(r, "get_json"):
                results[name] = r.get_json()
    return jsonify(results)


@app.route("/api/archetypes")
def get_archetypes():
    """Return archetype summary table."""
    summary = ARCH_BUNDLE["summary"].to_dict("records")
    return jsonify(summary)


@app.route("/api/market")
def get_market():
    """Return $/WAR market data by age."""
    merged = DATA.merge(
        PROFILES.set_index("player_id")[["archetype_label"]],
        on="player_id", how="left"
    )
    fa = merged[merged["service_years"] >= 6].copy()
    fa["dollar_per_war"] = fa["salary_M"] / fa["WAR"].clip(lower=0.5)
    fa = fa[fa["dollar_per_war"] < 40]

    by_age = (fa.groupby(fa["age"].astype(int))["dollar_per_war"]
                .median().reset_index()
                .rename(columns={"age":"age","dollar_per_war":"median_dolar_per_war"}))
    by_age = by_age[(by_age["age"] >= 22) & (by_age["age"] <= 40)]

    return jsonify(by_age.to_dict("records"))



# ── Pitcher Pipeline ──────────────────────────────────────────────────────────
PITCHER_DATA     = None
PITCHER_ARCH_MODELS = {}
PITCHER_PROFILES = None
PITCHER_BUNDLE   = None
PITCHER_TRAINED  = None

def load_pitcher_pipeline():
    global PITCHER_DATA, PITCHER_PROFILES, PITCHER_BUNDLE, PITCHER_TRAINED, PITCHER_ARCH_MODELS
    print("Loading pitcher pipeline from pickle...")
    try:
        import pickle
        bundle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pitcher_model_bundle.pkl")
        print(f"Pitcher bundle path: {bundle_path}, exists: {os.path.exists(bundle_path)}")
        with open(bundle_path, 'rb') as f:
            bundle = pickle.load(f)
        PITCHER_DATA = bundle['PITCHER_DATA']
        PITCHER_PROFILES = bundle['PITCHER_PROFILES']
        PITCHER_BUNDLE = bundle['PITCHER_BUNDLE']
        PITCHER_TRAINED = bundle['PITCHER_TRAINED']
        PITCHER_ARCH_MODELS = bundle.get('PITCHER_ARCH_MODELS', {})
        print(f"Pitcher pipeline loaded: {len(PITCHER_DATA):,} pitcher-seasons")
    except Exception as e:
        import traceback
        print(f"PITCHER PIPELINE ERROR: {e}")
        traceback.print_exc()


@app.route("/api/pitcher/<name>")
def get_pitcher(name):
    from models.pitcher_prediction import project_pitcher, estimate_pitcher_contract
    from models.pitcher_clustering import classify_pitcher

    if PITCHER_DATA is None:
        return jsonify({"error": "Pitcher pipeline not loaded"}), 503

    name_clean = name.strip()
    history = find_player_in_data(PITCHER_DATA, name_clean)
    if history.empty:
        return jsonify({"error": f"Pitcher '{name}' not found"}), 404
    if history["name"].nunique() > 1:
        return jsonify({"ambiguous": True, "matches": history["name"].unique().tolist()})

    pitcher_name = history["name"].iloc[0]
    history = PITCHER_DATA[PITCHER_DATA["name"] == pitcher_name].copy()
    arch_info = classify_pitcher(history, PITCHER_BUNDLE)
    latest = history.sort_values("season").iloc[-1]

    contract_years = request.args.get("contract_years", 3, type=int)
    proposed_aav   = request.args.get("aav", None, type=float)

    proj_years = max(contract_years, 5)  # always project at least 5 years
    proj     = project_pitcher(history, PITCHER_TRAINED, PITCHER_BUNDLE, n_years=proj_years, archetype_models=PITCHER_ARCH_MODELS)
    contract = estimate_pitcher_contract(proj, contract_years, proposed_aav)

    current_year  = 2026
    latest_season = int(latest["season"])
    current_age   = int(latest["age"]) + (current_year - latest_season)

    career = {
        "seasons":     int(history["season"].nunique()),
        "career_WAR":  round(float(history["WAR"].sum()), 1),
        "peak_WAR":    round(float(history["WAR"].max()), 1),
        "latest_year": latest_season,
        "latest_WAR":  round(float(latest["WAR"]), 1),
        "age":         current_age,
        "is_active":   latest_season >= 2024,
        "role":        str(latest.get("role","SP")),
        "ERA":         round(float(latest.get("ERA", 4.50)), 2),
        "FIP":         round(float(latest.get("FIP", 4.50)), 2),
        "WHIP":        round(float(latest.get("WHIP", 1.35)), 3),
        "SO9":         round(float(latest.get("SO9", 8.0)), 1),
        "BB9":         round(float(latest.get("BB9", 3.0)), 1),
        "HR9":         round(float(latest.get("HR9", 1.2)), 2),
        "K_pct":       round(float(latest.get("K_pct", 0.22)) * 100, 1),
        "BB_pct":      round(float(latest.get("BB_pct", 0.085)) * 100, 1),
        "ERA_plus":    int(latest.get("ERA_plus", 100)),
        "IP":          round(float(latest.get("IP") or latest.get("IP_x") or 0), 1),
        "GS":          int(float(latest.get("GS", 0) or 0)),
    }

    war_history = history.sort_values("season")[["season","age","WAR"]].to_dict("records")
    proj_table  = proj[["season","age","war_p10","war_p50","war_p90","war_adj","contract_value_M","archetype"]].to_dict("records")

    from models.pitcher_clustering import get_secondary_tags
    secondary_tags = get_secondary_tags(latest)

    return jsonify({
        "name":           pitcher_name,
        "archetype":      arch_info,
        "secondary_tags": secondary_tags,
        "career":         career,
        "war_history":    war_history,
        "projection":     proj_table,
        "contract":       contract,
        "type":           "pitcher",
    })


@app.route("/api/pitchers")
def list_pitchers():
    if PITCHER_DATA is None:
        return jsonify([])
    def fix_encoding(name):
        try:
            return name.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            return name
    names = sorted(set(fix_encoding(n) for n in PITCHER_DATA["name"].unique().tolist()))
    return jsonify(names)


@app.route("/api/roster_pitchers/<team>")
def get_roster_pitchers(team):
    if PITCHER_DATA is None:
        return jsonify([])
    cache_key = f"pitchers_{team}"
    if cache_key in roster_cache:
        return jsonify(roster_cache[cache_key])
    from models.pitcher_clustering import classify_pitcher
    from models.pitcher_prediction import project_pitcher, estimate_pitcher_contract

    recent = list(PITCHER_DATA[
        (PITCHER_DATA["season"] == 2025) & (PITCHER_DATA["team"] == team)
    ]["name"].unique())

    if len(recent) < 3:
        recent2024 = list(PITCHER_DATA[
            (PITCHER_DATA["season"] == 2024) & (PITCHER_DATA["team"] == team)
        ]["name"].unique())
        recent = recent + [p for p in recent2024 if p not in recent]

    roster = []
    for name in recent:
        try:
            history  = find_player_in_data(PITCHER_DATA, name).copy()
            if history.empty: continue
            latest   = history.sort_values("season").iloc[-1]
            arch     = classify_pitcher(history, PITCHER_BUNDLE)
            proj     = project_pitcher(history, PITCHER_TRAINED, PITCHER_BUNDLE, n_years=3, archetype_models=PITCHER_ARCH_MODELS)
            proj_war = round(float(proj["war_p50"].iloc[0]), 1)
            current_age = int(latest["age"]) + (2026 - int(latest["season"]))
            roster.append({
                "name":       name,
                "role":       str(latest.get("role","SP")),
                "age":        current_age,
                "latest_WAR": round(float(latest["WAR"]), 1),
                "proj_war":   proj_war,
                "ERA":        round(float(latest.get("ERA", 4.50)), 2),
                "FIP":        round(float(latest.get("FIP", 4.50)), 2),
                "archetype":  arch["archetype_label"],
                "is_active":  int(latest["season"]) >= 2024,
            })
        except Exception:
            continue

    roster.sort(key=lambda x: x["latest_WAR"], reverse=True)
    return jsonify(roster)






@app.route("/api/roster/<team>")
def get_roster(team):
    """Return all players on a team's 2024-2025 roster with basic projections."""
    cache_key = team
    if cache_key in roster_cache:
        return jsonify(roster_cache[cache_key])
    from models.prediction import project_player, estimate_contract
    from models.clustering import classify_player

    # Use 2025 only, fall back to 2024 if fewer than 5 players
    recent_2025 = list(DATA[(DATA["season"] == 2025) & (DATA["team"] == team)]["name"].unique())
    if len(recent_2025) >= 5:
        team_players = recent_2025
    else:
        recent_2024 = list(DATA[(DATA["season"] == 2024) & (DATA["team"] == team)]["name"].unique())
        team_players = recent_2025 + [p for p in recent_2024 if p not in recent_2025]

    if len(team_players) == 0:
        return jsonify([])

    roster = []
    for name in team_players:
        try:
            history = DATA[DATA["name"] == name].copy()
            if history.empty:
                continue

            latest  = history.sort_values("season").iloc[-1]
            arch    = classify_player(history, ARCH_BUNDLE)

            # Quick 3-year projection
            proj    = project_player(history, TRAINED, ARCH_BUNDLE, n_years=3, archetype_models=ARCH_MODELS)
            proj_war = round(float(proj["war_p50"].iloc[0]), 1)

            current_year   = 2026
            latest_season  = int(latest["season"])
            last_known_age = int(latest["age"])
            current_age    = last_known_age + (current_year - latest_season)

            roster.append({
                "name":       name,
                "position":   str(latest["position"]),
                "age":        current_age,
                "latest_WAR": round(float(latest["WAR"]), 1),
                "proj_war":   proj_war,
                "wRC_plus":   int(latest.get("wRC_plus", 100)),
                "archetype":  arch["archetype_label"],
                "is_active":  latest_season >= 2024,
            })
        except Exception:
            continue

    roster.sort(key=lambda x: x["latest_WAR"], reverse=True)
    roster_cache[cache_key] = roster
    return jsonify(roster)



# ── Name normalization ───────────────────────────────────────────────────────
def normalize_name(name):
    """Normalize player names to handle encoding differences between data sources."""
    if not isinstance(name, str):
        return str(name)
    # Fix latin-1 misread as UTF-8 (e.g. AcuÃ±a -> Acuña)
    try:
        fixed = name.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        fixed = name
    # Strip accents for comparison
    return unicodedata.normalize('NFKD', fixed).encode('ascii', 'ignore').decode('ascii').lower().strip()

def find_player_in_data(df, name, team=None):
    """Find a player in the dataframe using normalized name matching.
    If team is provided, prefer matches from that team to resolve name collisions.
    """
    norm_target = normalize_name(name)
    # Try exact normalized match first
    mask = df['name'].apply(lambda n: normalize_name(n) == norm_target)
    result = df[mask]
    if not result.empty:
        # If team provided and multiple players match, prefer team match
        if team and 'team' in df.columns:
            team_match = result[result['team'] == team]
            if not team_match.empty:
                return team_match
            # If no team match found, this might be a different player with same name
            # Check if any match has a very different team — if so, return empty
            matched_teams = set(result['team'].unique()) - {'Multiple'}
            if team not in matched_teams and len(matched_teams) > 0:
                return pd.DataFrame()  # Different player, no data
        return result
    # Try without suffix (Jr., Sr., II, III)
    norm_no_suffix = norm_target.replace(' jr.','').replace(' sr.','').replace(' ii','').replace(' iii','').strip()
    mask2 = df['name'].apply(lambda n: normalize_name(n).replace(' jr.','').replace(' sr.','').replace(' ii','').replace(' iii','').strip() == norm_no_suffix)
    result2 = df[mask2]
    if not result2.empty:
        if team and 'team' in df.columns:
            team_match = result2[result2['team'] == team]
            if not team_match.empty:
                return team_match
        return result2
    return pd.DataFrame()


def normalize_name(name):
    """Normalize player names to handle encoding differences between data sources."""
    if not isinstance(name, str):
        return str(name)
    # Fix latin-1 misread as UTF-8 (e.g. AcuÃ±a -> Acuña)
    try:
        fixed = name.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        fixed = name
    # Strip accents for comparison
    return unicodedata.normalize('NFKD', fixed).encode('ascii', 'ignore').decode('ascii').lower().strip()

def find_player_in_data(df, name):
    """Find a player in the dataframe using normalized name matching."""
    norm_target = normalize_name(name)
    # Try exact normalized match first
    mask = df['name'].apply(lambda n: normalize_name(n) == norm_target)
    result = df[mask]
    if not result.empty:
        return result
    # Try without suffix (Jr., Sr., II, III)
    norm_no_suffix = norm_target.replace(' jr.','').replace(' sr.','').replace(' ii','').replace(' iii','').strip()
    mask2 = df['name'].apply(lambda n: normalize_name(n).replace(' jr.','').replace(' sr.','').replace(' ii','').replace(' iii','').strip() == norm_no_suffix)
    return df[mask2]

# ── MLB Stats API Integration ─────────────────────────────────────────────────

import urllib.request
import json
import ssl

# Team abbreviation mapping: MLB API -> BBRef
MLB_API_TO_BBREF = {
    "AZ":"ARI", "ATH":"OAK", "CWS":"CHW", "WSH":"WSN",
    "ATL":"ATL", "BAL":"BAL", "BOS":"BOS", "CHC":"CHC",
    "CIN":"CIN", "CLE":"CLE", "COL":"COL", "DET":"DET",
    "HOU":"HOU", "KC":"KCR",  "LAA":"LAA", "LAD":"LAD",
    "MIA":"MIA", "MIL":"MIL", "MIN":"MIN", "NYM":"NYM",
    "NYY":"NYY", "PHI":"PHI", "PIT":"PIT", "SD":"SDP",
    "SF":"SFG",  "SEA":"SEA", "STL":"STL", "TB":"TBR",
    "TEX":"TEX", "TOR":"TOR",
}

BBREF_TO_MLB_ID = {
    "ARI":109, "OAK":133, "ATL":144, "BAL":110, "BOS":111,
    "CHC":112, "CHW":145, "CIN":113, "CLE":114, "COL":115,
    "DET":116, "HOU":117, "KCR":118, "LAA":108, "LAD":119,
    "MIA":146, "MIL":158, "MIN":142, "NYM":121, "NYY":147,
    "PHI":143, "PIT":134, "SDP":135, "SFG":137, "SEA":136,
    "STL":138, "TBR":139, "TEX":140, "TOR":141, "WSN":120,
}

live_roster_cache = {}
team_stats_cache = {}  # League-wide team summary stats

def fetch_live_roster(bbref_team):
    """Fetch current 2026 roster from MLB Stats API."""
    if bbref_team in live_roster_cache:
        return live_roster_cache[bbref_team]

    team_id = BBREF_TO_MLB_ID.get(bbref_team)
    if not team_id:
        return None

    try:
        ctx = ssl.create_default_context()
        url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=40Man"
        with urllib.request.urlopen(url, timeout=10, context=ctx) as r:
            data = json.loads(r.read())

        roster = []
        for p in data.get("roster", []):
            status = p.get("status", {}).get("description", "Active")
            # Only include active, IL-10, IL-15, IL-60 players (skip NRI, minors)
            if any(x in status for x in ["Active", "Day", "Injured", "Bereavement"]) or status == "Active":
                roster.append({
                    "name":     p["person"]["fullName"],
                    "position": p["position"]["abbreviation"],
                    "jersey":   p.get("jerseyNumber", ""),
                    "status":   status,
                    "on_il":    "Day" in status or "Injured" in status,
                })

        live_roster_cache[bbref_team] = roster
        return roster
    except Exception as e:
        print(f"MLB API error for {bbref_team}: {e}")
        return None


@app.route("/api/live_roster/<team>")
def get_live_roster(team):
    """Get current 2026 roster - uses pre-warmed cache if available."""
    # Return cached version instantly if available
    if team in live_roster_cache:
        cached = live_roster_cache[team]
        # Compute proj_war on first access if not yet done
        needs_proj = any(p.get('proj_war') is None for p in cached.get('hitters',[]) + cached.get('pitchers',[]))
        if needs_proj:
            import json, os
            proj_cache_path = os.path.join(os.path.dirname(__file__), 'data', f'proj_cache_{team}.json')
            # Try loading from disk first
            if os.path.exists(proj_cache_path):
                try:
                    with open(proj_cache_path, 'r') as f:
                        proj_map = json.load(f)
                    for p in cached.get('hitters', []) + cached.get('pitchers', []):
                        if p['name'] in proj_map:
                            p['proj_war'] = proj_map[p['name']]
                    print(f"  💾 Loaded proj_war from disk for {team}")
                    needs_proj = False
                except:
                    pass

            if needs_proj:
                from models.prediction import project_player
                from models.pitcher_prediction import project_pitcher
                proj_map = {}
                for p in cached.get('hitters', []):
                    if p.get('proj_war') is None and p.get('in_db'):
                        try:
                            hist = find_player_in_data(DATA, p['name'])
                            if not hist.empty:
                                p['proj_war'] = get_proj_war(hist, p['name'])
                            else:
                                p['proj_war'] = 'N/A'
                        except:
                            p['proj_war'] = 'N/A'
                        proj_map[p['name']] = p['proj_war']
                for p in cached.get('pitchers', []):
                    if p.get('proj_war') is None and p.get('in_db'):
                        try:
                            hist = find_player_in_data(PITCHER_DATA, p['name'])
                            if not hist.empty:
                                proj = project_pitcher(hist, PITCHER_TRAINED, PITCHER_BUNDLE, n_years=3, archetype_models=PITCHER_ARCH_MODELS)
                                p['proj_war'] = round(float(proj["war_p50"].iloc[0]), 1)
                            else:
                                p['proj_war'] = 'N/A'
                        except:
                            p['proj_war'] = 'N/A'
                        proj_map[p['name']] = p['proj_war']
                # Save to disk for future restarts
                try:
                    with open(proj_cache_path, 'w') as f:
                        json.dump(proj_map, f)
                    print(f"  💾 Saved proj_war to disk for {team}")
                except Exception as e:
                    print(f"  ⚠️ Could not save proj cache: {e}")
        return jsonify(cached)

    from models.prediction import project_player, estimate_contract
    from models.clustering import classify_player
    from models.pitcher_clustering import classify_pitcher
    from models.pitcher_prediction import project_pitcher

    live = fetch_live_roster(team)
    if live is None:
        return jsonify({"error": f"Could not fetch live roster for {team}"}), 404

    hitters  = []
    pitchers = []

    for player in live:
        name     = player["name"]
        position = player["position"]
        is_pitcher = position in ["P", "SP", "RP", "CL"]

        # Try to find in our database
        if is_pitcher:
            history = find_player_in_data(PITCHER_DATA, name) if PITCHER_DATA is not None else None

            if history is not None and not history.empty:
                try:
                    latest   = history.sort_values("season").iloc[-1]
                    arch     = classify_pitcher(history, PITCHER_BUNDLE)
                    proj     = project_pitcher(history, PITCHER_TRAINED, PITCHER_BUNDLE, n_years=3, archetype_models=PITCHER_ARCH_MODELS)
                    proj_war = round(float(proj["war_p50"].iloc[0]), 1)
                    pitchers.append({
                        "name":       name,
                        "position":   position,
                        "jersey":     player.get("jersey",""),
                        "role":       str(latest.get("role","SP")),
                        "age":        int(latest["age"]) + (2026 - int(latest["season"])),
                        "latest_WAR": round(float(latest["WAR"]), 1),
                        "proj_war":   proj_war,
                        "ERA":        round(float(latest.get("ERA", 4.50)), 2),
                        "FIP":        round(float(latest.get("FIP", 4.50)), 2),
                        "archetype":  arch["archetype_label"],
                        "in_db":      True,
                        "on_il":      player.get("on_il", False),
                        "status":     player.get("status", "Active"),
                        "war_season": int(latest["season"]),
                    })
                except Exception:
                    pitchers.append({"name":name,"position":position,"jersey":player["jersey"],"role":position,"age":"?","latest_WAR":"N/A","proj_war":"N/A","ERA":"N/A","FIP":"N/A","archetype":"Unknown","in_db":False})
            else:
                pitchers.append({"name":name,"position":position,"jersey":player["jersey"],"role":position,"age":"?","latest_WAR":"N/A","proj_war":"N/A","ERA":"N/A","FIP":"N/A","archetype":"Insufficient Data","in_db":False})
        else:
            history = find_player_in_data(DATA, name)

            if not history.empty:
                try:
                    latest   = history.sort_values("season").iloc[-1]
                    arch     = classify_player(history, ARCH_BUNDLE)
                    proj     = project_player(history, TRAINED, ARCH_BUNDLE, n_years=3, archetype_models=ARCH_MODELS)
                    proj_war = round(float(proj["war_p50"].iloc[0]), 1)
                    hitters.append({
                        "name":       name,
                        "position":   position,
                        "jersey":     player["jersey"],
                        "age":        int(latest["age"]) + (2026 - int(latest["season"])),
                        "latest_WAR": get_twp_latest_war(name, int(latest["season"]), round(float(latest["WAR"]), 1)),
                        "proj_war":   proj_war,
                        "wRC_plus":   int(latest.get("wRC_plus", 100)),
                        "archetype":  arch["archetype_label"],
                        "in_db":      True,
                        "on_il":      player.get("on_il", False),
                        "status":     player.get("status", "Active"),
                        "war_season": int(latest["season"]),
                    })
                except Exception:
                    hitters.append({"name":name,"position":position,"jersey":player["jersey"],"age":"?","latest_WAR":"N/A","proj_war":"N/A","wRC_plus":"N/A","archetype":"Unknown","in_db":False})
            else:
                hitters.append({"name":name,"position":position,"jersey":player["jersey"],"age":"?","latest_WAR":"N/A","proj_war":"N/A","wRC_plus":"N/A","archetype":"Insufficient Data","in_db":False})

    hitters.sort(key=lambda x: x["latest_WAR"] if isinstance(x["latest_WAR"], (int,float)) else -99, reverse=True)
    pitchers.sort(key=lambda x: x["latest_WAR"] if isinstance(x["latest_WAR"], (int,float)) else -99, reverse=True)

    return jsonify({"hitters": hitters, "pitchers": pitchers, "source": "MLB Stats API 2026"})


def load_proj_caches_only():
    """On cloud: load proj_war caches from disk without hitting MLB API."""
    import json, threading
    teams = [
        'ARI','ATL','BAL','BOS','CHC','CHW','CIN','CLE','COL','DET',
        'HOU','KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','OAK',
        'PHI','PIT','SDP','SEA','SFG','STL','TBR','TEX','TOR','WSN'
    ]
    loaded = 0
    for team in teams:
        proj_path = os.path.join(os.path.dirname(__file__), 'data', f'proj_cache_{team}.json')
        if os.path.exists(proj_path):
            try:
                with open(proj_path, 'r') as f:
                    proj_map = json.load(f)
                # Store as minimal cache entry so on-demand load skips recompute
                roster_proj_cache[team] = proj_map
                loaded += 1
            except:
                pass
    print(f"Loaded proj caches for {loaded}/30 teams from disk")

# Global proj cache (separate from live_roster_cache)
roster_proj_cache = {}

def prewarm_cache():
    """Pre-compute projections for all 30 teams in parallel."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _warm_team(bbref_team, team_id):
        """Process a single team — runs in parallel with other teams."""
        from models.prediction import project_player, estimate_contract
        from models.clustering import classify_player
        from models.pitcher_clustering import classify_pitcher
        from models.pitcher_prediction import project_pitcher
        import urllib.request, json, ssl

        ctx = ssl.create_default_context()
        try:
            url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=40Man"
            with urllib.request.urlopen(url, timeout=15, context=ctx) as r:
                data = json.loads(r.read())

            hitters, pitchers = [], []

            for player in data.get("roster", []):
                name     = player["person"]["fullName"]
                position = player["position"]["abbreviation"]
                status   = player.get("status", {}).get("description", "Active")
                if not any(x in status for x in ["Active","Day","Injured","Bereavement"]):
                    continue
                on_il = "Day" in status or "Injured" in status
                is_pitcher = position in ["P","SP","RP","CL"]

                if is_pitcher and PITCHER_DATA is not None:
                    history = find_player_in_data(PITCHER_DATA, name)
                    if not history.empty:
                        try:
                            latest = history.sort_values("season").iloc[-1]
                            arch   = classify_pitcher(history, PITCHER_BUNDLE)
                            # proj_war computed on-demand, not at startup
                            pitchers.append({
                                "name": name, "position": position,
                                "jersey": player.get("jerseyNumber",""),
                                "role": str(latest.get("role","SP")),
                                "age": int(latest["age"]) + (2026 - int(latest["season"])),
                                "latest_WAR": round(float(latest["WAR"]), 1),
                                "proj_war": None,
                                "ERA": round(float(latest.get("ERA", 4.50)), 2),
                                "FIP": round(float(latest.get("FIP", 4.50)), 2),
                                "archetype": arch["archetype_label"],
                                "in_db": True, "on_il": on_il, "status": status,
                                "war_season": int(latest["season"]),
                            })
                        except Exception:
                            pitchers.append({"name":name,"position":position,"jersey":player.get("jerseyNumber",""),"role":position,"age":"?","latest_WAR":"N/A","proj_war":"N/A","ERA":"N/A","FIP":"N/A","archetype":"Insufficient Data","in_db":False,"on_il":on_il,"status":status})
                    else:
                        pitchers.append({"name":name,"position":position,"jersey":player.get("jerseyNumber",""),"role":position,"age":"?","latest_WAR":"N/A","proj_war":"N/A","ERA":"N/A","FIP":"N/A","archetype":"Insufficient Data","in_db":False,"on_il":on_il,"status":status})
                elif not is_pitcher:
                    history = find_player_in_data(DATA, name)
                    if not history.empty:
                        try:
                            latest = history.sort_values("season").iloc[-1]
                            arch   = classify_player(history, ARCH_BUNDLE)
                            # proj_war computed on-demand, not at startup
                            hitters.append({
                                "name": name, "position": position,
                                "jersey": player.get("jerseyNumber",""),
                                "age": int(latest["age"]) + (2026 - int(latest["season"])),
                                "latest_WAR": get_twp_latest_war(name, int(latest["season"]), round(float(latest["WAR"]), 1)),
                                "proj_war": None,
                                "wRC_plus": int(latest.get("wRC_plus", 100)),
                                "archetype": arch["archetype_label"],
                                "in_db": True, "on_il": on_il, "status": status,
                                "war_season": int(latest["season"]),
                            })
                        except Exception:
                            hitters.append({"name":name,"position":position,"jersey":player.get("jerseyNumber",""),"age":"?","latest_WAR":"N/A","proj_war":"N/A","wRC_plus":"N/A","archetype":"Unknown","in_db":False,"on_il":on_il,"status":status})
                    else:
                        hitters.append({"name":name,"position":position,"jersey":player.get("jerseyNumber",""),"age":"?","latest_WAR":"N/A","proj_war":"N/A","wRC_plus":"N/A","archetype":"Insufficient Data","in_db":False,"on_il":on_il,"status":status})

            hitters.sort(key=lambda x: x["latest_WAR"] if isinstance(x["latest_WAR"],(int,float)) else -99, reverse=True)
            pitchers.sort(key=lambda x: x["latest_WAR"] if isinstance(x["latest_WAR"],(int,float)) else -99, reverse=True)
            return bbref_team, {"hitters": hitters, "pitchers": pitchers, "source": "MLB Stats API 2026"}
        except Exception as e:
            print(f"  ❌ Failed {bbref_team}: {e}")
            return bbref_team, None

    def _warm_all():
        import time
        t0 = time.time()
        print("🔥  Pre-warming all 30 teams in parallel...")
        # Use 15 workers — enough to parallelize all 30 teams without hammering the API
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = {executor.submit(_warm_team, team, tid): team
                      for team, tid in BBREF_TO_MLB_ID.items()}
            done = 0
            for future in as_completed(futures):
                bbref_team, result = future.result()
                if result:
                    live_roster_cache[bbref_team] = result
                    done += 1
                    h = len(result["hitters"])
                    p = len(result["pitchers"])
                    print(f"  ✅ [{done}/30] {bbref_team}: {h} hitters, {p} pitchers")
        elapsed = round(time.time() - t0)
        print(f"✅  All rosters cached in {elapsed}s!")

        # Now compute proj_war for all teams and build team stats cache
        print("📊  Computing proj_war for all teams...")
        from models.prediction import project_player
        from models.pitcher_prediction import project_pitcher
        import json, os

        for bbref_team in BBREF_TO_MLB_ID.keys():
            if bbref_team not in live_roster_cache:
                continue
            cached = live_roster_cache[bbref_team]
            proj_cache_path = os.path.join(os.path.dirname(__file__), 'data', f'proj_cache_{bbref_team}.json')

            # Load from disk if available
            proj_map = {}
            if os.path.exists(proj_cache_path):
                try:
                    with open(proj_cache_path, 'r') as f:
                        proj_map = json.load(f)
                except:
                    pass

            needs_save = False
            for p in cached.get('hitters', []):
                if p['name'] in proj_map:
                    p['proj_war'] = proj_map[p['name']]
                elif p.get('proj_war') is None and p.get('in_db'):
                    try:
                        hist = find_player_in_data(DATA, p['name'])
                        if not hist.empty:
                            p['proj_war'] = get_proj_war(hist, p['name'])
                        else:
                            p['proj_war'] = 'N/A'
                    except:
                        p['proj_war'] = 'N/A'
                    proj_map[p['name']] = p['proj_war']
                    needs_save = True

            for p in cached.get('pitchers', []):
                if p['name'] in proj_map:
                    p['proj_war'] = proj_map[p['name']]
                elif p.get('proj_war') is None and p.get('in_db'):
                    try:
                        hist = find_player_in_data(PITCHER_DATA, p['name'])
                        if not hist.empty:
                            proj = project_pitcher(hist, PITCHER_TRAINED, PITCHER_BUNDLE, n_years=3, archetype_models=PITCHER_ARCH_MODELS)
                            p['proj_war'] = round(float(proj["war_p50"].iloc[0]), 1)
                        else:
                            p['proj_war'] = 'N/A'
                    except:
                        p['proj_war'] = 'N/A'
                    proj_map[p['name']] = p['proj_war']
                    needs_save = True

            if needs_save:
                try:
                    with open(proj_cache_path, 'w') as f:
                        json.dump(proj_map, f)
                except:
                    pass

            # Compute team stats with proj_war now populated
            stats = compute_team_stats(bbref_team)
            if stats:
                team_stats_cache[bbref_team] = stats

        print("✅  Team stats ready!")

    t = threading.Thread(target=_warm_all, daemon=True)
    t.start()



@app.route("/api/comps/<name>")
def get_player_comps(name):
    """Find 5 most similar historical players based on statistical profile."""
    import unicodedata, numpy as np
    from sklearn.preprocessing import StandardScaler

    history = find_player_in_data(DATA, name)
    if history.empty:
        return jsonify({"error": f"Player not found: {name}"}), 404

    raw_name = str(history["name"].iloc[0])
    try:
        player_name = raw_name.encode("latin1").decode("utf8")
    except Exception:
        player_name = unicodedata.normalize("NFC", raw_name)
    latest = history.sort_values("season").iloc[-1]
    player_age = int(latest["age"])
    player_latest_season = int(latest["season"])

    COMP_FEATURES = ["wRC_plus","ISO","BB_pct","K_pct","Def","sprint_speed","WAR_3yr_avg","OBP","SLG"]

    def get_profile(row):
        return [float(row.get(f, 0) or 0) for f in COMP_FEATURES]

    target = get_profile(latest)

    def normalize_name(raw):
        s = str(raw)
        # Try latin1->utf8 fix for double-encoded names
        for enc in [("latin1","utf8"),("cp1252","utf8"),("utf8","utf8")]:
            try:
                s2 = s.encode(enc[0]).decode(enc[1])
                return unicodedata.normalize("NFC", s2)
            except Exception:
                continue
        return unicodedata.normalize("NFC", s)

    candidates = []
    for pid, grp in DATA.groupby("player_id"):
        cname = normalize_name(grp["name"].iloc[0])
        if cname == player_name:
            continue
        # Match by age, exclude seasons too recent to be comps
        age_match = grp[
            (grp["age"] >= player_age - 3) &
            (grp["age"] <= player_age + 3) &
            (grp["season"] <= player_latest_season - 3)
        ]
        if len(age_match) == 0:
            continue
        # Pick the season with the closest feature vector to target
        best_dist = float("inf")
        best_row = None
        for _, row in age_match.iterrows():
            vec = get_profile(row)
            d = sum((a - b) ** 2 for a, b in zip(target, vec))
            if d < best_dist:
                best_dist = d
                best_row = row
        if best_row is None:
            continue
        candidates.append({
            "name": cname,
            "season": int(best_row["season"]),
            "age": int(best_row["age"]),
            "vec": get_profile(best_row),
            "peak_war": round(float(grp["WAR"].max()), 1),
            "career_war": round(float(grp["WAR"].sum()), 1),
        })

    if not candidates:
        return jsonify({"comps": []})

    # Normalize and compute distances
    all_vecs = [target] + [c["vec"] for c in candidates]
    mat = np.array(all_vecs)
    scaler = StandardScaler()
    mat_scaled = scaler.fit_transform(mat)
    target_vec = mat_scaled[0]
    dists = [float(np.linalg.norm(mat_scaled[i+1] - target_vec)) for i in range(len(candidates))]

    for i, c in enumerate(candidates):
        c["distance"] = dists[i]
        del c["vec"]

    candidates.sort(key=lambda x: x["distance"])

    # Era-stratified selection: guarantee spread across 2000-2024
    era_buckets = {
        "2000-2009": [c for c in candidates if c["season"] <= 2009],
        "2010-2016": [c for c in candidates if 2010 <= c["season"] <= 2016],
        "2017-2024": [c for c in candidates if c["season"] >= 2017],
    }

    selected = []
    used_names = set()

    # Take best (closest) from each era bucket first
    for bucket in era_buckets.values():
        for c in bucket:
            if c["name"] not in used_names:
                selected.append(c)
                used_names.add(c["name"])
                break

    # Fill remaining slots with globally closest not yet selected
    for c in candidates:
        if len(selected) >= 5:
            break
        if c["name"] not in used_names:
            selected.append(c)
            used_names.add(c["name"])

    top5 = selected[:5]

    max_dist = max(c["distance"] for c in top5) if top5 else 1
    for c in top5:
        c["similarity"] = round((1 - c["distance"] / (max_dist * 1.5)) * 100)
        del c["distance"]

    from models.clustering import classify_player
    for c in top5:
        try:
            hist = DATA[DATA["name"] == c["name"]]
            # Use only seasons up to comp season for accurate archetype
            hist_to_season = hist[hist["season"] <= c["season"]]
            if hist_to_season.empty:
                hist_to_season = hist
            seasons_300pa = int((
                pd.to_numeric(hist_to_season["PA"], errors="coerce").fillna(0) >= 300
            ).sum())
            arch = classify_player(hist_to_season, ARCH_BUNDLE, seasons_300pa_override=seasons_300pa)
            c["archetype"] = arch["archetype_label"]
        except Exception:
            c["archetype"] = "Unknown"

    return jsonify({"comps": top5, "player": player_name, "age": player_age})


@app.route("/api/pitcher_comps/<name>")
def get_pitcher_comps(name):
    """Find 5 most similar historical pitchers."""
    import unicodedata, numpy as np
    from sklearn.preprocessing import StandardScaler

    if PITCHER_DATA is None:
        return jsonify({"comps": []})

    history = find_player_in_data(PITCHER_DATA, name)
    if history.empty:
        return jsonify({"error": f"Pitcher not found: {name}"}), 404

    raw_name = str(history["name"].iloc[0])
    try:
        player_name = raw_name.encode("latin1").decode("utf8")
    except Exception:
        player_name = unicodedata.normalize("NFC", raw_name)
    latest = history.sort_values("season").iloc[-1]
    player_age = int(latest["age"])
    player_latest_season = int(latest["season"])
    role = str(latest.get("role", "SP"))

    COMP_FEATURES = ["ERA","FIP","WHIP","SO9","BB9","HR9","K_pct","BB_pct"]

    def get_profile(row):
        return [float(row.get(f, 0) or 0) for f in COMP_FEATURES]

    target = get_profile(latest)

    candidates = []
    for pid, grp in PITCHER_DATA.groupby("player_id"):
        raw_cname = str(grp["name"].iloc[0])
        try:
            cname = raw_cname.encode("latin1").decode("utf8")
        except Exception:
            cname = unicodedata.normalize("NFC", raw_cname)
        if cname == player_name:
            continue
        if str(grp["role"].iloc[-1]) != role:
            continue
        age_match = grp[
            (grp["age"] >= player_age - 3) &
            (grp["age"] <= player_age + 3) &
            (grp["season"] <= player_latest_season - 3)
        ]
        if len(age_match) == 0:
            continue
        best = age_match.sort_values("season").iloc[0]
        candidates.append({
            "name": cname,
            "season": int(best["season"]),
            "age": int(best["age"]),
            "vec": get_profile(best),
            "peak_war": round(float(grp["WAR"].max()), 1),
            "career_war": round(float(grp["WAR"].sum()), 1),
            "ERA": round(float(best.get("ERA", 4.5)), 2),
        })

    if not candidates:
        return jsonify({"comps": []})

    all_vecs = [target] + [c["vec"] for c in candidates]
    mat = np.array(all_vecs)
    scaler = StandardScaler()
    mat_scaled = scaler.fit_transform(mat)
    target_vec = mat_scaled[0]
    dists = [float(np.linalg.norm(mat_scaled[i+1] - target_vec)) for i in range(len(candidates))]

    for i, c in enumerate(candidates):
        c["distance"] = dists[i]
        del c["vec"]

    candidates.sort(key=lambda x: x["distance"])
    top5 = candidates[:5]

    max_dist = max(c["distance"] for c in top5) if top5 else 1
    for c in top5:
        c["similarity"] = round((1 - c["distance"] / (max_dist * 1.5)) * 100)
        del c["distance"]

    from models.pitcher_clustering import classify_pitcher
    for c in top5:
        try:
            hist = PITCHER_DATA[PITCHER_DATA["name"] == c["name"]]
            hist_to_season = hist[hist["season"] <= c["season"]]
            if hist_to_season.empty:
                hist_to_season = hist
            arch = classify_pitcher(hist_to_season, PITCHER_BUNDLE)
            c["archetype"] = arch["archetype_label"]
        except Exception:
            c["archetype"] = "Unknown"

    return jsonify({"comps": top5, "player": player_name, "age": player_age})


@app.route("/api/historic_player/<name>/<int:season>")
def get_historic_player(name, season):
    """Look up any player's stats and archetype for a given season."""
    import unicodedata
    from models.clustering import classify_player

    if DATA is None or DATA.empty:
        return jsonify({"error": "Data not loaded yet"}), 503

    # Normalize search name
    search = unicodedata.normalize("NFC", name.strip().lower())

    # Find matching player rows
    def norm(n):
        try:
            n2 = str(n).encode("latin1").decode("utf8")
        except Exception:
            n2 = str(n)
        return unicodedata.normalize("NFC", n2).strip().lower()

    mask = DATA["name"].apply(norm) == search
    player_data = DATA[mask]

    if player_data.empty:
        # Try partial match
        mask2 = DATA["name"].apply(norm).str.contains(search, regex=False)
        player_data = DATA[mask2]
        if player_data.empty:
            return jsonify({"error": f"Player not found: {name}"}), 404

    # Get the requested season
    season_data = player_data[player_data["season"] == season]
    if season_data.empty:
        available = sorted(player_data["season"].unique().tolist())
        return jsonify({"error": f"No data for {name} in {season}", "available_seasons": available}), 404

    row = season_data.iloc[0]

    def safe(val, decimals=1):
        try:
            if val is None or (isinstance(val, float) and (val != val)):
                return None
            return round(float(val), decimals)
        except:
            return None

    def safe_int(val):
        try:
            if val is None or (isinstance(val, float) and (val != val)):
                return None
            return int(val)
        except:
            return None

    # Archetype — use full history for career context (seasons_300pa),
    # but only evaluate stats from the requested season row
    try:
        # Build a single-row df with correct seasons_300pa from full history
        eval_row = season_data.copy()
        seasons_300pa = int((pd.to_numeric(player_data["PA"], errors="coerce").fillna(0) >= 300).sum())
        # Only count seasons UP TO and including the requested season
        seasons_300pa_to_date = int((
            pd.to_numeric(
                player_data[player_data["season"] <= season]["PA"],
                errors="coerce"
            ).fillna(0) >= 300
        ).sum())
        eval_row["seasons_300pa"] = seasons_300pa_to_date
        arch_result = classify_player(eval_row, ARCH_BUNDLE, seasons_300pa_override=seasons_300pa_to_date)
        archetype = arch_result.get("archetype_label", "Unknown")
        archetype_class = arch_result.get("archetype_class", "")
    except Exception as e:
        archetype = "Unknown"
        archetype_class = ""

    # Career WAR up to this season
    career_to_date = player_data[player_data["season"] <= season]["WAR"].sum()

    # Display name
    raw_name = str(player_data["name"].iloc[0])
    try:
        display_name = raw_name.encode("latin1").decode("utf8")
    except:
        display_name = unicodedata.normalize("NFC", raw_name)

    result = {
        "name": display_name,
        "season": season,
        "age": safe_int(row.get("age")),
        "team": str(row.get("team", "")).strip(),
        "position": str(row.get("position", "")).strip(),
        "archetype": archetype,
        "archetype_class": archetype_class,
        "stats": {
            "WAR": safe(row.get("WAR")),
            "WAR_3yr_avg": safe(row.get("WAR_3yr_avg")),
            "career_WAR_to_date": safe(career_to_date),
            "PA": safe_int(row.get("PA")),
            "AVG": safe(row.get("AVG"), 3),
            "OBP": safe(row.get("OBP"), 3),
            "SLG": safe(row.get("SLG"), 3),
            "OPS": safe(row.get("OPS"), 3),
            "HR": safe_int(row.get("HR")),
            "SB": safe_int(row.get("SB")),
            "wRC_plus": safe_int(row.get("wRC_plus")),
            "ISO": safe(row.get("ISO"), 3),
            "BB_pct": safe(row.get("BB_pct"), 1),
            "K_pct": safe(row.get("K_pct"), 1),
            "Def": safe(row.get("Def")),
            "sprint_speed": safe(row.get("sprint_speed")),
            "oWAR": safe(row.get("oWAR")),
            "dWAR": safe(row.get("dWAR")),
        },
        "available_seasons": sorted(player_data["season"].unique().tolist())
    }

    return jsonify(result)


@app.route("/api/historic_pitcher/<name>/<int:season>")
def get_historic_pitcher(name, season):
    """Look up any pitcher's stats and archetype for a given season."""
    import unicodedata
    from models.pitcher_clustering import classify_pitcher

    if PITCHER_DATA is None or PITCHER_DATA.empty:
        return jsonify({"error": "Pitcher data not loaded yet"}), 503

    def norm(n):
        try:
            n2 = str(n).encode("latin1").decode("utf8")
        except Exception:
            n2 = str(n)
        return unicodedata.normalize("NFC", n2).strip().lower()

    search = unicodedata.normalize("NFC", name.strip().lower())
    mask = PITCHER_DATA["name"].apply(norm) == search
    player_data = PITCHER_DATA[mask]

    if player_data.empty:
        mask2 = PITCHER_DATA["name"].apply(norm).str.contains(search, regex=False)
        player_data = PITCHER_DATA[mask2]
        if player_data.empty:
            return jsonify({"error": f"Pitcher not found: {name}"}), 404

    season_data = player_data[player_data["season"] == season]
    if season_data.empty:
        available = sorted(player_data["season"].unique().tolist())
        return jsonify({"error": f"No data for {name} in {season}", "available_seasons": available}), 404

    row = season_data.iloc[0]

    def safe(val, decimals=2):
        try:
            if val is None or (isinstance(val, float) and (val != val)):
                return None
            return round(float(val), decimals)
        except:
            return None

    def safe_int(val):
        try:
            if val is None or (isinstance(val, float) and (val != val)):
                return None
            return int(val)
        except:
            return None

    # Archetype — season specific
    try:
        arch_result = classify_pitcher(season_data, PITCHER_BUNDLE)
        archetype = arch_result.get("archetype_label", "Unknown")
        archetype_class = arch_result.get("archetype_class", "")
    except Exception:
        archetype = "Unknown"
        archetype_class = ""

    raw_name = str(player_data["name"].iloc[0])
    try:
        display_name = raw_name.encode("latin1").decode("utf8")
    except:
        display_name = unicodedata.normalize("NFC", raw_name)

    career_to_date = player_data[player_data["season"] <= season]["WAR"].sum()

    result = {
        "name": display_name,
        "season": season,
        "age": safe_int(row.get("age")),
        "team": str(row.get("team", "")).strip(),
        "role": str(row.get("role", "")).strip(),
        "archetype": archetype,
        "archetype_class": archetype_class,
        "stats": {
            "WAR": safe(row.get("WAR")),
            "WAR_3yr_avg": safe(row.get("WAR_3yr_avg")),
            "career_WAR_to_date": safe(career_to_date),
            "ERA": safe(row.get("ERA")),
            "FIP": safe(row.get("FIP")),
            "WHIP": safe(row.get("WHIP")),
            "SO9": safe(row.get("SO9")),
            "BB9": safe(row.get("BB9")),
            "HR9": safe(row.get("HR9")),
            "H9": safe(row.get("H9")),
            "K_pct": safe(row.get("K_pct")),
            "BB_pct": safe(row.get("BB_pct")),
            "ERA_plus": safe_int(row.get("ERA_plus")),
            "IP": safe(row.get("IP_y")),
            "SO": safe_int(row.get("SO")),
            "BB": safe_int(row.get("BB")),
            "SV": safe_int(row.get("SV")),
            "GS": safe_int(row.get("GS_y")),
            "GB_pct": safe(row.get("GB_pct")),
            "oWAR": safe(row.get("RAA")),
            "dWAR": safe(row.get("WAA")),
        },
        "available_seasons": sorted(player_data["season"].unique().tolist())
    }

    return jsonify(result)



@app.route("/api/market_intel")
def get_market_intel():
    """Return sell-high and buy-low candidates based on WAR vs 3yr avg."""
    import pandas as pd
    import numpy as np

    TWP_PLAYERS = ['Shohei Ohtani']

    def fix_encoding(name):
        try:
            return name.encode('latin1').decode('utf-8')
        except:
            return name

    def get_notes(row, is_pitcher=False):
        notes = []
        age = row['age']
        war = float(row['WAR'])
        delta = float(row['delta'])
        peak = float(row['peak_WAR']) if not pd.isna(row['peak_WAR']) else war
        prev = float(row['WAR_prev']) if not pd.isna(row['WAR_prev']) else war
        name = row['name']

        if name in TWP_PLAYERS:
            return "TWP: Pitcher WAR only"

        if delta > 0:  # sell high
            if age >= 33:
                notes.append("AGE RISK")
            if war >= peak * 0.95:
                notes.append("CAREER PEAK")
        else:  # buy low
            if prev < 1.0 and peak >= 5.0 and delta < -2.5:
                notes.append("INJURY RELATED")
            elif prev < 0.5 and war < 1.5:
                notes.append("INJURY RELATED")

        return ", ".join(notes) if notes else ""

    def process(df, min_pa=None, min_ip=None, is_pitcher=False):
        df = df[df['season'] == 2025].copy()
        df['name'] = df['name'].apply(fix_encoding)
        if min_pa is not None:
            df = df[df['PA'] >= min_pa]
        if min_ip is not None:
            df = df[df['IP_x'] >= min_ip]
        df = df.dropna(subset=['WAR_3yr_avg'])
        df['delta'] = df['WAR'] - df['WAR_3yr_avg']
        df = df.dropna(subset=['delta'])

        # Filter out TWP pitcher side
        if is_pitcher:
            df = df[df['name'] != 'Shohei Ohtani']

        # Require meaningful service history
        df = df[df['service_years'].fillna(0) >= 4]

        # Exclude true ascending breakouts: at/near career peak AND young
        df = df[~((df['WAR'] >= df['peak_WAR'] * 0.95) & (df['age'] <= 31))]

        # Sell high: exclude elite consistent performers (3yr avg >= 6.0)
        if is_pitcher == False:
            df_sell = df[df['WAR_3yr_avg'] < 6.0]
        else:
            df_sell = df[df['WAR_3yr_avg'] < 6.0]

        # Buy low: cap age at 35 — older players declining isn't a buy opportunity
        df_buy = df[df['age'] <= 35]

        records = []
        for _, row in df.iterrows():
            records.append({
                'name': row['name'],
                'team': row['team'],
                'age': int(row['age']),
                'war': round(float(row['WAR']), 1),
                'war_3yr': round(float(row['WAR_3yr_avg']), 1),
                'delta': round(float(row['delta']), 1),
                'peak_war': round(float(row['peak_WAR']), 1) if not pd.isna(row['peak_WAR']) else None,
                'position': str(row.get('role', row.get('position', ''))),
                'notes': get_notes(row, is_pitcher),
                'is_pitcher': is_pitcher
            })

        rdf = pd.DataFrame(records)
        # Sell high: exclude elite consistent performers (3yr avg >= 6.0)
        sell_df = rdf[rdf['war_3yr'] < 6.0]
        sell_high = sell_df.nlargest(5, 'delta').to_dict('records')
        sell_high_ext = sell_df.nlargest(10, 'delta').to_dict('records')
        # Buy low: cap age at 35
        buy_df = rdf[rdf['age'] <= 35]
        buy_low = buy_df.nsmallest(5, 'delta').to_dict('records')
        buy_low_ext = buy_df.nsmallest(10, 'delta').to_dict('records')
        return sell_high, buy_low, sell_high_ext, buy_low_ext

    try:
        h = pd.read_csv('data/real_seasons.csv')
        p = pd.read_csv('data/pitcher_seasons.csv')
        h_sell, h_buy, h_sell_ext, h_buy_ext = process(h, min_pa=200, is_pitcher=False)
        p_sell, p_buy, p_sell_ext, p_buy_ext = process(p, min_ip=40, is_pitcher=True)
        return jsonify({
            'hitter_sell_high': h_sell,
            'hitter_buy_low': h_buy,
            'pitcher_sell_high': p_sell,
            'pitcher_buy_low': p_buy,
            'hitter_sell_high_ext': h_sell_ext,
            'hitter_buy_low_ext': h_buy_ext,
            'pitcher_sell_high_ext': p_sell_ext,
            'pitcher_buy_low_ext': p_buy_ext
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/api/payroll/<team>")
def get_team_payroll(team):
    """Return full contract/payroll data for a team."""
    import json, os
    contracts_path = os.path.join(os.path.dirname(__file__), 'data', 'contracts_clean.json')
    try:
        with open(contracts_path, 'r') as f:
            all_contracts = json.load(f)
    except Exception as e:
        return jsonify({"error": f"Could not load contracts: {e}"}), 500

    # Map team abbreviations to cots slugs
    TEAM_SLUG_MAP = {
        "ARI": "arizona-diamondbacks", "COL": "colorado-rockies",
        "LAD": "los-angeles-dodgers", "SDP": "san-diego-padres",
        "SFG": "san-francisco-giants", "CHC": "chicago-cubs",
        "CIN": "cincinnati-reds", "MIL": "milwaukee-brewers",
        "PIT": "pittsburgh-pirates", "STL": "st-louis-cardinals",
        "ATL": "atlanta-braves", "MIA": "miami-marlins",
        "NYM": "new-york-mets", "PHI": "philadelphia-phillies",
        "WSN": "washington-nationals", "BAL": "baltimore-orioles",
        "BOS": "boston-red-sox", "NYY": "new-york-yankees",
        "TBR": "tampa-bay-rays", "TOR": "toronto-blue-jays",
        "CHW": "chicago-white-sox", "CLE": "cleveland-guardians",
        "DET": "detroit-tigers", "KCR": "kansas-city-royals",
        "MIN": "minnesota-twins", "OAK": "athletics",
        "HOU": "houston-astros", "LAA": "los-angeles-angels",
        "SEA": "seattle-mariners", "TEX": "texas-rangers",
    }

    slug = TEAM_SLUG_MAP.get(team.upper(), team.lower())
    team_data = all_contracts.get(slug, {})
    # Support both old list format and new {players, summary} format
    if isinstance(team_data, list):
        contracts = team_data
        summary = {}
    else:
        contracts = team_data.get('players', [])
        summary = team_data.get('summary', {})

    # Get years range
    all_years = set()
    for c in contracts:
        all_years.update(int(yr) for yr in c['year_salaries'].keys())
    years = sorted(yr for yr in all_years if yr >= 2025)

    # Build player rows
    rows = []
    for c in contracts:
        ys = {int(yr): v for yr, v in c['year_salaries'].items()}
        future_yrs = {yr: v for yr, v in ys.items() if yr >= 2025}
        if not future_yrs:
            continue

        # Try to match WAR from DATA using normalized name matching
        war = None
        archetype = None
        if DATA is not None:
            import unicodedata
            def norm_name(n):
                s = str(n)
                try:
                    s = s.encode("latin1").decode("utf8")
                except:
                    pass
                return unicodedata.normalize("NFC", s).strip().lower()

            search_name = norm_name(c['name'])
            data_names = DATA['name'].apply(norm_name)
            # Try exact match first, then first+last name
            exact = DATA[(data_names == search_name) & (DATA['season'] == 2025)]
            if exact.empty:
                # Try matching first and last name separately
                parts = search_name.split()
                if len(parts) >= 2:
                    # Require last name to match exactly as a word
                    mask = data_names.apply(lambda n: parts[-1] == n.split()[-1] if n.split() else False)
                    exact = DATA[mask & (DATA['season'] == 2025)]
            if not exact.empty:
                war = round(float(exact.iloc[0].get('WAR', 0)), 1)
                try:
                    from models.clustering import classify_player
                    arch = classify_player(exact, ARCH_BUNDLE)
                    archetype = arch.get('archetype_label', None)
                except:
                    pass

        rows.append({
            'name': c['name'],
            'pos': c['pos'].upper(),
            'years': c['years'],
            'total_value': c['total_value'],
            'start_year': c['start_year'],
            'end_year': c['end_year'],
            'year_salaries': future_yrs,
            'war_2025': war,
            'archetype': archetype,
        })

    # Year totals
    year_totals = {}
    for yr in years:
        year_totals[yr] = round(sum(
            r['year_salaries'].get(yr, 0) for r in rows
        ), 1)

    # Luxury tax threshold (2026 estimate)
    LUX_TAX = 241.0

    # Convert summary keys to int years
    summary_clean = {}
    for key, yr_dict in summary.items():
        summary_clean[key] = {int(yr): v for yr, v in yr_dict.items()}

    return jsonify({
        'team': team.upper(),
        'years': years,
        'contracts': rows,
        'year_totals': year_totals,
        'summary': summary_clean,
        'luxury_tax_threshold': LUX_TAX,
    })


@app.route("/api/trade_contracts")
def get_trade_contracts():
    """Return contract data for a list of players and their teams."""
    import json, os, unicodedata
    names = request.args.getlist('names')
    teams = request.args.getlist('teams')

    contracts_path = os.path.join(os.path.dirname(__file__), 'data', 'contracts_clean.json')
    try:
        with open(contracts_path, 'r') as f:
            all_contracts = json.load(f)
    except:
        return jsonify({})

    TEAM_SLUG_MAP = {
        "ARI":"arizona-diamondbacks","COL":"colorado-rockies","LAD":"los-angeles-dodgers",
        "SDP":"san-diego-padres","SFG":"san-francisco-giants","CHC":"chicago-cubs",
        "CIN":"cincinnati-reds","MIL":"milwaukee-brewers","PIT":"pittsburgh-pirates",
        "STL":"st-louis-cardinals","ATL":"atlanta-braves","MIA":"miami-marlins",
        "NYM":"new-york-mets","PHI":"philadelphia-phillies","WSN":"washington-nationals",
        "BAL":"baltimore-orioles","BOS":"boston-red-sox","NYY":"new-york-yankees",
        "TBR":"tampa-bay-rays","TOR":"toronto-blue-jays","CHW":"chicago-white-sox",
        "CLE":"cleveland-guardians","DET":"detroit-tigers","KCR":"kansas-city-royals",
        "MIN":"minnesota-twins","OAK":"athletics","HOU":"houston-astros",
        "LAA":"los-angeles-angels","SEA":"seattle-mariners","TEX":"texas-rangers",
    }

    def norm(n):
        s = str(n)
        try: s = s.encode('latin1').decode('utf8')
        except: pass
        return unicodedata.normalize('NFC', s).strip().lower()

    results = {}
    for name, team in zip(names, teams):
        slug = TEAM_SLUG_MAP.get(team.upper(), '')
        team_data = all_contracts.get(slug, {})
        players = team_data.get('players', []) if isinstance(team_data, dict) else team_data
        summary = team_data.get('summary', {}) if isinstance(team_data, dict) else {}

        search = norm(name)
        contract = None
        for p in players:
            if norm(p['name']) == search:
                contract = p
                break
        if not contract:
            parts = search.split()
            if len(parts) >= 2:
                for p in players:
                    pn = norm(p['name'])
                    if parts[-1] in pn and parts[0] in pn:
                        contract = p
                        break

        team_payroll_2026 = 0
        pay_opt = summary.get('payroll_options', {})
        if pay_opt:
            yr_data = {int(k): v for k, v in pay_opt.items()}
            team_payroll_2026 = yr_data.get(2026, 0)

        results[name] = {
            'contract': contract,
            'team': team.upper(),
            'team_payroll_2026': team_payroll_2026,
        }

    return jsonify(results)


def compute_team_stats(bbref_team):
    """Compute summary stats for one team from cached roster."""
    import json, os
    roster = live_roster_cache.get(bbref_team)
    if not roster:
        return None

    hitters = [h for h in roster.get('hitters', []) if isinstance(h.get('latest_WAR'), (int, float))]
    pitchers = [p for p in roster.get('pitchers', []) if isinstance(p.get('latest_WAR'), (int, float))]

    # Use all positive WAR players — over 162 games everyone contributes
    def get_war(p):
        w = p.get('proj_war') if isinstance(p.get('proj_war'), (int,float)) else p.get('latest_WAR', 0)
        return w if isinstance(w, (int,float)) else 0

    hitters = [h for h in hitters if get_war(h) > 0]
    pitchers = [p for p in pitchers if get_war(p) > 0]

    hitter_war = round(sum(get_war(h) for h in hitters), 1)
    pitcher_war = round(sum(get_war(p) for p in pitchers), 1)
    total_war = round(hitter_war + pitcher_war, 1)

    # Win projection: calibrated scaling
    proj_wins = round(46 + (total_war * 0.77), 1)
    proj_wins = max(55, min(108, proj_wins))

    # Hitting stats
    wrc_vals = [h.get('wRC_plus', 100) for h in hitters if isinstance(h.get('wRC_plus'), (int,float))]
    avg_wrc = round(sum(wrc_vals) / len(wrc_vals), 0) if wrc_vals else 100

    # Pitching stats
    era_vals = [p.get('ERA') for p in pitchers if isinstance(p.get('ERA'), (int,float)) and p.get('ERA', 99) < 9]
    fip_vals = [p.get('FIP') for p in pitchers if isinstance(p.get('FIP'), (int,float)) and p.get('FIP', 99) < 9]
    avg_era = round(sum(era_vals) / len(era_vals), 2) if era_vals else 4.50
    avg_fip = round(sum(fip_vals) / len(fip_vals), 2) if fip_vals else 4.50

    # Payroll from contracts
    contracts_path = os.path.join(os.path.dirname(__file__), 'data', 'contracts_clean.json')
    payroll_2026 = 0
    try:
        with open(contracts_path, 'r') as f:
            contracts = json.load(f)
        team_data = contracts.get(BBREF_TO_COTS.get(bbref_team, ''), {})
        players = team_data.get('players', []) if isinstance(team_data, dict) else []
        summary = team_data.get('summary', {}) if isinstance(team_data, dict) else {}

        # Sum individual player 2026 salaries (most reliable source)
        payroll_2026 = round(sum(
            p.get('year_salaries', {}).get(2026, p.get('year_salaries', {}).get('2026', 0))
            for p in players
        ), 1)
    except:
        pass

    # $/WAR efficiency
    dollar_per_war = round(payroll_2026 / max(total_war, 0.5), 1) if payroll_2026 else None

    return {
        'team': bbref_team,
        'total_war': total_war,
        'hitter_war': round(hitter_war, 1),
        'pitcher_war': round(pitcher_war, 1),
        'proj_wins': proj_wins,
        'avg_wrc': int(avg_wrc),
        'avg_era': avg_era,
        'avg_fip': avg_fip,
        'payroll': round(payroll_2026, 1),
        'dollar_per_war': dollar_per_war,
        'roster_size': len(hitters) + len(pitchers),
    }

# Mapping from BBRef abbrev to Cots slug
BBREF_TO_COTS = {
    "ARI":"arizona-diamondbacks","COL":"colorado-rockies","LAD":"los-angeles-dodgers",
    "SDP":"san-diego-padres","SFG":"san-francisco-giants","CHC":"chicago-cubs",
    "CIN":"cincinnati-reds","MIL":"milwaukee-brewers","PIT":"pittsburgh-pirates",
    "STL":"st-louis-cardinals","ATL":"atlanta-braves","MIA":"miami-marlins",
    "NYM":"new-york-mets","PHI":"philadelphia-phillies","WSN":"washington-nationals",
    "BAL":"baltimore-orioles","BOS":"boston-red-sox","NYY":"new-york-yankees",
    "TBR":"tampa-bay-rays","TOR":"toronto-blue-jays","CHW":"chicago-white-sox",
    "CLE":"cleveland-guardians","DET":"detroit-tigers","KCR":"kansas-city-royals",
    "MIN":"minnesota-twins","OAK":"athletics","HOU":"houston-astros",
    "LAA":"los-angeles-angels","SEA":"seattle-mariners","TEX":"texas-rangers",
}


@app.route("/api/team_needs/<team>")
def get_team_needs(team):
    """Return positional needs for a team based on roster proj WAR vs league averages."""
    import json, os

    # League average proj WAR by position group (based on our model data)
    LEAGUE_AVG = {
        'C':  1.8, '1B': 1.5, '2B': 2.0, '3B': 1.8,
        'SS': 2.5, 'OF': 2.0,
        'SP': 2.2, 'RP': 0.8
    }

    POS_GROUP = {
        'C':'C', '1B':'1B', '2B':'2B', '3B':'3B', 'SS':'SS',
        'LF':'OF', 'CF':'OF', 'RF':'OF', 'DH':'OF',
        'SP':'SP', 'RP':'RP', 'CL':'RP', 'P':'RP'
    }

    # Load proj cache
    proj_path = os.path.join(os.path.dirname(__file__), 'data', f'proj_cache_{team}.json')
    proj_map = {}
    if os.path.exists(proj_path):
        with open(proj_path) as f:
            proj_map = json.load(f)

    # Get live roster
    if team not in live_roster_cache:
        return jsonify({"error": "Roster not loaded yet"}), 503

    roster = live_roster_cache[team]
    hitters = roster.get('hitters', [])
    pitchers = roster.get('pitchers', [])

    # Build position -> list of proj WAR
    pos_war = {pos: [] for pos in LEAGUE_AVG}

    for p in hitters:
        raw_pos = p.get('position', '')
        grp = POS_GROUP.get(raw_pos)
        if grp:
            war = proj_map.get(p['name'], p.get('proj_war'))
            try:
                war = float(war) if war not in (None, 'N/A') else None
            except:
                war = None
            # Unknown players (prospects/international) count as replacement level
            # so they don't trigger false HOLE flags
            if war is None:
                war = LEAGUE_AVG.get(grp, 1.5) * 0.5  # half league avg = replacement
            pos_war[grp].append(war)

    for p in pitchers:
        role = p.get('role', 'RP')
        grp = 'SP' if role in ('SP', 'Starter') else 'RP'
        war = proj_map.get(p['name'], p.get('proj_war'))
        try:
            war = float(war) if war not in (None, 'N/A') else None
        except:
            war = None
        if war is None:
            war = LEAGUE_AVG.get(grp, 0.8) * 0.5
        pos_war[grp].append(war)

    # Calculate needs
    needs = []
    for pos, wars in pos_war.items():
        avg = LEAGUE_AVG[pos]
        if not wars:
            best = 0.0
        else:
            # Use top 1 for C/SP (single player), top 3 for OF, top 5 for SP staff
            if pos == 'SP':
                best = sum(sorted(wars, reverse=True)[:5]) / 5 if wars else 0.0
            elif pos == 'OF':
                best = sum(sorted(wars, reverse=True)[:3]) / 3 if wars else 0.0
            elif pos == 'RP':
                best = sum(sorted(wars, reverse=True)[:3]) / 3 if wars else 0.0
            else:
                best = max(wars)

        gap = avg - best
        has_hole = best < 0.5  # no qualified player

        if gap > 0:
            if has_hole:
                severity = 'HOLE'
            elif gap >= 1.2:
                severity = 'CRITICAL'
            elif gap >= 0.7:
                severity = 'HIGH'
            elif gap >= 0.3:
                severity = 'MODERATE'
            else:
                severity = 'MINOR'
            needs.append({
                'position': pos,
                'team_war': round(best, 1),
                'league_avg': avg,
                'gap': round(gap, 1),
                'severity': severity,
                'has_hole': has_hole
            })

    # Sort by severity then gap
    needs.sort(key=lambda x: (0 if x['severity']=='HIGH' else 1, -x['gap']))

    return jsonify({
        'team': team,
        'needs': needs[:5]  # top 5 needs
    })


@app.route("/api/historic_standings/<int:year>")
def get_historic_standings(year):
    """Get actual standings + WAR data for a given season."""
    import urllib.request, ssl, json as _json
    
    ctx = ssl.create_default_context()
    
    # Get standings from MLB API
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={year}&standingsTypes=regularSeason"
    try:
        with urllib.request.urlopen(url, timeout=10, context=ctx) as r:
            standings_data = _json.loads(r.read())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    # WS winners by year
    WS_WINNERS = {
        2000:'NYY',2001:'ARI',2002:'LAA',2003:'MIA',2004:'BOS',
        2005:'CHW',2006:'STL',2007:'BOS',2008:'PHI',2009:'NYY',
        2010:'SFG',2011:'STL',2012:'SFG',2013:'BOS',2014:'SFG',
        2015:'KCR',2016:'CHC',2017:'HOU',2018:'BOS',2019:'WSN',
        2020:'LAD',2021:'ATL',2022:'HOU',2023:'TEX',2024:'LAD',2025:'LAD'
    }
    
    # Reverse map: MLB API team ID → our abbreviation
    TEAM_MAP = {v: k for k, v in BBREF_TO_MLB_ID.items()}

    # Franchise aliases - map old CSV codes to current codes for WAR lookup
    FRANCHISE_ALIASES = {
        'ANA': 'LAA',  # Anaheim Angels → LA Angels
        'FLA': 'MIA',  # Florida Marlins → Miami Marlins
        'MON': 'WSN',  # Montreal Expos → Washington Nationals
        'TBD': 'TBR',  # Tampa Bay Devil Rays → Rays
        'CAL': 'LAA',  # California Angels
    }

    def get_war(abbr, war_dict):
        """Get WAR for a team, checking aliases if not found."""
        if abbr in war_dict:
            return war_dict[abbr]
        # Check if any alias maps to this abbr
        for old_code, new_code in FRANCHISE_ALIASES.items():
            if new_code == abbr and old_code in war_dict:
                return war_dict[old_code]
        return 0

    # Get WAR data from our CSV — deduplicate first (some seasons have duplicate rows)
    hitters_yr = DATA[DATA['season']==year].drop_duplicates(subset=['name','team','season'])
    hitter_war = hitters_yr.groupby('team')['WAR'].sum().to_dict()
    if PITCHER_DATA is not None:
        pitchers_yr = PITCHER_DATA[PITCHER_DATA['season']==year].drop_duplicates(subset=['name','team','season'])
        pitcher_war = pitchers_yr.groupby('team')['WAR'].sum().to_dict()
    else:
        pitcher_war = {}

    teams = []
    for division in standings_data.get('records', []):
        div_name = division.get('division', {}).get('name', '')
        for t in division.get('teamRecords', []):
            name = t['team']['name']
            abbr = TEAM_MAP.get(t['team']['id'], '')
            wins = t.get('wins', 0)
            losses = t.get('losses', 0)
            gb = t.get('gamesBack', '-')
            playoff = t.get('clinched', False) or t.get('divisionChamp', False)
            
            h_war = round(get_war(abbr, hitter_war), 1)
            p_war = round(get_war(abbr, pitcher_war), 1)
            total_war = round(h_war + p_war, 1)
            
            teams.append({
                'team': abbr,
                'name': name,
                'division': div_name,
                'wins': wins,
                'losses': losses,
                'win_pct': round(wins/(wins+losses), 3) if (wins+losses) > 0 else 0,
                'games_back': gb,
                'hit_war': h_war,
                'pitch_war': p_war,
                'total_war': total_war,
                'ws_winner': WS_WINNERS.get(year) == abbr,
                'made_playoffs': playoff,
            })
    
    # Sort by wins
    teams.sort(key=lambda x: x['wins'], reverse=True)
    return jsonify({'year': year, 'teams': teams})

@app.route("/api/team_stats")
def get_team_stats():
    """Return league-wide team summary stats."""
    # Recompute any teams that have proj_war now available
    for bbref_team in BBREF_TO_MLB_ID.keys():
        if bbref_team in live_roster_cache:
            stats = compute_team_stats(bbref_team)
            if stats:
                team_stats_cache[bbref_team] = stats

    teams = sorted(team_stats_cache.values(), key=lambda x: x['total_war'], reverse=True)
    return jsonify({'teams': teams, 'count': len(teams)})



@app.route("/api/historic_roster/<team>/<int:season>")
def get_historic_roster(team, season):
    """Return full historical roster for a team/season from our dataset."""
    import unicodedata
    from models.clustering import classify_player
    from models.pitcher_clustering import classify_pitcher

    if DATA is None or PITCHER_DATA is None:
        return jsonify({"error": "Data not loaded"}), 503

    # Handle legacy team abbreviations
    TEAM_ALIASES = {
        'ANA': 'LAA', 'FLA': 'MIA', 'MON': 'WSN',
        'TBD': 'TBR', 'ATH': 'OAK',
    }
    search_team = TEAM_ALIASES.get(team.upper(), team.upper())
    # Also search legacy names if looking for current team
    REVERSE_ALIASES = {v: k for k, v in TEAM_ALIASES.items()}
    legacy_team = REVERSE_ALIASES.get(search_team)

    def norm_name(n):
        s = str(n)
        try: s = s.encode('latin1').decode('utf8')
        except: pass
        return unicodedata.normalize('NFC', s).strip()

    # Filter hitters
    mask = DATA['season'] == season
    if legacy_team:
        mask &= DATA['team'].isin([search_team, legacy_team])
    else:
        mask &= DATA['team'] == search_team
    team_hitters = DATA[mask].copy()
    # Deduplicate — keep highest WAR row per player (handles 2TM/3TM splits)
    team_hitters = team_hitters.sort_values('WAR', ascending=False).drop_duplicates('name', keep='first')

    # Filter pitchers
    pmask = PITCHER_DATA['season'] == season
    if legacy_team:
        pmask &= PITCHER_DATA['team'].isin([search_team, legacy_team])
    else:
        pmask &= PITCHER_DATA['team'] == search_team
    team_pitchers = PITCHER_DATA[pmask].copy()
    # Deduplicate pitchers too
    team_pitchers = team_pitchers.sort_values('WAR', ascending=False).drop_duplicates('name', keep='first')

    if team_hitters.empty and team_pitchers.empty:
        # Try legacy alias
        alt = REVERSE_ALIASES.get(team.upper(), TEAM_ALIASES.get(team.upper()))
        return jsonify({"error": f"No data for {team} in {season}", "try_alias": alt}), 404

    hitters = []
    for _, row in team_hitters.iterrows():
        name = norm_name(row['name'])
        try:
            history = DATA[DATA['name'] == row['name']]
            season_data = history[history['season'] <= season]
            if season_data.empty:
                season_data = history
            seasons_300pa = int((pd.to_numeric(season_data['PA'], errors='coerce').fillna(0) >= 300).sum())
            arch = classify_player(season_data, ARCH_BUNDLE, seasons_300pa_override=seasons_300pa)
            archetype = arch.get('archetype_label', 'Unknown')
        except:
            archetype = 'Unknown'
        
        hitters.append({
            'name': name,
            'position': str(row.get('position', '?')),
            'age': int(row.get('age', 0)),
            'WAR': round(float(row.get('WAR', 0)), 1),
            'wRC_plus': int(row.get('wRC_plus', 100)) if pd.notna(row.get('wRC_plus')) else 100,
            'AVG': round(float(row.get('AVG', 0)), 3) if pd.notna(row.get('AVG')) else None,
            'OBP': round(float(row.get('OBP', 0)), 3) if pd.notna(row.get('OBP')) else None,
            'SLG': round(float(row.get('SLG', 0)), 3) if pd.notna(row.get('SLG')) else None,
            'HR': int(row.get('HR', 0)) if pd.notna(row.get('HR')) else 0,
            'archetype': archetype,
        })

    pitchers = []
    for _, row in team_pitchers.iterrows():
        name = norm_name(row['name'])
        try:
            history = PITCHER_DATA[PITCHER_DATA['name'] == row['name']]
            season_data = history[history['season'] <= season]
            if season_data.empty:
                season_data = history
            arch = classify_pitcher(season_data, PITCHER_BUNDLE)
            archetype = arch.get('archetype_label', 'Unknown')
        except:
            archetype = 'Unknown'

        pitchers.append({
            'name': name,
            'role': str(row.get('role', 'SP')),
            'age': int(row.get('age', 0)),
            'WAR': round(float(row.get('WAR', 0)), 1),
            'ERA': round(float(row.get('ERA', 4.5)), 2) if pd.notna(row.get('ERA')) else None,
            'FIP': round(float(row.get('FIP', 4.5)), 2) if pd.notna(row.get('FIP')) else None,
            'IP': round(float(row.get('IP_y', 0)), 1) if pd.notna(row.get('IP_y')) else 0,
            'SO': int(row.get('SO', 0)) if pd.notna(row.get('SO')) else 0,
            'archetype': archetype,
        })

    hitters.sort(key=lambda x: x['WAR'], reverse=True)
    pitchers.sort(key=lambda x: x['WAR'], reverse=True)

    return jsonify({
        'team': team.upper(),
        'season': season,
        'hitters': hitters,
        'pitchers': pitchers,
        'total_war': round(sum(h['WAR'] for h in hitters) + sum(p['WAR'] for p in pitchers), 1),
    })

# Load pipeline synchronously at module level
load_pipeline()

if __name__ == "__main__":
    print("\nDashboard running at: http://localhost:5000\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)