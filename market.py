import numpy as np
import pandas as pd

# ============================================================
# 1. INPUT DATA (loaded from scores.json)
# ============================================================

import json

with open("scores.json", "r") as f:
    raw = json.load(f)

players = raw["players"]

match_names_completed = list(raw["matches"].keys())

# Full schedule of match names (completed + future)
all_match_names = [f"Match {i}" for i in range(1, 41)] + [
    "Qualifier", "Eliminator 1", "Eliminator 2", "Final"
]

scores = pd.DataFrame(index=all_match_names, columns=players, dtype=float)

for match_name, score_list in raw["matches"].items():
    scores.loc[match_name] = score_list
# ============================================================
# 2. SETTINGS
# ============================================================

N_SIMS = 100_000
PLAYOFF_MULTIPLIER = 2

ABILITY_UNCERTAINTY_MULT = 1.35
RESIDUAL_VOL_INFLATION = 1.15
MATCH_EFFECT_VOL_INFLATION = 1.20
T_DF = 5
DRIFT_FRACTION = 0.18   # per-match ability drift as fraction of residual SD
RECENCY_HALF_LIFE = 8.0
PARTICIPATION_WINDOW = 4   # look at last N completed matches

CHAMPIONS = {"OGK", "Zaafir", "Subhi", "Usman"}
CHAMPION_BONUS = 10.0

MIN_SCORE = 0.0
MAX_SCORE = 1200.0

PLAYER_SD_PRIOR_STRENGTH = 5.0

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ============================================================
# 3. DETECT COMPLETED MATCHES
#    Any row with all numeric values is treated as completed.
# ============================================================

completed_mask = scores.notna().all(axis=1)
completed_scores = scores.loc[completed_mask].copy()
future_scores = scores.loc[~completed_mask].copy()

current_totals = completed_scores.sum(axis=0)

official_table = current_totals.sort_values(ascending=False).reset_index()
official_table.columns = ["player", "current_total"]

future_match_names = future_scores.index.tolist()
future_is_playoff = np.array([
    name in ["Qualifier", "Eliminator 1", "Eliminator 2", "Final"]
    for name in future_match_names
])

# ============================================================
# 4. DNP / FITTING MASKS
#    Rules:
#    - washout: all scores equal → exclude entire match
#    - DNP rule: non-submissions get lowest_real_score - 30
#      so if gap between min and second-lowest unique is exactly 30,
#      the min scores are DNPs
#    - if no 30-point gap exists, everyone played → no DNPs
# ============================================================

observed_mask = pd.DataFrame(True, index=completed_scores.index, columns=completed_scores.columns)
fit_match_mask = pd.Series(True, index=completed_scores.index)

for match in completed_scores.index:
    row = completed_scores.loc[match].astype(float)

    # All equal => washout, no information
    if row.nunique() == 1:
        fit_match_mask.loc[match] = False
        observed_mask.loc[match, :] = False
        continue

    sorted_unique = sorted(row.dropna().unique())

    # Check for the 30-point DNP gap
    if len(sorted_unique) >= 2 and (sorted_unique[1] - sorted_unique[0]) == 30:
        min_score = sorted_unique[0]
        dnp_players = row[row == min_score].index.tolist()
        observed_mask.loc[match, dnp_players] = False
    # else: everyone played, all scores are real observations

fit_scores = completed_scores.loc[fit_match_mask].copy()
fit_observed_mask = observed_mask.loc[fit_match_mask].copy()
# ============================================================
# 4b. PARTICIPATION RATE ESTIMATION
#     For each player, estimate probability they submit a team
#     in any future match, based on recent submission history.
# ============================================================

n_completed = len(completed_scores)
window_start = max(0, n_completed - PARTICIPATION_WINDOW)
recent_matches = completed_scores.index[window_start:]

participation_rate = pd.Series(1.0, index=players)

for p in players:
    submitted = 0
    total = 0
    for m in recent_matches:
        total += 1
        if not fit_match_mask.loc[m]:
            total -= 1
            continue
        if observed_mask.loc[m, p]:
            submitted += 1
    
    if total > 0:
        raw_rate = submitted / total
        # Curve: floor at 0.20, ceiling at 0.95
        # Maps 0.0 -> 0.20, 1.0 -> 0.95 with a curve that
        # punishes missing more aggressively than it rewards attending
        floor = 0.20
        ceiling = 0.95
        participation_rate.loc[p] = floor + (ceiling - floor) * (raw_rate ** 0.7)
    else:
        participation_rate.loc[p] = 0.5
# ============================================================
# 5. ADDITIVE FIT (RECENCY-WEIGHTED + CONVERGENCE CHECK)
#    score_it = mu + player_effect_i + match_effect_t + residual_it
# ============================================================

observed_values = fit_scores.where(fit_observed_mask).stack()

if len(observed_values) == 0:
    mu = 500.0
else:
    mu = observed_values.mean()

# Recency weights: half-life of 4 matches, most recent = weight 1.0
# --- recency weights ---
# Weight each fit match by its distance from the CURRENT match count,
# not from the last fit match. This ensures players who haven't
# played recently have their old data properly decayed.

total_completed = len(completed_scores)  # includes washouts, DNP matches — everything

recency_weight_map = {}
for m in fit_scores.index:
    # position of this match in the full completed schedule (0-indexed)
    match_position = completed_scores.index.get_loc(m)
    # distance from the most recent completed match
    distance = (total_completed - 1) - match_position
    recency_weight_map[m] = np.exp(-np.log(2) / RECENCY_HALF_LIFE * distance)
player_effect = pd.Series(0.0, index=players)
match_effect = pd.Series(0.0, index=completed_scores.index)

CONVERGENCE_TOL = 1e-6

for iteration in range(250):
    prev_player_effect = player_effect.copy()

    # player effects (recency-weighted)
    for p in players:
        vals = []
        wts = []
        for m in fit_scores.index:
            if fit_observed_mask.loc[m, p]:
                vals.append(fit_scores.loc[m, p] - mu - match_effect.loc[m])
                wts.append(recency_weight_map[m])
        if len(vals) > 0:
            player_effect.loc[p] = np.average(np.array(vals), weights=np.array(wts))
        else:
            player_effect.loc[p] = 0.0

    player_effect -= player_effect.mean()

    # match effects (unweighted — each is specific to that slate)
    for m in fit_scores.index:
        vals = []
        for p in players:
            if fit_observed_mask.loc[m, p]:
                vals.append(fit_scores.loc[m, p] - mu - player_effect.loc[p])
        match_effect.loc[m] = np.mean(vals) if len(vals) > 0 else 0.0

    match_effect.loc[fit_scores.index] -= match_effect.loc[fit_scores.index].mean()

    # convergence check
    delta = np.max(np.abs(player_effect - prev_player_effect))
    if delta < CONVERGENCE_TOL:
        break

# Non-fit matches (e.g. washouts) get zero match effect
for m in completed_scores.index:
    if not fit_match_mask.loc[m]:
        match_effect.loc[m] = 0.0

# ============================================================
# 6. HISTORICAL IMPUTATION FOR FITTING ONLY
# ============================================================
#No longer being used
#fitted_history = completed_scores.copy()

#for m in completed_scores.index:
 #   for p in players:
  #      # only impute DNPs for matches used in fitting
   #     if fit_match_mask.loc[m] and (not observed_mask.loc[m, p]):
    #        fitted_history.loc[m, p] = mu + player_effect.loc[p] + match_effect.loc[m]

# ============================================================
# 7. RESIDUAL VOL ESTIMATION
# ============================================================

player_rows = []
all_residuals = []

for p in players:
    residuals = []
    observed_count = 0

    for m in fit_scores.index:
        fitted_val = mu + player_effect.loc[p] + match_effect.loc[m]
        if fit_observed_mask.loc[m, p]:
            resid = fit_scores.loc[m, p] - fitted_val
            residuals.append(resid)
            all_residuals.append(resid)
            observed_count += 1

    raw_resid_sd = float(np.std(residuals, ddof=1)) if len(residuals) >= 2 else np.nan

    player_rows.append({
        "player": p,
        "observed_count": observed_count,
        "player_effect": player_effect.loc[p],
        "raw_resid_sd": raw_resid_sd
    })

params_df = pd.DataFrame(player_rows)

pooled_resid_sd = np.std(all_residuals, ddof=1) if len(all_residuals) > 1 else 80.0
pooled_resid_sd = max(pooled_resid_sd, 40.0)

def shrink_sd(raw_sd, n, prior_sd, strength):
    if np.isnan(raw_sd):
        return prior_sd
    return (n * raw_sd + strength * prior_sd) / (n + strength)

params_df["resid_sd"] = params_df.apply(
    lambda row: max(20.0, shrink_sd(row["raw_resid_sd"], row["observed_count"], pooled_resid_sd, PLAYER_SD_PRIOR_STRENGTH)),
    axis=1
)

params_df["current_total"] = params_df["player"].map(current_totals)
params_df["champion_bonus"] = params_df["player"].apply(
    lambda p: CHAMPION_BONUS if p in CHAMPIONS else 0.0
)
params_df["estimated_mean"] = mu + params_df["player_effect"] + params_df["champion_bonus"]

fit_match_effects = match_effect.loc[fit_scores.index]
match_sd = float(np.std(fit_match_effects.values, ddof=1)) if len(fit_match_effects) >= 2 else 40.0
match_sd = max(match_sd, 20.0)
params_df["participation_rate"] = params_df["player"].map(participation_rate)
# ============================================================
# 8. SIMULATION (RANDOM WALK WITH PARTICIPATION)
# ============================================================

def simulate_one_season(params_df, rng, future_match_names, future_is_playoff):
    base_mean = params_df["estimated_mean"].values
    resid_sd = params_df["resid_sd"].values
    obs_count = np.maximum(params_df["observed_count"].values, 2)
    current = params_df["current_total"].values
    part_rate = params_df["participation_rate"].values

    ability_sd = ABILITY_UNCERTAINTY_MULT * resid_sd / np.sqrt(obs_count)
    drift_sd = DRIFT_FRACTION * resid_sd

    current_mean = rng.normal(base_mean, ability_sd)

    running = current.copy()

    for t, match_name in enumerate(future_match_names):
        if t > 0:
            current_mean += rng.normal(0, drift_sd)

        slate_effect = rng.normal(0, MATCH_EFFECT_VOL_INFLATION * match_sd)

        z = rng.standard_t(df=T_DF, size=len(players))
        z = z / np.sqrt(T_DF / (T_DF - 2))
        residual = RESIDUAL_VOL_INFLATION * resid_sd * z

        score_t = current_mean + slate_effect + residual
        score_t = np.clip(score_t, MIN_SCORE, MAX_SCORE)

        if future_is_playoff[t]:
            score_t *= PLAYOFF_MULTIPLIER

        # Participation: each player either submits or gets DNP
        plays = rng.random(size=len(players)) < part_rate
        if plays.sum() > 0:
            dnp_score = score_t[plays].min() - 30.0
            dnp_score = max(dnp_score, MIN_SCORE)
        else:
            dnp_score = MIN_SCORE

        score_t = np.where(plays, score_t, dnp_score)

        running += score_t

    return running

# ============================================================
# 9. MONTE CARLO
# ============================================================

n_players = len(players)
win_counts = np.zeros(n_players, dtype=int)
top3_counts = np.zeros(n_players, dtype=int)
top8_counts = np.zeros(n_players, dtype=int)
sum_final_scores = np.zeros(n_players)
final_scores_store = np.zeros((N_SIMS, n_players))

for sim in range(N_SIMS):
    final_scores = simulate_one_season(params_df, rng, future_match_names, future_is_playoff)
    final_scores_store[sim] = final_scores
    sum_final_scores += final_scores

    order = np.argsort(-final_scores)
    win_counts[order[0]] += 1
    top3_counts[order[:3]] += 1
    top8_counts[order[:8]] += 1

# ============================================================
# 10. RESULTS
# ============================================================

results = params_df[[
    "player", "current_total", "observed_count", "champion_bonus",
    "estimated_mean", "resid_sd", "participation_rate"
]].copy()

results["exp_final"] = sum_final_scores / N_SIMS
results["win_prob"] = win_counts / N_SIMS
results["top3_prob"] = top3_counts / N_SIMS
results["top8_prob"] = top8_counts / N_SIMS
results["p10_final"] = np.percentile(final_scores_store, 10, axis=0)
results["p50_final"] = np.percentile(final_scores_store, 50, axis=0)
results["p90_final"] = np.percentile(final_scores_store, 90, axis=0)

def fair_odds(p):
    return np.where(p > 0, 1 / p, np.inf)

def add_vig(prob_array, vig=0.03):
    book_probs = prob_array + vig
    book_probs = np.clip(book_probs, 1e-9, 0.999999)
    return book_probs

results["win_odds"] = fair_odds(results["win_prob"])
results["top3_odds"] = fair_odds(results["top3_prob"])
results["top8_odds"] = fair_odds(results["top8_prob"])

results["win_prob_book"] = add_vig(results["win_prob"].values, vig=0.03)
results["top3_prob_book"] = add_vig(results["top3_prob"].values, vig=0.03)
results["top8_prob_book"] = add_vig(results["top8_prob"].values, vig=0.03)

results["win_odds_book"] = fair_odds(results["win_prob_book"])
results["top3_odds_book"] = fair_odds(results["top3_prob_book"])
results["top8_odds_book"] = fair_odds(results["top8_prob_book"])

MAX_DECIMAL_ODDS = 51.0
results["win_odds_book"] = np.minimum(results["win_odds_book"], MAX_DECIMAL_ODDS)
results["top3_odds_book"] = np.minimum(results["top3_odds_book"], MAX_DECIMAL_ODDS)
results["top8_odds_book"] = np.minimum(results["top8_odds_book"], MAX_DECIMAL_ODDS)

results = results.sort_values("win_prob", ascending=False).reset_index(drop=True)

# ============================================================
# 11. OUTPUT FILES
# ============================================================

results.to_csv("market_results.csv", index=False)
official_table.to_csv("official_current_standings.csv", index=False)