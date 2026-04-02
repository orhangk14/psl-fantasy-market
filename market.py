import numpy as np
import pandas as pd

# ============================================================
# 1. INPUT DATA
#    Add completed matches here.
#    Leave future matches as np.nan.
# ============================================================

players = [
    "OGK", "Zaafir", "Subhi", "Usman", "Haris", "Rumail", "Shahzaib",
    "Rayaan", "Zayaan", "Umran", "Subhani", "Ibaad", "Gondy",
    "Sameer A", "Affan", "Khizer B", "Reza", "Rafay"
]

match_names = [
    "Match 1", "Match 2", "Match 3", "Match 4", "Match 5", "Match 6",
    "Match 7", "Match 8", "Match 9", "Match 10", "Match 11", "Match 12",
    "Match 13", "Match 14", "Match 15", "Match 16", "Match 17", "Match 18",
    "Match 19", "Match 20", "Match 21", "Match 22", "Match 23", "Match 24",
    "Match 25", "Match 26", "Match 27", "Match 28", "Match 29", "Match 30",
    "Match 31", "Match 32", "Match 33", "Match 34", "Match 35", "Match 36",
    "Match 37", "Match 38", "Match 39", "Match 40",
    "Qualifier", "Eliminator 1", "Eliminator 2", "Final"
]

scores = pd.DataFrame(index=match_names, columns=players, dtype=float)

scores.loc["Match 1"] = [691.5, 771.5, 535.0, 939.0, 786.0, 769.5, 605.0, 407.0, 407.0, 757.5, 862.0, 483.0, 802.5, 437.0, 804.5, 621.0, 407.0, 653.0]
scores.loc["Match 2"] = [675.0, 903.0, 422.0, 476.0, 767.0, 452.0, 422.0, 581.5, 633.0, 973.0, 422.0, 598.0, 806.0, 622.0, 422.0, 607.0, 422.0, 784.0]
scores.loc["Match 3"] = [930.0, 915.0, 986.0, 849.0, 588.0, 698.0, 474.0, 787.0, 722.0, 833.0, 428.0, 474.0, 885.0, 458.0, 495.0, 428.0, 879.0, 597.0]
scores.loc["Match 4"] = [641.0, 739.0, 698.5, 762.0, 667.5, 801.0, 532.5, 532.5, 789.5, 760.0, 846.5, 532.5, 708.5, 532.5, 880.0, 562.5, 532.5, 753.5]
scores.loc["Match 5"] = [561.0, 739.0, 618.0, 737.0, 591.0, 454.0, 454.0, 633.0, 454.0, 794.0, 660.0, 778.0, 728.0, 781.0, 813.0, 615.0, 484.0, 613.0]
scores.loc["Match 6"] = [815.0, 688.0, 420.0, 657.0, 665.0, 420.0, 420.0, 469.0, 622.5, 777.0, 614.0, 603.0, 654.5, 777.0, 450.0, 478.0, 566.0, 689.0]
scores.loc["Match 7"] = [0.0] * 18  # washed out
scores.loc["Match 8"] = [967.5, 881, 556, 912.5, 1046, 586, 970, 747.5, 753.5, 933.5, 885, 556, 862, 556, 914, 919, 973, 893]

# ============================================================
# 2. SETTINGS
# ============================================================

N_SIMS = 100_000
PLAYOFF_MULTIPLIER = 2

ABILITY_UNCERTAINTY_MULT = 1.35
RESIDUAL_VOL_INFLATION = 1.15
MATCH_EFFECT_VOL_INFLATION = 1.20
T_DF = 5
DRIFT_FRACTION = 0.15   # per-match ability drift as fraction of residual SD

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
#    - if all scores in a completed match are equal, exclude the
#      entire match from model fitting (washout / no information)
#    - but it still counts in official standings and time decay
#    - otherwise, bottom score(s) are treated as DNP for fitting
# ============================================================

observed_mask = pd.DataFrame(True, index=completed_scores.index, columns=completed_scores.columns)
fit_match_mask = pd.Series(True, index=completed_scores.index)

for match in completed_scores.index:
    row = completed_scores.loc[match].astype(float)

    # All equal => no cross-sectional information
    if row.nunique() == 1:
        fit_match_mask.loc[match] = False
        observed_mask.loc[match, :] = False
        continue

    min_score = row.min()
    dnp_players = row[row == min_score].index.tolist()
    observed_mask.loc[match, dnp_players] = False

fit_scores = completed_scores.loc[fit_match_mask].copy()
fit_observed_mask = observed_mask.loc[fit_match_mask].copy()
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
RECENCY_HALF_LIFE = 4.0
n_fit_matches = len(fit_scores)
match_indices = np.arange(n_fit_matches, dtype=float)
recency_weights = np.exp(np.log(2) / RECENCY_HALF_LIFE * (match_indices - (n_fit_matches - 1)))
recency_weight_map = dict(zip(fit_scores.index, recency_weights))

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

# ============================================================
# 8. SIMULATION (RANDOM WALK ABILITY)
# ============================================================

def simulate_one_season(params_df, rng, future_match_names, future_is_playoff):
    base_mean = params_df["estimated_mean"].values
    resid_sd = params_df["resid_sd"].values
    obs_count = np.maximum(params_df["observed_count"].values, 2)
    current = params_df["current_total"].values

    ability_sd = ABILITY_UNCERTAINTY_MULT * resid_sd / np.sqrt(obs_count)
    drift_sd = DRIFT_FRACTION * resid_sd

    # Draw initial ability level (where we think they are NOW)
    current_mean = rng.normal(base_mean, ability_sd)

    running = current.copy()

    for t, match_name in enumerate(future_match_names):
        # Ability drifts each match after the first
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
    "estimated_mean", "resid_sd"
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
    book_probs = prob_array * (1.0 + vig)
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
fitted_history.to_csv("fitted_history_with_imputations.csv")