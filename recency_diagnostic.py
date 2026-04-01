#!/usr/bin/env python3
"""
recency_diagnostic_v3.py
Slate-adjusted, DNP-excluded split-half diagnostic.
Run: python recency_diagnostic_v3.py
"""

import numpy as np
import pandas as pd

players = [
    "OGK", "Zaafir", "Subhi", "Usman", "Haris", "Rumail", "Shahzaib",
    "Rayaan", "Zayaan", "Umran", "Subhani", "Ibaad", "Gondy",
    "Sameer A", "Affan", "Khizer B", "Reza", "Rafay"
]

match_names = [
    "Match 1", "Match 2", "Match 3", "Match 4", "Match 5", "Match 6", "Match 7"
]

scores = pd.DataFrame(index=match_names, columns=players, dtype=float)

scores.loc["Match 1"] = [691.5, 771.5, 535.0, 939.0, 786.0, 769.5, 605.0, 407.0, 407.0, 757.5, 862.0, 483.0, 802.5, 437.0, 804.5, 621.0, 407.0, 653.0]
scores.loc["Match 2"] = [675.0, 903.0, 422.0, 476.0, 767.0, 452.0, 422.0, 581.5, 633.0, 973.0, 422.0, 598.0, 806.0, 622.0, 422.0, 607.0, 422.0, 784.0]
scores.loc["Match 3"] = [930.0, 915.0, 986.0, 849.0, 588.0, 698.0, 474.0, 787.0, 722.0, 833.0, 428.0, 474.0, 885.0, 458.0, 495.0, 428.0, 879.0, 597.0]
scores.loc["Match 4"] = [641.0, 739.0, 698.5, 762.0, 667.5, 801.0, 532.5, 532.5, 789.5, 760.0, 846.5, 532.5, 708.5, 532.5, 880.0, 562.5, 532.5, 753.5]
scores.loc["Match 5"] = [561.0, 739.0, 618.0, 737.0, 591.0, 454.0, 454.0, 633.0, 454.0, 794.0, 660.0, 778.0, 728.0, 781.0, 813.0, 615.0, 484.0, 613.0]
scores.loc["Match 6"] = [815.0, 688.0, 420.0, 657.0, 665.0, 420.0, 420.0, 469.0, 622.5, 777.0, 614.0, 603.0, 654.5, 777.0, 450.0, 478.0, 566.0, 689.0]
scores.loc["Match 7"] = [0.0] * 18

# --- DNP detection and fit mask ---

observed_mask = pd.DataFrame(True, index=scores.index, columns=scores.columns)
fit_match_mask = pd.Series(True, index=scores.index)

for match in scores.index:
    row = scores.loc[match].astype(float)
    if row.nunique() == 1:
        fit_match_mask.loc[match] = False
        observed_mask.loc[match, :] = False
        continue
    min_score = row.min()
    dnp_players = row[row == min_score].index.tolist()
    observed_mask.loc[match, dnp_players] = False

fit_scores = scores.loc[fit_match_mask].copy()
fit_observed_mask = observed_mask.loc[fit_match_mask].copy()

n_fit = len(fit_scores)

# --- slate-adjusted deviations using ONLY observed (non-DNP) scores ---

deviations = pd.DataFrame(np.nan, index=fit_scores.index, columns=players)

for match in fit_scores.index:
    obs_cols = fit_observed_mask.loc[match]
    obs_scores = fit_scores.loc[match, obs_cols]
    match_mean = obs_scores.mean()
    for p in players:
        if fit_observed_mask.loc[match, p]:
            deviations.loc[match, p] = fit_scores.loc[match, p] - match_mean

# --- split halves ---

mid = n_fit // 2
early_dev = deviations.iloc[:mid].mean()
late_dev  = deviations.iloc[mid:].mean()

both_valid = early_dev.notna() & late_dev.notna()
early_clean = early_dev[both_valid]
late_clean  = late_dev[both_valid]

rho = early_clean.corr(late_clean, method="spearman")

print("=" * 60)
print("RECENCY DIAGNOSTIC v3 (SLATE-ADJUSTED, DNP-EXCLUDED)")
print("=" * 60)
print(f"Fit matches used:        {n_fit}")
print(f"Early split (first {mid}):  {fit_scores.index[:mid].tolist()}")
print(f"Late  split (last  {n_fit - mid}):  {fit_scores.index[mid:].tolist()}")
print(f"Players with observed data in both halves: {both_valid.sum()}")
print()

# --- show DNPs per match for transparency ---

print("DNPs detected per match:")
for match in fit_scores.index:
    dnp_list = [p for p in players if not observed_mask.loc[match, p]]
    print(f"  {match}: {dnp_list}")
print()

print(f"Spearman rank correlation (slate-adjusted, DNP-excluded):  {rho:.3f}")
print()

if rho >= 0.7:
    print("VERDICT:  rho >= 0.70")
    print("          Skill rankings stable. Recency weighting unnecessary.")
elif rho >= 0.5:
    print("VERDICT:  rho between 0.50 and 0.70")
    print("          Moderate instability. Borderline.")
else:
    print("VERDICT:  rho < 0.50")
    print("          Skill rankings genuinely shifting between halves.")
    print("          Recency weighting is justified.")

print()

# --- per-player table ---

comparison = pd.DataFrame({
    "player": early_clean.index,
    "early_obs": [deviations.iloc[:mid][p].notna().sum() for p in early_clean.index],
    "late_obs":  [deviations.iloc[mid:][p].notna().sum() for p in early_clean.index],
    "early_dev": early_clean.values,
    "late_dev": late_clean.values,
    "shift": (late_clean - early_clean).values
}).sort_values("shift", ascending=False)

print("Per-player slate-adjusted deviations (DNPs excluded):")
print("early_obs / late_obs = number of real submissions in each half")
print("-" * 70)
print(comparison.to_string(index=False, float_format="%.1f"))
print("=" * 60)