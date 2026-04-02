#!/usr/bin/env python3
"""
drift_calibration.py
Uses last season's data to estimate ability drift.
Run: python drift_calibration.py
"""

import numpy as np
import pandas as pd

players = [
    "Zaki", "Zaafir", "Usman", "OGK", "Subhi", "Umran", "Haider",
    "Zayaan", "Sameer", "WICKGIF", "Rumail", "Khizar", "Ibrahim G",
    "Owais", "Raheem", "Subhani", "Zayd"
]

raw_scores = {
    "Match 1":  [358,355.5,378.5,599,530.5,402.5,508,241.5,385,571,625.5,271.5,241.5,241.5,241.5,241.5,241.5],
    "Match 2":  [631,683.5,627,416,612,498.5,513.5,386,459,479,554.5,422.5,386,609,386,386,386],
    "Match 3":  [735,607,301,682,816,628.5,780.5,331,601,924,811,727.5,301,414,301,301,301],
    "Match 4":  [650.5,652,454,551,644,541,454,656,523.5,547,727.5,571,583,659,454,454,484],
    "Match 5":  [728.5,638,550.5,596,537.5,823.5,488,518,521,667.5,701,698,645.5,488,625,604.5,504.5],
    "Match 6":  [666,583.5,408,598.5,614.5,646,408,602,572.5,543.5,786.5,769.5,590.5,590.5,438,654.5,761],
    "Match 7":  [739,723,733.5,769.5,594.5,771,557,648,587,787.5,727,744.5,718.5,627.5,557,700,770],
    "Match 8":  [630,582.5,620,655.5,689,489.5,459.5,687.5,452,836,615.5,452,452,767.5,452,482,670.5],
    "Match 9":  [424,481.5,456,399.5,515,526,361.5,477.5,520,500.5,482,361.5,559.5,361.5,361.5,666.5,391.5],
    "Match 10": [475,487.5,511,456.5,541,507.5,320,320,659.5,532,350,320,562,320,320,320,320],
    "Match 11": [540,461,421.5,430.5,525.5,448,367.5,484.5,407.5,586.5,543.5,534.5,367.5,574,367.5,465.5,490],
    "Match 12": [640,713,654.5,665,760.5,566,536,536,536,608.5,631,603.5,739,696,536,536,660],
    "Match 14": [331,342,544.5,361,445.5,424.5,301,301,362,710,422,450.5,301,447,301,451.5,301],
    "Match 15": [448,649.5,709,626.5,517,621.5,448,730,534,672.5,478,448,571,509,448,448,448],
    "Match 16": [418.5,634,644,612.5,513.5,523,np.nan,418.5,448.5,549.5,637.5,594.5,581.5,573.5,418.5,513,418.5],
    "Match 17": [557.5,558,608,583.5,704.5,640.5,np.nan,548.5,631.5,722,597,538,581,566,656,623,653.5],
    "Match 18": [492.5,316.5,423,419,406.5,411,np.nan,375.5,329.5,376,345,286.5,371,286.5,435,286.5,286.5],
    "Match 19": [531.5,592,720.5,671.5,776,615.5,np.nan,725.5,516.5,632.5,548,486.5,634,486.5,np.nan,670,486.5],
    "Match 20": [640,634,686,632,449.5,479.5,np.nan,634,729.5,449.5,579.5,576,449.5,449.5,np.nan,703,449.5],
    "Match 21": [563,459,404,462,413.5,410,np.nan,421,443,406.5,538,439.5,424,374,np.nan,561.5,470.5],
    "Match 22": [439.5,497,566.5,785.5,543.5,621.5,np.nan,612.5,469.5,589.5,606,512.5,540.5,439.5,np.nan,492.5,481.5],
    "Match 23": [294,504.5,393,357.5,490.5,345,np.nan,468,594,314,559.5,556.5,438.5,294,np.nan,492,407],
    "Match 24": [384.5,499.5,249.5,326,400,537.5,np.nan,293,387.5,388.5,319,266,436,219.5,np.nan,349,219.5],
    "Match 26": [310.5,481,457,567.5,512.5,565,np.nan,587,531.5,535,381,458.5,340.5,310.5,np.nan,310.5,310.5],
    "Match 27": [805.5,669.5,849.5,689,816.5,793,np.nan,664.5,625,908,835.5,901,595,595,np.nan,703.5,595],
    "Match 28": [714.5,415,602,299.5,309,412.5,np.nan,649,394,593,538.5,269.5,269.5,269.5,np.nan,269.5,269.5],
    "Match 29": [387.5,307.5,484.5,277.5,473.5,476.5,np.nan,508,431.5,401.5,518.5,479.5,277.5,277.5,np.nan,277.5,277.5],
    "Match 31": [639,655,677,527,746.5,692,np.nan,527,597,557,736.5,527,623,np.nan,np.nan,527,527],
    "Match 32": [569,589,752,704.5,678.5,571,np.nan,468,678.5,610.5,645,438,438,np.nan,np.nan,438,438],
    "Match 33": [375.5,601.5,557.5,625.5,692,555.5,np.nan,447,375.5,405.5,491,375.5,375.5,np.nan,np.nan,567,375.5],
    "Match 34": [528,578,722,677.5,776.5,528,np.nan,528,622.5,729.5,596.5,528,528,np.nan,np.nan,528,528],
}

scores = pd.DataFrame(raw_scores, index=players).T

# --- DNP detection ---

observed_mask = pd.DataFrame(True, index=scores.index, columns=players)
fit_match_mask = pd.Series(True, index=scores.index)

for match in scores.index:
    row = scores.loc[match].astype(float)
    valid = row.dropna()

    if len(valid) == 0 or valid.nunique() == 1:
        fit_match_mask.loc[match] = False
        observed_mask.loc[match, :] = False
        continue

    min_score = valid.min()
    for p in players:
        if pd.isna(row[p]):
            observed_mask.loc[match, p] = False
        elif row[p] == min_score:
            observed_mask.loc[match, p] = False

fit_scores = scores.loc[fit_match_mask].copy()
fit_observed_mask = observed_mask.loc[fit_match_mask].copy()

# --- compute slate-adjusted residuals ---

residuals = pd.DataFrame(np.nan, index=fit_scores.index, columns=players)

for match in fit_scores.index:
    obs = fit_observed_mask.loc[match]
    obs_vals = fit_scores.loc[match, obs].astype(float)
    match_mean = obs_vals.mean()
    for p in players:
        if fit_observed_mask.loc[match, p]:
            residuals.loc[match, p] = fit_scores.loc[match, p] - match_mean

# --- autocorrelation by lag ---

match_list = fit_scores.index.tolist()
n_matches = len(match_list)

print("=" * 60)
print("DRIFT CALIBRATION FROM LAST SEASON")
print("=" * 60)
print(f"Informative matches: {n_matches}")
print()

max_lag = min(15, n_matches - 2)

print(f"{'Lag':>4s}  {'Correlation':>12s}  {'Pairs':>6s}")
print("-" * 30)

lag_corrs = []

for lag in range(1, max_lag + 1):
    pair_x = []
    pair_y = []
    for i in range(n_matches - lag):
        m1 = match_list[i]
        m2 = match_list[i + lag]
        for p in players:
            r1 = residuals.loc[m1, p]
            r2 = residuals.loc[m2, p]
            if not np.isnan(r1) and not np.isnan(r2):
                pair_x.append(r1)
                pair_y.append(r2)
    
    if len(pair_x) >= 5:
        corr = np.corrcoef(pair_x, pair_y)[0, 1]
    else:
        corr = np.nan
    
    lag_corrs.append(corr)
    print(f"{lag:4d}  {corr:12.3f}  {len(pair_x):6d}")

print()

# --- estimate implied drift ---

lag1 = lag_corrs[0] if len(lag_corrs) > 0 else 0.5

# Under a random walk model with residual noise:
#   observed_corr(lag=1) = var(ability) / (var(ability) + var(residual))
#   and this decays with lag
#
# A simpler interpretation: if lag-1 autocorrelation of slate-adjusted
# residuals is rho, then the fraction of variance that is "persistent
# ability" vs "noise + drift" can be read directly.

print(f"Lag-1 autocorrelation: {lag1:.3f}")
print()

if lag1 > 0:
    # Check if correlation decays roughly linearly or exponentially
    lags_valid = [(i+1, c) for i, c in enumerate(lag_corrs) if not np.isnan(c)]
    
    if len(lags_valid) >= 5:
        mid_lag = lags_valid[len(lags_valid)//2]
        far_lag = lags_valid[-1]
        
        print(f"Mid-range (lag {mid_lag[0]}): {mid_lag[1]:.3f}")
        print(f"Far-range (lag {far_lag[0]}): {far_lag[1]:.3f}")
        print()
        
        if lag1 > 0.3 and far_lag[1] < 0.1:
            print("PATTERN: Correlation decays from moderate to near-zero.")
            print("         Consistent with drifting ability.")
            print()
        elif lag1 < 0.15:
            print("PATTERN: Even lag-1 correlation is very low.")
            print("         Ability is highly volatile match-to-match.")
            print()
        elif far_lag[1] > 0.2:
            print("PATTERN: Correlation persists even at long lags.")
            print("         Stable ability component exists.")
            print()

    # Rough drift calibration
    # If residuals have SD = sigma_r, and ability drifts with SD = sigma_d per match,
    # then lag-k autocorrelation ≈ var_a / (var_a + var_r) * decay(k)
    # 
    # Practical mapping:
    #   lag1 ~ 0.3-0.4 => moderate drift, DRIFT_FRACTION ~ 0.08-0.12
    #   lag1 ~ 0.1-0.2 => heavy drift, DRIFT_FRACTION ~ 0.15-0.25
    #   lag1 ~ 0.5+    => low drift, DRIFT_FRACTION ~ 0.03-0.06

    if lag1 >= 0.5:
        suggested = "0.03 - 0.06"
    elif lag1 >= 0.3:
        suggested = "0.08 - 0.12"
    elif lag1 >= 0.15:
        suggested = "0.12 - 0.18"
    else:
        suggested = "0.18 - 0.25"
    
    print(f"SUGGESTED DRIFT_FRACTION RANGE: {suggested}")
else:
    print("Lag-1 autocorrelation is zero or negative.")
    print("Ability signal is essentially noise. Drift is very high.")
    print("SUGGESTED DRIFT_FRACTION RANGE: 0.20 - 0.30")

print()

# --- split-half diagnostic on last year for comparison ---

mid = n_matches // 2
early_dev = residuals.iloc[:mid].mean()
late_dev = residuals.iloc[mid:].mean()
both = early_dev.notna() & late_dev.notna()

rho_split = early_dev[both].corr(late_dev[both], method="spearman")
print(f"Split-half Spearman (for comparison with this year): {rho_split:.3f}")
print("=" * 60)