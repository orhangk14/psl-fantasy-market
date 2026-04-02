#!/usr/bin/env python3
"""
drift_comparison.py
Compare fixed-mean vs random-walk odds impact.
Run: python drift_comparison.py
"""

import numpy as np
import pandas as pd

# Paste your full model code here but STOP before section 8.
# Then run both simulation variants and compare.

# --- We'll simulate a simplified version to show the effect ---

np.random.seed(42)

# Fake but representative parameters for 5 players
names =    ["Leader", "Contender", "Mid", "Chaser", "Longshot"]
means =    [720,       700,         680,   650,      620]
resid_sd = [85,        90,          95,    100,      110]
current =  [5000,      4800,        4500,  4200,     3800]

N_FUTURE = 37
N_SIMS = 50_000

def run_sims(drift_frac, label):
    rng = np.random.default_rng(42)
    wins = np.zeros(5)
    
    for _ in range(N_SIMS):
        ability_draw = rng.normal(means, np.array(resid_sd) / np.sqrt(6))
        running = np.array(current, dtype=float)
        current_mean = ability_draw.copy()
        
        for t in range(N_FUTURE):
            if t > 0 and drift_frac > 0:
                current_mean += rng.normal(0, drift_frac * np.array(resid_sd))
            
            slate = rng.normal(0, 50)
            noise = rng.normal(0, np.array(resid_sd))
            score = current_mean + slate + noise
            score = np.clip(score, 0, 1200)
            running += score
        
        wins[np.argmax(running)] += 1
    
    probs = wins / N_SIMS
    print(f"\n{label}")
    print("-" * 45)
    for i, n in enumerate(names):
        odds = 1/probs[i] if probs[i] > 0 else 999
        print(f"  {n:12s}  {probs[i]:.3f}  ({odds:.1f})")

run_sims(0.00, "FIXED MEAN (no drift)")
run_sims(0.10, "RANDOM WALK (drift = 0.10)")
run_sims(0.15, "RANDOM WALK (drift = 0.15)")
run_sims(0.25, "RANDOM WALK (drift = 0.25)")