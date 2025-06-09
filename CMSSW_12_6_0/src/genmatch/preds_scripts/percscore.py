import numpy as np
import ROOT

file = ROOT.TFile("ntup_onnxfix_0906.root")
tree = file.Get("tree")

# Global containers
global_preds = []
global_sig_scores = []

for i, evt in enumerate(tree):
    preds = np.array(evt.preds)
    sig_inds = evt.sig_ind

    if len(preds) == 0 or len(sig_inds) == 0:
        continue

    global_preds.extend(preds)  # all tracks
    for sig_idx in sig_inds:
        if sig_idx >= len(preds):
            continue
        global_sig_scores.append(preds[sig_idx])  # only signal tracks

    # --- Per-event percentiles ---
    sorted_scores = np.sort(preds)
    print(f"\nEvent {i} — {len(sig_inds)} signal tracks:")
    for sig_idx in sig_inds:
        if sig_idx >= len(preds):
            continue
        score = preds[sig_idx]
        percentile = 100.0 * np.searchsorted(sorted_scores, score, side='right') / len(sorted_scores)
        print(f"  Track {sig_idx}: Score = {score:.4f} → Percentile = {percentile:.2f}%")

# --- Global percentile calculation ---
global_preds = np.array(global_preds)
global_sig_scores = np.array(global_sig_scores)

sorted_global = np.sort(global_preds)

print("\n=== Global Signal Track Percentiles ===")
percentiles = []
for score in global_sig_scores:
    percentile = 100.0 * np.searchsorted(sorted_global, score, side='right') / len(sorted_global)
    percentiles.append(percentile)

percentiles = np.array(percentiles)

print(f"Signal track count: {len(global_sig_scores)}")
print(f"Average percentile: {np.mean(percentiles):.2f}%")
print(f"Median percentile:  {np.median(percentiles):.2f}%")
print(f"Fraction of signal tracks in top 10%: {(percentiles > 90).mean():.2%}")
print(f"Fraction of signal tracks in top 5%:  {(percentiles > 95).mean():.2%}")
print(f"Fraction of signal tracks in top 1%:  {(percentiles > 99).mean():.2%}")

