import ROOT
from collections import Counter
from collections import defaultdict
import numpy as np

infile = "ntup_ttbarhad3k.root"
Infile = ROOT.TFile(infile, 'READ')

tree = Infile.Get('tree')

score_thresh = 0.4422

cut_deltaR   = 0.1
cut_dca      = 0.007
cut_dca_sig  = 1.0
cut_cptopv   = 0.9

recovered_tp = 0
added_fp     = 0

fn_feature_mins = defaultdict(list)
# Initialize global counters
TP_model = FP_model = FN_model = TN_model = 0
TP_ivf = FP_ivf = FN_ivf = TN_ivf = 0

for i, evt in enumerate(tree):
    sig_set = set(evt.sig_ind)         # true signal tracks
    ivf_set = set(evt.SVtrk_ind)       # IVF-selected tracks
    preds = evt.preds                  # model score per track
    all_tracks = set(range(len(preds)))  # all track indices

    # Model-predicted signal tracks
    model_set = set(idx for idx, score in enumerate(preds) if (not np.isnan(score)) and score >= score_thresh)

    # IVF
    TP_ivf += len([idx for idx in ivf_set if idx in sig_set])
    FP_ivf += len([idx for idx in ivf_set if idx not in sig_set])
    FN_ivf += len([idx for idx in sig_set if idx not in ivf_set])
    TN_ivf += len([idx for idx in all_tracks if idx not in ivf_set and idx not in sig_set])

    # Model
    TP_model += len([idx for idx in model_set if idx in sig_set])
    FP_model += len([idx for idx in model_set if idx not in sig_set])
    FN_model += len([idx for idx in sig_set if idx not in model_set])
    TN_model += len([idx for idx in all_tracks if idx not in model_set and idx not in sig_set])

    fn_tracks = [idx for idx in sig_set if idx not in model_set]
    tp_tracks = [idx for idx in sig_set if idx in model_set]
    non_selected = set(idx for idx in range(len(preds)) if idx not in model_set)

    # If no FNs, skip event
    if not fn_tracks:
        continue
    
    candidate_tracks = set()

    trk_1 = evt.trk_1
    trk_2 = evt.trk_2
    deltaR = evt.deltaR
    dca = evt.dca
    dca_sig = evt.dca_sig
    cptopv = evt.cptopv
    pvtoPCA_1 = evt.pvtoPCA_1
    pvtoPCA_2 = evt.pvtoPCA_2
    dotprod_1 = evt.dotprod_1
    dotprod_2 = evt.dotprod_2
    pair_mom = evt.pair_mom
    pair_mass = evt.pair_invmass

    edge_features = {
        'deltaR': deltaR,
        'dca': dca,
        'dca_sig': dca_sig,
        'cptopv': cptopv,
        'pvtoPCA_1': pvtoPCA_1,
        'pvtoPCA_2': pvtoPCA_2,
        'dotprod_1': dotprod_1,
        'dotprod_2': dotprod_2,
        'pair_mom': pair_mom,
        'pair_mass': pair_mass,
    }

    for k in range(len(trk_1)):
        i1, i2 = trk_1[k], trk_2[k]

        # Only consider edge if it connects a TP to a non-selected track
        if i1 in tp_tracks and i2 in non_selected:
            t_tp, t_other = i1, i2
        elif i2 in tp_tracks and i1 in non_selected:
            t_tp, t_other = i2, i1
        else:
            continue

        # Apply cuts on edge features
       # if (deltaR[k] < cut_deltaR and
       #     dca[k] < cut_dca and
       #     dca_sig[k] < cut_dca_sig and
       #     cptopv[k] < cut_cptopv):
       #     candidate_tracks.add(t_other)

    # Evaluate recovered tracks
    for idx in candidate_tracks:
        if idx in sig_set:
            recovered_tp += 1
        else:
            added_fp += 1

    
    #for fn_idx in fn_tracks:
    #    matched_edges = [
    #    k for k in range(len(trk_1))
    #    if ((trk_1[k] == fn_idx and trk_2[k] in tp_tracks) or
    #        (trk_2[k] == fn_idx and trk_1[k] in tp_tracks))
    #    ]
    #    if not matched_edges:
    #        continue  # no edges for this FN track

    #    for feat_name, feat_array in edge_features.items():
    #        values = [feat_array[k] for k in matched_edges if not np.isnan(feat_array[k])]
    #        if values:
    #            fn_feature_mins[feat_name].append(min(values))

# Metric functions
def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def signal_accuracy(tp, fn):
    return recall(tp, fn)  # Equivalent to recall, renamed for clarity

def background_rejection(tn, fp):
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def f1(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

# Report IVF
print("=== IVF Stats ===")
print(f"TP: {TP_ivf}, FP: {FP_ivf}, FN: {FN_ivf}, TN: {TN_ivf}")
print(f"Precision: {precision(TP_ivf, FP_ivf):.4f}")
print(f"Recall:    {recall(TP_ivf, FN_ivf):.4f}")
print(f"F1 Score:  {f1(TP_ivf, FP_ivf, FN_ivf):.4f}")
print(f"Signal Accuracy:       {signal_accuracy(TP_ivf, FN_ivf):.4f}")
print(f"Background Rejection:  {background_rejection(TN_ivf, FP_ivf):.4f}")

# Report Model
print("\n=== Model Stats ===")
print(f"TP: {TP_model}, FP: {FP_model}, FN: {FN_model}, TN: {TN_model}")
print(f"Precision: {precision(TP_model, FP_model):.4f}")
print(f"Recall:    {recall(TP_model, FN_model):.4f}")
print(f"F1 Score:  {f1(TP_model, FP_model, FN_model):.4f}")
print(f"Signal Accuracy:       {signal_accuracy(TP_model, FN_model):.4f}")
print(f"Background Rejection:  {background_rejection(TN_model, FP_model):.4f}")

#for feat_name, mins in fn_feature_mins.items():
#    arr = np.array(mins)
#    print(f"{feat_name:15}: mean min = {np.mean(arr):.4f}, median = {np.median(arr):.4f}, min = {np.min(arr):.4f}, max = {np.max(arr):.4f}, N = {len(arr)}")
print(f"Cut thresholds:")
print(f"  deltaR   < {cut_deltaR}")
print(f"  dca      < {cut_dca}")
print(f"  dca_sig  < {cut_dca_sig}")
print(f"  cptopv   < {cut_cptopv}")
print()
print(f"Recovered TPs:   {recovered_tp}")
print(f"Added FPs:       {added_fp}")
print(f"TP/FP Ratio:     {recovered_tp / added_fp if added_fp else 'inf'}")
