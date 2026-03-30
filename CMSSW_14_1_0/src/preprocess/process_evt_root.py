#!/usr/bin/env python3
import argparse
import gc
import warnings

import awkward as ak
import numpy as np
import uproot

warnings.filterwarnings("ignore")


TRACK_FEATURES = [
    "trk_eta", "trk_phi", "trk_ip2d", "trk_ip3d", "trk_dz", "trk_dzsig",
    "trk_ip2dsig", "trk_ip3dsig", "trk_p", "trk_pt", "trk_nValid",
    "trk_nValidPixel", "trk_nValidStrip", "trk_charge",
]

EDGE_FEATURES = [
    "dca", "deltaR", "dca_sig", "cptopv", "pvtoPCA_1", "pvtoPCA_2",
    "dotprod_1", "dotprod_2", "pair_mom", "pair_invmass",
]

SCALAR_KEYS = {
    "jet_pt", "jet_eta",
    "truth_has_sv", "truth_has_b", "truth_has_btoc", "truth_has_c",
}

TRACK_DUMMY_VALUES = {
    "trk_eta": -999.0, "trk_phi": -999.0, "trk_ip2d": -999.0, "trk_ip3d": -999.0,
    "trk_dz": -999.0, "trk_dzsig": -999.0, "trk_ip2dsig": -999.0,
    "trk_ip3dsig": -999.0, "trk_p": -999.0, "trk_pt": -999.0,
    "trk_nValid": -1.0, "trk_nValidPixel": -1.0,
    "trk_nValidStrip": -1.0, "trk_charge": -3.0,
}

BRANCH_TYPES = {
    "trk_eta": "var * float32",
    "trk_phi": "var * float32",
    "trk_ip2d": "var * float32",
    "trk_ip3d": "var * float32",
    "trk_dz": "var * float32",
    "trk_dzsig": "var * float32",
    "trk_ip2dsig": "var * float32",
    "trk_ip3dsig": "var * float32",
    "trk_p": "var * float32",
    "trk_pt": "var * float32",
    "trk_nValid": "var * float32",
    "trk_nValidPixel": "var * float32",
    "trk_nValidStrip": "var * float32",
    "trk_charge": "var * float32",
    "dca": "var * float32",
    "deltaR": "var * float32",
    "dca_sig": "var * float32",
    "cptopv": "var * float32",
    "pvtoPCA_1": "var * float32",
    "pvtoPCA_2": "var * float32",
    "dotprod_1": "var * float32",
    "dotprod_2": "var * float32",
    "pair_mom": "var * float32",
    "pair_invmass": "var * float32",
    "trk_1": "var * int32",
    "trk_2": "var * int32",
    "trk_label": "var * int32",
    "trk_hadidx": "var * int32",
    "trk_flav": "var * int32",
    "edge_label": "var * float32",
    "jet_pt": "float32",
    "jet_eta": "float32",
    "truth_has_sv": "bool",
    "truth_has_b": "bool",
    "truth_has_btoc": "bool",
    "truth_has_c": "bool",
}


def _to_numpy_1d(jagged_item, dtype):
    arr = np.asarray(jagged_item, dtype=dtype)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def init_output_buffer():
    out = {feat: [] for feat in TRACK_FEATURES}
    out.update({name: [] for name in EDGE_FEATURES})
    out.update({
        "trk_1": [], "trk_2": [],
        "trk_label": [], "trk_hadidx": [], "trk_flav": [],
        "edge_label": [],
        "jet_pt": [], "jet_eta": [],
        "truth_has_sv": [], "truth_has_b": [],
        "truth_has_btoc": [], "truth_has_c": [],
    })
    return out


def to_write_dict(out):
    write_dict = {}
    for key, values in out.items():
        if key in SCALAR_KEYS:
            write_dict[key] = np.asarray(values)
        else:
            write_dict[key] = ak.from_iter(values)
    return write_dict


def flush_chunk(tree_out, out):
    n_events = len(out["jet_pt"])
    if n_events == 0:
        return 0

    write_dict = to_write_dict(out)
    tree_out.extend(write_dict)

    return n_events


def create_edge_index(
    trk_1, trk_2,
    dca, delta_r, dca_sig,
    cptopv, pv_to_pca_1, pv_to_pca_2,
    dotprod_1, dotprod_2,
    pair_mom, pair_invmass,
    trk_hadidx, trk_flav,
    fully_connected=False,
):
    trk_1_np = _to_numpy_1d(trk_1, np.int64)
    trk_2_np = _to_numpy_1d(trk_2, np.int64)

    dca_np = _to_numpy_1d(dca, np.float32)
    delta_r_np = _to_numpy_1d(delta_r, np.float32)
    dca_sig_np = _to_numpy_1d(dca_sig, np.float32)
    cptopv_np = _to_numpy_1d(cptopv, np.float32)
    pv_to_pca_1_np = _to_numpy_1d(pv_to_pca_1, np.float32)
    pv_to_pca_2_np = _to_numpy_1d(pv_to_pca_2, np.float32)
    dotprod_1_np = _to_numpy_1d(dotprod_1, np.float32)
    dotprod_2_np = _to_numpy_1d(dotprod_2, np.float32)
    pair_mom_np = _to_numpy_1d(pair_mom, np.float32)
    pair_invmass_np = _to_numpy_1d(pair_invmass, np.float32)

    if fully_connected:
        final_mask = np.ones_like(trk_1_np, dtype=bool)
    else:
        final_mask = (
            np.isfinite(cptopv_np)
            & np.isfinite(pv_to_pca_1_np)
            & np.isfinite(pv_to_pca_2_np)
            & (cptopv_np < 20)
            & (pv_to_pca_1_np < 20)
            & (pv_to_pca_2_np < 20)
        )

    src = trk_1_np[final_mask]
    dst = trk_2_np[final_mask]

    edge_features = {
        "dca": dca_np[final_mask],
        "deltaR": delta_r_np[final_mask],
        "dca_sig": dca_sig_np[final_mask],
        "cptopv": cptopv_np[final_mask],
        "pvtoPCA_1": pv_to_pca_1_np[final_mask],
        "pvtoPCA_2": pv_to_pca_2_np[final_mask],
        "dotprod_1": dotprod_1_np[final_mask],
        "dotprod_2": dotprod_2_np[final_mask],
        "pair_mom": pair_mom_np[final_mask],
        "pair_invmass": pair_invmass_np[final_mask],
    }

    edge_labels = (
        (trk_hadidx[src] == trk_hadidx[dst])
        & (trk_flav[src] == trk_flav[dst])
        & (trk_hadidx[src] >= 0)
    ).astype(np.float32)

    return src, dst, edge_features, edge_labels


def main():
    parser = argparse.ArgumentParser("Create ROOT event-graph training samples")
    parser.add_argument("-d", "--data", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--in-tree", default="tree")
    parser.add_argument("--out-tree", default="tree")
    parser.add_argument("-s", "--start", type=int, default=0)
    parser.add_argument("-e", "--end", type=int, default=-1)
    parser.add_argument("-ds", "--downsample-fraction", type=float, default=0.1)
    parser.add_argument("--fully-connected", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--pt-branch", default="jet_pt")
    parser.add_argument("--eta-branch", default="jet_eta")
    parser.add_argument("--write-chunk-size", type=int, default=100)
    parser.add_argument("--read-chunk-size", type=int, default=500)
    parser.add_argument("--compression", choices=["none", "zlib1"], default="none")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    branches_to_read = list(TRACK_FEATURES) + [
        "trk_1", "trk_2", "deltaR", "dca", "dca_sig",
        "cptopv", "pvtoPCA_1", "pvtoPCA_2",
        "dotprod_1", "dotprod_2", "pair_mom", "pair_invmass",
        "trk_label", "trk_hadidx", "trk_flav",
    ]

    with uproot.open(args.data) as f:
        tree = f[args.in_tree]
        num_events = tree.num_entries
        tree_keys = set(tree.keys())

    pt_branch_present = args.pt_branch in tree_keys
    eta_branch_present = args.eta_branch in tree_keys

    if pt_branch_present:
        branches_to_read.append(args.pt_branch)
    if eta_branch_present and args.eta_branch != args.pt_branch:
        branches_to_read.append(args.eta_branch)

    start = max(0, args.start)
    stop = num_events if args.end < 0 else min(args.end, num_events)

    read_chunk_size = max(1, args.read_chunk_size)
    write_chunk_size = max(1, args.write_chunk_size)

    compression = None if args.compression == "none" else uproot.ZLIB(1)

    print(f"Processing {args.data} -> {args.output}")

    out = init_output_buffer()
    kept = 0
    skipped_no_edges = 0
    skipped_downsample = 0

    with uproot.open(args.data) as fin, uproot.recreate(args.output, compression=compression) as fout:
        tree = fin[args.in_tree]
        tree_out = fout.mktree(args.out_tree, BRANCH_TYPES)

        for chunk_start in range(start, stop, read_chunk_size):
            chunk_stop = min(chunk_start + read_chunk_size, stop)

            chunk = tree.arrays(
                branches_to_read,
                entry_start=chunk_start,
                entry_stop=chunk_stop,
                library="ak",
            )

            n_chunk = chunk_stop - chunk_start

            for i in range(n_chunk):
                labels_evt = _to_numpy_1d(chunk["trk_label"][i], np.int64)
                hadidx_evt = _to_numpy_1d(chunk["trk_hadidx"][i], np.int64)
                flav_evt = _to_numpy_1d(chunk["trk_flav"][i], np.int64)

                if len(labels_evt) == 0:
                    continue

                hf_mask = np.isin(labels_evt, [2, 3, 4])
                if hf_mask.sum() < 2 and rng.random() > args.downsample_fraction:
                    skipped_downsample += 1
                    continue

                clean_tracks = {}
                for feat in TRACK_FEATURES:
                    vals = _to_numpy_1d(chunk[feat][i], np.float32)
                    bad = ~np.isfinite(vals)
                    if bad.any():
                        vals = vals.copy()
                        vals[bad] = TRACK_DUMMY_VALUES[feat]
                    clean_tracks[feat] = vals

                src, dst, edge_feat_evt, edge_label_evt = create_edge_index(
                    chunk["trk_1"][i], chunk["trk_2"][i],
                    chunk["dca"][i], chunk["deltaR"][i], chunk["dca_sig"][i],
                    chunk["cptopv"][i], chunk["pvtoPCA_1"][i], chunk["pvtoPCA_2"][i],
                    chunk["dotprod_1"][i], chunk["dotprod_2"][i],
                    chunk["pair_mom"][i], chunk["pair_invmass"][i],
                    hadidx_evt, flav_evt,
                    fully_connected=args.fully_connected,
                )

                if len(src) == 0:
                    skipped_no_edges += 1
                    continue

                for feat in TRACK_FEATURES:
                    out[feat].append(clean_tracks[feat].tolist())

                out["trk_1"].append(src.astype(np.int32).tolist())
                out["trk_2"].append(dst.astype(np.int32).tolist())

                for feat in EDGE_FEATURES:
                    vals = edge_feat_evt[feat].astype(np.float32)
                    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                    out[feat].append(vals.tolist())

                out["trk_label"].append(labels_evt.astype(np.int32).tolist())
                out["trk_hadidx"].append(hadidx_evt.astype(np.int32).tolist())
                out["trk_flav"].append(flav_evt.astype(np.int32).tolist())
                out["edge_label"].append(edge_label_evt.astype(np.float32).tolist())

                jet_pt = float(chunk[args.pt_branch][i]) if pt_branch_present else 50.0
                jet_eta = float(chunk[args.eta_branch][i]) if eta_branch_present else 0.0

                if not np.isfinite(jet_pt):
                    jet_pt = 50.0
                if not np.isfinite(jet_eta):
                    jet_eta = 0.0

                out["jet_pt"].append(np.float32(jet_pt))
                out["jet_eta"].append(np.float32(jet_eta))
                out["truth_has_sv"].append(np.bool_(np.isin(labels_evt, [2, 3, 4]).any()))
                out["truth_has_b"].append(np.bool_((labels_evt == 2).any()))
                out["truth_has_btoc"].append(np.bool_((labels_evt == 3).any()))
                out["truth_has_c"].append(np.bool_((labels_evt == 4).any()))

                kept += 1

                if len(out["jet_pt"]) >= write_chunk_size:
                    flush_chunk(tree_out, out)
                    out = init_output_buffer()

            del chunk
            gc.collect()

        flush_chunk(tree_out, out)

    print(f"Done. Wrote {kept} events")
    print(f"Skipped (downsample): {skipped_downsample}")
    print(f"Skipped (no edges):   {skipped_no_edges}")


if __name__ == "__main__":
    main()
