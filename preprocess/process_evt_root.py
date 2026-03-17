import argparse
import warnings

import awkward as ak
import numpy as np
import uproot

warnings.filterwarnings("ignore")


TRACK_FEATURES = [
    "trk_eta",
    "trk_phi",
    "trk_ip2d",
    "trk_ip3d",
    "trk_dz",
    "trk_dzsig",
    "trk_ip2dsig",
    "trk_ip3dsig",
    "trk_p",
    "trk_pt",
    "trk_nValid",
    "trk_nValidPixel",
    "trk_nValidStrip",
    "trk_charge",
]

EDGE_FEATURES = [
    "dca",
    "deltaR",
    "dca_sig",
    "cptopv",
    "pvtoPCA_1",
    "pvtoPCA_2",
    "dotprod_1",
    "dotprod_2",
    "pair_mom",
    "pair_invmass",
]

SCALAR_KEYS = {
    "jet_pt",
    "jet_eta",
    "truth_has_sv",
    "truth_has_b",
    "truth_has_btoc",
    "truth_has_c",
}


# Per-feature fallback values for non-finite track entries.
TRACK_DUMMY_VALUES = {
    "trk_eta": -999.0,
    "trk_phi": -999.0,
    "trk_ip2d": -999.0,
    "trk_ip3d": -999.0,
    "trk_dz": -999.0,
    "trk_dzsig": -999.0,
    "trk_ip2dsig": -999.0,
    "trk_ip3dsig": -999.0,
    "trk_p": -999.0,
    "trk_pt": -999.0,
    "trk_nValid": -1.0,
    "trk_nValidPixel": -1.0,
    "trk_nValidStrip": -1.0,
    "trk_charge": -3.0,
}


def _to_numpy_1d(jagged_item, dtype):
    """Convert a single-event awkward content to a flat numpy array."""
    arr = np.asarray(jagged_item, dtype=dtype)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def init_output_buffer():
    out = {feat: [] for feat in TRACK_FEATURES}
    out.update({name: [] for name in EDGE_FEATURES})
    out.update(
        {
            "trk_1": [],
            "trk_2": [],
            "trk_label": [],
            "trk_hadidx": [],
            "trk_flav": [],
            "edge_label": [],
            "jet_pt": [],
            "jet_eta": [],
            "truth_has_sv": [],
            "truth_has_b": [],
            "truth_has_btoc": [],
            "truth_has_c": [],
        }
    )
    return out


def to_write_dict(out):
    write_dict = {}
    for key, values in out.items():
        if key in SCALAR_KEYS:
            write_dict[key] = np.asarray(values)
        else:
            write_dict[key] = ak.Array(values)
    return write_dict


def flush_chunk(fout, tree_name, out, tree_written):
    n_events = len(out["jet_pt"])
    if n_events == 0:
        return tree_written, 0
    write_dict = to_write_dict(out)
    if not tree_written:
        fout[tree_name] = write_dict
        tree_written = True
    else:
        fout[tree_name].extend(write_dict)
    return tree_written, n_events


def create_edge_index(
    trk_1,
    trk_2,
    dca,
    delta_r,
    dca_sig,
    cptopv,
    pv_to_pca_1,
    pv_to_pca_2,
    dotprod_1,
    dotprod_2,
    pair_mom,
    pair_invmass,
    trk_hadidx,
    trk_flav,
    fully_connected=False,
):
    """
    Build event edge index, edge features, and binary edge labels.

    edge_label = 1 when both tracks share hadron index and flavour and hadron index is valid.
    """
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
        # Keep same selection used in the existing pt pipeline.
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
    parser.add_argument("-d", "--data", required=True, help="Input ROOT file")
    parser.add_argument("-o", "--output", required=True, help="Output ROOT file")
    parser.add_argument("--in-tree", default="tree", help="Input tree name")
    parser.add_argument("--out-tree", default="tree", help="Output tree name")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start event index")
    parser.add_argument("-e", "--end", type=int, default=-1, help="End event index (exclusive), -1 means all")
    parser.add_argument(
        "-ds",
        "--downsample-fraction",
        type=float,
        default=0.1,
        help="Fraction of events with <2 HF tracks to keep",
    )
    parser.add_argument(
        "--fully-connected",
        action="store_false",
        help="Use all provided candidate edges without cptopv/pvtoPCA filtering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for deterministic downsampling",
    )
    parser.add_argument(
        "--pt-branch",
        default="jet_pt",
        help="Input event-level pt branch to copy; fallback is 50.0 when missing",
    )
    parser.add_argument(
        "--eta-branch",
        default="jet_eta",
        help="Input event-level eta branch to copy; fallback is 0.0 when missing",
    )
    parser.add_argument(
        "--write-chunk-size",
        type=int,
        default=2000,
        help="Number of kept events per ROOT write chunk",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    with uproot.open(args.data) as f:
        tree = f[args.in_tree]
        num_events = tree.num_entries

        # Load required branches once; this matches existing script behavior.
        track_data = {feat: tree[feat].array() for feat in TRACK_FEATURES}
        edge_raw = {
            "trk_1": tree["trk_1"].array(),
            "trk_2": tree["trk_2"].array(),
            "deltaR": tree["deltaR"].array(),
            "dca": tree["dca"].array(),
            "dca_sig": tree["dca_sig"].array(),
            "cptopv": tree["cptopv"].array(),
            "pvtoPCA_1": tree["pvtoPCA_1"].array(),
            "pvtoPCA_2": tree["pvtoPCA_2"].array(),
            "dotprod_1": tree["dotprod_1"].array(),
            "dotprod_2": tree["dotprod_2"].array(),
            "pair_mom": tree["pair_mom"].array(),
            "pair_invmass": tree["pair_invmass"].array(),
        }

        trk_label = tree["trk_label"].array()
        trk_hadidx = tree["trk_hadidx"].array()
        trk_flav = tree["trk_flav"].array()

        pt_branch_present = args.pt_branch in tree.keys()
        eta_branch_present = args.eta_branch in tree.keys()
        pt_vals = tree[args.pt_branch].array() if pt_branch_present else None
        eta_vals = tree[args.eta_branch].array() if eta_branch_present else None

    start = max(0, args.start)
    stop = num_events if args.end < 0 else min(args.end, num_events)

    out = init_output_buffer()
    write_chunk_size = max(1, args.write_chunk_size)
    written_events = 0

    kept = 0
    skipped_no_edges = 0
    skipped_downsample = 0

    with uproot.recreate(args.output) as fout:
        tree_written = False
        for evt in range(start, stop):
            if(evt%10==0): print(evt)
            labels_evt = _to_numpy_1d(trk_label[evt], np.int64)
            hadidx_evt = _to_numpy_1d(trk_hadidx[evt], np.int64)
            flav_evt = _to_numpy_1d(trk_flav[evt], np.int64)

            n_tracks = len(labels_evt)
            if n_tracks == 0:
                continue

            hf_mask = np.isin(labels_evt, [2, 3, 4])
            if hf_mask.sum() < 2 and rng.random() > args.downsample_fraction:
                skipped_downsample += 1
                continue

            # Build/clean track features.
            clean_tracks = {}
            for feat in TRACK_FEATURES:
                vals = _to_numpy_1d(track_data[feat][evt], np.float32)
                bad = ~np.isfinite(vals)
                if bad.any():
                    vals = vals.copy()
                    vals[bad] = TRACK_DUMMY_VALUES[feat]
                clean_tracks[feat] = vals

            src, dst, edge_feat_evt, edge_label_evt = create_edge_index(
                edge_raw["trk_1"][evt],
                edge_raw["trk_2"][evt],
                edge_raw["dca"][evt],
                edge_raw["deltaR"][evt],
                edge_raw["dca_sig"][evt],
                edge_raw["cptopv"][evt],
                edge_raw["pvtoPCA_1"][evt],
                edge_raw["pvtoPCA_2"][evt],
                edge_raw["dotprod_1"][evt],
                edge_raw["dotprod_2"][evt],
                edge_raw["pair_mom"][evt],
                edge_raw["pair_invmass"][evt],
                hadidx_evt,
                flav_evt,
                fully_connected=args.fully_connected,
            )

            if len(src) == 0:
                skipped_no_edges += 1
                continue

            # Append event payload.
            for feat in TRACK_FEATURES:
                out[feat].append(clean_tracks[feat])

            out["trk_1"].append(src.astype(np.int32))
            out["trk_2"].append(dst.astype(np.int32))

            for feat in EDGE_FEATURES:
                vals = edge_feat_evt[feat].astype(np.float32)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                out[feat].append(vals)

            out["trk_label"].append(labels_evt.astype(np.int32))
            out["trk_hadidx"].append(hadidx_evt.astype(np.int32))
            out["trk_flav"].append(flav_evt.astype(np.int32))
            out["edge_label"].append(edge_label_evt.astype(np.float32))

            jet_pt = float(pt_vals[evt]) if pt_branch_present else 50.0
            jet_eta = float(eta_vals[evt]) if eta_branch_present else 0.0
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
                tree_written, n_flushed = flush_chunk(fout, args.out_tree, out, tree_written)
                written_events += n_flushed
                out = init_output_buffer()
                print(f"Flushed {n_flushed} events (total written: {written_events})")

            if kept % 1000 == 0:
                print(f"Processed {evt - start + 1} events, kept {kept}")

        tree_written, n_flushed = flush_chunk(fout, args.out_tree, out, tree_written)
        written_events += n_flushed

    print(f"Wrote {kept} events to {args.output}:{args.out_tree}")
    print(f"Skipped (downsample): {skipped_downsample}")
    print(f"Skipped (no edges):   {skipped_no_edges}")


if __name__ == "__main__":
    main()

