import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from pathlib import Path
import os
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from utils.models.base_model import Classifier_base


class EdgeMLPConv(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super().__init__(aggr=None)
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_channels, out_channels),
        )
        self.edge_score = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        msg = self.mlp(torch.cat([x_i, x_j, edge_attr], dim=1))
        w = torch.tanh(self.edge_score(msg)).squeeze(-1)
        self._alpha = w
        return msg * (1.0 + w).unsqueeze(-1) * 0.5

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        alpha = softmax(self._alpha, index)
        return scatter_add(inputs * alpha.unsqueeze(-1), index, dim=0, dim_size=dim_size)


class GraphEventGNN(Classifier_base):
    """
    Graph model for event-level track/edge multitask training.
    Uses node labels + edge labels with explicit validity masks.
    """

    integer_features = {
        "global_features": [],
        "cpf_candidates": [],
        "npf_candidates": [],
        "vtx_features": [],
        "lt_candidates": [],
    }

    def __init__(self, config, outdim=128, **kwargs):
        super().__init__(config, **kwargs)

        self.classes = {name: [name] for name in config["truths"]}
        self.track_feature_names = list(config.get("cpf_candidates", []))
        self.edge_feature_names = list(config.get("npf_candidates", []))

        self.sv_w = float(config.get("sv_loss_weight", 4.76))
        self.sub_w = float(config.get("sub_loss_weight", 1.5))
        self.edge_w = float(config.get("edge_loss_weight", 2.50))
        self.metric_w = float(config.get("metric_loss_weight", 0.0))
        self._train_component_history = None
        self._val_component_history = None
        self.sv_loss_fn = nn.CrossEntropyLoss()
        self.sub_loss_fn = nn.CrossEntropyLoss()
        self.edge_loss_fn = nn.BCEWithLogitsLoss()
        self.outdim = outdim
        self.sub_gate_floor = float(config.get("sub_gate_floor", 0.05))

        in_node = len(self.track_feature_names)
        in_edge = len(self.edge_feature_names)

        self.bn0 = nn.LayerNorm(in_node, eps=1e-6)
        self.dummy_token = nn.Parameter(torch.randn(in_node))
        self.proj_skip = nn.Linear(in_node, outdim)

        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(in_edge, eps=1e-6),
            nn.Linear(in_edge, outdim),
            nn.ReLU(),
            nn.Linear(outdim, outdim * 2),
            nn.ReLU(),
            nn.Linear(outdim * 2, outdim),
        )
        self.edge_att = nn.Sequential(
            nn.Linear(outdim, outdim // 2),
            nn.ReLU(),
            nn.Linear(outdim // 2, 1),
        )

        self.edge_mlpconv1 = EdgeMLPConv(in_node, outdim, outdim // 2)
        self.edge_mlpconv2 = EdgeMLPConv(outdim // 2, outdim, outdim)
        self.bn1 = nn.LayerNorm(outdim // 2, eps=1e-6)
        self.bn2 = nn.LayerNorm(outdim, eps=1e-6)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.gate_layer = nn.Linear(outdim, outdim)

        # 3-way node type: nonSV / PV / SV
        self.sv_pred = nn.Sequential(
            nn.Linear(outdim, outdim // 2),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(outdim // 2, 3),
        )
        # Sub-SV head uses node embedding + per-node edge context + per-node edge score context.
        # SV subclass: original labels 2/3/4 -> 0/1/2
        self.sub_context_proj = nn.Sequential(
            nn.LayerNorm(outdim * 2 + 1, eps=1e-6),
            nn.Linear(outdim * 2 + 1, outdim * 2),
            nn.ELU(),
            nn.Dropout(0.2),
        )
        self.sv_sub_pred = nn.Sequential(
            nn.Linear(outdim * 2, outdim * 2),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(outdim * 2, outdim),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(outdim, 3),
        )
        self.edge_pred = nn.Sequential(
            nn.Linear(outdim * 3, outdim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(outdim, outdim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(outdim // 2, 1),
        )

    def get_dataset_kwargs(self):
        return {
            "node_label_key": "trk_label",
            "track_valid_key": "track_valid",
            "edge_label_key": "edge_label",
            "edge_valid_key": "edge_valid",
        }

    @staticmethod
    def _build_node_type_truth(node_labels):
        # 0: nonSV (1,5,6), 1: PV (0), 2: SV (2,3,4)
        node_type_truth = torch.zeros_like(node_labels)
        node_type_truth[node_labels == 0] = 1
        sv_mask = (node_labels == 2) | (node_labels == 3) | (node_labels == 4)
        node_type_truth[sv_mask] = 2
        return node_type_truth

    def forward(self, x_in, edge_index, edge_attr):
        # x_in: [1,N,F], edge_index: [1,2,E], edge_attr: [1,E,Fe]
        x = x_in.squeeze(0)
        e_idx = edge_index.squeeze(0).long()
        e_attr = edge_attr.squeeze(0)

        invalid_mask = x[:, 0] == -999.0
        x = torch.where(invalid_mask.unsqueeze(1), self.dummy_token.expand_as(x), x)
        x = self.bn0(x)

        e_attr_enc = self.edge_encoder(e_attr) if e_attr.numel() > 0 else e_attr.new_zeros((0, self.proj_skip.out_features))
        if e_idx.shape[1] > 0:
            e_scores = self.edge_att(e_attr_enc).squeeze(-1)
            alpha = softmax(e_scores, e_idx[1])
            e_attr_msg = e_attr_enc * alpha.unsqueeze(1)
            x1 = self.edge_mlpconv1(x, e_idx, e_attr_msg)
            x2 = self.edge_mlpconv2(self.drop1(F.leaky_relu(self.bn1(x1))), e_idx, e_attr_msg)
            x2 = self.drop2(F.relu(self.bn2(x2)))
        else:
            x2 = self.proj_skip(x)

        skip = self.proj_skip(x)
        gate = torch.sigmoid(self.gate_layer(skip))
        xf = gate * skip + (1.0 - gate) * x2

        sv_logits = self.sv_pred(xf)
        node_edge_ctx, node_edge_score_ctx = self._build_sub_context(xf, e_idx, e_attr_enc)
        sub_in = torch.cat([xf, node_edge_ctx, node_edge_score_ctx], dim=1)
        sub_in = self.sub_context_proj(sub_in)
        sv_sub_logits_raw = self.sv_sub_pred(sub_in)
        sv_prob = torch.softmax(sv_logits, dim=1)[:, 2:3]  # P(node is SV)
        sv_gate = self.sub_gate_floor + (1.0 - self.sub_gate_floor) * sv_prob
        sv_sub_logits = sv_sub_logits_raw * sv_gate

        if e_idx.shape[1] > 0:
            src, dst = e_idx
            edge_feat_input = torch.cat([xf[src], xf[dst], e_attr_enc], dim=1)
            edge_logits = self.edge_pred(edge_feat_input)
        else:
            edge_logits = xf.new_zeros((0, 1))

        return sv_logits, sv_sub_logits, edge_logits, xf

    def _build_sub_context(self, xf, e_idx, e_attr_enc):
        n_nodes = xf.shape[0]
        if e_idx.shape[1] == 0:
            return xf.new_zeros((n_nodes, self.outdim)), xf.new_zeros((n_nodes, 1))

        src, dst = e_idx
        # Symmetric aggregation over both endpoints.
        node_idx = torch.cat([src, dst], dim=0)
        edge_ctx = torch.cat([e_attr_enc, e_attr_enc], dim=0)
        edge_score = self.edge_att(e_attr_enc).squeeze(-1)
        edge_score = torch.cat([edge_score, edge_score], dim=0)

        deg = xf.new_zeros(n_nodes)
        deg.index_add_(0, node_idx, xf.new_ones(node_idx.shape[0]))
        deg = deg.clamp_min(1.0)

        node_edge_ctx = xf.new_zeros((n_nodes, self.outdim))
        if edge_ctx.dtype != node_edge_ctx.dtype:
            edge_ctx = edge_ctx.to(node_edge_ctx.dtype)
        node_edge_ctx.index_add_(0, node_idx, edge_ctx)
        node_edge_ctx = node_edge_ctx / deg.unsqueeze(1)

        node_edge_score_ctx = xf.new_zeros(n_nodes)
        if edge_score.dtype != node_edge_score_ctx.dtype:
            edge_score = edge_score.to(node_edge_score_ctx.dtype)
        node_edge_score_ctx.index_add_(0, node_idx, edge_score)
        node_edge_score_ctx = (node_edge_score_ctx / deg).unsqueeze(1)

        return node_edge_ctx, node_edge_score_ctx

    def _loss_on_batch(self, x_tuple, device):
        trk, edg, eidx, edge_y, node_y, trk_valid, edge_valid = x_tuple
        trk = torch.as_tensor(trk, device=device, dtype=torch.float32)
        edg = torch.as_tensor(edg, device=device, dtype=torch.float32)
        eidx = torch.as_tensor(eidx, device=device, dtype=torch.int64)
        edge_y = torch.as_tensor(edge_y, device=device, dtype=torch.float32)
        node_y = torch.as_tensor(node_y, device=device, dtype=torch.int64)
        trk_valid = torch.as_tensor(trk_valid, device=device, dtype=torch.float32)
        edge_valid = torch.as_tensor(edge_valid, device=device, dtype=torch.float32)
        B, Ntrk, _ = trk.shape
        Nedge = edg.shape[1]

        # Valid nodes are non-padding tracks with valid node labels.
        node_keep = (trk_valid > 0.5) & (node_y >= 0)
        node_counts = node_keep.sum(dim=1)  # [B]
        valid_graph_mask = node_counts > 0
        n_graph = int(valid_graph_mask.sum().item())
        if n_graph == 0:
            zero = trk.new_tensor(0.0, requires_grad=True)
            return zero, 0.0, {
                "sv": 0.0,
                "sub": 0.0,
                "edge": 0.0,
                "sv_acc": 0.0,
                "sub_acc": 0.0,
                "edge_acc": 0.0,
            }

        # Flatten kept nodes and build node->event map.
        event_ids_nodes = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, Ntrk)
        node_event = event_ids_nodes[node_keep]  # [N_total]
        x_nodes_all = trk[node_keep]            # [N_total, F]
        node_labels_all = node_y[node_keep]     # [N_total]

        # Map local node indices [B, Ntrk] -> global flattened node indices.
        node_map = torch.full((B, Ntrk), -1, dtype=torch.long, device=device)
        node_map[node_keep] = torch.arange(x_nodes_all.shape[0], device=device, dtype=torch.long)

        # Build global edge list from padded edge arrays.
        src_local = eidx[:, 0, :]
        dst_local = eidx[:, 1, :]
        edge_base = edge_valid > 0.5
        idx_in_range = (
            (src_local >= 0)
            & (dst_local >= 0)
            & (src_local < Ntrk)
            & (dst_local < Ntrk)
        )
        edge_mask = edge_base & idx_in_range

        src_safe = src_local.clamp(min=0, max=max(Ntrk - 1, 0))
        dst_safe = dst_local.clamp(min=0, max=max(Ntrk - 1, 0))
        src_global = node_map.gather(1, src_safe)
        dst_global = node_map.gather(1, dst_safe)
        edge_mask = edge_mask & (src_global >= 0) & (dst_global >= 0)

        event_ids_edges = torch.arange(B, device=device, dtype=torch.long).unsqueeze(1).expand(B, Nedge)
        edge_event = event_ids_edges[edge_mask]         # [E_total]
        edge_index_global = torch.stack(
            (src_global[edge_mask], dst_global[edge_mask]),
            dim=0,
        )                                              # [2, E_total]
        edge_attr_all = edg[edge_mask]                # [E_total, Fe]
        edge_targets_all = edge_y[edge_mask].float()  # [E_total]

        # Single forward over one disconnected large graph.
        sv_logits, sv_sub_logits, edge_logits, _ = self(
            x_nodes_all.unsqueeze(0),
            edge_index_global.unsqueeze(0),
            edge_attr_all.unsqueeze(0),
        )

        # Helper tensors for per-event means.
        node_counts_f = node_counts.float().clamp_min(1.0)
        valid_graph_idx = torch.where(valid_graph_mask)[0]

        sv_truth = self._build_node_type_truth(node_labels_all)
        sv_pred = sv_logits.argmax(1)

        # SV loss/acc: mean over events of per-event means.
        sv_loss_vec = F.cross_entropy(sv_logits, sv_truth, reduction="none")
        sv_loss_sum_evt = trk.new_zeros(B)
        sv_loss_sum_evt.index_add_(0, node_event, sv_loss_vec)
        sv_loss_evt = sv_loss_sum_evt[valid_graph_idx] / node_counts_f[valid_graph_idx]
        sv_loss = sv_loss_evt.mean()

        sv_corr_vec = (sv_pred == sv_truth).float()
        sv_corr_sum_evt = trk.new_zeros(B)
        sv_corr_sum_evt.index_add_(0, node_event, sv_corr_vec)
        sv_acc_evt = sv_corr_sum_evt[valid_graph_idx] / node_counts_f[valid_graph_idx]
        sv_acc = sv_acc_evt.mean()

        # Sub-SV loss/acc: average over events that contain SV nodes.
        sv_mask = sv_truth == 2
        if torch.any(sv_mask):
            sub_truth = node_labels_all[sv_mask] - 2
            sub_logits = sv_sub_logits[sv_mask]
            sub_event = node_event[sv_mask]

            sub_loss_vec = F.cross_entropy(sub_logits, sub_truth, reduction="none")
            sub_sum_evt = trk.new_zeros(B)
            sub_cnt_evt = trk.new_zeros(B)
            sub_sum_evt.index_add_(0, sub_event, sub_loss_vec)
            sub_cnt_evt.index_add_(0, sub_event, torch.ones_like(sub_loss_vec))
            sub_evt_mask = sub_cnt_evt > 0
            sub_loss = (sub_sum_evt[sub_evt_mask] / sub_cnt_evt[sub_evt_mask]).mean()

            sub_corr_vec = (sub_logits.argmax(1) == sub_truth).float()
            sub_corr_evt = trk.new_zeros(B)
            sub_corr_evt.index_add_(0, sub_event, sub_corr_vec)
            sub_acc = (sub_corr_evt[sub_evt_mask] / sub_cnt_evt[sub_evt_mask]).mean()
        else:
            sub_loss = sv_loss.new_tensor(0.0)
            sub_acc = sv_loss.new_tensor(0.0)

        # Edge loss/acc: average over events that contain valid edges.
        if edge_logits.numel() > 0:
            edge_logits_flat = edge_logits.view(-1)
            edge_loss_vec = F.binary_cross_entropy_with_logits(
                edge_logits_flat,
                edge_targets_all,
                reduction="none",
            )
            edge_sum_evt = trk.new_zeros(B)
            edge_cnt_evt = trk.new_zeros(B)
            edge_sum_evt.index_add_(0, edge_event, edge_loss_vec)
            edge_cnt_evt.index_add_(0, edge_event, torch.ones_like(edge_loss_vec))
            edge_evt_mask = edge_cnt_evt > 0
            edge_loss = (edge_sum_evt[edge_evt_mask] / edge_cnt_evt[edge_evt_mask]).mean()

            edge_pred = (torch.sigmoid(edge_logits_flat) >= 0.5).float()
            edge_corr_vec = (edge_pred == edge_targets_all).float()
            edge_corr_evt = trk.new_zeros(B)
            edge_corr_evt.index_add_(0, edge_event, edge_corr_vec)
            edge_acc = (edge_corr_evt[edge_evt_mask] / edge_cnt_evt[edge_evt_mask]).mean()
        else:
            edge_loss = sv_loss.new_tensor(0.0)
            edge_acc = sv_loss.new_tensor(0.0)

        total_loss = self.sv_w * sv_loss + self.sub_w * sub_loss + self.edge_w * edge_loss + self.metric_w * 0.0

        return (
            total_loss,
            float(sv_acc.detach().cpu().item()),
            {
                "sv": float(sv_loss.detach().cpu().item()),
                "sub": float(sub_loss.detach().cpu().item()),
                "edge": float(edge_loss.detach().cpu().item()),
                "sv_acc": float(sv_acc.detach().cpu().item()),
                "sub_acc": float(sub_acc.detach().cpu().item()),
                "edge_acc": float(edge_acc.detach().cpu().item()),
            },
        )

    def update(
        self,
        dataloader,
        loss_fn,
        optimizer,
        scheduler=None,
        batch_lr=False,
        attack=None,
        scaler=None,
        device="cpu",
        verbose=True,
        gradient_accumulation_steps=1,
    ):
        self.train()
        t0 = time.time()
        losses, accs = [], []
        comps = {"sv": [], "sub": [], "edge": [], "sv_acc": [], "sub_acc": [], "edge_acc": []}
        profile = os.getenv("B_HIVE_PROFILE", "0") == "1"
        profile_interval = max(1, int(os.getenv("B_HIVE_PROFILE_INTERVAL", "50")))
        last_iter_end = time.time()
        sum_data_s = 0.0
        sum_compute_s = 0.0
        sum_opt_s = 0.0
        n_seen = 0
        n_batches = 0

        device_type = "cuda" if "cuda" in str(device) else "cpu"
        use_amp = bool(getattr(self, "mixed_precision", False) and device_type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        for b, (x, truth, w, p) in enumerate(dataloader):
            iter_start = time.time()
            data_wait_s = iter_start - last_iter_end
            compute_start = time.time()
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
                loss, acc, c = self._loss_on_batch(x, device)
            compute_s = time.time() - compute_start
            loss = loss / max(1, gradient_accumulation_steps)
            if use_amp and (scaler is not None):
                scaler.scale(loss).backward()
            else:
                loss.backward()

            opt_s = 0.0
            if (b + 1) % max(1, gradient_accumulation_steps) == 0:
                opt_start = time.time()
                if use_amp and (scaler is not None):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if batch_lr and (scheduler is not None):
                    scheduler.step()
                opt_s = time.time() - opt_start

            losses.append(float(loss.detach().cpu().item()) * max(1, gradient_accumulation_steps))
            accs.append(acc)
            for k in comps:
                comps[k].append(c[k])

            bsz = int(len(truth)) if hasattr(truth, "__len__") else 0
            n_seen += max(0, bsz)
            n_batches += 1
            sum_data_s += data_wait_s
            sum_compute_s += compute_s
            sum_opt_s += opt_s
            iter_total_s = max(1e-9, time.time() - iter_start)
            if profile and ((b + 1) % profile_interval == 0):
                print(
                    f"[PROFILE][GraphEventGNN][train] batch={b+1} "
                    f"data_ms={sum_data_s / n_batches * 1e3:.2f} "
                    f"compute_ms={sum_compute_s / n_batches * 1e3:.2f} "
                    f"opt_ms={sum_opt_s / n_batches * 1e3:.2f} "
                    f"batch_s={1.0 / iter_total_s:.2f} "
                    f"evt_s={bsz / iter_total_s:.2f}",
                    flush=True,
                )
            last_iter_end = time.time()

        if len(losses) and ((b + 1) % max(1, gradient_accumulation_steps) != 0):
            if use_amp and (scaler is not None):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if (not batch_lr) and (scheduler is not None):
            scheduler.step()

        self._last_update_metrics = {k: float(np.mean(v) if len(v) else 0.0) for k, v in comps.items()}
        if self._train_component_history is not None:
            for k in comps:
                self._train_component_history[k].append(self._last_update_metrics[k])
        active_run = mlflow.active_run()
        if active_run and self._train_component_history is not None:
            step = len(self._train_component_history["sv"]) - 1
            m = self._last_update_metrics
            mlflow.log_metric("graph_train_sv_loss", m["sv"], step=step)
            mlflow.log_metric("graph_train_sub_loss", m["sub"], step=step)
            mlflow.log_metric("graph_train_edge_loss", m["edge"], step=step)
            mlflow.log_metric("graph_train_sv_acc", m["sv_acc"], step=step)
            mlflow.log_metric("graph_train_sub_acc", m["sub_acc"], step=step)
            mlflow.log_metric("graph_train_edge_acc", m["edge_acc"], step=step)

        return float(np.mean(losses) if losses else 0.0), float(np.mean(accs) if accs else 0.0), time.time() - t0

    def validate_model(
        self,
        dataloader,
        loss_fn,
        device="cpu",
        verbose=True,
        terminal_plot=False,
    ):
        self.eval()
        t0 = time.time()
        losses, accs = [], []
        comps = {"sv": [], "sub": [], "edge": [], "sv_acc": [], "sub_acc": [], "edge_acc": []}
        profile = os.getenv("B_HIVE_PROFILE", "0") == "1"
        profile_interval = max(1, int(os.getenv("B_HIVE_PROFILE_INTERVAL", "50")))
        last_iter_end = time.time()
        sum_data_s = 0.0
        sum_compute_s = 0.0
        n_seen = 0
        n_batches = 0
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        use_amp = bool(getattr(self, "mixed_precision", False) and device_type == "cuda")
        with torch.no_grad():
            for b, (x, truth, w, p) in enumerate(dataloader):
                iter_start = time.time()
                data_wait_s = iter_start - last_iter_end
                compute_start = time.time()
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
                    loss, acc, c = self._loss_on_batch(x, device)
                compute_s = time.time() - compute_start
                losses.append(float(loss.detach().cpu().item()))
                accs.append(acc)
                for k in comps:
                    comps[k].append(c[k])
                bsz = int(len(truth)) if hasattr(truth, "__len__") else 0
                n_seen += max(0, bsz)
                n_batches += 1
                sum_data_s += data_wait_s
                sum_compute_s += compute_s
                iter_total_s = max(1e-9, time.time() - iter_start)
                if profile and ((b + 1) % profile_interval == 0):
                    print(
                        f"[PROFILE][GraphEventGNN][val] batch={b+1} "
                        f"data_ms={sum_data_s / n_batches * 1e3:.2f} "
                        f"compute_ms={sum_compute_s / n_batches * 1e3:.2f} "
                        f"batch_s={1.0 / iter_total_s:.2f} "
                        f"evt_s={bsz / iter_total_s:.2f}",
                        flush=True,
                    )
                last_iter_end = time.time()
        self._last_validate_metrics = {k: float(np.mean(v) if len(v) else 0.0) for k, v in comps.items()}
        if self._val_component_history is not None:
            for k in comps:
                self._val_component_history[k].append(self._last_validate_metrics[k])
        if verbose:
            m = self._last_validate_metrics
            print(
                "[VAL][GraphEventGNN] "
                f"sv_loss={m['sv']:.4f} sub_loss={m['sub']:.4f} edge_loss={m['edge']:.4f} "
                f"sv_acc={m['sv_acc']:.4f} sub_acc={m['sub_acc']:.4f} edge_acc={m['edge_acc']:.4f}"
            )
        active_run = mlflow.active_run()
        if active_run and self._val_component_history is not None:
            step = len(self._val_component_history["sv"]) - 1
            m = self._last_validate_metrics
            mlflow.log_metric("graph_val_sv_loss", m["sv"], step=step)
            mlflow.log_metric("graph_val_sub_loss", m["sub"], step=step)
            mlflow.log_metric("graph_val_edge_loss", m["edge"], step=step)
            mlflow.log_metric("graph_val_sv_acc", m["sv_acc"], step=step)
            mlflow.log_metric("graph_val_sub_acc", m["sub_acc"], step=step)
            mlflow.log_metric("graph_val_edge_acc", m["edge_acc"], step=step)
        return float(np.mean(losses) if losses else 0.0), float(np.mean(accs) if accs else 0.0), time.time() - t0

    def _plot_multitask_metrics(self, output_dir):
        if self._train_component_history is None or self._val_component_history is None:
            return
        if len(self._train_component_history["sv"]) == 0:
            return

        x = np.arange(1, len(self._train_component_history["sv"]) + 1)

        # Loss components
        fig, ax = plt.subplots()
        ax.set_title("Graph Multitask Loss Components")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid()
        color_map = {"sv": "tab:blue", "sub": "tab:orange", "edge": "tab:green"}
        for k, label in [("sv", "SV"), ("sub", "Sub-SV"), ("edge", "Edge")]:
            c = color_map[k]
            ax.plot(x, self._train_component_history[k], "-o", color=c, label=f"Train {label}")
            ax.plot(x, self._val_component_history[k], "--o", color=c, label=f"Val {label}")
        ax.legend(ncol=2)
        fig.savefig(os.path.join(output_dir, "multitask_loss.png"))
        fig.savefig(os.path.join(output_dir, "multitask_loss.pdf"))
        plt.close(fig)

        # Accuracy components
        fig, ax = plt.subplots()
        ax.set_title("Graph Multitask Accuracy Components")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.grid()
        color_map_acc = {"sv_acc": "tab:blue", "sub_acc": "tab:orange", "edge_acc": "tab:green"}
        for k, label in [("sv_acc", "SV"), ("sub_acc", "Sub-SV"), ("edge_acc", "Edge")]:
            c = color_map_acc[k]
            ax.plot(x, self._train_component_history[k], "-o", color=c, label=f"Train {label}")
            ax.plot(x, self._val_component_history[k], "--o", color=c, label=f"Val {label}")
        ax.legend(ncol=2)
        fig.savefig(os.path.join(output_dir, "multitask_acc.png"))
        fig.savefig(os.path.join(output_dir, "multitask_acc.pdf"))
        plt.close(fig)

        np.savez(
            os.path.join(output_dir, "multitask_metrics.npz"),
            train_sv_loss=np.array(self._train_component_history["sv"]),
            val_sv_loss=np.array(self._val_component_history["sv"]),
            train_sub_loss=np.array(self._train_component_history["sub"]),
            val_sub_loss=np.array(self._val_component_history["sub"]),
            train_edge_loss=np.array(self._train_component_history["edge"]),
            val_edge_loss=np.array(self._val_component_history["edge"]),
            train_sv_acc=np.array(self._train_component_history["sv_acc"]),
            val_sv_acc=np.array(self._val_component_history["sv_acc"]),
            train_sub_acc=np.array(self._train_component_history["sub_acc"]),
            val_sub_acc=np.array(self._val_component_history["sub_acc"]),
            train_edge_acc=np.array(self._train_component_history["edge_acc"]),
            val_edge_acc=np.array(self._val_component_history["edge_acc"]),
        )

    def train_model(self, *args, **kwargs):
        output_dir = kwargs.get("directory", args[2] if len(args) > 2 else None)
        self._train_component_history = {"sv": [], "sub": [], "edge": [], "sv_acc": [], "sub_acc": [], "edge_acc": []}
        self._val_component_history = {"sv": [], "sub": [], "edge": [], "sv_acc": [], "sub_acc": [], "edge_acc": []}
        multitask_path = os.path.join(output_dir, "multitask_metrics.npz") if output_dir else None
        if multitask_path and os.path.isfile(multitask_path):
            try:
                mm = np.load(multitask_path, allow_pickle=True)
                self._train_component_history = {
                    "sv": mm["train_sv_loss"].tolist(),
                    "sub": mm["train_sub_loss"].tolist(),
                    "edge": mm["train_edge_loss"].tolist(),
                    "sv_acc": mm["train_sv_acc"].tolist(),
                    "sub_acc": mm["train_sub_acc"].tolist(),
                    "edge_acc": mm["train_edge_acc"].tolist(),
                }
                self._val_component_history = {
                    "sv": mm["val_sv_loss"].tolist(),
                    "sub": mm["val_sub_loss"].tolist(),
                    "edge": mm["val_edge_loss"].tolist(),
                    "sv_acc": mm["val_sv_acc"].tolist(),
                    "sub_acc": mm["val_sub_acc"].tolist(),
                    "edge_acc": mm["val_edge_acc"].tolist(),
                }
            except Exception:
                pass
        train_metrics, validation_metrics = super().train_model(*args, **kwargs)
        if output_dir:
            self._plot_multitask_metrics(output_dir)
        return train_metrics, validation_metrics

    def train_model_iter(
        self,
        training_data,
        validation_data,
        directory,
        loss_fn,
        total_number_iterations,
        N_iters_per_save,
        attack=None,
        optimizer=None,
        scheduler=None,
        batch_lr=False,
        device=None,
        nepochs=0,
        best_loss_val=np.inf,
        resume_epochs=0,
        train_metrics=None,
        validation_metrics=None,
        terminal_plot=False,
        torch_compile_mode=None,
        keep_all_epochs_models=True,
        gradient_accumulation_steps=1,
        **kwargs,
    ):
        if train_metrics is None:
            train_metrics = {"loss": [], "acc": []}
        if validation_metrics is None:
            validation_metrics = {"loss": [], "acc": []}
        if N_iters_per_save <= 0:
            raise ValueError("N_iters_per_save must be > 0")

        self._train_component_history = {"sv": [], "sub": [], "edge": [], "sv_acc": [], "sub_acc": [], "edge_acc": []}
        self._val_component_history = {"sv": [], "sub": [], "edge": [], "sv_acc": [], "sub_acc": [], "edge_acc": []}

        mlflow_parent_run = mlflow.active_run()
        if mlflow_parent_run:
            mlflow.start_run(nested=True, run_name="model_performance_metrics")
            mlflow_nested_run = mlflow.active_run()
            print(
                "Active nested MLflow run name: "
                f"{mlflow_nested_run.info.run_name}, run_id: {mlflow_nested_run.info.run_id}"
            )
        else:
            mlflow_nested_run = None

        device_type = "cuda" if "cuda" in str(device) else "cpu"
        use_amp = bool(getattr(self, "mixed_precision", False) and device_type == "cuda")
        scaler = torch.amp.GradScaler(device, enabled=use_amp)
        profile = os.getenv("B_HIVE_PROFILE", "0") == "1"
        profile_interval = max(1, int(os.getenv("B_HIVE_PROFILE_INTERVAL", "50")))

        expected_checkpoints = int(math.ceil(float(total_number_iterations) / float(N_iters_per_save)))
        if os.path.isfile(f"{directory}/train_time.npy"):
            train_time = np.load(f"{directory}/train_time.npy").tolist()
        else:
            train_time = []
        if os.path.isfile(f"{directory}/val_time.npy"):
            val_time = np.load(f"{directory}/val_time.npy").tolist()
        else:
            val_time = []

        # In iteration mode, resume_epochs corresponds to last saved model index (iteration number).
        global_b = int(resume_epochs) if resume_epochs else len(train_metrics.get("loss", [])) * N_iters_per_save
        global_b = max(0, min(global_b, int(total_number_iterations)))
        checkpoint_idx = len(train_metrics.get("loss", []))

        if global_b >= total_number_iterations:
            if directory:
                self._plot_multitask_metrics(directory)
            if mlflow_nested_run:
                mlflow.end_run()
            return train_metrics, validation_metrics

        accum_counter = 0
        last_iter_end = time.time()
        prof_data_s = 0.0
        prof_compute_s = 0.0
        prof_opt_s = 0.0
        prof_batches = 0
        while global_b < total_number_iterations:
            self.train()
            training_data.dataset.shuffleFileList()

            window_start = time.time()
            window_loss = 0.0
            window_acc = 0.0
            window_n = 0
            window_comp = {"sv": 0.0, "sub": 0.0, "edge": 0.0, "sv_acc": 0.0, "sub_acc": 0.0, "edge_acc": 0.0}

            optimizer.zero_grad(set_to_none=True)

            for x, truth, w, p in training_data:
                iter_start = time.time()
                data_wait_s = iter_start - last_iter_end
                compute_start = time.time()
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
                    loss, acc, c = self._loss_on_batch(x, device)
                compute_s = time.time() - compute_start
                raw_loss = float(loss.detach().cpu().item())
                loss = loss / max(1, gradient_accumulation_steps)

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_counter += 1
                opt_s = 0.0
                if accum_counter % max(1, gradient_accumulation_steps) == 0:
                    opt_start = time.time()
                    if use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if batch_lr and (scheduler is not None):
                        scheduler.step()
                    opt_s = time.time() - opt_start

                window_loss += raw_loss
                window_acc += float(acc)
                window_n += 1
                for k in window_comp:
                    window_comp[k] += float(c[k])
                global_b += 1

                bsz = int(len(truth)) if hasattr(truth, "__len__") else 0
                prof_data_s += data_wait_s
                prof_compute_s += compute_s
                prof_opt_s += opt_s
                prof_batches += 1
                iter_total_s = max(1e-9, time.time() - iter_start)
                if profile and (global_b % profile_interval == 0):
                    print(
                        f"[PROFILE][GraphEventGNN][train][iter] batch={global_b} "
                        f"data_ms={prof_data_s / max(1, prof_batches) * 1e3:.2f} "
                        f"compute_ms={prof_compute_s / max(1, prof_batches) * 1e3:.2f} "
                        f"opt_ms={prof_opt_s / max(1, prof_batches) * 1e3:.2f} "
                        f"batch_s={1.0 / iter_total_s:.2f} "
                        f"evt_s={bsz / iter_total_s:.2f}",
                        flush=True,
                    )
                last_iter_end = time.time()

                if global_b >= total_number_iterations or (global_b % N_iters_per_save == 0):
                    # Flush tail accumulated gradients before validation/checkpoint.
                    if accum_counter % max(1, gradient_accumulation_steps) != 0:
                        if use_amp:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    loss_training = window_loss / max(window_n, 1)
                    acc_training = window_acc / max(window_n, 1)
                    train_comp_avg = {k: (window_comp[k] / max(window_n, 1)) for k in window_comp}
                    t_train = time.time() - window_start

                    train_metrics["loss"].append(loss_training)
                    train_metrics["acc"].append(acc_training)
                    train_time.append(t_train)
                    if self._train_component_history is not None:
                        for k in window_comp:
                            self._train_component_history[k].append(train_comp_avg[k])

                    val_start = time.time()
                    loss_validation, acc_validation, _ = self.validate_model(
                        validation_data,
                        loss_fn,
                        device=device,
                        terminal_plot=terminal_plot,
                    )
                    t_val = time.time() - val_start
                    validation_metrics["loss"].append(loss_validation)
                    validation_metrics["acc"].append(acc_validation)
                    val_time.append(t_val)

                    checkpoint = {
                        "epoch": global_b,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                        "loss_train": loss_training,
                        "acc_train": acc_training,
                        "loss_val": loss_validation,
                        "acc_val": acc_validation,
                    }
                    torch.save(checkpoint, f"{directory}/model_{global_b}.pt")

                    if (not keep_all_epochs_models) and (global_b - N_iters_per_save > 0):
                        old_path = f"{directory}/model_{global_b - N_iters_per_save}.pt"
                        if os.path.exists(old_path):
                            os.remove(old_path)

                    if loss_validation < best_loss_val:
                        best_loss_val = loss_validation
                        torch.save(checkpoint, f"{directory}/best_model.pt")

                    np.save(f"{directory}/train_time.npy", np.array(train_time, dtype=np.float32))
                    np.save(f"{directory}/val_time.npy", np.array(val_time, dtype=np.float32))
                    np.savez(
                        f"{directory}/training_metrics",
                        loss=train_metrics["loss"],
                        acc=train_metrics["acc"],
                        allow_pickle=True,
                    )
                    np.savez(
                        f"{directory}/validation_metrics",
                        loss=validation_metrics["loss"],
                        acc=validation_metrics["acc"],
                        allow_pickle=True,
                    )

                    if mlflow_nested_run:
                        mlflow.log_metric("Training Loss", loss_training, step=global_b)
                        mlflow.log_metric("Training Accuracy", acc_training, step=global_b)
                        mlflow.log_metric("graph_train_sv_loss", train_comp_avg["sv"], step=global_b)
                        mlflow.log_metric("graph_train_sub_loss", train_comp_avg["sub"], step=global_b)
                        mlflow.log_metric("graph_train_edge_loss", train_comp_avg["edge"], step=global_b)
                        mlflow.log_metric("graph_train_sv_acc", train_comp_avg["sv_acc"], step=global_b)
                        mlflow.log_metric("graph_train_sub_acc", train_comp_avg["sub_acc"], step=global_b)
                        mlflow.log_metric("graph_train_edge_acc", train_comp_avg["edge_acc"], step=global_b)
                        mlflow.log_metric("Validation Loss", loss_validation, step=global_b)
                        mlflow.log_metric("Validation Accuracy", acc_validation, step=global_b)

                    if directory:
                        self._plot_multitask_metrics(directory)

                    checkpoint_idx += 1
                    print(
                        f"[iter {global_b}/{total_number_iterations}] "
                        f"train_loss={loss_training:.4f} train_acc={acc_training:.4f} "
                        f"val_loss={loss_validation:.4f} val_acc={acc_validation:.4f} "
                        f"| train(sv/sub/edge)=({train_comp_avg['sv']:.4f}/{train_comp_avg['sub']:.4f}/{train_comp_avg['edge']:.4f}) "
                        f"| val(sv/sub/edge)=({self._last_validate_metrics['sv']:.4f}/{self._last_validate_metrics['sub']:.4f}/{self._last_validate_metrics['edge']:.4f}) "
                        f"| val_acc(sv/sub/edge)=({self._last_validate_metrics['sv_acc']:.4f}/{self._last_validate_metrics['sub_acc']:.4f}/{self._last_validate_metrics['edge_acc']:.4f})"
                    )

                    window_start = time.time()
                    window_loss = 0.0
                    window_acc = 0.0
                    window_n = 0
                    window_comp = {"sv": 0.0, "sub": 0.0, "edge": 0.0, "sv_acc": 0.0, "sub_acc": 0.0, "edge_acc": 0.0}

                    if global_b >= total_number_iterations:
                        break

            if (not batch_lr) and (scheduler is not None):
                scheduler.step()

        if directory:
            self._plot_multitask_metrics(directory)
        if mlflow_nested_run:
            try:
                mlflow.end_run()
                print("MLflow nested run ended.")
            except Exception as e:
                print(f"Error ending MLflow run: {e}")
        return train_metrics, validation_metrics

    def predict_model(self, dataloader, output, loss_fn, device, attack=None):
        self.eval()
        sv_logits_all = []
        sv_truth_all = []
        sub_logits_all = []
        sub_truth_all = []
        edge_logits_all = []
        edge_truth_all = []
        process_all = []

        with torch.no_grad():
            for x, truth, w, p in dataloader:
                trk, edg, eidx, edge_y, node_y, trk_valid, edge_valid = x
                trk = torch.as_tensor(trk, device=device, dtype=torch.float32)
                edg = torch.as_tensor(edg, device=device, dtype=torch.float32)
                eidx = torch.as_tensor(eidx, device=device, dtype=torch.int64)
                edge_y = torch.as_tensor(edge_y, device=device, dtype=torch.float32)
                node_y = torch.as_tensor(node_y, device=device, dtype=torch.int64)
                trk_valid = torch.as_tensor(trk_valid, device=device, dtype=torch.float32)
                edge_valid = torch.as_tensor(edge_valid, device=device, dtype=torch.float32)

                for i in range(trk.shape[0]):
                    tmask = trk_valid[i] > 0.5
                    x_nodes = trk[i][tmask]
                    node_labels = node_y[i][tmask]
                    valid_label = node_labels >= 0

                    if x_nodes.numel() == 0 or not torch.any(valid_label):
                        continue

                    x_nodes = x_nodes[valid_label]
                    node_labels = node_labels[valid_label]
                    sv_truth = self._build_node_type_truth(node_labels)

                    emask = edge_valid[i] > 0.5
                    edge_idx_i = eidx[i][:, emask]
                    edge_attr_i = edg[i][emask]
                    edge_y_i = edge_y[i][emask]
                    if edge_idx_i.numel() > 0:
                        n_nodes = x_nodes.shape[0]
                        keep = (
                            (edge_idx_i[0] >= 0)
                            & (edge_idx_i[1] >= 0)
                            & (edge_idx_i[0] < n_nodes)
                            & (edge_idx_i[1] < n_nodes)
                        )
                        edge_idx_i = edge_idx_i[:, keep]
                        edge_attr_i = edge_attr_i[keep]
                        edge_y_i = edge_y_i[keep]

                    sv_logits, sv_sub_logits, edge_logits, _ = self(
                        x_nodes.unsqueeze(0),
                        edge_idx_i.unsqueeze(0),
                        edge_attr_i.unsqueeze(0),
                    )

                    sv_logits_all.append(sv_logits.detach().cpu().numpy())
                    sv_truth_all.append(sv_truth.detach().cpu().numpy())

                    sv_mask = sv_truth == 2
                    if torch.any(sv_mask):
                        sub_logits_all.append(sv_sub_logits[sv_mask].detach().cpu().numpy())
                        sub_truth_all.append((node_labels[sv_mask] - 2).detach().cpu().numpy())

                    if edge_logits.numel() > 0:
                        edge_logits_all.append(edge_logits.view(-1).detach().cpu().numpy())
                        edge_truth_all.append(edge_y_i.detach().cpu().numpy())

                    process_all.append(np.array([p[i]], dtype=np.float32))

        node_sv_logits = np.concatenate(sv_logits_all, axis=0) if sv_logits_all else np.empty((0, 3), dtype=np.float32)
        node_sv_truth = np.concatenate(sv_truth_all, axis=0) if sv_truth_all else np.empty((0,), dtype=np.int64)
        sub_sv_logits = np.concatenate(sub_logits_all, axis=0) if sub_logits_all else np.empty((0, 3), dtype=np.float32)
        sub_sv_truth = np.concatenate(sub_truth_all, axis=0) if sub_truth_all else np.empty((0,), dtype=np.int64)
        edge_logits_arr = np.concatenate(edge_logits_all, axis=0) if edge_logits_all else np.empty((0,), dtype=np.float32)
        edge_truth_arr = np.concatenate(edge_truth_all, axis=0) if edge_truth_all else np.empty((0,), dtype=np.float32)
        procs_arr = np.concatenate(process_all, axis=0) if process_all else np.empty((0,), dtype=np.float32)

        node_sv_prob = torch.softmax(torch.as_tensor(node_sv_logits), dim=1).cpu().numpy() if len(node_sv_logits) else np.empty((0, 3), dtype=np.float32)
        sub_sv_prob = torch.softmax(torch.as_tensor(sub_sv_logits), dim=1).cpu().numpy() if len(sub_sv_logits) else np.empty((0, 3), dtype=np.float32)
        edge_prob = torch.sigmoid(torch.as_tensor(edge_logits_arr)).cpu().numpy() if len(edge_logits_arr) else np.empty((0,), dtype=np.float32)

        predictions = node_sv_prob
        truths_arr = node_sv_truth
        kinematics = np.empty((len(truths_arr), 2), dtype=np.float32)

        np.save(output["prediction"].path, predictions)
        np.save(output["truth"].path, truths_arr)
        np.save(output["kinematics"].path, kinematics)
        np.save(output["process"].path, procs_arr)
        np.savez(output["inference_metrics"].path, time=0.0, loss=0.0, acc=0.0, allow_pickle=True)

        graph_out = Path(output["prediction"].path).parent / "graph_task_outputs.npz"
        np.savez(
            graph_out,
            node_sv_logits=node_sv_logits,
            node_sv_prob=node_sv_prob,
            node_sv_truth=node_sv_truth,
            sub_sv_logits=sub_sv_logits,
            sub_sv_prob=sub_sv_prob,
            sub_sv_truth=sub_sv_truth,
            edge_logits=edge_logits_arr,
            edge_prob=edge_prob,
            edge_truth=edge_truth_arr,
            process=procs_arr,
            allow_pickle=True,
        )
        return predictions, truths_arr, kinematics, procs_arr, 0.0
