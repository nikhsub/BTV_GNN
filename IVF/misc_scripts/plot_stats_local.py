import json
import matplotlib.pyplot as plt
import argparse
import numpy as np

parser = argparse.ArgumentParser("Stats check")
parser.add_argument("-f", "--infile", default="", help="Path to input json file")
parser.add_argument("-t", "--test", default=False, action="store_true", help="Show test curves")
args = parser.parse_args()

show_test = args.test

# -----------------------------------------------------
# Load JSON
# -----------------------------------------------------
with open(args.infile, 'r') as f:
    data = json.load(f)

epochs_data = data["epochs"]
best_epoch = data["best_epoch"]["epoch"]

# Helper to detect missing values
def get_list(key):
    return np.array([e.get(key, None) for e in epochs_data])


# -----------------------------------------------------
# Extract lists
# -----------------------------------------------------
epochs = np.array([e["epoch"] for e in epochs_data])

train_tot_loss  = get_list("train_tot_loss")
train_node_loss = get_list("train_node_loss")
train_edge_loss = get_list("train_edge_loss")

test_node_loss  = get_list("test_node_loss")
test_edge_loss  = get_list("test_edge_loss")

# Validation if present
val_node_loss   = get_list("val_node_loss")
val_edge_loss   = get_list("val_edge_loss")

# AUC metrics
test_edge_auc = get_list("test_edge_auc")

auc_primary   = get_list("test_auc_primary")
auc_pileup    = get_list("test_auc_pileup")
auc_fromB     = get_list("test_auc_fromB")
auc_fromBC    = get_list("test_auc_fromBC")
auc_fromC     = get_list("test_auc_fromC")
auc_other     = get_list("test_auc_other")
auc_fake      = get_list("test_auc_fake")

# Validation AUCs (if present)
metric          = get_list("metric")
val_edge_auc    = get_list("val_edge_auc")
val_auc_primary = get_list("val_auc_primary")
val_auc_pileup  = get_list("val_auc_pileup")
val_auc_fromB   = get_list("val_auc_fromB")
val_auc_fromBC  = get_list("val_auc_fromBC")
val_auc_fromC   = get_list("val_auc_fromC")
val_auc_other   = get_list("val_auc_other")
val_auc_fake    = get_list("val_auc_fake")

valid_val_mask = [v!=-1 for v in val_edge_auc]
valid_val_epochs = epochs[valid_val_mask]

valid_val_node_loss   = val_node_loss[valid_val_mask]
valid_val_edge_loss   = val_edge_loss[valid_val_mask]
valid_metric	      = metric[valid_val_mask]
valid_val_edge_auc    = val_edge_auc[valid_val_mask]
valid_val_auc_primary = val_auc_primary[valid_val_mask]
valid_val_auc_pileup  = val_auc_pileup[valid_val_mask]
valid_val_auc_fromB   = val_auc_fromB[valid_val_mask]
valid_val_auc_fromBC  = val_auc_fromBC[valid_val_mask]
valid_val_auc_fromC   = val_auc_fromC[valid_val_mask]
valid_val_auc_other   = val_auc_other[valid_val_mask]
valid_val_auc_fake    = val_auc_fake[valid_val_mask]

# -----------------------------------------------------
# Plotting
# -----------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# ========= LOSS PLOTS =========
# Style groups
train_style = dict(marker='o', linestyle='-', linewidth=2)
val_style   = dict(marker='s', linestyle='--', linewidth=2)
test_style  = dict(marker=',', linestyle=':', linewidth=2)

ax1.plot(epochs, train_tot_loss,  label="Train Total",  color='blue', **train_style)
ax1.plot(epochs, train_node_loss, label="Train Node",   color='orange', **train_style)
ax1.plot(epochs, train_edge_loss, label="Train Edge",   color='green', **train_style)

ax1.plot(valid_val_epochs, valid_val_node_loss, label="Val Node", color='orange', **val_style)
ax1.plot(valid_val_epochs, valid_val_edge_loss, label="Val Edge", color='green', **val_style)

if(show_test):
	ax1.plot(epochs, test_node_loss,  label="Test Node", color='orange', **test_style)
	ax1.plot(epochs, test_edge_loss,  label="Test Edge", color='green', **test_style)

ax1.axvline(best_epoch, color='grey', linestyle='--', label=f"Best Epoch: {best_epoch}")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Loss Curves")
ax1.grid(True)
ax1.legend()

# ========= AUC + METRIC PLOTS =========
# Edge/test AUC (highlight)
if(show_test):
	ax2.plot(epochs, test_edge_auc, label="Edge AUC (test)", color='green', **test_style)
ax2.plot(valid_val_epochs, valid_val_edge_auc, label="Edge AUC (val)", color='green', **val_style)

# Metric
ax2.plot(valid_val_epochs, valid_metric, label="Metric", marker='*', markersize=10, linewidth=2, color='black')

color_map = {
    "FromC":   "#6A0DAD",   # purple
    "FromB":  "#FF1493",   # deep pink
    "Pileup":   "#8B4513",   # saddle brown
    "Primary":  "#FF8C00",   # dark orange (replaces teal)
    "Fake":   "#708090",   # slate grey
    "Other":   "#FFD700",   # gold
    "FromBC":    "#00CED1",   # dark turquoise
}


# Per-class small-marker AUCs
class_auc_series = {
    "Primary": auc_primary,
    "Pileup":  auc_pileup,
    "FromB":   auc_fromB,
    "FromBC":  auc_fromBC,
    "FromC":   auc_fromC,
    "Other":   auc_other,
    "Fake":    auc_fake
}

if(show_test):
	for label, auc_list in class_auc_series.items():
	    ax2.plot(epochs, auc_list, marker=',', linestyle=':', linewidth=1, color=color_map[label])

val_class_auc_series = {
    "Primary": valid_val_auc_primary,
    "Pileup":  valid_val_auc_pileup,
    "FromB":   valid_val_auc_fromB,
    "FromBC":  valid_val_auc_fromBC,
    "FromC":   valid_val_auc_fromC,
    "Other":   valid_val_auc_other,
    "Fake":    valid_val_auc_fake
}

for label, val_auc_list in val_class_auc_series.items():
            ax2.plot(valid_val_epochs, val_auc_list, label=label, marker='s', markersize=4, linestyle='--', linewidth=1, color=color_map[label])

ax2.axvline(best_epoch, color='grey', linestyle='--', label=f"Best Epoch: {best_epoch}")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("AUC / Metric")
ax2.set_title("AUC Curves")
ax2.grid(True)
ax2.legend(ncol=2)

plt.tight_layout()
plt.show(block=False)
plt.pause(360)
plt.close
