import torch
from collections import Counter

# --- Load the saved PyTorch file ---
data_list = torch.load("evtvaldata_test1_1211.pt")

print(f"Loaded {len(data_list)} event graphs")

# ---------------------------------------------------------
# 1. Inspect first 5 graphs (same as before)
# ---------------------------------------------------------
for i, data in enumerate(data_list[:5]):
    print(f"\nEvent {i}:")
    print(f"  x (node features): {data.x.shape}")
    print(f"  y (node one-hot labels): {data.y.shape}")
    print(f"  edge_index:        {data.edge_index.shape}")
    print(f"  edge_attr:         {data.edge_attr.shape}")

    # show first few rows of one-hot
    print("  First 5 node one-hot labels:\n", data.y[:5])

    # convert to indices for readability
    print("  First 10 label indices:", torch.argmax(data.y, dim=1)[:10])


# ---------------------------------------------------------
# 2. Compute GLOBAL distribution across all events
# ---------------------------------------------------------
all_label_indices = []

for data in data_list:
    # Convert [N,7] one-hot → [N] integer class
    labels = torch.argmax(data.y, dim=1)
    all_label_indices.extend(labels.tolist())

label_counts = Counter(all_label_indices)

print("\n=================================================")
print("GLOBAL LABEL DISTRIBUTION (validation set)")
print("=================================================")

for label in range(7):
    print(f"  Label {label}: {label_counts.get(label, 0)} tracks")

# ---------------------------------------------------------
# 3. Pileup check (label = 1)
# ---------------------------------------------------------
PILEUP_LABEL = 1
num_pileup = label_counts.get(PILEUP_LABEL, 0)

print("\n----------------------------------------------")
print("PILEUP CHECK:")
print("----------------------------------------------")
print(f"  Pileup tracks in validation set: {num_pileup}")

if num_pileup == 0:
    print("  ❌ No pileup tracks → AUC = NaN (expected)")
elif num_pileup == 1:
    print("  ⚠️ Only ONE pileup track → AUC undefined → NaN")
else:
    print("  ✔️ Pileup exists → AUC should be computable")

