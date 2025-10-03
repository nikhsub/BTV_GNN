import matplotlib.pyplot as plt

# Thresholds
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
x_vals = list(range(len(thresholds)))  # numeric x positions

# -------------------------
# One common track data
# -------------------------
one_IVF_eff  = [65.3, 65.3, 65.3, 65.3, 65.3, 65.3, 65.3, 65.3, 65.3]
one_RECO_eff = [63.4, 65.7, 65.7, 65.1, 64.9, 64.8, 64.6, 64.2, 63.4]
one_GEN_eff =  [64.6, 64.6, 64.6, 64.6, 64.6, 64.6, 64.6, 64.6, 64.6]

one_IVF_fake  = [57.4, 57.4, 57.4, 57.4, 57.4, 57.4, 57.4, 57.4, 57.4]
one_RECO_fake = [54.9, 58.9, 58.9, 56.2, 55.0, 53.6, 51.9, 49.6, 46.1]
one_GEN_fake  = [11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5]

# -------------------------
# Two common track data
# -------------------------
two_IVF_eff  = [55.1, 55.1, 55.1, 55.1, 55.1, 55.1, 55.1, 55.1, 55.1]
two_RECO_eff = [54.2, 54.8, 54.7, 54.7, 54.7, 54.9, 54.9, 54.9, 54.5]
two_GEN_eff  = [56.6, 56.6, 56.6, 56.6, 56.6, 56.6, 56.6, 56.6, 56.6]

two_IVF_fake  = [64.1, 64.1, 64.1, 64.1, 64.1, 64.1, 64.1, 64.1, 64.1]
two_RECO_fake = [64.6, 65.7, 65.1, 63.1, 62.0, 60.1, 59.0, 56.9, 53.6]
two_GEN_fake  = [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22]

# -------------------------
# Plot 1: Efficiency (One common track)
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(x_vals, one_IVF_eff, 'o--', color='blue', label="IVF")
plt.plot(x_vals, one_RECO_eff, 's-', color='red', label="RECO")
plt.plot(x_vals, one_GEN_eff, 's-', color='orange', label="GEN")
plt.xticks(x_vals, thresholds)
plt.xlabel("Model threshold")
plt.ylabel("Efficiency (%)")
plt.title("Efficiency vs Threshold (One common track)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("eff_1track.png")

# -------------------------
# Plot 2: Fake Rate (One common track)
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(x_vals, one_IVF_fake, 'o--', color='blue', label="IVF")
plt.plot(x_vals, one_RECO_fake, 's-', color='red', label="RECO")
plt.plot(x_vals, one_GEN_fake, 's-', color='orange', label="GEN")
plt.xticks(x_vals, thresholds)
plt.xlabel("Model threshold")
plt.ylabel("Fake Rate (%)")
plt.title("Fake Rate vs Threshold (One common track)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("fake_1track.png")

# -------------------------
# Plot 3: Efficiency (Two common tracks)
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(x_vals, two_IVF_eff, 'o--', color='blue', label="IVF")
plt.plot(x_vals, two_RECO_eff, 's-', color='red', label="RECO")
plt.plot(x_vals, two_GEN_eff, 's-', color='orange', label="GEN")
plt.xticks(x_vals, thresholds)
plt.xlabel("Model threshold")
plt.ylabel("Efficiency (%)")
plt.title("Efficiency vs Threshold (Two common tracks)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("eff_2track.png")

# -------------------------
# Plot 4: Fake Rate (Two common tracks)
# -------------------------
plt.figure(figsize=(7,5))
plt.plot(x_vals, two_IVF_fake, 'o--', color='blue', label="IVF")
plt.plot(x_vals, two_RECO_fake, 's-', color='red', label="RECO")
plt.plot(x_vals, two_GEN_fake, 's-', color='orange', label="GEN")
plt.xticks(x_vals, thresholds)
plt.xlabel("Model threshold")
plt.ylabel("Fake Rate (%)")
plt.title("Fake Rate vs Threshold (Two common tracks)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("fake_2track.png")

