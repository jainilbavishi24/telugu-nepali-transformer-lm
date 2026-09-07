import json
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

JSON_PATH = "Nepalese/eval/attention_results/attention_metrics.json"

# Output files
ENTROPY_OUTPUT = "Nepalese/eval/attention_entropy_per_head.png"
DISTANCE_OUTPUT = "Nepalese/eval/mean_attention_distance_per_head.png"


# ============================================================
# LOAD JSON
# ============================================================

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

entropy = np.array(data["entropy"])
distance = np.array(data["mean_attention_distance"])

num_layers = data["num_layers"]
num_heads = data["num_heads"]

print("Entropy shape:", entropy.shape)
print("Distance shape:", distance.shape)


# ============================================================
# X-AXIS
# ============================================================

heads = np.arange(num_heads)

# Width of each individual bar
bar_width = 0.09


# ============================================================
# 1. ATTENTION ENTROPY
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

for layer in range(num_layers):

    # Offset each layer slightly so that the 8 bars
    # belonging to each head appear next to each other.
    offset = (
        layer - (num_layers - 1) / 2
    ) * bar_width

    ax.bar(
        heads + offset,
        entropy[layer],
        width=bar_width,
        label=f"Layer {layer}"
    )

ax.set_xlabel("Attention Head", fontsize=12)
ax.set_ylabel("Mean Attention Entropy", fontsize=12)

ax.set_title(
    "Attention Entropy per Head across Layers",
    fontsize=15
)

ax.set_xticks(heads)
ax.set_xticklabels(
    [f"Head {h}" for h in heads]
)

ax.legend(
    title="Layer",
    ncol=2
)

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    ENTROPY_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 2. MEAN ATTENTION DISTANCE
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7))

for layer in range(num_layers):

    offset = (
        layer - (num_layers - 1) / 2
    ) * bar_width

    ax.bar(
        heads + offset,
        distance[layer],
        width=bar_width,
        label=f"Layer {layer}"
    )

ax.set_xlabel("Attention Head", fontsize=12)
ax.set_ylabel("Mean Attention Distance (tokens)", fontsize=12)

ax.set_title(
    "Mean Attention Distance per Head across Layers",
    fontsize=15
)

ax.set_xticks(heads)
ax.set_xticklabels(
    [f"Head {h}" for h in heads]
)

ax.legend(
    title="Layer",
    ncol=2
)

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    DISTANCE_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("Saved:")
print(ENTROPY_OUTPUT)
print(DISTANCE_OUTPUT)