import os
import sys
import json
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
import sentencepiece as spm
import yaml
from huggingface_hub import hf_hub_download


# =========================================================
# Project imports
# =========================================================

root_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Telugu.model.transformer import LanguageModel


# =========================================================
# Configuration
# =========================================================

LANGUAGE = "Telugu"

CONFIG_FILE = f"{LANGUAGE}/configs/config1.yaml"

REPO_ID = "iconically-mine/LMA-individual-project"

MODEL_FILE = f"{LANGUAGE}/best_{LANGUAGE.lower()}.pt"

TEST_BIN = f"data_bins/{LANGUAGE.lower()}/test_train.bin"

OUTPUT_DIR = "attention_results"

NUM_EXAMPLES = 5

SEQ_LEN = 128


# =========================================================
# Device
# =========================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# =========================================================
# Load config
# =========================================================

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)


# =========================================================
# Load model
# =========================================================

def load_model():

    model = LanguageModel(
        vocab_size=config["model"]["vocab_size"],
        max_seq_len=config["model"]["max_seq_len"],
        embed_dim=config["model"]["embed_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        ff_hidden_dim=config["model"]["ff_hidden_dim"],
        dropout=config["model"]["dropout"],
        attn_dropout=config["model"]["attn_dropout"],
        rope_base=config["model"]["rope_base"],
        pad_token_id=config["model"]["pad_token_id"],
        tie_embeddings=config["model"]["tie_embeddings"],
    )

    model_path = f".hf_cache/{LANGUAGE}/best_{LANGUAGE.lower()}.pt"

    if not os.path.exists(model_path):

        print("Downloading checkpoint...")

        os.makedirs(
            f".hf_cache/{LANGUAGE}",
            exist_ok=True
        )

        hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILE,
            local_dir=".hf_cache"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print("Model loaded.")

    return model


# =========================================================
# Attention entropy
# =========================================================

def attention_entropy(attention):

    """
    attention:
        (B, H, T, T)

    Returns:
        entropy:
        (B, H, T)
    """

    # Avoid log(0)
    eps = 1e-12

    attention_safe = attention.clamp(min=eps)

    entropy = -(
        attention_safe *
        torch.log(attention_safe)
    ).sum(dim=-1)

    return entropy


# =========================================================
# Mean attention distance
# =========================================================

def attention_distance(attention):

    """
    attention:
        (B, H, T, T)

    Returns:
        distance:
        (B, H, T)
    """

    B, H, T, _ = attention.shape

    positions = torch.arange(
        T,
        device=attention.device,
        dtype=attention.dtype
    )

    # |i - j|
    distances = torch.abs(
        positions[:, None] -
        positions[None, :]
    )

    # Sum_j attention[i,j] * |i-j|
    distance = (
        attention * distances
    ).sum(dim=-1)

    return distance


# =========================================================
# Generate heatmap
# =========================================================

def save_heatmap(
    attention,
    layer,
    head,
    example_id,
    output_dir
):

    """
    attention:
        (T, T)
    """

    plt.figure(figsize=(8, 7))

    plt.imshow(
        attention,
        aspect="auto",
        origin="lower"
    )

    plt.xlabel("Key position")
    plt.ylabel("Query position")

    plt.title(
        f"Layer {layer}, Head {head}, "
        f"Example {example_id}"
    )

    plt.colorbar(
        label="Attention weight"
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        f"heatmap_example{example_id}"
        f"_layer{layer}_head{head}.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()


# =========================================================
# Main evaluation
# =========================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    model = load_model()

    # -----------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------

    tokenizer_path = config["tokenizer"]["model_path"]

    sp = spm.SentencePieceProcessor(
        model_file=tokenizer_path
    )

    # -----------------------------------------------------
    # Load test tokens
    # -----------------------------------------------------

    test_tokens = np.memmap(
        TEST_BIN,
        dtype=np.uint16,
        mode="r"
    )

    required = SEQ_LEN

    if len(test_tokens) < required:

        raise ValueError(
            "Test set is too small."
        )

    # -----------------------------------------------------
    # Pick deterministic examples
    # -----------------------------------------------------

    rng = np.random.default_rng(42)

    starts = rng.choice(
        len(test_tokens) - required + 1,
        size=NUM_EXAMPLES,
        replace=False
    )

    # -----------------------------------------------------
    # Storage for metrics
    # -----------------------------------------------------

    num_layers = config["model"]["num_layers"]
    num_heads = config["model"]["num_heads"]

    entropy_values = np.zeros(
        (num_layers, num_heads),
        dtype=np.float64
    )

    distance_values = np.zeros(
        (num_layers, num_heads),
        dtype=np.float64
    )

    num_processed = 0

    # =====================================================
    # Process examples
    # =====================================================

    for example_id, start in enumerate(starts):

        print(
            f"\nProcessing example "
            f"{example_id + 1}/{NUM_EXAMPLES}"
        )

        token_ids = test_tokens[
            start:start + SEQ_LEN
        ].astype(np.int64)

        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        # -------------------------------------------------
        # Forward pass
        # -------------------------------------------------

        with torch.no_grad():

            logits, all_attentions = model(
                input_ids,
                return_attention=True
            )

        # -------------------------------------------------
        # Process every layer
        # -------------------------------------------------

        for layer, attention in enumerate(
            all_attentions
        ):

            # attention:
            # (B, H, T, T)

            entropy = attention_entropy(
                attention
            )

            distance = attention_distance(
                attention
            )

            # Average over batch and query positions
            layer_entropy = entropy.mean(
                dim=(0, 2)
            )

            layer_distance = distance.mean(
                dim=(0, 2)
            )

            entropy_values[layer] += (
                layer_entropy.cpu().numpy()
            )

            distance_values[layer] += (
                layer_distance.cpu().numpy()
            )

            # -------------------------------------------------
            # Save heatmaps
            # -------------------------------------------------

            # Save every head for every layer
            for head in range(num_heads):

                heatmap = attention[
                    0,
                    head
                ].cpu().numpy()

                save_heatmap(
                    heatmap,
                    layer,
                    head,
                    example_id,
                    OUTPUT_DIR
                )

        num_processed += 1

    # =====================================================
    # Average across examples
    # =====================================================

    entropy_values /= num_processed
    distance_values /= num_processed

    # =====================================================
    # Print results
    # =====================================================

    print("\n")
    print("=" * 70)
    print("ATTENTION ENTROPY")
    print("=" * 70)

    for layer in range(num_layers):

        print(
            f"Layer {layer}: "
            f"mean={entropy_values[layer].mean():.4f}"
        )

    print("\n")
    print("=" * 70)
    print("MEAN ATTENTION DISTANCE")
    print("=" * 70)

    for layer in range(num_layers):

        print(
            f"Layer {layer}: "
            f"mean={distance_values[layer].mean():.4f}"
        )

    # =====================================================
    # Save JSON
    # =====================================================

    results = {
        "num_examples": NUM_EXAMPLES,
        "seq_len": SEQ_LEN,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "entropy": entropy_values.tolist(),
        "mean_attention_distance": distance_values.tolist(),
    }

    output_file = os.path.join(
        OUTPUT_DIR,
        "attention_metrics.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print(
        f"\nSaved metrics to {output_file}"
    )

    print(
        f"Heatmaps saved to {OUTPUT_DIR}/"
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()