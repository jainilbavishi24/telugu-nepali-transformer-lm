# Telugu & Nepali Transformer LMs — Built from Scratch

Two independent GPT-style decoder-only language models (~25M parameters each), written in
plain PyTorch and trained end to end: corpus collection → data cleaning → tokenizer training
→ pretraining → reasoning finetuning → evaluation.

One model is for **Telugu** (high-resource, ~104M speakers), the other for **Nepali**
(low-resource, ~32M speakers). The two corpora are completely independent of each other —
no machine translation and no synthetic pretraining data.

Everything in `model/` is hand-written: scaled dot-product attention, multi-head attention,
rotary positional embeddings, pre-norm transformer blocks. No `transformers` model classes,
no pretrained weights.

📄 **[Full write-up with all results and figures → `Report.pdf`](Report.pdf)**

---

## Headline results

| | Telugu | Nepali |
|---|---|---|
| Corpus size | 641.8M tokens | 519.8M tokens |
| Self-collected share | 24.3% | 15.1% |
| Tokenizer vocab | 8,000 (BPE) | 6,000 (BPE) |
| Fertility (tokens/word) | 2.20 | 1.67 |
| Unknown-token rate | 0.11% | 0.23% |
| Test perplexity | 38.96 | 23.87 |
| Bits per token | 0.769 | 0.618 |

**Reasoning finetuning** (1,000 held-out examples, strict exact match after normalization):

| | Pretrained | Finetuned |
|---|---|---|
| Telugu — strict accuracy | 0.0% | **70.4%** |
| Telugu — format compliance | 0.0% | **99.6%** |
| Nepali — strict accuracy | 0.0% | **85.9%** |
| Nepali — format compliance | 0.0% | **99.4%** |

Both gains are significant under McNemar's test (p < 1e-200); no example that the pretrained
model got right was broken by finetuning.

---

## Architecture

Identical for both languages apart from vocabulary size — 25,117,696 trainable parameters.

```
Input IDs
   ↓
Token Embedding + Dropout        (tied with the LM head, saves ~16% of params)
   ↓
Transformer Block × 8            pre-norm:  x + MHA(LN(x))  →  x + FFN(LN(x))
   ↓                             8 heads × 64 dims, RoPE applied to Q/K per layer
Final LayerNorm                  FFN = Linear → GELU → Linear, 3× expansion (1536)
   ↓
LM Head (Linear)
   ↓
Logits
```

| Hyperparameter | Value |
|---|---|
| Embedding dim | 512 |
| Layers | 8 |
| Heads | 8 (head dim 64) |
| FFN hidden dim | 1536 (3×) |
| Max sequence length | 512 (RoPE is relative, so this is the *trained* window) |
| Dropout | 0.1 |
| RoPE base | 10,000 |
| Tied embeddings | yes |

**Why RoPE** over learned or sinusoidal positional embeddings: it adds no learnable
parameters, and because it encodes *relative* position by rotating Q and K, the trained
context window is not a hard architectural ceiling.

---

## Corpus collection

The brief required ~500M training tokens per language, with at least 20% self-collected
(not taken from an existing prepared corpus).

**Telugu** — 641,752,789 tokens

| Source | Tokens | Share |
|---|---|---|
| Sangraha (verified, non-synthetic split) | 486,127,738 | 75.75% |
| Wikipedia | 76,881,984 | 11.98% |
| Wikisource | 70,721,157 | 11.02% |
| Wiktionary | 7,315,982 | 1.14% |
| Wikibooks | 705,928 | 0.11% |

Wikimedia alone cleared the 20% self-collected bar — it turned out to hold a large library
of human-written Telugu articles, out-of-copyright novels and textbooks.

**Nepali** — 519,817,461 tokens

| Source | Tokens | Share |
|---|---|---|
| Sangraha (verified, non-synthetic split) | 441,377,006 | 84.91% |
| Common Crawl | 27,290,417 | 5.25% |
| Gorkhapatra | 17,621,812 | 3.39% |
| Wikipedia | 14,242,998 | 2.74% |
| Ekantipur | 10,916,167 | 2.10% |
| Ratopati | 8,265,098 | 1.59% |
| OnlineKhabar | 103,963 | 0.02% |

Nepali was much harder. Wikimedia yielded only a fraction of what was needed, so this half
leaned on news-site scraping and Common Crawl. Because Nepali shares the Devanagari script
with Hindi and several other languages, a language-identification model was used to keep
only genuine Nepali text. Even so, the self-collected share landed at 15.1%, short of the
20% target.

### Cleaning pipeline

The same four stages for both languages (`{Language}/tokenizer/data_cleaning/`):

1. **Basic filter** — strips HTML tags, URLs and control characters; normalizes whitespace
   and Unicode.
2. **Exact dedup** — hashes every document and drops hash collisions.
3. **Quality filter** — thresholds on language ratio, sentence-repetition ratio, digit
   ratio and lexical diversity; drops documents that are too short or too long.
4. **Near dedup** — MinHash + LSH to remove near-duplicates.

---

## Tokenizers

SentencePiece BPE — chosen because BPE is the current default and adapts well to Indic
morphology. The corpus is split 99/1/1 into train/val/test, and **every** component
(tokenizer included) is fitted on the train split only.

| Metric | Telugu | Nepali |
|---|---|---|
| Vocab size | 8,000 | 6,000 |
| Vocab utilization | 99.96% | 99.95% |
| Total words | 291,052,982 | 310,885,940 |
| Fertility (tokens/word) | 2.20 | 1.67 |
| Unknown tokens | 678,964 (0.11%) | 1,197,408 (0.23%) |
| Docs containing an unknown | 12.80% | 16.74% |
| Mean / max token length | 4.47 / 15 | 4.30 / 15 |

---

## Pretraining

One full pass over each corpus on 2× T4 GPUs, with mixed precision, `torch.compile`, and a
token-budgeted cosine schedule (`max_steps` is derived from the token count, so the decay
bottoms out exactly when the budget runs out).

| Hyperparameter | Value |
|---|---|
| Epochs | 1 |
| Tokens per optimizer step | 32,768 (batch 64 × seq 512) |
| Steps | 12,319 (Telugu) / 14,670 (Nepali) |
| Learning rate | 6.0e-4, cosine to 10% |
| Warmup steps | 300 |
| Weight decay | 0.1 (2-D params only) |
| Betas / eps | 0.9, 0.95 / 1e-8 |
| Grad clip | 1.0 |
| Seed | 42 |

### Intrinsic metrics

| Language | Total bytes | Perplexity | Bits per token |
|---|---|---|---|
| Telugu | 61,712,860 | 38.959 | 0.769 |
| Nepali | 52,036,231 | 23.868 | 0.618 |

The Nepali model coming out *ahead* was the surprise of the project. Two likely causes: the
Telugu model saw marginally fewer steps, and the Nepali corpus was assembled from a far
wider range of sources, so source diversity may have helped.

### Generation evaluation

Reference-based and diversity metrics across four decoding settings (pretrained models):

| Language | Decoding | BLEU | chrF | ROUGE-L | Rep-2 | Distinct-2 |
|---|---|---|---|---|---|---|
| Telugu | Greedy | 1.42 | 11.95 | 0.0201 | 0.8260 | 0.1740 |
| Telugu | Temp 0.5 | 1.82 | 16.15 | 0.0351 | 0.6367 | 0.3633 |
| Telugu | Temp 1.0 | 0.57 | 23.57 | 0.0290 | 0.0468 | 0.9532 |
| Telugu | Temp 1.5 | 0.18 | 23.11 | 0.0211 | 0.0020 | 0.9980 |
| Nepali | Greedy | 2.50 | 18.00 | 0.0361 | 0.8032 | 0.1968 |
| Nepali | Temp 0.5 | 1.79 | 20.58 | 0.0368 | 0.5409 | 0.4591 |
| Nepali | Temp 1.0 | 1.04 | 24.82 | 0.0418 | 0.0513 | 0.9487 |
| Nepali | Temp 1.5 | 0.19 | 22.86 | 0.0107 | 0.0034 | 0.9966 |

The same trade-off shows up in both: greedy decoding degenerates into repetition (Rep-2
above 0.80), while high temperature maximizes diversity at the cost of fidelity.

### Attention analysis

Per-head attention entropy and mean attention distance, measured layer by layer. Both models
show the same qualitative pattern: **early layers attend locally, deeper layers attend far**.
In the Nepali model, Layer 0 has heads with mean distances under 6 tokens while Layer 6 has
heads reaching past 40 — consistent with lower layers handling local/positional context and
deeper layers integrating distant context. The Telugu model varies more sharply across heads,
mixing strongly local and strongly long-range heads in the same layer.

Plots live in `{Language}/outputs/{pretrained,finetuned}/`.

---

## Reasoning finetuning

### Dataset

Synthetic `question → reasoning → answer` triplets, generated from 19 hand-written templates
across 7 families: `direct`, `direct_yes_no`, `equality`, `two_hop`,
`two_hop_reversed_wording`, `three_hop`, `ranking`.

The generator (`generate_dataset.py`) instantiates each template's logic, **computes the
answer in code**, and only then renders the language-specific text — so the ground truth is
derived, never generated by a model. Entity names are disjoint across splits, which stops the
model from memorizing name→answer associations.

20,000 train / 2,500 validation / 2,500 test examples per language, with native-script field
labels (`ప్రశ్న / తర్కం / జవాబు`, `प्रश्न / तर्क / उत्तर`).

### Hyperparameters

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Learning rate | 2.0e-5 |
| Batch size | 8 |
| Grad accumulation | 4 |
| Weight decay | 0.01 |
| Seed | 20260914 |

### Evaluation

Three metrics, all against a pretrained baseline on the same 1,000 examples:

- **Strict** — normalized generation must equal the reference exactly.
- **Loose** — reference appears anywhere in the generation.
- **Format compliance** — did the model learn the `question/reasoning/answer` format at all?

| Metric | Telugu pretrained | Telugu finetuned | Nepali pretrained | Nepali finetuned |
|---|---|---|---|---|
| Strict | 0.0% | 70.4% | 0.0% | 85.9% |
| Loose | 26.5% | 83.0% | 24.2% | 89.7% |
| Format compliance | 0.0% | 99.6% | 0.0% | 99.4% |

Per-family breakdowns are in `{Language}/outputs/finetuned/reasoning_evaluation_results.json`.
Easy families saturate (`direct_yes_no` hits 100% strict in Telugu) while `ranking` stays
hardest in both languages — it has the lowest chance level (1/6) and needs a full ordering
rather than a single comparison.

Two negative results worth recording:

- **Intrinsic metrics did not move at all** after finetuning. The likely reason is that
  general language tokens, which have nothing to do with the reasoning task, overwhelmingly
  dominate the perplexity computation.
- **Generation metrics and attention patterns barely moved either**, suggesting the
  finetuning taught an output format and a task-specific mapping without reorganizing the
  underlying generation mechanism.

---

## Repository layout

```
Telugu/ , Nepalese/          one self-contained pipeline per language
├── tokenizer/
│   ├── data_collection/     Wikimedia dumps, Common Crawl, news scrapers, Sangraha
│   ├── data_cleaning/       the 4-stage pipeline + orchestrator (pipeline.py)
│   ├── utils/               corpus splitting, token counting, tokenizer evaluation
│   ├── models/              trained SentencePiece BPE models (.model / .vocab)
│   ├── train_tokenizer.py
│   └── report.ipynb         tokenizer analysis notebook
├── model/
│   ├── attention.py         scaled dot-product + multi-head attention
│   ├── positional.py        rotary positional embeddings
│   └── transformer.py       FFN, pre-norm block, full LM
├── train/
│   ├── tokenize_and_pack.py corpus → flat uint16 token stream (.bin)
│   ├── stream_jsonl.py      memory-bounded JSONL streaming
│   ├── lm_dataset.py
│   └── train.py             pretraining loop (AMP, DDP, resumable, MFU readout)
├── eval/
│   ├── metrics.py           perplexity, bits per token, BLEU/chrF/ROUGE-L, rep-n, distinct-n
│   ├── reason_evaluation.py pretrained vs finetuned, with McNemar's test
│   └── generation + attention evaluation
│       Nepalese/: evaluate_generation.py, evaluate_attention.py
│       Telugu/:   generation.evaluation.py, attention.evaluation.py
├── configs/config1.yaml     fully commented pretraining config
└── outputs/                 metrics JSON + every figure in the report

generate_dataset.py          synthetic reasoning dataset generator
reasoning_templates.json     19 templates × 7 reasoning families
finetune_reasoning.py        reasoning finetuning script
Report.pdf                   full write-up
```

---

## Reproducing

```bash
pip install -r requirements.txt
```

Most scripts pull their models and data straight from the Hugging Face Hub, so there is
little else to set up. Every figure in the report is reproducible from the scripts in
`{Language}/eval/`.

To pretrain from scratch:

```bash
python Telugu/train/tokenize_and_pack.py --config Telugu/configs/config1.yaml
python Telugu/train/train.py            --config Telugu/configs/config1.yaml
```

To finetune on the reasoning task (generate the dataset first — it writes
`reasoning_dataset/{telugu,nepali}/{train,validation,test}.jsonl` plus a manifest):

```bash
python generate_dataset.py

python finetune_reasoning.py \
  --language telugu \
  --hf-file Telugu/best_telugu.pt \
  --train-data reasoning_dataset/telugu/train.jsonl \
  --validation-data reasoning_dataset/telugu/validation.jsonl
```

Pass `--hf-file` explicitly: the flag defaults to the Nepali checkpoint regardless of
`--language`, and loading the wrong one fails on the vocabulary-size mismatch (8,000 vs
6,000).

To evaluate:

```bash
python Telugu/eval/reason_evaluation.py        # pretrained vs finetuned reasoning
python Telugu/eval/generation.evaluation.py    # BLEU / chrF / ROUGE-L / rep-n / distinct-n
python Telugu/eval/attention.evaluation.py     # per-head entropy and attention distance

python Nepalese/eval/reason_evaluation.py      # same three for Nepali
python Nepalese/eval/evaluate_generation.py
python Nepalese/eval/evaluate_attention.py
```

---

## Artifacts

- **[Dataset on Hugging Face](https://huggingface.co/datasets/iconically-mine/LMA-indivudal-project)** — packed pretraining token streams and the reasoning finetuning dataset
- **[Models on Hugging Face](https://huggingface.co/iconically-mine/LMA-individual-project)** — pretrained and finetuned checkpoints
- Mirrors: [finetuning dataset](https://drive.google.com/drive/folders/13DEU1NXyCrVLY-h-vFyCJZ2qb9HXdrKY?usp=sharing) · [finetuned models](https://drive.google.com/drive/folders/129S8VG_Jpdi_JnSzyg41c05HCvD9C4mz?usp=sharing)

Pretraining corpora and packed `.bin` files are not in the repo (see `.gitignore`) — they
live in the Hugging Face dataset repo above.

---

## Notes

Built over about a month as an individual course project for Language Modeling & Applications,
in three phases: corpus + tokenizer, architecture + pretraining, and reasoning finetuning +
evaluation. The commit history follows those phases.

Author: A Sai Rithvik (2023102060)
