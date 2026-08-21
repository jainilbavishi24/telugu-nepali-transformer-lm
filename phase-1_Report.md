# Phase-1 Report

Telugu-Data: [Onedrive link](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/sairithvik_achutuni_students_iiit_ac_in/IgAbPacchgqmSbVdKT_TAu02AV_TzGa3lH0fX3qgQWTVQYo?e=X4PVYz)

Nepalese-Data: [Onedrive link](https://iiithydstudents-my.sharepoint.com/:f:/g/personal/sairithvik_achutuni_students_iiit_ac_in/IgB_rhkn1aGUT7zohmQDVkhFAbUbEFUwaCcprLqYDNjeslo?e=PPGckq)

## Overview

This project builds two independent pipelines for a higher-resource language (Telugu) and a lower-resource language (Nepalese), with the goal of producing cleaned corpora and SentencePiece BPE tokenizers suitable for downstream language-model training.

The two pipelines follow the same overall structure:

1. Source-specific corpus collection
2. Basic language/script filtering
3. Exact deduplication
4. Quality-based filtering
5. Near deduplication
6. Train/validation/test split
7. SentencePiece tokenizer training and tokenizer evaluation


---

## Common Pipeline Design

### Basic filtering
The basic filter removes HTML, comments, URLs, control characters, noise, and normalizes whitespace. It also requires a minimum document length and a language-script ratio threshold.


### Exact deduplication
The exact deduplicator uses normalized text hashing (Unicode normalization + whitespace normalization + SHA-256) to remove formatting-only duplicates.

### Quality filtering
The advanced quality filter removes documents with:

- high digit ratio
- many long tokens
- low lexical diversity
- excessive n-gram repetition
- excessive sentence repetition

### Near deduplication
The near-deduplication stage uses MinHash + LSH over word shingles. The configuration is:

- 64 permutations
- shingle size = 5
- LSH threshold = 0.95
- removal threshold = 0.98

### Split strategy
The corpus is split into train, validation, and test with a roughly 98/1/1 ratio.



> Please refer to `Telugu/report.ipynb` and `Nepalese/report.ipynb` for more detailed presentation


## Telugu (Higher-Resource Language)

### Data sources
The Telugu pipeline combines five sources:

- Wikipedia
- Sangraha
- Wikisource
- Wiktionary
- Wikibooks

### Corpus merge stats

- Total merged documents: 1,235,592
- Total merged characters: 2,500,476,214
- Source distribution:
  - Wikipedia: 71,295 documents, 384,680,141 characters
  - Sangraha: 890,055 documents, 1,803,087,684 characters
  - Wikisource: 182,084 documents, 252,768,509 characters
  - Wiktionary: 91,511 documents, 53,679,959 characters
  - Wikibooks: 647 documents, 6,259,921 characters

### Cleaning and deduplication stats

- Input to cleaning: 1,235,582 documents
- Documents kept after cleaning: 1,204,196
- Documents removed: 31,386
- Character reduction: 2,500,465,992 to 2,315,230,491
- Character removal percentage: 7.41%

Near deduplication summary:

- Documents processed: 1,235,583
- Documents kept: 1,235,582
- Documents removed by near dedup: 1
- Removed characters: 4,527
- Candidate pairs checked: 25,931
- Pairs above threshold: 1

This shows that the final Telugu corpus is mostly unique after filtering.

### Split stats

- Total valid documents: 1,204,196
- Train: 1,179,934 (97.99%)
- Validation: 12,137 (1.01%)
- Test: 12,125 (1.01%)

Character counts:

- Train: 2,269,069,645
- Validation: 22,806,398
- Test: 23,354,448
- Total: 2,315,230,491

### Tokenizer configuration

- Model type: SentencePiece BPE
- Vocabulary size: 24,000
- Character coverage: 0.9995

### Tokenizer evaluation statistics
The notebook analysis for Telugu reports the following evaluation metrics for the final tokenizer:

- Total tokens in the corpus: 641,752,789
- Total words: 291,052,982
- Fertility: 2.20 tokens per word
- Total unknown tokens: 678,964
- Documents with unknown tokens: 12.80%
- Unknown token percentage: 0.11%
- Vocabulary size: 8,000
- Vocabulary used: 99.96%
- Average token length: 4.47
- Minimum token length: 0
- Maximum token length: 15

The source token distribution:

- Sangraha: 75.75% of tokens
- Wikipedia: 11.98%
- Wikisource: 11.02%
- Wiktionary: 1.14%
- Wikibooks: 0.11%


---

## Nepalese (Lower-Resource Language)

### Data sources
The Nepalese pipeline incorporates several sources:

- Ekantipur
- Gorkhapatra
- OnlineKhabar
- Ratopati
- Common Crawl
- Sangraha
- Wikipedia

### Corpus merge and initial cleaning stats

- Raw merged documents: 1,901,195
- Final cleaned training corpus: 977,991 documents

This large reduction reflects the stricter language filtering and later quality-control passes.

### Exact deduplication stats

- Input documents: 1,875,655
- Kept documents: 980,283
- Duplicate documents removed: 895,372
- Duplicate percentage: 47.74%
- Input characters: 3,763,613,472
- Output characters: 2,023,493,337
- Duplicate characters removed: 1,740,120,135

This is one of the largest reductions in the overall pipeline and indicates significant repeated content in the merged Nepalese corpus.

### Quality filtering stats

- Total documents: 980,283
- Kept: 978,667
- Removed: 1,616
- Input characters: 2,023,493,337
- Output characters: 2,012,273,007
- Characters removed: 11,220,330

Primary removal reasons:

- Sentence repetition: 1,142
- N-gram repetition: 429
- Low lexical diversity: 43
- High digit ratio: 2

This suggests that repetition-based degradation is the dominant issue in the Nepalese corpus, more than structural corruption.

### Tokenizer configuration

- Model type: SentencePiece BPE
- Vocabulary size: 6,000
- Character coverage: 0.9995
- Normalization: nmt_nfkc
- Threads: 16

### Tokenizer evaluation statistics
The notebook analysis for Nepalese reports the following evaluation metrics:

- Total tokens in the corpus: 519,817,461
- Total words: 310,885,940
- Fertility: 1.67 tokens per word
- Total unknown tokens: 1,197,408
- Documents with unknown tokens: 16.74%
- Unknown token percentage: 0.23%
- Vocabulary size: 6,000
- Vocabulary used: 99.95%
- Average token length: 4.30
- Minimum token length: 0
- Maximum token length: 15

The source composition:

- Sangraha: 84.91% of tokens
- Nepalese Common Crawl corpus: 5.25%
- Wikipedia: 2.74%
- Gorkhapatra: 3.39%
- Ekantipur: 2.10%
- Ratopati: 1.59%
- OnlineKhabar: 0.02%


---

## Comparative Summary

### Telugu
- Higher-resource language
- Larger final corpus size
- More balanced multi-source composition
- SentencePiece BPE tokenizer with strong coverage and lower unknown-token rate
- Fertility: 2.20
- Unknown-token doc rate: 12.80%

### Nepalese
- Lower-resource language
- Greater duplication and more repetition-heavy content
- Stronger concentration in Sangraha data
- SentencePiece BPE tokenizer with smaller vocabulary and higher unknown-token rate
- Fertility: 1.67
- Unknown-token doc rate: 16.74%



