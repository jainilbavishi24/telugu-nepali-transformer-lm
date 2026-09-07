# PHASE-2

In this phase, I implemented a GPT-style model (code located in `{Language}\models\`) that I then pre-trained with the corpus collected in phase-1. I then did my analysis, which I will note in this report.


> I will split this report in 2 sections, - Implementation, - Results


## IMPLEMENTATION

As mentioned before, the code files of implementation of both models is in [Telugu](Telugu/model/) and [Nepalese](Nepalese/model/), both of these code files are completely identical (copy-paste). The transformer implementation is spread out in 3 files (attention.py, positional.py, transformer.py).

### ATTENTION

**Scaled Dot Product Attention**: The foundational equation is implemented in this function `scaled_dot_product_attention` in `attention.py`:
```py

## The formula is implemented in this way in python/torch

attn_logits = (torch.matmul(q, k.transpose(-2, -1))/ math.sqrt(d_k))
output = torch.matmul(attention, v)

```

**Multi-Head Attention**: I implemented multihead attention in `MultiHeadAttention` in `attention.py`:


```py

## Abbridged MHA implementation

self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
self.o_proj = nn.Linear(embed_dim, input_dim)

qkv = self.qkv_proj(x)
values, attention = scaled_dot_product_attention(q, k, v, self.attn_dropout, mask=mask)

o = self.o_proj(values)

return o
```

### POSITIONAL EMBEDDING

I implemented Rotary Positional Embedding. The reasons for choosing it over Learned PE or Sinusoidal PE:

- Sinusoidal PE has fixed sequence length over training time
- In RoPE there are no learnable parameters, hence making it more extensible

Essentially I just chose RoPE because it is imo the most popular PE and is relatively easy to implement.


RoPE works by rotating Q and K matrices by doing matrix multiplications before the attention mechanism over every layer.

Implementation is in positional.py.

```py
## Abbridged Implementation of RoPE
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    return torch.stack((-x[..., 1], x[..., 0]), dim=-1).flatten(-2)

def apply_rotary_pos_emb(x, cos, sin, rotary_dim):
    rotated, remainder = x[..., :rotary_dim], x[..., rotary_dim:]
    rotated = rotated * cos + rotate_half(rotated) * sin
    return torch.cat((rotated, remainder), dim=-1)
```

### Transformer

Implemented like:

Self-attention Sublayer:
LayerNorm -> Self Attention -> Dropout

Feed-forward Sublayer:
LayerNorm -> FFN -> Dropout

```py
# -------------------------
# Self-attention sublayer
# -------------------------
residual = x

x_norm = self.ln1(x)

if return_attention:
    attn_out, attention = self.attn(
        x_norm,
        mask=mask,
        return_attention=True,
    )
else:
    attn_out = self.attn(
        x_norm,
        mask=mask,
        return_attention=False,
    )

x = residual + self.attn_dropout(attn_out)
# -------------------------
# Feed-forward sublayer
# -------------------------
residual = x

x_norm = self.ln2(x)
ff_out = self.ffn(x_norm)

x = residual + ff_out

if return_attention:
    return x, attention

return x
```

I have glossed over other less important details. They can be found in models/. i believe I went over the most important blocks and how I implemented them.


## RESULTS

### Intrinsic Metrics

#### Perplexity and Bits-per-Byte
Calculated in `{Language}/eval/metrics.py`:

**Telugu**


![telugu_metrics](Telugu/eval/intrinsic_metrics.png)


**Nepalese**


![Nepalese_metrics](Nepalese/eval/intrinsic_metrics.png)

#### Possible Interpretations

Telugu model has a higher perplexity (38.96) compared with the Nepalese model (23.87). This indicates that the Telugu model is less confident in predicting the next token on its test set and therefore models the language less effectively under the current setup. The Telugu model also has a slightly higher bits-per-byte/token metric (0.769 vs. 0.618), further indicating higher uncertainty.

This is a bit surprising, but I think it can be due to training/data collection. Telugu model was trained for (marginally) fewer steps that nepalese model. In data collection, for nepalese I collected text from a large collection of sources, so the diversity might have positively effected the model. 


### Generation Evaluation

**Telugu**


![telugu_metrics](Telugu/eval/generation_metrics.png)


**Nepalese**


![Nepalese_metrics](Nepalese/eval/generation_metrics.png)



#### Possible Interpretation

The Nepali model generally performs better in reference-based generation metrics, while both models show a similar decoding trade-off:

- **BLEU**: Nepali is higher for greedy (2.50 vs. 1.42) and T=0.5 (1.79 vs. 1.82, essentially similar), while both drop substantially at higher temperatures. This suggests moderate-temperature generation can improve reference overlap, but higher sampling temperatures reduce it.
- **chrF**: Nepali consistently has higher chrF than Telugu at greedy (18.00 vs. 11.95), T=0.5 (20.58 vs. 16.15), and T=1.0 (24.82 vs. 23.57). At T=1.5, the scores are very similar (22.86 vs. 23.11).
- **ROUGE-L**: Nepali has slightly better ROUGE-L at greedy and T=1.0, while Telugu is slightly better at T=0.5 and T=1.5. Overall, both have very low ROUGE-L, indicating limited exact sequence overlap.
- **Repetition**: Both models suffer from severe repetition under greedy decoding, with Rep-2/Rep-3 around 0.8. Increasing temperature greatly reduces repetition. At T=1.0, Telugu has slightly lower repetition than Nepali (Rep-2: 0.0468 vs. 0.0513), while at T=1.5 both are almost completely non-repetitive.
- **Diversity**: Both models show the same strong increase in Distinct-1/2 as temperature increases. Telugu reaches slightly higher diversity at T=1.0 and T=1.5, whereas Nepali has higher diversity at T=0.5


**coherence and language correctness**:
Note that, I cant read either of these languages (nepalese a little bit maybe). I gave it to GPT and verified that the language was coherent although did not make sense.



### Attention Evaluation

#### Attention Heatmaps

**Telugu**

- Layer 0 (early layer):
![layer0](Telugu/eval/telugu_layer_0.png)

- Layer 3 (middle layer):
![layer1](Telugu/eval/telugu_layer_3.png)


- Layer 7 (late layer):
![layer7](Telugu/eval/telugu_layer_7.png)


Example text for all these plots are:

పనిచేసింది. అల్ ఇండియా అస్సాం రైతుల సమావేశానికి కూడా నాయకత్వం వహించింది. 1972 లో భారత ప్రభుత్వం ఆమెను పద్మశ్రీతో సత్కరించింది. మళ్లీ 2002 లో, భారత ప్రభుత్వం ఆమె గౌరవార్థం స్మారక స్టాంపును విడుదల చేసింది. మరణం, వారసత్వం సైకియాని తన 72 వ పుట్టినరోజు నాడు, 1972 మార్చి 16 న క్యాన్సర్తో మరణించింది ఆమె మరణానికి కొన్ని నెలల ముందు,1972 లో, భారత ప్రభుత్వం ఆమెకు పద్మశ్రీ పురస్కారం ప్రదానం చేసింది. 2002 లో పోస్ట్ డిపార్ట్మెంట్ ఆఫ్ పోస్ట్స్ ద్వారా సంఘ సంస్కర్తల స్మారక తపాలా బిళ్ళను విడుదల చేసి ప్రభుత్వం ఆమెను సత్కరించింది. గౌహ



**Nepalese**

- Layer 0 (early layer):
![layer0](Nepalese/eval/nepalese_layer_0.png)

- Layer 3 (middle layer):
![layer1](Nepalese/eval/nepalese_layer_3.png)


- Layer 7 (late layer):
![layer7](Nepalese/eval/nepalese_layer_7.png)


Example text for all these plots are:


बजाना (विविध मनोरञ्जनात्मक मैथिली गीतहरूको श्रव्यसँगालो), प्रकाशकः गङ्गा क्यासेट, नयादिल्ली, भारतः २०६३ भजन दिव्यानन्द (चट्याङ मास्टरद्वारा रचित नेपाली भक्तिरचनाहरूको रूका श्रव्यसँगालो), प्रकाशकः कान्ता गौतम, काठमाडौँ, नेपालः २०६१ वन-गीत (संरक्षणसम्बन्धी चेतनामूलक मैथिली गीतहरूको श्रव्यसँगालो), प्रकाशकः बाइसेप एसटी, काठमाडौँ, नेपालः २०६१ शैक्षिक कृतिहरू खिस्सा-पिहानी (मै


#### Attention Entropy

**Telugu**

![entropy_tel](Telugu/eval/attention_entropy_per_head.png)

**Nepalese**

![entropy_nep](Nepalese/eval/attention_entropy_per_head.png)


#### Mean Attention Distance

**Telugu**

![att_dist](Telugu/eval/mean_attention_distance_per_head.png)


**Nepalese**

![att_dist](Nepalese/eval/mean_attention_distance_per_head.png)

Heads with low attention distance and relatively low entropy are the strongest candidates for local behavior. In particular for Nepalese are:

- Layer 0, Head 7: entropy ≈ 0.68, distance ≈ 4.40
- Layer 0, Head 0: entropy ≈ 0.73, distance ≈ 5.30
- Layer 1, Head 2: entropy ≈ 1.65, distance ≈ 3.12
- Layer 3, Head 1: entropy ≈ 0.77, distance ≈ 2.97
- Layer 4, Head 0: entropy ≈ 1.33, distance ≈ 5.11
- Layer 4, Head 6: entropy ≈ 1.09, distance ≈ 6.14

For Telugu:

- Layer 0, Head 0: entropy ≈ 0.38, distance ≈ 3.7
- Layer 4, Head 0: entropy ≈ 0.90, distance ≈ 3.9
- Layer 3, Head 3: entropy ≈ 1.49, distance ≈ 4.0
- Layer 2, Head 5: entropy ≈ 1.35, distance ≈ 5.9
- Layer 2, Head 3: entropy ≈ 1.07, distance ≈ 6.2
- Layer 1, Head 2: entropy ≈ 2.04, distance ≈ 6.5
- Layer 1, Head 7: entropy ≈ 1.54, distance ≈ 6.8

Heads with substantially larger attention distances indicate that they incorporate information from much farther back in the sequence. These heads for Nepalese maybe:

- Layer 6, Head 6: distance ≈ 43.67
- Layer 5, Head 4: distance ≈ 38.79
- Layer 7, Head 2: distance ≈ 35.43
- Layer 6, Head 4: distance ≈ 25.64
- Layer 7, Head 7: distance ≈ 23.23

For telugu:

- Layer 5, Head 5: distance ≈ 43.6, entropy ≈ 1.90
- Layer 5, Head 1: distance ≈ 40.3, entropy ≈ 2.08
- Layer 7, Head 0: distance ≈ 31.8, entropy ≈ 2.84
- Layer 7, Head 7: distance ≈ 31.2, entropy ≈ 2.69
- Layer 2, Head 6: distance ≈ 30.0, entropy ≈ 3.15
- Layer 3, Head 0: distance ≈ 29.1, entropy ≈ 2.70
- Layer 4, Head 2: distance ≈ 29.3, entropy ≈ 3.00


the Telugu model shows substantial variation across heads, with both highly local heads and strongly long-range heads. The deeper layers generally contain more long-range heads than the earliest layers.

The Nepali model shows a clear tendency for early layers to have shorter attention distances, while several deeper layers develop substantially longer-range attention. For example, Layer 0 contains heads with distances below 6 tokens, whereas Layer 6 contains heads reaching over 40 tokens. This suggests that the model may use lower layers primarily for local contextual and positional information and deeper layers for integrating more distant context

Both models exhibit the same broad pattern: some heads specialize in local/positional attention while others attend over much longer distances, particularly in deeper layers. However, the degree of specialization differs. The Telugu model contains an especially concentrated local head in the first layer (Layer 0–Head 0, with entropy around 0.38 and distance around 3.7), while the Nepali model's corresponding local heads have somewhat higher entropy and distance (e.g. Layer 0–Head 0: ≈0.73 and 5.3 tokens). At the long-range end, the two models are quite similar: Nepali reaches about 43.7 tokens (Layer 6–Head 6), while Telugu reaches roughly 43 tokens (Layer 5–Head 5).
