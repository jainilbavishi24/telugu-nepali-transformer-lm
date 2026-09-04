import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .positional import RotaryPositionalEmbedding


class FeedForward(nn.Module):
    """
    GPT-style MLP:
        Linear -> GELU -> Dropout -> Linear -> Dropout

    Uses the common 4 * hidden_dim expansion.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * embed_dim

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class TransformerBlock(nn.Module):
    """
    Pre-LN decoder-only Transformer block:

        x
        |
        +--> LayerNorm --> Self Attention --> Dropout --+
        |                                                |
        +---------------------------------------------->+
        |
        +--> LayerNorm --> MLP --> Dropout -------------+
        |                                                |
        +---------------------------------------------->+
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        rope_base: float = 10_000.0,
        max_seq_len: int = 512,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        head_dim = embed_dim // num_heads

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.rope = RotaryPositionalEmbedding(
            head_dim=head_dim,
            max_len=max_seq_len,
            base=rope_base,
        )

        # attn_dropout applies to the attention *probabilities* (after softmax,
        # before the value matmul), which is a different regularizer from the
        # sublayer-output dropout below. 0.0 leaves it off entirely: the module
        # skips the Dropout call rather than running an identity op on the
        # (B, H, T, T) tensor.
        self.attn = MultiHeadAttention(
            input_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout if attn_dropout and attn_dropout > 0 else None,
            rope=self.rope,
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=ff_hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
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


class LanguageModel(nn.Module):
    """
    Decoder-only GPT-style language model.

    Input:
        input_ids: (B, T)

    Output:
        logits: (B, T, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 512,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_hidden_dim: int | None = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        rope_base: float = 10_000.0,
        pad_token_id: int | None = None,
        tie_embeddings: bool = True,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.tie_embeddings = tie_embeddings

        # Token embedding only.
        #
        # With RoPE, we do NOT need a learned positional embedding.
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_token_id,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    rope_base=rope_base,
                    max_seq_len=max_seq_len,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(embed_dim)

        self.lm_head = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False,
        )

        self._reset_parameters()

        # Weight tying:
        #
        # input embedding matrix:
        #       E : vocab_size x embed_dim
        #
        # LM head:
        #       E^T : embed_dim x vocab_size
        #
        # This is standard for many GPT-style LMs.
        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    def _reset_parameters(self):
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        nn.init.ones_(self.final_ln.weight)
        nn.init.zeros_(self.final_ln.bias)

    def build_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns:

            (1, seq_len, seq_len)

        where 1 means attention is allowed.

        Example for T=4:

            [[1,0,0,0],
             [1,1,0,0],
             [1,1,1,0],
             [1,1,1,1]]
        """

        return torch.tril(
            torch.ones(
                1,
                seq_len,
                seq_len,
                device=device,
                dtype=torch.bool,
            )
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        """
        input_ids:
            (B, T)

        attention_mask:
            Optional padding mask of shape (B, T)

            1 = valid token
            0 = padding

        returns:
            logits: (B, T, vocab_size)

        optionally:
            attention maps from every layer
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape (batch_size, sequence_length)"
            )

        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        # ------------------------------------------------
        # 1. Token embedding
        # ------------------------------------------------
        x = self.token_embedding(input_ids)

        x = self.embedding_dropout(x)

        # ------------------------------------------------
        # 2. Causal mask
        # ------------------------------------------------
        causal_mask = self.build_causal_mask(
            seq_len=seq_len,
            device=input_ids.device,
        )

        # ------------------------------------------------
        # 3. Combine causal + padding masks
        # ------------------------------------------------
        #
        # causal_mask:
        #       (1, T, T)
        #
        # attention_mask:
        #       (B, T)
        #
        # We need:
        #       (B, 1, T, T)
        #
        # such that:
        #   - future tokens are masked
        #   - padding keys are masked
        #
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(
                    "attention_mask must have shape (B, T)"
                )

            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    "attention_mask must have shape "
                    f"({batch_size}, {seq_len})"
                )

            attention_mask = attention_mask.to(
                device=input_ids.device,
                dtype=torch.bool,
            )

            # (B, T) -> (B, 1, 1, T)
            padding_mask = attention_mask[:, None, None, :]

            # (1, T, T) -> (1, 1, T, T)
            causal_mask = causal_mask[:, None, :, :]

            # -> (B, 1, T, T)
            mask = causal_mask & padding_mask

        else:
            # (1, T, T)
            mask = causal_mask

        # ------------------------------------------------
        # 4. Transformer blocks
        # ------------------------------------------------
        all_attentions = []

        for block in self.blocks:
            if return_attention:
                x, attention = block(
                    x,
                    mask=mask,
                    return_attention=True,
                )
                all_attentions.append(attention)
            else:
                x = block(
                    x,
                    mask=mask,
                    return_attention=False,
                )

        # ------------------------------------------------
        # 5. Final LayerNorm
        # ------------------------------------------------
        x = self.final_ln(x)

        # ------------------------------------------------
        # 6. Vocabulary projection
        # ------------------------------------------------
        logits = self.lm_head(x)

        if return_attention:
            return logits, all_attentions

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """
        Autoregressive generation.

        input_ids:
            (B, T)

        returns:
            (B, T + generated_tokens)
        """

        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the most recent max_seq_len tokens.
            context = input_ids[:, -self.max_seq_len:]

            logits = self(context)

            # Only the final token matters for the next prediction.
            next_token_logits = logits[:, -1, :]

            # Temperature scaling.
            next_token_logits = next_token_logits / temperature

            # Optional top-k filtering.
            if top_k is not None:

                if top_k <= 0:
                    raise ValueError("top_k must be positive")

                k = min(top_k, next_token_logits.size(-1))

                values, _ = torch.topk(
                    next_token_logits,
                    k=k,
                )

                threshold = values[:, [-1]]

                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold,
                    float("-inf"),
                )

            # Convert logits -> probabilities.
            probs = torch.softmax(
                next_token_logits,
                dim=-1,
            )

            # Sample next token.
            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            # Append.
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

        return input_ids