import torch
import torch.nn.functional as F
import math

import os
import sys
import json

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Nepalese.model.transformer import LanguageModel
from huggingface_hub import hf_hub_download
from Nepalese.train.lm_dataset import make_dataloader

import yaml

import numpy as np
import sentencepiece as spm

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


CONFIG_FILE = 'Nepalese/configs/config1.yaml'
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)


LANGUAGE ='Nepalese'
REPO_ID ='iconically-mine/LMA-individual-project'
FILE_NAME = 'finetuned/Nepalese/best_nepalese_finetuned.pt'

hf_hub_download(
    repo_id=REPO_ID,
    filename=FILE_NAME,
    local_dir=f'.hf_cache/'
)

class Metrics:
    def __init__(self, stride=256, seq_len=512, batch_size=16,  model_path: str = f'.hf_cache/{LANGUAGE}/best_{LANGUAGE.lower()}.pt'):
        self.model = LanguageModel(
            vocab_size=config['model']['vocab_size'],
            max_seq_len=config['model']['max_seq_len'],
            embed_dim=config['model']['embed_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            ff_hidden_dim=config['model']['ff_hidden_dim'],
            dropout=config['model']['dropout'],
            attn_dropout=config['model']['attn_dropout'],
            rope_base=config['model']['rope_base'],
            pad_token_id=config['model']['pad_token_id'],
            tie_embeddings=config['model']['tie_embeddings']
        )

        self._init_model_from_checkpoint(model_path)
        self._init_dataloaders(stride=stride, seq_len=seq_len, batch_size=batch_size)
        self.test_jsonl_path = self._download_test_jsonl(repo_id='iconically-mine/LMA-indivudal-project', filename='nepalese_test.jsonl', local_dir='.hf_cache/Nepalese')

    def _download_test_jsonl(self, repo_id: str, filename: str, local_dir: str = ".hf_cache") -> str:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            repo_type="dataset"
        )
        return f"{local_dir}/{filename}"


    def _init_dataloaders(self, stride: int = 256, seq_len: int = 512, batch_size: int = 16):
        self.test_loader = make_dataloader(
            bin_path=f'data_bins/{LANGUAGE.lower()}/test_train.bin',
            seq_len=config['model']['max_seq_len'],
            stride=stride,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False
        )

    def _init_model_from_checkpoint(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError(f"Checkpoint at {model_path} does not contain 'model_state_dict'.")
        self.model.to(device)
        self.model.eval()


    def compute_PPL_BPB(self):
        total_loss = 0.0
        total_tokens = 0

        for x, y in self.test_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
                total_loss += loss.item()
                total_tokens += y.numel()

            print(f"Processed batch: {x.size(0)} samples, total tokens: {total_tokens}, cumulative loss: {total_loss}")

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        total_bits = avg_loss / math.log(2)

        bpb = total_loss / self.count_test_bytes(self);
        return ppl, bpb

    def count_test_bytes(self, print_result=False):
        total_bytes = 0

        with open(self.test_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                text = record["text"]
                total_bytes += len(text.encode("utf-8"))

        if print_result:
            print(f'Total number of bytes in test set: {total_bytes}')

        return total_bytes

    def generate_examples(
        self,
        num_examples: int = 100,
        prefix_len: int = 128,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        greedy: bool = False,
        output_path: str = "Nepalese/generation_examples_1_0.json",
    ):
        test_tokens = np.memmap(
            f"data_bins/{LANGUAGE.lower()}/test_train.bin",
            dtype=np.uint16,
            mode="r",
        )

        # Load the SentencePiece tokenizer
        sp = spm.SentencePieceProcessor(
            model_file=config["tokenizer"]["model_path"]
        )

        # Need prefix_len + max_new_tokens contiguous tokens
        required_tokens = prefix_len + max_new_tokens

        if len(test_tokens) < required_tokens:
            raise ValueError("Test set is too small for generation.")

        # Reproducible random starting positions
        rng = np.random.default_rng(42)

        starts = rng.choice(
            len(test_tokens) - required_tokens + 1,
            size=num_examples,
            replace=False,
        )

        results = []

        for i, start in enumerate(starts):

            # 128-token prefix
            prefix_ids = test_tokens[
                start : start + prefix_len
            ].astype(np.int64)

            # Actual next 128 tokens from the test set
            reference_ids = test_tokens[
                start + prefix_len :
                start + required_tokens
            ].astype(np.int64)

            # Convert prefix to [1, prefix_len]
            prefix = torch.tensor(
                prefix_ids,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            # Generate continuation
            with torch.no_grad():
                output = self.model.generate(
                    prefix,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    greedy=greedy,
                )

            # Only keep newly generated tokens
            generated_ids = output[0, prefix_len:]

            # Decode
            prefix_text = sp.decode(prefix_ids.tolist())
            reference_text = sp.decode(reference_ids.tolist())
            generated_text = sp.decode(generated_ids.tolist())

            results.append({
                "id": i,
                "start_token": int(start),
                "prefix": prefix_text,
                "reference": reference_text,
                "generated": generated_text,
            })

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_examples} examples")

        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                results,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Saved {num_examples} examples to {output_path}")



if __name__ == "__main__":
    metrics = Metrics()
    # print(f"Perplexity: {metrics.compute_PPL()}")
    # print(f"Bits per token: {metrics.compute_BPB()}")
    # print(f"Total bytes in test set: {metrics.count_test_bytes(print_result=True)}")

    # metrics.generate_examples(
    #     num_examples=100,
    #     prefix_len=128,
    #     max_new_tokens=128,
    #     temperature=1.5,
    #     greedy=True,
    #     output_path="Nepalese/generation_examples_greedy.json",
    # )

    ppl, bpb = metrics.compute_PPL_BPB()
    print(f"Perplexity: {ppl}")
    print(f"Bits per token: {bpb}")

    metrics.generate_examples(
        num_examples=100,
        prefix_len=128,
        max_new_tokens=128,
        temperature=1.5,
        output_path="Nepalese/generation_examples_1_5_finetuned.json",
    )

    metrics.generate_examples(
        num_examples=100,
        prefix_len=128,
        max_new_tokens=128,
        temperature=1.0,
        output_path="Nepalese/generation_examples_1_0_finetuned.json",
    )

    metrics.generate_examples(
        num_examples=100,
        prefix_len=128,
        max_new_tokens=128,
        temperature=0.5,
        output_path="Nepalese/generation_examples_0_5_finetuned.json",
    )

    metrics.generate_examples(
        num_examples=100,
        prefix_len=128,
        max_new_tokens=128,
        temperature=1.5,
        greedy=True,
        output_path="Nepalese/generation_examples_greedy_finetuned.json",
    )