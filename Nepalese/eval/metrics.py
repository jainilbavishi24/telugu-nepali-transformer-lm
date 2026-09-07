import torch
import torch.nn.functional as F
import math

import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Nepalese.model.transformer import LanguageModel
from huggingface_hub import hf_hub_download
from Nepalese.train.lm_dataset import make_dataloader

import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')


CONFIG_FILE = 'Nepalese/configs/config1.yaml'
with open(CONFIG_FILE, 'r') as f:
    config = yaml.safe_load(f)


LANGUAGE ='Nepalese'
REPO_ID ='iconically-mine/LMA-individual-project'
FILE_NAME = f'{LANGUAGE}/best_{LANGUAGE.lower()}.pt'

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


    def compute_PPL(self):
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
        return ppl


    def test(self, stride: int = 256, seq_len: int = 512, batch_size: int = 16,  bin_path: str=f'data_bins/{LANGUAGE.lower()}/test_train.bin'):
        test_loader = make_dataloader(
            bin_path=bin_path,
            seq_len=seq_len,
            stride=stride,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False
        )

        for x, y in test_loader:
            print(f"x shape: {x.shape}, y shape: {y.shape}")
            with torch.no_grad():
                logits = self.model(x.to(device))

            print(f"logits shape: {logits.shape}")
            break


# def compute_PPL()

if __name__ == "__main__":
    metrics = Metrics()
    print(f"Perplexity: {metrics.compute_PPL()}")