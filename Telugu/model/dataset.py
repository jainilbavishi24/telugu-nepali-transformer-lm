import sentencepiece as spm
import torch

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class CausalLMDataset(Dataset):
    def __init__(self, tokenized_data, seq_len: int, stride: int):
        self.seq_len = seq_len
        self.examples = []

        for i in range(len(tokenized_data)):
            tokens = tokenized_data[i]["input_ids"]
            if len(tokens) < seq_len + 1:
                continue

            for start in range(0, len(tokens) - seq_len, stride):
                chunk = tokens[start : start + seq_len + 1]
            
                if len(chunk) < seq_len + 1:
                    break

                input_ids = chunk[:-1]
                labels = chunk[1:]

                self.examples.append(
                    (
                        torch.tensor(input_ids, dtype=torch.long),
                        torch.tensor(labels, dtype=torch.long)
                    )
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_datasets(config):
    data_cfg = config["data"]
    tokenizer_cfg = config["tokenizer"]
    model_cfg = config["model"]

    # --------------------------------------------------------
    # Load SentencePiece tokenizer
    # --------------------------------------------------------

    tokenizer = spm.SentencePieceProcessor(
        model_file=tokenizer_cfg["model_path"]
    )

    vocab_size = tokenizer.get_piece_size()

    print(f"Tokenizer vocabulary size: {vocab_size}")

    # --------------------------------------------------------
    # Load JSONL files from HuggingFace
    # --------------------------------------------------------

    repo_id = data_cfg["repo_id"]

    train_file = data_cfg["train_file"]
    val_file = data_cfg["val_file"]
    test_file = data_cfg["test_file"]

    dataset = load_dataset(
        "json",
        data_files={
            "train": (
                f"https://huggingface.co/datasets/"
                f"{repo_id}/resolve/main/{train_file}"
            ),
            "validation": (
                f"https://huggingface.co/datasets/"
                f"{repo_id}/resolve/main/{val_file}"
            ),
            "test": (
                f"https://huggingface.co/datasets/"
                f"{repo_id}/resolve/main/{test_file}"
            ),
        },
    )

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    print(f"Train documents: {len(train_data)}")
    print(f"Validation documents: {len(val_data)}")
    print(f"Test documents: {len(test_data)}")

    # --------------------------------------------------------
    # Determine text column
    # --------------------------------------------------------

    text_column = data_cfg["text_column"]

    if text_column not in train_data.column_names:
        raise ValueError(
            f"Column '{text_column}' not found.\n"
            f"Available columns: {train_data.column_names}"
        )

    # --------------------------------------------------------
    # Tokenization function
    # --------------------------------------------------------

    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()

    def tokenize_example(example):
        text = example[text_column]

        if not isinstance(text, str):
            text = str(text)

        token_ids = tokenizer.encode_as_ids(text)

        # Add BOS if tokenizer defines one.
        if bos_id >= 0:
            token_ids = [bos_id] + token_ids

        # Add EOS if tokenizer defines one.
        if eos_id >= 0:
            token_ids = token_ids + [eos_id]

        return {
            "input_ids": token_ids
        }

    # --------------------------------------------------------
    # Tokenize each split
    # --------------------------------------------------------

    print("Tokenizing train split...")
    train_tokenized = train_data.map(
        tokenize_example,
        remove_columns=train_data.column_names,
        num_proc=1,
    )

    print("Tokenizing validation split...")
    val_tokenized = val_data.map(
        tokenize_example,
        remove_columns=val_data.column_names,
        num_proc=1,
    )

    print("Tokenizing test split...")
    test_tokenized = test_data.map(
        tokenize_example,
        remove_columns=test_data.column_names,
        num_proc=1,
    )

    # --------------------------------------------------------
    # Build causal LM datasets
    # --------------------------------------------------------

    seq_len = model_cfg["max_seq_len"]
    stride = data_cfg["stride"]

    train_dataset = CausalLMDataset(
        train_tokenized,
        seq_len=seq_len,
        stride=stride,
    )

    val_dataset = CausalLMDataset(
        val_tokenized,
        seq_len=seq_len,
        stride=stride,
    )

    test_dataset = CausalLMDataset(
        test_tokenized,
        seq_len=seq_len,
        stride=stride,
    )

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        tokenizer,
    )


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    config,
):
    """
    Create PyTorch DataLoaders for all three splits.
    """

    training_cfg = config["training"]

    batch_size = training_cfg["batch_size"]
    num_workers = training_cfg["num_workers"]

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )