from typing import Tuple, Optional

import torch
from datasets import load_dataset
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2TokenizerFast

from src.modules.transformer_translation_model import Seq2SeqTransformer


def generate_square_subsequent_mask(size: int, device: torch.device) -> Tensor:
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src: Tensor, tgt: Tensor, device, pad_idx: int = 1) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).to(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train(model: Optional[torch.nn.Module] = None,
          epochs: int = 20,
          batch_size: int = 2,
          lr: float = 3e-4,
          device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
          pad_idx: int = 1):
    if not model:
        model = Seq2SeqTransformer()
    model.to(device)
    model.train()

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    tokenizer = GPT2TokenizerFast.from_pretrained('../nmt_tokenizer')
    tokenizer.pad_token = tokenizer.eos_token
    pad_idx = tokenizer.eos_token_id

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = load_dataset('opus_books', 'en-ru', split='train')['translation'][:2]

    train_dataloader = DataLoader(dataset, batch_size=batch_size)

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            epoch_loss = 0
            for d in train_dataloader:
                src = d['en']
                tgt = d['ru']
                src = tokenizer(src, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
                tgt = tokenizer(tgt, return_tensors='pt', padding='max_length', truncation=True, max_length=64)

                src_ids = src['input_ids'].transpose(1, 0).to(device)
                tgt_ids = tgt['input_ids'].transpose(1, 0).to(device)
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_ids, tgt_ids, device)

                logits = model(src_ids, tgt_ids, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,
                               src_padding_mask)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_ids.reshape(-1))
                print(torch.argmax(logits, dim=0))
                loss.backward()
                torch.save(model.state_dict(), './checkpoint_model')
                epoch_loss += loss.item()
                optimizer.zero_grad()
                optimizer.step()
            print(f'Epoch loss: {epoch_loss / len(train_dataloader)}')
            pbar.update(1)


def greedy_decode(model: nn.Module, src: Tensor, src_mask: Tensor, max_len: int, start_symbol: str, device):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).to(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)
                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
    return ys


def main():
    torch.manual_seed(0)
    train()


if __name__ == '__main__':
    main()
