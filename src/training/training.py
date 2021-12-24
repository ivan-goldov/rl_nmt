from typing import Tuple

import torch
from datasets import load_dataset
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2TokenizerFast

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


def train(model: torch.nn.Module,
          epochs: int = 20,
          batch_size: int = 32,
          lr: float = 3e-4,
          device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
          pad_idx: int = 1):
    model.to(device)
    model.train()

    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    losses = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = load_dataset('opus_books', 'en-ru', split='train')['translation'][15]
    tokenizer = GPT2TokenizerFast.from_pretrained('../../nmt_tokenizer')
    train_dataloader = DataLoader(dataset, batch_size=batch_size)

    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            for d in train_dataloader:
                src = d['en']
                tgt = d['ru']
                src = tokenizer(src, return_tensors='pt', padding='max_length', truncation=True, max_length=32)
                tgt = tokenizer(tgt, return_tensors='pt', padding='max_length', truncation=True, max_length=32)

                src_ids = src['input_ids'].transpose(1, 0).to(device)
                # src_mask = src['attention_mask'].bool().reshape((batch_size, -1)).to(device)
                tgt_ids = tgt['input_ids'].transpose(1, 0).to(device)
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_ids, tgt_ids, device)
                # tgt_mask = tgt['attention_mask'].bool().reshape((batch_size, -1)).to(device)
                # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(source, tgt_input)

                # logits = model(src_ids, tgt_ids)
                logits = model(src_ids, tgt_ids, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
                # print(logits.shape)
                # print(logits.reshape(-1, logits.shape[-1]).shape)

                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_ids.reshape(-1))
                loss.backward()
                torch.save(model.state_dict(), './checkpoint_model')
                print(loss.item())
                optimizer.zero_grad()
                optimizer.step()
                # return 0
            pbar.update(1)

    # return losses / len(train_dataloader)


# def greedy_decode(model, src, src_mask, max_len, start_symbol):
#     src = src.to(DEVICE)
#     src_mask = src_mask.to(DEVICE)
#
#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
#     for i in range(max_len - 1):
#         memory = memory.to(DEVICE)
#         tgt_mask = (generate_square_subsequent_mask(ys.size(0))
#                     .type(torch.bool)).to(DEVICE)
#         out = model.decode(ys, memory, tgt_mask)
#         out = out.transpose(0, 1)
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.item()
#
#         ys = torch.cat([ys,
#                         torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
#         if next_word == EOS_IDX:
#             break
#     return ys


# def evaluate(model):
#     model.eval()
#     losses = 0
#
#     val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
#     val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
#
#     for src, tgt in val_dataloader:
#         src = src.to(DEVICE)
#         tgt = tgt.to(DEVICE)
#
#         tgt_input = tgt[:-1, :]
#
#         src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
#
#         logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
#
#         tgt_out = tgt[1:, :]
#         loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
#         losses += loss.item()
#
#     return losses / len(val_dataloader)

def main():
    torch.manual_seed(0)
    transformer = Seq2SeqTransformer()
    train(transformer)


if __name__ == '__main__':
    main()
