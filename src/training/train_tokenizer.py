from datasets import Dataset, load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from torch import nn
from transformers import GPT2TokenizerFast


def batch_iterator(batch_size: int = 512):
    dataset = load_dataset('opus_books', 'en-ru', split='train')['translation']
    for i in range(0, len(dataset), batch_size):
        yield ' '.join(d['ru'] for d in dataset[i: i + batch_size])
    for i in range(0, len(dataset), batch_size):
        yield ' '.join(d['en'] for d in dataset[i: i + batch_size])


def main():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=15_000, special_tokens=['<|endoftext|>'])
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    new_tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    new_tokenizer.save_pretrained('../nmt_tokenizer')


if __name__ == '__main__':
    main()
