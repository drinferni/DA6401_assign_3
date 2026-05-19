import torch
import spacy
from datasets import load_dataset
from collections import Counter

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.raw_data = load_dataset("bentrevett/multi30k", split=self.split)
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError("Spacy models not found. Run: python -m spacy download de_core_news_sm en_core_web_sm")

        self.special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']
        self.unk_idx = 0
        self.pad_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3

        self.src_vocab = None
        self.tgt_vocab = None
        self.src_inv_vocab = None
        self.tgt_inv_vocab = None

    def _tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def _tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self, min_freq=2):
        """
        Builds the vocabulary mapping for src (de) and tgt (en).
        Only use the training split to build the vocabulary.
        """
        src_counter = Counter()
        tgt_counter = Counter()

        for item in self.raw_data:
            src_counter.update(self._tokenize_de(item['de']))
            tgt_counter.update(self._tokenize_en(item['en']))

        self.src_vocab = {token: i for i, token in enumerate(self.special_tokens)}
        self.tgt_vocab = {token: i for i, token in enumerate(self.special_tokens)}

        for word, freq in src_counter.items():
            if freq >= min_freq and word not in self.src_vocab:
                self.src_vocab[word] = len(self.src_vocab)

        for word, freq in tgt_counter.items():
            if freq >= min_freq and word not in self.tgt_vocab:
                self.tgt_vocab[word] = len(self.tgt_vocab)

        self.src_inv_vocab = {i: token for token, i in self.src_vocab.items()}
        self.tgt_inv_vocab = {i: token for token, i in self.tgt_vocab.items()}

        print(f"Vocab built. Source (DE) size: {len(self.src_vocab)}, Target (EN) size: {len(self.tgt_vocab)}")

    def process_data(self):
        """
        Convert English and German sentences into integer token lists.
        Includes <sos> at start and <eos> at the end.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        processed_data = []

        for item in self.raw_data:
            src_tokens = self._tokenize_de(item['de'])
            src_indices = [self.sos_idx] + \
                          [self.src_vocab.get(token, self.unk_idx) for token in src_tokens] + \
                          [self.eos_idx]

            tgt_tokens = self._tokenize_en(item['en'])
            tgt_indices = [self.sos_idx] + \
                          [self.tgt_vocab.get(token, self.unk_idx) for token in tgt_tokens] + \
                          [self.eos_idx]

            processed_data.append({
                'src': torch.tensor(src_indices, dtype=torch.long),
                'tgt': torch.tensor(tgt_indices, dtype=torch.long)
            })

        return processed_data