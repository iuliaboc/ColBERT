import os
import ujson

from functools import partial
from utils.utils import print_message
from modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples

from utils.runs import Run

import random
import torch

class EagerBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)

        self.triples_path = args.triples
        self.second_triples_path = args.second_triples
        self._reset_triples()

    def _reset_triples(self):
        self.reader = open(self.triples_path, mode='r', encoding="utf-8")
        self.second_reader = open(self.second_triples_path, mode='r', encoding="utf-8")
        self.position1 = 0
        self.position2 = 0

    def __iter__(self):
        return self

    def __next__(self):
        queries, positives, negatives = [], [], []

        probability_of_selection = 0.8
        if torch.rand(1).item() <= probability_of_selection:
            for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
                if (self.position + line_idx) % self.nranks != self.rank:
                    continue
                query, pos, neg = line.strip().split('\t')

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)

            self.position1 += line_idx + 1
        else:
            for line_idx, line in zip(range(self.bsize * self.nranks), self.second_reader):
                if (self.position + line_idx) % self.nranks != self.rank:
                    continue
                query, pos, neg = line.strip().split('\t')

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)

            self.position2 += line_idx + 1

        if len(queries) < self.bsize:
            raise StopIteration

        return self.collate(queries, positives, negatives)

    def collate(self, queries, positives, negatives):
        assert len(queries) == len(positives) == len(negatives) == self.bsize

        return self.tensorize_triples(queries, positives, negatives, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples()

        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None

#%%


# %%
