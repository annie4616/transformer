# data import
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mohamedlotfy50/wmt-2014-english-german")

print("Path to dataset files:", path)

# import pakages
import torch
import torch.nn as nn
import os
import csv
from torch.nn.utils.rnn import pad_sequence

# file reading
csv_path = os.path.join(path, "wmt14_translate_de-en_train.csv")

# None 또는 공백 제거
de_sentences = [] #s for s in de_sentences if s and s.strip()
en_sentences = [] #s for s in en_sentences if s and s.strip()

with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        de_sentences.append(row["de"])
        en_sentences.append(row["en"])
        if i >= 10000:
            break

print(en_sentences[0])
print(de_sentences[0])

#tokenizer
def tokenize(text):
    if not isinstance(text, str):
        return []
    return text.lower().strip().split()

from collections import Counter

# vocab 생성

# 모든 문장 토큰 수집
tokens = [token for sentence in en_sentences for token in tokenize(sentence)]

# 토큰 빈도로 vocab 생성
vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
for i, token in enumerate(Counter(tokens).keys(), start=4):
    vocab[token] = i

# vocab 사이즈 확인
print("Vocab size:", len(vocab))

# encoding(en)
def encode(sentence, vocab):
    tokens = ["<bos>"] + tokenize(sentence) + ["<eos>"]
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]

from torch.nn.utils.rnn import pad_sequence

# 100개만 샘플로 사용
encoded_seqs = [torch.tensor(encode(s, vocab)) for s in en_sentences[:]]

# 패딩 적용 (batch_first=True로 shape: [batch, seq])
padded_batch = pad_sequence(encoded_seqs, batch_first=True, padding_value=vocab["<pad>"])

print("Batch shape:", padded_batch.shape)  # ex: (100, max_seq_len) 이 shape을 하나로 정해줘야 함 / 아니면 토크나이저를 쓰기

# embedding/padding
embedding_dim = 512
embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)

embedded_output = embedding(padded_batch)  # shape: (batch, seq_len, embedding_dim)

print("Embedded shape:", embedded_output.shape)

# tokenize(ge)
def tokenize(text):
    if not isinstance(text, str):
        return []
    return text.lower().strip().split()

from collections import Counter

tokens_de = [token for sentence in de_sentences for token in tokenize(sentence)]

# vocab 생성

# special tokens 포함한 vocab 만들기
vocab_de = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
for i, token in enumerate(Counter(tokens_de).keys(), start=4):
    vocab_de[token] = i

print("독일어 vocab 크기:", len(vocab_de))

# encoding
def encode(sentence, vocab):
    tokens = ["<bos>"] + tokenize(sentence) + ["<eos>"]
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]

# 샘플용으로 100개 문장만 사용
encoded_de = [torch.tensor(encode(s, vocab_de)) for s in de_sentences[:]]

padded_de = pad_sequence(encoded_de, batch_first=True, padding_value=vocab_de["<pad>"])

print("Padded 독일어 텐서 shape:", padded_de.shape)

#embedding
embedding_dim = 512
embedding_de = nn.Embedding(num_embeddings=len(vocab_de), embedding_dim=embedding_dim)

embedded_de = embedding_de(padded_de)  # shape: (batch, seq_len, embedding_dim)
print("Embedded 독일어 텐서 shape:", embedded_de.shape)

# positional encoding
def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # shape: (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

    return pe  # shape: (seq_len, d_model)