# -*- conding: utf-8 -*-
# @Time    : 2026/4/5  19:25
# @Author  : psi

import numpy as np


def cosine_similarity(embedding1, embedding2):
    # 归一化
    emb1 = embedding1 / np.linalg.norm(embedding1, axis=1, keepdims=True)
    emb2 = embedding2 / np.linalg.norm(embedding2, axis=1, keepdims=True)
    # emb1 = embedding1
    # emb2 = embedding2
    # 点积
    sim = np.dot(emb1, emb2.T)  # (1, 81)

    return sim


def clu_similarity(embeddings1, embeddings2, topk=[1, 3, 5, 10]):

    result = {}
    for i in range(embeddings1.shape[0]):
        e1 = np.expand_dims(embeddings1[i, :], axis=0)

        r = cosine_similarity(e1, embeddings2)
        arr = r[0]

        sorted_idx = np.argsort(arr)[::-1]
        sorted_idx = sorted_idx.tolist()

        for t in topk:
            if t not in result:
                result[t] = []
            if i in sorted_idx[:t]:
                result[t].append(1)
            else:
                result[t].append(0)

    scores = {k: sum(v)/len(v) for k, v in result.items()}

    return scores

