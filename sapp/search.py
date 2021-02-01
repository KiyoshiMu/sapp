import json
import logging
import os
import pickle
from typing import List

import faiss
import numpy as np
from numpy.linalg import norm

from sapp.infer import BertServer

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

DEFAULT_INDEX = "index_s.bin"
DEFAULT_EMBED = "embed_s.pkl"


def l2norm(feature_v):
    return feature_v / np.expand_dims(norm(feature_v, axis=1), axis=1)


def make_index(embed_p):
    with open(embed_p, "rb") as target:
        embed = pickle.load(target)
    embed_shape = embed[0].shape
    assert embed_shape == (768,), f"{embed_shape} is not (768,)"
    logger.info("embedding shape is %s", embed_shape)
    indexer = faiss.IndexFlatIP(embed_shape[0])
    indexer.add(l2norm(embed))
    faiss.write_index(indexer, DEFAULT_INDEX)


def load_index(index_p=DEFAULT_INDEX, embed_p=DEFAULT_EMBED) -> faiss.IndexFlatIP:
    if not os.path.exists(index_p):
        make_index(embed_p)
    indexer = faiss.read_index(index_p)
    return indexer


def compose(case: dict, keep_key=False) -> str:
    if keep_key:
        tmp = [f"{k}: {v}" for k, v in case.items()]
    else:
        tmp = list(case.values())
    return " ".join(tmp)


def load_cases(case_p="cases.json") -> List[dict]:
    with open(case_p, "r", encoding="utf8") as target:
        return json.load(target)


def load_path(path_p):
    with open(path_p, "r", encoding="utf8") as target:
        return json.load(target)


class Server:
    def __init__(
        self,
        index_p=DEFAULT_INDEX,
        embed_p=DEFAULT_EMBED,
        case_p="cases.json",
        path_p="idx_path.json",
    ) -> None:
        self.bert = BertServer()
        self.indexer = load_index(index_p=index_p, embed_p=embed_p)
        self.case = load_cases(case_p)
        self.path = load_path(path_p)

    def search(self, text, top=10):
        embed = self.bert.predict(text)
        _, ids = self.indexer.search(l2norm(embed), k=top)
        # logger.info(ids[0], distences[0] )
        return {
            "cases": [self.case[idx] for idx in ids[0]],
            "imgs": [self.path[idx] for idx in ids[0]],
        }
