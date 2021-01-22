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

DEFAULT_INDEX = "index.bin"


def l2norm(feature_v):
    return feature_v / np.expand_dims(norm(feature_v, axis=1), axis=1)


def make_index():
    if os.path.exists(DEFAULT_INDEX):
        return

    with open("embed.pkl", "rb") as target:
        embed = pickle.load(target)
    embed_shape = embed[0].shape
    assert embed_shape == (768,), f"{embed_shape} is not (768,)"
    logger.info("embedding shape is %s", embed_shape)
    indexer = faiss.IndexFlatIP(embed_shape[0])
    indexer.add(l2norm(embed))
    faiss.write_index(indexer, DEFAULT_INDEX)


def load_index() -> faiss.IndexFlatIP:
    if not os.path.exists(DEFAULT_INDEX):
        make_index()
    indexer = faiss.read_index(DEFAULT_INDEX)
    return indexer


def compose(case: dict, keep_key=False) -> str:
    if keep_key:
        tmp = [f"{k}: {v}" for k, v in case.items()]
    else:
        tmp = list(case.values())
    return " ".join(tmp)


def load_cases() -> List[dict]:
    with open("cases.json", "r", encoding="utf8") as target:
        return json.load(target)


class Server:
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertServer()
        self.indexer = load_index()
        self.case = load_cases()

    def search(self, text, top=10):
        embed = self.bert.predict(text)
        distences, ids = self.indexer.search(l2norm(embed), k=top)
        # logger.info(ids[0], distences[0] )
        return [self.case[idx] for idx in ids[0]]


if __name__ == "__main__":
    # ["plasma cell neoplasm"]
    case = {
        "PARTICLES": "CELLULAR.",
        "MEGAKARYOCYTES AND PLATELETS": "PRESENT.",
        "EXTRINSIC CELLS": "NONE IDENTIFIED.",
        "ERYTHROPOIESIS": "NORMOBLASTIC.",
        "GRANULOPOIESIS": "ACTIVE WITH NORMAL MATURATION.",
        "RETICULUM CELLS, PLASMA CELLS AND LYMPHOCYTES": "MODEST INCREASE IN PLASMA CELLS, SOME BILOBED. 5-8% MNC PATCHY.",
        "HEMOSIDERIN": "GRADE 2 IN STORES. SCANT IN PRECURSORS.",
        "COMMENT": "NO OVERT EVIDENCE OF NHL. QUERY PLASMA CELL DYSCRASIA. CORRELATE WITH SPEP/QI. FLOW AND BIOPSY TO FOLLOW.",
    }
    text = compose(case)
    server = Server()
    print(server.search(text))
