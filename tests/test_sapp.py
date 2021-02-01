from sapp import __version__
from sapp.search import Server, compose

index_p = "index_s.bin"
embed_p = "embed_s.pkl"
case_p = "cases.json"
path_p = "idx_path.json"


def test_version():
    assert __version__ == "0.1.0"


def test_search():
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
    server = Server(index_p=index_p, embed_p=embed_p, case_p=case_p, path_p=path_p)
    ret = server.search(text, top=10)
    print(ret)
    assert len(ret["imgs"] == 10)
